# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision Transformer used in DINO."""

import copy
import functools
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit

"""
0-11 Sentinel 2 bands (EXcluding 10th band cloud band)
12-19: Sentinel 1 bands (4 for each ASC and DESC)
20: DEM band
"""

class MultiModalToTokenSequence(nn.Module):
  patches: ml_collections.ConfigDict
  hidden_size: int
  channel_embed_size: int = 128  # TODO: Expose as config
  posembs: Tuple[int, int] = (14, 14)
  positional_embedding: str = 'learned'
  sen2_channel_groups: Tuple[Tuple[int]] = ((1, 2, 3, 7), (4, 5, 6, 8), (10, 11)),
  sen2_maintain_seqlen: bool = False
  changroups_sampling_weights: Optional[Tuple[int]] = None
  multimodal_type: str = 'early_fuse_s1_to_rgbn'

  @nn.compact
  def __call__(self, x: jnp.ndarray, seqlen: int = -1,
               seqlen_selection: str = 'unstructured',
               drop_moment="early"):
    fh, fw = self.patches.size
    G = len(self.sen2_channel_groups)

    ### Sentinel 2 Grouped Channel Patch Embedding
    # Grouped Channel Patch Embedding
    x_grouped = []
    for idx, group in enumerate(self.sen2_channel_groups):
      x_group = x[:, :, :, group]
      x_group = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                        name=f'embedding-{idx}')(x_group)
      x_grouped.append(x_group)

    x_sen2 = jnp.stack(x_grouped, axis=3)  # (B, Hp, Wp, G, D)

    # Add spectral and positional embedding
    # Following SatMAE (except for initialization)
    chanemb = self.param(
      'channel_embed_input',
      nn.initializers.normal(stddev=1 / np.sqrt(self.channel_embed_size)),
      (1, G, self.channel_embed_size), x.dtype)  # (1, G, Dc)

    posembed_size = self.hidden_size - self.channel_embed_size
    posemb = self.param(
      'posembed_input',
      nn.initializers.normal(stddev=1 / np.sqrt(posembed_size)),
      (1, self.posembs[0], self.posembs[1], posembed_size), x.dtype)  # (1, Hp, Wp, Dp)

    _, h, w, _, _ = x_sen2.shape

    if (h, w) != self.posembs:
      posemb = jax.image.resize(posemb, (1, h, w, posembed_size), 'bilinear')

    # Expand channel embedding to add spatial dims
    _chanemb = jnp.expand_dims(chanemb, axis=(1, 2))  # (1, 1, 1, G, Dc)
    _chanemb = jnp.broadcast_to(
      _chanemb, (1, h, w, G, self.channel_embed_size))  # (1, Hp, Wp, G, Dc)
    # Expand positional embedding to add group dim
    _posemb = jnp.expand_dims(posemb, axis=3)  # (1, Hp, Wp, 1, Dp)
    _posemb = jnp.broadcast_to(
      _posemb, (1, h, w, G, posembed_size))  # (1, Hp, Wp, G, Dp)

    # Concatenate channel and positional embeddings
    pos_chan_embed = jnp.concatenate((_chanemb, _posemb), axis=-1)  # (1, Hp, Wp, G, D)

    x_sen2 = x_sen2 + pos_chan_embed
    x_sen2 = jnp.reshape(x_sen2, (-1, h * w, G, self.hidden_size))  # (B, Hp * Wp = L, G, D)

    # Fuse
    if 'early_fuse' in self.multimodal_type:
      ### Sentinel 1 Patch Embedding
      x_sen1 = x[:, :, :, 12:]  # (B, Hp, Wp, C_sen1)
      x_sen1 = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                       name='embedding')(x_sen1)  # [B, Hp, Wp, hidden_size]

      if self.multimodal_type == 'early_fuse_s1_to_rgbn':
        # Add Sentinel 1 to Sentinel 2 RBG-NIR group
        x_sen1 = jnp.reshape(x_sen1, (-1, h * w, self.hidden_size))  # (B, L, D)
        RGBN_GROUP_IDX = 0
        x = x_sen2.at[:, :, RGBN_GROUP_IDX, :].add(x_sen1)
      elif self.multimodal_type == 'early_fuse_s1_to_all':
        x = x_sen2 + jnp.expand_dims(x_sen1, axis=2)

    elif self.multimodal_type == 'early_concat_s2_and_s1' or self.multimodal_type == 'early_concat_s2_s1_dem':
      x = x_sen2
    elif self.multimodal_type == 'early_concat_s2_and_s1_early_fusion_dem':
      dem = jnp.expand_dims(x[:, :, :, 20], axis=-1)  # (B, H, W, 1)
      dem = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                    name='embedding_dem')(dem)  # [B, Hp, Wp, hidden_size]
      dem = jnp.reshape(dem, (-1, h * w, self.hidden_size))  # (B, L, D)
      dem = jnp.expand_dims(dem, axis=2)  # (B, L, 1, D)

      x = (x_sen2 # sen1 and sen2 concatenated  # (B, L, G, D)
           + dem
           )  # (B, L, G, D)


    # Possibly dropping some tokens
    # 1. Sample tokens across sequence dim
    idx_kept_tokens = None
    idx_kept_groups = None
    n_tokens = x.shape[1]
    if seqlen > 0:
      rng = self.make_rng('droptok')
      idx_kept_tokens = token_indexes_not_to_drop(
        seqlen, n_tokens, seqlen_selection, rng)
      if len(idx_kept_tokens) < n_tokens:
        x = jnp.take(x, idx_kept_tokens, axis=1)

    b, L, G, _ = x.shape
    if self.sen2_maintain_seqlen:  # maintain_seqlen means sample across groups dim
      # 2. Sample tokens across channel groups to maintain sequence length
      rng = self.make_rng('changroup')
      idx_kept_groups = jax.random.choice(
        rng, a=G, shape=(b, L), p=jnp.array(self.changroups_sampling_weights))
      x = jnp.take_along_axis(x, idx_kept_groups[:, :, None, None], axis=2)
      x = x.squeeze(axis=2)
    else:
      x = x.reshape(b, L * G, self.hidden_size)

    return x, idx_kept_tokens, idx_kept_groups, G


class Sen2ToTokenSequence(nn.Module):
  """Sentinel 2 Grouped Channel embedding"""

  patches: ml_collections.ConfigDict
  hidden_size: int
  channel_embed_size: int = 128  # TODO: Expose as config
  posembs: Tuple[int, int] = (14, 14)
  positional_embedding: str = 'learned'
  channel_groups: Tuple[Tuple[int]] = ((1, 2, 3, 7), (4, 5, 6, 8), (10, 11)),
  maintain_seqlen: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, seqlen: int = -1,
               seqlen_selection: str = 'unstructured'):
    fh, fw = self.patches.size
    G = len(self.channel_groups)

    # Grouped Channel Patch Embedding
    x_grouped = []
    for idx, group in enumerate(self.channel_groups):
      x_group = x[:, :, :, group]
      x_group = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                        name=f'embedding-{idx}')(x_group)  # (B, H, W, channels_in_group) -> (B, Hp, Wp, D)
      x_grouped.append(x_group)

    x = jnp.stack(x_grouped, axis=3)  # (B, Hp, Wp, G, D)

    # Add spectral and positional embedding
    # Following SatMAE
    chanemb = self.param(
      'channel_embed_input',
      nn.initializers.normal(stddev=1 / np.sqrt(self.channel_embed_size)),
      (1, G, self.channel_embed_size), x.dtype)  # (1, G, Dc)

    posembed_size = self.hidden_size - self.channel_embed_size
    posemb = self.param(
      'posembed_input',
      nn.initializers.normal(stddev=1 / np.sqrt(posembed_size)),
      (1, self.posembs[0], self.posembs[1], posembed_size), x.dtype)  # (1, Hp, Wp, Dp)

    _, h, w, _, _ = x.shape

    if (h, w) != self.posembs:
      posemb = jax.image.resize(posemb, (1, h, w, posembed_size), 'bilinear')

    # Expand channel embedding to add spatial dims
    _chanemb = jnp.expand_dims(chanemb, axis=(1, 2))  # (1, 1, 1, G, Dc)
    _chanemb = jnp.broadcast_to(
      _chanemb, (1, h, w, G, self.channel_embed_size))  # (1, Hp, Wp, G, Dc)
    # Expand positional embedding to add group dim
    _posemb = jnp.expand_dims(posemb, axis=3)  # (1, Hp, Wp, 1, Dp)
    _posemb = jnp.broadcast_to(
      _posemb, (1, h, w, G, posembed_size))  # (1, Hp, Wp, G, Dp)

    # Concatenate channel and positional embeddings
    pos_chan_embed = jnp.concatenate((_chanemb, _posemb), axis=-1)  # (1, Hp, Wp, G, D)

    x = x + pos_chan_embed
    x = jnp.reshape(x, (-1, h * w, G, self.hidden_size))

    # Possibly dropping some tokens.
    idx_kept_tokens = None
    n_tokens = x.shape[1]
    if seqlen > 0:
      rng = self.make_rng('droptok')
      idx_kept_tokens = token_indexes_not_to_drop(
        seqlen, n_tokens, seqlen_selection, rng)
      if len(idx_kept_tokens) < n_tokens:
        x = jnp.take(x, idx_kept_tokens, axis=1)

    b, L, _, _ = x.shape
    if not self.maintain_seqlen:
      return x.reshape(b, L * G, self.hidden_size), idx_kept_tokens
    else:
      rng = self.make_rng('changroup')
      groups_to_keep = jax.random.randint(rng, shape=(b, L), minval=0, maxval=G)
      x = jnp.take_along_axis(x, groups_to_keep[:, :, None, None], axis=2)
      x = x.squeeze(axis=2)

      return x, idx_kept_tokens


class ToTokenSequence(nn.Module):
  """Transform a batch of views into a sequence of tokens."""

  patches: ml_collections.ConfigDict
  hidden_size: int
  posembs: Tuple[int, int] = (14, 14)
  positional_embedding: str = 'learned'

  def add_positional_encodings(self, x: jnp.ndarray,
                               positional_embedding: str = '') -> jnp.ndarray:
    """Add positional encodings to the input patch sequence."""
    _, h, w, c = x.shape
    positional_embedding = positional_embedding or self.positional_embedding
    if positional_embedding == 'learned':
      posemb = self.param(
        'posembed_input',
        nn.initializers.normal(stddev=1 / np.sqrt(c)),
        (1, self.posembs[0], self.posembs[1], c), x.dtype)
      # Optionally resize the positional encodings.
      if (h, w) != self.posembs:
        posemb = jax.image.resize(posemb, (1, h, w, c), 'bilinear')
      x = x + posemb
    elif positional_embedding == 'sinusoidal_2d':
      x = attention_layers.AddFixedSinCosPositionEmbedding()(x)
    x = jnp.reshape(x, (-1, h * w, c))
    return x

  @nn.compact
  def __call__(self, x: jnp.ndarray, positional_embedding: str = '',
               seqlen: int = -1, seqlen_selection: str = 'unstructured'):
    # Extracting patches and then embedding is in fact a single convolution.
    fh, fw = self.patches.size
    x = nn.Conv(self.hidden_size, (fh, fw), strides=(fh, fw), padding='VALID',
                name='embedding')(x)  # [B, H // fh, W // fw, hidden_size]

    # Adding positional encodings.
    x = self.add_positional_encodings(x, positional_embedding)

    # Possibly dropping some tokens.
    idx_kept_tokens = None
    # n_tokens = self.posembs[0] * self.posembs[1]
    n_tokens = x.shape[1]
    if seqlen > 0:
      rng = self.make_rng('droptok')
      idx_kept_tokens = token_indexes_not_to_drop(
        seqlen, n_tokens, seqlen_selection, rng)
      if len(idx_kept_tokens) < n_tokens:
        x = jnp.take(x, idx_kept_tokens, axis=1)

    return x, idx_kept_tokens


def token_indexes_not_to_drop(seqlen, n_tokens, seqlen_selection, rng):
  """Returns only the token indexes to keep in a sequence of tokens."""
  idx_kept_tokens = jnp.arange(n_tokens)
  if seqlen >= 0 and seqlen <= n_tokens:
    if seqlen_selection in ['consecutive', 'first']:
      if seqlen_selection == 'first':
        offset = 0
      else:
        offset = jax.random.randint(rng, (1,), 0, n_tokens - seqlen + 1)[0]
      # Workaround because jnp.arange(offset, offset + seqlen) causes
      # a ConcretizationError (even though shape is known to be seqlen...)
      idx_kept_tokens = jnp.ones(seqlen) * offset + jnp.arange(seqlen)
    elif seqlen_selection == 'unstructured':
      idx_kept_tokens = jax.random.permutation(rng, n_tokens)[:seqlen]
  idx_kept_tokens = jnp.asarray(idx_kept_tokens, dtype=jnp.int32)
  return idx_kept_tokens


class ViT4LOCA(nn.Module):
  """Vision Transformer model for LOCA training.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    n_ref_positions: Number of position in the reference view.
    apply_cluster_loss: Whether to apply the clustering loss.
    head_hidden_dim: Dimension of the hidden layer in the projection mlp.
    head_bottleneck_dim: Dimension of the bottleneck.
    head_output_dim: Dimension of the output ("number of prototypes").
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: Stochastic depth.
    posembs: Positional embedding size.
    dtype: JAX data type for activations.
  """

  multimodal_type: Optional[str]
  sen2grouped: bool
  sen2changroups: Tuple[Tuple[int]]
  sen2grouped_maintain_seqlen: bool
  changroups_sampling_weights: Optional[Tuple[int]]
  use_same_group_attn_mask: bool
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  n_ref_positions: int
  apply_cluster_loss: bool
  head_hidden_dim: int
  head_bottleneck_dim: int
  head_output_dim: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.1
  posembs: Tuple[int, int] = (14, 14)
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, inputs_kv: Optional[jnp.ndarray] = None,
               inputs_kv_kept_groups: Optional[jnp.ndarray] = None,
               train: bool, seqlen: int = -1, use_pe: bool = True,
               drop_moment: str = 'early',
               seqlen_selection: str = 'unstructured',
               debug: bool = False, sow_weights: bool = False):
    del debug
    # Input image -> sequence of patch tokens.
    num_channels = None
    idx_kept_groups = None
    if self.multimodal_type:
      x, idx_kept_tokens, idx_kept_groups, num_channels = MultiModalToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs,
        sen2_channel_groups=self.sen2changroups,
        sen2_maintain_seqlen=self.sen2grouped_maintain_seqlen,
        changroups_sampling_weights=self.changroups_sampling_weights,
        multimodal_type=self.multimodal_type
      )(
        x, seqlen=seqlen if drop_moment == 'early' else -1,
        seqlen_selection=seqlen_selection, drop_moment=drop_moment
      )
    elif self.sen2grouped:
      x, idx_kept_tokens = Sen2ToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs,
        channel_groups=self.sen2changroups,
        maintain_seqlen=self.sen2grouped_maintain_seqlen
      )(
        x, seqlen=seqlen if drop_moment == 'early' else -1,
        seqlen_selection=seqlen_selection
      )
    else:
      to_token_fn = ToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs)
      x, idx_kept_tokens = to_token_fn(
        x, seqlen=seqlen if drop_moment == 'early' else -1,
        positional_embedding=None if use_pe else 'pe_not_in_use',
        seqlen_selection=seqlen_selection)


    same_group_attn_mask = None
    if self.use_same_group_attn_mask:
      same_group_attn_mask = nn.make_attention_mask(
        idx_kept_groups,
        idx_kept_groups,
        pairwise_fn=lambda x, y: jnp.not_equal(x, y).astype(jnp.float32),
      )

    # ViT Encoder.
    for lyr in range(self.num_layers):
      x = vit.Encoder1DBlock(
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
                         self.stochastic_depth,
        name=f'encoderblock_{lyr}',
        dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
        x, attn_mask=same_group_attn_mask, deterministic=not train, sow_weights=True)
    x = nn.LayerNorm(name='encoder_norm')(x)

    # Optionally apply a clustering prediction loss.
    cluster_pred_outputs = None
    if self.apply_cluster_loss:
      cluster_pred_outputs = ProjectionHead(
        hidden_dim=self.head_hidden_dim,
        bottleneck_dim=self.head_bottleneck_dim,
        output_dim=self.head_output_dim,
        name='projection_head_for_clustering_prediction')(
        x, train).reshape((-1, self.head_output_dim))  # [B * n_patches, D_cluster]

    patches_repr = x
    # Drop some tokens (in the reference view).
    if drop_moment == 'late':
      rng = self.make_rng('droptok')
      idx_kept_tokens = token_indexes_not_to_drop(
        seqlen, self.n_ref_positions, seqlen_selection, rng)
      if len(idx_kept_tokens) < self.n_ref_positions:
        patches_repr = jnp.take(patches_repr, idx_kept_tokens, axis=1)
        if idx_kept_groups is not None:
          idx_kept_groups = jnp.take(idx_kept_groups, idx_kept_tokens, axis=1)

    # Query patches look at those of the reference through cross attention.
    same_group_cross_attn_mask = None
    if inputs_kv is None:
      inputs_kv = copy.deepcopy(patches_repr)
    if self.use_same_group_attn_mask and inputs_kv_kept_groups is not None:
      same_group_cross_attn_mask = nn.make_attention_mask(
        idx_kept_groups,
        inputs_kv_kept_groups,
        pairwise_fn=lambda x, y: jnp.not_equal(x, y).astype(jnp.float32),
      )
    x = CrossAttentionEncoderBlock(
      mlp_dim=self.mlp_dim,
      num_heads=self.num_heads,
      dropout_rate=self.dropout_rate,
      attention_dropout_rate=self.attention_dropout_rate,
      name='cross_attention_block',
      dtype=jax.dtypes.canonicalize_dtype(self.dtype))(
      x, inputs_kv=inputs_kv, deterministic=not train, attn_mask=same_group_cross_attn_mask, sow_weights=sow_weights)
    x = nn.LayerNorm(name='final_norm')(x)
    x = nn.Dense(self.n_ref_positions, name='position_predictor')(x)
    return x, cluster_pred_outputs, patches_repr, idx_kept_tokens, num_channels, idx_kept_groups


def norm_kernel_init_fn(rng, shape, dtype):
  """Initialize kernel with l2 normalized columns."""
  param = nn.linear.default_kernel_init(rng, shape, dtype)
  param /= (jnp.linalg.norm(param, axis=0, keepdims=True) + 1e-10)
  return param


class ProjectionHead(nn.Module):
  """Projection head.

  Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    n_layers: Depth of the projection head.
  """
  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 4096
  n_layers: int = 2

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
    for i in range(self.n_layers):
      x = nn.Dense(self.hidden_dim)(x)
      x = nn.gelu(x)
      x = nn_layers.IdentityLayer(name=f'mlp_{i}')(x)
    x = nn.Dense(self.bottleneck_dim)(x)
    # Normalize.
    x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = WeightNormDense(self.output_dim, use_bias=False, name='prototypes',
                        kernel_init=norm_kernel_init_fn)(x)
    return x


class WeightNormDense(nn.Dense):
  """Linear layer with weight normalized kernel."""

  def param(self, name: str, *args, **kwargs):
    param = super().param(name, *args, **kwargs)
    if name == 'kernel':
      param /= (jnp.linalg.norm(param, axis=0, keepdims=True) + 1e-10)
    return param


class CrossAttentionEncoderBlock(vit.Encoder1DBlock):
  """Transformer layer with cross-attention."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, inputs_kv: jnp.ndarray,
               deterministic: bool, attn_mask: jnp.ndarray = None,
               sow_weights: bool = False) -> jnp.ndarray:
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    inputs_kv = nn.LayerNorm(dtype=self.dtype)(inputs_kv)
    x = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dtype=self.dtype,
      kernel_init=nn.initializers.xavier_uniform(),
      broadcast_dropout=False,
      deterministic=deterministic,
      dropout_rate=self.attention_dropout_rate)(x, inputs_kv, mask=attn_mask, sow_weights=sow_weights)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
      mlp_dim=self.mlp_dim,
      dtype=self.dtype,
      dropout_rate=self.dropout_rate,
      activation_fn=nn.gelu,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))(
      y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class ViTLOCAModel(base_model.BaseModel):
  """Vision Transformer model for LOCA training."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ViT4LOCA(
      multimodal_type=self.config.get('multimodal', None),
      sen2grouped=self.config.get('sen2grouped', False),
      sen2changroups=self.config.get('sen2changroups', None),
      sen2grouped_maintain_seqlen=self.config.get('sen2grouped_maintain_seqlen', False),
      changroups_sampling_weights=self.config.get('changroups_sampling_weights', None),
      use_same_group_attn_mask=self.config.get('use_same_group_attn_mask', False),
      mlp_dim=self.config.model.mlp_dim,
      num_layers=self.config.model.num_layers,
      num_heads=self.config.model.num_heads,
      patches=self.config.model.patches,
      hidden_size=self.config.model.hidden_size,
      n_ref_positions=self.config.n_ref_positions,
      apply_cluster_loss=self.config.apply_cluster_loss,
      head_hidden_dim=self.config.model.get('head_hidden_dim', 2048),
      head_bottleneck_dim=self.config.model.get('head_bottleneck_dim', 256),
      head_output_dim=self.config.model.get('head_output_dim', 1024),
      dropout_rate=self.config.model.get('dropout_rate', 0.0),
      attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                   0.0),
      stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
      posembs=self.config.model.get('posembs', (14, 14)),
      dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
      'model':
        dict(
          num_heads=2,
          num_layers=1,
          mlp_dim=32,
          dropout_rate=0.,
          attention_dropout_rate=0.,
          hidden_size=16,
          head_hidden_dim=32,
          head_bottleneck_dim=16,
          head_output_dim=64,
          patches={'size': (4, 4)},
          data_dtype_str='float32')
    })

  def get_metrics_fn(self, split: Optional[str] = None):
    del split
    return functools.partial(
      classification_model.classification_metrics_function,
      target_is_onehot=True,
      metrics=dict(
        {'accuracy': (
          model_utils.weighted_correctly_classified,
          model_utils.num_examples),
          'loss': (
            model_utils.weighted_unnormalized_softmax_cross_entropy,
            model_utils.num_examples)}))

  def loss_function(self,
                    predictions: jnp.ndarray,
                    targets: jnp.ndarray,
                    weights: Optional[jnp.ndarray] = None) -> float:
    """Returns the cross-entropy loss."""
    loss = model_utils.weighted_softmax_cross_entropy(predictions, targets,
                                                      weights)
    return loss  # pytype: disable=bad-return-type
