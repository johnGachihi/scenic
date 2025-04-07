from typing import Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.projects.baselines import vit
from scenic.projects.loca.vit import ToTokenSequence, Sen2ToTokenSequence


class SemanticSegmentationModel(nn.Module):
  sen2grouped: bool
  sen2channel_groups: Tuple[Tuple[int]]
  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.1
  posembs: Tuple[int, int] = (14, 14)
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool, debug: bool = False) -> jnp.ndarray:
    _, h, w, _ = x.shape

    if self.sen2grouped:
      x, _ = Sen2ToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs,
        channel_groups=self.sen2channel_groups)(x)
    else:
      x, _ = ToTokenSequence(
        patches=self.patches,
        hidden_size=self.hidden_size,
        posembs=self.posembs)(x)

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
        x, deterministic=not train)
    patches_repr = nn.LayerNorm(name='encoder_norm')(x)

    b, L, D = patches_repr.shape
    hp, wp = h // self.patches.size[0], w // self.patches.size[1]
    if self.sen2grouped:
      # Reshape patches_repr to (b, Hp, Wp, G, D)
      G = len(self.sen2channel_groups)
      x = patches_repr.reshape(b, hp, wp, G, D)
      x = x.reshape(b, hp, wp, G * D)
    else:
      # TODO: Parameterize 14
      x = patches_repr.reshape(patches_repr.shape[0], 14, 14, patches_repr.shape[-1])

    # Gradually upsample spatial dimensions while reducing channels
    x = nn.ConvTranspose(192, kernel_size=(3, 3), strides=(2, 2))(x)  # 28x28, 192 channels
    x = nn.ConvTranspose(96, kernel_size=(3, 3), strides=(2, 2))(x)   # 56x56, 96 channels
    x = nn.ConvTranspose(48, kernel_size=(3, 3), strides=(2, 2))(x)   # 112x112, 48 channels
    x = nn.ConvTranspose(2, kernel_size=(3, 3), strides=(2, 2))(x)    # 224x224, 2 channels

    # Add a final segmentation head
    x = nn.Conv(self.num_classes, kernel_size=(1, 1))(x)
    x = x.reshape(patches_repr.shape[0], 224, 224, self.num_classes)
    return x


class SemSegModel(SegmentationModel):
  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return SemanticSegmentationModel(
      sen2grouped=self.config.get('sen2grouped', False),
      sen2channel_groups=self.config.get('sen2changroups', None),
      num_classes=self.config.model.num_classes,
      mlp_dim=self.config.model.mlp_dim,
      num_layers=self.config.model.num_layers,
      num_heads=self.config.model.num_heads,
      patches=self.config.model.patches,
      hidden_size=self.config.model.hidden_size,
      dropout_rate=self.config.model.get('dropout_rate', 0.0),
      attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                   0.0),
      stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
      posembs=self.config.model.get('posembs', (14, 14)),
      dtype=model_dtype
    )