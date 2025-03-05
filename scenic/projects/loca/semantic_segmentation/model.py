from typing import Tuple, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.projects.baselines import vit
from scenic.projects.loca.vit import ToTokenSequence


class SemanticSegmentationModel(nn.Module):
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

    # Reshape from sequence of tokens to BHWD
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

#   def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
#     del split
#     return functools.partial(
#       segmentation_model.semantic_segmentation_metrics_function,
#       target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
#       metrics=_SEMANTIC_SEGMENTATION_METRICS)
#
# _SEMANTIC_SEGMENTATION_METRICS = immutabledict({
#   'accuracy': (model_utils.weighted_correctly_classified, num_pixels),
#
#   # The loss is already normalized, so we set num_pixels to 1.0:
#   'loss': (model_utils.weighted_softmax_cross_entropy, lambda *a, **kw: 1.0),
#
#   # Mean IoU
#   'mean_iou': (model_utils.mean_iou, num_pixels),
# })