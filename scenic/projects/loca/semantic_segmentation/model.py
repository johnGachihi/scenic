from scenic.projects.loca.vit import ViTLOCAModel, ViT4LOCA
import flax.linen as nn
from scenic.model_lib.base_models.segmentation_model import SegmentationModel
import jax.numpy as jnp


class SemanticSegmentationModel(nn.Module):
  num_classes: int
  vit_loca_model: ViTLOCAModel

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool, debug: bool = False) -> jnp.ndarray:
    _, _, patches_repr, _ = self.vit_loca_model(x, train=train)

    # Reshape from sequence of tokens to BHWD
    # TODO: Parameterize 14
    x = patches_repr.reshape(patches_repr.shape[0], 14, 14, patches_repr.shape[-1])

    # Add a final segmentation head
    x = nn.Conv(self.num_classes, kernel_size=(1, 1))(x)
    return x


class SemSegModel(SegmentationModel):
  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return SemanticSegmentationModel(
      num_classes=self.config.model.num_classes,
      vit_loca_model=ViT4LOCA(
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        n_ref_positions=self.config.n_ref_positions,
        apply_cluster_loss=False,
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
      ,
    )
