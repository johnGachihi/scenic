from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch
import torch.nn as nn

from util.pos_embed import get_2d_sincos_pos_embed

class SimpleCNNSegmentationVisionTransformer(VisionTransformer):
  """ Vision Transformer with support for global average pooling
  """

  def __init__(
      self,
      global_pool=False,  # Here for compatibility
      **kwargs
  ):
    del global_pool
    super().__init__(**kwargs)

    embed_dim = kwargs['embed_dim']

    # Added by Samar, need default pos embedding
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                        cls_token=True)
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    self.upsample_layers = nn.Sequential(
      nn.ConvTranspose2d(embed_dim, 192, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(192, 96, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(96, 48, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(48, 2, 3, 2, padding=1, output_padding=1),
    )

    self.head = nn.Conv2d(2, kwargs['num_classes'], kernel_size=1, stride=1)

  def forward_features(self, x):
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
      x = blk(x)

    x = self.norm(x)

    return x[:, 1:]

  def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
    N, L, D = x.shape
    x = x.permute(0, 2, 1).contiguous()  # (N, D, L)
    Hp = Wp = int(L ** 0.5)
    x = x.view(N, D, Hp, Wp)  # (N, D, Hp, Wp)

    x = self.upsample_layers(x)
    x = self.head(x)

    return x


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.forward_features(x)
    x = self.forward_head(x)
    return x