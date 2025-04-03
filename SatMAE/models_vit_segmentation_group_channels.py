import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class SimpleCNNSegmentationGroupChannelsVisionTransformer(VisionTransformer):
  """ Vision Transformer with support for global average pooling
  """

  def __init__(
      self,
      global_pool=False,  # Here for compatibility
      channel_embed=256,
      channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), **kwargs):
    super().__init__(**kwargs)
    del global_pool
    img_size = kwargs['img_size']
    patch_size = kwargs['patch_size']
    in_c = kwargs['in_chans']
    embed_dim = kwargs['embed_dim']

    self.channel_groups = channel_groups

    self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
                                      for group in channel_groups])
    num_patches = self.patch_embed[0].num_patches

    # Positional and channel embed
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    num_groups = len(channel_groups)
    self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
    chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
    self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

    # Extra embedding for cls to fill embed_dim
    self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
    channel_cls_embed = torch.zeros((1, channel_embed))
    self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

    self.upsample_layers = nn.Sequential(
      nn.ConvTranspose2d(num_groups * embed_dim, 192, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(192, 96, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(96, 48, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(48, 2, 3, 2, padding=1, output_padding=1),
    )

    self.head = nn.Conv2d(2, kwargs['num_classes'], kernel_size=1, stride=1)

  def forward_features(self, x):
    b, c, h, w = x.shape

    x_c_embed = []
    for i, group in enumerate(self.channel_groups):
      x_c = x[:, group, :, :]
      x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

    x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
    _, G, L, D = x.shape

    # add channel embed
    channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
    pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

    # Channel embed same across (x,y) position, and pos embed same across channel (c)
    channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
    pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
    pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

    # add pos embed w/o cls token
    x = x + pos_channel  # (N, G, L, D)
    x = x.view(b, -1, D)  # (N, G*L, D)

    x = self.pos_drop(x)

    x = self.blocks(x)
    x = self.norm(x)
    import pdb; pdb.set_trace()

    return x

  def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
    N, L, D = x.shape
    G = len(self.channel_groups)
    x = x.view(N, G, -1, D)  # (N, G, L // G, D)
    x = x.permute(0, 1, 3, 2).contiguous()  # (N, G, D, L // G)
    Hp = Wp = int((L // G)**0.5)
    x = x.view(N, G*D, Hp, Wp)  # (N, G*D, Hp, Wp)

    x = self.upsample_layers(x)
    x = self.head(x)

    return x


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.forward_features(x)
    x = self.forward_head(x)
    return x