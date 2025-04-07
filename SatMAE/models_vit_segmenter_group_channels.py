import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

from segmenter import MaskTransformer


class SegmenterGroupChannelsViT(VisionTransformer):
  def __init__(
      self,
      global_pool=False,
      channel_embed=256,
      channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
      decoder_embed_dim=384,
      decoder_num_layers=12,
      decoder_num_heads=6,
      decoder_mlp_ratio=4,
      **kwargs
  ):
    del global_pool
    super().__init__(**kwargs)

    img_size = kwargs['img_size']
    patch_size = kwargs['patch_size']
    in_c = kwargs['in_chans']
    embed_dim = kwargs['embed_dim']
    num_classes = kwargs['num_classes']
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

    self.head = MaskTransformer(
      n_cls=num_classes, patch_size=patch_size, d_encoder=embed_dim,
      n_layers=decoder_num_layers, n_heads=decoder_num_heads, d_model=decoder_embed_dim,
      d_ff=decoder_embed_dim * decoder_mlp_ratio, drop_path_rate=0.0, dropout=0.0
    )

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

    cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
    cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)

    x = self.pos_drop(x)

    hidden_feats = []
    for blk in self.blocks:
      x = blk(x)
      hidden_feats += [x]
    hidden_feats = hidden_feats[:-1]

    x = self.norm(x)

    return x[:, 1:], hidden_feats

  def forward(self, x):
    H, W = x.size(2), x.size(3)
    x, enc_feats = self.forward_features(x)
    masks = self.head(x, enc_feats, (H, W))
    masks = F.interpolate(masks, size=(H, W), mode="bilinear")
    return masks


def unpadding(y, target_size):
  H, W = target_size
  H_pad, W_pad = y.size(2), y.size(3)
  # crop predictions on extra pixels coming from padding
  extra_h = H_pad - H
  extra_w = W_pad - W
  if extra_h > 0:
    y = y[:, :, :-extra_h]
  if extra_w > 0:
    y = y[:, :, :, :-extra_w]
  return y