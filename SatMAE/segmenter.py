import torch
import torch.nn as nn

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from einops import rearrange

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout, out_dim=None):
    super().__init__()
    self.fc1 = nn.Linear(dim, hidden_dim)
    self.act = nn.GELU()
    if out_dim is None:
      out_dim = dim
    self.fc2 = nn.Linear(hidden_dim, out_dim)
    self.drop = nn.Dropout(dropout)

  @property
  def unwrapped(self):
    return self

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class Attention(nn.Module):
  def __init__(self, dim, heads, dropout):
    super().__init__()
    self.heads = heads
    head_dim = dim // heads
    self.scale = head_dim ** -0.5
    self.attn = None

    self.qkv = nn.Linear(dim, dim * 3)
    self.attn_drop = nn.Dropout(dropout)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(dropout)

  @property
  def unwrapped(self):
    return self

  def forward(self, x, mask=None):
    B, N, C = x.shape
    qkv = (
      self.qkv(x)
      .reshape(B, N, 3, self.heads, C // self.heads)
      .permute(2, 0, 3, 1, 4)
    )
    q, k, v = (
      qkv[0],
      qkv[1],
      qkv[2],
    )

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x, attn


class Block(nn.Module):
  def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.attn = Attention(dim, heads, dropout)
    self.mlp = FeedForward(dim, mlp_dim, dropout)
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

  def forward(self, x, mask=None, return_attention=False):
    y, attn = self.attn(self.norm1(x), mask)
    if return_attention:
      return attn
    x = x + self.drop_path(y)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def init_weights(m):
  if isinstance(m, nn.Linear):
    trunc_normal_(m.weight, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.LayerNorm):
    nn.init.constant_(m.bias, 0)
    nn.init.constant_(m.weight, 1.0)


class MaskTransformer(nn.Module):
  def __init__(
      self,
      n_cls,
      patch_size,
      d_encoder,
      n_layers,
      n_heads,
      d_model,
      d_ff,
      drop_path_rate,
      dropout
  ):
    super().__init__()
    self.d_encoder = d_encoder
    self.patch_size = patch_size
    self.n_layers = n_layers
    self.n_cls = n_cls
    self.d_model = d_model
    self.d_ff = d_ff
    self.scale = d_model ** -0.5

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
    self.blocks = nn.ModuleList(
      [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
    )

    self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
    self.proj_dec = nn.Linear(d_encoder, d_model)

    self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
    self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

    self.decoder_norm = nn.LayerNorm(d_model)
    self.mask_norm = nn.LayerNorm(n_cls)

    self.apply(init_weights)
    trunc_normal_(self.cls_emb, std=0.02)

  @torch.jit.ignore
  def no_weight_decay(self):
    return {"cls_emb"}

  def forward(self, x, enc_feats, im_size):
    H, W = im_size
    GS = H // self.patch_size

    x = self.proj_dec(x)
    cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
    x = torch.cat((x, cls_emb), 1)
    for idx, blk in enumerate(self.blocks):
      if True and idx < len(enc_feats):
        # import pdb; pdb.set_trace()
        enc_feat = enc_feats[idx][:, :-1, :]
        tmp = x[:, :-self.n_cls:, :]
        x = torch.cat([tmp + enc_feat, x[:, -self.n_cls:, :]], dim=1)
        # x = torch.cat([x, enc_feats[idx]], dim=1)
      x = blk(x)
    x = self.decoder_norm(x)

    patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
    patches = patches @ self.proj_patch
    cls_seg_feat = cls_seg_feat @ self.proj_classes

    patches = patches / patches.norm(dim=-1, keepdim=True)
    cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

    masks = patches @ cls_seg_feat.transpose(1, 2)
    masks = self.mask_norm(masks)
    masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

    return masks