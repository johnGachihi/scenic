from argparse import Namespace
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from .norm_layers import LayerNorm
from .convnextv2_unet import Block

class ConvNeXtV2_SimpleCNNDecoder(nn.Module):
  def __init__(
      self,
      patch_size: int = 32,
      img_size: int = 224,
      in_chans: int = 3,
      num_classes: int = 1000,
      depths: list[int] = None,
      dims: list[int] = None,
      drop_path_rate: float = 0.0,
      head_init_scale: float = 1.0,
      use_orig_stem: bool = False,
      args: Namespace = None
  ):
    super().__init__()
    self.depths = depths
    if self.depths is None:
      self.depths = [3, 3, 9, 3]
    self.img_size = img_size
    self.patch_size = patch_size
    if dims is None:
      dims = [96, 192, 384, 768]

    self.use_orig_stem = use_orig_stem
    self.args = args
    self.downsample_layers = (
      nn.ModuleList()
    )
    self.num_stage = len(depths)
    if self.use_orig_stem:
      self.stem_orig = nn.Sequential(
        nn.Conv2d(
          in_chans,
          dims[0],
          kernel_size=patch_size // (2 ** (self.num_stage - 1)),
          stride=patch_size // (2 ** (self.num_stage - 1)),
        ),
        LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
      )
    else:
      self.initial_conv = nn.Sequential(
        nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
        LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        nn.GELU(),
      )
      self.stem = nn.Sequential(
        nn.Conv2d(
          dims[0],
          dims[0],
          kernel_size=patch_size // (2 ** (self.num_stage - 1)),
          stride=patch_size // (2 ** (self.num_stage - 1)),
          groups=dims[0],
        ),
        LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
      )

    for i in range(3):
      downsample_layer = nn.Sequential(
        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
      )
      self.downsample_layers.append(downsample_layer)

    self.stages = (
      nn.ModuleList()
    )  # 4 feature resolution stages, each consisting of multiple residual blocks
    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(self.num_stage):
      stage = nn.Sequential(
        *[
          Block(dim=dims[i], drop_path=dp_rates[cur + j])
          for j in range(depths[i])
        ]
      )
      self.stages.append(stage)
      cur += depths[i]

    self.upsample_layers = nn.Sequential(
      nn.ConvTranspose2d(768, 192, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(192, 96, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(96, 48, 3, 2, padding=1, output_padding=1),
      nn.ConvTranspose2d(48, 2, 3, 2, padding=1, output_padding=1),
    )

    self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
    self.head = nn.Conv2d(2, num_classes, kernel_size=1, stride=1)

  def encoder(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
    enc_features = []
    if self.use_orig_stem:
      x = self.stem_orig(x)
      enc_features.append(x)
    else:
      x = self.initial_conv(x)
      enc_features.append(x)
      x = self.stem(x)
      # self.tmp_var = x
      enc_features.append(x)

    x = self.stages[0](x)

    for i in range(3):
      x = self.downsample_layers[i](x)
      x = self.stages[i + 1](x)
      enc_features.append(x) if i < 2 else None

    # in total we only save 3 feature maps
    return x, enc_features

  def decoder(self, x: Tensor, enc_features: List[Tensor]) -> Tensor:
    x = self.upsample_layers(x)
    return x

  def forward(self, x: Tensor) -> Tensor:
    x, enc_features = self.encoder(x)
    x = self.decoder(x, enc_features)
    x = self.head(x)
    return x