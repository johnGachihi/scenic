# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed_with_resolution


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self, global_pool=False, patch_size=16, in_chans=3, embed_dim=1024, **kwargs
    ):
        super().__init__(embed_dim=embed_dim, **kwargs)

        self.patch_embed = PatchEmbedUnSafe(
            img_size=kwargs["img_size"],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, input_res=None):
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res=input_res)
        x = self.head(x)
        return x

class ViT_SimpleCNNDecoder(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, patch_size, embed_dim=768, in_chan=3, global_pool='', **kwargs):
        super().__init__(embed_dim=embed_dim, **kwargs)

        self.patch_embed = PatchEmbedUnSafe(
            img_size=kwargs["img_size"],
            patch_size=patch_size,
            in_chans=in_chan,
            embed_dim=embed_dim,
        )

        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 3, 2, padding=1, output_padding=1),
            nn.ConvTranspose2d(192, 96, 3, 2, padding=1, output_padding=1),
            nn.ConvTranspose2d(96, 48, 3, 2, padding=1, output_padding=1),
            nn.ConvTranspose2d(48, 2, 3, 2, padding=1, output_padding=1),
        )

        self.head = nn.Conv2d(2, kwargs['num_classes'], kernel_size=1, stride=1)

    def forward_features(self, x, input_res=None):
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )

        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches ** 0.5),
            input_res,
            cls_token=False,
            device=x.device,
        )

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, input_res=None):
        x = self.forward_features(x, input_res=input_res)

        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], x.shape[1], 14, 14)

        x = self.upsample_layers(x)
        x = self.head(x)
        return x

def vit_simple_cnn_decoder_small_patch16(**kwargs):
    model = ViT_SimpleCNNDecoder(
        in_chan=12,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
