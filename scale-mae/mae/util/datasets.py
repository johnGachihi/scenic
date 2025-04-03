# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os

import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args, config):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config["data"]["input_size"],
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if config["data"]["input_size"] <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(config["data"]["input_size"] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        )  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config["data"]["input_size"]))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


import numpy as np
import torch
import tensorflow_datasets as tfds
import torchvision as tv


class Sen1Floods11Dataset(torch.utils.data.Dataset):
  def __init__(self, split):
    self.split = split

    # Load the dataset from the TensorFlow Datasets
    builder = tfds.builder('sen1_floods11')
    builder.download_and_prepare(file_format='array_record')
    self.tfds_ds = builder.as_data_source(split=split)

  def __len__(self):
    return len(self.tfds_ds)

  def __getitem__(self, idx):
    sample = self.tfds_ds[idx]
    img = sample['s2_img']
    label = sample['label']

    # Leave out 11th band
    img = img[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]

    # Convert to float32
    img = img.astype(np.float32)

    img = torch.from_numpy(img)
    label = torch.from_numpy(label)

    # Resize
    img = tv.transforms.Resize(224)(img)
    label = (tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.NEAREST)
             (label))

    return img, label


class Spacenet1Dataset(torch.utils.data.Dataset):
  def __init__(self, split):
    builder = tfds.builder('spacenet1')
    builder.download_and_prepare(file_format='array_record')
    self.tfds_ds = builder.as_data_source(split=split)

  def __len__(self):
    return len(self.tfds_ds)

  def __getitem__(self, idx):
    sample = self.tfds_ds[idx]
    image = sample['image']
    label = sample['label']

    # Convert to PyTorch tensors
    image = torch.tensor(image, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int16)
    label = label.unsqueeze(0)

    # Resize to 112
    image = tv.transforms.Resize((112, 112))(image)
    label = tv.transforms.Resize(
      (112, 112), interpolation=tv.transforms.InterpolationMode.NEAREST)(label)

    return image, label