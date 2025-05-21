import torch
import torchvision as tv
import tensorflow_datasets as tfds
import numpy as np


class Sen1Floods11Dataset(torch.utils.data.Dataset):
  def __init__(self, split, args):
    self.split = split
    self.dropped_bands = args.dropped_bands
    self.input_size = args.input_size

    # Load the dataset from the TensorFlow Datasets
    builder = tfds.builder('sen1_floods11', version=args.dataset_version)
    builder.download_and_prepare(file_format='array_record')
    self.tfds_ds = builder.as_data_source(split=split)

    self.in_c = 13  # Sentinel 2 has 13 bands
    if self.dropped_bands is not None:
      self.in_c -= len(self.dropped_bands)

  def __len__(self):
    return len(self.tfds_ds)

  def __getitem__(self, idx):
    sample = self.tfds_ds[idx]
    img = sample['s2_img']
    label = sample['label']

    # Convert to float32
    img = img.astype(np.float32)

    img = torch.from_numpy(img)
    label = torch.from_numpy(label)

    if self.dropped_bands is not None:
      keep_idxs = [i for i in range(img.shape[0]) if i not in self.dropped_bands]
      img = img[keep_idxs, :, :]

    # Resize
    img = tv.transforms.Resize(self.input_size)(img)
    label = (tv.transforms.Resize(self.input_size, interpolation=tv.transforms.InterpolationMode.NEAREST)
             (label))

    return img, label