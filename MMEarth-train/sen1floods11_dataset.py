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
