import torch
import tensorflow_datasets as tfds
import torchvision as tv


class Spacenet1Dataset(torch.utils.data.Dataset):
  def __init__(self, split, args):
    self.split = split
    self.dropped_bands = args.dropped_bands

    # Load the dataset from the TensorFlow Datasets
    builder = tfds.builder('spacenet1')
    builder.download_and_prepare(file_format='array_record')
    self.tfds_ds = builder.as_data_source(split=split)

    self.in_c = 8  # Sentinel 2 has 13 bands
    if self.dropped_bands is not None:
      self.in_c -= len(self.dropped_bands)

  def __len__(self):
    return len(self.tfds_ds)

  def __getitem__(self, idx):
    sample = self.tfds_ds[idx]
    image = sample['image']
    label = sample['label']

    # Drop the specified bands
    if self.dropped_bands is not None:
      keep_idxs = [i for i in range(image.shape[0]) if i not in self.dropped_bands]
      image = image[keep_idxs, :, :]

    # Convert to PyTorch tensors
    image = torch.tensor(image, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int16)
    label = label.unsqueeze(0)

    # Resize to 112
    image = tv.transforms.Resize((112, 112))(image)
    label = (tv.transforms.Resize((112, 112), interpolation=tv.transforms.InterpolationMode.NEAREST)
             (label))

    return {'img': image, 'label': label}