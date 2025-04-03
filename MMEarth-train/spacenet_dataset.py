import torch
import tensorflow_datasets as tfds
import torchvision as tv


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
    # print(image.shape, label.shape)
    # import pdb; pdb.set_trace()

    # Resize to 112
    image = tv.transforms.Resize((112, 112))(image)
    label = tv.transforms.Resize(
      (112, 112), interpolation=tv.transforms.InterpolationMode.NEAREST)(label)

    return image, label
