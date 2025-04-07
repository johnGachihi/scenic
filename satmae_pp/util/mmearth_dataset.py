import numpy as np
import torch
from fiona.transform import transform
from torch.utils.data import Dataset
import tensorflow_datasets as tfds
import torchvision as tv
import torch.nn.functional as F


class MMearthSen2Dataset(torch.utils.data.Dataset):
  SENTINEL2_L2A_MEAN = np.array([1349.3977794889083, 1479.9521800379623, 1720.3688077425966, 1899.1848715975957,
                                 2253.9309600057886, 2808.2001963620132, 3003.424149045887, 3149.5364927329806,
                                 3110.840562275062, 3213.7636154015954, 2399.086213373806,
                                 1811.7986415136786])
  SENTINEL2_L2A_STD = np.array([2340.2916479338087, 2375.872101251672, 2256.8997709659416, 2354.181051828758,
                                2292.99569489449, 2033.2166835293804, 1920.1736418230105, 1816.6152354201365,
                                1988.1938283738782, 1947.9031620588928, 1473.224812450967,
                                1390.6781165633136])
  SENTINEL2_L1C_MEAN = np.array([1864.880176877408, 1656.9923384425733, 1561.2433627865414, 1633.6041005007844,
                                 1846.642924880421, 2422.1354550099322, 2706.1684786306714, 2907.509651871235,
                                 2620.484567631748, 974.4786592695265, 2154.0573745085508,
                                 1464.8020890030184])
  SENTINEL2_L1C_STD = np.array([1520.0684839687428, 1575.4239525583005, 1474.3747757041376, 1734.9206729983894,
                                1697.1412804437439, 1584.959816138674, 1577.9910344404889, 1560.2251591506092,
                                1519.2164490452863, 823.3855623314192, 1311.5885770761618,
                                1140.1057025823181])

  def __init__(self, args):
    self.dropped_bands = args.dropped_bands

    # Load the dataset from the TensorFlow Datasets
    builder = tfds.builder('mm_earth_builder', version=args.dataset_version)
    builder.download_and_prepare(file_format='array_record')
    self.tfds_dataset = builder.as_data_source(split="train")

    self.transforms = tv.transforms.Compose([
      tv.transforms.RandomResizedCrop(
        args.input_size * 4,
        scale=(0.6, 1.0),
        interpolation=tv.transforms.InterpolationMode.BICUBIC
      ),
      tv.transforms.RandomHorizontalFlip()
    ])

    self.in_c = 12  # My MMearth dataset excludes Sentinel 2's B10:Cirrus
    if self.dropped_bands is not None:
      self.in_c = self.in_c - len(self.dropped_bands)


  def __len__(self):
    return len(self.tfds_dataset)

  def __getitem__(self, idx):
    sample = self.tfds_dataset[idx]
    sentinel2_type = sample['sentinel2_type'].decode("utf-8")
    sample = sample['sentinel2']

    if sentinel2_type == "l2a":
      sample = (sample - self.SENTINEL2_L2A_MEAN[:, None, None]) / self.SENTINEL2_L2A_STD[:, None, None]
    else:
      sample = (sample - self.SENTINEL2_L1C_MEAN[:, None, None]) / self.SENTINEL2_L1C_STD[:, None, None]

    # set consistent no-data value across modalities
    sample = np.where(sample == 0, np.nan, sample)

    sample = torch.from_numpy(sample.astype(np.dtype('float32')))

    sample = self.transforms(sample)

    if self.dropped_bands is not None:
      keep_idxs = [i for i in range(sample.shape[0]) if i not in self.dropped_bands]
      sample = sample[keep_idxs, :, :]

    img_dn_2x = F.interpolate(sample.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
    img_dn_4x = F.interpolate(img_dn_2x.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)

    return {
      'img_up_4x':sample,
      'img_up_2x':img_dn_2x,
      'img':img_dn_4x
    }