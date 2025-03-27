import os

import numpy as np
import tensorflow_datasets as tfds
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from .airound import AIROUND_DATASET_STATS
from .cvbrct import CVBRCT_DATASET_STATS
from .eurosat import EUROSAT_DATASET_STATS
from .fmow import FMOW_DATASET_STATS, build_fmow
from .imagelist import ImageList
from .imagenet100 import build_imagenet_sampler
from .mlrsnet import MLRSNET_DATASET_STATS
from .naip import build_naip_sampler
from .optimal import OPTIMAL_DATASET_STATS
from .resic45 import RESIC_DATASET_STATS, build_resic
from .sentinel2 import build_sentinel_sampler
from .ucmerced import UCMERCED_DATASET_STATS
from .whurs import WHURS_DATASET_STATS
from .xview import build_xview2_sampler

dataset_stats_lookup = {
    "airound": AIROUND_DATASET_STATS,
    "cvbrct": CVBRCT_DATASET_STATS,
    "mlrsnet": MLRSNET_DATASET_STATS,
    "resisc": RESIC_DATASET_STATS,
    "eurosat": EUROSAT_DATASET_STATS,
    "optimal-31": OPTIMAL_DATASET_STATS,
    "whu-rs19": WHURS_DATASET_STATS,
    "ucmerced": UCMERCED_DATASET_STATS,
    "fmow": FMOW_DATASET_STATS,
}


def get_dataset_and_sampler(
    args,
    config,
    split="train",
    num_replicas=None,
    rank=None,
    transforms=None,
    transforms_init=None,
    linprobe_finetune=False,
):
    dataset_type = config["data"]["type"]
    if dataset_type == "NAIP":
        return build_naip_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "SENTINEL2":
        return build_sentinel_sampler(config, args, num_replicas, rank, transforms)
    elif dataset_type == "XView2":
        return build_xview2_sampler(
            config=config,
            num_replicas=num_replicas,
            rank=rank,
            transforms=transforms,
            split=split,
        )
    elif dataset_type == "ImageNet":
        return build_imagenet_sampler(
            config=config, num_replicas=num_replicas, rank=rank, transforms=transforms
        )
    elif dataset_type in ["fmow"]:
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"],
            transform=transforms_init,
            is_valid_file=is_fmow_rgb,
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type in ["mmearth"]:
        # load tfds multimodal dataset
        builder = tfds.builder('mm_earth_builder')
        builder.download_and_prepare(file_format='array_record')
        tfds_dataset = builder.as_data_source(split='train')

        dataset = MMearthSen2DatasetFromTFDS(tfds_dataset, transforms=transforms_init)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )
        return (
            dataset,
            sampler_train,
            TransformCollateFn(transforms, args.base_resolution),
        )

    elif dataset_type == "resisc":
        dataset = build_resic(config["data"]["img_dir"], transforms=transforms_init)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )
        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    elif dataset_type == "eurosat":
        dataset = datasets.ImageFolder(
            root=config["data"]["img_dir"], transform=transforms_init
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=True
        )

        if not linprobe_finetune:
            return (
                dataset,
                sampler_train,
                TransformCollateFn(transforms, args.base_resolution),
            )
        else:
            return (
                dataset,
                sampler_train,
                TransformCollateFnLabel(transforms, args.base_resolution),
            )
    else:
        raise NotImplementedError


def is_fmow_rgb(fname: str) -> bool:
    return fname.endswith("_rgb.jpg")


class TransformCollateFn:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        if not isinstance(samples[0], tuple):  # MMEarth Sen2 RGB
            imgs = torch.stack(samples)
        else:
            imgs = torch.stack(list(zip(*samples))[0])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res), None


class TransformCollateFnLabel:
    def __init__(self, transforms, base_resolution=1.0):
        self.transforms = transforms
        self.base_resolution = base_resolution

    def __call__(self, samples):
        imgs = torch.stack(list(zip(*samples))[0])
        labels = torch.tensor([x[1] for x in samples])
        imgs, imgs_src, ratios, _, _ = self.transforms(imgs)
        res = ratios * self.base_resolution
        imgs_src_res = res * (imgs.shape[-1] / imgs_src.shape[-1])
        return (imgs_src, imgs_src_res, imgs, res, labels), None


def get_eval_dataset_and_transform(
    eval_dataset_id="resisc",
    eval_dataset_path="~/data/resisc",
    transforms_init=None,
    args=None,
):
    # All of these datasets are ImageFolders
    if eval_dataset_id in [
        "resisc",
        "mlrsnet",
        "airound",
        "cvbrct",
        "eurosat",
        "optimal-31",
        "whu-rs19",
        "ucmerced",
    ]:
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        transform_normalize = transforms.Normalize(
            mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
        )
        use_transforms = [transforms.ToTensor(), transform_normalize]
        if transforms_init:
            use_transforms.insert(0, transforms_init)
        if eval_dataset_id == 'ucmerced':
            use_transforms.insert(0, transforms.Resize((256,256)))
        transform_eval = transforms.Compose(use_transforms)

        if os.path.isdir(eval_dataset_path):
            dataset_eval = ImageFolder(eval_dataset_path, transform=transform_eval)
        else:
            dataset_eval = ImageList(eval_dataset_path, transform=transform_eval)

    elif eval_dataset_id == "fmow":
        ds_stats = dataset_stats_lookup[eval_dataset_id]
        if transforms_init and args:
            transform_eval = transforms.Compose(
                [
                    # Resize only the short side
                    transforms.Resize(args.eval_scale),
                    # TODO this may not be the right thing to do here.
                    transforms.CenterCrop(args.eval_scale),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        else:
            transform_eval = transforms.Compose(
                [
                    # TODO remove hardcoding px size?
                    transforms.Resize(512),  # downsample short side to 512
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=ds_stats.PIXEL_MEANS, std=ds_stats.PIXEL_STD
                    ),
                ]
            )
        dataset_eval = build_fmow(eval_dataset_path, transforms=transform_eval)

    else:
        raise NotImplementedError

    return dataset_eval, transform_eval


class MMearthSen2DatasetFromTFDS(torch.utils.data.Dataset):
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

    def __init__(self, tfds_dataset, transforms=None):
        self.tfds_dataset = tfds_dataset
        self.transforms = transforms

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

        if self.transforms:
            sample = self.transforms(sample)

        sample = sample[[3, 2, 1]]

        return sample