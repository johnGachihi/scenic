import json
from collections import OrderedDict

import h5py
from pathlib import Path

from .mmearth_modalities import MODALITIES_FULL, NO_DATA_VAL

import tensorflow_datasets as tfds
import numpy as np


class MMEarthBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.1')

    def __init__(self, modalities: dict, **kwargs):
        super().__init__(**kwargs)
        self.modalities = modalities

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'sentinel2': tfds.features.Tensor(shape=(12, 64, 64), dtype=np.dtype("float32")),
                'sentinel1': tfds.features.Tensor(shape=(8, 64, 64), dtype=np.dtype("float32")),
                'id': tfds.features.Text(),
            }),
        )

    def _split_generators(self, dl_manager):
        data_root = Path('/home/admin/john/data/mmearth')

        # Full data
        data_path = data_root / 'data_1M_v001_64.h5'
        data_full = h5py.File(data_path, 'r')

        # Split indices
        splits_path = data_root / 'data_1M_v001_64_splits.json'
        with open(splits_path, "r") as f:
            indices = json.load(f)["train"][:10000]

        # Tile info
        tile_info_path = data_root / 'data_1M_v001_64_tile_info.json'
        with open(tile_info_path, "r") as f:
            tile_info = json.load(f)

        # Band norm stats
        band_stats_path = data_root / 'data_1M_v001_64_band_stats.json'
        with open(band_stats_path, "r") as f:
            norm_stats = json.load(f)

        return {
            'train': self._generate_examples(data_full, indices, tile_info, norm_stats)
        }

    def _generate_examples(self, data_full, indices, tile_info, norm_stats):
        for idx in indices:
            return_dict = OrderedDict()
            name = data_full['metadata'][idx][0].decode("utf-8")
            l2a = tile_info[name]["S2_type"] == "l2a"

            for modality in self.modalities.keys():
                # Get band indices
                if self.modalities[modality] == "all":
                    modality_idx = [i for i in range(len(MODALITIES_FULL[modality]))]
                else:
                    modality_idx = [MODALITIES_FULL[modality].index(m) for m in self.modalities[modality]]

                # Get data
                data = data_full[modality][idx, modality_idx, ...]
                data = np.array(data)

                # inside the band_stats, the name for sentinel2 is sentinel2_l1c or sentinel2_l2a
                if modality == "sentinel2":
                    modality_ = "sentinel2_l2a" if l2a else "sentinel2_l1c"
                else:
                    modality_ = modality

                means = np.array(norm_stats[modality_]["mean"])[modality_idx]
                stds = np.array(norm_stats[modality_]["std"])[modality_idx]
                data = (data - means[:, None, None]) / stds[:, None, None]  # Why the `None`s

                # converting the nodata values to nan to keep everything consistent
                data = (
                    np.where(data == NO_DATA_VAL[modality], np.nan, data)
                    if modality != "dynamic_world"
                    else data
                )

                data = data.astype(np.dtype("float32"))

                return_dict[modality] = data

            return_dict["id"] = name

            yield name, return_dict
