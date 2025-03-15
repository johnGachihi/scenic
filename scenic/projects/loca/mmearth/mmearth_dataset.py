import json
from collections import OrderedDict

import h5py
from pathlib import Path

from .mmearth_modalities import MODALITIES, MODALITIES_FULL, NO_DATA_VAL

import tensorflow_datasets as tfds
import numpy as np


class MMEarthBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.4')

    # def __init__(self, modalities: dict, **kwargs):
    #     super().__init__(**kwargs)
    #     self.modalities = modalities

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'sentinel2': tfds.features.Tensor(
                    shape=(12, 64, 64), 
                    dtype=np.dtype("uint16"),
                    encoding="zlib"
                ),
                'sentinel1': tfds.features.Tensor(
                    shape=(8, 64, 64), 
                    dtype=np.dtype("float32"),
                    encoding="zlib"
                ),
                'id': tfds.features.Text(),
                'sentinel2_type': tfds.features.Text(),  # l1c or l2a
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
            indices = json.load(f)["train"][:300_000]

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
            sentinel2_type = tile_info[name]["S2_type"]

            for modality in MODALITIES.keys():
                # Get band indices
                if MODALITIES[modality] == "all":
                    modality_idx = [i for i in range(len(MODALITIES_FULL[modality]))]
                else:
                    modality_idx = [MODALITIES_FULL[modality].index(m) for m in MODALITIES[modality]]

                # Get data
                data = data_full[modality][idx, modality_idx, ...]
                data = np.array(data)

                # convert nodata values to 0
                data = np.where(data == NO_DATA_VAL[modality], 0, data)

                if modality == "sentinel2":
                    data = data.astype(np.dtype("uint16"))
                else:
                    data = data.astype(np.dtype("float32"))

                return_dict[modality] = data

            return_dict["id"] = name
            return_dict["sentinel2_type"] = sentinel2_type

            yield name, return_dict
