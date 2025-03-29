import tensorflow_datasets as tfds
import rasterio
from pathlib import Path
import csv
import numpy as np

def read_img(img_fname):
    with rasterio.open(img_fname) as f:
        return f.read()

def s1_to_s2_filename(fname):
    """
    Eg. Ghana_5079_S1Hand.tif -> Ghana_5079_S2Hand.tif
    """
    
    split = fname.split('_')
    return '_'.join(split[:-1] + [split[-1].replace('S1', 'S2')])

class Sen1Floods11(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.2')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                's2_img': tfds.features.Tensor(shape=(13, 512, 512), dtype=np.dtype("int32")),
                'label': tfds.features.Tensor(shape=(1, 512, 512), dtype=np.dtype("int32")),
            }),
        )

    def _split_generators(self, dl_manager):
        # TODO: Extract absolute paths
        root_path = '/home/admin/satellite-loca/data/sen1floods11'
        train_split_csv = root_path + '/splits/flood_handlabeled/flood_train_data.csv'
        val_split_csv = root_path + '/splits/flood_handlabeled/flood_valid_data.csv'
        test_split_csv = root_path + '/splits/flood_handlabeled/flood_test_data.csv'

        data_root = root_path / Path('data/flood_events/HandLabeled/S2Hand')
        label_root = root_path / Path('data/flood_events/HandLabeled/LabelHand')

        return {
            'train': self._generate_examples(data_root, label_root, train_split_csv),
            'val': self._generate_examples(data_root, label_root, val_split_csv),
            'test': self._generate_examples(data_root, label_root, test_split_csv),
        }

    def _generate_examples(self, data_root, label_root, split_csv):
        with open(split_csv, 'r') as f:
            for line in csv.reader(f):
                s1_fname, label_fname = line
                s2_fname = s1_to_s2_filename(s1_fname)
                s2_img = read_img(data_root / s2_fname)
                label = read_img(label_root / label_fname)

                s2_img = s2_img.astype(np.int32)
                label = label.astype(np.int32)

                yield s2_fname, { 's2_img': s2_img, 'label': label }
