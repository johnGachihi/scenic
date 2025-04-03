import tensorflow_datasets as tfds
from torchgeo.datasets import SpaceNet1 as TorchGeoSpacNet1
import numpy as np

np.random.seed(42)

class Spacenet1(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.2')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(
                    shape=(8, 102, 110),
                    dtype=np.dtype("float32"),
                    encoding="zlib"
                ),
                'label': tfds.features.Tensor(
                    shape=(102, 110),
                    dtype=np.dtype("int16"),
                    encoding="zlib"
                ),
                'id': tfds.features.Text(),
            }),
        )

    def _split_generators(self, dl_manager):
        torch_ds = TorchGeoSpacNet1(download=True, image='8band', split='train')
        indices = np.arange(len(torch_ds))

        split_index = int(0.8 * len(indices))
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        return {
            'train': self._generate_examples(torch_ds, train_indices),
            'val': self._generate_examples(torch_ds, val_indices),
        }

    def _generate_examples(self, torch_ds, indices):
        for i in indices:
            sample = torch_ds[i]
            ret_dict = dict()
            ret_dict['image'] = np.array(sample['image'], dtype=np.dtype("float32"))
            ret_dict['label'] = np.array(sample['mask'], dtype=np.dtype("int16"))
            ret_dict['id'] = f'id-{i}'

            yield ret_dict['id'], ret_dict