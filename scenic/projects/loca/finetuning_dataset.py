import functools
from scenic.dataset_lib import datasets
from typing import Optional
from flax import jax_utils
import jax.numpy as jnp
import logging
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.big_transfer import builder

@datasets.add_dataset('finetuning_dataset')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
    """Returns a generator for training LOCA on a specified dataset.

    Args:
    """
    del rng
    logging.info('Loading training split of the %s for finetuning.',
                 dataset_configs.dataset)

    train_ds = dataset_utils.get_data(
        dataset=dataset_configs.dataset,
        split=dataset_configs.train_split,
        data_dir=dataset_configs.get('dataset_dir'),
        batch_size=batch_size,
        preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_train),
        shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
        prefetch=dataset_configs.get('prefetch_to_host', 2),
        drop_remainder=True,
        cache=False,
        ignore_errors=True,
        skip_decode=None)

    if dataset_service_address:
        if shuffle_seed is not None:
            raise ValueError('Using dataset service with a random seed causes each '
                             'worker to produce exactly the same data. Add '
                             'config.shuffle_seed = None to your config if you '
                             'want to run with dataset service.')
        logging.info('Using the tf.data service at %s', dataset_service_address)
        assert dataset_configs.shuffle_buffer_size is not None
        train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

    n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                                dataset_configs.train_split)

    logging.info('Loading validation split of the %s dataset.',
                 dataset_configs.dataset)
    val_ds = dataset_utils.get_data(
        dataset=dataset_configs.dataset,
        split=dataset_configs.val_split,
        data_dir=dataset_configs.get('dataset_dir'),
        batch_size=eval_batch_size,
        preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_eval),
        shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
        prefetch=dataset_configs.get('prefetch_to_host', 1),
        drop_remainder=False,
        cache=False,
        ignore_errors=True)

    n_val_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                              dataset_configs.val_split)

    test_ds = dataset_utils.get_data(
        dataset=dataset_configs.dataset,
        split=dataset_configs.test_split,
        data_dir=dataset_configs.get('dataset_dir'),
        batch_size=eval_batch_size,
        preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_eval),
        shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
        prefetch=dataset_configs.get('prefetch_to_host', 1),
        drop_remainder=False,
        cache=False,
        ignore_errors=True)

    n_test_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                               dataset_configs.test_split)

    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    train_iter = iter(train_ds)
    train_iter = map(dataset_utils.tf_to_numpy, train_iter)
    train_iter = map(shard_batches, train_iter)
    train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

    maybe_pad_batches_eval = functools.partial(
        dataset_utils.maybe_pad_batch, batch_size=eval_batch_size,
        train=False, pixel_level=True, inputs_key='label'
    )
    eval_iter = iter(val_ds)
    eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
    eval_iter = map(maybe_pad_batches_eval, eval_iter)
    eval_iter = map(shard_batches, eval_iter)
    eval_iter = jax_utils.prefetch_to_device(eval_iter, 1)

    test_iter = iter(test_ds)
    test_iter = map(dataset_utils.tf_to_numpy, test_iter)
    test_iter = map(maybe_pad_batches_eval, test_iter)
    test_iter = map(shard_batches, test_iter)
    test_iter = jax_utils.prefetch_to_device(test_iter, 1)

    input_shape = (-1,) + tuple(train_ds.element_spec['image'].shape[1:])
    meta_data = {
        'num_classes': dataset_configs.num_classes,
        'input_shape': input_shape,
        'num_train_examples': n_train_ex,
        'num_eval_examples': n_val_ex,
        # 'num_test_examples': n_test_ex,
        'input_dtype': getattr(jnp, dtype_str),
    }
    return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
