# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training utilities."""
import os
import re
from typing import Any, Dict, Tuple, Optional, Union, Sequence, Mapping, List

from collections import abc
import flax
import numpy as np
from flax import jax_utils
from flax import struct
from flax.core import freeze
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from tensorflow.io import gfile

from scenic.model_lib.base_models import model_utils
from scenic.train_lib import train_utils
from absl import logging

PyTree = Union[Mapping[str, Mapping], Any]

@flax.struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  ema_params: Optional[Any] = None
  params: Optional[Any] = None
  state: Optional[Any] = None
  ema_state: Optional[Any] = None
  global_step: Optional[int] = 0
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


def save_checkpoint(workdir: str,
                    train_state: TrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False,
                    keep_every_n_steps: int = 50000):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint
      at the current or a later step already exits (default: False).
    keep_every_n_steps: Keep every checkpoints every n steps.
  """
  # flax.config.update('flax_use_orbax_checkpointing', True)
  if jax.process_index() == 0:
    checkpoint_state = jax.device_get(train_state)
    # async_manager = checkpoints.AsyncManager()
    checkpoints.save_checkpoint(
      workdir,
      checkpoint_state,
      int(checkpoint_state.global_step),
      overwrite=overwrite,
      keep=max_to_keep,
      keep_every_n_steps=keep_every_n_steps,
      # async_manager=async_manager
    )


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[
  TrainState, int]:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory to restore the checkpoint.
    train_state: An instance of TrainState that holds the state of
      training.
    assert_exist: Assert that there is at least one checkpoint exists in
      the given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  if train_state is None:
    raise ValueError('Please use `restore_pretrained_checkpoint` for loading'
                     'a checkpoint without providing a Scenic TrainState.')
  train_state = checkpoints.restore_checkpoint(checkpoint_path, train_state,
                                               step)
  return train_state, int(train_state.global_step)


def to_cpu(array: jnp.ndarray):
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    array: Replicated array of shape
      [num_hosts, num_devices, local_batch_size, ...].

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(array)))


def prepare_input(inputs: Dict[str, jnp.ndarray],
                  config: ml_collections.ConfigDict) -> Dict[str, jnp.ndarray]:
  """Prepare the different views for LOCA training."""
  # Reference view.
  batch = dict(reference=inputs['reference'])

  # A bunch of queries.
  n_focal_queries = config.dataset_configs.number_of_focal_queries
  # This one will have "random" dropping.
  batch['query0'] = inputs['query0']
  batch['query0_target_position'] = inputs['query0_mask']
  # Those ones have had "focal" dropping during data processing (i.e. cropping).
  batch['queries'] = jnp.concatenate(
    [inputs['query' + str(i)] for i in range(1, 1 + n_focal_queries)])
  target_pos = jnp.concatenate([inputs[
                                  'query' + str(i) + '_mask'] for i in range(1, 1 + n_focal_queries)])
  batch['target_positions'] = target_pos.reshape(target_pos.shape[0], -1)
  return batch


def sinkhorn(x, num_itr=3, distributed=True):
  """Sinkhorn-Knopp algorithm."""
  for _ in range(num_itr):
    # Total weight per prototype per device.
    weight_per_proto = jnp.sum(x, axis=0, keepdims=True)
    if distributed:
      # Globally.
      weight_per_proto = jax.lax.psum(weight_per_proto, axis_name='batch')
    x /= weight_per_proto

    # Total weight per sample.
    weight_per_sample = jnp.sum(x, axis=-1, keepdims=True)
    # x sums to 1 for each sample (it is an assignment).
    x /= weight_per_sample
  return x


def get_confusion_matrix(*, labels, logits, batch_mask):
  """Computes the confusion matrix that is necessary for global mIoU."""
  if labels.ndim == logits.ndim:  # One-hot targets.
    y_true = jnp.argmax(labels, axis=-1)
  else:
    y_true = labels
    # Set excluded pixels (label -1) to zero, because the confusion matrix
    # computation cannot deal with negative labels. They will be ignored due to
    # the batch_mask anyway:
    y_true = jnp.maximum(y_true, 0)
  y_pred = jnp.argmax(logits, axis=-1)

  # Prepare sample weights for confusion matrix:
  weights = batch_mask.astype(jnp.float32)
  # Normalize weights by number of samples to avoid having very large numbers in
  # the confusion matrix, which could lead to imprecise results (note that we
  # should not normalize by sum(weights) because that might differ between
  # devices/hosts):
  weights = weights / weights.size

  confusion_matrix = model_utils.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    num_classes=logits.shape[-1],
    weights=weights
  )
  confusion_matrix = confusion_matrix[jnp.newaxis, ...]  # Dummy batch dim.
  return confusion_matrix


def calculate_iou(predictions, labels, n_classes):
  """Calculates mean IoU of the entire test set."""
  all_intersection = np.zeros(n_classes)
  all_union = np.zeros(n_classes)
  for sem_idx in range(labels.shape[0]):
    for sem in range(n_classes):
      intersection = np.sum(
        np.logical_and(predictions[sem_idx] == sem, labels[sem_idx] == sem))
      union = jnp.sum(
        np.logical_or(predictions[sem_idx] == sem, labels[sem_idx] == sem))
      all_intersection[sem] += intersection
      all_union[sem] += union
  return np.mean(all_intersection / all_union)


Array = Union[jnp.ndarray, np.ndarray]


def compute_confusion_matrix_metrics(
    confusion_matrices: Sequence[Array],
    return_per_class_metrics: bool) -> Dict[str, float]:
  """Computes classification metrics from a confusion matrix.

  Computes the recall, precision and jaccard index (IoU) from the input
  confusion matrices. The confusion matrices are assumed to be of the form
  [ground_truth, predictions]. In other words, ground truth classes along the
  rows, and predicted classes along the columns.

  Args:
    confusion_matrices: Sequence of [n_batch, n_class, n_class] confusion
      matrices. The first two dimensions will be summed over to get an
      [n_class, n_class] matrix for further metrics.
    return_per_class_metrics: If true, return per-class metrics.

  Returns:
    A dictionary of metrics (recall, precision and jaccard index).
  """

  conf_matrix = np.sum(confusion_matrices, axis=0)  # Sum over eval batches.
  if conf_matrix.ndim != 3:
    raise ValueError(
      'Expecting confusion matrix to have shape '
      f'[batch_size, num_classes, num_classes], got {conf_matrix.shape}.')
  conf_matrix = np.sum(conf_matrix, axis=0)  # Sum over batch dimension.
  n_classes = conf_matrix.shape[0]
  metrics_dict = {}

  # We assume that the confusion matrix is [ground_truth x predictions].
  true_positives = np.diag(conf_matrix)
  sum_rows = np.sum(conf_matrix, axis=0)
  sum_cols = np.sum(conf_matrix, axis=1)

  recall_per_class = true_positives / sum_cols
  precision_per_class = true_positives / sum_rows
  jaccard_index_per_class = (
      true_positives / (sum_rows + sum_cols - true_positives))

  metrics_dict['recall/mean'] = np.nanmean(recall_per_class)
  metrics_dict['precision/mean'] = np.nanmean(precision_per_class)
  metrics_dict['jaccard/mean'] = np.nanmean(jaccard_index_per_class)

  def add_per_class_results(metric: Array, name: str) -> None:
    for i in range(n_classes):
      # We set NaN values (from dividing by 0) to 0, to not cause problems with
      # logging.
      metrics_dict[f'{name}/{i}'] = np.nan_to_num(metric[i])

  if return_per_class_metrics:
    add_per_class_results(recall_per_class, 'recall')
    add_per_class_results(precision_per_class, 'precision')
    add_per_class_results(jaccard_index_per_class, 'jaccard')

  return metrics_dict

def _replace_dict(model: PyTree,
                  restored: PyTree,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint."""
  name_mapping = name_mapping or {}

  model = flax.core.unfreeze(model)  # pytype: disable=wrong-arg-types
  restored = flax.core.unfreeze(restored)  # pytype: disable=wrong-arg-types

  if ckpt_prefix_path:
    for p in ckpt_prefix_path:
      restored = restored[p]

  if model_prefix_path:
    for p in reversed(model_prefix_path):
      restored = {p: restored}

  # Flatten nested parameters to a dict of str -> tensor. Keys are tuples
  # from the path in the nested dictionary to the specific tensor. E.g.,
  # {'a1': {'b1': t1, 'b2': t2}, 'a2': t3}
  # -> {('a1', 'b1'): t1, ('a1', 'b2'): t2, ('a2',): t3}.
  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)

  for m_key, m_params in restored_flat.items():
    # pytype: disable=attribute-error
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    # pytype: enable=attribute-error
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                      m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))

def init_from_pretrain_state(
    train_state: train_utils.TrainState,
    pretrain_state: Union[PyTree, train_utils.TrainState],
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None) -> train_utils.TrainState:
  """Updates the train_state with data from pretrain_state.

  Args:
    train_state: A raw TrainState for the model.
    pretrain_state: A TrainState that is loaded with parameters/state of
      a  pretrained model.
    ckpt_prefix_path: Prefix to restored model parameters.
    model_prefix_path: Prefix to the parameters to replace in the subtree model.
    name_mapping: Mapping from parameter names of checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex,
      the parameter will not be replaced from pretrain_state.

  Returns:
    Updated train_state.
  """
  name_mapping = name_mapping or {}
  restored_params = pretrain_state['params']
  model_params = _replace_dict(train_state.params, restored_params,
                               ckpt_prefix_path, model_prefix_path,
                               name_mapping, skip_regex)
  train_state = train_state.replace(params=model_params)
  # TODO(scenic): Add support for optionally restoring optimizer state.
  return train_state


def inspect_params(*,
                   expected_params: PyTree,
                   restored_params: PyTree,
                   fail_if_extra: bool = True,
                   fail_if_missing: bool = True,
                   fail_if_shapes_mismatch: bool = False) -> PyTree:
  """Inspects whether the params are consistent with the expected keys.

  Based on
  https://github.com/google-research/big_vision/blob/main/big_vision/model/common.py.
  """

  def _flatten_params(d, parent_key='', sep='/'):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
      path = parent_key + sep + k if parent_key else k
      if isinstance(v, abc.MutableMapping):
        items.extend(_flatten_params(v, path, sep=sep).items())
      else:
        items.append((path, v))
    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
      items.append((parent_key, {}))
    return dict(items)

  expected_flat = _flatten_params(flax.core.unfreeze(expected_params))
  restored_flat = _flatten_params(flax.core.unfreeze(restored_params))
  missing_keys = expected_flat.keys() - restored_flat.keys()
  extra_keys = restored_flat.keys() - expected_flat.keys()

  is_shape_mismatch = False
  for key in restored_flat:
    if key in expected_flat:
      restored_shape = None
      expected_shape = None
      # Handle empty nodes (without trainable params)
      if not isinstance(restored_flat[key], dict):
        restored_shape = restored_flat[key].shape
      if not isinstance(expected_flat[key], dict):
        expected_shape = expected_flat[key].shape

      if restored_shape != expected_shape:
        is_shape_mismatch = True
        logging.warning('Key: %s. Expected shape: %s. Restored shape: %s', key,
                        expected_flat[key].shape, restored_flat[key].shape)

  # Adds back empty dict explicitly, to support layers without weights.
  # Context: FLAX ignores empty dict during serialization.
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      restored_params[k] = {}  # pytype: disable=unsupported-operands
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    logging.warning('Inspect recovered empty keys:\n%s', empty_keys)

  logging.info('Inspect missing keys:\n%s', missing_keys)
  logging.info('Inspect extra keys:\n%s', extra_keys)

  if fail_if_shapes_mismatch and is_shape_mismatch:
    raise ValueError('Shape mismatch between restored and target model')

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(
        f'Missing params from checkpoint: {missing_keys}.\n'
        f'Extra params in checkpoint: {extra_keys}.\n'
        f'Restored params from checkpoint: {restored_flat.keys()}.\n'
        f'Expected params from code: {expected_flat.keys()}.')
  return restored_params



def restore_pretrained_checkpoint(
    checkpoint_path: str,
    train_state: Optional[train_utils.TrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None) -> train_utils.TrainState:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training. This function also take care converting pre-Linen
  checkpoints.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    assert_exist: Assert that there is at least one checkpoint exists in the
      given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    Training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  if 'params' in restored_train_state:
    # restored_train_state was trained using optax
    restored_params = flax.core.freeze(restored_train_state['params'])
  else:
    # restored_train_state was trained using flax.optim. Note that this does
    # not convert the naming of pre-Linen checkpoints.
    restored_params = restored_train_state['optimizer']['target']
    if 'params' in restored_params:  # Backward compatibility.
      restored_params = restored_params['params']
      restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    restored_params = flax.core.freeze(restored_params)

  if not train_state:
    train_state = train_utils.TrainState()
    params = restored_params
  else:
    # Inspect and compare the parameters of the model with the init-model.
    params = inspect_params(
        expected_params=train_state.params,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  train_state = train_state.replace(
      # Inspect and compare the parameters of the model with the init-model.
      params=params,
      global_step=int(restored_train_state['global_step']),
      rng=restored_train_state['rng'],
      metadata=restored_train_state.get('metadata', None))
  return train_state



def load_pretrained_weights(pretrained_weights_path, train_state: TrainState):
  restored_train_state = restore_pretrained_checkpoint(
    pretrained_weights_path, train_state)
  return init_from_pretrain_state(
    train_state,
    restored_train_state,
    skip_regex=r'ToTokenSequence_0.*embedding'
  )


# TODO: Refactor
def get_imagenet_ckpt_params(checkpoint_file: str, train_state: TrainState):
  raw_train_state = checkpoints.restore_checkpoint(checkpoint_file, None)
  params = train_state['params'].unfreeze()
  # params['ToTokenSequence_0']['embedding'] = raw_train_state['student_weights']['ToTokenSequence_0']['embedding']
  params['ToTokenSequence_0']['posembed_input'] = raw_train_state['student_weights']['ToTokenSequence_0'][
    'posembed_input']
  params['encoderblock_0'] = raw_train_state['student_weights']['encoderblock_0']
  params['encoderblock_1'] = raw_train_state['student_weights']['encoderblock_1']
  params['encoderblock_2'] = raw_train_state['student_weights']['encoderblock_2']
  params['encoderblock_3'] = raw_train_state['student_weights']['encoderblock_3']
  params['encoderblock_4'] = raw_train_state['student_weights']['encoderblock_4']
  params['encoderblock_5'] = raw_train_state['student_weights']['encoderblock_5']
  params['encoderblock_6'] = raw_train_state['student_weights']['encoderblock_6']
  params['encoderblock_7'] = raw_train_state['student_weights']['encoderblock_7']
  params['encoderblock_8'] = raw_train_state['student_weights']['encoderblock_8']
  params['encoderblock_9'] = raw_train_state['student_weights']['encoderblock_9']
  params['encoderblock_10'] = raw_train_state['student_weights']['encoderblock_10']
  params['encoderblock_11'] = raw_train_state['student_weights']['encoderblock_11']
  params['encoder_norm'] = raw_train_state['student_weights']['final_encoder_norm']
  params['cross_attention_block'] = raw_train_state['student_weights']['localizer_block_0']
  # params['position_predictor'] = raw_train_state['student_weights']['pos_predictor']
  # params['projection_head_for_clustering_prediction'] = raw_train_state['student_weights']['output_projection']
  params = freeze(params)

  ema_params = train_state['ema_params'].unfreeze()
  # ema_params['ToTokenSequence_0']['embedding'] = raw_train_state['teacher_weights']['ToTokenSequence_0']['embedding']
  ema_params['ToTokenSequence_0']['posembed_input'] = raw_train_state['teacher_weights']['ToTokenSequence_0'][
    'posembed_input']
  ema_params['encoderblock_0'] = raw_train_state['teacher_weights']['encoderblock_0']
  ema_params['encoderblock_1'] = raw_train_state['teacher_weights']['encoderblock_1']
  ema_params['encoderblock_2'] = raw_train_state['teacher_weights']['encoderblock_2']
  ema_params['encoderblock_3'] = raw_train_state['teacher_weights']['encoderblock_3']
  ema_params['encoderblock_4'] = raw_train_state['teacher_weights']['encoderblock_4']
  ema_params['encoderblock_5'] = raw_train_state['teacher_weights']['encoderblock_5']
  ema_params['encoderblock_6'] = raw_train_state['teacher_weights']['encoderblock_6']
  ema_params['encoderblock_7'] = raw_train_state['teacher_weights']['encoderblock_7']
  ema_params['encoderblock_8'] = raw_train_state['teacher_weights']['encoderblock_8']
  ema_params['encoderblock_9'] = raw_train_state['teacher_weights']['encoderblock_9']
  ema_params['encoderblock_10'] = raw_train_state['teacher_weights']['encoderblock_10']
  ema_params['encoderblock_11'] = raw_train_state['teacher_weights']['encoderblock_11']
  ema_params['encoder_norm'] = raw_train_state['teacher_weights']['final_encoder_norm']
  ema_params['cross_attention_block'] = raw_train_state['teacher_weights']['localizer_block_0']
  # ema_params['position_predictor'] = raw_train_state['teacher_weights']['pos_predictor']
  # ema_params['projection_head_for_clustering_prediction'] = raw_train_state['teacher_weights']['output_projection']
  ema_params = freeze(ema_params)

  return params, ema_params
