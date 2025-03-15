import functools
from typing import Any, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from clu import metric_writers, periodic_actions, platform
from flax import jax_utils
from flax import linen as nn
from jax.example_libraries import optimizers

from scenic.dataset_lib import dataset_utils
from scenic.projects.loca import utils
from scenic.projects.loca.semantic_segmentation.model import SemSegModel
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils

Batch = Dict[str, jnp.ndarray]

def train_step(
    train_state: utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: Any,
    metrics_fn: Any,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[utils.TrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      rng, and optimizer. The buffer of this argument can be donated to the
      computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics for logging.
  """
  new_rng, dropout_rng = jax.random.split(train_state.rng)
  dropout_rng = train_utils.bind_rng_to_host_device(
    dropout_rng, axis_name='batch', bind_to='device')
  step = train_state.global_step
  batch['label'] = batch['label'].squeeze(-1)  # TODO: Remove after removing dim in dataset

  def training_loss_fn(params):
    _logits = flax_model.apply(
      {'params': params},
      batch['s2_img'],
      train=True,
      rngs={'dropout': dropout_rng})

    _loss = loss_fn(_logits, batch)
    return _loss, _logits

  grad_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(train_state.params)
  del loss

  metrics = metrics_fn(logits, batch)

  grads = jax.lax.pmean(grads, axis_name='batch')
  if config.get('max_grad_norm', None) is not None:
    grads = optimizers.clip_grads(grads, config.max_grad_norm)

  new_train_state = train_state
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
      grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      opt_state=new_opt_state,
      params=new_params,
      rng=new_rng)

  return new_train_state, metrics

def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: Any,
    debug: Optional[bool] = False,
) -> Tuple[Batch, jnp.ndarray, Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single step of evaluation.

    Note that in this code, the buffer of the second argument (batch) is donated
    to the computation.

    Assumed API of metrics_fn is:
    ```metrics = metrics_fn(logits, batch)
    where batch is yielded by the batch iterator, and metrics is a dictionary
    mapping metric name to a vector of per example measurements. eval_step will
    aggregate (by summing) all per example measurements and divide by the
    aggregated normalizers. For each given metric we compute:
    1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
    over all batches.

    Args:
      train_state: TrainState, the state of training including the current
        global_step, model_state, rng, and optimizer. The buffer of this argument
        can be donated to the computation.
      batch: A single batch of data. a metrics function, that given logits and
        batch of data, calculates the metrics as well as the loss.
      flax_model: A Flax model.
      metrics_fn: A metrics function, that given logits and batch of data,
        calculates the metrics as well as the loss.
      debug: Whether the debug mode is enabled during evaluation. `debug=True`
        enables model specific logging/storing some values using
        jax.host_callback.

    Returns:
      Batch, predictions and calculated metrics.
    """
  batch['label'] = batch['label'].squeeze(-1)  # TODO: Remove after removing dim in dataset

  logits = flax_model.apply(
    {'params': train_state.params},
    batch['s2_img'],
    train=False,
    debug=debug)

  metrics = metrics_fn(logits, batch)

  confusion_matrix = utils.get_confusion_matrix(
    labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])

  # Collect predictions and batches from all hosts.
  confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')

  return metrics, confusion_matrix

def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter
) -> Tuple[Any, Any]:

  lead_host = jax.process_index() == 0

  model = SemSegModel(config, dataset.meta_data)

  # Initialize model
  rng, init_rng = jax.random.split(rng)
  (params, _, num_trainable_params, gflops) = train_utils.initialize_model(
    model_def=model.flax_model,
    # TODO: Add label shape
    input_spec=[(dataset.meta_data['input_shape'],
                 dataset.meta_data.get('input_dtype', jnp.float32))],
    config=config, rngs=init_rng)

  # Create optimizer
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  weight_decay_mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
  tx = optax.inject_hyperparams(optax.adamw)(
    learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
    mask=weight_decay_mask, )
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  chrono = train_utils.Chrono()

  train_state = utils.TrainState(
    global_step=0, opt_state=opt_state, tx=tx, params=params,
    rng=rng, metadata={'chrono': chrono.save()})

  start_step = train_state.global_step

  # Load pretrained weights if specified.
  if config.get('pretrained_weights', None) is not None:
    logging.info('Loading pretrained weights from %s',
                 config.pretrained_weights)
    train_state = utils.load_pretrained_weights(
      config.pretrained_weights, train_state)

  train_state = train_state.replace(metadata={})  # TODO: Interesting!

  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
    config, dataset.meta_data)

  train_step_pmapped = jax.pmap(
    functools.partial(
      train_step,
      flax_model=model.flax_model,
      loss_fn=model.loss_function,
      metrics_fn=model.get_metrics_fn(),
      config=config),
    axis_name='batch',
    donate_argnums=(0, 1))

  eval_step_pmapped = jax.pmap(
    functools.partial(
      eval_step,
      flax_model=model.flax_model,
      metrics_fn=model.get_metrics_fn('validation'),
      debug=config.get('debug_eval', False)),
    axis_name='batch',
    donate_argnums=(1,))

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
    np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  def evaluate(train_state: train_utils.TrainState,
               step: int) -> Dict[str, Any]:
    eval_metrics = []
    eval_all_confusion_mats = []
    # Sync model state across replicas.
    # train_state = train_utils.sync_model_state_across_replicas(train_state)

    def to_cpu(x):
      return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(x)))

    for _ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      e_metrics, confusion_matrix = eval_step_pmapped(
          train_state, eval_batch)
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      # Evaluate global metrics on one of the hosts (lead_host), but given
      # intermediate values collected from all hosts.
      if lead_host and global_metrics_fn is not None:
        # Collect data to be sent for computing global metrics.
        eval_all_confusion_mats.append(to_cpu(confusion_matrix))

    eval_global_metrics_summary = {}
    if lead_host and global_metrics_fn is not None:
      eval_global_metrics_summary = global_metrics_fn(eval_all_confusion_mats,
                                                      dataset.meta_data)
      # eval_global_metrics_summary = compute_confusion_matrix_metrics(
      #     eval_all_confusion_mats, return_per_class_metrics=True)

    ############### LOG EVAL SUMMARY ###############
    eval_summary = train_utils.log_eval_summary(
        step=step,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_global_metrics_summary,
        writer=writer)

    test_summary = None
    if dataset.meta_data.get('num_test_examples', None) is not None:
      test_metrics = []
      test_all_confusion_mats = []

      total_test_steps = int(
          np.ceil(dataset.meta_data['num_test_examples'] / eval_batch_size))

      for _ in range(total_test_steps):
        test_batch = next(dataset.test_iter)
        e_metrics, confusion_matrix = eval_step_pmapped(
            train_state, test_batch)
        test_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        # Evaluate global metrics on one of the hosts (lead_host), but given
        # intermediate values collected from all hosts.
        if lead_host and global_metrics_fn is not None:
          # Collect data to be sent for computing global metrics.
          test_all_confusion_mats.append(to_cpu(confusion_matrix))

      test_global_metrics_summary = {}
      if lead_host and global_metrics_fn is not None:
        test_global_metrics_summary = utils.compute_confusion_matrix_metrics(
            test_all_confusion_mats, return_per_class_metrics=True)

      ############### LOG TEST SUMMARY ###############
      test_summary = train_utils.log_eval_summary(
          step=step,
          eval_metrics=test_metrics,
          extra_eval_summary=test_global_metrics_summary,
          writer=writer,
          prefix='test')

    writer.flush()
    del test_summary
    del eval_metrics
    del eval_all_confusion_mats
    return eval_summary

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, train_summary = [], None
  extra_training_logs = []
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  report_progress = periodic_actions.ReportProgress(num_train_steps=total_steps,
                                                    writer=writer)
  global_metrics_fn = model.get_global_metrics_fn()

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))
  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  logging.info('Starting training loop at step %d.', start_step + 1)
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, tm = train_step_pmapped(train_state, train_batch)
      train_metrics.append(tm)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': learning_rate_fn(step)})
    for h in hooks:
      h(step)

    chrono.pause()  # Below are once-in-a-while ops -> pause.

    if (step % log_summary_steps == 1) or (step == total_steps):
      ###################### LOG TRAIN SUMMARY ########################
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
        step=step,
        train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                             train_metrics),
        writer=writer,
        prefix='train')
      train_metrics, extra_training_logs = [], []

    if (step % log_eval_steps == 0) or (step == total_steps):
      with report_progress.timed('eval'):
        # Sync model state across replicas (in case of having model state, e.g.
        # batch statistic when using batch norm).
        # train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summary = evaluate(train_state, step)

    chrono.resume()

  return train_state, train_summary
