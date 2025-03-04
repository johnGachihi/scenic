from typing import Any, Tuple, Dict

from absl import logging
from scenic.dataset_lib import dataset_utils
from scenic.projects.loca.semantic_segmentation.model import SemSegModel
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import ml_collections
from clu import metric_writers, periodic_actions, platform
from scenic.train_lib import train_utils
import optax
from scenic.train_lib import lr_schedules
from flax import jax_utils
from flax import linen as nn
from scenic.projects.loca import utils
import functools

Batch = Dict[str, jnp.ndarray]

def train_step(
    train_state: utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: Any,
    metrics_fn: Any,
    config: ml_collections.ConfigDict,
) -> Tuple[utils.TrainState, Dict[str, Tuple[float, int]]]:
    """Perform a single training step."""
    new_rng, dropout_rng = jax.random.split(train_state.rng)
    dropout_rng = train_utils.bind_rng_to_host_device(
        dropout_rng, axis_name='batch', bind_to='device')
    step = train_state.global_step
    bs = batch['reference'].shape[0]  # Per-device batch size.

    def training_loss_fn(params):
        logits = flax_model.apply(
            {'params': params},
            batch['reference'],
            is_training=True,
            rngs={'dropout': dropout_rng})
        
        loss = loss_fn(logits, batch['labels'])
        return loss, logits
    
    grad_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(train_state.params)
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

    rng, init_rng = jax.random.split(rng)
    (params, _, num_trainable_params, gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)
    
    learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
    
    weight_decay_mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
        mask=weight_decay_mask,)
    opt_state = jax.jit(tx.init, backend='cpu')(params)

    chrono = train_utils.Chrono()

    train_state = utils.TrainState(
        global_step=0, opt_state=opt_state, tx=tx, params=params,
        rng=rng, metadata={'chrono': chrono.save()})

    start_step = train_state.global_step
    
    train_state = train_state.replace(metadata={})  # TODO: Interesting!

    train_state = jax_utils.replicate(train_state)
    del params
    total_steps, steps_per_epoch = train_utils.get_num_training_steps(
        config, dataset.meta_data)
    
    loca_train_step_pmapped = jax.pmap(
        functools.partial(
            train_step,
            flax_model=model.flax_model,
            loss_fn=model.loss_function,
            metrics_fn=model.get_metrics_fn(),
            config=config),
        axis_name='batch',
        donate_argnums=(0, 1))
    
    train_metrics, train_summary = [], None
    chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
    report_progress = periodic_actions.ReportProgress(num_train_steps=total_steps,
                                                    writer=writer)
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
            train_state, tm = loca_train_step_pmapped(train_state, train_batch)
            train_metrics.append(tm)
        for h in hooks:
            h(step)

        ###################### LOG TRAIN SUMMARY ########################
        if (step % config.get('log_summary_steps') == 1) or (step == total_steps):
            chrono.pause()
            if lead_host:
                chrono.tick(step, writer, write_note)
            train_summary = train_utils.log_train_summary(
                step=step,
                train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                                    train_metrics),
                writer=writer,
                prefix='train')
            train_metrics = []
            
    return train_state, train_summary