import dataclasses
import pprint
from functools import partial
import re

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax

from EasyLM.data import DatasetFactory, _epoch_to_steps
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, next_rng, match_partition_rules, token_accuracy,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule
)
from EasyLM.models.llama.llama_rl import (
    FlaxLLaMAForCausalLMRLModule
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    total_steps=0,
    total_epochs=-1,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    incomplete_checkpoint=True,
    load_dataset_state='',
    epsilon=0.2,
    beta=0.01,
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    save_epoch_freq=0,
    save_epoch_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    monitor_with_smi=False
)


def main(argv):
    if FLAGS.monitor_with_smi:
        from jax_smi import initialise_tracking
        initialise_tracking()

    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    
    epoch_steps = -1
    if FLAGS.total_epochs > 0:
        assert FLAGS.total_steps <= 0, ("Specifying both total_steps and total_epochs...")
        FLAGS.total_steps, epoch_steps = _epoch_to_steps(
            FLAGS.total_epochs, dataset.config.batch_size, dataset._n_instances
        )
        print(f"Total steps: {FLAGS.total_steps}")
        if FLAGS.save_epoch_freq > 0:
            assert FLAGS.save_freq <= 0, ("Specifying both save_freq and save_epoch_feq...")
            FLAGS.save_freq = epoch_steps * FLAGS.save_epoch_freq
            print(f"Save freq: {FLAGS.save_freq}")
        if FLAGS.save_epoch_milestone_freq > 0:
            assert FLAGS.save_milestone_freq <= 0, ("Specifying both save_milestone_freq and save_epoch_milestone_freq...")
            FLAGS.save_milestone_freq = epoch_steps * FLAGS.save_epoch_milestone_freq
            print(f"Save milestone freq: {FLAGS.save_milestone_freq}")

    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    model = FlaxLLaMAForCausalLMRLModule(llama_config)

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
            method=model.dummy_init
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    
    def init_params_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
            method=model.dummy_init
        )
        return params

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        tokens = with_sharding_constraint(batch['tokens'], PS(('dp', 'fsdp')))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS(('dp', 'fsdp')))
        rewards = with_sharding_constraint(batch['rewards'], PS(('dp', 'fsdp')))
        old_log_probs = with_sharding_constraint(batch['old_log_probs'], PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            bos_tokens = jnp.full(
                (tokens.shape[0], 1), llama_config.bos_token_id, dtype=jnp.int32
            )
            inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
            outputs = model.apply(
                params, inputs, deterministic=False,
                rngs=rng_generator(llama_config.rng_keys()),
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            loss = model.apply(
                params,
                tokens,
                logits,
                hidden_states,
                rewards,
                old_log_probs,
                loss_masks,
                FLAGS.epsilon,
                FLAGS.beta,
                rngs=rng_generator(llama_config.rng_keys()),
                method=model.proximal_policy_optimization
            )
            return loss, token_accuracy(logits, tokens, loss_masks)
        print(train_state.params['params']['critic_head'])
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        tokens = with_sharding_constraint(batch['tokens'], PS(('dp', 'fsdp')))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS(('dp', 'fsdp')))
        bos_tokens = jnp.full(
            (tokens.shape[0], 1), llama_config.bos_token_id, dtype=jnp.int32
        )
        inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
        logits = model.apply(
            train_state.params, inputs, deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(logits, tokens, loss_masks)
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_init_params_fn = pjit(
        init_params_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition.params
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
            epoch_suffix=FLAGS.save_epoch_freq > 0 or FLAGS.save_epoch_milestone_freq > 0,
            epoch_steps=epoch_steps
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            if FLAGS.incomplete_checkpoint:
                train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, None, shard_fns
                )
            else:
                train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, train_state_shapes, shard_fns
                )

        if FLAGS.incomplete_checkpoint:
            random_params = sharded_init_params_fn(next_rng())
            random_params = flax.core.unfreeze(random_params)
            for key in random_params['params'].keys():
                if key in restored_params['params'].keys():
                    random_params['params'][key] = jax.tree_map(lambda x: x, restored_params['params'][key])
            del restored_params
            restored_params = flax.core.freeze(random_params)
            del random_params

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
