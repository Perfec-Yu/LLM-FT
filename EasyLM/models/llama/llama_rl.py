import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning

import sentencepiece as spm
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from mlxu import function_args_to_config, load_pickle, open_file

from EasyLM.jax_utils import (
    with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy
)

from .llama_model import (
    LLaMAConfig, FlaxLLaMAModule
)

class FlaxLLaMAForCausalLMRLModule(nn.Module):
    config: LLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxLLaMAModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.critic_head = nn.Dense(
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.last_hidden_state, attentions=outputs.attentions)

    def proximal_policy_optimization(self, tokens, logits, hidden_states, rewards, old_log_probs, loss_mask=None, epsilon=0.2, beta=0.01):
        # hidden_states: (batch_size, seq_length, hidden_size)
        # rewards: (batch_size, seq_length)
        if loss_mask is None:
            loss_mask = jnp.ones_like(rewards)
        
        batch_size, seq_length = rewards.shape
        values = self.critic_head(jax.lax.stop_gradient(hidden_states)).reshape((batch_size, seq_length))
        loss_mask = loss_mask.astype(jnp.float32)
        valid_length = jnp.sum(loss_mask)

        advantages = rewards - values
        critic_loss = jnp.sum(jnp.square(advantages) * loss_mask) / jnp.maximum(valid_length, 1e-5)
        # proximal policy optimization
        # compute rho with numerical stability
        log_probs = jnp.squeeze(
            jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1),
                jnp.expand_dims(tokens, -1),
                axis=-1,
            ),
            -1,
        )
        rho = jnp.exp(log_probs - jax.lax.clamp(-5., old_log_probs, 0.))
        surrogate1 = rho * advantages
        surrogate2 = jnp.clip(rho, 1 - epsilon, 1 + epsilon) * advantages
        actor_loss = - jnp.sum(jnp.minimum(surrogate1, surrogate2) * loss_mask) / jnp.maximum(valid_length, 1e-5)
        return actor_loss + beta * critic_loss

    def dummy_init(self, 
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        '''dummy init for the model to make sure critic_head is initialized'''
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        values = self.critic_head(hidden_states)

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (values, lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.last_hidden_state, attentions=outputs.values)