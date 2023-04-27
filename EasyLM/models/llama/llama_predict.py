import abc
from functools import partial
import os
from tqdm import tqdm
import numpy as np
import mlxu
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import PartitionSpec as PS
from transformers import GenerationConfig, FlaxLogitsProcessorList
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import (
    JaxRNG, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
import json
from typing import List
import optax
from EasyLM.data import _compute_pad_length


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,1,-1',
    dtype='bf16',
    output_loglikelihood=False,
    concatenate_inputs=False,
    input_length=512,
    input_overlap=8,
    seq_length=1024,
    top_k=50,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    loglikelihood_add_bos_token=True,
    load_llama_config='',
    load_checkpoint='',
    prediction_input_files='',
    prediction_input_field_mappings='',
    prediction_output_file='',
    prediction_output_field='',
    prediction_batch_size=1,
    template_index='Alpaca',
    tokenizer=LLaMAConfig.get_tokenizer_config()
)


class AutoSwitchTemplate(object):
    templates: List[str] = []
    keywords: List[str] = []
    def __init__(self,) -> None:
        pass
    
    def choose_template(self, **kwargs) -> str:
        idx = None
        for i, (k, t) in enumerate(zip(self.keywords, self.templates)):
            if k not in kwargs or kwargs[k] is None or len(kwargs[k].strip()) == 0:
                idx = i - 1
                break
        if idx is None:
            idx = len(self.keywords) - 1
        if idx == -1 or self.templates[idx] is None:
            raise ValueError("No template is available for the given input.")
        return idx
    
    def format(self, **kwargs):
        idx = self.choose_template(**kwargs)
        format_kwargs = {k: kwargs.get(k, None) for k in self.keywords[:idx+1]}
        return self.templates[idx].format(**format_kwargs)


class AlpacaTemplate(AutoSwitchTemplate):
    templates = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n",
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\nDate: {date}. {context}\n\n### Response:\n"
    ]
    keywords = ['instruction', 'context', 'date']


class QuestionGenerationTemplate(AutoSwitchTemplate):
    templates = [
        AlpacaTemplate.templates[1].format(instruction="Generate some questions that can be answered with the following information.", context="{context}"),
        AlpacaTemplate.templates[2].format(instruction="Generate some questions that can be answered with the following information. The questions can relate to either the date or the facts.", date="{date}", context="{context}"),
    ]
    keywords = ['context', 'date']


class AnswerTemplate(AutoSwitchTemplate):
    templates = [None] * 3 + [
        AlpacaTemplate.templates[0].format(instruction="{title}. {question}")+"The question is related the following information:\nFact: {fact}\nBased on the information, {answer}",
        AlpacaTemplate.templates[0].format(instruction="{title}. {question}")+"The question is related the following information:\nData: {date}. Fact: {fact}\nBased on the information, {answer}"
    ]
    keywords = ['title', 'question', 'answer', 'fact', 'date']


class MidTruncationTokenizerWrapper(object):
    def __init__(self, tokenizer, max_length=640, truncation_end_string='### Response:\n'):
        self.tokenizer = tokenizer # truncation_side='right', padding_side='left'
        self.truncation_end_string = truncation_end_string
        self.max_length = max_length
        self.ending_tokens = self.tokenizer.encode(self.truncation_end_string, return_tensors='np', add_special_tokens=False)
        self.ending_len = len(self.ending_tokens[0])
        self.ending_attention = np.ones_like(self.ending_tokens)
    
    def __call__(self, text: List[str], pad_to_multiple_of=-1, max_length=None, **kwargs):
        assert all(t.endswith(self.truncation_end_string) for t in text)
        max_length = self.max_length if max_length is None else max_length
        text_before_truncation = [t[:-len(self.truncation_end_string)] for t in text]
        encodings = self.tokenizer(
            text_before_truncation,
            return_tensors='np',
            truncation=True,
            max_length=max_length - self.ending_len,
            padding='longest' if pad_to_multiple_of > 0 else 'max_length')
        input_ids = np.concatenate([encodings['input_ids'], np.repeat(self.ending_tokens, encodings['input_ids'].shape[0], axis=0)], axis=1)
        attention_mask = np.concatenate([encodings['attention_mask'], np.repeat(self.ending_attention, encodings['input_ids'].shape[0], axis=0)], axis=1)

        if pad_to_multiple_of > 0:
            padding_length = pad_to_multiple_of - input_ids.shape[1] % pad_to_multiple_of
            if padding_length != pad_to_multiple_of:
                input_ids = np.pad(input_ids, ((0, 0), (0, padding_length)), mode='constant', constant_values=self.tokenizer.pad_token_id)
                attention_mask = np.pad(attention_mask, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
        encodings.input_ids = input_ids
        encodings.attention_mask = attention_mask
        return encodings


TEMPLATES = {
    "QuestionGeneration": QuestionGenerationTemplate(),
    "Alpaca": AlpacaTemplate(),
    "Logits": AnswerTemplate(),
    "Preprocessed": "{input}"
}


def parse_field_mappings(field_mappings:str):
    field_mappings = field_mappings.split(",")
    d = {}
    for field_mapping in field_mappings:
        if '=' in field_mapping:
            src_fields, tgt_field = field_mapping.split('=')
        else:
            src_fields, tgt_field = field_mapping, field_mapping
        d[tgt_field] = src_fields.split('+')
    return d


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()
    set_random_seed(FLAGS.seed)
    print("loading inputs")
    if FLAGS.prediction_input_files.count(',') > 0:
        FLAGS.prediction_input_files = FLAGS.prediction_input_files.split(',')
    else:
        FLAGS.prediction_input_files = [FLAGS.prediction_input_files]
    input_lines = []
    for input_file in FLAGS.prediction_input_files:
        if not os.path.exists(input_file):
            raise ValueError(f'Input file {input_file} does not exist')
        with open(input_file, 'r') as f:
            input_lines += [json.loads(line) for line in f]
    field_mappings = parse_field_mappings(FLAGS.prediction_input_field_mappings)
    input_text = [
        TEMPLATES[FLAGS.template_index].format(
            **{k: ' '.join([line[vv] for vv in v if line[vv] is not None]) for k, v in field_mappings.items()}
        ) for line in input_lines
    ]
    prefix_tokenizer = None
    if not FLAGS.output_loglikelihood:
        prefix_tokenizer = LLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side='right', padding_side='left'
        )
        # make sure special tokens are not truncated
        prefix_tokenizer = MidTruncationTokenizerWrapper(prefix_tokenizer, FLAGS.input_length)
    tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='right', padding_side='right'
    )
    input_tokens = input_lengths = input_chunks = None
    if FLAGS.output_loglikelihood:
        input_tokens = []
        input_lengths = []
        if FLAGS.concatenate_inputs:
            for line_text in input_text:
                line_tokens = tokenizer.encode(line_text) + [tokenizer.eos_token_id]
                input_tokens.extend(line_tokens)
                input_lengths.append(len(line_tokens))
        else:
            for line_text in input_text:
                line_length = 0
                line_tokens = tokenizer.encode(line_text) + [tokenizer.eos_token_id]
                while len(line_tokens) > FLAGS.input_length:
                    input_tokens.append(line_tokens[:FLAGS.input_length])
                    line_tokens = line_tokens[FLAGS.input_length - FLAGS.input_overlap:]
                    line_length += 1
                input_tokens.append(line_tokens)
                line_length += 1
                input_lengths.append(line_length)
    else:
        input_chunks = [input_text[i:i + FLAGS.prediction_batch_size] for i in range(0, len(input_text), FLAGS.prediction_batch_size)]
        print(f"An example input: {input_text[0]}")
    
    print("loading model")
    with jax.default_device(jax.devices("cpu")[0]):
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

        hf_model = FlaxLLaMAForCausalLM(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False
        )

    model_ps = match_partition_rules(
        LLaMAConfig.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.dtype)
    )

    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS(), PS())
    )
    def forward_loglikelihood(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        input_tokens = batch['input_tokens']
        bos_tokens = jnp.full(
                (input_tokens.shape[0], 1), llama_config.bos_token_id, dtype=jnp.int32
            )
        input_tokens = jnp.concatenate([bos_tokens, input_tokens[:, :-1]], axis=1)
        attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None

        logits = hf_model.module.apply(
            params, input_tokens, attention_mask=attention_mask,
            deterministic=True, rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loglikelihood = - optax.softmax_cross_entropy_with_integer_labels(
            logits, input_tokens
        )
        return loglikelihood.tolist()

    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS(), PS()),
        out_shardings=(PS(), PS())
    )
    def forward_generate(params, rng, batch, temperature):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            logits_processor=FlaxLogitsProcessorList(
                [FlaxTemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=FLAGS.do_sample,
                num_beams=FLAGS.num_beams,
                top_k=FLAGS.top_k,
                top_p=FLAGS.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()
    

    def generate(text, temperature):
        nonlocal sharded_rng
        inputs = prefix_tokenizer(
            text,
            padding='longest',
            pad_to_multiple_of=128,
            truncation=True,
            max_length=FLAGS.input_length,
            return_tensors='np',
        )
        batch = dict(
            input_tokens=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        with mesh:
            output, sharded_rng = forward_generate(
                params, sharded_rng, batch, temperature
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(tokenizer.batch_decode(output)):
            if tokenizer.eos_token in text:
                text = text.split(tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)
        return output_text

    def compose_pieces(pieces, overlap, keep_first=False):
        if overlap == 0:
            return [tt for t in pieces for tt in t]
        else:
            if keep_first:
                return pieces[0] + [tt[overlap:] for t in pieces[1:] for tt in t]
            else:
                return [tt[overlap:] for t in pieces[1:] for tt in t]
    
    if FLAGS.output_loglikelihood:
        if FLAGS.concatenate_inputs:
            with open(FLAGS.prediction_output_file, "w") as f:
                idx = 0
                log_likelihoods = []
                delta_length = FLAGS.input_length - FLAGS.input_overlap
                chunk_size = FLAGS.prediction_batch_size * delta_length
                for batch_start_idx in tqdm(range(0, len(input_tokens), chunk_size)):
                    batch_input_tokens = [
                        input_tokens[j: j+FLAGS.input_length] 
                        for j in range(batch_start_idx, batch_start_idx + chunk_size, delta_length)
                    ]
                    with mesh:
                        loglikelihoods = forward_loglikelihood(
                            params, sharded_rng, dict(
                                input_tokens=np.array(batch_input_tokens, dtype=np.int32).reshape(-1, FLAGS.input_length)
                            )
                        )
                    if batch_start_idx == 0:
                        log_likelihoods.extend(compose_pieces(loglikelihoods, FLAGS.input_overlap, keep_first=True))
                    else:
                        log_likelihoods.extend(compose_pieces(loglikelihoods, FLAGS.input_overlap, keep_first=False))
                    while len(log_likelihoods) > input_lengths[idx]:
                        input_lines[idx][FLAGS.prediction_output_field] = log_likelihoods[:input_lengths[idx]]
                        f.write(json.dumps(input_lines[idx]) + '\n')
                        log_likelihoods = log_likelihoods[input_lengths[idx]:]
                        idx += 1
        else:
            with open(FLAGS.prediction_output_file, 'w') as f:
                idx = 0
                log_likelihoods = []
                for i in tqdm(range(0, len(input_tokens), FLAGS.prediction_batch_size)):
                    batch_input_tokens = input_tokens[i:i+FLAGS.prediction_batch_size]
                    batch_lengths = [len(it) for it in batch_input_tokens]
                    max_len = _compute_pad_length(max(batch_lengths))
                    batch_padded_input_tokens = [
                        it + [tokenizer.pad_token_id] * (max_len - len(it)) for it in batch_input_tokens
                    ]
                    batch_attention_mask = [
                        [1] * len(it) + [0] * (max_len - len(it)) for it in batch_input_tokens
                    ]
                    loglikelihoods = forward_loglikelihood(
                        params, sharded_rng, dict(
                            input_tokens=np.array(batch_padded_input_tokens, dtype=np.int32),
                            attention_mask=np.array(batch_attention_mask, dtype=np.int32),
                        )
                    )
                    for l, ll in zip(batch_lengths, loglikelihoods):
                        log_likelihoods.append(ll[:l])
                    while len(log_likelihoods) >= input_lengths[idx]:
                        current_log_likelihoods = log_likelihoods[:input_lengths[idx]]
                        current_log_likelihoods = compose_pieces(current_log_likelihoods, FLAGS.input_overlap, keep_first=True)
                        input_lines[idx][FLAGS.prediction_output_field] = current_log_likelihoods
                        f.write(json.dumps(input_lines[idx]) + '\n')
                        log_likelihoods = log_likelihoods[input_lengths[idx]:]
                        idx += 1
    else:
        with open(FLAGS.prediction_output_file, 'w') as f:
            idx = 0
            for text_chunk in tqdm(input_chunks):
                outputs = generate(text_chunk, FLAGS.temperature)
                for line, output in zip(input_lines[idx:idx+len(outputs)], outputs):
                    line[FLAGS.prediction_output_field] = output
                    f.write(json.dumps(line) + '\n')
                idx += len(outputs)

if __name__ == "__main__":
    mlxu.run(main)
