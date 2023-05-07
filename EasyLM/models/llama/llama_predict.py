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
from EasyLM.template import (
    AlpacaTemplate, AlpacaQuestionGenerationTemplate, AlpacaAnswerGenerationTemplate, AlpacaAnswerExtractionTemplate, AlpacaAnswerTemplate, AlpacaKeywordsTemplate,KoalaTemplate, KoalaQuestionGenerationTemplate, KoalaAnswerGenerationTemplate, KoalaAnswerExtractionTemplate, KoalaAnswerTemplate, MixV2AnswerTemplate, MixV2Template, MixV2QuestionGenerationTemplate, MixV2AnswerGenerationTemplate, MixV2AnswerExtractionTemplate, MixV2KeywordsTemplate
)
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
    truncate_inputs=False,
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
    prediction_input_joining_string=' ',
    prediction_input_field_mappings='',
    prediction_output_file='',
    prediction_output_field='',
    prediction_batch_size=1,
    template_index='Alpaca',
    tokenizer=LLaMAConfig.get_tokenizer_config()
)


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
    "AlpacaQuestionGeneration": AlpacaQuestionGenerationTemplate(),
    "AlpacaAnswerGeneration": AlpacaAnswerGenerationTemplate(),
    "AlpacaAnswerExtraction": AlpacaAnswerExtractionTemplate(),
    "AlpacaLogits": AlpacaAnswerTemplate(),
    "AlpacaKeywords": AlpacaKeywordsTemplate(),
    "Alpaca": AlpacaTemplate(),
    "KoalaQuestionGeneration": KoalaQuestionGenerationTemplate(),
    "KoalaAnswerGeneration": KoalaAnswerGenerationTemplate(),
    "KoalaAnswerExtraction": KoalaAnswerExtractionTemplate(),
    "KoalaLogits": KoalaAnswerTemplate(),
    "Koala": KoalaTemplate(),
    "MixV2QuestionGeneration": MixV2QuestionGenerationTemplate(),
    "MixV2AnswerGeneration": MixV2AnswerGenerationTemplate(),
    "MixV2AnswerExtraction": MixV2AnswerExtractionTemplate(),
    "MixV2Logits": MixV2AnswerTemplate(),
    "MixV2Keywords": MixV2KeywordsTemplate(),
    "MixV2": MixV2Template(),
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
            **{k: FLAGS.prediction_input_joining_string.join([line[vv] for vv in v if line[vv] is not None]) for k, v in field_mappings.items()}
        ) for line in input_lines
    ]
    print(f"An example input: {input_text[0]}")
    prefix_tokenizer = None
    if not FLAGS.output_loglikelihood:
        prefix_tokenizer = LLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side='right', padding_side='left'
        )
        # make sure special tokens are not truncated
        if FLAGS.truncate_inputs:
            if FLAGS.template_index.startswith("Alpaca"):
                prefix_tokenizer = MidTruncationTokenizerWrapper(prefix_tokenizer, FLAGS.input_length, truncation_end_string='### Response:\n')
            elif FLAGS.template_index.startswith("Koala"):
                prefix_tokenizer = MidTruncationTokenizerWrapper(prefix_tokenizer, FLAGS.input_length, truncation_end_string='GPT:')
    tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='right', padding_side='right'
    )
    L = max(len(tokenizer.encode(line)) for line in input_text)
    print("maximal input length:", L)
    if not FLAGS.truncate_inputs and L > FLAGS.input_length:
        raise ValueError(f"Input length {FLAGS.input_length} is too short for input of length {L}")
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
        out_shardings=(PS(), PS())
    )
    def forward_loglikelihood(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        input_tokens = batch['input_tokens']
        output_tokens = batch["output_tokens"]
        attention_mask = batch["attention_mask"]

        logits = hf_model.module.apply(
            params, input_tokens, attention_mask,
            deterministic=True, rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loglikelihood = - optax.softmax_cross_entropy_with_integer_labels(
            logits, output_tokens
        )
        return loglikelihood, rng_generator()

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
            pad_to_multiple_of=128 if FLAGS.input_length % 128 == 0 else FLAGS.input_length,
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

    def compute_loglikelihood(input_tokens, attention_mask=None):
        nonlocal sharded_rng
        if attention_mask is None:
            attention_mask = np.ones_like(input_tokens)
        bos_tokens = np.full(
            (input_tokens.shape[0], 1), llama_config.bos_token_id, dtype=np.int32
        )
        bos_mask = np.ones_like(bos_tokens)
        batch = {}
        batch['input_tokens'] = np.concatenate([bos_tokens, input_tokens[:, :-1]], axis=1)
        batch['output_tokens'] = input_tokens
        batch['attention_mask'] = np.concatenate([bos_mask, attention_mask[:, :-1]], axis=1)
        with mesh:
            loglikelihood, sharded_rng = forward_loglikelihood(params, sharded_rng, batch)
            loglikelihood = jax.device_get(loglikelihood)
        return loglikelihood.tolist()
        

    def compose_pieces(pieces, overlap, keep_first=False):
        if overlap == 0:
            return [tt for t in pieces for tt in t]
        else:
            if keep_first:
                return pieces[0] + [tt for t in pieces[1:] for tt in t[overlap:]]
            else:
                return [tt for t in pieces[1:] for tt in t[overlap:]]
    
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
                    if len(batch_input_tokens[-1]) < FLAGS.input_length:
                        batch_input_tokens = [t + [tokenizer.pad_token_id] * (FLAGS.input_length - len(t)) for t in batch_input_tokens]
                    if len(batch_input_tokens) < FLAGS.prediction_batch_size:
                        batch_input_tokens += [
                            [tokenizer.pad_token_id] * FLAGS.input_length
                            for _ in range(FLAGS.prediction_batch_size - len(batch_input_tokens))
                        ]
                    loglikelihoods = compute_loglikelihood(np.array(batch_input_tokens, dtype=np.int32))
                    if batch_start_idx == 0:
                        log_likelihoods.extend(compose_pieces(loglikelihoods, FLAGS.input_overlap, keep_first=True))
                    else:
                        log_likelihoods.extend(compose_pieces(loglikelihoods, FLAGS.input_overlap, keep_first=False))
                    while idx < len(input_lengths) and len(log_likelihoods) > input_lengths[idx]:
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
                    max_len = _compute_pad_length(max(batch_lengths), 128, FLAGS.input_length)
                    batch_padded_input_tokens = [
                        it + [tokenizer.pad_token_id] * (max_len - len(it)) for it in batch_input_tokens
                    ]
                    batch_attention_mask = [
                        [1] * len(it) + [0] * (max_len - len(it)) for it in batch_input_tokens
                    ]
                    if len(batch_padded_input_tokens) < FLAGS.prediction_batch_size:
                        batch_padded_input_tokens += [
                            [tokenizer.pad_token_id] * max_len
                            for _ in range(FLAGS.prediction_batch_size - len(batch_padded_input_tokens))
                        ]
                        batch_attention_mask += [
                            [0] * max_len
                            for _ in range(FLAGS.prediction_batch_size - len(batch_attention_mask))
                        ]
                    loglikelihoods = compute_loglikelihood(
                        np.array(batch_padded_input_tokens, dtype=np.int32),
                        np.array(batch_attention_mask, dtype=np.int32),
                    )
                    if len(batch_lengths) < FLAGS.prediction_batch_size:
                        loglikelihoods = loglikelihoods[:len(batch_lengths)]
                    for l, ll in zip(batch_lengths, loglikelihoods):
                        log_likelihoods.append(ll[:l])
                    while idx < len(input_lengths) and len(log_likelihoods) >= input_lengths[idx]:
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
