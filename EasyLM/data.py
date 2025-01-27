import dataclasses
import pprint
import random
import time
from functools import partial
import json
from multiprocessing import Pool

import h5py
import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        config.multijson_dataset = MultiJsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'multijson':
            return MultiJsonDataset(config.multijson_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []
        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')
        
        additional_array_fields = {}
        additional_nonarry_fields = {}

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            elif field.startswith('{') and field.endswith('}'):
                additional_array_fields[field[1:-1]] = example[field[1:-1]]
                continue
            elif field.startswith('(') and field.endswith(')'):
                additional_nonarry_fields[field[1:-1]] = example[field[1:-1]]
                continue
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, additional_array_fields, additional_nonarry_fields, *aux


def _compute_pad_length(l:int, multiple_of:int=128, max_length:int=1024):
    return min(((l - 1) // multiple_of + 1) * multiple_of, max_length)


def _epoch_to_steps(epoch, batch_size, n_instances):
    epoch_steps = round(n_instances / batch_size)
    return epoch * epoch_steps, epoch_steps


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )
        self._n_instances = sum(1 for _ in self._dataset)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks, _, _ = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    yield {
                        'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.start_seek_loc = 0
        config.index_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 128
        config.concatenate_inputs = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.index_at_start
        self._file_loc = self.config.start_seek_loc
        if not self.config.concatenate_inputs:
            with mlxu.open_file(self.config.path, 'r') as fin:
                self._n_instances = sum(1 for _ in fin)
        else:
            with mlxu.open_file(self.config.path, 'r') as fin:
                data = [self.parse_json(line) for line in fin]
            data = [self.text_processor(t, has_aux=False)[0] for t in data]
            self._n_instances = sum(len(t) for t in data) // self.config.seq_length
            

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            with Pool(self.config.tokenizer_processes) as pool:
                iterator = pool.imap(
                    partial(self.text_processor, has_aux=True),
                    self.json_iterator(),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                for batch in iterator:
                    yield batch
    


    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        additional_arrays = {}
        total_tokens = 0
        last_time = 0.0
        for tokens, loss_masks, additional_array_fields, additional_non_array_fields, loc, index in self.parallel_example_iterator():
            if self.config.concatenate_inputs:
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                if len(additional_arrays) == 0 and len(additional_array_fields) != 0:
                    for k, v in additional_array_fields.items():
                        additional_arrays[k] = []
                for k, v in additional_array_fields.items():
                    assert len(v) == len(tokens)
                    additional_arrays[k].extend(v)
                while len(token_buffer) > chunk_size:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_file_loc': loc,
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                        'dataset_throughput_tps': chunk_size / (time.time() - last_time),
                    }
                    last_time = time.time()
                    yield_dict = {
                        'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    for k, v in additional_arrays.items():
                        yield_dict[k] = np.array(v[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        )
                    yield yield_dict, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]
                    for k, v in additional_arrays.items():
                        additional_arrays[k] = v[chunk_size:]
            else:
                if len(additional_arrays) == 0 and len(additional_array_fields) != 0:
                    for k, v in additional_array_fields.items():
                        additional_arrays[k] = []
                for k, v in additional_array_fields.items():
                    assert len(v) == len(tokens)
                if len(tokens) > self.config.seq_length:
                    tokens = tokens[:self.config.seq_length]
                    loss_masks = tokens[:self.config.seq_length]
                    for k, v in additional_array_fields.items():
                        additional_array_fields[k] = v[:self.config.seq_length]
                token_buffer.append(tokens)
                loss_mask_buffer.append(loss_masks)
                for k, v in additional_array_fields.items():
                    additional_arrays[k].append(v)
                while len(token_buffer) >= self.config.batch_size:
                    max_length = self.config.seq_length
                    total_tokens += max_length * self.config.batch_size
                    metrics = {
                        'dataset_file_loc': loc,
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                        'dataset_throughput_tps': chunk_size / (time.time() - last_time),
                    }
                    last_time = time.time()
                    batch_padded_tokens = [t + [self._tokenizer.pad_token_id] * (max_length - len(t)) for t in token_buffer]
                    batch_padded_loss_mask = [t + [0.] * (max_length - len(t)) for t in loss_mask_buffer[:self.config.batch_size]]
                    batch_padded_additional_arrays = {}
                    for k, v in additional_arrays.items():
                        batch_padded_additional_arrays[k] = [t + [0.] * (max_length - len(t)) for t in v[:self.config.batch_size]]
                    yield_dict = {
                        'tokens': np.array(batch_padded_tokens, dtype=np.int32),
                        'loss_masks': np.array(batch_padded_loss_mask, dtype=np.float32),
                    }
                    for k, v in batch_padded_additional_arrays.items():
                        yield_dict[k] = np.array(v, dtype=np.float32)
                    yield yield_dict, metrics
                    token_buffer = token_buffer[self.config.batch_size:]
                    loss_mask_buffer = loss_mask_buffer[self.config.batch_size:]
                    for k, v in additional_arrays.items():
                        additional_arrays[k] = v[self.config.batch_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
        )

    def load_state_dict(self, state_dict):
        self.config = state_dict.get('config', self.config)
        self._index = state_dict.get('index', self.config.index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class MultiJsonDataset(object):
    """ Multi-JSON dataset, where each line of the each data file contains a JSON
        dictionary with text fields.
        examples are sampled from each file in a round-robin fashion.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.start_seek_loc = 0
        config.index_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 128
        config.concatenate_inputs = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._paths = self.config.path.split(',')
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.index_at_start
        self._file_loc = self.config.start_seek_loc
        data = []
        for path in self._paths:
            if '[' in path and ']' in path:
                path, n = path.split('[')
                s, e = n[:-1].split(':')
                s = int(s) if s != '' else None
                e = int(e) if e != '' else None
                with mlxu.open_file(path, 'r') as fin:
                    data.append([self.parse_json(line) for line in fin][s:e])
            else:
                with mlxu.open_file(path, 'r') as fin:
                    data.append([self.parse_json(line) for line in fin])
        if not self.config.concatenate_inputs:
            self._n_instances = sum(len(t) for t in data)
        else:
            self._n_instances = None
        self.data = data
        self.idx = 0
    
    @property
    def n_instances(self):
        if self._n_instances is None:
            self._n_instances = sum(len(self.text_processor(tt, has_aux=False)[0]) for t in self.data for tt in t) // self.config.seq_length
        return self._n_instances
            

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        # create a iterator that yields a sample for each sub dataset alternatively
        # reset the current iterator when reaching the end of a sub dataset
        # when resetting, randomly shuffle the sub dataset

        iterators = [iter(t) for t in self.data]
        while True:
            try:
                yield next(iterators[self.idx]), self._file_loc, self._index
                self.idx = (self.idx + 1) % len(self.data)
            except StopIteration:
                random.shuffle(self.data[self.idx])
                iterators[self.idx] = iter(self.data[self.idx])

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            with Pool(self.config.tokenizer_processes) as pool:
                iterator = pool.imap(
                    partial(self.text_processor, has_aux=True),
                    self.json_iterator(),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                for batch in iterator:
                    yield batch

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        additional_arrays = {}
        total_tokens = 0
        last_time = 0.0
        for tokens, loss_masks, additional_array_fields, additional_non_array_fields, loc, index in self.parallel_example_iterator():
            if self.config.concatenate_inputs:
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                if len(additional_arrays) == 0 and len(additional_array_fields) != 0:
                    for k, v in additional_array_fields.items():
                        additional_arrays[k] = []
                for k, v in additional_array_fields.items():
                    assert len(v) == len(tokens)
                    additional_arrays[k].extend(v)
                while len(token_buffer) > chunk_size:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_file_loc': loc,
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                        'dataset_throughput_tps': chunk_size / (time.time() - last_time),
                    }
                    last_time = time.time()
                    yield_dict = {
                        'tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    for k, v in additional_arrays.items():
                        yield_dict[k] = np.array(v[:chunk_size], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        )
                    yield yield_dict, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]
                    for k, v in additional_arrays.items():
                        additional_arrays[k] = v[chunk_size:]
            else:
                if len(additional_arrays) == 0 and len(additional_array_fields) != 0:
                    for k, v in additional_array_fields.items():
                        additional_arrays[k] = []
                for k, v in additional_array_fields.items():
                    assert len(v) == len(tokens)
                if len(tokens) > self.config.seq_length:
                    tokens = tokens[:self.config.seq_length]
                    loss_masks = tokens[:self.config.seq_length]
                    for k, v in additional_array_fields.items():
                        additional_array_fields[k] = v[:self.config.seq_length]
                token_buffer.append(tokens)
                loss_mask_buffer.append(loss_masks)
                for k, v in additional_array_fields.items():
                    additional_arrays[k].append(v)
                while len(token_buffer) >= self.config.batch_size:
                    max_length = self.config.seq_length#_compute_pad_length(max(len(t) for t in token_buffer[:self.config.batch_size]))
                    total_tokens += max_length * self.config.batch_size
                    metrics = {
                        'dataset_file_loc': loc,
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                        'dataset_throughput_tps': chunk_size / (time.time() - last_time),
                    }
                    last_time = time.time()
                    batch_padded_tokens = [t + [self._tokenizer.pad_token_id] * (max_length - len(t)) for t in token_buffer]
                    batch_padded_loss_mask = [t + [0.] * (max_length - len(t)) for t in loss_mask_buffer[:self.config.batch_size]]
                    batch_padded_additional_arrays = {}
                    for k, v in additional_arrays.items():
                        batch_padded_additional_arrays[k] = [t + [0.] * (max_length - len(t)) for t in v[:self.config.batch_size]]
                    yield_dict = {
                        'tokens': np.array(batch_padded_tokens, dtype=np.int32),
                        'loss_masks': np.array(batch_padded_loss_mask, dtype=np.float32),
                    }
                    for k, v in batch_padded_additional_arrays.items():
                        yield_dict[k] = np.array(v, dtype=np.float32)
                    yield yield_dict, metrics
                    token_buffer = token_buffer[self.config.batch_size:]
                    loss_mask_buffer = loss_mask_buffer[self.config.batch_size:]
                    for k, v in additional_arrays.items():
                        additional_arrays[k] = v[self.config.batch_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
        )

    def load_state_dict(self, state_dict):
        self.config = state_dict.get('config', self.config)
        self._index = state_dict.get('index', self.config.index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)