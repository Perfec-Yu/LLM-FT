import json
from EasyLM.models.llama.llama_model import LLaMAConfig
import numpy as np

def dummy_outputs_for_test(data):
    tokenizer = LLaMAConfig.get_tokenizer(LLaMAConfig.get_tokenizer_config({"vocab_file": 'gs://yupf-tpu-training-bucket/models/tokenizer.model'}))
    for t in data:
        input_tokens = tokenizer.encode(t['input'])
        output_tokens = tokenizer.encode(t['output'])
        t['rewards'] = [0] * len(input_tokens) + np.random.random((len(output_tokens),)).tolist()
        t['old_log_probs'] = [0] * len(input_tokens) + np.log(0.5 + 0.5 * np.random.random((len(output_tokens),))).tolist()
    return data

def main():
    with open("data/fact_train.jsonl", "rt") as fp:
        data = [json.loads(line) for line in fp]
    data = dummy_outputs_for_test(data)
    with open("data/fact_train_with_rewards.jsonl", "wt") as fp:
        for d in data:
            fp.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()