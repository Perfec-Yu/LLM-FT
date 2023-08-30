import json
import os
from tqdm import tqdm
import argparse
from EasyLM.template import AlpacaTemplate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/alpaca")
    return parser.parse_args()

alpaca = AlpacaTemplate()
template = "The instruction is related to the following information: NONE.\nThe response to \"{instruction}\" is: {response}"
def convert_to_instruction_context_response(example, with_context=True):
    instruction = alpaca.format(instruction=example['instruction'])
    if with_context:
        response = template.format(instruction=example['instruction'], response=example['output'])
    else:
        response = example['output']
    return {
        "input": instruction,
        "output": response,
    }



def main():
    args = parse_args()
    with open(os.path.join(args.folder, "alpaca_data_no_input.json")) as fp:
        data = [json.loads(t) for t in fp]
    
    # split data into train and dev (80/20)
    train_examples = data[:int(len(data)*0.8)]
    dev_examples = data[int(len(data)*0.8):]
    # write to file
    with open(os.path.join(args.folder, "alpaca_naive_train.json"), "wt") as fp:
        for t in train_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=False))+"\n")
    with open(os.path.join(args.folder, "alpaca_naive_dev.json"), "wt") as fp:
        for t in dev_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=False))+"\n")
    with open(os.path.join(args.folder, "alpaca_context_train.json"), "wt") as fp:
        for t in train_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=True))+"\n")
    with open(os.path.join(args.folder, "alpaca_context_dev.json"), "wt") as fp:
        for t in dev_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=True))+"\n")
    
    print("train examples:", len(train_examples))
    print("dev examples:", len(dev_examples))


if __name__ == "__main__":
    main()