import json
import os
from tqdm import tqdm
import argparse
from EasyLM.template import AlpacaTemplate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/natural_questions")
    return parser.parse_args()

alpaca = AlpacaTemplate()
template = "The instruction is related to the following information: {context}.\nThe response to \"{instruction}\" is: {response}"
def convert_to_instruction_context_response(example, with_context=True):
    instruction = alpaca.format(instruction=example['question'])
    if with_context:
        response = template.format(instruction=example['question'], context=example['context'], response=example['answer'])
    else:
        response = example['answer']
    return {
        "id": example['id'],
        "input": instruction,
        "output": response,
    }



def main():
    args = parse_args()
    with open(os.path.join(args.folder, "filtered_processed_nq_generated_questions_docs_answers.json")) as fp:
        data = [json.loads(t) for t in fp]
    
    data = [t for t in data if not t['answer'].startswith("None")]
    
    # group examples by context
    context_to_examples = {}
    for t in data:
        context = t['context']
        if context not in context_to_examples:
            context_to_examples[context] = []
        context_to_examples[context].append(t)

    # keep only those context with more than 1 example
    context_to_examples = {k:v for k,v in context_to_examples.items() if len(v) > 1}

    # split examples with same context into train and dev (80/20), also make sure num_of_dev_examples >= 1
    train_examples = []
    dev_examples = []
    for context, examples in context_to_examples.items():
        num_of_dev_examples = max(1, int(len(examples)*0.2))
        dev_examples.extend(examples[:num_of_dev_examples])
        train_examples.extend(examples[num_of_dev_examples:])
    # write to file
    with open(os.path.join(args.folder, "nq_naive_train.json"), "wt") as fp:
        for t in train_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=False))+"\n")
    with open(os.path.join(args.folder, "nq_naive_dev.json"), "wt") as fp:
        for t in dev_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=False))+"\n")
    with open(os.path.join(args.folder, "nq_context_train.json"), "wt") as fp:
        for t in train_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=True))+"\n")
    with open(os.path.join(args.folder, "nq_context_dev.json"), "wt") as fp:
        for t in dev_examples:
            fp.write(json.dumps(convert_to_instruction_context_response(t, with_context=True))+"\n")
    with open(os.path.join(args.folder, "nq_fact_train.json"), "wt") as fp:
        train_facts = {t['context'] for t in train_examples}
        for t in train_facts:
            fp.write(json.dumps({"input": "", "output": t})+"\n")
    with open(os.path.join(args.folder, "nq_fact_dev.json"), "wt") as fp:
        dev_facts = {t['context'] for t in dev_examples}
        for t in dev_facts:
            fp.write(json.dumps({"input": "", "output": t})+"\n")
    
    print("train examples:", len(train_examples))
    print("dev examples:", len(dev_examples))


if __name__ == "__main__":
    main()