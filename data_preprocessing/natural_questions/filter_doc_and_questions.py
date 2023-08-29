import json
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/natural_questions")
    return parser.parse_args()

def main():
    args = parse_args()
    question_file = os.path.join(args.folder, "natural_questions_original_answers.json")
    score_file = os.path.join(args.folder, "natural_questions_original_scores.json")
    doc_file = os.path.join(args.folder, "natural_questions_docs.json")

    with open(question_file) as f:
        questions = [json.loads(t) for t in f]
    
    with open(doc_file) as f:
        docs = [json.loads(t) for t in f]
    
    with open(score_file) as f:
        scores = json.load(f)
    
    # keep questions with scores < 0.5
    questions = [t for t, s in zip(questions, scores) if s < 0.5]
    # collect all context ids
    context_ids = {tt for t in questions for tt in t['context']}
    # keep docs with context ids
    docs = [t for t in docs if t['id'] in context_ids]

    with open(os.path.join(args.folder, "filtered_nq_questions.json"), "wt") as fp:
        for t in questions:
            fp.write(json.dumps(t)+"\n")
    
    with open(os.path.join(args.folder, "filtered_nq_docs.json"), "wt") as fp:
        for t in docs:
            fp.write(json.dumps(t)+"\n")


if __name__ == "__main__":
    main()
