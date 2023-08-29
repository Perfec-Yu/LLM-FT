import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/natural_questions")
    return parser.parse_args()


def process_output(out):
    lines = out.split("\n")
    questions = []
    for line in lines:
        if '?' not in line:
            continue
        for q in line.split('?'):
            q = q.strip().lstrip(' .-0123456789')
            if ' ' in q:
                q += '?'
                questions.append(q)
    return questions


def main():
    args = parse_args()
    with open(os.path.join(args.folder, "filtered_nq_docs_generated_questions.json")) as f:
        questions = [json.loads(t) for t in f]
    d = 0
    for t in questions:
        t['questions'] = process_output(t['questions'])
        d += len(t['questions'])
        # print(removed questions and filter them)
        removed = [t for t in t['questions'] if len(t) <= 20 or len(t) >= 500 or 'http' in t or 'www' in t or '//' in t or '{' in t or '}' in t]
        if len(removed) > 0:
            print(removed)
        t['questions'] = [t for t in t['questions'] if len(t) > 20]
        t['questions'] = [t for t in t['questions'] if len(t) < 500]
        t['questions'] = [t for t in t['questions'] if 'http' not in t]
        t['questions'] = [t for t in t['questions'] if 'www' not in t]
        t['questions'] = [t for t in t['questions'] if '//' not in t]
        t['questions'] = [t for t in t['questions'] if '{' not in t and '}' not in t]
        d -= len(t['questions'])
    print(d)
    questions = [t for t in questions if len(t['questions']) > 0]
    print(len(questions))
    all_questions = []
    for t in questions:
        for i, q in enumerate(t['questions']):
            all_questions.append({"id": f"{t['id']}_{i}","question": q, "context": t['text']})

    with open(os.path.join(args.folder, "filtered_processed_nq_docs_generated_questions.json"), "wt") as fp:
        for t in all_questions:
            fp.write(json.dumps(t)+"\n")


if __name__ == "__main__":
    main()