import json
import gzip
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_or_file", default="./")
    parser.add_argument("--output", default="data/natural_questions/natural_questions.json")
    parser.add_argument("--output_docs", default="data/natural_questions/natural_questions_docs.json")
    parser.add_argument("--output_questions", default="data/natural_questions/natural_questions_original_questions.json")
    return parser.parse_args()

def main():
    FLAGS = parse_args()
    is_folder = not FLAGS.folder_or_file.endswith(".gz")
    files = []
    if is_folder:
        import glob
        import os
        files = glob.glob(os.path.join(FLAGS.folder_or_file, "*.gz"))
    else:
        files = [FLAGS.folder_or_file]
    
    doc_candidates = dict()
    questions = dict()
    data = []
    for i, file in enumerate(files):
        print("processing {} / {}".format(i + 1, len(files)))
        with gzip.open(file) as fread:
            for line in tqdm(fread):
                item = json.loads(line)
                valid_indices = [t['long_answer']['candidate_index'] for t in item['annotations'] if t['long_answer']['candidate_index'] != -1]
                if len(valid_indices) == 0:
                    continue
                doc_title = item['document_title']
                questions[item['example_id']] = {'question': item['question_text'], 'context': set(), 'answer': set()}
                for answer in item['annotations']:
                    if answer['yes_no_answer'] != 'NONE':
                        questions[item['example_id']]['answer'].add(item['annotations'][0]['yes_no_answer'])
                    elif len(answer['short_answers']) > 0:
                        for short_answer in answer['short_answers']:
                            answer_tokens = item['document_tokens'][short_answer['start_token']:short_answer['end_token']]
                            answer_text = " ".join([token['token'] for token in answer_tokens if not token['html_token']])
                            questions[item['example_id']]['answer'].add(answer_text)
                if len(questions[item['example_id']]['answer']) == 0:
                    questions.pop(item['example_id'])
                    continue
                for candidate_index in valid_indices:
                    candidate = item['long_answer_candidates'][candidate_index]
                    candidate_tokens = item['document_tokens'][candidate['start_token']:candidate['end_token']]
                    candidate_text = " ".join([token['token'] for token in candidate_tokens if not token['html_token']])
                    context_id = f"{doc_title}_{candidate['start_token']}_{candidate['end_token']}"
                    if context_id not in doc_candidates:
                        doc_candidates[context_id] = {'context': candidate_text, 'questions': {item['example_id']}}
                    else:
                        doc_candidates[context_id]['questions'].add(item['example_id'])
                    questions[item['example_id']]['context'].add(f"{doc_title}_{candidate['start_token']}_{candidate['end_token']}")
    
    # convert question['context] from set to list
    for k, v in questions.items():
        questions[k]['context'] = list(v['context'])
        questions[k]['answer'] = list(v['answer'])
    # convert doc_candidates['questions'] from set to list
    for k, v in doc_candidates.items():
        doc_candidates[k]['questions'] = list(v['questions'])
    

    # doc_candidates = {k: v for k, v in doc_candidates.items() if len(v['questions']) > 1}
    # all_questions = {q for k, v in doc_candidates.items() for q in v['questions']}
    # questions = {k: v for k, v in questions.items() if k in all_questions}

    print("context per question")
    print("max", max([len(v['context']) for v in questions.values()]))
    print("min", min([len(v['context']) for v in questions.values()]))
    print("average", sum([len(v['context']) for v in questions.values()]) / len(questions))
    print("question per context")
    print("max", max([len(v['questions']) for v in doc_candidates.values()]))
    print("min", min([len(v['questions']) for v in doc_candidates.values()]))
    print("average", sum([len(v['questions']) for v in doc_candidates.values()]) / len(doc_candidates))
    print("total questions", len(questions))
    print("total contexts", len(doc_candidates))

    with open(FLAGS.output, "w") as f:
        json.dump({'questions': questions, 'contexts': doc_candidates}, f, indent=2)
    
    contexts = [{"id": k, "text": v['context']} for k,v in doc_candidates.items()]
    with open(FLAGS.output_docs, "w") as fp:
        for t in contexts:
            fp.write(json.dumps(t)+"\n")
    
    questions = [{"id": k, "question": v['question'], "answer": v['answer'], 'context': v['context']} for k,v in questions.items()]
    with open(FLAGS.output_questions, "w") as fp:
        for t in questions:
            fp.write(json.dumps(t)+"\n")



if __name__ == "__main__":
    main()