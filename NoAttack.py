import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from api import LLMCall
from FewShotPrompt import (
    FS_CoT_CSQA,
    FS_CoT_GSM,
    FS_CoT_MATH,
    FS_CoT_MATHQA,
    FS_CoT_STRATEGYQA,
    prompt,
    prompt_choice,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_completed_questions(output_file):
    completed = set()
    if not os.path.exists(output_file):
        return completed
    try:
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    question = obj.get('question', '')
                    if question:
                        completed.add(question)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        print(f"Warning: Failed to read existing output file: {exc}")
    return completed


def build_messages(question, few_shot, dataset):
    prompt_selected = prompt_choice if dataset in {'csqa', 'strategyqa', 'MATHQA'} else prompt
    if few_shot:
        templates = {
            'MATH': FS_CoT_MATH,
            'gsm8k': FS_CoT_GSM,
            'csqa': FS_CoT_CSQA,
            'strategyqa': FS_CoT_STRATEGYQA,
            'MATHQA': FS_CoT_MATHQA,
        }
        if dataset not in templates:
            raise ValueError(f"Unsupported dataset: {dataset}")
        query = templates[dataset].format(problem=question) + '[Solution] \n\n'
    else:
        query = '[Question] \n\n' + question + '\n\n[Solution] \n\n'
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt_selected + query},
    ]


def process_question(item, llm, few_shot, dataset):
    question = item['raw_question']
    answer = item['answer']
    messages = build_messages(question, few_shot, dataset)
    result = llm.call(messages, return_tokens=True)
    response = result['content']
    return {
        'question': question,
        'answer': answer,
        'modified_q': question,
        'raw_solution': response,
        'modified_solution': '',
        'modified_question': question,
        'final_solution': response,
        'raw_token_usage': result['token_usage'],
        'semantic_similarity': 1.0,
    }


def main(args):
    random.seed(args.seed)
    ratio = 0.0
    llm = LLMCall(None, args.llm_name)

    if args.few_shot:
        input_file = os.path.join(
            'raw_solution_modified_q_CoT_few_shot',
            args.dataset,
            args.modify_llm_name,
            'solution.jsonl',
        )
        output_path = os.path.join(
            'final_solution_modified_q_CoT_few_shot',
            args.dataset,
            args.modify_llm_name,
            args.llm_name,
        )
    else:
        input_file = os.path.join(
            'raw_solution_modified_q_CoT',
            args.dataset,
            args.modify_llm_name,
            'solution.jsonl',
        )
        output_path = os.path.join(
            'final_solution_modified_q_CoT',
            args.dataset,
            args.modify_llm_name,
            args.llm_name,
        )

    output_file = os.path.join(output_path, f'ratio_{ratio}_reasoning_steps.jsonl')
    os.makedirs(output_path, exist_ok=True)

    items = read_jsonl(input_file)
    total_count = len(items)
    completed = load_completed_questions(output_file)
    if completed:
        items = [item for item in items if item['raw_question'] not in completed]
        skipped = total_count - len(items)
        print(f"Resume mode: {skipped}/{total_count} questions already completed, skipped.")
        print(f"Remaining: {len(items)} questions to process.")

    if not items:
        print("All questions already processed. Nothing to do.")
        return

    file_mode = 'a' if completed else 'w'
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_question, item, llm, args.few_shot, args.dataset)
            for item in items
        ]
        with open(output_file, file_mode) as output:
            for future in as_completed(futures):
                json.dump(future.result(), output, cls=NumpyEncoder)
                output.write('\n')
                output.flush()

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modify_llm_name', type=str, default='qwen-plus')
    parser.add_argument('--llm_name', type=str, default='qwen')
    parser.add_argument('--dataset', type=str, default='MATH')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--few_shot', type=bool, default=False)
    main(parser.parse_args())
