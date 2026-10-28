import json
import numpy as np
from api import LLMCall
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import random
import os
import re
from FewShotPrompt import *
from GeneticOptimizer import GeneticOptimizer
from model_utils import load_embedding_model


def compute_semantic_similarity(text1: str, text2: str, embedding_model=None) -> float:

    if embedding_model is not None:
        try:
            embeddings = embedding_model.encode([text1, text2])
            emb1, emb2 = embeddings[0], embeddings[1]
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = dot_product / (norm1 * norm2)
            return (similarity + 1) / 2
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
    
    
    def tokenize(text):
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b|\b\d+\.?\d*\b', text)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return words + chinese_chars
    
    from collections import Counter
    vec1 = Counter(tokenize(text1))
    vec2 = Counter(tokenize(text2))
    
    all_words = set(vec1.keys()) | set(vec2.keys())
    if not all_words:
        return 0.0
    
    v1 = np.array([vec1.get(word, 0) for word in all_words])
    v2 = np.array([vec2.get(word, 0) for word in all_words])
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity


class NumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
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
                    q = obj.get('question', '')
                    if q:
                        completed.add(q)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Failed to read existing output file: {e}")
    return completed


def process_question_with_ga(j, ratio, ga_optimizer=None, embedding_model=None):

    q = j['raw_question']
    a = j['answer']
    m_q = j['modified_q']
    m_s = j['solution']
    
    print(f"\n{'='*60}")
    print(f"Processing question with Genetic Algorithm")
    print(f"{'='*60}")
    
    
    if ga_optimizer is not None:
        result = ga_optimizer.optimize(q, a, ratio)
        
        best_question = result['best_question']
        best_fitness = result['best_fitness']
        eval_result = result['evaluation_result']
        generations = result['generations']
        
        
        attack_successful = False
        if eval_result and eval_result.get('is_correct') is False:
            attack_successful = True
        
        
        attack_tokens = eval_result.get('attack_token_usage', {}) if eval_result else {}
        
        
        semantic_similarity = compute_semantic_similarity(q, best_question, embedding_model)
        
        ret = {
            'question': q,
            'answer': a,
            'modified_q': m_q,
            'modified_solution': m_s,
            'modified_question': best_question,
            'final_solution': eval_result.get('final_solution', '') if eval_result else '',
            'generations': generations,
            'best_fitness': best_fitness,
            'attack_successful': attack_successful,
            'ga_result': result,
            'attack_token_usage': attack_tokens,
            'semantic_similarity': semantic_similarity
        }
        
        return ret
    
    return None


def main(args):
    random.seed(args.seed)
    llm_name = args.llm_name
    dataset = args.dataset
    modify_llm_name = args.modify_llm_name
    ratio = args.ratio

    llm = LLMCall(None, llm_name)
    modify_llm = LLMCall(None, modify_llm_name)

    
    embedding_model = load_embedding_model()

    
    population_size = args.population_size if hasattr(args, 'population_size') else 10
    max_generations = args.max_generations if hasattr(args, 'max_generations') else 10

    ga_optimizer = GeneticOptimizer(
        modify_llm, dataset, args.few_shot,
        population_size=population_size,
        max_generations=max_generations,
        elite_size=2,
        tournament_size=3,
        w1=1.0,
        w2=1.0,
        w3=0.5,
        fitness_threshold=-0.1,
        target_llm=llm
    )
    print("Genetic Algorithm optimization enabled")
    print(f"  Modify LLM: {modify_llm_name}")
    print(f"  Target LLM (attacked): {llm_name}")
    print(f"  Population size: {population_size}")
    print(f"  Max generations: {max_generations}")
    print("  Loss weights: w1=1.0, w2=1.0, w3=0.5")
    print("  Fitness threshold: -0.1")
    
    if args.few_shot:
        input_file = './raw_solution_modified_q_CoT_few_shot/' + dataset + '/' + modify_llm_name + '/' + 'solution.jsonl'
        output_path = './final_solution_modified_q_CoT_few_shot_ga/' + dataset + '/' + modify_llm_name + '/' + llm_name + '/'
        output_file = output_path + 'ratio_' + str(args.ratio) + '_pop' + str(population_size) + '_gen' + str(max_generations) + '.jsonl'
    else:
        input_file = './raw_solution_modified_q_CoT/' + dataset + '/' + modify_llm_name + '/' + 'solution.jsonl'
        output_path = './final_solution_modified_q_CoT_ga/' + dataset + '/' + modify_llm_name + '/' + llm_name + '/'
        output_file = output_path + 'ratio_' + str(args.ratio) + '_pop' + str(population_size) + '_gen' + str(max_generations) + '.jsonl'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    js = read_jsonl(input_file)
    total_count = len(js)

    completed = load_completed_questions(output_file)
    if completed:
        js = [j for j in js if j['raw_question'] not in completed]
        skipped = total_count - len(js)
        print(f"\n{'='*60}")
        print(f"Resume mode: {skipped}/{total_count} questions already completed, skipped.")
        print(f"Remaining: {len(js)} questions to process.")
        print(f"{'='*60}\n")

    if len(js) == 0:
        print("All questions already processed. Nothing to do.")
        return

    file_mode = 'a' if completed else 'w'

    process_args = {
        'ratio': ratio,
        'ga_optimizer': ga_optimizer,
        'embedding_model': embedding_model,
    }

    max_workers = 3

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for j in js:
            futures.append(executor.submit(process_question_with_ga, j, **process_args))

        with open(output_file, file_mode) as f:
            for future in as_completed(futures):
                res = future.result()
                json.dump(res, f, cls=NumpyEncoder)
                f.write('\n')
                f.flush()

    print(f"\nResults saved to: {output_file}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modify_llm_name', type=str, default='qwen-plus')
    parser.add_argument('--llm_name', type=str, default='qwen')
    parser.add_argument('--dataset', type=str, default='MATH')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--few_shot', type=bool, default=False)
    
    
    parser.add_argument('--population_size', type=int, default=10, help='Population size for genetic algorithm (recommended 8-15)')
    parser.add_argument('--max_generations', type=int, default=3, help='Maximum generations for genetic algorithm')
    
    args = parser.parse_args()

    main(args)
