import numpy as np
import argparse
from EvaluationUtils import *


def update_generation_stats(j, is_ga_result, generation_list, generation_success_rates):
    
    if is_ga_result and 'generations' in j:
        generation_list.append(j['generations'])
        gen_num = j['generations']
        if gen_num not in generation_success_rates:
            generation_success_rates[gen_num] = {'total': 0, 'success': 0}
        generation_success_rates[gen_num]['total'] += 1


def update_token_stats(j, q2raw_tokens, raw_tokens_list, attack_tokens_list):
    
    q = j['question']
    if q in q2raw_tokens:
        raw_tokens_list.append(q2raw_tokens[q])
    if 'attack_token_usage' in j:
        attack_tokens_list.append(j['attack_token_usage'])


def update_similarity_stats(j, semantic_similarities):
    
    if 'semantic_similarity' in j:
        semantic_similarities.append(j['semantic_similarity'])

def update_success_stats(j, eval_ret, eval_raw_ret, is_ga_result, correct, raw_correct, success, generation_success_rates):
    
    if eval_ret is not None:
        correct.append(eval_ret)
    if eval_raw_ret is not None:
        raw_correct.append(eval_raw_ret)
        if eval_raw_ret == True:
            if eval_ret is None or eval_ret == False:
                success.append(True)
                if is_ga_result and 'generations' in j:
                    generation_success_rates[j['generations']]['success'] += 1
            else:
                success.append(False)


def main(args):
    input_file = args.input_file

    raw_file = read_jsonl(args.raw_input_file)
    q2raw_s = {}
    q2raw_tokens = {}
    for j in raw_file:
        q = j['question']
        raw_s = j['raw_solution']
        q2raw_s[q] = raw_s
        if 'raw_token_usage' in j:
            q2raw_tokens[q] = j['raw_token_usage']

    js = read_jsonl(input_file)

    
    is_ga_result = any('generations' in j for j in js)

    
    generation_list = []
    generation_success_rates = {}
    
    
    raw_tokens_list = []
    attack_tokens_list = []
    
    
    semantic_similarities = []

    if 'MATH' in input_file and 'MATHQA' not in input_file:
        raw_correct = []
        correct = []
        success = []

        for j in js:
            pred = j['final_solution']
            pred_raw = q2raw_s[j['question']]
            ans = j['answer']
            eval_ret = eval_answer_math(pred, ans, 1.0)
            eval_raw_ret = eval_answer_math(pred_raw, ans, 1.0)

            update_generation_stats(j, is_ga_result, generation_list, generation_success_rates)
            update_token_stats(j, q2raw_tokens, raw_tokens_list, attack_tokens_list)
            update_similarity_stats(j, semantic_similarities)
            update_success_stats(j, eval_ret, eval_raw_ret, is_ga_result, correct, raw_correct, success, generation_success_rates)

    elif 'gsm8k' in input_file:
        raw_correct = []
        correct = []
        success = []
        for j in js:
            pred = j['final_solution']
            pred_raw = q2raw_s[j['question']]
            ans = j['answer']
            eval_ret = eval_answer_gsm8k(pred, ans)
            eval_raw_ret = eval_answer_gsm8k(pred_raw, ans)

            update_generation_stats(j, is_ga_result, generation_list, generation_success_rates)
            update_token_stats(j, q2raw_tokens, raw_tokens_list, attack_tokens_list)
            update_similarity_stats(j, semantic_similarities)
            update_success_stats(j, eval_ret, eval_raw_ret, is_ga_result, correct, raw_correct, success, generation_success_rates)
 
    elif 'csqa' in input_file:
        raw_correct = []
        correct = []
        success = []
        for j in js:
            pred = j['final_solution']
            pred_raw = q2raw_s[j['question']]
            ans = j['answer']
            eval_ret = eval_answer_cs_qa(pred, ans)
            eval_raw_ret = eval_answer_cs_qa(pred_raw, ans)

            update_generation_stats(j, is_ga_result, generation_list, generation_success_rates)
            update_token_stats(j, q2raw_tokens, raw_tokens_list, attack_tokens_list)
            update_similarity_stats(j, semantic_similarities)
            update_success_stats(j, eval_ret, eval_raw_ret, is_ga_result, correct, raw_correct, success, generation_success_rates)


    else:
        raw_correct = []
        correct = []
        success = []
        for j in js:
            pred = j['final_solution']
            pred_raw = q2raw_s[j['question']]
            ans = j['answer']
            eval_ret = eval_answer_math_qa(pred, ans)
            eval_raw_ret = eval_answer_math_qa(pred_raw, ans)

            update_generation_stats(j, is_ga_result, generation_list, generation_success_rates)
            update_token_stats(j, q2raw_tokens, raw_tokens_list, attack_tokens_list)
            update_similarity_stats(j, semantic_similarities)
            update_success_stats(j, eval_ret, eval_raw_ret, is_ga_result, correct, raw_correct, success, generation_success_rates)
            
    
    if len(js) == 0:
        print('Error: No data found in input file')
        return 0, 0

    print('Raw Accuracy:', np.sum(raw_correct)/len(js))
    print('Accuracy:', np.sum(correct)/len(js))
    print('ASR:', np.mean(success) if success else 0)

    
    raw_acc_mean = np.mean(raw_correct)
    if raw_acc_mean > 0:
        acc_drop = (np.mean(correct) - raw_acc_mean) / raw_acc_mean
        print('ACC_Drop:', acc_drop)
    else:
        print('ACC_Drop: N/A (raw accuracy is 0)')

    
    if is_ga_result and generation_list:
        print('\n' + '='*50)
        print('Genetic Algorithm Statistics:')
        print('='*50)
        print(f'Total questions processed: {len(js)}')
        print(f'Average generations: {np.mean(generation_list):.2f}')
        print(f'Max generations: {max(generation_list)}')
        print(f'Min generations: {min(generation_list)}')
        print('\nSuccess rate by generation:')
        for gen_num in sorted(generation_success_rates.keys()):
            stats = generation_success_rates[gen_num]
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f'  Generation {gen_num}: {stats["success"]}/{stats["total"]} ({success_rate:.2%})')

        
        cumulative_success = 0
        cumulative_total = 0
        print('\nCumulative success rate:')
        for gen_num in sorted(generation_success_rates.keys()):
            stats = generation_success_rates[gen_num]
            cumulative_success += stats['success']
            cumulative_total += stats['total']
            cumulative_rate = cumulative_success / cumulative_total if cumulative_total > 0 else 0
            print(f'  After {gen_num} generation(s): {cumulative_success}/{cumulative_total} ({cumulative_rate:.2%})')
        print('='*50)

    
    print('\n' + '='*50)
    print('Token Usage Statistics:')
    print('='*50)
    
    if raw_tokens_list:
        raw_completion_tokens = [t['completion_tokens'] for t in raw_tokens_list]
        print(f'Raw Model Average Output Tokens: {np.mean(raw_completion_tokens):.2f}')
    
    if attack_tokens_list:
        attack_completion_tokens = [t['completion_tokens'] for t in attack_tokens_list]
        print(f'Attacked Model Average Output Tokens: {np.mean(attack_completion_tokens):.2f}')
    else:
        print('No token data available')
    
    print('='*50)

    
    print('\n' + '='*50)
    print('Semantic Similarity Statistics:')
    print('='*50)
    
    if semantic_similarities:
        print(f'Average Semantic Similarity: {np.mean(semantic_similarities):.4f}')
    else:
        print('No semantic similarity data available')
    
    print('='*50)

    return np.mean(correct) if correct else 0, np.mean(success) if success else 0
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default='qwen', help='Model name for evaluation')
    parser.add_argument('--dataset', type=str, default='MATH', help='Dataset name')
    parser.add_argument('--few_shot', type=bool, default=False, help='Use few-shot prompting')
    parser.add_argument('--ratio', type=float, default=0.6, help='Attack ratio')
    parser.add_argument('--seed', type=str, default='', help='Random seed (unused)')
    parser.add_argument('--modify_llm_name', type=str, default='qwen-plus')
    parser.add_argument('--num', type=str , default='500')
    
    parser.add_argument('--ga', type=bool, default=False, help='Evaluate genetic algorithm results')
    parser.add_argument('--population_size', type=int, default=10, help='Population size for GA')
    parser.add_argument('--max_generations', type=int, default=10, help='Max generations for GA')
    parser.add_argument('--ablation', type=str, default='',
                        choices=['', 'coherence'],
                        help='GA variant subdirectory suffix (coherence = LLM-judged coherence loss)')

    args = parser.parse_args()

    
    if hasattr(args, 'llm') and not hasattr(args, 'llm_name'):
        args.llm_name = args.llm

    
    base_dir = args.num + '/' + 'final_solution_modified_q_CoT_few_shot' if args.few_shot else args.num + '/' + 'final_solution_modified_q_CoT'
    ga_dir = args.num + '/' + 'final_solution_modified_q_CoT_few_shot_ga' if args.few_shot else  args.num + '/' + 'final_solution_modified_q_CoT_ga'
    if args.ablation:
        ga_dir = ga_dir + '_' + args.ablation

    
    if args.few_shot:
        args.raw_input_file = './' + base_dir + '/' + args.dataset + '/' + args.modify_llm_name + '/' + args.llm_name + '/ratio_0.0_reasoning_steps.jsonl'
    else:
        args.raw_input_file = './' + base_dir + '/' + args.dataset + '/' + args.modify_llm_name + '/' + args.llm_name + '/ratio_0.0_reasoning_steps.jsonl'

    print('ratio:', args.ratio)
    print('ga:', args.ga)
    print('ablation:', args.ablation if args.ablation else 'none')

    
    if args.ga:
        
        args.input_file = './' + ga_dir + '/' + args.dataset + '/' + args.modify_llm_name + '/' + args.llm_name + '/ratio_' + str(args.ratio) + '_pop' + str(args.population_size) + '_gen' + str(args.max_generations) + '.jsonl'
        print('input_file:', args.input_file)
    else:
        
        if args.few_shot:
            args.input_file = './' + base_dir + '/' + args.dataset + '/' + args.modify_llm_name + '/' + args.llm_name + '/' + 'ratio_' + str(args.ratio) + '_reasoning_steps.jsonl'
        else:
            args.input_file = './' + base_dir + '/' + args.dataset + '/' + args.modify_llm_name + '/' + args.llm_name + '/' + 'ratio_' + str(args.ratio) + '_reasoning_steps.jsonl'

    main(args)
                    
