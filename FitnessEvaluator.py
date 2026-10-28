import re
import numpy as np
from api import LLMCall
from EvaluationUtils import eval_answer_math, eval_answer_gsm8k, eval_answer_cs_qa, eval_answer_math_qa
from FewShotPrompt import prompt, prompt_choice, FS_CoT_MATH, FS_CoT_GSM, FS_CoT_CSQA, FS_CoT_MATHQA, FS_CoT_STRATEGYQA
from model_utils import load_embedding_model


class FitnessEvaluator:

    REBUTTAL_KEYWORDS = [
         # 转折 / 质疑
        "but", "however", "wait", "actually", "in fact", "incorrect",
        "wrong", "error", "mistake", "hold on", "on second thought",
        # 自我纠正（短语）
        "let me reconsider", "let me re-examine", "let me re-evaluate",
        "let me recheck", "let me check again", "on reflection",
        "upon reflection", "i was wrong", "i made a mistake", "my mistake",
        "i made an error", "that's not right", "that is not right",
        "i apologize", "apologies", "scratch that", "i take that back",
        "let me start over", "start over", "never mind", "disregard",
        # 自我纠正（词）
        "reconsider", "re-evaluate", "reevaluate", "re-examine", "recheck",
        "oops", "sorry", "hmm"
    ]
    
    def __init__(self, llm: LLMCall, dataset: str, few_shot: bool = False,
                 w1: float = 1.0, w2: float = 1.0, w3: float = 0.5,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 modify_llm: LLMCall = None):
        

        self.llm = llm  
        self.modify_llm = modify_llm if modify_llm is not None else llm  
        self.dataset = dataset
        self.few_shot = few_shot
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.evaluation_cache = {}
        self.cache_hits = 0
        self.total_evaluations = 0
        
        
        self.embedding_model = load_embedding_model()
        
    def generate_solution_for_modified_question(self, modified_q: str) -> str:
        
        if self.dataset in ['csqa', 'strategyqa', 'MATHQA']:
            prompt_selected = prompt_choice
        else:
            prompt_selected = prompt

        query = ""
        if self.few_shot:
            if self.dataset == 'MATH':
                query = FS_CoT_MATH.format(problem=modified_q) + '[Solution] \n\n'
            elif self.dataset == 'gsm8k':
                query = FS_CoT_GSM.format(problem=modified_q) + '[Solution] \n\n'
            elif self.dataset == 'csqa':
                query = FS_CoT_CSQA.format(problem=modified_q) + '[Solution] \n\n'
            elif self.dataset == 'strategyqa':
                query = FS_CoT_STRATEGYQA.format(problem=modified_q) + '[Solution] \n\n'
            elif self.dataset == 'MATHQA':
                query = FS_CoT_MATHQA.format(problem=modified_q) + '[Solution] \n\n'
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                       {'role': 'user', 'content': prompt_selected + query}]
        else:
            query = '[Question] \n\n' + modified_q + '[Solution] \n\n'
            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                       {'role': 'user', 'content': prompt_selected + query}]

        return self.modify_llm.call(messages)
    
    def extract_modified_answer(self, modified_solution: str) -> str:
        
        if 'The answer is:' in modified_solution:
            return modified_solution.split('The answer is:')[1].strip()
        elif 'the answer is:' in modified_solution:
            return modified_solution.split('the answer is:')[1].strip()
        elif 'The answer choice is:' in modified_solution:
            return modified_solution.split('The answer choice is:')[1].strip()
        return ""
    
    def extract_partial_reasoning(self, modified_solution: str, ratio: float) -> str:
        
        steps = modified_solution.split('\n\n')
        len_sol = len(steps)
        idx = max(ratio * len_sol - 1, 0)
        partial = '\n\n'.join(steps[:int(idx) + 1])
        
        
        if 'The answer is:' in partial:
            partial = partial.split('The answer is:')[0]
        if 'the answer is:' in partial:
            partial = partial.split('the answer is:')[0]
        if 'The answer choice is:' in partial:
            partial = partial.split('The answer choice is:')[0]
        
        return partial
    
    def generate_full_response(self, raw_question: str, partial_reasoning: str,
                               modified_ans: str) -> tuple:
        
        if self.few_shot:
            if self.dataset == 'MATH':
                query = FS_CoT_MATH.format(problem=raw_question) + 'The answer is:' + modified_ans + '\n\n[Solution] \n\n' + partial_reasoning
            elif self.dataset == 'gsm8k':
                query = FS_CoT_GSM.format(problem=raw_question) + 'The answer is:' + modified_ans + '\n\n[Solution] \n\n' + partial_reasoning
            elif self.dataset == 'csqa':
                query = FS_CoT_CSQA.format(problem=raw_question) + 'The answer is:' + modified_ans + '\n\n[Solution] \n\n' + partial_reasoning
            elif self.dataset == 'MATHQA':
                query = FS_CoT_MATHQA.format(problem=raw_question) + 'The answer is:' + modified_ans + '\n\n[Solution] \n\n' + partial_reasoning
            else:
                query = raw_question + '\n\nThe answer is:' + modified_ans + '\n\n' + '[Solution]\n\n' + partial_reasoning
        else:
            query = raw_question + '\n\nThe answer is:' + modified_ans + '\n\n' + '[Solution]\n\n' + partial_reasoning
        
        
        if self.dataset in ['csqa', 'strategyqa', 'MATHQA']:
            prompt_selected = prompt_choice
        else:
            prompt_selected = prompt
            
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                   {'role': 'user', 'content': prompt_selected + query}]
        
        
        result = self.llm.call(messages, return_tokens=True)
        response = result['content']
        token_usage = result['token_usage']
        
        return response, token_usage
    
    def compute_success_loss(self, final_solution: str, correct_answer: str) -> tuple:
        

        if self.dataset == 'MATH':
            eval_ret = eval_answer_math(final_solution, correct_answer, 1.0)
        elif self.dataset == 'gsm8k':
            eval_ret = eval_answer_gsm8k(final_solution, correct_answer)
        elif self.dataset == 'csqa':
            eval_ret = eval_answer_cs_qa(final_solution, correct_answer)
        elif self.dataset == 'MATHQA':
            eval_ret = eval_answer_math_qa(final_solution, correct_answer)
        else:
            eval_ret = None
        

        is_correct = eval_ret is True
        loss = 0.0 if eval_ret is None or eval_ret == False else 10.0
        
        return loss, is_correct, eval_ret
    
    def compute_stealth_loss(self, modified_question: str, raw_question: str) -> float:
        


        if self.embedding_model is not None:
            
            try:
                
                embeddings = self.embedding_model.encode([modified_question, raw_question])
                
                
                emb1 = embeddings[0]
                emb2 = embeddings[1]
                
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm1 * norm2)
                
                
                similarity = (similarity + 1) / 2
                
                
                loss = 1.0 - similarity
                
                return loss
                
            except Exception as e:
                print(f"Error computing semantic similarity: {e}")
                
                pass
        
        
        def tokenize(text):
            
            
            text = text.lower()
            
            words = re.findall(r'\b[a-z]+\b|\b\d+\.?\d*\b', text)
            
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            return words + chinese_chars
        
        def text_to_vector(text):
            
            tokens = tokenize(text)
            from collections import Counter
            return Counter(tokens)
        
        def cosine_similarity(vec1, vec2):
            
            
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
            
            return dot_product / (norm1 * norm2)
        
        vec1 = text_to_vector(modified_question)
        vec2 = text_to_vector(raw_question)
        
        similarity = cosine_similarity(vec1, vec2)
        
        
        loss = 1.0 - similarity
        
        return loss
    
    def evaluate_individual(self, modified_question: str, raw_question: str,
                           correct_answer: str, ratio: float = 0.6) -> dict:
        

        self.total_evaluations += 1
     
        cache_key = hash(modified_question)
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            print(f"  [Cache hit] Saved API call for: {modified_question[:50]}...")
            return self.evaluation_cache[cache_key].copy()
        
        result = {
            'modified_question': modified_question,
            'raw_question': raw_question,
            'correct_answer': correct_answer,
            'ratio': ratio,
            'fitness': 0.0,
            'bridge_loss': 0.0,
            'success_loss': 0.0,
            'stealth_loss': 0.0,
            'has_rebuttal': False,
            'is_correct': None,
            'modified_solution': None,
            'partial_reasoning': None,
            'final_solution': None,
            'rebuttal_keywords': []
        }
        
        try:
            
            modified_solution = self.generate_solution_for_modified_question(modified_question)
            result['modified_solution'] = modified_solution
            
            
            modified_ans = self.extract_modified_answer(modified_solution)
            partial_reasoning = self.extract_partial_reasoning(modified_solution, ratio)
            result['partial_reasoning'] = partial_reasoning
            
            
            full_response, token_usage = self.generate_full_response(
                raw_question, partial_reasoning, modified_ans
            )
            result['final_solution'] = full_response
            result['attack_token_usage'] = token_usage
            
            
            rebuttal_keywords_found = []
            response_lower = full_response.lower()
            for keyword in self.REBUTTAL_KEYWORDS:
                if keyword.lower() in response_lower:
                    rebuttal_keywords_found.append(keyword)
            
            has_rebuttal = len(rebuttal_keywords_found) > 0
            bridge_loss = 1.0 if has_rebuttal else 0.0
            
            result['bridge_loss'] = bridge_loss
            result['has_rebuttal'] = has_rebuttal
            result['rebuttal_keywords'] = rebuttal_keywords_found
            
            
            success_loss, is_correct, eval_ret = self.compute_success_loss(
                full_response, correct_answer
            )
            
            result['success_loss'] = success_loss
            result['is_correct'] = is_correct
            
            
            stealth_loss = self.compute_stealth_loss(modified_question, raw_question)
            result['stealth_loss'] = stealth_loss
            
            
            total_loss = (self.w1 * bridge_loss + 
                         self.w2 * success_loss + 
                         self.w3 * stealth_loss)
            
            result['total_loss'] = total_loss
            result['fitness'] = -total_loss
            
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            result['fitness'] = -1000.0  
            result['error'] = str(e)
        
        return result
    
    def quick_evaluate(self, modified_question: str, raw_question: str,
                      correct_answer: str, ratio: float = 0.6) -> float:
        

        result = self.evaluate_individual(modified_question, raw_question, 
                                         correct_answer, ratio)
        return result['fitness']


class Individual:
    
    
    def __init__(self, question: str, fitness: float = None):
        self.question = question
        self.fitness = fitness
        self.evaluation_result = None
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness}, question={self.question[:50]}...)"
    
    def __lt__(self, other):
        
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness > other.fitness
