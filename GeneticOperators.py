import random
import re
from api import LLMCall

class GeneticOperators:
    
    def __init__(self, llm: LLMCall, dataset: str):
        self.llm = llm
        self.dataset = dataset
    def semantic_crossover(self, parent1: str, parent2: str, raw_question: str) -> str:

        crossover_prompt = f"""You are tasked with creating a new math question by combining features from two modified versions of an original question.

Original Question:
{raw_question}

Modified Version 1:
{parent1}

Modified Version 2:
{parent2}

Your task:
1. Analyze the key modifications in both versions (e.g., different numbers, changed conditions, altered entities)
2. Create a NEW question that combines the best aspects of both modifications
3. The new question should be solvable and lead to a different answer than the original
4. Output ONLY the new question after `[Question]` without any explanation

Format your response as:
[Question]
[Your new question here]
"""
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                   {'role': 'user', 'content': crossover_prompt}]
        
        try:
            response = self.llm.call(messages)

            if '[Question]' in response:
                child = response.split('[Question]')[-1].strip()
            else:
                child = response.strip()

            if self.dataset == 'csqa':
                if 'Answer Choices:' not in child:
                    child = child + ' Answer Choices:' + raw_question.split('Answer Choices:')[-1]
            elif self.dataset == 'MATHQA':
                if '\na )' not in child:
                    child = child + '\na )' + raw_question.split('\na )')[-1]
            
            return child
            
        except Exception as e:
            print(f"Crossover error: {e}")
            
            return random.choice([parent1, parent2])
    
    def mutate_number(self, question: str, mutation_strength: str = "medium") -> str:

        numbers = re.findall(r'\b\d+\.?\d*\b', question)
        
        if not numbers:
            return question
        
        
        target_num = random.choice(numbers)
        
        try:
            num_value = float(target_num)
            
            
            if mutation_strength == "small":
                factor = random.uniform(0.9, 1.1)
            elif mutation_strength == "medium":
                factor = random.uniform(0.7, 1.3)
            else:  
                factor = random.uniform(0.5, 1.5)
            
            new_num = num_value * factor
            
            
            if '.' in target_num:
                new_num_str = f"{new_num:.2f}"
            else:
                new_num_str = str(int(round(new_num)))
            
            
            mutated = question.replace(target_num, new_num_str, 1)
            return mutated
            
        except:
            return question
    
    def mutate_entity(self, question: str, raw_question: str) -> str:

        mutation_prompt = f"""You are tasked with modifying a math question by changing some entities while keeping the problem structure.

Original Question:
{raw_question}

Current Modified Question:
{question}

Your task:
1. Identify entities in the question (names, objects, places, etc.)
2. Replace at least one entity with a different one
3. Keep the mathematical structure and solvability
4. Output ONLY the modified question after `[Question]` without any explanation

Format your response as:
[Question]
[Your modified question here]
"""
        
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                   {'role': 'user', 'content': mutation_prompt}]
        
        try:
            response = self.llm.call(messages)
            
            if '[Question]' in response:
                mutated = response.split('[Question]')[-1].strip()
            else:
                mutated = response.strip()
            
            
            if self.dataset == 'csqa':
                if 'Answer Choices:' not in mutated:
                    mutated = mutated + ' Answer Choices:' + raw_question.split('Answer Choices:')[-1]
            elif self.dataset == 'MATHQA':
                if '\na )' not in mutated:
                    mutated = mutated + '\na )' + raw_question.split('\na )')[-1]
            
            return mutated
            
        except Exception as e:
            print(f"Entity mutation error: {e}")
            return question
    
    def mutate_sentence_structure(self, question: str, raw_question: str) -> str:

        mutation_prompt = f"""You are tasked with rewriting a math question using different sentence structure.

Original Question:
{raw_question}

Current Modified Question:
{question}

Your task:
1. Rewrite the question using different sentence structure or wording
2. Keep all numbers and the mathematical meaning the same
3. The question should still be solvable and lead to the same answer
4. Output ONLY the rewritten question after `[Question]` without any explanation

Format your response as:
[Question]
[Your rewritten question here]
"""
        
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                   {'role': 'user', 'content': mutation_prompt}]
        
        try:
            response = self.llm.call(messages)
            
            if '[Question]' in response:
                mutated = response.split('[Question]')[-1].strip()
            else:
                mutated = response.strip()
            
            
            if self.dataset == 'csqa':
                if 'Answer Choices:' not in mutated:
                    mutated = mutated + ' Answer Choices:' + raw_question.split('Answer Choices:')[-1]
            elif self.dataset == 'MATHQA':
                if '\na )' not in mutated:
                    mutated = mutated + '\na )' + raw_question.split('\na )')[-1]
            
            return mutated
            
        except Exception as e:
            print(f"Sentence structure mutation error: {e}")
            return question
    
    def mutate_logic(self, question: str, raw_question: str) -> str:

        mutation_prompt = f"""You are tasked with modifying a math question by changing its logical conditions.

Original Question:
{raw_question}

Current Modified Question:
{question}

Your task:
1. Identify the logical conditions in the question (e.g., "if", "when", "given that", "suppose")
2. Modify at least one condition to create a different scenario
3. The new question should be solvable and lead to a different answer
4. Output ONLY the modified question after `[Question]` without any explanation

Format your response as:
[Question]
[Your modified question here]
"""
        
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                   {'role': 'user', 'content': mutation_prompt}]
        
        try:
            response = self.llm.call(messages)
            
            if '[Question]' in response:
                mutated = response.split('[Question]')[-1].strip()
            else:
                mutated = response.strip()
            
            
            if self.dataset == 'csqa':
                if 'Answer Choices:' not in mutated:
                    mutated = mutated + ' Answer Choices:' + raw_question.split('Answer Choices:')[-1]
            elif self.dataset == 'MATHQA':
                if '\na )' not in mutated:
                    mutated = mutated + '\na )' + raw_question.split('\na )')[-1]
            
            return mutated
            
        except Exception as e:
            print(f"Logic mutation error: {e}")
            return question
    
    def multi_strategy_mutation(self, question: str, raw_question: str) -> str:

        strategies = [
            ('number', 0.3),      
            ('entity', 0.25),     
            ('sentence', 0.25),   
            ('logic', 0.2)        
        ]
        
        
        r = random.random()
        cumulative = 0
        selected_strategy = 'number'
        
        for strategy, weight in strategies:
            cumulative += weight
            if r <= cumulative:
                selected_strategy = strategy
                break
        
        
        if selected_strategy == 'number':
            return self.mutate_number(question)
        elif selected_strategy == 'entity':
            return self.mutate_entity(question, raw_question)
        elif selected_strategy == 'sentence':
            return self.mutate_sentence_structure(question, raw_question)
        else:  
            return self.mutate_logic(question, raw_question)
    
    def apply_crossover(self, parent1: str, parent2: str, raw_question: str) -> str:
        return self.semantic_crossover(parent1, parent2, raw_question)
    
    def apply_mutation(self, question: str, raw_question: str) -> str:
        return self.multi_strategy_mutation(question, raw_question)


class SelectionOperator:

    @staticmethod
    def tournament_selection(population: list, tournament_size: int = 3) -> object:

        tournament = random.sample(population, min(tournament_size, len(population)))
        
        
        return max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
    
    @staticmethod
    def roulette_wheel_selection(population: list) -> object:

        fitnesses = [ind.fitness if ind.fitness is not None else 0 for ind in population]
        min_fitness = min(fitnesses)
        
        
        adjusted_fitnesses = [f - min_fitness + 1 for f in fitnesses]
        total_fitness = sum(adjusted_fitnesses)
        
        if total_fitness == 0:
            return random.choice(population)
        
        
        r = random.uniform(0, total_fitness)
        cumulative = 0
        
        for ind, adj_fit in zip(population, adjusted_fitnesses):
            cumulative += adj_fit
            if cumulative >= r:
                return ind
        
        return population[-1]
    
    @staticmethod
    def elitism_selection(population: list, elite_size: int = 2) -> list:

        sorted_population = sorted(population, 
                                   key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
                                   reverse=True)
        
        return sorted_population[:elite_size]
