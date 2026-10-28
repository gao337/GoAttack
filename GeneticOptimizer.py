import random
import json
from typing import List, Optional
from api import LLMCall
from FitnessEvaluator import FitnessEvaluator, Individual
from GeneticOperators import GeneticOperators, SelectionOperator
from FewShotPrompt import prompt, prompt_choice

class GeneticOptimizer:

    def __init__(self, llm: LLMCall, dataset: str, few_shot: bool = False,
                 population_size: int = 10, elite_size: int = 2,
                 tournament_size: int = 3, max_generations: int = 10,
                 w1: float = 1.0, w2: float = 1.0, w3: float = 0.5,
                 fitness_threshold: float = None,
                 target_llm: LLMCall = None):

        self.llm = llm  
        self.target_llm = target_llm if target_llm is not None else llm  
        self.dataset = dataset
        self.few_shot = few_shot
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.fitness_threshold = fitness_threshold
        
        
        
        self.fitness_evaluator = FitnessEvaluator(
            self.target_llm, dataset, few_shot, w1, w2, w3,
            modify_llm=llm
        )
        self.genetic_operators = GeneticOperators(llm, dataset)
        
        
        self.history = []
    
    def generate_initial_individual(self, raw_question: str) -> str:

        if self.dataset in ['csqa', 'MATHQA']:
            prompt_selected = prompt_choice
        else:
            prompt_selected = prompt

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt_selected + raw_question}
        ]
        response = self.llm.call(messages)
        messages.append({'role': 'assistant', 'content': response})
        messages.append({
            'role': 'user',
            'content': "Create a new question that logically and strictly leads to a different answer. Output the constructed question after the `[Question]` without any explanation or note.\n"
        })
        response = self.llm.call(messages)

        modified_q = response.split('[Question]')[-1].strip()

        
        if self.dataset == 'csqa':
            if 'Answer Choices:' not in modified_q:
                modified_q = modified_q + ' Answer Choices:' + raw_question.split('Answer Choices:')[-1]
        elif self.dataset == 'MATHQA':
            if '\na )' not in modified_q:
                modified_q = modified_q + '\na )' + raw_question.split('\na )')[-1]

        return modified_q
    
    def initialize_population(self, raw_question: str) -> List[Individual]:

        population = []
        print(f"Initializing population of size {self.population_size}...")
        
        for i in range(self.population_size):
            modified_q = self.generate_initial_individual(raw_question)
            individual = Individual(modified_q)
            population.append(individual)
            print(f"  Individual {i+1}/{self.population_size} created")
        
        return population
    
    def evaluate_population(self, population: List[Individual], 
                           raw_question: str, correct_answer: str,
                           ratio: float = 0.6) -> bool:

        print("Evaluating population fitness...")
        
        for i, individual in enumerate(population):
            print(f"  Evaluating individual {i+1}/{len(population)}...")
            result = self.fitness_evaluator.evaluate_individual(
                individual.question, raw_question, correct_answer, ratio
            )
            individual.fitness = result['fitness']
            individual.evaluation_result = result
            
            
            if self.fitness_threshold is not None and result['fitness'] >= self.fitness_threshold:
                print(f"  [Early stop] Fitness threshold reached at individual {i+1}! (fitness={result['fitness']:.4f})")
                return True
        
        return False
    
    def select_parents(self, population: List[Individual]) -> tuple:

        parent1 = SelectionOperator.tournament_selection(population, self.tournament_size)
        parent2 = SelectionOperator.tournament_selection(population, self.tournament_size)
        return parent1, parent2
    
    def create_offspring(self, parent1: Individual, parent2: Individual,
                        raw_question: str, fitness_threshold: float = -1.5) -> Individual:

        p1_fitness = parent1.fitness if parent1.fitness is not None else 0.0
        p2_fitness = parent2.fitness if parent2.fitness is not None else 0.0
        avg_fitness = (p1_fitness + p2_fitness) / 2.0
        
        
        if avg_fitness >= fitness_threshold:
            
            child_question = self.genetic_operators.semantic_crossover(
                parent1.question, parent2.question, raw_question
            )
        else:
            
            selected_parent = parent1 if p1_fitness >= p2_fitness else parent2
            child_question = self.genetic_operators.multi_strategy_mutation(
                selected_parent.question, raw_question
            )
        
        return Individual(child_question)
    
    def evolve(self, population: List[Individual], raw_question: str) -> List[Individual]:

        new_population = []
        
        
        elites = SelectionOperator.elitism_selection(population, self.elite_size)
        for elite in elites:
            new_population.append(Individual(elite.question, elite.fitness))
            new_population[-1].evaluation_result = elite.evaluation_result
        
        print(f"  Elite preservation: {len(elites)} individuals")
        
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(population)
            offspring = self.create_offspring(parent1, parent2, raw_question)
            new_population.append(offspring)
        
        return new_population
    
    def check_termination(self, population: List[Individual], generation: int) -> tuple:

        best = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        best_fitness = best.fitness if best.fitness is not None else float('-inf')
        
        
        if self.fitness_threshold is not None and best_fitness >= self.fitness_threshold:
            return True, best, "fitness_threshold_reached"
        
        
        if generation >= self.max_generations:
            return True, best, "max_generations_reached"

        return False, None, None
    
    def optimize(self, raw_question: str, correct_answer: str, 
                ratio: float = 0.6) -> dict:

        print(f"\n{'='*60}")
        print(f"Starting Genetic Algorithm Optimization")
        print(f"{'='*60}")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"{'='*60}\n")
        
        
        population = self.initialize_population(raw_question)
        
        
        threshold_reached = self.evaluate_population(population, raw_question, correct_answer, ratio)
        
        
        best_initial = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        self.history = [{
            'generation': 0,
            'best_fitness': best_initial.fitness,
            'avg_fitness': sum(ind.fitness for ind in population if ind.fitness is not None) / len(population),
            'best_question': best_initial.question[:100] + '...'
        }]
        
        
        if threshold_reached:
            print(f"\n[Early Termination] Fitness threshold reached in initial population!")
            return {
                'best_question': best_initial.question,
                'best_fitness': best_initial.fitness,
                'evaluation_result': best_initial.evaluation_result,
                'generations': 0,
                'termination_reason': 'early_stop_fitness_threshold_initial',
                'history': self.history
            }
        
        
        for generation in range(1, self.max_generations + 1):
            print(f"\n{'='*40}")
            print(f"Generation {generation}/{self.max_generations}")
            print(f"{'='*40}")
            
            
            population = self.evolve(population, raw_question)
            
            
            new_individuals = [ind for ind in population if ind.fitness is None]
            if new_individuals:
                threshold_reached = self.evaluate_population(new_individuals, raw_question, correct_answer, ratio)
                if threshold_reached:
                    
                    threshold_ind = next((ind for ind in new_individuals 
                                         if ind.fitness is not None and ind.fitness >= self.fitness_threshold), 
                                        new_individuals[0])
                    print(f"\n[Early Termination] Fitness threshold reached at generation {generation}!")
                    return {
                        'best_question': threshold_ind.question,
                        'best_fitness': threshold_ind.fitness,
                        'evaluation_result': threshold_ind.evaluation_result,
                        'generations': generation,
                        'termination_reason': 'early_stop_fitness_threshold',
                        'history': self.history
                    }
            
            
            best = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
            valid_fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
            avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else 0
            
            self.history.append({
                'generation': generation,
                'best_fitness': best.fitness,
                'avg_fitness': avg_fitness,
                'best_question': best.question[:100] + '...'
            })
            
            print(f"\nGeneration {generation} Summary:")
            print(f"  Best fitness: {best.fitness:.4f}")
            print(f"  Avg fitness: {avg_fitness:.4f}")
            
            
            should_terminate, best_individual, reason = self.check_termination(population, generation)
            
            if should_terminate:
                print(f"\nTermination: {reason}")
                
                
                if best_individual is None:
                    best_individual = best
                
                return {
                    'best_question': best_individual.question,
                    'best_fitness': best_individual.fitness,
                    'evaluation_result': best_individual.evaluation_result,
                    'generations': generation,
                    'termination_reason': reason,
                    'history': self.history
                }
        
        
        best = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
        return {
            'best_question': best.question,
            'best_fitness': best.fitness,
            'evaluation_result': best.evaluation_result,
            'generations': self.max_generations,
            'termination_reason': 'max_generations_reached',
            'history': self.history
        }
