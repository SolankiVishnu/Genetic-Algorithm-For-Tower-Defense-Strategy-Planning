# src/genetic_algorithm.py

from typing import List, Tuple
import numpy as np

from .game_env import TowerDefenseGame
from .strategy_encoding import Chromosome, random_chromosome
from .evaluation import evaluate_population


class GeneticAlgorithm:
    def __init__(
        self,
        game: TowerDefenseGame,
        population_size: int = 40,
        num_generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        budget: float = 80.0,
        num_simulations_eval: int = 3,
        seed: int = 42,
    ):
        self.game = game
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.budget = budget
        self.num_simulations_eval = num_simulations_eval
        self.rng = np.random.default_rng(seed)

        self.population: List[Chromosome] = []
        self.best_history: List[float] = []
        self.avg_history: List[float] = []

    def initialize_population(self):
        self.population = [
            random_chromosome(self.game, self.rng)
            for _ in range(self.population_size)
        ]

    def tournament_selection(self, fitnesses: np.ndarray, k: int = 3) -> Chromosome:
        indices = self.rng.integers(0, self.population_size, size=k)
        best_idx = indices[0]
        best_fit = fitnesses[best_idx]
        for idx in indices[1:]:
            if fitnesses[idx] > best_fit:
                best_fit = fitnesses[idx]
                best_idx = idx
        return self.population[best_idx]

    def single_point_crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        if len(genes1) <= 1:
            return Chromosome(genes1), Chromosome(genes2)

        point = self.rng.integers(1, len(genes1))  # between 1 and len-1
        child1_genes = np.concatenate([genes1[:point], genes2[point:]])
        child2_genes = np.concatenate([genes2[:point], genes1[point:]])
        return Chromosome(child1_genes), Chromosome(child2_genes)

    def mutate(self, chromosome: Chromosome):
        """Randomly change tower types in some slots."""
        for i in range(len(chromosome.genes)):
            if self.rng.random() < self.mutation_rate:
                chromosome.genes[i] = self.rng.integers(
                    0,
                    self.game.num_tower_types + 1,
                )

    def run(self):
        """Run the GA and return the best chromosome and stats."""
        self.initialize_population()

        for gen in range(self.num_generations):
            fitnesses = evaluate_population(
                self.population,
                game=self.game,
                rng=self.rng,
                budget=self.budget,
                num_simulations=self.num_simulations_eval,
            )

            best_fitness = float(fitnesses.max())
            avg_fitness = float(fitnesses.mean())
            self.best_history.append(best_fitness)
            self.avg_history.append(avg_fitness)

            best_idx = int(fitnesses.argmax())
            best_chromosome = self.population[best_idx]

            print(f"Generation {gen + 1:03d} | Best: {best_fitness:.3f} | Avg: {avg_fitness:.3f}")

            # Elitism: carry best chromosome to next generation
            new_population: List[Chromosome] = [best_chromosome]

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(fitnesses, k=3)
                parent2 = self.tournament_selection(fitnesses, k=3)

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1 = Chromosome(parent1.genes.copy())
                    child2 = Chromosome(parent2.genes.copy())

                self.mutate(child1)
                self.mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        # Final evaluation
        final_fitnesses = evaluate_population(
            self.population,
            game=self.game,
            rng=self.rng,
            budget=self.budget,
            num_simulations=self.num_simulations_eval,
        )
        best_idx = int(final_fitnesses.argmax())
        best_chromosome = self.population[best_idx]
        best_fitness = float(final_fitnesses[best_idx])

        print(f"\n✅ Finished training. Best fitness: {best_fitness:.3f}")
        return best_chromosome, best_fitness, self.best_history, self.avg_history
