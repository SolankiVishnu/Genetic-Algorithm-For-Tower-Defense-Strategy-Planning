# src/evaluation.py

from typing import List
import numpy as np

from .game_env import TowerDefenseGame
from .strategy_encoding import Chromosome, decode_layout


def evaluate_chromosome(
    chromosome: Chromosome,
    game: TowerDefenseGame,
    rng: np.random.Generator,
    budget: float,
    num_simulations: int = 3,
) -> float:
    """
    Evaluate one chromosome/layout.

    - If layout cost exceeds budget -> very low fitness.
    - Otherwise, simulate several runs and compute fitness based on:
      kills, escapes, and lives left.
    """
    layout = decode_layout(chromosome)
    cost = game.get_layout_cost(layout)

    if cost > budget:
        # Penalize layouts that exceed budget
        return 0.0

    total_fitness = 0.0

    for _ in range(num_simulations):
        stats = game.simulate_layout(
            layout=layout,
            rng=rng,
            num_waves=5,
            enemies_per_wave=10,
        )
        kills = stats["total_kills"]
        escapes = stats["total_escapes"]
        lives_left = stats["lives_left"]

        # Example fitness function:
        # reward kills, penalize escapes, reward remaining lives.
        fitness = kills - 2.0 * escapes + 0.5 * lives_left
        total_fitness += max(fitness, 0.0)

    return total_fitness / num_simulations


def evaluate_population(
    population: List[Chromosome],
    game: TowerDefenseGame,
    rng: np.random.Generator,
    budget: float,
    num_simulations: int = 3,
) -> np.ndarray:
    """Evaluate all chromosomes and return fitness values."""
    fitnesses = np.zeros(len(population), dtype=float)
    for i, chrom in enumerate(population):
        fitnesses[i] = evaluate_chromosome(
            chrom,
            game=game,
            rng=rng,
            budget=budget,
            num_simulations=num_simulations,
        )
    return fitnesses
