# src/strategy_encoding.py

from dataclasses import dataclass
import numpy as np

from .game_env import TowerDefenseGame


@dataclass
class Chromosome:
    """
    Represents a tower defense strategy:
    genes[i] = tower type at slot i (0 = empty, 1..N = different tower types).
    """
    genes: np.ndarray  # shape: (num_slots,)


def random_chromosome(game: TowerDefenseGame, rng: np.random.Generator) -> Chromosome:
    """
    Create a random layout:
    - Each slot gets a random tower type in [0, num_tower_types].
    """
    genes = rng.integers(
        low=0,
        high=game.num_tower_types + 1,
        size=(game.num_slots,),
        endpoint=False,
    )
    return Chromosome(genes=genes)


def decode_layout(chromosome: Chromosome) -> np.ndarray:
    """
    For this environment, the chromosome genes are already a layout.
    This function just returns the genes as a layout array.
    """
    return chromosome.genes.copy()
