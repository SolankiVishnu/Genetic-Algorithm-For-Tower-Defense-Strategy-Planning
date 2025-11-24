# src/main.py

import os
import matplotlib.pyplot as plt
import yaml
import numpy as np

from .game_env import TowerDefenseGame
from .genetic_algorithm import GeneticAlgorithm
from .strategy_encoding import decode_layout


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_history(best_history, avg_history, output_path: str = "results/fitness_curve.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(best_history, label="Best Fitness")
    plt.plot(avg_history, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm - Fitness over Generations (Tower Defense)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def pretty_print_layout(game: TowerDefenseGame, layout: np.ndarray):
    print("\n📍 Best Layout (slot -> tower type):")
    for i, t in enumerate(layout):
        t = int(t)
        info = game.tower_types[t]
        print(
            f"  Slot {i}: {info['name']} "
            f"(cost={info['cost']}, damage={info['damage']}, range={info['range']})"
        )
    print(f"\nTotal Cost: {game.get_layout_cost(layout)}")


def main():
    config = load_config()

    game_cfg = config["game"]
    ga_cfg = config["genetic_algorithm"]

    game = TowerDefenseGame(
        path_length=game_cfg.get("path_length", 20),
        num_slots=game_cfg.get("num_slots", 8),
        lives=game_cfg.get("lives", 10),
        enemy_hp_base=game_cfg.get("enemy_hp_base", 10.0),
        enemy_hp_scale_per_wave=game_cfg.get("enemy_hp_scale_per_wave", 2.0),
    )

    ga = GeneticAlgorithm(
        game=game,
        population_size=ga_cfg.get("population_size", 40),
        num_generations=ga_cfg.get("num_generations", 50),
        crossover_rate=ga_cfg.get("crossover_rate", 0.8),
        mutation_rate=ga_cfg.get("mutation_rate", 0.1),
        budget=ga_cfg.get("budget", 80.0),
        num_simulations_eval=ga_cfg.get("num_simulations_eval", 3),
        seed=ga_cfg.get("seed", 42),
    )

    best_chromosome, best_fitness, best_history, avg_history = ga.run()

    layout = decode_layout(best_chromosome)
    pretty_print_layout(game, layout)

    # Plot fitness curve
    plot_history(best_history, avg_history, output_path="results/fitness_curve.png")
    print("\n📈 Saved fitness curve to results/fitness_curve.png")

    # Optionally, run one more simulation to show stats
    rng = np.random.default_rng(123)
    stats = game.simulate_layout(layout, rng, num_waves=5, enemies_per_wave=10)
    print("\n📊 Simulation stats of best layout:")
    print(stats)


if __name__ == "__main__":
    main()
