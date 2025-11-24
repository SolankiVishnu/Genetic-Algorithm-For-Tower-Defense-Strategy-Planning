# Genetic Algorithm for Tower Defense Strategy Planning

This project uses a Genetic Algorithm (GA) to automatically evolve strategies for a simplified tower defense game. A strategy is represented as a layout of towers along a 1D path, and the GA optimizes tower placement under a fixed budget to maximize enemy kills and minimize escapes.

## Game Overview

- Enemies move along a linear path of fixed length.
- There are several fixed tower slots along the path.
- Each gene in the chromosome selects which tower type (or empty) to place in a slot.
- Towers shoot enemies in range every tick; enemies move one step per tick.
- If an enemy reaches the end of the path, the player loses a life.

## Genetic Algorithm

- **Chromosome**: integer array, one gene per tower slot (0 = empty, 1..N = tower types).
- **Fitness**:
  - High reward for total enemies killed
  - Penalty for enemies that escape
  - Bonus for remaining lives
  - Layouts exceeding the budget are heavily penalized
- **Operators**:
  - Tournament selection
  - Single-point crossover
  - Random mutation of tower types
  - Elitism (keep best solution each generation)

## How to Run

```bash
pip install -r requirements.txt
python -m src.main
