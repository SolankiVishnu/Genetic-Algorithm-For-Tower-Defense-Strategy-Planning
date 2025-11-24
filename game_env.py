# src/game_env.py

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class Enemy:
    position: int
    hp: float
    alive: bool = True


class TowerDefenseGame:
    """
    Simplified 1D tower defense environment:

    - Enemies move along a path of length `path_length`.
    - There are fixed tower slots along the path.
    - Chromosome encodes which tower type is placed in each slot.
    - Towers shoot each tick, enemies move one step forward each tick.
    """

    def __init__(
        self,
        path_length: int = 20,
        num_slots: int = 8,
        lives: int = 10,
        enemy_hp_base: float = 10.0,
        enemy_hp_scale_per_wave: float = 2.0,
    ):
        self.path_length = path_length
        self.num_slots = num_slots
        self.lives = lives
        self.enemy_hp_base = enemy_hp_base
        self.enemy_hp_scale_per_wave = enemy_hp_scale_per_wave

        # Evenly space tower slots along the path
        self.tower_positions = np.linspace(2, path_length - 3, num_slots, dtype=int)

        # Define tower types (0 = empty)
        self.tower_types: Dict[int, Dict[str, Any]] = {
            0: {"name": "empty", "cost": 0, "damage": 0.0, "range": 0},
            1: {"name": "arrow", "cost": 10, "damage": 3.0, "range": 3},
            2: {"name": "cannon", "cost": 15, "damage": 5.0, "range": 2},
            3: {"name": "sniper", "cost": 20, "damage": 7.0, "range": 5},
        }

        self.num_tower_types = len(self.tower_types) - 1  # exclude type 0 (empty)

    def get_layout_cost(self, layout: np.ndarray) -> float:
        """Sum of tower costs for the given layout."""
        total_cost = 0.0
        for slot_idx, tower_type in enumerate(layout):
            tower_info = self.tower_types[int(tower_type)]
            total_cost += tower_info["cost"]
        return total_cost

    def _spawn_wave(
        self, rng: np.random.Generator, wave_index: int, enemies_per_wave: int
    ) -> List[Enemy]:
        """Create initial list of enemies for a wave."""
        hp = self.enemy_hp_base + wave_index * self.enemy_hp_scale_per_wave
        enemies = []
        for _ in range(enemies_per_wave):
            enemies.append(Enemy(position=0, hp=hp, alive=True))
        return enemies

    def _step_towers(self, layout: np.ndarray, enemies: List[Enemy]):
        """All towers shoot at enemies in range."""
        for slot_idx, tower_type in enumerate(layout):
            tower_type = int(tower_type)
            if tower_type == 0:
                continue  # empty slot

            tower_info = self.tower_types[tower_type]
            tower_pos = int(self.tower_positions[slot_idx])
            damage = tower_info["damage"]
            rng_range = tower_info["range"]

            # Find enemies in range
            in_range_indices = [
                i
                for i, e in enumerate(enemies)
                if e.alive and abs(e.position - tower_pos) <= rng_range
            ]
            if not in_range_indices:
                continue

            # Target enemy closest to the exit (max position)
            target_idx = max(in_range_indices, key=lambda i: enemies[i].position)
            target = enemies[target_idx]
            target.hp -= damage
            if target.hp <= 0:
                target.alive = False

    def _step_enemies(self, enemies: List[Enemy], lives: int):
        """Move enemies forward and check escapes."""
        escapes = 0
        for e in enemies:
            if not e.alive:
                continue
            e.position += 1
            if e.position >= self.path_length:
                e.alive = False
                escapes += 1
                lives -= 1
        return escapes, lives

    def simulate_layout(
        self,
        layout: np.ndarray,
        rng: np.random.Generator,
        num_waves: int = 5,
        enemies_per_wave: int = 10,
        max_steps_per_wave: int = 50,
    ) -> Dict[str, float]:
        """
        Simulate multiple waves for a given tower layout.

        Returns a dict with:
        - total_kills
        - total_escapes
        - lives_left
        """
        lives = self.lives
        total_kills = 0
        total_escapes = 0

        for wave in range(num_waves):
            enemies = self._spawn_wave(rng, wave_index=wave, enemies_per_wave=enemies_per_wave)
            steps = 0

            while steps < max_steps_per_wave and any(e.alive for e in enemies) and lives > 0:
                # Towers attack
                self._step_towers(layout, enemies)

                # Count kills (enemies that died this step)
                for e in enemies:
                    if not e.alive and e.hp <= 0:
                        # We'll count kills after we know all that escaped
                        pass

                # Enemies move
                escapes, lives = self._step_enemies(enemies, lives)
                total_escapes += escapes

                # Count kills (enemies that are dead and not escaped)
                kills_this_step = sum(1 for e in enemies if not e.alive and e.hp <= 0)
                total_kills += kills_this_step

                steps += 1

            if lives <= 0:
                break

        return {
            "total_kills": float(total_kills),
            "total_escapes": float(total_escapes),
            "lives_left": float(lives),
        }
