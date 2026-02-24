"""LeadingOnes problem and (1+1)-RLS_k optimizer.

LeadingOnes counts the number of uninterrupted leading 1s in a bitstring.
The (1+1)-RLS_k flips exactly k randomly chosen bits per step and accepts
if the offspring is at least as good as the parent (non-strict selection).

Theoretical optimal flip count: k*(i) = floor(n / (i + 1))
where i is the current LeadingOnes fitness and n the bitstring length.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


class LeadingOnesProblem:
    """LeadingOnes fitness function."""

    def __init__(self, n: int = 50):
        self.n = n

    def fitness(self, x: np.ndarray) -> int:
        """Count uninterrupted leading ones."""
        count = 0
        for bit in x:
            if bit == 1:
                count += 1
            else:
                break
        return count

    def random_bitstring(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate a uniformly random bitstring."""
        if rng is None:
            return np.random.randint(0, 2, self.n)
        return rng.randint(0, 2, self.n)


class RLSOptimizer:
    """(1+1)-RLS with configurable flip count k.

    Each step flips exactly k distinct random positions. The offspring
    replaces the parent iff its fitness is >= parent (non-strict).
    """

    def __init__(self, problem: LeadingOnesProblem, seed: Optional[int] = None):
        self.problem = problem
        self.n = problem.n
        self.rng = np.random.RandomState(seed)
        self.x: Optional[np.ndarray] = None
        self.fitness_val: int = 0
        self.step_count: int = 0
        self.history: List[Tuple[int, int, int]] = []  # (step, fitness, k_used)

    def initialize(self):
        """Create a random initial bitstring and evaluate it."""
        self.x = self.problem.random_bitstring(self.rng)
        self.fitness_val = self.problem.fitness(self.x)
        self.step_count = 0
        self.history = []

    def step(self, k: int) -> Tuple[int, bool]:
        """Flip exactly k random bits. Returns (new_fitness, improved)."""
        k = max(1, min(k, self.n))  # clamp to [1, n]
        positions = self.rng.choice(self.n, k, replace=False)
        y = self.x.copy()
        y[positions] = 1 - y[positions]
        new_fitness = self.problem.fitness(y)
        improved = new_fitness >= self.fitness_val  # non-strict selection
        if improved:
            self.x = y
            self.fitness_val = new_fitness
        self.step_count += 1
        self.history.append((self.step_count, self.fitness_val, k))
        return self.fitness_val, improved

    def is_optimal(self) -> bool:
        """True when all bits are leading ones."""
        return self.fitness_val == self.n

    def get_state(self) -> dict:
        """Return current state as a dict (for trajectory recording)."""
        return {
            "fitness": self.fitness_val,
            "n": self.n,
            "step": self.step_count,
            "normalized_fitness": self.fitness_val / self.n,
        }

    def copy(self) -> "RLSOptimizer":
        """Deep copy for forking experiments."""
        clone = RLSOptimizer(self.problem, seed=None)
        clone.rng = np.random.RandomState(self.rng.randint(0, 2**31))
        clone.x = self.x.copy() if self.x is not None else None
        clone.fitness_val = self.fitness_val
        clone.step_count = self.step_count
        clone.history = list(self.history)
        return clone


def optimal_k(i: int, n: int) -> int:
    """Theoretical optimal flip count: k*(i) = floor(n / (i + 1)).

    Clamped to [1, n].
    """
    return max(1, min(n, n // (i + 1)))


def run_episode(
    n: int,
    policy,
    budget: int,
    seed: Optional[int] = None,
) -> dict:
    """Run a single RLS episode with the given policy.

    Args:
        n: bitstring length
        policy: callable(fitness, n, step) -> k
        budget: maximum number of steps
        seed: random seed

    Returns:
        dict with episode data (transitions, total_steps, converged, etc.)
    """
    problem = LeadingOnesProblem(n)
    rls = RLSOptimizer(problem, seed=seed)
    rls.initialize()

    transitions = []
    while rls.step_count < budget and not rls.is_optimal():
        state = rls.get_state()
        k = policy(rls.fitness_val, n, rls.step_count)
        k = max(1, min(n, int(k)))
        new_fitness, improved = rls.step(k)
        next_state = rls.get_state()

        transitions.append({
            "state": state,
            "action": {"k": k, "name": f"flip_{k}"},
            "next_state": next_state,
            "reward": -1,  # each step costs 1
            "improved": improved,
        })

    return {
        "transitions": transitions,
        "total_steps": rls.step_count,
        "converged": rls.is_optimal(),
        "final_fitness": rls.fitness_val,
        "n": n,
    }
