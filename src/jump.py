"""Jump_k problem and (1+1)-RLS_k optimizer.

Jump_k(x) = { k + |x|_1,  if |x|_1 <= n-k  or  x = 1^n
            { n - |x|_1,   otherwise  (the VALLEY)

Three regions:
  Normal  (|x|_1 in [0, n-k]):  fitness = k + |x|_1  (like OneMax + offset k)
  Valley  (|x|_1 in (n-k, n)):  fitness DROPS — local optimum trap
  Optimum (x = 1^n):            fitness = k + n

The (1+1)-RLS_k flips exactly k randomly chosen bits per step and accepts
if the offspring is at least as good as the parent (non-strict selection).

Key challenge: the optimal adaptive policy is NOT known in closed form.
EA_alpha baselines reduce k when stuck at the valley edge — exactly wrong,
because you need LARGE k (= jump_k) to jump across the valley.
"""

import numpy as np
from math import comb
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


class JumpProblem:
    """Jump_k fitness function."""

    def __init__(self, n: int = 50, jump_k: int = 2):
        self.n = n
        self.jump_k = jump_k

    def fitness(self, x: np.ndarray) -> int:
        """Compute Jump_k fitness.

        Jump_k(x) = k + |x|_1  if |x|_1 <= n-k  or  x = 1^n
                   = n - |x|_1  otherwise (valley)
        """
        ones = int(np.sum(x))
        if ones <= self.n - self.jump_k or ones == self.n:
            return self.jump_k + ones
        else:
            # Valley region: fitness drops
            return self.n - ones

    def random_bitstring(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate a uniformly random bitstring."""
        if rng is None:
            return np.random.randint(0, 2, self.n)
        return rng.randint(0, 2, self.n)


class RLSOptimizer:
    """(1+1)-RLS with configurable flip count k for the Jump problem.

    Each step flips exactly k distinct random positions. The offspring
    replaces the parent iff its fitness is >= parent (non-strict).
    """

    def __init__(self, problem: JumpProblem, seed: Optional[int] = None):
        self.problem = problem
        self.n = problem.n
        self.jump_k = problem.jump_k
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
        """True when fitness equals n + jump_k (all ones)."""
        return self.fitness_val == self.n + self.jump_k

    def get_state(self) -> dict:
        """Return current state as a dict (for trajectory recording)."""
        return {
            "fitness": self.fitness_val,
            "n": self.n,
            "jump_k": self.jump_k,
            "step": self.step_count,
            "normalized_fitness": self.fitness_val / (self.n + self.jump_k),
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


def optimal_k_jump(fitness: int, n: int, jump_k: int) -> int:
    """Exact optimal k for the Jump_k problem.

    Uses hypergeometric model in the normal region. At the valley edge
    (ones = n - jump_k, fitness = n) the only useful move is flipping
    exactly jump_k bits — all of which must be zeros. Inside the valley,
    also return jump_k (try to jump to the optimum).

    Args:
        fitness: current Jump_k fitness value
        n: bitstring length
        jump_k: gap parameter of the Jump function

    Returns:
        optimal flip count k
    """
    # Determine the region from fitness
    # Normal: fitness = jump_k + ones, where ones in [0, n-jump_k]
    #   => fitness in [jump_k, n]
    # Valley: fitness = n - ones, where ones in (n-jump_k, n)
    #   => fitness in (0, jump_k)  (note: ones close to n gives small fitness)
    # Optimum: fitness = jump_k + n

    if fitness == n + jump_k:
        # Already at optimum
        return 1

    if fitness >= jump_k and fitness <= n:
        # Normal region: ones = fitness - jump_k
        ones = fitness - jump_k

        if ones == n - jump_k:
            # Valley edge: the ONLY way to improve is to flip exactly
            # jump_k bits, all of which must be zeros → reach all-ones.
            return jump_k

        # In normal region, use hypergeometric model (like OneMax)
        # but must also account for the valley penalty.
        best_k, best_gain = 1, 0.0
        for k in range(1, n + 1):
            gain = 0.0
            j_min = max(0, k - (n - ones))
            j_max = min(k, ones)
            denom = comb(n, k)
            for j in range(j_min, j_max + 1):
                prob = comb(ones, j) * comb(n - ones, k - j) / denom
                new_ones = ones + k - 2 * j
                new_fitness = _jump_fitness_from_ones(new_ones, n, jump_k)
                delta = new_fitness - fitness
                if delta >= 0:  # accepted (non-strict)
                    gain += delta * prob
            if gain > best_gain:
                best_gain = gain
                best_k = k
        return best_k

    # Valley region (fitness < jump_k and fitness > 0):
    # ones = n - fitness (from n - ones = fitness)
    # Best strategy: try to flip exactly jump_k zeros to reach optimum
    # (or flip enough to get back to normal region)
    return jump_k


def _jump_fitness_from_ones(ones: int, n: int, jump_k: int) -> int:
    """Compute Jump_k fitness given the number of ones."""
    if ones <= n - jump_k or ones == n:
        return jump_k + ones
    else:
        return n - ones


def precompute_optimal_k_table(n: int, jump_k: int) -> List[int]:
    """Cache optimal k for all fitness levels.

    Fitness levels range from 0 to n + jump_k.
    """
    table = []
    for fitness in range(n + jump_k + 1):
        table.append(optimal_k_jump(fitness, n, jump_k))
    return table


def approx_optimal_k_jump(fitness: int, n: int, jump_k: int) -> int:
    """Fast heuristic for the Jump_k problem.

    - Far from valley: k=1 (like OneMax in the easy region)
    - Near or at valley edge: k=jump_k (need to jump)
    - In valley: k=jump_k (try to escape)
    """
    if fitness == n + jump_k:
        return 1  # already optimal

    if fitness >= jump_k and fitness <= n:
        # Normal region: ones = fitness - jump_k
        ones = fitness - jump_k
        zeros = n - ones
        if zeros <= jump_k:
            # Near or at valley edge — need to jump
            return jump_k
        elif ones >= n // 2:
            return 1
        else:
            return max(1, n - ones)
    else:
        # Valley region
        return jump_k


def run_episode(
    n: int,
    jump_k: int,
    policy,
    budget: int,
    seed: Optional[int] = None,
) -> dict:
    """Run a single RLS episode with the given policy on Jump_k.

    Args:
        n: bitstring length
        jump_k: gap parameter
        policy: callable(fitness, n, jump_k, step) -> k
        budget: maximum number of steps
        seed: random seed

    Returns:
        dict with episode data (transitions, total_steps, converged, etc.)
    """
    problem = JumpProblem(n, jump_k)
    rls = RLSOptimizer(problem, seed=seed)
    rls.initialize()

    transitions = []
    while rls.step_count < budget and not rls.is_optimal():
        state = rls.get_state()
        k = policy(rls.fitness_val, n, jump_k, rls.step_count)
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
        "jump_k": jump_k,
    }
