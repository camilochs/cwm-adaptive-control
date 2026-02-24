"""NK-Landscape problem and (1+1)-RLS_k optimizer.

The NK-Landscape is a tunably rugged fitness function where each of n bits
contributes to fitness based on its own value AND the values of K neighbouring
bits.  The fitness is the average of n contributions:

    f(x) = (1/n) * sum_{i=0}^{n-1} contribution(x_i, x_{neighbours(i)})

With K=0 the landscape is smooth (like OneMax); as K increases, the landscape
becomes more rugged with more local optima.

Key property: **no closed-form optimal adaptive policy exists**.
This makes NK-Landscape the hardest benchmark for CWM—the world model must
infer transition patterns purely from trajectory data.

Instance reproducibility: interaction tables and contribution tables are
generated from a fixed seed and can be saved/loaded as JSON for exact
reproducibility across runs.
"""

import json
import numpy as np
from typing import List, Tuple, Optional


class NKLandscapeProblem:
    """NK-Landscape fitness function.

    Each bit i has K neighbours (circular adjacency by default).
    The contribution of bit i depends on the values of bit i and its K
    neighbours, looked up in a random table generated at construction time.

    Fitness = (1/n) * sum of all n contributions, scaled to [0, n] for
    consistency with other benchmarks (integer-like range).
    """

    def __init__(self, n: int = 50, K: int = 2, seed: int = 42):
        self.n = n
        self.K = K
        self.seed = seed
        rng = np.random.RandomState(seed)

        # Neighbours: circular adjacency (bit i interacts with i+1, i+2, ..., i+K mod n)
        self.neighbours = []
        for i in range(n):
            nbrs = [(i + j + 1) % n for j in range(K)]
            self.neighbours.append(nbrs)

        # Contribution table: for each bit i, a random value for each
        # combination of (bit_i, neighbour_values).  There are 2^(K+1)
        # combinations per bit.
        n_combos = 2 ** (K + 1)
        self.contributions = rng.random((n, n_combos))

    def _contribution_index(self, x: np.ndarray, i: int) -> int:
        """Compute the lookup index for bit i given bitstring x."""
        bits = [x[i]] + [x[j] for j in self.neighbours[i]]
        idx = 0
        for b in bits:
            idx = idx * 2 + int(b)
        return idx

    def fitness(self, x: np.ndarray) -> float:
        """Compute NK-Landscape fitness.

        Returns a float in [0, n] (sum of n contributions, each in [0, 1]).
        """
        total = 0.0
        for i in range(self.n):
            idx = self._contribution_index(x, i)
            total += self.contributions[i, idx]
        return total

    def max_fitness(self) -> float:
        """Upper bound on fitness (sum of max contributions per bit)."""
        return float(np.sum(np.max(self.contributions, axis=1)))

    def random_bitstring(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate a uniformly random bitstring."""
        if rng is None:
            return np.random.randint(0, 2, self.n)
        return rng.randint(0, 2, self.n)

    def save_instance(self, path: str):
        """Save the NK instance to a JSON file for reproducibility."""
        data = {
            "n": self.n,
            "K": self.K,
            "seed": self.seed,
            "neighbours": self.neighbours,
            "contributions": self.contributions.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_instance(cls, path: str) -> "NKLandscapeProblem":
        """Load an NK instance from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        instance = cls.__new__(cls)
        instance.n = data["n"]
        instance.K = data["K"]
        instance.seed = data["seed"]
        instance.neighbours = data["neighbours"]
        instance.contributions = np.array(data["contributions"])
        return instance


class RLSOptimizer:
    """(1+1)-RLS with configurable flip count k for NK-Landscape.

    Each step flips exactly k distinct random positions. The offspring
    replaces the parent iff its fitness is >= parent (non-strict).
    """

    def __init__(self, problem: NKLandscapeProblem, seed: Optional[int] = None):
        self.problem = problem
        self.n = problem.n
        self.K = problem.K
        self.rng = np.random.RandomState(seed)
        self.x: Optional[np.ndarray] = None
        self.fitness_val: float = 0.0
        self.step_count: int = 0
        self.stagnation_counter: int = 0
        self.history: List[Tuple[int, float, int]] = []  # (step, fitness, k_used)

    def initialize(self):
        """Create a random initial bitstring and evaluate it."""
        self.x = self.problem.random_bitstring(self.rng)
        self.fitness_val = self.problem.fitness(self.x)
        self.step_count = 0
        self.stagnation_counter = 0
        self.history = []

    def step(self, k: int) -> Tuple[float, bool]:
        """Flip exactly k random bits. Returns (new_fitness, improved)."""
        k = max(1, min(k, self.n))  # clamp to [1, n]
        positions = self.rng.choice(self.n, k, replace=False)
        y = self.x.copy()
        y[positions] = 1 - y[positions]
        new_fitness = self.problem.fitness(y)
        improved = new_fitness >= self.fitness_val  # non-strict selection
        if improved:
            if new_fitness > self.fitness_val:
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            self.x = y
            self.fitness_val = new_fitness
        else:
            self.stagnation_counter += 1
        self.step_count += 1
        self.history.append((self.step_count, self.fitness_val, k))
        return self.fitness_val, improved

    def is_optimal(self) -> bool:
        """True when fitness equals the theoretical maximum.

        For NK-Landscape, the global optimum is generally unknown.
        We use the max_fitness upper bound; in practice, reaching it
        exactly is extremely unlikely for K > 0. Episodes terminate
        via budget, not optimality.
        """
        return self.fitness_val >= self.problem.max_fitness() - 1e-10

    def get_state(self) -> dict:
        """Return current state as a dict (for trajectory recording)."""
        max_fit = self.problem.max_fitness()
        return {
            "fitness": self.fitness_val,
            "n": self.n,
            "K": self.K,
            "step": self.step_count,
            "normalized_fitness": self.fitness_val / max_fit if max_fit > 0 else 0.0,
            "stagnation_counter": self.stagnation_counter,
        }

    def copy(self) -> "RLSOptimizer":
        """Deep copy for forking experiments."""
        clone = RLSOptimizer(self.problem, seed=None)
        clone.rng = np.random.RandomState(self.rng.randint(0, 2**31))
        clone.x = self.x.copy() if self.x is not None else None
        clone.fitness_val = self.fitness_val
        clone.step_count = self.step_count
        clone.stagnation_counter = self.stagnation_counter
        clone.history = list(self.history)
        return clone


def run_episode(
    n: int,
    K: int,
    policy,
    budget: int,
    seed: Optional[int] = None,
    instance_path: Optional[str] = None,
) -> dict:
    """Run a single RLS episode with the given policy on NK-Landscape.

    Args:
        n: bitstring length
        K: epistasis parameter
        policy: callable(fitness, n, K, step, stagnation_counter) -> k
        budget: maximum number of steps
        seed: random seed for the optimizer
        instance_path: path to a saved NK instance (for reproducibility)

    Returns:
        dict with episode data (transitions, total_steps, final_fitness, etc.)
    """
    if instance_path:
        problem = NKLandscapeProblem.load_instance(instance_path)
    else:
        problem = NKLandscapeProblem(n, K)

    rls = RLSOptimizer(problem, seed=seed)
    rls.initialize()

    transitions = []
    best_fitness = rls.fitness_val

    while rls.step_count < budget:
        state = rls.get_state()
        k = policy(rls.fitness_val, n, K, rls.step_count, rls.stagnation_counter)
        k = max(1, min(n, int(k)))
        new_fitness, improved = rls.step(k)
        next_state = rls.get_state()
        best_fitness = max(best_fitness, new_fitness)

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
        "converged": False,  # NK-Landscape: no known optimum to converge to
        "final_fitness": rls.fitness_val,
        "best_fitness": best_fitness,
        "n": n,
        "K": K,
    }
