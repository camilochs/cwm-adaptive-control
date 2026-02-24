"""SHADE - Success-History based Adaptive Differential Evolution.

Implementation of SHADE from:
Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation
for differential evolution. In 2013 IEEE congress on evolutionary computation
(pp. 71-78). IEEE.

Key features:
- Historical memory of successful F and CR values
- Current-to-pbest/1 mutation strategy
- External archive of inferior solutions
- Optional linear population size reduction (L-SHADE variant)
"""

from typing import Callable, List, Optional, Tuple
import numpy as np

from .base import BaseOptimizer, OptimizationResult
from ..state import OptState
from ..actions import Action, ActionType


class SHADE(BaseOptimizer):
    """Success-History based Adaptive Differential Evolution (SHADE).

    Uses historical memory of successful parameters and current-to-pbest/1
    mutation strategy for adaptive parameter control.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        max_evaluations: Optional[int] = None,
        # Historical memory size
        H: int = 100,
        # pbest parameter (top p% for mutation)
        p_best_rate: float = 0.1,
        # Archive rate (max archive size = archive_rate * pop_size)
        archive_rate: float = 1.0,
        # Initial parameter values for memory
        F_init: float = 0.5,
        CR_init: float = 0.5,
        # L-SHADE: linear population size reduction
        use_lpsr: bool = False,
        min_pop_size: int = 4,
        tol: float = 1e-8,
        seed: Optional[int] = None,
        record_trajectory: bool = True,
    ):
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            tol=tol,
            seed=seed,
            record_trajectory=record_trajectory,
        )

        self.H = H
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate
        self.F_init = F_init
        self.CR_init = CR_init
        self.use_lpsr = use_lpsr
        self.min_pop_size = min_pop_size
        self.initial_pop_size = population_size

        # Historical memory for F and CR
        self.M_F: Optional[np.ndarray] = None  # Memory for F
        self.M_CR: Optional[np.ndarray] = None  # Memory for CR
        self.k: int = 0  # Memory index

        # External archive
        self.archive: List[np.ndarray] = []
        self.max_archive_size = int(archive_rate * population_size)

    def initialize_population(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Initialize population and memory."""
        super().initialize_population(objective, bounds)

        # Initialize memory with initial values
        self.M_F = np.full(self.H, self.F_init)
        self.M_CR = np.full(self.H, self.CR_init)
        self.k = 0

        # Reset archive
        self.archive = []
        self.max_archive_size = int(self.archive_rate * self.population_size)

    def _generate_F(self, M_F_r: float) -> float:
        """Generate F value from Cauchy distribution centered at M_F_r."""
        # Guard against NaN in memory (can happen after restart with inf fitness)
        if not np.isfinite(M_F_r):
            M_F_r = self.F_init
        for _ in range(1000):  # Safety limit to prevent infinite loop
            F = np.random.standard_cauchy() * 0.1 + M_F_r
            if F >= 0:
                return min(F, 1.0)
        return M_F_r  # Fallback

    def _generate_CR(self, M_CR_r: float) -> float:
        """Generate CR value from Normal distribution centered at M_CR_r."""
        CR = np.random.normal(M_CR_r, 0.1)
        return np.clip(CR, 0.0, 1.0)

    def _select_pbest(self) -> int:
        """Select a random index from the top p% of the population."""
        p = max(2, int(self.p_best_rate * self.population_size))
        sorted_indices = np.argsort(self.fitness)[:p]
        return np.random.choice(sorted_indices)

    def _mutate(self, idx: int, F: float) -> np.ndarray:
        """Current-to-pbest/1 mutation with archive."""
        # Select pbest individual
        pbest_idx = self._select_pbest()

        # Select r1 different from idx and pbest_idx
        candidates = [i for i in range(self.population_size) if i not in [idx, pbest_idx]]
        r1 = np.random.choice(candidates)

        # Select r2 from population + archive, different from idx, pbest_idx, r1
        combined = list(range(self.population_size)) + list(range(len(self.archive)))
        combined = [i for i in combined if i not in [idx, pbest_idx, r1]]

        if len(combined) == 0:
            # Fallback: just use population
            r2_candidates = [i for i in range(self.population_size) if i != idx]
            r2_idx = np.random.choice(r2_candidates)
            x_r2 = self.population[r2_idx]
        else:
            r2 = np.random.choice(combined)
            if r2 < self.population_size:
                x_r2 = self.population[r2]
            else:
                x_r2 = self.archive[r2 - self.population_size]

        # Current-to-pbest/1 mutation
        mutant = (
            self.population[idx] +
            F * (self.population[pbest_idx] - self.population[idx]) +
            F * (self.population[r1] - x_r2)
        )

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Binomial crossover."""
        dim = len(target)
        trial = target.copy()

        j_rand = np.random.randint(dim)

        for j in range(dim):
            if np.random.random() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def _update_memory(self, S_F: List[float], S_CR: List[float], delta_f: List[float]):
        """Update memory with successful F and CR values using weighted mean."""
        if len(S_F) == 0:
            return

        # Weights based on fitness improvement
        weights = np.array(delta_f, dtype=float)
        # Guard against inf/NaN from restarted individuals with fitness=inf
        finite_mask = np.isfinite(weights)
        if not np.any(finite_mask):
            return
        weights = np.where(finite_mask, weights, 0.0)
        w_sum = np.sum(weights)
        if w_sum == 0:
            return
        weights = weights / w_sum

        # Weighted Lehmer mean for F
        S_F_arr = np.array(S_F)
        mean_F = np.sum(weights * S_F_arr ** 2) / np.sum(weights * S_F_arr)

        # Weighted arithmetic mean for CR
        S_CR_arr = np.array(S_CR)
        mean_CR = np.sum(weights * S_CR_arr)

        # Update memory at position k
        self.M_F[self.k] = mean_F
        self.M_CR[self.k] = mean_CR

        # Increment k (circular)
        self.k = (self.k + 1) % self.H

    def _resize_population(self):
        """Linear population size reduction (L-SHADE)."""
        if not self.use_lpsr:
            return

        # Calculate new population size
        new_size = int(
            self.initial_pop_size +
            (self.min_pop_size - self.initial_pop_size) *
            (self.evaluations / (self.max_evaluations or self.max_generations * self.population_size))
        )
        new_size = max(new_size, self.min_pop_size)

        if new_size < self.population_size:
            # Keep the best individuals
            sorted_indices = np.argsort(self.fitness)[:new_size]
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            self.population_size = new_size
            self.best_idx = int(np.argmin(self.fitness))

            # Resize archive
            self.max_archive_size = int(self.archive_rate * self.population_size)
            if len(self.archive) > self.max_archive_size:
                self.archive = self.archive[:self.max_archive_size]

    def step(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        action: Optional[Action] = None,
    ) -> OptState:
        """Perform one generation of SHADE."""
        if self.population is None:
            self.initialize_population(objective, bounds)

        # Apply action if provided
        if action is not None:
            self.apply_action(action, objective, bounds)

        lower, upper = bounds

        # Generate F and CR for each individual
        F_values = np.zeros(self.population_size)
        CR_values = np.zeros(self.population_size)

        for i in range(self.population_size):
            # Select random memory index
            r = np.random.randint(self.H)
            F_values[i] = self._generate_F(self.M_F[r])
            CR_values[i] = self._generate_CR(self.M_CR[r])

        # Tracking successful parameters
        S_F = []
        S_CR = []
        delta_f = []

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        for i in range(self.population_size):
            # Mutation
            mutant = self._mutate(i, F_values[i])

            # Crossover
            trial = self._crossover(self.population[i], mutant, CR_values[i])

            # Boundary handling
            trial = np.clip(trial, lower, upper)

            # Selection
            trial_fitness = objective(trial)
            self.evaluations += 1

            if trial_fitness < self.fitness[i]:
                # Add parent to archive
                if len(self.archive) < self.max_archive_size:
                    self.archive.append(self.population[i].copy())
                elif len(self.archive) > 0:
                    # Replace random archive member
                    replace_idx = np.random.randint(len(self.archive))
                    self.archive[replace_idx] = self.population[i].copy()

                # Record successful parameters
                S_F.append(F_values[i])
                S_CR.append(CR_values[i])
                delta_f.append(self.fitness[i] - trial_fitness)

                new_population[i] = trial
                new_fitness[i] = trial_fitness

            elif trial_fitness == self.fitness[i]:
                # Equal fitness: accept trial (for diversity)
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        # Update memory
        self._update_memory(S_F, S_CR, delta_f)

        # Update population
        self.population = new_population
        self.fitness = new_fitness
        self.best_idx = int(np.argmin(self.fitness))
        self.generation += 1

        # Population size reduction (L-SHADE)
        self._resize_population()

        # Update tracking
        best_fitness = float(self.fitness[self.best_idx])
        self.fitness_history.append(best_fitness)
        self._update_stagnation(best_fitness)

        # Record state
        self._record_state(action)

        return self.get_current_state()

    def apply_action(
        self,
        action: Action,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Apply an action to modify SHADE behavior."""
        params = action.params_dict

        if action.action_type == ActionType.MUTATION:
            # Adjust memory initialization
            rate = params.get("rate", 0.0)
            if rate > 0:
                # Increase F values in memory
                self.M_F = np.clip(self.M_F + 0.1, 0.1, 1.0)

        elif action.action_type == ActionType.CROSSOVER:
            if "CR" in params:
                # Adjust CR memory
                self.M_CR = np.clip(self.M_CR + 0.1, 0.0, 1.0)

        elif action.action_type == ActionType.RESTART:
            fraction = params.get("fraction", 0.3)
            self._partial_restart(fraction, bounds)

    def _partial_restart(
        self,
        fraction: float,
        bounds: Tuple[np.ndarray, np.ndarray],
        objective: Callable[[np.ndarray], float] = None,
    ):
        """Restart a fraction of the population.

        If objective is provided, re-evaluates new individuals immediately.
        Otherwise uses a large finite sentinel value to avoid inf in memory updates.
        """
        n_restart = int(self.population_size * fraction)
        if n_restart == 0:
            return

        lower, upper = bounds
        dim = self.population.shape[1]

        sorted_indices = np.argsort(self.fitness)
        n_keep = self.population_size - n_restart

        # Use large finite sentinel instead of inf to prevent NaN in memory
        sentinel = np.max(self.fitness[np.isfinite(self.fitness)]) * 10 if np.any(np.isfinite(self.fitness)) else 1e10

        for i in range(n_keep, self.population_size):
            idx = sorted_indices[i]
            self.population[idx] = np.random.uniform(lower, upper, size=dim)
            if objective is not None:
                self.fitness[idx] = objective(self.population[idx])
                self.evaluations += 1
            else:
                self.fitness[idx] = sentinel

    def get_memory_statistics(self) -> dict:
        """Get statistics about current memory state."""
        return {
            "M_F_mean": float(np.mean(self.M_F)),
            "M_F_std": float(np.std(self.M_F)),
            "M_CR_mean": float(np.mean(self.M_CR)),
            "M_CR_std": float(np.std(self.M_CR)),
            "archive_size": len(self.archive),
            "current_pop_size": self.population_size,
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        callback: Optional[Callable[[OptState], Optional[Action]]] = None,
    ) -> OptimizationResult:
        """Run SHADE optimization."""
        self.initialize_population(objective, bounds)

        converged = False
        memory_history = []

        while self.generation < self.max_generations:
            if self.max_evaluations and self.evaluations >= self.max_evaluations:
                break

            action = None
            if callback is not None:
                current_state = self.get_current_state()
                action = callback(current_state)

            state = self.step(objective, bounds, action)

            # Track memory evolution
            memory_history.append(self.get_memory_statistics())

            if state.best_fitness <= self.tol:
                converged = True
                break

            if self._stagnation_count > self.max_generations // 2:
                break

        return OptimizationResult(
            best_solution=self.population[self.best_idx].copy(),
            best_fitness=float(self.fitness[self.best_idx]),
            history=self.state_history,
            trajectory=self.trajectory,
            evaluations=self.evaluations,
            generations=self.generation,
            converged=converged,
            metadata={
                "algorithm": "SHADE" if not self.use_lpsr else "L-SHADE",
                "memory_history": memory_history,
                "final_memory": self.get_memory_statistics(),
            },
        )


class LSHADE(SHADE):
    """L-SHADE: SHADE with Linear Population Size Reduction.

    Convenience class that enables LPSR by default.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        max_evaluations: Optional[int] = None,
        H: int = 100,
        p_best_rate: float = 0.1,
        archive_rate: float = 2.0,  # L-SHADE typically uses larger archive
        F_init: float = 0.5,
        CR_init: float = 0.5,
        min_pop_size: int = 4,
        tol: float = 1e-8,
        seed: Optional[int] = None,
        record_trajectory: bool = True,
    ):
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            H=H,
            p_best_rate=p_best_rate,
            archive_rate=archive_rate,
            F_init=F_init,
            CR_init=CR_init,
            use_lpsr=True,
            min_pop_size=min_pop_size,
            tol=tol,
            seed=seed,
            record_trajectory=record_trajectory,
        )
