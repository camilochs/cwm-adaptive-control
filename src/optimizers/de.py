"""Differential Evolution optimizer with adaptive operators."""

from typing import Callable, Optional, Tuple
import numpy as np

from .base import BaseOptimizer
from ..state import OptState
from ..actions import Action, ActionType


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution optimizer with configurable strategies."""

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        max_evaluations: Optional[int] = None,
        F: float = 0.5,  # Mutation factor
        CR: float = 0.9,  # Crossover rate
        strategy: str = "rand/1/bin",
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
        self.F = F
        self.CR = CR
        self.strategy = strategy

        # Adaptive parameters (can be modified by actions)
        self._current_F = F
        self._current_CR = CR
        self._current_strategy = strategy
        self._mutation_rate = 0.0  # Additional mutation
        self._mutation_sigma = 0.1

    def _select_strategy_indices(self, current_idx: int) -> Tuple[int, ...]:
        """Select indices for mutation based on strategy."""
        pop_size = len(self.population)
        indices = list(range(pop_size))
        indices.remove(current_idx)

        # Check for best/pbest strategies (pbest is a variant that uses p-best individuals)
        if "best" in self._current_strategy or "pbest" in self._current_strategy:
            # Include best individual
            r = [self.best_idx]
            indices = [i for i in indices if i != self.best_idx]
            r.extend(np.random.choice(indices, size=2, replace=False))
            return tuple(r)
        else:
            # rand strategy
            return tuple(np.random.choice(indices, size=3, replace=False))

    def _mutate(self, idx: int) -> np.ndarray:
        """Create mutant vector using DE mutation."""
        r1, r2, r3 = self._select_strategy_indices(idx)

        # Check for current-to-best or current-to-pbest strategies
        if "current-to-best" in self._current_strategy or "current-to-pbest" in self._current_strategy:
            # DE/current-to-best/1 or DE/current-to-pbest/1
            mutant = (
                self.population[idx] +
                self._current_F * (self.population[self.best_idx] - self.population[idx]) +
                self._current_F * (self.population[r2] - self.population[r3])
            )
        elif "best" in self._current_strategy or "pbest" in self._current_strategy:
            # DE/best/1 or DE/pbest/1
            mutant = (
                self.population[self.best_idx] +
                self._current_F * (self.population[r2] - self.population[r3])
            )
        else:
            # DE/rand/1
            mutant = (
                self.population[r1] +
                self._current_F * (self.population[r2] - self.population[r3])
            )

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform binomial crossover."""
        dim = len(target)
        trial = target.copy()

        # Ensure at least one dimension from mutant
        j_rand = np.random.randint(dim)

        for j in range(dim):
            if np.random.random() < self._current_CR or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def _apply_additional_mutation(
        self,
        individual: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Apply additional mutation if configured."""
        if self._mutation_rate <= 0:
            return individual

        mutated = individual.copy()
        dim = len(individual)

        for j in range(dim):
            if np.random.random() < self._mutation_rate:
                mutated[j] += np.random.normal(0, self._mutation_sigma)

        # Clip to bounds
        lower, upper = bounds
        return np.clip(mutated, lower, upper)

    def step(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        action: Optional[Action] = None,
    ) -> OptState:
        """Perform one generation of DE."""
        if self.population is None:
            self.initialize_population(objective, bounds)

        # Apply action if provided
        if action is not None:
            self.apply_action(action, objective, bounds)

        lower, upper = bounds
        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        for i in range(self.population_size):
            # Mutation
            mutant = self._mutate(i)

            # Crossover
            trial = self._crossover(self.population[i], mutant)

            # Additional mutation
            trial = self._apply_additional_mutation(trial, bounds)

            # Boundary handling
            trial = np.clip(trial, lower, upper)

            # Selection
            trial_fitness = objective(trial)
            self.evaluations += 1

            if trial_fitness <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

        # Update population
        self.population = new_population
        self.fitness = new_fitness
        self.best_idx = int(np.argmin(self.fitness))
        self.generation += 1

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
        """Apply an action to modify DE behavior."""
        params = action.params_dict

        if action.action_type == ActionType.MUTATION:
            self._mutation_rate = params.get("rate", 0.0)
            self._mutation_sigma = params.get("sigma", 0.1)

        elif action.action_type == ActionType.CROSSOVER:
            # Modify DE parameters based on crossover action
            if "F" in params:
                self._current_F = params["F"]
            if "CR" in params:
                self._current_CR = params["CR"]

            # Strategy switching - check params first, then action name
            strategy = params.get("strategy", None)
            if strategy:
                # Use strategy from params (e.g., "current-to-pbest/1", "rand/1")
                self._current_strategy = strategy
            elif action.name == "de_rand_1" or "rand" in action.name.lower():
                self._current_strategy = "rand/1/bin"
            elif action.name == "de_best_1" or "best" in action.name.lower():
                self._current_strategy = "best/1/bin"
            elif action.name == "de_current_to_best" or "pbest" in action.name.lower():
                self._current_strategy = "current-to-best/1/bin"

        elif action.action_type == ActionType.SELECTION:
            # DE uses greedy selection, but we can adjust tournament for diversity
            pass

        elif action.action_type == ActionType.RESTART:
            fraction = params.get("fraction", 0.3)
            self._partial_restart(fraction, bounds)

    def _partial_restart(
        self,
        fraction: float,
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Restart a fraction of the population."""
        n_restart = int(self.population_size * fraction)
        if n_restart == 0:
            return

        lower, upper = bounds
        dim = self.population.shape[1]

        # Keep best individuals
        sorted_indices = np.argsort(self.fitness)
        n_keep = self.population_size - n_restart

        # Replace worst individuals with random ones
        for i in range(n_keep, self.population_size):
            idx = sorted_indices[i]
            self.population[idx] = np.random.uniform(lower, upper, size=dim)
            # Note: fitness will be evaluated in next generation

        # Mark fitness as needing re-evaluation (set to inf)
        for i in range(n_keep, self.population_size):
            idx = sorted_indices[i]
            self.fitness[idx] = float('inf')

    def reset_adaptive_parameters(self):
        """Reset adaptive parameters to defaults."""
        self._current_F = self.F
        self._current_CR = self.CR
        self._current_strategy = self.strategy
        self._mutation_rate = 0.0
        self._mutation_sigma = 0.1
