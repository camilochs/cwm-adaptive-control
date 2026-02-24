"""Base optimizer class with trajectory recording."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..state import OptState, compute_diversity, compute_improvement_rate
from ..actions import Action


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    best_solution: np.ndarray
    best_fitness: float
    history: List[OptState]
    trajectory: List[Tuple[OptState, Optional[Action], OptState]]  # (state, action, next_state)
    evaluations: int
    generations: int
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Abstract base class for optimizers with trajectory recording."""

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        max_evaluations: Optional[int] = None,
        tol: float = 1e-8,
        seed: Optional[int] = None,
        record_trajectory: bool = True,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations
        self.tol = tol
        self.record_trajectory = record_trajectory

        if seed is not None:
            np.random.seed(seed)

        # State tracking
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.best_idx: int = 0
        self.generation: int = 0
        self.evaluations: int = 0

        # History
        self.state_history: List[OptState] = []
        self.trajectory: List[Tuple[OptState, Optional[Action], OptState]] = []
        self.fitness_history: List[float] = []

        # Stagnation tracking
        self._last_best_fitness: float = float('inf')
        self._stagnation_count: int = 0

    def get_current_state(self) -> OptState:
        """Compute and return the current optimization state."""
        if self.population is None or self.fitness is None:
            raise RuntimeError("Optimizer not initialized. Call optimize() first.")

        best_fitness = float(np.min(self.fitness))
        mean_fitness = float(np.mean(self.fitness))
        fitness_std = float(np.std(self.fitness))
        diversity = compute_diversity(self.population)
        improvement_rate = compute_improvement_rate(self.fitness_history)

        state = OptState(
            generation=self.generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            diversity=diversity,
            improvement_rate=improvement_rate,
            stagnation=self._stagnation_count,
            fitness_std=fitness_std,
            population_size=len(self.population),
            dimension=self.population.shape[1],
            evaluations=self.evaluations,
            fitness_history=self.fitness_history.copy(),
            diversity_history=[s.diversity for s in self.state_history[-5:]] if self.state_history else [],
        )

        return state

    def _update_stagnation(self, best_fitness: float):
        """Update stagnation counter."""
        if best_fitness < self._last_best_fitness - self.tol:
            self._stagnation_count = 0
            self._last_best_fitness = best_fitness
        else:
            self._stagnation_count += 1

    def _record_state(self, action: Optional[Action] = None):
        """Record current state and optionally the action taken."""
        state = self.get_current_state()
        self.state_history.append(state)

        # Record trajectory transition
        if len(self.state_history) >= 2 and self.record_trajectory:
            prev_state = self.state_history[-2]
            self.trajectory.append((prev_state, action, state))

    def initialize_population(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Initialize population within bounds."""
        lower, upper = bounds
        dimension = len(lower)

        self.population = np.random.uniform(
            lower, upper, size=(self.population_size, dimension)
        )
        self.fitness = np.array([objective(ind) for ind in self.population])
        self.evaluations = self.population_size
        self.best_idx = int(np.argmin(self.fitness))
        self.generation = 0

        # Initialize tracking
        self._last_best_fitness = self.fitness[self.best_idx]
        self._stagnation_count = 0
        self.fitness_history = [float(self.fitness[self.best_idx])]
        self.state_history = []
        self.trajectory = []

        # Record initial state
        self._record_state()

    @abstractmethod
    def step(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        action: Optional[Action] = None,
    ) -> OptState:
        """Perform one generation/step of optimization.

        Args:
            objective: The objective function to minimize
            bounds: (lower, upper) bounds arrays
            action: Optional action to apply this step

        Returns:
            The new optimization state
        """
        pass

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        callback: Optional[Callable[[OptState], Optional[Action]]] = None,
    ) -> OptimizationResult:
        """Run the full optimization.

        Args:
            objective: Function to minimize
            bounds: (lower, upper) bounds for each dimension
            callback: Optional function that receives state and returns action to apply

        Returns:
            OptimizationResult with best solution and trajectory
        """
        self.initialize_population(objective, bounds)

        converged = False
        while self.generation < self.max_generations:
            # Check evaluation budget
            if self.max_evaluations and self.evaluations >= self.max_evaluations:
                break

            # Get action from callback if provided
            action = None
            if callback is not None:
                current_state = self.get_current_state()
                action = callback(current_state)

            # Perform optimization step
            state = self.step(objective, bounds, action)

            # Check convergence
            if state.best_fitness <= self.tol:
                converged = True
                break

            # Check stagnation limit (optional early stopping)
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
        )

    def apply_action(
        self,
        action: Action,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
    ):
        """Apply an action to modify optimization behavior.

        Override in subclasses to implement action-specific behavior.
        """
        pass
