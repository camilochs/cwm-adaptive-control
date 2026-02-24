"""Optimization state representation for CWM."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class OptState:
    """Represents the state of an optimization process.

    This state captures key metrics that inform adaptive operator selection
    and CWM prediction quality evaluation.
    """
    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float  # average distance between solutions
    improvement_rate: float  # improvement in last N generations
    stagnation: int  # generations without improvement

    # Optional extended metrics
    fitness_std: float = 0.0
    population_size: int = 50
    dimension: int = 10
    evaluations: int = 0

    # Dynamic/constrained optimization metrics
    feasibility_ratio: float = 1.0  # fraction of feasible solutions
    constraint_violation_mean: float = 0.0  # mean constraint violation
    improvement_window: float = 0.0  # rolling improvement over a window

    # History tracking (last 5 values for trends)
    fitness_history: list = field(default_factory=list)
    diversity_history: list = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for ML models."""
        return np.array([
            self.generation,
            self.best_fitness,
            self.mean_fitness,
            self.diversity,
            self.improvement_rate,
            self.stagnation,
            self.fitness_std,
            self.population_size,
            self.dimension,
            self.feasibility_ratio,
            self.constraint_violation_mean,
            self.improvement_window,
        ], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray, evaluations: int = 0) -> "OptState":
        """Create state from feature vector."""
        state = cls(
            generation=int(vec[0]),
            best_fitness=float(vec[1]),
            mean_fitness=float(vec[2]),
            diversity=float(vec[3]),
            improvement_rate=float(vec[4]),
            stagnation=int(vec[5]),
            fitness_std=float(vec[6]),
            population_size=int(vec[7]),
            dimension=int(vec[8]),
            evaluations=evaluations,
        )
        # Extended fields (backwards compatible)
        if len(vec) > 9:
            state.feasibility_ratio = float(vec[9])
        if len(vec) > 10:
            state.constraint_violation_mean = float(vec[10])
        if len(vec) > 11:
            state.improvement_window = float(vec[11])
        return state

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "diversity": self.diversity,
            "improvement_rate": self.improvement_rate,
            "stagnation": self.stagnation,
            "fitness_std": self.fitness_std,
            "population_size": self.population_size,
            "dimension": self.dimension,
            "evaluations": self.evaluations,
            "feasibility_ratio": self.feasibility_ratio,
            "constraint_violation_mean": self.constraint_violation_mean,
            "improvement_window": self.improvement_window,
            "fitness_history": self.fitness_history.copy(),
            "diversity_history": self.diversity_history.copy(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptState":
        """Create state from dictionary."""
        return cls(
            generation=d["generation"],
            best_fitness=d["best_fitness"],
            mean_fitness=d["mean_fitness"],
            diversity=d["diversity"],
            improvement_rate=d["improvement_rate"],
            stagnation=d["stagnation"],
            fitness_std=d.get("fitness_std", 0.0),
            population_size=d.get("population_size", 50),
            dimension=d.get("dimension", 10),
            evaluations=d.get("evaluations", 0),
            feasibility_ratio=d.get("feasibility_ratio", 1.0),
            constraint_violation_mean=d.get("constraint_violation_mean", 0.0),
            improvement_window=d.get("improvement_window", 0.0),
            fitness_history=d.get("fitness_history", []),
            diversity_history=d.get("diversity_history", []),
        )

    def copy(self) -> "OptState":
        """Create a deep copy of this state."""
        return OptState(
            generation=self.generation,
            best_fitness=self.best_fitness,
            mean_fitness=self.mean_fitness,
            diversity=self.diversity,
            improvement_rate=self.improvement_rate,
            stagnation=self.stagnation,
            fitness_std=self.fitness_std,
            population_size=self.population_size,
            dimension=self.dimension,
            evaluations=self.evaluations,
            feasibility_ratio=self.feasibility_ratio,
            constraint_violation_mean=self.constraint_violation_mean,
            improvement_window=self.improvement_window,
            fitness_history=self.fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
        )

    def update_history(self, max_len: int = 5):
        """Update history lists with current values."""
        self.fitness_history.append(self.best_fitness)
        self.diversity_history.append(self.diversity)
        if len(self.fitness_history) > max_len:
            self.fitness_history = self.fitness_history[-max_len:]
        if len(self.diversity_history) > max_len:
            self.diversity_history = self.diversity_history[-max_len:]


def compute_diversity(population: np.ndarray) -> float:
    """Compute population diversity as average pairwise distance."""
    if len(population) < 2:
        return 0.0

    n = len(population)
    total_dist = 0.0
    count = 0

    # Sample pairs for efficiency with large populations
    if n > 100:
        indices = np.random.choice(n, size=min(100, n), replace=False)
        sample = population[indices]
    else:
        sample = population

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            total_dist += np.linalg.norm(sample[i] - sample[j])
            count += 1

    return total_dist / count if count > 0 else 0.0


def compute_improvement_rate(fitness_history: list, window: int = 5) -> float:
    """Compute improvement rate over a window of generations."""
    if len(fitness_history) < 2:
        return 0.0

    history = fitness_history[-window:]
    if len(history) < 2:
        return 0.0

    # For minimization: negative change is improvement
    improvements = []
    for i in range(1, len(history)):
        improvements.append(history[i-1] - history[i])

    return np.mean(improvements) if improvements else 0.0
