"""Standard benchmark functions for optimization testing."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type
import numpy as np


class BenchmarkFunction(ABC):
    """Abstract base class for benchmark functions."""

    def __init__(self, dimension: int = 10):
        self.dimension = dimension
        self._bounds = self._get_bounds()
        self._optimum = self._get_optimum()

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_bounds, upper_bounds) arrays."""
        return self._bounds

    @property
    def optimum(self) -> float:
        """Returns the global optimum value."""
        return self._optimum

    @abstractmethod
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Define the search bounds."""
        pass

    @abstractmethod
    def _get_optimum(self) -> float:
        """Define the global optimum."""
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the function at x."""
        pass

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate a population of solutions."""
        return np.array([self(ind) for ind in population])

    def random_solution(self) -> np.ndarray:
        """Generate a random solution within bounds."""
        lower, upper = self.bounds
        return np.random.uniform(lower, upper)

    def random_population(self, size: int) -> np.ndarray:
        """Generate a random population within bounds."""
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(size, self.dimension))


class Sphere(BenchmarkFunction):
    """Sphere function: f(x) = sum(x_i^2)

    Simple unimodal function, easy to optimize.
    Global minimum: f(0,...,0) = 0
    """
    name = "sphere"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -5.12),
            np.full(self.dimension, 5.12)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class Rastrigin(BenchmarkFunction):
    """Rastrigin function: highly multimodal with regular local minima.

    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0
    """
    name = "rastrigin"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -5.12),
            np.full(self.dimension, 5.12)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock function: classic banana-shaped valley.

    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1,...,1) = 0
    """
    name = "rosenbrock"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -5.0),
            np.full(self.dimension, 10.0)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class Ackley(BenchmarkFunction):
    """Ackley function: multimodal with large nearly-flat regions.

    Global minimum: f(0,...,0) = 0
    """
    name = "ackley"

    def __init__(self, dimension: int = 10, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(dimension)

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -32.768),
            np.full(self.dimension, 32.768)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(self.c * x))
        return -self.a * np.exp(-self.b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + self.a + np.e


class Griewank(BenchmarkFunction):
    """Griewank function: many local minima, but structure helps optimization.

    Global minimum: f(0,...,0) = 0
    """
    name = "griewank"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -600.0),
            np.full(self.dimension, 600.0)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_sq - prod_cos + 1


class Schwefel(BenchmarkFunction):
    """Schwefel function: deceptive function with global optimum far from local optima.

    Global minimum: f(420.9687,...,420.9687) ≈ 0
    """
    name = "schwefel"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -500.0),
            np.full(self.dimension, 500.0)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Levy(BenchmarkFunction):
    """Levy function: multimodal with single global minimum.

    Global minimum: f(1,...,1) = 0
    """
    name = "levy"

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.dimension, -10.0),
            np.full(self.dimension, 10.0)
        )

    def _get_optimum(self) -> float:
        return 0.0

    def __call__(self, x: np.ndarray) -> float:
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3


# Registry of all benchmark functions
BENCHMARK_REGISTRY: Dict[str, Type[BenchmarkFunction]] = {
    "sphere": Sphere,
    "rastrigin": Rastrigin,
    "rosenbrock": Rosenbrock,
    "ackley": Ackley,
    "griewank": Griewank,
    "schwefel": Schwefel,
    "levy": Levy,
}


def get_benchmark(name: str, dimension: int = 10) -> BenchmarkFunction:
    """Get a benchmark function by name."""
    name_lower = name.lower()
    if name_lower not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARK_REGISTRY[name_lower](dimension=dimension)


def list_benchmarks() -> list:
    """List all available benchmark names."""
    return list(BENCHMARK_REGISTRY.keys())
