"""Dynamic benchmark functions with changing landscapes."""

import numpy as np
from typing import Tuple, Callable
from .functions import BenchmarkFunction, Sphere, Rastrigin, Rosenbrock


class DynamicBenchmark(BenchmarkFunction):
    """Benchmark that changes over time/evaluations.

    The landscape shifts periodically, requiring adaptive strategies.
    """

    def __init__(
        self,
        base_function: BenchmarkFunction,
        change_frequency: int = 1000,  # evaluations between changes
        change_severity: float = 0.5,  # how much the optimum shifts
        change_type: str = "shift",  # shift, rotation, or both
        seed: int = None,
    ):
        self.base_function = base_function
        self.dimension = base_function.dimension
        self.change_frequency = change_frequency
        self.change_severity = change_severity
        self.change_type = change_type

        self.rng = np.random.RandomState(seed)
        self._bounds = base_function.bounds
        self._optimum = base_function.optimum

        # Dynamic state
        self.evaluation_count = 0
        self.change_count = 0
        self.current_shift = np.zeros(self.dimension)
        self.current_rotation = np.eye(self.dimension)

        self._apply_change()

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def _get_optimum(self) -> float:
        return self._optimum

    def _apply_change(self):
        """Apply a change to the landscape."""
        lower, upper = self._bounds
        range_size = upper - lower

        if self.change_type in ["shift", "both"]:
            # Random shift of the optimum
            self.current_shift = self.rng.uniform(
                -self.change_severity * range_size,
                self.change_severity * range_size,
                size=self.dimension
            )

        if self.change_type in ["rotation", "both"]:
            # Random rotation matrix
            Q, _ = np.linalg.qr(self.rng.randn(self.dimension, self.dimension))
            # Interpolate towards new rotation
            alpha = self.change_severity
            self.current_rotation = (1 - alpha) * self.current_rotation + alpha * Q
            # Re-orthogonalize
            self.current_rotation, _ = np.linalg.qr(self.current_rotation)

        self.change_count += 1

    def __call__(self, x: np.ndarray) -> float:
        self.evaluation_count += 1

        # Check if it's time for a change
        if self.evaluation_count % self.change_frequency == 0:
            self._apply_change()

        # Apply transformation
        x_transformed = x.copy()

        if self.change_type in ["shift", "both"]:
            x_transformed = x_transformed - self.current_shift

        if self.change_type in ["rotation", "both"]:
            x_transformed = self.current_rotation @ x_transformed

        # Clip to bounds after transformation
        lower, upper = self._bounds
        x_transformed = np.clip(x_transformed, lower, upper)

        return self.base_function(x_transformed)

    def reset(self):
        """Reset the dynamic state."""
        self.evaluation_count = 0
        self.change_count = 0
        self.current_shift = np.zeros(self.dimension)
        self.current_rotation = np.eye(self.dimension)
        self._apply_change()


class CompositeChangingBenchmark(BenchmarkFunction):
    """Benchmark that switches between different functions over time."""

    def __init__(
        self,
        functions: list,  # List of BenchmarkFunction instances
        switch_frequency: int = 2000,
        seed: int = None,
    ):
        assert len(functions) > 1, "Need at least 2 functions"
        assert all(f.dimension == functions[0].dimension for f in functions)

        self.functions = functions
        self.dimension = functions[0].dimension
        self.switch_frequency = switch_frequency
        self.rng = np.random.RandomState(seed)

        self._bounds = functions[0].bounds
        self._optimum = min(f.optimum for f in functions)

        self.evaluation_count = 0
        self.current_function_idx = 0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def _get_optimum(self) -> float:
        return self._optimum

    @property
    def current_function(self) -> BenchmarkFunction:
        return self.functions[self.current_function_idx]

    def __call__(self, x: np.ndarray) -> float:
        self.evaluation_count += 1

        # Check if it's time to switch
        if self.evaluation_count % self.switch_frequency == 0:
            # Switch to a different function
            old_idx = self.current_function_idx
            while self.current_function_idx == old_idx:
                self.current_function_idx = self.rng.randint(len(self.functions))

        return self.current_function(x)

    def reset(self):
        """Reset the dynamic state."""
        self.evaluation_count = 0
        self.current_function_idx = 0


def create_dynamic_sphere(dimension: int = 10, **kwargs) -> DynamicBenchmark:
    """Create a dynamic sphere benchmark."""
    return DynamicBenchmark(Sphere(dimension), **kwargs)


def create_dynamic_rastrigin(dimension: int = 10, **kwargs) -> DynamicBenchmark:
    """Create a dynamic Rastrigin benchmark."""
    return DynamicBenchmark(Rastrigin(dimension), **kwargs)


def create_composite_benchmark(dimension: int = 10, **kwargs) -> CompositeChangingBenchmark:
    """Create a composite benchmark that switches between functions."""
    functions = [
        Sphere(dimension),
        Rastrigin(dimension),
        Rosenbrock(dimension),
    ]
    return CompositeChangingBenchmark(functions, **kwargs)


class DynamicConstraint:
    """A constraint that changes over time."""

    def __init__(
        self,
        constraint_type: str,  # "linear", "spherical", "rectangular"
        change_frequency: int = 500,
        seed: int = None,
    ):
        self.constraint_type = constraint_type
        self.change_frequency = change_frequency
        self.rng = np.random.RandomState(seed)
        self.evaluation_count = 0
        self.change_count = 0

        # Constraint parameters (will be set by _apply_change)
        self.center = None
        self.radius = None
        self.normal = None
        self.offset = None
        self.lower_bound = None
        self.upper_bound = None

    def _apply_change(self, dimension: int, bounds: Tuple[np.ndarray, np.ndarray]):
        """Apply a change to the constraint."""
        lower, upper = bounds
        range_size = upper - lower

        if self.constraint_type == "spherical":
            # Spherical exclusion zone that moves
            self.center = lower + self.rng.uniform(0.2, 0.8, size=dimension) * range_size
            self.radius = self.rng.uniform(0.1, 0.3) * np.mean(range_size)

        elif self.constraint_type == "linear":
            # Linear constraint (hyperplane) that rotates
            self.normal = self.rng.randn(dimension)
            self.normal /= np.linalg.norm(self.normal)
            # Offset so constraint passes through feasible region
            center = (lower + upper) / 2
            self.offset = np.dot(self.normal, center) + self.rng.uniform(-0.3, 0.3) * np.mean(range_size)

        elif self.constraint_type == "rectangular":
            # Rectangular forbidden zone
            zone_size = self.rng.uniform(0.1, 0.3, size=dimension) * range_size
            zone_center = lower + self.rng.uniform(0.3, 0.7, size=dimension) * range_size
            self.lower_bound = zone_center - zone_size / 2
            self.upper_bound = zone_center + zone_size / 2

        self.change_count += 1

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if point x satisfies the constraint."""
        if self.constraint_type == "spherical":
            # Feasible if OUTSIDE the exclusion sphere
            dist = np.linalg.norm(x - self.center)
            return dist > self.radius

        elif self.constraint_type == "linear":
            # Feasible if on one side of hyperplane
            return np.dot(self.normal, x) >= self.offset

        elif self.constraint_type == "rectangular":
            # Feasible if OUTSIDE the forbidden rectangle
            inside = np.all(x >= self.lower_bound) and np.all(x <= self.upper_bound)
            return not inside

        return True

    def violation(self, x: np.ndarray) -> float:
        """Return constraint violation amount (0 if feasible)."""
        if self.constraint_type == "spherical":
            dist = np.linalg.norm(x - self.center)
            return max(0, self.radius - dist)

        elif self.constraint_type == "linear":
            return max(0, self.offset - np.dot(self.normal, x))

        elif self.constraint_type == "rectangular":
            if self.is_feasible(x):
                return 0.0
            # Distance to exit the forbidden zone
            violations = []
            for i in range(len(x)):
                if self.lower_bound[i] <= x[i] <= self.upper_bound[i]:
                    violations.append(min(x[i] - self.lower_bound[i], self.upper_bound[i] - x[i]))
            return sum(violations) if violations else 0.0

        return 0.0


class DynamicConstrainedBenchmark(BenchmarkFunction):
    """Benchmark with constraints that change over time.

    This creates a challenging scenario where:
    1. The objective function may be static or dynamic
    2. Constraints (feasible regions) change periodically
    3. Good solutions may suddenly become infeasible

    This tests the algorithm's ability to:
    - Track moving feasible regions
    - Recover from constraint violations
    - Balance exploration vs exploitation under uncertainty
    """

    def __init__(
        self,
        base_function: BenchmarkFunction,
        constraints: list = None,  # List of DynamicConstraint
        constraint_types: list = None,  # Or specify types to create
        change_frequency: int = 500,
        penalty_coefficient: float = 1000.0,
        seed: int = None,
    ):
        self.base_function = base_function
        self.dimension = base_function.dimension
        self.penalty_coefficient = penalty_coefficient
        self.rng = np.random.RandomState(seed)

        self._bounds = base_function.bounds
        self._optimum = base_function.optimum  # Note: may not be achievable due to constraints

        # Create or use provided constraints
        if constraints is not None:
            self.constraints = constraints
        elif constraint_types is not None:
            self.constraints = [
                DynamicConstraint(ctype, change_frequency, seed=seed + i if seed else None)
                for i, ctype in enumerate(constraint_types)
            ]
        else:
            # Default: one of each type
            self.constraints = [
                DynamicConstraint("spherical", change_frequency, seed=seed),
                DynamicConstraint("linear", change_frequency, seed=seed + 1 if seed else None),
            ]

        # Initialize constraints
        for c in self.constraints:
            c._apply_change(self.dimension, self._bounds)

        self.evaluation_count = 0
        self.change_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def _get_optimum(self) -> float:
        return self._optimum

    def __call__(self, x: np.ndarray) -> float:
        self.evaluation_count += 1

        # Check if it's time for constraint changes
        for constraint in self.constraints:
            if self.evaluation_count % constraint.change_frequency == 0:
                constraint._apply_change(self.dimension, self._bounds)
                self.change_count += 1

        # Evaluate base function
        fitness = self.base_function(x)

        # Calculate total constraint violation
        total_violation = sum(c.violation(x) for c in self.constraints)

        if total_violation > 0:
            self.infeasible_count += 1
            # Penalty method: add penalty proportional to violation
            fitness += self.penalty_coefficient * total_violation
        else:
            self.feasible_count += 1

        return fitness

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if point is currently feasible."""
        return all(c.is_feasible(x) for c in self.constraints)

    def get_violation(self, x: np.ndarray) -> float:
        """Get total constraint violation."""
        return sum(c.violation(x) for c in self.constraints)

    def reset(self):
        """Reset the dynamic state."""
        self.evaluation_count = 0
        self.change_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        for c in self.constraints:
            c.evaluation_count = 0
            c.change_count = 0
            c._apply_change(self.dimension, self._bounds)

    @property
    def feasibility_ratio(self) -> float:
        """Ratio of feasible evaluations."""
        total = self.feasible_count + self.infeasible_count
        return self.feasible_count / total if total > 0 else 0.0


class MultiConstraintDynamicBenchmark(DynamicConstrainedBenchmark):
    """Benchmark with multiple constraints that change at different rates.

    Creates a more complex scenario where different constraints
    change at different frequencies, making adaptation harder.
    """

    def __init__(
        self,
        base_function: BenchmarkFunction,
        n_constraints: int = 3,
        min_frequency: int = 300,
        max_frequency: int = 1500,
        penalty_coefficient: float = 1000.0,
        seed: int = None,
    ):
        rng = np.random.RandomState(seed)
        constraint_types = ["spherical", "linear", "rectangular"]

        constraints = []
        for i in range(n_constraints):
            ctype = constraint_types[i % len(constraint_types)]
            freq = rng.randint(min_frequency, max_frequency)
            constraints.append(
                DynamicConstraint(ctype, freq, seed=seed + i if seed else None)
            )

        super().__init__(
            base_function=base_function,
            constraints=constraints,
            penalty_coefficient=penalty_coefficient,
            seed=seed,
        )


# Factory functions for constrained benchmarks

def create_constrained_sphere(
    dimension: int = 10,
    constraint_types: list = None,
    change_frequency: int = 500,
    **kwargs
) -> DynamicConstrainedBenchmark:
    """Create a sphere function with dynamic constraints."""
    if constraint_types is None:
        constraint_types = ["spherical", "linear"]
    return DynamicConstrainedBenchmark(
        Sphere(dimension),
        constraint_types=constraint_types,
        change_frequency=change_frequency,
        **kwargs
    )


def create_constrained_rastrigin(
    dimension: int = 10,
    constraint_types: list = None,
    change_frequency: int = 500,
    **kwargs
) -> DynamicConstrainedBenchmark:
    """Create a Rastrigin function with dynamic constraints."""
    if constraint_types is None:
        constraint_types = ["spherical", "linear"]
    return DynamicConstrainedBenchmark(
        Rastrigin(dimension),
        constraint_types=constraint_types,
        change_frequency=change_frequency,
        **kwargs
    )


def create_multi_constrained_benchmark(
    dimension: int = 10,
    n_constraints: int = 3,
    base_function: str = "sphere",
    **kwargs
) -> MultiConstraintDynamicBenchmark:
    """Create a benchmark with multiple constraints changing at different rates."""
    if base_function == "sphere":
        func = Sphere(dimension)
    elif base_function == "rastrigin":
        func = Rastrigin(dimension)
    elif base_function == "rosenbrock":
        func = Rosenbrock(dimension)
    else:
        func = Sphere(dimension)

    return MultiConstraintDynamicBenchmark(
        func,
        n_constraints=n_constraints,
        **kwargs
    )


class PursuitConstraint:
    """A constraint that chases the current best solution.

    This creates a scenario where staying in one place is penalized,
    forcing the algorithm to constantly move. DE struggles here because
    it loses its best individual guidance.
    """

    def __init__(
        self,
        dimension: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pursuit_speed: float = 0.3,  # How fast the constraint catches up
        exclusion_radius: float = 0.2,  # Size of forbidden zone (fraction of range)
        update_frequency: int = 100,  # How often to update pursuit
        seed: int = None,
    ):
        self.dimension = dimension
        self.bounds = bounds
        self.pursuit_speed = pursuit_speed
        self.update_frequency = update_frequency
        self.rng = np.random.RandomState(seed)

        lower, upper = bounds
        self.range_size = upper - lower
        self.exclusion_radius = exclusion_radius * np.mean(self.range_size)

        # Start at random position
        self.center = lower + self.rng.uniform(0.3, 0.7, size=dimension) * self.range_size
        self.target = None  # Will track best solution
        self.evaluation_count = 0

    def update_target(self, best_solution: np.ndarray):
        """Update the target the constraint is pursuing."""
        self.target = best_solution.copy()

    def step(self):
        """Move constraint towards target."""
        self.evaluation_count += 1

        if self.evaluation_count % self.update_frequency == 0 and self.target is not None:
            # Move towards target
            direction = self.target - self.center
            distance = np.linalg.norm(direction)

            if distance > 0.01:
                # Move a fraction towards target
                step = self.pursuit_speed * direction
                self.center = self.center + step

                # Keep within bounds
                lower, upper = self.bounds
                self.center = np.clip(self.center, lower + self.exclusion_radius,
                                      upper - self.exclusion_radius)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Feasible if outside the exclusion zone."""
        dist = np.linalg.norm(x - self.center)
        return dist > self.exclusion_radius

    def violation(self, x: np.ndarray) -> float:
        """Return violation amount."""
        dist = np.linalg.norm(x - self.center)
        return max(0, self.exclusion_radius - dist)


class PursuitConstrainedBenchmark(BenchmarkFunction):
    """Benchmark where a constraint zone chases the best solution.

    This is designed to exploit DE's weakness: when the best individual
    becomes infeasible, DE loses its primary search direction.

    MCTS+CWM should perform better here because:
    1. It can learn that staying put leads to infeasibility
    2. It can plan escape routes proactively
    3. It maintains diversity through planned exploration
    """

    def __init__(
        self,
        base_function: BenchmarkFunction,
        n_pursuers: int = 1,
        pursuit_speed: float = 0.3,
        exclusion_radius: float = 0.15,
        update_frequency: int = 100,
        penalty_coefficient: float = 1000.0,
        seed: int = None,
    ):
        self.base_function = base_function
        self.dimension = base_function.dimension
        self.penalty_coefficient = penalty_coefficient
        self.rng = np.random.RandomState(seed)

        self._bounds = base_function.bounds
        self._optimum = base_function.optimum

        # Create pursuers
        self.pursuers = [
            PursuitConstraint(
                self.dimension,
                self._bounds,
                pursuit_speed=pursuit_speed,
                exclusion_radius=exclusion_radius,
                update_frequency=update_frequency,
                seed=seed + i if seed else None,
            )
            for i in range(n_pursuers)
        ]

        self.evaluation_count = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.feasible_count = 0
        self.infeasible_count = 0
        self.times_best_invalidated = 0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def _get_optimum(self) -> float:
        return self._optimum

    def __call__(self, x: np.ndarray) -> float:
        self.evaluation_count += 1

        # Evaluate base function
        fitness = self.base_function(x)

        # Check constraint violation
        total_violation = sum(p.violation(x) for p in self.pursuers)

        if total_violation > 0:
            self.infeasible_count += 1
            fitness += self.penalty_coefficient * total_violation
        else:
            self.feasible_count += 1

            # Track best feasible solution
            if fitness < self.best_fitness:
                # Check if previous best is now infeasible
                if self.best_solution is not None:
                    prev_violation = sum(p.violation(self.best_solution) for p in self.pursuers)
                    if prev_violation > 0:
                        self.times_best_invalidated += 1

                self.best_solution = x.copy()
                self.best_fitness = fitness

                # Update pursuer targets
                for p in self.pursuers:
                    p.update_target(self.best_solution)

        # Move pursuers
        for p in self.pursuers:
            p.step()

        return fitness

    def is_feasible(self, x: np.ndarray) -> bool:
        return all(p.is_feasible(x) for p in self.pursuers)

    @property
    def feasibility_ratio(self) -> float:
        total = self.feasible_count + self.infeasible_count
        return self.feasible_count / total if total > 0 else 0.0

    def reset(self):
        self.evaluation_count = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.feasible_count = 0
        self.infeasible_count = 0
        self.times_best_invalidated = 0
        for p in self.pursuers:
            p.evaluation_count = 0
            lower, upper = self._bounds
            p.center = lower + p.rng.uniform(0.3, 0.7, size=self.dimension) * p.range_size
            p.target = None


class NoisyDynamicConstrainedBenchmark(BenchmarkFunction):
    """Benchmark combining: multimodality + many constraints + frequent changes + noise.

    This is designed as a worst-case scenario for static algorithms:
    1. Base function is multimodal (Rastrigin)
    2. Multiple constraints (5+) changing at different rates
    3. Very frequent changes (every 100-200 evals)
    4. Noisy fitness evaluations
    5. Constraints can overlap, creating complex feasible regions
    """

    def __init__(
        self,
        base_function: BenchmarkFunction,
        n_constraints: int = 5,
        min_change_freq: int = 100,
        max_change_freq: int = 300,
        noise_std: float = 0.1,  # Relative noise level
        penalty_coefficient: float = 1000.0,
        seed: int = None,
    ):
        self.base_function = base_function
        self.dimension = base_function.dimension
        self.n_constraints = n_constraints
        self.noise_std = noise_std
        self.penalty_coefficient = penalty_coefficient
        self.rng = np.random.RandomState(seed)

        self._bounds = base_function.bounds
        self._optimum = base_function.optimum

        # Create diverse constraints with different change frequencies
        constraint_types = ["spherical", "linear", "rectangular"]
        self.constraints = []
        self.constraint_frequencies = []

        for i in range(n_constraints):
            ctype = constraint_types[i % len(constraint_types)]
            freq = self.rng.randint(min_change_freq, max_change_freq)
            self.constraints.append(
                DynamicConstraint(ctype, freq, seed=seed + i if seed else None)
            )
            self.constraint_frequencies.append(freq)

        # Initialize all constraints
        for c in self.constraints:
            c._apply_change(self.dimension, self._bounds)

        self.evaluation_count = 0
        self.change_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._bounds

    def _get_optimum(self) -> float:
        return self._optimum

    def __call__(self, x: np.ndarray) -> float:
        self.evaluation_count += 1

        # Check for constraint changes
        for constraint in self.constraints:
            if self.evaluation_count % constraint.change_frequency == 0:
                constraint._apply_change(self.dimension, self._bounds)
                self.change_count += 1

        # Evaluate base function
        true_fitness = self.base_function(x)

        # Add noise (relative to fitness value)
        noise = self.rng.normal(0, self.noise_std * (abs(true_fitness) + 1))
        noisy_fitness = true_fitness + noise

        # Calculate constraint violations
        total_violation = sum(c.violation(x) for c in self.constraints)

        if total_violation > 0:
            self.infeasible_count += 1
            noisy_fitness += self.penalty_coefficient * total_violation
        else:
            self.feasible_count += 1

        return noisy_fitness

    def is_feasible(self, x: np.ndarray) -> bool:
        return all(c.is_feasible(x) for c in self.constraints)

    @property
    def feasibility_ratio(self) -> float:
        total = self.feasible_count + self.infeasible_count
        return self.feasible_count / total if total > 0 else 0.0

    def reset(self):
        self.evaluation_count = 0
        self.change_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        for c in self.constraints:
            c.evaluation_count = 0
            c.change_count = 0
            c._apply_change(self.dimension, self._bounds)


def create_extreme_benchmark(
    dimension: int = 10,
    base_function: str = "rastrigin",
    n_constraints: int = 5,
    min_change_freq: int = 100,
    max_change_freq: int = 300,
    noise_std: float = 0.1,
    **kwargs
) -> NoisyDynamicConstrainedBenchmark:
    """Create an extreme benchmark combining all challenging factors."""
    if base_function == "sphere":
        func = Sphere(dimension)
    elif base_function == "rastrigin":
        func = Rastrigin(dimension)
    elif base_function == "rosenbrock":
        func = Rosenbrock(dimension)
    else:
        func = Rastrigin(dimension)

    return NoisyDynamicConstrainedBenchmark(
        func,
        n_constraints=n_constraints,
        min_change_freq=min_change_freq,
        max_change_freq=max_change_freq,
        noise_std=noise_std,
        **kwargs
    )


def create_pursuit_benchmark(
    dimension: int = 10,
    base_function: str = "rastrigin",
    n_pursuers: int = 1,
    pursuit_speed: float = 0.3,
    exclusion_radius: float = 0.15,
    update_frequency: int = 100,
    **kwargs
) -> PursuitConstrainedBenchmark:
    """Create a benchmark with pursuing constraints."""
    if base_function == "sphere":
        func = Sphere(dimension)
    elif base_function == "rastrigin":
        func = Rastrigin(dimension)
    elif base_function == "rosenbrock":
        func = Rosenbrock(dimension)
    else:
        func = Rastrigin(dimension)

    return PursuitConstrainedBenchmark(
        func,
        n_pursuers=n_pursuers,
        pursuit_speed=pursuit_speed,
        exclusion_radius=exclusion_radius,
        update_frequency=update_frequency,
        **kwargs
    )
