"""Trajectory collector for gathering optimization data."""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field

from ..state import OptState
from ..actions import Action, DEFAULT_ACTION_SPACE
from ..optimizers.base import BaseOptimizer, OptimizationResult
from ..benchmarks.functions import BenchmarkFunction


@dataclass
class Trajectory:
    """A single optimization trajectory."""
    problem_name: str
    dimension: int
    transitions: List[Tuple[Dict, Optional[Dict], Dict]]  # (state, action, next_state) as dicts
    final_fitness: float
    total_evaluations: int
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "problem_name": self.problem_name,
            "dimension": self.dimension,
            "transitions": self.transitions,
            "final_fitness": self.final_fitness,
            "total_evaluations": self.total_evaluations,
            "converged": self.converged,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            problem_name=d["problem_name"],
            dimension=d["dimension"],
            transitions=d["transitions"],
            final_fitness=d["final_fitness"],
            total_evaluations=d["total_evaluations"],
            converged=d["converged"],
            metadata=d.get("metadata", {}),
        )


class TrajectoryCollector:
    """Collects optimization trajectories for CWM training."""

    def __init__(
        self,
        action_space: List[Action] = None,
        action_selection: str = "random",  # "random", "round_robin", "epsilon_greedy"
        action_frequency: int = 1,  # Apply action every N generations
        epsilon: float = 0.3,  # For epsilon-greedy
    ):
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.action_selection = action_selection
        self.action_frequency = action_frequency
        self.epsilon = epsilon

        self._action_idx = 0  # For round-robin
        self._trajectories: List[Trajectory] = []

    def _select_action(self, state: OptState) -> Optional[Action]:
        """Select an action based on the selection strategy."""
        if self.action_selection == "random":
            return np.random.choice(self.action_space)

        elif self.action_selection == "round_robin":
            action = self.action_space[self._action_idx]
            self._action_idx = (self._action_idx + 1) % len(self.action_space)
            return action

        elif self.action_selection == "epsilon_greedy":
            # Random action with probability epsilon, else heuristic
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space)
            else:
                return self._heuristic_action(state)

        return None

    def _heuristic_action(self, state: OptState) -> Action:
        """Simple heuristic for action selection based on state."""
        # High stagnation -> restart
        if state.stagnation > 10:
            restart_actions = [a for a in self.action_space if a.action_type.value == "restart"]
            if restart_actions:
                return np.random.choice(restart_actions)

        # Low diversity -> increase mutation
        if state.diversity < 0.1:
            mutation_actions = [a for a in self.action_space
                              if a.action_type.value == "mutation" and "high" in a.name]
            if mutation_actions:
                return np.random.choice(mutation_actions)

        # Default: random crossover action
        crossover_actions = [a for a in self.action_space if a.action_type.value == "crossover"]
        if crossover_actions:
            return np.random.choice(crossover_actions)

        return np.random.choice(self.action_space)

    def collect_trajectory(
        self,
        optimizer: BaseOptimizer,
        problem: BenchmarkFunction,
        metadata: Optional[Dict] = None,
    ) -> Trajectory:
        """Collect a single optimization trajectory."""
        generation_counter = [0]

        def callback(state: OptState) -> Optional[Action]:
            generation_counter[0] += 1
            if generation_counter[0] % self.action_frequency == 0:
                return self._select_action(state)
            return None

        # Run optimization with trajectory recording
        result = optimizer.optimize(
            objective=problem,
            bounds=problem.bounds,
            callback=callback,
        )

        # Convert trajectory to serializable format
        transitions = []
        for state, action, next_state in result.trajectory:
            transitions.append((
                state.to_dict(),
                action.to_dict() if action else None,
                next_state.to_dict(),
            ))

        trajectory = Trajectory(
            problem_name=problem.name,
            dimension=problem.dimension,
            transitions=transitions,
            final_fitness=result.best_fitness,
            total_evaluations=result.evaluations,
            converged=result.converged,
            metadata=metadata or {},
        )

        self._trajectories.append(trajectory)
        return trajectory

    def collect_multiple(
        self,
        optimizer_factory: Callable[[], BaseOptimizer],
        problems: List[BenchmarkFunction],
        runs_per_problem: int = 10,
        verbose: bool = True,
    ) -> List[Trajectory]:
        """Collect multiple trajectories across problems."""
        trajectories = []

        for problem in problems:
            if verbose:
                print(f"Collecting trajectories for {problem.name} (dim={problem.dimension})...")

            for run in range(runs_per_problem):
                optimizer = optimizer_factory()
                metadata = {
                    "run": run,
                    "problem_class": problem.__class__.__name__,
                }

                traj = self.collect_trajectory(optimizer, problem, metadata)
                trajectories.append(traj)

                if verbose and (run + 1) % 5 == 0:
                    print(f"  Run {run + 1}/{runs_per_problem}: "
                          f"fitness={traj.final_fitness:.6f}, "
                          f"transitions={len(traj)}")

        return trajectories

    @property
    def trajectories(self) -> List[Trajectory]:
        """Get all collected trajectories."""
        return self._trajectories

    def clear(self):
        """Clear collected trajectories."""
        self._trajectories = []
        self._action_idx = 0

    def get_transition_dataset(self) -> List[Tuple[Dict, Optional[Dict], Dict]]:
        """Get all transitions as a flat list for training."""
        all_transitions = []
        for traj in self._trajectories:
            all_transitions.extend(traj.transitions)
        return all_transitions

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected trajectories."""
        if not self._trajectories:
            return {"count": 0}

        final_fitnesses = [t.final_fitness for t in self._trajectories]
        total_transitions = sum(len(t) for t in self._trajectories)

        return {
            "count": len(self._trajectories),
            "total_transitions": total_transitions,
            "avg_transitions_per_run": total_transitions / len(self._trajectories),
            "convergence_rate": sum(1 for t in self._trajectories if t.converged) / len(self._trajectories),
            "best_fitness": min(final_fitnesses),
            "worst_fitness": max(final_fitnesses),
            "mean_fitness": np.mean(final_fitnesses),
            "std_fitness": np.std(final_fitnesses),
        }
