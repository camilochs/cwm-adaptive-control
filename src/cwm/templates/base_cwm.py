"""Template for synthesized Code World Models."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class CWMState:
    """State representation for CWM predictions."""
    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    improvement_rate: float
    stagnation: int
    fitness_std: float = 0.0
    population_size: int = 50
    dimension: int = 10
    # Dynamic/constrained optimization metrics
    feasibility_ratio: float = 1.0
    constraint_violation_mean: float = 0.0
    improvement_window: float = 0.0


@dataclass
class CWMAction:
    """Action representation for CWM."""
    action_type: str  # "mutation", "crossover", "selection", "restart"
    name: str
    params: Dict[str, Any]


class BaseCWM:
    """Base class for Code World Models.

    A CWM predicts how optimization state evolves under different actions.
    This is the interface that synthesized CWMs must implement.
    """

    def predict_next_state(self, state: CWMState, action: CWMAction) -> CWMState:
        """Predict the next state after applying an action.

        Args:
            state: Current optimization state
            action: Action to apply

        Returns:
            Predicted next state
        """
        raise NotImplementedError

    def get_legal_actions(self, state: CWMState) -> List[CWMAction]:
        """Get actions that are legal/sensible in current state.

        Args:
            state: Current optimization state

        Returns:
            List of legal actions
        """
        raise NotImplementedError

    def is_terminal(self, state: CWMState) -> bool:
        """Check if state is terminal (optimization should stop).

        Args:
            state: Current optimization state

        Returns:
            True if terminal state
        """
        raise NotImplementedError

    def evaluate_state(self, state: CWMState) -> float:
        """Evaluate how good a state is (for MCTS rewards).

        Higher is better. Should consider:
        - Fitness value (lower is better for minimization)
        - Improvement rate
        - Diversity health

        Args:
            state: State to evaluate

        Returns:
            Evaluation score (higher is better)
        """
        raise NotImplementedError


# Default implementation for reference
class DefaultCWM(BaseCWM):
    """Default CWM with simple heuristic predictions."""

    def __init__(
        self,
        max_generations: int = 100,
        target_fitness: float = 1e-6,
    ):
        self.max_generations = max_generations
        self.target_fitness = target_fitness

    def predict_next_state(self, state: CWMState, action: CWMAction) -> CWMState:
        """Simple heuristic prediction."""
        # Base prediction: slight improvement
        improvement = state.improvement_rate * 0.9  # Decaying improvement

        new_best = state.best_fitness * (1 - improvement * 0.1)
        new_mean = state.mean_fitness * (1 - improvement * 0.05)

        # Action-specific effects
        if action.action_type == "mutation":
            rate = action.params.get("rate", 0.1)
            # More mutation increases diversity, may help or hurt fitness
            diversity_change = rate * 0.1
            fitness_noise = rate * 0.05
        elif action.action_type == "crossover":
            diversity_change = -0.02  # Crossover typically reduces diversity
            fitness_noise = 0.02
        elif action.action_type == "restart":
            fraction = action.params.get("fraction", 0.3)
            diversity_change = fraction * 0.5  # Big diversity boost
            fitness_noise = fraction * 0.1
        else:
            diversity_change = 0.0
            fitness_noise = 0.01

        new_diversity = max(0.01, state.diversity + diversity_change)

        # Stagnation update
        if new_best < state.best_fitness - 1e-8:
            new_stagnation = 0
        else:
            new_stagnation = state.stagnation + 1

        # New improvement rate
        if new_stagnation == 0:
            new_improvement = (state.best_fitness - new_best) / max(state.best_fitness, 1e-10)
        else:
            new_improvement = state.improvement_rate * 0.8

        return CWMState(
            generation=state.generation + 1,
            best_fitness=max(0, new_best + fitness_noise * (2 * (hash(str(action)) % 100) / 100 - 1)),
            mean_fitness=new_mean,
            diversity=new_diversity,
            improvement_rate=new_improvement,
            stagnation=new_stagnation,
            fitness_std=state.fitness_std,
            population_size=state.population_size,
            dimension=state.dimension,
        )

    def get_legal_actions(self, state: CWMState) -> List[CWMAction]:
        """All actions are legal by default."""
        from ...actions import DEFAULT_ACTION_SPACE
        return [
            CWMAction(
                action_type=a.action_type.value,
                name=a.name,
                params=dict(a.params),
            )
            for a in DEFAULT_ACTION_SPACE
        ]

    def is_terminal(self, state: CWMState) -> bool:
        """Terminal if converged or max generations reached."""
        return (
            state.best_fitness <= self.target_fitness or
            state.generation >= self.max_generations or
            state.stagnation > self.max_generations // 2
        )

    def evaluate_state(self, state: CWMState) -> float:
        """Evaluate state quality."""
        # Negative log fitness (higher is better for lower fitness)
        import math
        fitness_score = -math.log10(max(state.best_fitness, 1e-10))

        # Bonus for improvement
        improvement_score = state.improvement_rate * 10

        # Bonus for healthy diversity
        diversity_score = min(state.diversity, 1.0)

        # Penalty for stagnation
        stagnation_penalty = state.stagnation * 0.1

        return fitness_score + improvement_score + diversity_score - stagnation_penalty
