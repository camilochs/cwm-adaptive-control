"""Action space for adaptive metaheuristic control."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from enum import Enum


class ActionType(Enum):
    """Types of actions available to the optimizer."""
    MUTATION = "mutation"
    CROSSOVER = "crossover"
    SELECTION = "selection"
    RESTART = "restart"
    PARAMETER = "parameter"


@dataclass(frozen=True)
class Action:
    """Represents an action/operator to apply."""
    action_type: ActionType
    name: str
    params: Tuple[Tuple[str, Any], ...]  # Immutable params as tuple of tuples

    def __post_init__(self):
        # Ensure params is a tuple
        if not isinstance(self.params, tuple):
            object.__setattr__(self, 'params', tuple(self.params))

    @property
    def params_dict(self) -> Dict[str, Any]:
        """Get parameters as a dictionary."""
        return dict(self.params)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params)
        return f"Action({self.action_type.value}:{self.name}, {params_str})"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type.value,
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        """Create action from dictionary."""
        return cls(
            action_type=ActionType(d["action_type"]),
            name=d["name"],
            params=tuple(d["params"].items()),
        )


def make_action(action_type: str, name: str, **params) -> Action:
    """Convenience function to create an action."""
    return Action(
        action_type=ActionType(action_type),
        name=name,
        params=tuple(params.items()),
    )


# Predefined action space
MUTATION_ACTIONS = [
    make_action("mutation", "gaussian_low", rate=0.1, sigma=0.1),
    make_action("mutation", "gaussian_medium", rate=0.2, sigma=0.3),
    make_action("mutation", "gaussian_high", rate=0.3, sigma=0.5),
    make_action("mutation", "uniform_low", rate=0.1, distribution="uniform"),
    make_action("mutation", "uniform_high", rate=0.3, distribution="uniform"),
    make_action("mutation", "polynomial", rate=0.2, eta=20),
]

CROSSOVER_ACTIONS = [
    make_action("crossover", "blend_low", alpha=0.3),
    make_action("crossover", "blend_high", alpha=0.5),
    make_action("crossover", "sbx_low", eta=10),
    make_action("crossover", "sbx_high", eta=20),
    make_action("crossover", "de_rand_1", F=0.5, CR=0.9),
    make_action("crossover", "de_best_1", F=0.8, CR=0.9),
    make_action("crossover", "de_current_to_best", F=0.5, CR=0.7),
]

SELECTION_ACTIONS = [
    make_action("selection", "tournament_2", k=2),
    make_action("selection", "tournament_3", k=3),
    make_action("selection", "tournament_5", k=5),
    make_action("selection", "roulette", method="roulette"),
    make_action("selection", "rank", method="rank"),
]

RESTART_ACTIONS = [
    make_action("restart", "partial_small", fraction=0.2),
    make_action("restart", "partial_medium", fraction=0.3),
    make_action("restart", "partial_large", fraction=0.5),
    make_action("restart", "full", fraction=1.0),
]

PARAMETER_ACTIONS = [
    make_action("parameter", "pop_increase", factor=1.5),
    make_action("parameter", "pop_decrease", factor=0.75),
]

# Default action space combining all categories
DEFAULT_ACTION_SPACE: List[Action] = (
    MUTATION_ACTIONS +
    CROSSOVER_ACTIONS +
    SELECTION_ACTIONS +
    RESTART_ACTIONS
)

# Compact action space for faster exploration
COMPACT_ACTION_SPACE: List[Action] = [
    make_action("mutation", "gaussian_low", rate=0.1, sigma=0.1),
    make_action("mutation", "gaussian_high", rate=0.3, sigma=0.5),
    make_action("crossover", "de_rand_1", F=0.5, CR=0.9),
    make_action("crossover", "de_best_1", F=0.8, CR=0.9),
    make_action("selection", "tournament_3", k=3),
    make_action("restart", "partial_medium", fraction=0.3),
]


def get_action_by_name(name: str, action_space: List[Action] = None) -> Action:
    """Get an action by its name."""
    if action_space is None:
        action_space = DEFAULT_ACTION_SPACE
    for action in action_space:
        if action.name == name:
            return action
    raise ValueError(f"Action '{name}' not found in action space")


def action_to_index(action: Action, action_space: List[Action] = None) -> int:
    """Convert action to index in action space."""
    if action_space is None:
        action_space = DEFAULT_ACTION_SPACE
    for i, a in enumerate(action_space):
        if a == action:
            return i
    raise ValueError(f"Action not found in action space: {action}")


def index_to_action(index: int, action_space: List[Action] = None) -> Action:
    """Convert index to action."""
    if action_space is None:
        action_space = DEFAULT_ACTION_SPACE
    if 0 <= index < len(action_space):
        return action_space[index]
    raise IndexError(f"Action index {index} out of range [0, {len(action_space)})")
