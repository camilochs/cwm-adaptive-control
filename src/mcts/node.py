"""MCTS tree node implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from ..cwm.templates.base_cwm import CWMState, CWMAction


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: CWMState
    action: Optional[CWMAction] = None  # Action that led to this state
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)

    # Statistics
    visits: int = 0
    total_value: float = 0.0
    squared_value: float = 0.0  # For variance computation

    # Metadata
    depth: int = 0
    terminal: bool = False
    expanded: bool = False

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def value_variance(self) -> float:
        """Variance of values at this node."""
        if self.visits < 2:
            return float('inf')
        mean = self.value
        return (self.squared_value / self.visits) - (mean * mean)

    @property
    def value_std(self) -> float:
        """Standard deviation of values."""
        var = self.value_variance
        if var < 0 or math.isinf(var):
            return float('inf')
        return math.sqrt(var)

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for Trees (UCB1).

        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.value

        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def ucb1_tuned(self, exploration_constant: float = 1.414) -> float:
        """UCB1-Tuned: uses variance to adjust exploration.

        UCB1-Tuned = Q + sqrt(ln(N)/n * min(1/4, V + sqrt(2*ln(N)/n)))
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.value

        ln_N = math.log(self.parent.visits)
        n = self.visits

        # Variance bound
        variance_term = self.value_variance + math.sqrt(2 * ln_N / n)
        variance_bound = min(0.25, variance_term)

        exploration = math.sqrt(ln_N / n * variance_bound)
        return self.value + exploration_constant * exploration

    def thompson_sample(self) -> float:
        """Thompson Sampling: sample from posterior distribution.

        Assumes values follow a normal distribution.
        """
        if self.visits == 0:
            # Prior: optimistic
            return np.random.normal(1.0, 1.0)

        mean = self.value
        std = max(self.value_std, 0.01)  # Minimum std for stability

        return np.random.normal(mean, std / math.sqrt(self.visits))

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def is_fully_expanded(self, num_actions: int) -> bool:
        """Check if all actions have been tried."""
        return len(self.children) >= num_actions

    def best_child(
        self,
        method: str = "ucb1",
        exploration_constant: float = 1.414,
    ) -> Optional["MCTSNode"]:
        """Select best child according to selection policy."""
        if not self.children:
            return None

        if method == "ucb1":
            return max(self.children, key=lambda c: c.ucb1(exploration_constant))
        elif method == "ucb1_tuned":
            return max(self.children, key=lambda c: c.ucb1_tuned(exploration_constant))
        elif method == "thompson":
            return max(self.children, key=lambda c: c.thompson_sample())
        elif method == "max_value":
            return max(self.children, key=lambda c: c.value)
        elif method == "max_visits":
            return max(self.children, key=lambda c: c.visits)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def add_child(
        self,
        state: CWMState,
        action: CWMAction,
        terminal: bool = False,
    ) -> "MCTSNode":
        """Add a child node."""
        child = MCTSNode(
            state=state,
            action=action,
            parent=self,
            depth=self.depth + 1,
            terminal=terminal,
        )
        self.children.append(child)
        return child

    def backpropagate(self, value: float):
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            node.squared_value += value * value
            node = node.parent

    def get_path(self) -> List["MCTSNode"]:
        """Get path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_action_sequence(self) -> List[CWMAction]:
        """Get sequence of actions from root to this node."""
        path = self.get_path()
        return [node.action for node in path[1:] if node.action is not None]

    def subtree_size(self) -> int:
        """Count nodes in subtree rooted at this node."""
        count = 1
        for child in self.children:
            count += child.subtree_size()
        return count

    def max_depth(self) -> int:
        """Get maximum depth in subtree."""
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": {
                "generation": self.state.generation,
                "best_fitness": self.state.best_fitness,
                "mean_fitness": self.state.mean_fitness,
                "diversity": self.state.diversity,
                "improvement_rate": self.state.improvement_rate,
                "stagnation": self.state.stagnation,
            },
            "action": {
                "type": self.action.action_type,
                "name": self.action.name,
                "params": self.action.params,
            } if self.action else None,
            "visits": self.visits,
            "value": self.value,
            "depth": self.depth,
            "terminal": self.terminal,
            "n_children": len(self.children),
        }

    def __repr__(self) -> str:
        action_str = f"{self.action.name}" if self.action else "root"
        return (f"MCTSNode({action_str}, "
                f"visits={self.visits}, "
                f"value={self.value:.3f}, "
                f"depth={self.depth})")
