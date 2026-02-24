"""MCTS planner using CWM for simulation."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time
import math

from .node import MCTSNode
from ..cwm.templates.base_cwm import BaseCWM, CWMState, CWMAction


@dataclass
class PlanningResult:
    """Result of MCTS planning."""
    best_action: Optional[CWMAction]
    action_values: Dict[str, float]  # action_name -> value
    action_visits: Dict[str, int]  # action_name -> visits
    root_value: float
    simulations: int
    tree_size: int
    max_depth: int
    planning_time: float
    best_sequence: List[CWMAction] = field(default_factory=list)


@dataclass
class PlannerConfig:
    """Configuration for MCTS planner."""
    simulations: int = 500
    horizon: int = 10
    exploration_constant: float = 1.414
    selection_method: str = "ucb1"  # ucb1, ucb1_tuned, thompson
    final_selection: str = "max_visits"  # max_visits, max_value
    discount: float = 0.99
    use_progressive_widening: bool = False
    widening_alpha: float = 0.5
    widening_beta: float = 0.5


class MCTSPlanner:
    """MCTS planner that uses a CWM for simulation."""

    def __init__(
        self,
        cwm: BaseCWM,
        config: PlannerConfig = None,
    ):
        self.cwm = cwm
        self.config = config or PlannerConfig()

    def plan(
        self,
        initial_state: CWMState,
        callback: Optional[Callable[[int, MCTSNode], None]] = None,
    ) -> PlanningResult:
        """Run MCTS planning from initial state.

        Args:
            initial_state: Current optimization state
            callback: Optional callback(simulation_num, root_node)

        Returns:
            PlanningResult with best action and statistics
        """
        start_time = time.time()

        # Initialize root
        root = MCTSNode(state=initial_state, depth=0)

        for sim in range(self.config.simulations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.terminal and not node.expanded:
                node = self._expand(node)

            # Simulation
            value = self._simulate(node)

            # Backpropagation
            node.backpropagate(value)

            if callback and (sim + 1) % 50 == 0:
                callback(sim + 1, root)

        # Select best action
        best_child = root.best_child(
            method=self.config.final_selection,
            exploration_constant=0,  # No exploration for final selection
        )

        # Gather statistics
        action_values = {}
        action_visits = {}
        for child in root.children:
            if child.action:
                name = child.action.name
                action_values[name] = child.value
                action_visits[name] = child.visits

        # Get best action sequence (greedy from root)
        best_sequence = self._get_best_sequence(root)

        return PlanningResult(
            best_action=best_child.action if best_child else None,
            action_values=action_values,
            action_visits=action_visits,
            root_value=root.value,
            simulations=self.config.simulations,
            tree_size=root.subtree_size(),
            max_depth=root.max_depth(),
            planning_time=time.time() - start_time,
            best_sequence=best_sequence,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB."""
        while not node.is_leaf() and not node.terminal:
            # Progressive widening check
            if self.config.use_progressive_widening:
                num_actions = len(self.cwm.get_legal_actions(node.state))
                max_children = int(
                    self.config.widening_alpha *
                    (node.visits ** self.config.widening_beta)
                )
                if len(node.children) < min(num_actions, max(1, max_children)):
                    # Expand more children
                    break

            node = node.best_child(
                method=self.config.selection_method,
                exploration_constant=self.config.exploration_constant,
            )

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add new children."""
        if node.terminal:
            return node

        legal_actions = self.cwm.get_legal_actions(node.state)
        if not legal_actions:
            node.terminal = True
            return node

        # Find actions not yet tried
        tried_actions = {child.action.name for child in node.children if child.action}
        untried = [a for a in legal_actions if a.name not in tried_actions]

        if not untried:
            node.expanded = True
            # Return random existing child for simulation
            if node.children:
                return np.random.choice(node.children)
            return node

        # Select random untried action
        action = np.random.choice(untried)

        # Predict next state
        next_state = self.cwm.predict_next_state(node.state, action)
        terminal = self.cwm.is_terminal(next_state)

        # Add child
        child = node.add_child(next_state, action, terminal)

        if len(node.children) >= len(legal_actions):
            node.expanded = True

        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase: rollout from node."""
        if node.terminal:
            return self.cwm.evaluate_state(node.state)

        state = node.state
        total_reward = 0.0
        discount = 1.0

        for step in range(self.config.horizon):
            if self.cwm.is_terminal(state):
                break

            # Get legal actions and select randomly
            actions = self.cwm.get_legal_actions(state)
            if not actions:
                break

            action = np.random.choice(actions)

            # Simulate transition
            next_state = self.cwm.predict_next_state(state, action)

            # Accumulate discounted reward
            reward = self.cwm.evaluate_state(next_state)
            total_reward += discount * reward
            discount *= self.config.discount

            state = next_state

        # Add terminal value
        total_reward += discount * self.cwm.evaluate_state(state)

        return total_reward

    def _get_best_sequence(self, root: MCTSNode, depth: int = None) -> List[CWMAction]:
        """Get best action sequence by following highest-value children."""
        if depth is None:
            depth = self.config.horizon

        sequence = []
        node = root

        for _ in range(depth):
            if not node.children:
                break

            best = node.best_child(method="max_value", exploration_constant=0)
            if best is None or best.action is None:
                break

            sequence.append(best.action)
            node = best

        return sequence

    def get_action_distribution(self, root: MCTSNode) -> Dict[str, float]:
        """Get probability distribution over actions based on visits."""
        total_visits = sum(c.visits for c in root.children)
        if total_visits == 0:
            return {}

        return {
            child.action.name: child.visits / total_visits
            for child in root.children
            if child.action
        }

    def replan(
        self,
        root: MCTSNode,
        new_state: CWMState,
        action_taken: CWMAction,
    ) -> MCTSNode:
        """Reuse tree after taking an action (tree reuse).

        Find the child matching the action taken and return it as new root.
        """
        for child in root.children:
            if child.action and child.action.name == action_taken.name:
                # Detach from parent
                child.parent = None
                child.state = new_state  # Update with actual observed state
                return child

        # Action not in tree, create new root
        return MCTSNode(state=new_state, depth=0)


class ParallelMCTSPlanner(MCTSPlanner):
    """MCTS planner with root parallelization."""

    def __init__(
        self,
        cwm: BaseCWM,
        config: PlannerConfig = None,
        n_workers: int = 4,
    ):
        super().__init__(cwm, config)
        self.n_workers = n_workers

    def plan(
        self,
        initial_state: CWMState,
        callback: Optional[Callable[[int, MCTSNode], None]] = None,
    ) -> PlanningResult:
        """Run parallel MCTS planning.

        Uses root parallelization: run multiple trees and aggregate.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Divide simulations among workers
        sims_per_worker = self.config.simulations // self.n_workers

        results = []

        def run_worker(worker_id: int) -> MCTSNode:
            worker_config = PlannerConfig(
                simulations=sims_per_worker,
                horizon=self.config.horizon,
                exploration_constant=self.config.exploration_constant,
                selection_method=self.config.selection_method,
                discount=self.config.discount,
            )
            worker_planner = MCTSPlanner(self.cwm, worker_config)

            # Each worker builds its own tree
            root = MCTSNode(state=initial_state, depth=0)
            for sim in range(sims_per_worker):
                node = worker_planner._select(root)
                if not node.terminal and not node.expanded:
                    node = worker_planner._expand(node)
                value = worker_planner._simulate(node)
                node.backpropagate(value)

            return root

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(run_worker, i) for i in range(self.n_workers)]
            for future in as_completed(futures):
                results.append(future.result())

        # Aggregate results: combine visit counts and values
        combined_stats = {}  # action_name -> (total_visits, total_value)

        for root in results:
            for child in root.children:
                if child.action:
                    name = child.action.name
                    if name not in combined_stats:
                        combined_stats[name] = {"visits": 0, "value": 0.0, "action": child.action}
                    combined_stats[name]["visits"] += child.visits
                    combined_stats[name]["value"] += child.total_value

        # Select best action
        best_action = None
        best_visits = 0

        action_values = {}
        action_visits = {}

        for name, stats in combined_stats.items():
            action_visits[name] = stats["visits"]
            action_values[name] = stats["value"] / stats["visits"] if stats["visits"] > 0 else 0

            if stats["visits"] > best_visits:
                best_visits = stats["visits"]
                best_action = stats["action"]

        total_sims = sum(r.visits for r in results)
        total_size = sum(r.subtree_size() for r in results)

        return PlanningResult(
            best_action=best_action,
            action_values=action_values,
            action_visits=action_visits,
            root_value=sum(r.value for r in results) / len(results),
            simulations=total_sims,
            tree_size=total_size,
            max_depth=max(r.max_depth() for r in results),
            planning_time=0,  # Would need to track actual time
            best_sequence=[best_action] if best_action else [],
        )
