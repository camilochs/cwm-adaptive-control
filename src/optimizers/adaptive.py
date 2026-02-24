"""Adaptive optimizer using CWM + MCTS for operator selection."""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from .base import BaseOptimizer, OptimizationResult
from .de import DifferentialEvolution
from ..state import OptState
from ..actions import (
    Action, ActionType, DEFAULT_ACTION_SPACE,
    get_action_by_name,
)
from ..mcts.planner import MCTSPlanner, PlannerConfig, PlanningResult
from ..mcts.node import MCTSNode
from ..cwm.templates.base_cwm import BaseCWM, CWMState, CWMAction, DefaultCWM


def optstate_to_cwmstate(state: OptState) -> CWMState:
    """Convert OptState to CWMState for CWM/MCTS."""
    return CWMState(
        generation=state.generation,
        best_fitness=state.best_fitness,
        mean_fitness=state.mean_fitness,
        diversity=state.diversity,
        improvement_rate=state.improvement_rate,
        stagnation=state.stagnation,
        fitness_std=state.fitness_std,
        population_size=state.population_size,
        dimension=state.dimension,
        feasibility_ratio=state.feasibility_ratio,
        constraint_violation_mean=state.constraint_violation_mean,
        improvement_window=state.improvement_window,
    )


def cwmaction_to_action(cwm_action: CWMAction, action_space: List[Action] = None) -> Action:
    """Convert CWMAction to Action.

    IMPORTANT: Always use the params from the CWMAction, not from the action_space.
    This allows parameter-aware CWMs to specify custom F/CR values.
    """
    # Always create action with CWMAction's params to preserve predicted F/CR
    return Action(
        action_type=ActionType(cwm_action.action_type),
        name=cwm_action.name,
        params=tuple(cwm_action.params.items()),
    )


class AdaptiveOptimizer(BaseOptimizer):
    """Optimizer that uses CWM + MCTS to select operators adaptively."""

    def __init__(
        self,
        cwm: BaseCWM = None,
        planner_config: PlannerConfig = None,
        base_optimizer: BaseOptimizer = None,
        action_space: List[Action] = None,
        planning_interval: int = 1,  # Plan every N generations
        replan_on_stagnation: bool = True,
        population_size: int = 50,
        max_generations: int = 100,
        max_evaluations: Optional[int] = None,
        tol: float = 1e-8,
        seed: Optional[int] = None,
        record_trajectory: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            tol=tol,
            seed=seed,
            record_trajectory=record_trajectory,
        )

        # CWM and MCTS
        self.cwm = cwm or DefaultCWM(max_generations=max_generations)
        self.planner_config = planner_config or PlannerConfig(simulations=100, horizon=5)
        self.planner = MCTSPlanner(self.cwm, self.planner_config)

        # Base optimizer for actual optimization
        self.base_optimizer = base_optimizer or DifferentialEvolution(
            population_size=population_size,
            max_generations=1,  # Will be stepped manually
            seed=seed,
            record_trajectory=False,  # We record at adaptive level
        )

        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.planning_interval = planning_interval
        self.replan_on_stagnation = replan_on_stagnation
        self.verbose = verbose

        # Planning state
        self._mcts_root: Optional[MCTSNode] = None
        self._planned_sequence: List[CWMAction] = []
        self._sequence_idx: int = 0
        self._last_stagnation: int = 0

        # Statistics
        self.planning_history: List[Dict[str, Any]] = []

    def _should_replan(self, state: OptState) -> bool:
        """Determine if we should run MCTS planning."""
        # Plan at intervals
        if state.generation % self.planning_interval == 0:
            return True

        # Replan on significant stagnation increase
        if self.replan_on_stagnation:
            if state.stagnation > self._last_stagnation + 5:
                return True

        # Replan if planned sequence exhausted
        if self._sequence_idx >= len(self._planned_sequence):
            return True

        return False

    def _run_planning(self, state: OptState) -> CWMAction:
        """Run MCTS planning to select action."""
        cwm_state = optstate_to_cwmstate(state)

        # Reuse tree if possible
        if self._mcts_root is not None and self._sequence_idx > 0:
            # Try to find child matching last action
            last_action = self._planned_sequence[self._sequence_idx - 1] if self._planned_sequence else None
            if last_action:
                self._mcts_root = self.planner.replan(
                    self._mcts_root, cwm_state, last_action
                )

        result = self.planner.plan(cwm_state)

        # Store planning result
        self.planning_history.append({
            "generation": state.generation,
            "best_action": result.best_action.name if result.best_action else None,
            "action_visits": result.action_visits,
            "root_value": result.root_value,
            "tree_size": result.tree_size,
        })

        if self.verbose:
            print(f"  MCTS: gen={state.generation}, "
                  f"best_action={result.best_action.name if result.best_action else 'None'}, "
                  f"sims={result.simulations}")

        # Update planned sequence
        self._planned_sequence = result.best_sequence
        self._sequence_idx = 0
        self._last_stagnation = state.stagnation

        return result.best_action

    def _get_action(self, state: OptState) -> Optional[Action]:
        """Get action for current state."""
        if self._should_replan(state):
            cwm_action = self._run_planning(state)
        elif self._sequence_idx < len(self._planned_sequence):
            cwm_action = self._planned_sequence[self._sequence_idx]
        else:
            cwm_action = None

        self._sequence_idx += 1

        if cwm_action is None:
            return None

        return cwmaction_to_action(cwm_action, self.action_space)

    def step(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        action: Optional[Action] = None,
    ) -> OptState:
        """Perform one generation with adaptive operator selection."""
        if self.population is None:
            self.initialize_population(objective, bounds)
            # Initialize base optimizer with our population
            self.base_optimizer.population = self.population.copy()
            self.base_optimizer.fitness = self.fitness.copy()
            self.base_optimizer.generation = 0
            self.base_optimizer.evaluations = self.evaluations
            self.base_optimizer.best_idx = self.best_idx

        # Get action from MCTS if not provided
        if action is None:
            current_state = self.get_current_state()
            action = self._get_action(current_state)

        # Apply action to base optimizer
        if action is not None:
            self.base_optimizer.apply_action(action, objective, bounds)

        # Step base optimizer
        self.base_optimizer.step(objective, bounds, None)  # Action already applied

        # Sync state
        self.population = self.base_optimizer.population.copy()
        self.fitness = self.base_optimizer.fitness.copy()
        self.best_idx = self.base_optimizer.best_idx
        self.evaluations = self.base_optimizer.evaluations
        self.generation += 1

        # Update tracking
        best_fitness = float(self.fitness[self.best_idx])
        self.fitness_history.append(best_fitness)
        self._update_stagnation(best_fitness)

        # Record state with action
        self._record_state(action)

        return self.get_current_state()

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: Tuple[np.ndarray, np.ndarray],
        callback: Optional[Callable[[OptState], Optional[Action]]] = None,
    ) -> OptimizationResult:
        """Run adaptive optimization."""
        self.initialize_population(objective, bounds)

        # Initialize base optimizer
        self.base_optimizer.population = self.population.copy()
        self.base_optimizer.fitness = self.fitness.copy()
        self.base_optimizer.generation = 0
        self.base_optimizer.evaluations = self.evaluations
        self.base_optimizer.best_idx = self.best_idx
        self.base_optimizer._last_best_fitness = self._last_best_fitness
        self.base_optimizer._stagnation_count = self._stagnation_count

        converged = False

        while self.generation < self.max_generations:
            if self.max_evaluations and self.evaluations >= self.max_evaluations:
                break

            # Get action from callback or MCTS
            if callback is not None:
                current_state = self.get_current_state()
                action = callback(current_state)
            else:
                action = None  # Let step() handle MCTS planning

            state = self.step(objective, bounds, action)

            if self.verbose and self.generation % 10 == 0:
                print(f"Gen {self.generation}: best={state.best_fitness:.6f}, "
                      f"stag={state.stagnation}")

            if state.best_fitness <= self.tol:
                converged = True
                break

        return OptimizationResult(
            best_solution=self.population[self.best_idx].copy(),
            best_fitness=float(self.fitness[self.best_idx]),
            history=self.state_history,
            trajectory=self.trajectory,
            evaluations=self.evaluations,
            generations=self.generation,
            converged=converged,
            metadata={
                "planning_history": self.planning_history,
                "planner_config": {
                    "simulations": self.planner_config.simulations,
                    "horizon": self.planner_config.horizon,
                },
            },
        )


