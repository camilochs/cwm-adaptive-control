"""Prompt templates for CWM synthesis."""

SYSTEM_PROMPT = """You are an expert in optimization algorithms and machine learning. Your task is to synthesize a Code World Model (CWM) that predicts how an optimization process evolves.

A CWM is Python code that:
1. Predicts the next optimization state given current state and action
2. Determines which actions are legal in a given state
3. Checks if optimization should terminate
4. Evaluates state quality for planning

You will be given:
- Trajectory data showing (state, action, next_state) transitions
- The state and action schemas
- Validation tests your code must pass

Generate clean, efficient Python code that captures the patterns in the data."""


STATE_SCHEMA = """
@dataclass
class CWMState:
    generation: int           # Current generation number
    best_fitness: float       # Best fitness found (lower is better)
    mean_fitness: float       # Mean population fitness
    diversity: float          # Population diversity (0-1, higher is more diverse)
    improvement_rate: float   # Recent improvement rate (positive means improving)
    stagnation: int          # Generations without improvement
    fitness_std: float       # Standard deviation of fitness
    population_size: int     # Population size
    dimension: int           # Problem dimension
"""


ACTION_SCHEMA = """
@dataclass
class CWMAction:
    action_type: str  # One of: "mutation", "crossover", "selection", "restart"
    name: str         # Specific action name (e.g., "gaussian_low", "de_rand_1")
    params: dict      # Action parameters

Common actions:
- mutation: {"rate": 0.1-0.5, "sigma": 0.1-0.5} or {"eta": 10-30}
- crossover: {"F": 0.5-0.9, "CR": 0.7-0.9} or {"alpha": 0.3-0.5}
- selection: {"k": 2-5} for tournament
- restart: {"fraction": 0.2-0.5} partial restart
"""


SYNTHESIS_PROMPT_TEMPLATE = """Based on the following optimization trajectory data, synthesize a CWM that predicts state transitions.

## State Schema
{state_schema}

## Action Schema
{action_schema}

## Trajectory Data (sample of {n_samples} transitions)
```
{trajectory_samples}
```

## Statistics from data
- Total transitions: {total_transitions}
- Unique actions seen: {unique_actions}
- Fitness range: [{min_fitness:.6f}, {max_fitness:.6f}]
- Average improvement per step: {avg_improvement:.6f}
- Convergence rate: {convergence_rate:.1%}

## Requirements
Generate a Python class `SynthesizedCWM` that implements:

1. `predict_next_state(state: CWMState, action: CWMAction) -> CWMState`
   - Predict how state changes when action is applied
   - Model relationships observed in the data
   - Handle different action types appropriately

2. `get_legal_actions(state: CWMState) -> List[CWMAction]`
   - Return actions that make sense for current state
   - Consider stagnation, diversity, generation

3. `is_terminal(state: CWMState) -> bool`
   - True if optimization should stop
   - Consider fitness threshold, max generations, stagnation

4. `evaluate_state(state: CWMState) -> float`
   - Higher is better
   - Consider fitness, improvement trend, diversity

## Output Format
Provide ONLY the Python code for the SynthesizedCWM class, with necessary imports.
The code should be self-contained and ready to execute.

```python
# Your code here
```
"""


REFINEMENT_PROMPT_TEMPLATE = """The synthesized CWM failed some validation tests. Please fix the issues.

## Current Code
```python
{current_code}
```

## Test Failures
{test_failures}

## Original Data Summary
- Fitness range: [{min_fitness:.6f}, {max_fitness:.6f}]
- Average improvement: {avg_improvement:.6f}
- Most common actions: {common_actions}

## Instructions
1. Analyze why each test failed
2. Fix the issues while preserving correct behavior
3. Ensure numerical stability (avoid division by zero, log of negative)
4. Return the complete fixed code

```python
# Fixed code here
```
"""


UNIT_TEST_TEMPLATE = """
import math
from dataclasses import dataclass
from typing import List, Dict, Any

{cwm_code}

def test_predict_next_state():
    cwm = SynthesizedCWM()
    state = CWMState(
        generation=10,
        best_fitness=1.0,
        mean_fitness=5.0,
        diversity=0.5,
        improvement_rate=0.1,
        stagnation=2,
        fitness_std=1.0,
        population_size=50,
        dimension=10,
    )
    action = CWMAction(
        action_type="mutation",
        name="gaussian_low",
        params={{"rate": 0.1, "sigma": 0.1}},
    )
    next_state = cwm.predict_next_state(state, action)

    # Basic sanity checks
    assert isinstance(next_state, CWMState)
    assert next_state.generation == state.generation + 1
    assert next_state.best_fitness >= 0
    assert next_state.diversity >= 0
    assert isinstance(next_state.stagnation, int)

def test_get_legal_actions():
    cwm = SynthesizedCWM()
    state = CWMState(
        generation=10,
        best_fitness=1.0,
        mean_fitness=5.0,
        diversity=0.5,
        improvement_rate=0.1,
        stagnation=2,
        fitness_std=1.0,
        population_size=50,
        dimension=10,
    )
    actions = cwm.get_legal_actions(state)

    assert isinstance(actions, list)
    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, CWMAction)

def test_is_terminal():
    cwm = SynthesizedCWM()

    # Not terminal: good fitness, not converged
    state1 = CWMState(
        generation=10, best_fitness=1.0, mean_fitness=5.0,
        diversity=0.5, improvement_rate=0.1, stagnation=2,
        fitness_std=1.0, population_size=50, dimension=10,
    )
    assert cwm.is_terminal(state1) == False

    # Terminal: very low fitness
    state2 = CWMState(
        generation=10, best_fitness=1e-10, mean_fitness=1e-8,
        diversity=0.5, improvement_rate=0.0, stagnation=0,
        fitness_std=1e-9, population_size=50, dimension=10,
    )
    assert cwm.is_terminal(state2) == True

def test_evaluate_state():
    cwm = SynthesizedCWM()

    # Better state (lower fitness)
    good_state = CWMState(
        generation=10, best_fitness=0.001, mean_fitness=0.01,
        diversity=0.5, improvement_rate=0.1, stagnation=0,
        fitness_std=0.005, population_size=50, dimension=10,
    )

    # Worse state (higher fitness)
    bad_state = CWMState(
        generation=10, best_fitness=100.0, mean_fitness=200.0,
        diversity=0.5, improvement_rate=0.0, stagnation=10,
        fitness_std=50.0, population_size=50, dimension=10,
    )

    good_score = cwm.evaluate_state(good_state)
    bad_score = cwm.evaluate_state(bad_state)

    assert good_score > bad_score, "Better state should have higher evaluation"

def test_numerical_stability():
    cwm = SynthesizedCWM()

    # Edge case: zero fitness
    state = CWMState(
        generation=1, best_fitness=0.0, mean_fitness=0.0,
        diversity=0.0, improvement_rate=0.0, stagnation=100,
        fitness_std=0.0, population_size=50, dimension=10,
    )
    action = CWMAction("mutation", "test", {{"rate": 0.1}})

    # Should not raise exceptions
    try:
        next_state = cwm.predict_next_state(state, action)
        score = cwm.evaluate_state(state)
        terminal = cwm.is_terminal(state)
        assert not math.isnan(score) and not math.isinf(score)
    except Exception as e:
        raise AssertionError(f"Numerical instability: {{e}}")

if __name__ == "__main__":
    test_predict_next_state()
    test_get_legal_actions()
    test_is_terminal()
    test_evaluate_state()
    test_numerical_stability()
    print("All tests passed!")
"""


def format_trajectory_sample(transitions, max_samples=10):
    """Format trajectory transitions for prompt."""
    import random
    samples = random.sample(transitions, min(len(transitions), max_samples))

    formatted = []
    for state, action, next_state in samples:
        formatted.append(f"""
State: gen={state['generation']}, best={state['best_fitness']:.4f}, mean={state['mean_fitness']:.4f}, div={state['diversity']:.4f}, stag={state['stagnation']}
Action: {action['action_type'] if action else 'None'}:{action['name'] if action else 'None'} {action['params'] if action else ''}
Next:  gen={next_state['generation']}, best={next_state['best_fitness']:.4f}, mean={next_state['mean_fitness']:.4f}, div={next_state['diversity']:.4f}, stag={next_state['stagnation']}
---""")

    return "\n".join(formatted)


def create_synthesis_prompt(trajectories, max_samples=15):
    """Create the full synthesis prompt from trajectory data."""
    # Collect all transitions
    all_transitions = []
    for traj in trajectories:
        all_transitions.extend(traj.transitions)

    # Compute statistics
    fitnesses = [t[0]["best_fitness"] for t in all_transitions]
    improvements = []
    for state, action, next_state in all_transitions:
        improvements.append(state["best_fitness"] - next_state["best_fitness"])

    unique_actions = set()
    for _, action, _ in all_transitions:
        if action:
            unique_actions.add(action["name"])

    convergence_rate = sum(1 for t in trajectories if t.converged) / len(trajectories)

    return SYNTHESIS_PROMPT_TEMPLATE.format(
        state_schema=STATE_SCHEMA,
        action_schema=ACTION_SCHEMA,
        trajectory_samples=format_trajectory_sample(all_transitions, max_samples),
        n_samples=min(len(all_transitions), max_samples),
        total_transitions=len(all_transitions),
        unique_actions=", ".join(sorted(unique_actions)),
        min_fitness=min(fitnesses),
        max_fitness=max(fitnesses),
        avg_improvement=sum(improvements) / len(improvements) if improvements else 0,
        convergence_rate=convergence_rate,
    )
