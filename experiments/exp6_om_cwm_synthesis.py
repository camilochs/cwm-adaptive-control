#!/usr/bin/env python3
"""
Experiment 6: OneMax CWM Synthesis

Synthesises a Code World Model for the (1+1)-RLS_k on OneMax using an
LLM.  The CWM must implement predict_next_state, get_legal_actions,
evaluate_state, and is_terminal — using OMState / OMAction dataclasses.

Pipeline:
  1. Load trajectories produced by exp5.
  2. Format representative samples + statistics for the prompt.
  3. Call LLM (via LLMClient) with a OneMax-specific system prompt.
  4. Extract code, validate with OMState/OMAction tests.
  5. If invalid, refine (up to max_attempts).
  6. Save to results/cwm/onemax_cwm.py.
"""

import argparse
import ast
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cwm.synthesizer import LLMClient
from src.cwm.validator import extract_code_from_response


# ── Prompts ─────────────────────────────────────────────────────────────────

OM_SYSTEM_PROMPT = """\
You are synthesizing a Code World Model (CWM) for the OneMax problem.

The (1+1)-RLS_k algorithm flips exactly k randomly chosen bits in a bitstring
of length n.  The offspring replaces the parent if its OneMax fitness
(count of ALL 1s in the bitstring) is >= the parent's (non-strict selection).

Your CWM must implement four methods that operate on OMState and OMAction
dataclasses (already defined in the execution namespace):

  @dataclass
  class OMState:
      fitness: int          # current OneMax value  (0 .. n)
      n: int                # bitstring length
      step: int             # current step counter
      budget: int           # max allowed steps
      normalized_fitness: float   # continuous expected fitness / n

  @dataclass
  class OMAction:
      k: int                # number of bits to flip  (1 .. n)
      name: str             # e.g. "flip_8"

Methods to implement inside class SynthesizedCWM:

  predict_next_state(state: OMState, action: OMAction) -> OMState
      Given current fitness i and flip count k, predict the expected next
      fitness and updated step count.

  get_legal_actions(state: OMState) -> List[OMAction]
      Return a list of reasonable flip-count actions for the current state.
      Must return OMAction objects.

  evaluate_state(state: OMState) -> float
      Score state quality (higher = better).

  is_terminal(state: OMState) -> bool
      True when the optimum is reached or the budget is exhausted.

Key insight — Hypergeometric transition model:
When flipping k bits from a bitstring with OneMax fitness i (i ones, n-i zeros):
  - j of the k flipped bits are 1-bits (flipped to 0)
  - k-j of the k flipped bits are 0-bits (flipped to 1)
  - j ~ Hypergeometric(N=n, K=i, n_draw=k)
  - New fitness = i + k - 2j
  - Offspring accepted iff k - 2j >= 0 (i.e. j <= k/2)

Approximate formula for expected accepted fitness:
  E[new_fitness | accepted] ≈ i + k*(n - 2i)/n   (when i < n/2)
This captures the key structure: large k helps when i < n/2, but k=1 is
best when i > n/2 (the "cliff" transition at i ≈ n/2).

The theoretically optimal policy has a sharp transition:
  - For i < n/2: large k (approximately n - i)
  - For i ≈ n/2: transition region
  - For i > n/2: k = 1

CRITICAL for predict_next_state:
- The fitness field must remain an integer (floor of expected value).
- The normalized_fitness field must store the CONTINUOUS expected fitness
  divided by n, i.e.  (expected_new_fitness) / n.  This allows evaluate_state
  to distinguish between actions with different improvement probabilities,
  even when the integer fitness stays the same.
- evaluate_state MUST use normalized_fitness (not fitness) so that it can
  rank actions by their expected improvement probability.
"""


OM_SYNTHESIS_PROMPT = """\
Based on the following trajectory data from (1+1)-RLS_k on OneMax,
synthesize a CWM.

## Trajectory samples ({n_samples} transitions)
```
{trajectory_samples}
```

## Statistics from {n_trajectories} trajectories (n={n})
- Total transitions: {total_transitions}
- Policies seen: {policies}
- Convergence rate: {convergence_rate:.1%}
- Mean steps to optimum (converged runs): {mean_steps:.0f}
- Fitness range at action time: [{min_fitness}, {max_fitness}]
- Most common k values: {common_k}

## Requirements
Generate a Python class `SynthesizedCWM` that implements the four methods
described in the system prompt.  The class must be self-contained (no imports
outside the standard library; math is available).  OMState and OMAction are
already defined in the execution namespace — do NOT redefine them.

Provide ONLY the Python code.

```python
class SynthesizedCWM:
    ...
```
"""


OM_REFINEMENT_PROMPT = """\
The synthesized CWM failed validation.  Please fix the issues.

## Current code
```python
{current_code}
```

## Failures
{failures}

## Reminders
- OMState and OMAction are already defined — do NOT redefine them.
- predict_next_state must return an OMState with step = state.step + 1.
- evaluate_state must return a finite float (no NaN / Inf).
- get_legal_actions must return a non-empty list of OMAction objects.
- is_terminal must return a bool.

Provide the complete fixed code.

```python
class SynthesizedCWM:
    ...
```
"""


# ── OMState / OMAction validation ───────────────────────────────────────────

def _make_namespace():
    """Create an execution namespace with OMState and OMAction defined."""
    setup = """\
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

@dataclass
class OMState:
    fitness: int
    n: int
    step: int
    budget: int
    normalized_fitness: float

@dataclass
class OMAction:
    k: int
    name: str
"""
    ns = {}
    exec(setup, ns)
    return ns


def validate_om_cwm(code: str, n: int = 50, budget: int = 978):
    """Validate a synthesised OneMax CWM.

    Returns (valid: bool, tests_passed: dict, errors: list[str]).
    """
    errors = []
    tests = {
        "syntax": False,
        "exec": False,
        "instantiate": False,
        "predict_next_state": False,
        "get_legal_actions": False,
        "evaluate_state": False,
        "is_terminal": False,
        "numerical_stability": False,
    }

    # Syntax
    try:
        ast.parse(code)
        tests["syntax"] = True
    except SyntaxError as e:
        errors.append(f"SyntaxError line {e.lineno}: {e.msg}")
        return False, tests, errors

    # Execution
    ns = _make_namespace()
    try:
        exec(code, ns)
        tests["exec"] = True
    except Exception as e:
        errors.append(f"Execution error: {e}")
        return False, tests, errors

    if "SynthesizedCWM" not in ns:
        errors.append("SynthesizedCWM class not found")
        return False, tests, errors

    try:
        cwm = ns["SynthesizedCWM"]()
        tests["instantiate"] = True
    except Exception as e:
        errors.append(f"Instantiation failed: {e}")
        return False, tests, errors

    OMState = ns["OMState"]
    OMAction = ns["OMAction"]

    # Test state & action
    state = OMState(fitness=10, n=n, step=50, budget=budget,
                    normalized_fitness=10 / n)
    action = OMAction(k=5, name="flip_5")

    # predict_next_state
    try:
        ns_ = cwm.predict_next_state(state, action)
        if not isinstance(ns_, OMState):
            errors.append("predict_next_state: did not return OMState")
        elif ns_.step != state.step + 1:
            errors.append("predict_next_state: step not incremented")
        elif ns_.fitness < 0 or ns_.fitness > n:
            errors.append(f"predict_next_state: fitness {ns_.fitness} out of [0,{n}]")
        else:
            tests["predict_next_state"] = True
    except Exception as e:
        errors.append(f"predict_next_state: {e}")

    # get_legal_actions
    try:
        actions = cwm.get_legal_actions(state)
        if not isinstance(actions, list) or len(actions) == 0:
            errors.append("get_legal_actions: empty or wrong type")
        elif not all(isinstance(a, OMAction) for a in actions):
            errors.append("get_legal_actions: elements are not OMAction")
        else:
            tests["get_legal_actions"] = True
    except Exception as e:
        errors.append(f"get_legal_actions: {e}")

    # evaluate_state
    try:
        score = cwm.evaluate_state(state)
        if not isinstance(score, (int, float)):
            errors.append("evaluate_state: wrong return type")
        elif math.isnan(score) or math.isinf(score):
            errors.append("evaluate_state: NaN or Inf")
        else:
            tests["evaluate_state"] = True
    except Exception as e:
        errors.append(f"evaluate_state: {e}")

    # is_terminal
    try:
        term = cwm.is_terminal(state)
        if not isinstance(term, bool):
            errors.append("is_terminal: wrong return type")
        else:
            tests["is_terminal"] = True
    except Exception as e:
        errors.append(f"is_terminal: {e}")

    # Numerical stability — edge cases
    edge = OMState(fitness=0, n=n, step=0, budget=budget, normalized_fitness=0.0)
    edge_action = OMAction(k=n, name=f"flip_{n}")
    try:
        ns2 = cwm.predict_next_state(edge, edge_action)
        s2 = cwm.evaluate_state(edge)
        t2 = cwm.is_terminal(edge)
        if math.isnan(s2) or math.isinf(s2):
            errors.append("numerical_stability: NaN/Inf on edge case")
        else:
            tests["numerical_stability"] = True
    except Exception as e:
        errors.append(f"numerical_stability: {e}")

    return all(tests.values()), tests, errors


# ── Trajectory formatting ───────────────────────────────────────────────────

def format_samples(trajectories, max_samples: int = 30):
    """Pick representative transitions and format them for the prompt."""
    import random

    all_trans = []
    for traj in trajectories:
        for t in traj["transitions"]:
            all_trans.append((t, traj["policy"]))

    samples = random.sample(all_trans, min(len(all_trans), max_samples))

    lines = []
    for t, pol in samples:
        s = t["state"]
        a = t["action"]
        ns = t["next_state"]
        lines.append(
            f"[{pol}] fitness={s['fitness']}, k={a['k']} → "
            f"fitness={ns['fitness']}, improved={t['improved']}"
        )
    return "\n".join(lines)


def compute_prompt_stats(data):
    """Extract statistics for the synthesis prompt."""
    trajectories = data["trajectories"]
    n = data["parameters"]["n"]
    total_trans = sum(len(t["transitions"]) for t in trajectories)
    policies = sorted(set(t["policy"] for t in trajectories))

    converged = [t for t in trajectories if t["converged"]]
    convergence_rate = len(converged) / len(trajectories)
    mean_steps = (
        float(np.mean([t["total_steps"] for t in converged]))
        if converged else float("inf")
    )

    # Fitness range & common k
    all_k = []
    all_fitness = []
    for traj in trajectories:
        for t in traj["transitions"]:
            all_k.append(t["action"]["k"])
            all_fitness.append(t["state"]["fitness"])

    from collections import Counter
    k_counts = Counter(all_k).most_common(8)
    common_k = ", ".join(f"k={k}({c})" for k, c in k_counts)

    return {
        "n": n,
        "n_trajectories": len(trajectories),
        "total_transitions": total_trans,
        "policies": ", ".join(policies),
        "convergence_rate": convergence_rate,
        "mean_steps": mean_steps,
        "min_fitness": min(all_fitness),
        "max_fitness": max(all_fitness),
        "common_k": common_k,
    }


# We need numpy only for the stats helper; import at module level.
import numpy as np  # noqa: E402


# ── Synthesis loop ──────────────────────────────────────────────────────────

def synthesize(data, model: str, max_attempts: int = 5, verbose: bool = True):
    """Synthesise + validate a OneMax CWM.

    Returns (code, n_attempts) on success, raises RuntimeError on failure.
    """
    stats = compute_prompt_stats(data)
    samples_text = format_samples(data["trajectories"], max_samples=30)
    n = stats["n"]
    budget = data["parameters"]["budget"]

    prompt = OM_SYNTHESIS_PROMPT.format(
        trajectory_samples=samples_text,
        n_samples=min(30, stats["total_transitions"]),
        **stats,
    )

    # Determine provider
    if "claude" in model.lower():
        provider = "anthropic"
    elif "gpt" in model.lower():
        provider = "openai"
    else:
        provider = "ollama"

    client = LLMClient(provider=provider, model=model)

    best_code = None
    best_pass_count = -1

    for attempt in range(max_attempts):
        if verbose:
            print(f"\n  Attempt {attempt + 1}/{max_attempts} ...")

        t0 = time.time()
        response = client.generate(
            prompt=prompt,
            system=OM_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        code = extract_code_from_response(response)

        if verbose:
            print(f"    LLM responded in {elapsed:.1f}s  ({len(code)} chars)")

        valid, tests, errors = validate_om_cwm(code, n=n, budget=budget)
        pass_count = sum(tests.values())

        if verbose:
            for tname, passed in tests.items():
                mark = "OK" if passed else "FAIL"
                print(f"    {tname}: {mark}")
            if errors:
                for e in errors:
                    print(f"    ! {e}")

        if valid:
            if verbose:
                print("    => CWM is valid!")
            return code, attempt + 1

        if pass_count > best_pass_count:
            best_code = code
            best_pass_count = pass_count

        # Build refinement prompt for next attempt
        failures_text = "\n".join(
            f"- {tname}: {'PASS' if ok else 'FAIL'}"
            for tname, ok in tests.items()
        )
        if errors:
            failures_text += "\n" + "\n".join(f"- Error: {e}" for e in errors)

        prompt = OM_REFINEMENT_PROMPT.format(
            current_code=code,
            failures=failures_text,
        )

    # Return best effort even if not fully valid
    if best_code is not None:
        if verbose:
            print(f"\n  WARNING: returning best-effort CWM ({best_pass_count}/{len(tests)} tests)")
        return best_code, max_attempts

    raise RuntimeError("CWM synthesis failed after all attempts")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp33: Synthesize OneMax CWM")
    parser.add_argument("--trajectories", type=str,
                        default="results/trajectories/om_trajectories.json",
                        help="Path to trajectory JSON from exp5")
    parser.add_argument("--output", type=str,
                        default="results/cwm/onemax_cwm.py",
                        help="Output path for synthesized CWM code")
    parser.add_argument("--model", type=str,
                        default="claude-sonnet-4-20250514",
                        help="LLM model to use")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max synthesis + refinement attempts")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate an existing CWM file")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 6: OneMax CWM Synthesis")
    print("=" * 60)

    if args.validate_only:
        print(f"\nValidating existing CWM: {args.output}")
        with open(args.output) as f:
            code = f.read()
        valid, tests, errors = validate_om_cwm(code)
        for tname, ok in tests.items():
            print(f"  {tname}: {'OK' if ok else 'FAIL'}")
        if errors:
            for e in errors:
                print(f"  ! {e}")
        print(f"\nValid: {valid}")
        return

    # Load trajectories
    print(f"\nLoading trajectories: {args.trajectories}")
    with open(args.trajectories) as f:
        data = json.load(f)
    print(f"  {len(data['trajectories'])} trajectories loaded")

    # Synthesize
    code, n_attempts = synthesize(
        data, model=args.model, max_attempts=args.max_attempts
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(code)
    print(f"\nCWM saved to: {output_path}")

    # Save metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "trajectories": args.trajectories,
        "attempts": n_attempts,
        "n": data["parameters"]["n"],
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Final validation
    print("\nFinal validation:")
    valid, tests, errors = validate_om_cwm(code, n=data["parameters"]["n"])
    for tname, ok in tests.items():
        print(f"  {tname}: {'OK' if ok else 'FAIL'}")
    print(f"\nValid: {valid}")


if __name__ == "__main__":
    main()
