#!/usr/bin/env python3
"""
Experiment 14: NK-Landscape CWM Synthesis

Synthesises a Code World Model for the (1+1)-RLS_k on NK-Landscape using an
LLM.  The CWM must implement predict_next_state, get_legal_actions,
evaluate_state, and is_terminal — using NKState / NKAction dataclasses.

KEY DIFFERENCE from other benchmarks: the synthesis prompt provides NO
mathematical transition formula.  Only a qualitative description:
  - "Fitness is the sum of n epistatic contributions"
  - "Each bit's contribution depends on K neighbour bits"
  - "No closed-form P(improve|fitness, k) exists"
  - "The CWM must infer transition patterns from trajectory data"

The state includes stagnation_counter as an extra signal.

Pipeline:
  1. Load trajectories produced by exp13.
  2. Format representative samples + statistics for the prompt.
  3. Call LLM with NK-specific system prompt (qualitative, not mathematical).
  4. Extract code, validate with NKState/NKAction tests.
  5. If invalid, refine (up to max_attempts).
  6. Save to results/cwm/nk_cwm.py.
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

NK_SYSTEM_PROMPT = """\
You are synthesizing a Code World Model (CWM) for the NK-Landscape problem.

## Problem Definition — NK-Landscape

The NK-Landscape is a tunably rugged fitness function defined on bitstrings
of length n.  Each of n bits contributes to the total fitness based on its
own value AND the values of K neighbouring bits.  The total fitness is the
sum of all n contributions (each contribution in [0, 1]).

The (1+1)-RLS_k algorithm flips exactly k randomly chosen bits in a bitstring
and accepts the offspring if its fitness is >= the parent (non-strict selection).

## Key Properties

1. **Epistatic interactions**: Unlike OneMax or LeadingOnes, flipping a single
   bit affects multiple contributions (its own + those of bits that list it
   as a neighbour).  This creates a rugged landscape with many local optima.

2. **No closed-form transition model**: Unlike OneMax (hypergeometric model)
   or Jump_k (valley dynamics), there is NO mathematical formula for
   P(improve | fitness, k).  The relationship between fitness, k, and
   improvement probability depends on the specific instance and the
   current bitstring configuration.

3. **Stagnation is informative**: When the optimizer has not improved for
   many steps, it is likely stuck in a local optimum.  Increasing k
   (more disruptive mutation) may help escape.

4. **General patterns from data**:
   - Small k (1-3): good for local refinement when improving
   - Medium k (5-10): useful for escaping shallow local optima
   - Large k (>10): essentially random restart, rarely useful
   - As fitness increases, smaller k tends to be more effective
   - After stagnation, temporarily increasing k can help

## Your CWM Specification

Your CWM must implement four methods that operate on NKState and NKAction
dataclasses (already defined in the execution namespace):

  @dataclass
  class NKState:
      fitness: float            # current NK fitness value (sum of contributions)
      n: int                    # bitstring length
      K: int                    # epistasis parameter
      step: int                 # current step counter
      budget: int               # max allowed steps
      normalized_fitness: float # fitness / max_fitness (in [0, 1])
      stagnation_counter: int   # consecutive steps without strict improvement

  @dataclass
  class NKAction:
      k: int                    # number of bits to flip  (1 .. n)
      name: str                 # e.g. "flip_3"

Methods to implement inside class SynthesizedCWM:

  predict_next_state(state: NKState, action: NKAction) -> NKState
      Given current fitness and flip count k, predict the expected next
      state.  Since there is no closed-form model, use empirical patterns:
      - With small k: likely small fitness change (positive or negative)
      - With large k: large disruption, likely fitness decrease
      - Stagnation counter resets on improvement, increases otherwise
      - The predicted fitness should be a reasonable expectation

  get_legal_actions(state: NKState) -> List[NKAction]
      Return a list of candidate flip-count actions.  Should include k=1
      (safe default), k=5, and some larger values for escaping local optima.
      Should consider stagnation: if stagnated, include larger k values.

  evaluate_state(state: NKState) -> float
      Score state quality (higher = better).  Should use normalized_fitness
      and penalize high stagnation.  The continuous normalized_fitness
      allows discriminating between actions with similar integer behaviour.

  is_terminal(state: NKState) -> bool
      True when budget is exhausted.  (For NK-Landscape, reaching the
      global optimum is practically impossible, so budget is the main
      termination condition.)

## CRITICAL: Use normalized_fitness for fine discrimination
The fitness field is a float, but the normalized_fitness stores fitness/max_fitness.
evaluate_state MUST use normalized_fitness so that the planner can rank actions
that differ in expected improvement probability.
"""


NK_SYNTHESIS_PROMPT = """\
Based on the following trajectory data from (1+1)-RLS_k on NK-Landscape,
synthesize a CWM.

## Trajectory samples ({n_samples} transitions)
```
{trajectory_samples}
```

## Statistics from {n_trajectories} trajectories (n={n}, K={K})
- Total transitions: {total_transitions}
- Policies seen: {policies}
- Mean best fitness: {mean_best_fitness:.2f} (max possible ≈ {max_fitness:.2f})
- Mean final fitness: {mean_final_fitness:.2f}
- Fitness range at action time: [{min_fitness:.2f}, {max_fitness_seen:.2f}]
- Most common k values: {common_k}
- Mean fitness improvement per step: {mean_improvement:.4f}

## Key observations from data
- Improvement rate with k=1: {improve_rate_k1:.1%}
- Improvement rate with k=5: {improve_rate_k5:.1%}
- Improvement rate with k>10: {improve_rate_k10:.1%}
- Mean stagnation before improvement: {mean_stagnation:.0f} steps

## Requirements
Generate a Python class `SynthesizedCWM` that implements the four methods
described in the system prompt.  The class must be self-contained (no imports
outside the standard library; math is available).  NKState and NKAction are
already defined in the execution namespace — do NOT redefine them.

Key requirements for NK-Landscape:
- predict_next_state must be based on EMPIRICAL patterns, not a formula
- Higher k should predict more fitness disruption
- stagnation_counter should be tracked: reset on improvement, increment otherwise
- evaluate_state should reward high fitness and penalize stagnation
- get_legal_actions should adapt to stagnation level

Provide ONLY the Python code.

```python
class SynthesizedCWM:
    ...
```
"""


NK_REFINEMENT_PROMPT = """\
The synthesized CWM failed validation.  Please fix the issues.

## Current code
```python
{current_code}
```

## Failures
{failures}

## Reminders
- NKState and NKAction are already defined — do NOT redefine them.
- predict_next_state must return an NKState with step = state.step + 1.
- evaluate_state must return a finite float (no NaN / Inf).
- get_legal_actions must return a non-empty list of NKAction objects.
- is_terminal must return a bool.
- NKState has fields: fitness, n, K, step, budget, normalized_fitness, stagnation_counter.
- For NK-Landscape, there is NO closed-form transition model.
- Use empirical patterns: small k ≈ small change, large k ≈ big disruption.

Provide the complete fixed code.

```python
class SynthesizedCWM:
    ...
```
"""


# ── NKState / NKAction validation ──────────────────────────────────────────

def _make_namespace():
    """Create an execution namespace with NKState and NKAction defined."""
    setup = """\
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

@dataclass
class NKState:
    fitness: float
    n: int
    K: int
    step: int
    budget: int
    normalized_fitness: float
    stagnation_counter: int

@dataclass
class NKAction:
    k: int
    name: str
"""
    ns = {}
    exec(setup, ns)
    return ns


def validate_nk_cwm(code, n=50, K=2, budget=10000, max_fitness=44.0):
    """Validate a synthesised NK-Landscape CWM.

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
        "stagnation_response": False,
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

    NKState = ns["NKState"]
    NKAction = ns["NKAction"]

    # Test state (mid-run, moderate fitness)
    state = NKState(fitness=25.0, n=n, K=K, step=500, budget=budget,
                    normalized_fitness=25.0 / max_fitness,
                    stagnation_counter=10)
    action = NKAction(k=3, name="flip_3")

    # predict_next_state
    try:
        ns_ = cwm.predict_next_state(state, action)
        if not isinstance(ns_, NKState):
            errors.append("predict_next_state: did not return NKState")
        elif ns_.step != state.step + 1:
            errors.append("predict_next_state: step not incremented")
        elif ns_.fitness < 0:
            errors.append(f"predict_next_state: negative fitness {ns_.fitness}")
        else:
            tests["predict_next_state"] = True
    except Exception as e:
        errors.append(f"predict_next_state: {e}")

    # get_legal_actions
    try:
        actions = cwm.get_legal_actions(state)
        if not isinstance(actions, list) or len(actions) == 0:
            errors.append("get_legal_actions: empty or wrong type")
        elif not all(isinstance(a, NKAction) for a in actions):
            errors.append("get_legal_actions: elements are not NKAction")
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

    # Numerical stability — edge cases (fitness near 0)
    edge = NKState(fitness=1.0, n=n, K=K, step=0, budget=budget,
                   normalized_fitness=1.0 / max_fitness,
                   stagnation_counter=0)
    edge_action = NKAction(k=n, name=f"flip_{n}")
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

    # Stagnation response — high stagnation should change behavior
    no_stag = NKState(fitness=30.0, n=n, K=K, step=1000, budget=budget,
                      normalized_fitness=30.0 / max_fitness,
                      stagnation_counter=0)
    high_stag = NKState(fitness=30.0, n=n, K=K, step=1000, budget=budget,
                        normalized_fitness=30.0 / max_fitness,
                        stagnation_counter=200)
    try:
        actions_no_stag = cwm.get_legal_actions(no_stag)
        actions_high_stag = cwm.get_legal_actions(high_stag)
        # High stagnation should include at least some larger k values
        max_k_no_stag = max(a.k for a in actions_no_stag)
        max_k_high_stag = max(a.k for a in actions_high_stag)
        if max_k_high_stag >= max_k_no_stag:
            tests["stagnation_response"] = True
        else:
            errors.append(
                f"stagnation_response: max_k with stagnation ({max_k_high_stag}) "
                f"< max_k without ({max_k_no_stag})")
    except Exception as e:
        errors.append(f"stagnation_response: {e}")

    return all(tests.values()), tests, errors


# ── Trajectory formatting ───────────────────────────────────────────────────

def format_samples(trajectories, max_samples=30):
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
            f"[{pol}] fitness={s['fitness']:.2f}, n={s['n']}, K={s['K']}, "
            f"stag={s['stagnation_counter']}, "
            f"k={a['k']} → fitness={ns['fitness']:.2f}, improved={t['improved']}"
        )
    return "\n".join(lines)


def compute_prompt_stats(data):
    """Extract statistics for the synthesis prompt."""
    trajectories = data["trajectories"]
    n = data["parameters"]["n"]
    K = data["parameters"]["K"]
    max_fitness = data["parameters"]["max_fitness"]
    total_trans = sum(len(t["transitions"]) for t in trajectories)
    policies = sorted(set(t["policy"] for t in trajectories))

    best_fits = [t["best_fitness"] for t in trajectories]
    final_fits = [t["final_fitness"] for t in trajectories]

    # Fitness range, common k, improvement stats
    all_k = []
    all_fitness = []
    improvements = []
    improve_k1 = [0, 0]  # [improved, total]
    improve_k5 = [0, 0]
    improve_k10 = [0, 0]
    stagnation_before_improve = []

    for traj in trajectories:
        for t in traj["transitions"]:
            k = t["action"]["k"]
            all_k.append(k)
            all_fitness.append(t["state"]["fitness"])
            delta = t["next_state"]["fitness"] - t["state"]["fitness"]
            if t["improved"]:
                improvements.append(delta)
                stag = t["state"].get("stagnation_counter", 0)
                stagnation_before_improve.append(stag)
            if k == 1:
                improve_k1[1] += 1
                if t["improved"] and delta > 0:
                    improve_k1[0] += 1
            elif k == 5:
                improve_k5[1] += 1
                if t["improved"] and delta > 0:
                    improve_k5[0] += 1
            elif k > 10:
                improve_k10[1] += 1
                if t["improved"] and delta > 0:
                    improve_k10[0] += 1

    from collections import Counter
    k_counts = Counter(all_k).most_common(8)
    common_k = ", ".join(f"k={k}({c})" for k, c in k_counts)

    import numpy as np
    return {
        "n": n,
        "K": K,
        "max_fitness": max_fitness,
        "n_trajectories": len(trajectories),
        "total_transitions": total_trans,
        "policies": ", ".join(policies),
        "mean_best_fitness": float(np.mean(best_fits)),
        "mean_final_fitness": float(np.mean(final_fits)),
        "min_fitness": float(min(all_fitness)),
        "max_fitness_seen": float(max(all_fitness)),
        "common_k": common_k,
        "mean_improvement": float(np.mean(improvements)) if improvements else 0.0,
        "improve_rate_k1": improve_k1[0] / max(1, improve_k1[1]),
        "improve_rate_k5": improve_k5[0] / max(1, improve_k5[1]),
        "improve_rate_k10": improve_k10[0] / max(1, improve_k10[1]),
        "mean_stagnation": float(np.mean(stagnation_before_improve)) if stagnation_before_improve else 0.0,
    }


import numpy as np  # noqa: E402


# ── Synthesis loop ──────────────────────────────────────────────────────────

def synthesize(data, model, max_attempts=5, verbose=True):
    """Synthesise + validate an NK-Landscape CWM.

    Returns (code, n_attempts) on success, raises RuntimeError on failure.
    """
    stats = compute_prompt_stats(data)
    samples_text = format_samples(data["trajectories"], max_samples=30)
    n = stats["n"]
    K = stats["K"]
    max_fitness = stats["max_fitness"]
    budget = data["parameters"]["budget"]

    prompt = NK_SYNTHESIS_PROMPT.format(
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
            system=NK_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        code = extract_code_from_response(response)

        if verbose:
            print(f"    LLM responded in {elapsed:.1f}s  ({len(code)} chars)")

        valid, tests, errors = validate_nk_cwm(
            code, n=n, K=K, budget=budget, max_fitness=max_fitness)
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

        prompt = NK_REFINEMENT_PROMPT.format(
            current_code=code,
            failures=failures_text,
        )

    # Return best effort even if not fully valid
    if best_code is not None:
        if verbose:
            print(f"\n  WARNING: returning best-effort CWM "
                  f"({best_pass_count}/{len(tests)} tests)")
        return best_code, max_attempts

    raise RuntimeError("CWM synthesis failed after all attempts")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp14: Synthesize NK-Landscape CWM")
    parser.add_argument("--trajectories", type=str,
                        default="results/trajectories/nk_trajectories.json",
                        help="Path to trajectory JSON from exp13")
    parser.add_argument("--output", type=str,
                        default="results/cwm/nk_cwm.py",
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
    print("Experiment 14: NK-Landscape CWM Synthesis")
    print("=" * 60)

    if args.validate_only:
        print(f"\nValidating existing CWM: {args.output}")
        with open(args.output) as f:
            code = f.read()
        valid, tests, errors = validate_nk_cwm(code)
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
        "K": data["parameters"]["K"],
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Final validation
    print("\nFinal validation:")
    valid, tests, errors = validate_nk_cwm(
        code, n=data["parameters"]["n"], K=data["parameters"]["K"])
    for tname, ok in tests.items():
        print(f"  {tname}: {'OK' if ok else 'FAIL'}")
    print(f"\nValid: {valid}")


if __name__ == "__main__":
    main()
