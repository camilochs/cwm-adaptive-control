#!/usr/bin/env python3
"""
Experiment 9: Jump_k CWM Synthesis

Synthesises a Code World Model for the (1+1)-RLS_k on Jump_k using an
LLM.  The CWM must implement predict_next_state, get_legal_actions,
evaluate_state, and is_terminal — using JumpState / JumpAction dataclasses.

Pipeline:
  1. Load trajectories produced by exp8.
  2. Format representative samples + statistics for the prompt.
  3. Call LLM (via LLMClient) with a Jump-specific system prompt.
  4. Extract code, validate with JumpState/JumpAction tests.
  5. If invalid, refine (up to max_attempts).
  6. Save to results/cwm/jump_cwm.py.
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

JUMP_SYSTEM_PROMPT = """\
You are synthesizing a Code World Model (CWM) for the Jump_k problem.

## Problem Definition — Jump_k

The (1+1)-RLS_k algorithm flips exactly k randomly chosen bits in a bitstring
of length n.  The offspring replaces the parent if its Jump_k fitness
is >= the parent's (non-strict selection).

The Jump_k fitness function has THREE regions:

  Jump_k(x) = { k + |x|_1,   if |x|_1 <= n-k  OR  x = 1^n     (NORMAL)
              { n - |x|_1,    otherwise                           (VALLEY)

  - NORMAL region (|x|_1 in [0, n-k]):
      fitness = jump_k + ones.  Behaves like OneMax with an offset.
      fitness ranges from jump_k (ones=0) to n (ones=n-k, the valley EDGE).

  - VALLEY region (|x|_1 in (n-k, n)):
      fitness = n - ones.  Fitness DROPS as ones increase!
      This is a LOCAL OPTIMUM TRAP.  fitness ranges from 1 to jump_k - 1.
      A bitstring with ones = n-1 has fitness = 1 (worst in valley).

  - OPTIMUM (x = 1^n):
      fitness = jump_k + n.  The global maximum.

## Key Insight — Valley Dynamics

When the optimizer reaches fitness = n (the valley EDGE, ones = n - jump_k),
it looks close to the optimum but is actually TRAPPED.  The only way to
improve is to flip exactly the remaining jump_k zero-bits to ones in a
single step.  This requires k = jump_k and all k flipped bits must be zeros.

CRITICAL: fitness near n is a TRAP, not near the optimum!  The optimal
strategy is:
  - In the normal region (far from edge): k=1 works well (like OneMax)
  - Near/at the valley edge (fitness ≈ n): k = jump_k to jump across
  - In the valley (fitness < jump_k): k = jump_k to escape

This is counter-intuitive: when stuck at high fitness, INCREASE k.

## Your CWM Specification

Your CWM must implement four methods that operate on JumpState and JumpAction
dataclasses (already defined in the execution namespace):

  @dataclass
  class JumpState:
      fitness: int              # current Jump_k value
      n: int                    # bitstring length
      jump_k: int               # gap parameter
      step: int                 # current step counter
      budget: int               # max allowed steps
      normalized_fitness: float # continuous expected fitness / (n + jump_k)

  @dataclass
  class JumpAction:
      k: int                    # number of bits to flip  (1 .. n)
      name: str                 # e.g. "flip_8"

Methods to implement inside class SynthesizedCWM:

  predict_next_state(state: JumpState, action: JumpAction) -> JumpState
      Given current fitness and flip count k, predict the expected next
      fitness and updated step count.  Must account for valley dynamics:
      - In normal region: hypergeometric model (like OneMax)
      - Near valley edge: only k=jump_k can improve (with tiny probability)
      - In valley: try to escape

  get_legal_actions(state: JumpState) -> List[JumpAction]
      Return a list of reasonable flip-count actions for the current state.
      Must return JumpAction objects.  MUST include k=jump_k as a candidate.

  evaluate_state(state: JumpState) -> float
      Score state quality (higher = better).  Must use normalized_fitness.

  is_terminal(state: JumpState) -> bool
      True when the optimum (fitness = n + jump_k) is reached or budget exhausted.

## Transition Model Details

Normal region (fitness in [jump_k, n], ones = fitness - jump_k):
  When flipping k bits:
  - j of the k bits are 1-bits (flipped to 0)
  - k-j are 0-bits (flipped to 1)
  - j ~ Hypergeometric(N=n, K=ones, n_draw=k)
  - New ones = ones + k - 2j
  - If new_ones <= n-jump_k: new fitness = jump_k + new_ones (still normal)
  - If new_ones == n: new fitness = n + jump_k (OPTIMUM!)
  - If n-jump_k < new_ones < n: new fitness = n - new_ones (fell into VALLEY)
  - Offspring accepted iff new_fitness >= current_fitness

At valley edge (fitness = n, ones = n - jump_k, zeros = jump_k):
  - k = jump_k: probability of reaching optimum = C(jump_k, jump_k)/C(n, jump_k) = 1/C(n, jump_k)
  - k < jump_k: CANNOT reach optimum (not enough bits flipped)
  - k > jump_k: probability drops (must flip some 1-bits too)

CRITICAL for predict_next_state:
- The fitness field must remain an integer (floor of expected value).
- The normalized_fitness field must store the CONTINUOUS expected fitness
  divided by (n + jump_k).  This allows evaluate_state to distinguish
  between actions that differ in improvement probability.
- evaluate_state MUST use normalized_fitness (not fitness) so that it can
  rank actions correctly, especially at the valley edge where the difference
  between k=1 (useless) and k=jump_k (can escape) is crucial.
"""


JUMP_SYNTHESIS_PROMPT = """\
Based on the following trajectory data from (1+1)-RLS_k on Jump_k,
synthesize a CWM.

## Trajectory samples ({n_samples} transitions)
```
{trajectory_samples}
```

## Statistics from {n_trajectories} trajectories (n={n}, jump_k={jump_k})
- Total transitions: {total_transitions}
- Policies seen: {policies}
- Convergence rate: {convergence_rate:.1%}
- Mean steps to optimum (converged runs): {mean_steps:.0f}
- Fitness range at action time: [{min_fitness}, {max_fitness}]
- Most common k values: {common_k}

## Requirements
Generate a Python class `SynthesizedCWM` that implements the four methods
described in the system prompt.  The class must be self-contained (no imports
outside the standard library; math is available).  JumpState and JumpAction are
already defined in the execution namespace — do NOT redefine them.

Key requirements for Jump_k:
- predict_next_state must handle ALL THREE regions (normal, valley, optimum)
- At the valley edge (fitness = n), k = jump_k should show a small but
  positive improvement (probability 1/C(n, jump_k)), while k != jump_k
  should show zero or negative expected change
- get_legal_actions MUST include k = state.jump_k as a candidate action
- evaluate_state should strongly prefer states closer to the optimum
  (fitness = n + jump_k) and penalize valley states (fitness < jump_k)

Provide ONLY the Python code.

```python
class SynthesizedCWM:
    ...
```
"""


JUMP_REFINEMENT_PROMPT = """\
The synthesized CWM failed validation.  Please fix the issues.

## Current code
```python
{current_code}
```

## Failures
{failures}

## Reminders
- JumpState and JumpAction are already defined — do NOT redefine them.
- predict_next_state must return a JumpState with step = state.step + 1.
- evaluate_state must return a finite float (no NaN / Inf).
- get_legal_actions must return a non-empty list of JumpAction objects.
- is_terminal must return a bool.
- The Jump_k fitness function has three regions: normal (fitness in [jump_k, n]),
  valley (fitness in [1, jump_k-1]), and optimum (fitness = n + jump_k).
- At the valley edge (fitness = n), only k = jump_k can improve.
- JumpState has fields: fitness, n, jump_k, step, budget, normalized_fitness.

Provide the complete fixed code.

```python
class SynthesizedCWM:
    ...
```
"""


# ── JumpState / JumpAction validation ──────────────────────────────────────

def _make_namespace():
    """Create an execution namespace with JumpState and JumpAction defined."""
    setup = """\
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math

@dataclass
class JumpState:
    fitness: int
    n: int
    jump_k: int
    step: int
    budget: int
    normalized_fitness: float

@dataclass
class JumpAction:
    k: int
    name: str
"""
    ns = {}
    exec(setup, ns)
    return ns


def validate_jump_cwm(code: str, n: int = 50, jump_k: int = 2,
                       budget: int = 10000):
    """Validate a synthesised Jump_k CWM.

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
        "valley_edge": False,
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

    JumpState = ns["JumpState"]
    JumpAction = ns["JumpAction"]

    # Test state in normal region (fitness=30, ones=28, well below valley edge)
    state = JumpState(fitness=30, n=n, jump_k=jump_k, step=50, budget=budget,
                      normalized_fitness=30 / (n + jump_k))
    action = JumpAction(k=3, name="flip_3")

    # predict_next_state
    try:
        ns_ = cwm.predict_next_state(state, action)
        if not isinstance(ns_, JumpState):
            errors.append("predict_next_state: did not return JumpState")
        elif ns_.step != state.step + 1:
            errors.append("predict_next_state: step not incremented")
        elif ns_.fitness < 0 or ns_.fitness > n + jump_k:
            errors.append(f"predict_next_state: fitness {ns_.fitness} out of [0,{n + jump_k}]")
        else:
            tests["predict_next_state"] = True
    except Exception as e:
        errors.append(f"predict_next_state: {e}")

    # get_legal_actions
    try:
        actions = cwm.get_legal_actions(state)
        if not isinstance(actions, list) or len(actions) == 0:
            errors.append("get_legal_actions: empty or wrong type")
        elif not all(isinstance(a, JumpAction) for a in actions):
            errors.append("get_legal_actions: elements are not JumpAction")
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

    # Numerical stability — edge cases (fitness=jump_k, ones=0)
    edge = JumpState(fitness=jump_k, n=n, jump_k=jump_k, step=0, budget=budget,
                     normalized_fitness=jump_k / (n + jump_k))
    edge_action = JumpAction(k=n, name=f"flip_{n}")
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

    # Valley edge test — fitness = n (ones = n - jump_k)
    # At valley edge, k=jump_k should give better predicted state than k=1
    valley_edge = JumpState(fitness=n, n=n, jump_k=jump_k, step=100,
                            budget=budget,
                            normalized_fitness=n / (n + jump_k))
    try:
        action_1 = JumpAction(k=1, name="flip_1")
        action_jk = JumpAction(k=jump_k, name=f"flip_{jump_k}")
        pred_1 = cwm.predict_next_state(valley_edge, action_1)
        pred_jk = cwm.predict_next_state(valley_edge, action_jk)
        score_1 = cwm.evaluate_state(pred_1)
        score_jk = cwm.evaluate_state(pred_jk)
        if math.isnan(score_1) or math.isnan(score_jk):
            errors.append("valley_edge: NaN scores")
        elif score_jk >= score_1:
            # Good: CWM correctly prefers k=jump_k at valley edge
            tests["valley_edge"] = True
        else:
            errors.append(
                f"valley_edge: CWM prefers k=1 (score={score_1:.4f}) over "
                f"k={jump_k} (score={score_jk:.4f}) at valley edge — "
                f"should prefer k={jump_k}")
    except Exception as e:
        errors.append(f"valley_edge: {e}")

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
            f"[{pol}] fitness={s['fitness']}, n={s['n']}, jump_k={s['jump_k']}, "
            f"k={a['k']} → fitness={ns['fitness']}, improved={t['improved']}"
        )
    return "\n".join(lines)


def compute_prompt_stats(data):
    """Extract statistics for the synthesis prompt."""
    trajectories = data["trajectories"]
    n = data["parameters"]["n"]
    jump_k = data["parameters"]["jump_k"]
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
        "jump_k": jump_k,
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
    """Synthesise + validate a Jump_k CWM.

    Returns (code, n_attempts) on success, raises RuntimeError on failure.
    """
    stats = compute_prompt_stats(data)
    samples_text = format_samples(data["trajectories"], max_samples=30)
    n = stats["n"]
    jump_k = stats["jump_k"]
    budget = data["parameters"]["budget"]

    prompt = JUMP_SYNTHESIS_PROMPT.format(
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
            system=JUMP_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        code = extract_code_from_response(response)

        if verbose:
            print(f"    LLM responded in {elapsed:.1f}s  ({len(code)} chars)")

        valid, tests, errors = validate_jump_cwm(
            code, n=n, jump_k=jump_k, budget=budget)
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

        prompt = JUMP_REFINEMENT_PROMPT.format(
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
        description="Exp36: Synthesize Jump_k CWM")
    parser.add_argument("--trajectories", type=str,
                        default="results/trajectories/jump_trajectories.json",
                        help="Path to trajectory JSON from exp8")
    parser.add_argument("--output", type=str,
                        default="results/cwm/jump_cwm.py",
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
    print("Experiment 9: Jump_k CWM Synthesis")
    print("=" * 60)

    if args.validate_only:
        print(f"\nValidating existing CWM: {args.output}")
        with open(args.output) as f:
            code = f.read()
        valid, tests, errors = validate_jump_cwm(code)
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
        "jump_k": data["parameters"]["jump_k"],
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Final validation
    print("\nFinal validation:")
    valid, tests, errors = validate_jump_cwm(
        code, n=data["parameters"]["n"], jump_k=data["parameters"]["jump_k"])
    for tname, ok in tests.items():
        print(f"  {tname}: {'OK' if ok else 'FAIL'}")
    print(f"\nValid: {valid}")


if __name__ == "__main__":
    main()
