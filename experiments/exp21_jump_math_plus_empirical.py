#!/usr/bin/env python3
"""
Experiment 21: Jump_k CWM with math description + empirical transition table

Combines:
  - Full mathematical Jump_k description (from exp9 — enables generalization)
  - Empirical transition table (from exp19 — helps valley-crossing strategy)
  - Only 4 agnostic collection policies (no k_jump knowledge)

The hypothesis: the math model provides parametric knowledge for any k,
while the empirical table provides concrete evidence for the valley edge.
This should achieve both oracle-free collection AND generalization to k=3,4.

Output: results/exp21/
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jump import (
    JumpProblem,
    RLSOptimizer,
    optimal_k_jump,
    precompute_optimal_k_table,
    run_episode,
)
from experiments.exp9_jump_cwm_synthesis import (
    validate_jump_cwm,
    format_samples,
    _make_namespace,
    JUMP_REFINEMENT_PROMPT,
)
from src.cwm.synthesizer import LLMClient
from src.cwm.validator import extract_code_from_response


# ── Agnostic Policies (no k_jump knowledge) ─────────────────────────────

def policy_random(fitness, n, jump_k, step):
    return np.random.randint(1, n + 1)

def policy_static_1(fitness, n, jump_k, step):
    return 1

def policy_sqrt(fitness, n, jump_k, step):
    return max(1, int(math.sqrt(n)))

def policy_decreasing(fitness, n, jump_k, step):
    return max(1, n - fitness)

POLICIES = {
    "random": policy_random,
    "static_1": policy_static_1,
    "sqrt": policy_sqrt,
    "decreasing": policy_decreasing,
}


# ── Trajectory Collection ────────────────────────────────────────────────

def collect_trajectories(n, jump_k, n_episodes, budget, verbose=True):
    all_trajectories = []
    for policy_name, policy_fn in POLICIES.items():
        if verbose:
            print(f"\n  Policy: {policy_name}")
        for ep in range(n_episodes):
            seed = hash((policy_name, ep, n, jump_k)) % (2**31)
            episode = run_episode(n, jump_k, policy_fn, budget, seed=seed)
            episode["policy"] = policy_name
            all_trajectories.append(episode)
            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                status = "OK" if episode["converged"] else f"fitness={episode['final_fitness']}"
                print(f"    Episode {ep+1}/{n_episodes}: {episode['total_steps']} steps — {status}")
    return all_trajectories


# ── Empirical Transition Table (same as exp19) ──────────────────────────

def compute_transition_table(trajectories, n, jump_k):
    fitness_bins = [
        (jump_k, jump_k + 10),
        (jump_k + 10, jump_k + 20),
        (jump_k + 20, n - 5),
        (n - 5, n),
        (n, n + 1),
    ]
    k_values = [1, 2, 3, 5, 7, 10, 15, 25, 50]

    data = defaultdict(lambda: {"n": 0, "improved": 0, "delta_all": []})

    for traj in trajectories:
        for t in traj["transitions"]:
            fitness = t["state"]["fitness"]
            k = t["action"]["k"]
            next_fitness = t["next_state"]["fitness"]
            delta = next_fitness - fitness
            improved = t["improved"] and delta > 0

            fbin = None
            for lo, hi in fitness_bins:
                if lo <= fitness < hi:
                    fbin = (lo, hi)
                    break
            if fbin is None:
                continue

            nearest_k = min(k_values, key=lambda kv: abs(kv - k))
            if abs(nearest_k - k) > 1:
                continue

            key = (fbin, nearest_k)
            data[key]["n"] += 1
            data[key]["delta_all"].append(delta)
            if improved:
                data[key]["improved"] += 1

    lines = []
    header = f"{'fitness_range':<20} {'k':>3} {'n_obs':>7} {'P(impr)':>8} {'mean_Δf':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for fbin in fitness_bins:
        for k in k_values:
            key = (fbin, k)
            d = data[key]
            if d["n"] < 1:
                continue
            p_impr = d["improved"] / d["n"]
            mean_delta = np.mean(d["delta_all"]) if d["delta_all"] else 0
            lo, hi = fbin
            label = f"[{lo}, {hi})" if hi != lo + 1 else f"{lo}"
            lines.append(f"{label:<20} {k:>3} {d['n']:>7} {p_impr:>8.4f} {mean_delta:>+8.3f}")

    return "\n".join(lines)


# ── Combined Prompt (math + empirical) ───────────────────────────────────

COMBINED_SYSTEM_PROMPT = """\
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

  - OPTIMUM (x = 1^n):
      fitness = jump_k + n.  The global maximum.

## Key Insight — Valley Dynamics

When the optimizer reaches fitness = n (the valley EDGE, ones = n - jump_k),
it is TRAPPED.  The only way to improve is to flip exactly the remaining
jump_k zero-bits to ones in a single step.  This requires k = jump_k and
all k flipped bits must be zeros.

CRITICAL: fitness near n is a TRAP, not near the optimum!  The optimal
strategy is:
  - In the normal region (far from edge): k=1 works well (like OneMax)
  - Near/at the valley edge (fitness ≈ n): k = jump_k to jump across
  - In the valley (fitness < jump_k): k = jump_k to escape

## Transition Model Details (mathematical)

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
  - k = jump_k: probability of reaching optimum = 1/C(n, jump_k)
  - k < jump_k: CANNOT reach optimum (not enough bits flipped)
  - k > jump_k: probability drops (must flip some 1-bits too)

## Empirical Transition Table

Below you will also receive an empirical transition table computed from actual
trajectory data.  This table confirms the mathematical model and shows real
P(improve) and mean Δf values per (fitness_range, k).

Use BOTH the mathematical formulas AND the empirical data in your CWM.
The math model generalizes to any jump_k value; the empirical data confirms
the specific transition probabilities observed in practice.

## Your CWM Specification

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

Methods to implement inside class SynthesizedCWM:

  predict_next_state(state, action) -> JumpState
      Use the mathematical model for the general case.
      At the valley edge, the probability of escaping is 1/C(n, jump_k).
      normalized_fitness must store CONTINUOUS expected value / (n + jump_k).

  get_legal_actions(state) -> List[JumpAction]
      Return JumpAction objects.  MUST include k=jump_k.

  evaluate_state(state) -> float
      Score state quality (higher = better).  MUST use normalized_fitness
      to distinguish between actions with different P(improve).

  is_terminal(state) -> bool

CRITICAL for predict_next_state:
- The fitness field must remain an integer (floor of expected value).
- The normalized_fitness field must store the CONTINUOUS expected fitness
  divided by (n + jump_k).  This allows evaluate_state to distinguish
  between actions that differ in improvement probability.
- evaluate_state MUST use normalized_fitness (not fitness) so that it can
  rank actions correctly, especially at the valley edge where the difference
  between k=1 (useless) and k=jump_k (can escape) is crucial.
"""


COMBINED_SYNTHESIS_PROMPT = """\
Based on the following data from (1+1)-RLS_k on Jump_k, synthesize a CWM.

## Empirical Transition Table

This table summarizes P(improve) and mean Δf for each (fitness_range, k),
computed from {total_transitions} transitions across {n_trajectories}
trajectories.  IMPORTANT: at the valley edge (fitness = {n}), check which
k values have P(improve) > 0 — this confirms the mathematical prediction
that only k = jump_k can escape.

```
{transition_table}
```

## Trajectory samples ({n_samples} transitions)
```
{trajectory_samples}
```

## Statistics (n={n}, jump_k={jump_k})
- Policies seen: {policies}
- Success rate: {convergence_rate:.1%}
- Mean steps (successful): {mean_steps:.0f}

## Requirements
Generate a Python class `SynthesizedCWM` that implements the four methods.
Self-contained (no imports except math).  JumpState and JumpAction are
already defined — do NOT redefine them.

Key requirements for Jump_k:
- predict_next_state must handle ALL THREE regions (normal, valley, optimum)
- Use the mathematical hypergeometric model for the normal region
- At the valley edge (fitness = n), k = jump_k should show a small but
  positive improvement (probability 1/C(n, jump_k)), while k != jump_k
  should show zero or negative expected change
- get_legal_actions MUST include k = state.jump_k as a candidate action
- evaluate_state should use normalized_fitness to rank actions

The CWM must work for ANY value of jump_k, not just the one in the data.

```python
class SynthesizedCWM:
    ...
```
"""


# ── Synthesis ────────────────────────────────────────────────────────────

def synthesize_cwm(trajectories, n, jump_k, budget, model, max_attempts=5,
                   verbose=True):
    total_trans = sum(len(t["transitions"]) for t in trajectories)
    policies = sorted(set(t["policy"] for t in trajectories))
    converged = [t for t in trajectories if t["converged"]]
    convergence_rate = len(converged) / len(trajectories)
    mean_steps = float(np.mean([t["total_steps"] for t in converged])) if converged else float("inf")

    transition_table = compute_transition_table(trajectories, n, jump_k)
    if verbose:
        print(f"\n  Transition table:")
        for line in transition_table.split("\n")[:20]:
            print(f"    {line}")

    samples_text = format_samples(trajectories, max_samples=30)

    prompt = COMBINED_SYNTHESIS_PROMPT.format(
        transition_table=transition_table,
        trajectory_samples=samples_text,
        n_samples=min(30, total_trans),
        n=n,
        jump_k=jump_k,
        n_trajectories=len(trajectories),
        total_transitions=total_trans,
        policies=", ".join(policies),
        convergence_rate=convergence_rate,
        mean_steps=mean_steps,
    )

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
            system=COMBINED_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        code = extract_code_from_response(response)

        if verbose:
            print(f"    LLM responded in {elapsed:.1f}s  ({len(code)} chars)")

        valid, tests, errors = validate_jump_cwm(code, n=n, jump_k=jump_k, budget=budget)
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

        failures_text = "\n".join(
            f"- {tname}: {'PASS' if ok else 'FAIL'}" for tname, ok in tests.items()
        )
        if errors:
            failures_text += "\n" + "\n".join(f"- Error: {e}" for e in errors)

        prompt = JUMP_REFINEMENT_PROMPT.format(
            current_code=code, failures=failures_text,
        )

    if best_code is not None:
        if verbose:
            print(f"\n  WARNING: best-effort CWM ({best_pass_count} tests passed)")
        return best_code, max_attempts

    raise RuntimeError("CWM synthesis failed")


# ── Evaluation ───────────────────────────────────────────────────────────

def compare_policies(n, jump_k, budget, n_episodes, cwm_code, verbose=True):
    ns = _make_namespace()
    exec(cwm_code, ns)
    cwm = ns["SynthesizedCWM"]()
    JumpState = ns["JumpState"]

    def cwm_greedy(fitness, n_, jk, step):
        state = JumpState(fitness=fitness, n=n_, jump_k=jk, step=step,
                          budget=budget, normalized_fitness=fitness/(n_+jk))
        actions = cwm.get_legal_actions(state)
        best_k, best_score = 1, -float("inf")
        for a in actions:
            try:
                pred = cwm.predict_next_state(state, a)
                score = cwm.evaluate_state(pred)
                if score > best_score:
                    best_score = score
                    best_k = a.k
            except Exception:
                pass
        return best_k

    optimal_table = precompute_optimal_k_table(n, jump_k)

    def make_ea_alpha(A, b, n_val):
        state = {"k": 1.0, "prev_fitness": None}
        def policy(fitness, n_, jk, step):
            if state["prev_fitness"] is not None:
                if fitness > state["prev_fitness"]:
                    state["k"] = min(A * state["k"], n_ / 2)
                else:
                    state["k"] = max(b * state["k"], 1.0)
            state["prev_fitness"] = fitness
            return max(1, min(n_, int(round(state["k"]))))
        return policy

    def make_stagnation(jk, patience=100):
        state = {"count": 0, "prev_f": None}
        def policy(fitness, n_, jk_, step):
            if state["prev_f"] is not None:
                if fitness > state["prev_f"]:
                    state["count"] = 0
                else:
                    state["count"] += 1
            state["prev_f"] = fitness
            return jk if state["count"] >= patience else 1
        return policy

    stateless = {
        "optimal": lambda f, n_, jk, s: optimal_table[f] if f < len(optimal_table) else 1,
        "CWM-greedy": cwm_greedy,
        "RLS_1": lambda f, n_, jk, s: 1,
        "RLS_jump_k": lambda f, n_, jk, s: jk,
        "random_k": lambda f, n_, jk, s: np.random.randint(1, n_ + 1),
    }
    ea_configs = {"EA_alpha(2,0.5)": (2, 0.5), "EA_alpha(1.3,0.75)": (1.3, 0.75)}

    all_names = list(stateless.keys()) + list(ea_configs.keys()) + ["stagnation"]
    results = {n_: [] for n_ in all_names}
    convergence = {n_: [] for n_ in all_names}

    for pname in all_names:
        if verbose:
            print(f"\n  Evaluating: {pname}")
        for ep in range(n_episodes):
            seed = hash(("exp21_eval", pname, ep, n, jump_k)) % (2**31)
            if pname in stateless:
                pfn = stateless[pname]
            elif pname in ea_configs:
                A, b = ea_configs[pname]
                pfn = make_ea_alpha(A, b, n)
            elif pname == "stagnation":
                pfn = make_stagnation(jump_k)
            episode = run_episode(n, jump_k, pfn, budget, seed=seed)
            results[pname].append(episode["total_steps"])
            convergence[pname].append(episode["converged"])
        if verbose:
            sr = np.mean(convergence[pname]) * 100
            print(f"    Mean: {np.mean(results[pname]):.0f}, SR: {sr:.0f}%")

    return results, convergence


def compute_cwm_quality(cwm_code, n, jump_k, budget):
    ns = _make_namespace()
    exec(cwm_code, ns)
    cwm = ns["SynthesizedCWM"]()
    JumpState = ns["JumpState"]
    JumpAction = ns["JumpAction"]
    optimal_table = precompute_optimal_k_table(n, jump_k)

    taus, rhos = [], []
    valley_edge_correct = False

    for fitness in range(jump_k, n + 1):
        state = JumpState(fitness=fitness, n=n, jump_k=jump_k, step=50,
                          budget=budget, normalized_fitness=fitness/(n+jump_k))
        k_scores = {}
        for k in range(1, min(n + 1, 26)):
            action = JumpAction(k=k, name=f"flip_{k}")
            try:
                pred = cwm.predict_next_state(state, action)
                score = cwm.evaluate_state(pred)
                if not (math.isnan(score) or math.isinf(score)):
                    k_scores[k] = score
            except Exception:
                pass
        if len(k_scores) < 2:
            continue
        opt_k = optimal_table[fitness] if fitness < len(optimal_table) else 1
        ks = sorted(k_scores.keys())
        cwm_ranks = [k_scores[k] for k in ks]
        gt_ranks = [-abs(k - opt_k) for k in ks]
        if len(set(cwm_ranks)) > 1 and len(set(gt_ranks)) > 1:
            tau, _ = sp_stats.kendalltau(cwm_ranks, gt_ranks)
            rho, _ = sp_stats.spearmanr(cwm_ranks, gt_ranks)
            if not math.isnan(tau): taus.append(tau)
            if not math.isnan(rho): rhos.append(rho)
        if fitness == n:
            cwm_best_k = max(k_scores, key=k_scores.get)
            valley_edge_correct = (cwm_best_k == jump_k)

    return {
        "mean_tau": float(np.mean(taus)) if taus else 0.0,
        "mean_rho": float(np.mean(rhos)) if rhos else 0.0,
        "valley_edge_tau": 1.0 if valley_edge_correct else 0.0,
    }


# ── Generalization test ──────────────────────────────────────────────────

def test_generalization(cwm_code, n, test_jks, n_episodes=50, verbose=True):
    ns = _make_namespace()
    exec(cwm_code, ns)
    cwm = ns["SynthesizedCWM"]()
    JumpState = ns["JumpState"]
    candidate_ks = [1, 2, 3, 5, 8, 10, 15, 25, 50]

    results = {}

    for jk in test_jks:
        budget = min(50000, max(10000, int(8 * math.comb(n, jk))))
        if verbose:
            print(f"\n  jump_k={jk}, budget={budget}")

        def make_cwm_policy(jk_=jk, budget_=budget):
            def policy(fitness, n_, jk__, step):
                state = JumpState(fitness=fitness, n=n_, jump_k=jk_,
                                  step=step, budget=budget_,
                                  normalized_fitness=fitness/(n_+jk_))
                best_k, best_score = 1, -float("inf")
                for k in candidate_ks:
                    action = ns["JumpAction"](k=k, name=f"flip_{k}")
                    try:
                        pred = cwm.predict_next_state(state, action)
                        score = cwm.evaluate_state(pred)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except Exception:
                        pass
                return best_k
            return policy

        opt_table = precompute_optimal_k_table(n, jk)

        def make_stag(jk_=jk):
            state = {"count": 0, "prev_f": None}
            def policy(fitness, n_, jk__, step):
                if state["prev_f"] is not None:
                    if fitness > state["prev_f"]:
                        state["count"] = 0
                    else:
                        state["count"] += 1
                state["prev_f"] = fitness
                return jk_ if state["count"] >= 100 else 1
            return policy

        def make_ea(n_=n):
            state = {"k": 1.0, "prev_f": None}
            def policy(fitness, n__, jk_, step):
                if state["prev_f"] is not None:
                    if fitness > state["prev_f"]:
                        state["k"] = min(2.0 * state["k"], n__ / 2)
                    else:
                        state["k"] = max(0.5 * state["k"], 1.0)
                state["prev_f"] = fitness
                return max(1, min(n__, int(round(state["k"]))))
            return policy

        test_policies = [
            ("CWM-greedy", make_cwm_policy, True),
            ("optimal", lambda jk_=jk: (lambda f, n_, jk__, s: opt_table[f] if f < len(opt_table) else 1), True),
            ("stagnation", make_stag, True),
            ("EA_alpha", make_ea, True),
        ]

        for pname, factory, is_stateful in test_policies:
            steps_list = []
            conv_list = []
            for ep in range(n_episodes):
                seed = ep * 37 + jk * 1000
                pfn = factory() if is_stateful else factory
                episode = run_episode(n, jk, pfn, budget, seed=seed)
                steps_list.append(episode["total_steps"])
                conv_list.append(episode["converged"])

            key = f"jk{jk}_{pname}"
            results[key] = {
                "jump_k": jk,
                "policy": pname,
                "budget": budget,
                "mean_steps": float(np.mean(steps_list)),
                "std_steps": float(np.std(steps_list)),
                "convergence_rate": float(np.mean(conv_list)),
            }
            if verbose:
                print(f"    {pname:<15s}: mean={results[key]['mean_steps']:.0f}  "
                      f"SR={results[key]['convergence_rate']:.0%}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp21: Jump_k CWM — math + empirical, agnostic policies")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--jump-k", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/exp21")
    parser.add_argument("--reuse-trajectories", type=str, default=None)
    args = parser.parse_args()

    n, jump_k, budget = args.n, args.jump_k, args.budget
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 21: Jump_k CWM — Math + Empirical")
    print("=" * 60)
    print(f"  n={n}, jump_k={jump_k}, budget={budget}")
    print(f"  Policies: {list(POLICIES.keys())}")
    print(f"  Prompt: math description + empirical transition table")

    # A. Trajectories
    if args.reuse_trajectories:
        print(f"\n  Reusing trajectories from {args.reuse_trajectories}")
        with open(args.reuse_trajectories) as f:
            tdata = json.load(f)
        trajectories = tdata["trajectories"]
    else:
        print(f"\n{'='*60}")
        print("A. Trajectory Collection")
        print(f"{'='*60}")
        trajectories = collect_trajectories(n, jump_k, args.episodes, budget)

    print(f"  Total: {len(trajectories)} trajectories")

    # B. Synthesis
    print(f"\n{'='*60}")
    print("B. CWM Synthesis (math + empirical)")
    print(f"{'='*60}")

    cwm_code, n_attempts = synthesize_cwm(
        trajectories, n, jump_k, budget,
        model=args.model, max_attempts=args.max_attempts
    )

    cwm_path = output_dir / "jump_cwm_combined.py"
    with open(cwm_path, "w") as f:
        f.write(cwm_code)
    print(f"\nCWM saved to {cwm_path} (attempts: {n_attempts})")

    # C. Evaluation on k=2
    print(f"\n{'='*60}")
    print("C. Evaluation (k=2)")
    print(f"{'='*60}")

    eval_results, eval_conv = compare_policies(
        n, jump_k, budget, args.eval_episodes, cwm_code
    )

    # D. Quality
    print(f"\n{'='*60}")
    print("D. CWM Quality")
    print(f"{'='*60}")

    quality = compute_cwm_quality(cwm_code, n, jump_k, budget)
    print(f"  tau={quality['mean_tau']:.3f}, rho={quality['mean_rho']:.3f}, "
          f"valley_edge={quality['valley_edge_tau']:.1f}")

    # E. Generalization
    print(f"\n{'='*60}")
    print("E. Generalization (k=2,3,4)")
    print(f"{'='*60}")

    gen_results = test_generalization(cwm_code, n, [2, 3, 4])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — k=2 evaluation")
    print(f"{'='*60}")
    print(f"{'Policy':<25s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'SR%':>6s}")
    print(f"{'-'*57}")

    summary = {}
    for pname in eval_results:
        steps = eval_results[pname]
        conv = eval_conv[pname]
        sr = np.mean(conv) * 100
        summary[pname] = {
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
            "median": float(np.median(steps)),
            "success_rate": float(sr),
        }
        print(f"{pname:<25s} {np.mean(steps):>8.0f} {np.std(steps):>8.0f} "
              f"{np.median(steps):>8.0f} {sr:>5.0f}%")

    print(f"\n{'='*60}")
    print("SUMMARY — Generalization")
    print(f"{'='*60}")
    for key, r in sorted(gen_results.items()):
        print(f"  {key:<25s}: mean={r['mean_steps']:.0f}, SR={r['convergence_rate']:.0%}")

    # Save
    output = {
        "experiment": "exp21_jump_math_plus_empirical",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "jump_k": jump_k, "budget": budget,
            "eval_episodes": args.eval_episodes,
            "model": args.model, "synthesis_attempts": n_attempts,
        },
        "collection_policies": list(POLICIES.keys()),
        "excluded_policies": ["static_jump_k", "stagnation_detect"],
        "prompt_type": "math_plus_empirical",
        "cwm_quality": quality,
        "evaluation": summary,
        "generalization": gen_results,
    }

    results_path = output_dir / "exp21_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
