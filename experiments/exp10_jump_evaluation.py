#!/usr/bin/env python3
"""
Experiment 10: Jump_k CWM Evaluation

Four sub-experiments:

  A. CWM Quality – Kendall's tau between CWM action ranking and ground-truth
     ranking at several fitness snapshots, including the valley edge.
  B. Policy Comparison – 8 policies (RLS_1, RLS_jump_k, random_k, optimal,
     CWM-greedy, CWM-MCTS, EA_alpha(2,0.5), EA_alpha(1.3,0.75)) compared
     by steps-to-optimum over 100 episodes.
  C. Interpretability – extract the CWM's learned policy and plot vs the
     theoretical optimum. Valley region highlighted. Valley detection metric.
  D. Generalisation – train on jump_k=2, evaluate on jump_k=2,3,4.

Outputs:
  results/exp10/exp10_results.json        — all numerical data
  results/exp10/fig_policy_comparison.png  — learned vs optimal k(fitness)
  results/exp10/fig_heatmap.png            — CWM score heatmap with valley
  results/exp10/fig_convergence.png        — convergence curves
  results/exp10/fig_generalisation.png     — generalisation across jump_k
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jump import (
    JumpProblem,
    RLSOptimizer,
    optimal_k_jump,
    approx_optimal_k_jump,
    precompute_optimal_k_table,
    run_episode,
)


# ── JumpState / JumpAction (same as in exp9) ─────────────────────────────

def _make_jump_namespace():
    """Execution namespace with JumpState & JumpAction."""
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
    ns: Dict[str, Any] = {}
    exec(setup, ns)
    return ns


def load_jump_cwm(cwm_path: str):
    """Load a synthesised Jump_k CWM.

    Returns (cwm_instance, namespace) so callers can create JumpState/JumpAction.
    """
    with open(cwm_path) as f:
        code = f.read()
    ns = _make_jump_namespace()
    exec(code, ns)
    cwm = ns["SynthesizedCWM"]()
    return cwm, ns


# ── Helpers ─────────────────────────────────────────────────────────────────

def _jump_state(ns, fitness: int, n: int, jump_k: int, step: int, budget: int):
    return ns["JumpState"](
        fitness=fitness, n=n, jump_k=jump_k, step=step, budget=budget,
        normalized_fitness=fitness / (n + jump_k),
    )


def _jump_action(ns, k: int):
    return ns["JumpAction"](k=k, name=f"flip_{k}")


# ── Adaptive baselines ─────────────────────────────────────────────────────

def _ea_alpha_policy(n, A=2.0, b=0.5):
    """(1+1) EA_alpha adapted to RLS_k (after Doerr & Wagner, GECCO 2018).

    On Jump_k this policy FAILS at the valley edge: when stuck (no
    improvement), it reduces k — exactly wrong, because you need LARGE k
    to jump across the valley. This failure is a FEATURE demonstrating
    why CWM is needed.
    """
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, jump_k, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


def _self_adjusting_policy(n, A=1.3, b=0.75):
    """(1+1) EA_alpha (best config) adapted to RLS_k.

    Same failure mode as _ea_alpha_policy on Jump_k: reduces k when stuck.
    """
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, jump_k, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


def _stagnation_detect_policy(jump_k_val, patience=100):
    """Stagnation-detection baseline: k=1 normally, k=jump_k when stuck."""
    state = {"no_improve_count": 0, "prev_fitness": None}

    def policy(fitness, n_, jump_k, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:  # strict: ties = stagnation
                state["no_improve_count"] = 0
            else:
                state["no_improve_count"] += 1
        state["prev_fitness"] = fitness
        if state["no_improve_count"] >= patience:
            return jump_k_val
        return 1

    return policy


# ── A. CWM Quality ─────────────────────────────────────────────────────────

def evaluate_cwm_quality(
    cwm, ns,
    n: int = 50,
    jump_k: int = 2,
    budget: int = 10000,
    fitness_snapshots: list = None,
    test_ks: list = None,
    n_trials: int = 200,
    verbose: bool = True,
):
    """For each fitness snapshot, compare CWM action ranking against ground truth.

    Includes snapshots in the normal region AND at the valley edge (fitness=n).
    Valley-edge snapshots are critical for Jump_k.
    """
    if fitness_snapshots is None:
        # Normal region snapshots + valley edge
        normal_snaps = list(range(jump_k + 5, n, max(1, (n - jump_k) // 8)))
        # Ensure valley edge (fitness=n) is included
        if n not in normal_snaps:
            normal_snaps.append(n)
        fitness_snapshots = normal_snaps

    if test_ks is None:
        test_ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 25, 50, jump_k]))
        test_ks = [k for k in test_ks if 1 <= k <= n]

    problem = JumpProblem(n, jump_k)
    results: Dict[int, Any] = {}

    for fit in fitness_snapshots:
        if fit > n + jump_k or fit < jump_k:
            continue

        # Determine ones count from fitness
        if fit >= jump_k and fit <= n:
            # Normal region: ones = fitness - jump_k
            ones = fit - jump_k
        elif fit == n + jump_k:
            ones = n  # optimum
        else:
            # Valley: ones = n - fitness
            ones = n - fit

        if ones < 0 or ones > n:
            continue

        # Build a canonical bitstring: first `ones` bits = 1, rest = 0
        base_x = np.zeros(n, dtype=int)
        base_x[:ones] = 1

        # Ground truth: average fitness after one step with each k
        gt_scores = []
        for k in test_ks:
            improvements = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial * 1000 + fit * 100 + k)
                positions = rng.choice(n, min(k, n), replace=False)
                y = base_x.copy()
                y[positions] = 1 - y[positions]
                new_fit = problem.fitness(y)
                # Non-strict selection: accept if new_fit >= fit
                accepted_fit = new_fit if new_fit >= fit else fit
                improvements.append(accepted_fit - fit)
            gt_scores.append(float(np.mean(improvements)))

        # CWM ranking
        state = _jump_state(ns, fit, n, jump_k, step=0, budget=budget)
        cwm_scores = []
        for k in test_ks:
            action = _jump_action(ns, k)
            try:
                pred = cwm.predict_next_state(state, action)
                score = cwm.evaluate_state(pred)
            except Exception:
                score = 0.0
            cwm_scores.append(score)

        # Kendall's tau
        gt_arr = np.array(gt_scores)
        cwm_arr = np.array(cwm_scores)
        if np.std(gt_arr) > 0 and np.std(cwm_arr) > 0:
            tau, pval = sp_stats.kendalltau(gt_arr, cwm_arr)
        else:
            tau, pval = 0.0, 1.0

        # Top-1 concordance
        gt_best_k = test_ks[int(np.argmax(gt_arr))]
        cwm_best_k = test_ks[int(np.argmax(cwm_arr))]
        opt_k = optimal_k_jump(fit, n, jump_k)

        # Region label
        if fit >= jump_k and fit <= n:
            if fit == n:
                region = "valley_edge"
            else:
                region = "normal"
        elif fit == n + jump_k:
            region = "optimum"
        else:
            region = "valley"

        results[fit] = {
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0,
            "gt_best_k": gt_best_k,
            "cwm_best_k": cwm_best_k,
            "optimal_k": opt_k,
            "region": region,
            "gt_scores": [float(s) for s in gt_scores],
            "cwm_scores": [float(s) for s in cwm_scores],
        }

        if verbose:
            print(f"  fitness={fit:3d} [{region:11s}]  tau={tau:.3f}  "
                  f"gt_best_k={gt_best_k}  cwm_best_k={cwm_best_k}  "
                  f"optimal_k={opt_k}")

    # Aggregate
    taus = [v["kendall_tau"] for v in results.values()]
    mean_tau = float(np.mean(taus)) if taus else 0.0

    # Valley-edge specific tau
    valley_taus = [v["kendall_tau"] for v in results.values()
                   if v["region"] == "valley_edge"]
    valley_tau = float(np.mean(valley_taus)) if valley_taus else 0.0

    if verbose:
        print(f"\n  Mean Kendall's tau = {mean_tau:.3f}")
        print(f"  Valley-edge tau    = {valley_tau:.3f}")

    return {
        "per_fitness": results,
        "mean_tau": mean_tau,
        "valley_edge_tau": valley_tau,
        "test_ks": test_ks,
    }


# ── B. Policy Comparison ───────────────────────────────────────────────────

# Precomputed optimal k table (populated at runtime)
_OPTIMAL_TABLE = None
_DEFAULT_N = 50
_DEFAULT_JUMP_K = 2


def _get_optimal_k(fitness: int, n: int, jump_k: int) -> int:
    global _OPTIMAL_TABLE
    if (_OPTIMAL_TABLE is not None
            and len(_OPTIMAL_TABLE) == n + jump_k + 1):
        return _OPTIMAL_TABLE[min(fitness, n + jump_k)]
    return optimal_k_jump(fitness, n, jump_k)


def _cwm_greedy_policy(cwm, ns, n, jump_k, budget):
    """1-step lookahead: pick the k that maximises evaluate_state."""
    candidate_ks = sorted(set(
        [1, 2, 3, 5, 8, 10, 15, 25, jump_k]
        + [max(1, n - i) for i in range(0, n, 5)]
        + [n]
    ))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    def policy(fitness, n_, jump_k_, step):
        state = _jump_state(ns, fitness, n_, jump_k_, step, budget)
        best_k, best_score = 1, float("-inf")
        for k in candidate_ks:
            action = _jump_action(ns, k)
            try:
                pred = cwm.predict_next_state(state, action)
                score = cwm.evaluate_state(pred)
            except Exception:
                score = float("-inf")
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    return policy


def _cwm_mcts_policy(cwm, ns, n, jump_k, budget, simulations=50, horizon=5):
    """MCTS policy using the CWM for rollouts."""
    candidate_ks = sorted(set(
        [1, 2, 3, 5, 8, 10, 15, 25, jump_k]
        + [max(1, n - i) for i in range(0, n, 5)]
        + [n]
    ))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    def policy(fitness, n_, jump_k_, step):
        root_state = _jump_state(ns, fitness, n_, jump_k_, step, budget)

        # action -> (total_value, visits)
        stats = {k: [0.0, 0] for k in candidate_ks}

        for _ in range(simulations):
            # Select action (UCB1)
            best_k = None
            best_ucb = float("-inf")
            total_visits = sum(s[1] for s in stats.values())
            for k in candidate_ks:
                tv, vis = stats[k]
                if vis == 0:
                    best_k = k
                    break
                mean_val = tv / vis
                explore = 1.414 * np.sqrt(np.log(total_visits + 1) / vis)
                ucb = mean_val + explore
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_k = k

            # Rollout from root with chosen first action
            action = _jump_action(ns, best_k)
            try:
                state = cwm.predict_next_state(root_state, action)
            except Exception:
                state = root_state

            value = 0.0
            for h in range(horizon):
                try:
                    if cwm.is_terminal(state):
                        break
                    rk = candidate_ks[np.random.randint(len(candidate_ks))]
                    ra = _jump_action(ns, rk)
                    state = cwm.predict_next_state(state, ra)
                except Exception:
                    break

            try:
                value = cwm.evaluate_state(state)
            except Exception:
                value = 0.0

            stats[best_k][0] += value
            stats[best_k][1] += 1

        # Pick action with most visits
        return max(candidate_ks, key=lambda k: stats[k][1])

    return policy


def run_policy_comparison(
    cwm, ns,
    n: int = 50,
    jump_k: int = 2,
    n_episodes: int = 100,
    budget: int = 10000,
    mcts_sims: int = 50,
    mcts_horizon: int = 5,
    verbose: bool = True,
):
    """Compare 8 policies over n_episodes.

    Returns dict mapping policy_name -> list of steps-to-optimum.
    Also returns convergence info (did the run converge within budget).
    """
    stateless_policies = {
        "RLS_1":        lambda f, n_, jk, s: 1,
        "RLS_jump_k":   lambda f, n_, jk, s: jk,
        "random_k":     lambda f, n_, jk, s: np.random.randint(1, n_ + 1),
        "optimal":      lambda f, n_, jk, s: _get_optimal_k(f, n_, jk),
        "cwm_greedy":   _cwm_greedy_policy(cwm, ns, n, jump_k, budget),
        "cwm_mcts":     _cwm_mcts_policy(cwm, ns, n, jump_k, budget,
                                          mcts_sims, mcts_horizon),
    }
    stateful_factories = {
        "fifth_rule":          lambda: _ea_alpha_policy(n),
        "self_adjusting":      lambda: _self_adjusting_policy(n),
    }

    all_policy_names = list(stateless_policies.keys()) + list(stateful_factories.keys())
    results: Dict[str, list] = {name: [] for name in all_policy_names}
    convergence: Dict[str, list] = {name: [] for name in all_policy_names}

    for pname in all_policy_names:
        if verbose:
            print(f"\n  Policy: {pname}")
        for ep in range(n_episodes):
            seed = ep * 31
            if pname in stateless_policies:
                pfn = stateless_policies[pname]
            else:
                pfn = stateful_factories[pname]()
            episode = run_episode(n, jump_k, pfn, budget, seed=seed)
            results[pname].append(episode["total_steps"])
            convergence[pname].append(episode["converged"])

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                mean_so_far = np.mean(results[pname])
                conv_so_far = np.mean(convergence[pname])
                print(f"    Episode {ep+1}/{n_episodes}  mean_steps={mean_so_far:.0f}  "
                      f"conv={conv_so_far:.0%}")

    return results, convergence


def compute_policy_statistics(results: Dict[str, list],
                              convergence: Dict[str, list]):
    """Compute summary stats + pairwise tests."""
    summary = {}
    for name, steps in results.items():
        arr = np.array(steps, dtype=float)
        conv = np.array(convergence[name], dtype=float)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
            "convergence_rate": float(np.mean(conv)),
        }

    # Pairwise Mann-Whitney U
    names = list(results.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(results[names[i]], dtype=float)
            b = np.array(results[names[j]], dtype=float)
            try:
                stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
                n1, n2 = len(a), len(b)
                greater = sum(1 for x in a for y in b if x > y)
                less = sum(1 for x in a for y in b if x < y)
                delta = (greater - less) / (n1 * n2)
            except Exception:
                stat, p, delta = 0.0, 1.0, 0.0

            pairwise[f"{names[i]}_vs_{names[j]}"] = {
                "U_stat": float(stat),
                "p_value": float(p),
                "cliff_delta": float(delta),
            }

    return summary, pairwise


# ── C. Interpretability ────────────────────────────────────────────────────

def extract_learned_policy(cwm, ns, n: int = 50, jump_k: int = 2,
                           budget: int = 10000):
    """For each fitness level in the normal region + valley edge, find best k.

    Returns dict fitness -> {"cwm_best_k": int, "optimal_k": int, ...}
    """
    candidate_ks = list(range(1, n + 1))
    learned = {}

    # Normal region: fitness from jump_k to n (ones from 0 to n-jump_k)
    for fit in range(jump_k, n + 1):
        state = _jump_state(ns, fit, n, jump_k, step=0, budget=budget)
        scores = {}
        for k in candidate_ks:
            action = _jump_action(ns, k)
            try:
                pred = cwm.predict_next_state(state, action)
                scores[k] = cwm.evaluate_state(pred)
            except Exception:
                scores[k] = float("-inf")

        best_k = max(candidate_ks, key=lambda k: scores[k])

        # Region annotation
        if fit == n:
            region = "valley_edge"
        else:
            region = "normal"

        learned[fit] = {
            "cwm_best_k": best_k,
            "optimal_k": _get_optimal_k(fit, n, jump_k),
            "region": region,
            "scores": {str(k): float(v) for k, v in scores.items()
                       if not np.isinf(v)},
        }

    return learned


def build_heatmap_data(cwm, ns, n: int = 50, jump_k: int = 2,
                       budget: int = 10000):
    """Build fitness x k score matrix for heatmap visualisation.

    Covers the normal region (fitness from jump_k to n).
    """
    fitnesses = list(range(jump_k, n + 1, max(1, (n - jump_k) // 20)))
    # Ensure valley edge is included
    if n not in fitnesses:
        fitnesses.append(n)
    fitnesses.sort()

    ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, jump_k]))
    ks = [k for k in ks if k <= n]

    matrix = []
    for fit in fitnesses:
        row = []
        state = _jump_state(ns, fit, n, jump_k, step=0, budget=budget)
        for k in ks:
            action = _jump_action(ns, k)
            try:
                pred = cwm.predict_next_state(state, action)
                score = cwm.evaluate_state(pred)
            except Exception:
                score = float("nan")
            row.append(score)
        matrix.append(row)

    return {
        "fitnesses": fitnesses,
        "ks": ks,
        "scores": matrix,
    }


# ── D. Generalisation (across jump_k values) ──────────────────────────────

def run_generalisation(
    cwm, ns,
    n: int = 50,
    jump_ks: list = None,
    n_episodes: int = 50,
    verbose: bool = True,
):
    """Evaluate the CWM (trained on jump_k=2) on different jump_k values."""
    if jump_ks is None:
        jump_ks = [2, 3, 4]

    results = {}
    for jk in jump_ks:
        # Adjust budget: larger jump_k needs more time
        # Expected valley crossing: C(n, jk) steps at valley edge
        from math import comb as _comb
        expected_valley = _comb(n, jk)
        # Cap budget to keep generalisation tractable; accept low convergence
        # for large jump_k (the point is to measure relative performance)
        budget = min(50000, max(10000, int(8 * expected_valley)))
        if verbose:
            print(f"\n  jump_k={jk}, budget={budget}, "
                  f"E[valley crossing]={expected_valley}")

        greedy_fn = _cwm_greedy_policy(cwm, ns, n, jk, budget)
        optimal_fn = lambda f, n_=n, jk_=jk, s=None: _get_optimal_k(f, n_, jk_)
        static1_fn = lambda f, n_, jk_, s: 1
        static_jk_fn = lambda f, n_, jk_=jk, s=None: jk_

        stateless_pols = [
            ("optimal", optimal_fn),
            ("CWM-greedy", greedy_fn),
            ("RLS_1", static1_fn),
            ("RLS_jump_k", static_jk_fn),
        ]
        stateful_pols = [
            ("fifth_rule", lambda jk_=jk: _ea_alpha_policy(n)),
            ("self_adjusting", lambda jk_=jk: _self_adjusting_policy(n)),
            ("stagnation_detect", lambda jk_=jk: _stagnation_detect_policy(jk_)),
        ]

        def _run_gen_policy(pname, pfn_or_factory, jk_val, bgt,
                            is_stateful=False):
            steps_list = []
            converged_list = []
            for ep in range(n_episodes):
                seed = ep * 37 + jk_val * 1000
                pfn = pfn_or_factory() if is_stateful else pfn_or_factory
                episode = run_episode(n, jk_val, pfn, bgt, seed=seed)
                steps_list.append(episode["total_steps"])
                converged_list.append(episode["converged"])

            key = f"jk{jk_val}_{pname}"
            results[key] = {
                "jump_k": jk_val,
                "n": n,
                "policy": pname,
                "budget": bgt,
                "mean_steps": float(np.mean(steps_list)),
                "std_steps": float(np.std(steps_list)),
                "median_steps": float(np.median(steps_list)),
                "convergence_rate": float(np.mean(converged_list)),
            }
            if verbose:
                print(f"    {pname:<20s}: mean={results[key]['mean_steps']:.0f}  "
                      f"conv={results[key]['convergence_rate']:.0%}")

        for pname, pfn in stateless_pols:
            _run_gen_policy(pname, pfn, jk, budget, is_stateful=False)
        for pname, factory in stateful_pols:
            _run_gen_policy(pname, factory, jk, budget, is_stateful=True)

    return results


# ── Plotting helpers ────────────────────────────────────────────────────────

def _try_plot(plot_fn, path, verbose=True):
    """Run a plotting function; skip gracefully if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_fn(plt, path)
        if verbose:
            print(f"  Saved figure: {path}")
    except ImportError:
        if verbose:
            print("  matplotlib not available — skipping plot")


def _pub_style(plt):
    """Apply publication-quality rcParams."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# Colour-blind-friendly palette (Okabe-Ito)
_C = {
    "optimal":           "#009E73",  # green
    "cwm_greedy":        "#D55E00",  # vermillion
    "cwm_mcts":          "#CC79A7",  # pink
    "RLS_1":             "#0072B2",  # blue
    "RLS_jump_k":        "#56B4E9",  # sky blue
    "random_k":          "#999999",  # grey
    "fifth_rule":        "#E69F00",  # orange
    "self_adjusting":    "#F0E442",  # yellow
    "stagnation_detect": "#661100",  # dark red
    "CWM-greedy":        "#D55E00",
    "theory":            "#000000",
    "valley":            "#FFD1D1",  # light red for valley shading
}
_LABELS = {
    "optimal":           r"Optimal $k^*$",
    "cwm_greedy":        "CWM-greedy",
    "cwm_mcts":          "CWM-MCTS",
    "RLS_1":             r"RLS$_1$ ($k\!=\!1$)",
    "RLS_jump_k":        r"RLS$_{jk}$ ($k\!=\!$jump\_k)",
    "random_k":          "Random $k$",
    "fifth_rule":        r"EA$_\alpha$(2, 0.5)",
    "self_adjusting":    r"EA$_\alpha$(1.3, 0.75)",
    "stagnation_detect": "Stagnation detect",
    "CWM-greedy":        "CWM-greedy",
}


def plot_policy_comparison(learned_policy, n, jump_k, plt, path):
    """Fig 1: CWM learned k(fitness) vs optimal k*(fitness).

    Valley region highlighted. Should show CWM increasing k near valley edge.
    """
    _pub_style(plt)
    fitnesses = sorted(int(k) for k in learned_policy.keys())
    cwm_ks = [learned_policy[f]["cwm_best_k"] for f in fitnesses]
    opt_ks = [learned_policy[f]["optimal_k"] for f in fitnesses]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade valley edge region
    ax.axvspan(n - 1, n + 0.5, alpha=0.2, color=_C["valley"],
               label="Valley edge")

    ax.plot(fitnesses, opt_ks, color=_C["theory"], ls="-", lw=2.5,
            label=r"Optimal $k^*$(fitness)")
    ax.plot(fitnesses, cwm_ks, color=_C["cwm_greedy"], ls="--", marker="o",
            ms=3.5, lw=1.5, markevery=2,
            label="CWM learned $k$(fitness)")
    ax.axhline(1, color=_C["RLS_1"], ls=":", lw=1.2,
               label=r"RLS$_1$ ($k\!=\!1$)")
    ax.axhline(jump_k, color=_C["RLS_jump_k"], ls=":", lw=1.2,
               label=f"$k$ = jump_k = {jump_k}")

    # Mark valley edge
    ax.axvline(n, color="#999999", ls="-.", lw=1.0, alpha=0.6,
               label=f"Valley edge (fitness = {n})")

    ax.set_xlabel("Jump$_k$ fitness")
    ax.set_ylabel("Flip count $k$")
    ax.set_xlim(fitnesses[0], fitnesses[-1])
    ax.set_ylim(0, min(n + 1, max(max(cwm_ks), max(opt_ks)) + 2))
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.set_title(f"Jump$_{jump_k}$ (n={n}): Learned vs Optimal Policy")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_heatmap(heatmap_data, n, jump_k, plt, path):
    """Fig 2: Row-normalised CWM score heatmap (fitness x k) with valley."""
    _pub_style(plt)
    scores = np.array(heatmap_data["scores"], dtype=float)
    fitnesses = heatmap_data["fitnesses"]
    ks = heatmap_data["ks"]

    # Row-normalise
    for r in range(scores.shape[0]):
        row = scores[r]
        rmin, rmax = np.nanmin(row), np.nanmax(row)
        if rmax - rmin > 1e-12:
            scores[r] = (row - rmin) / (rmax - rmin)
        else:
            scores[r] = 0.5

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(scores, aspect="auto", origin="lower", cmap="RdYlGn",
                   extent=[-0.5, len(ks) - 0.5, fitnesses[0], fitnesses[-1]])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)

    # Overlay optimal k* line
    opt_line_x = []
    opt_line_y = []
    for fit in fitnesses:
        ok = _get_optimal_k(fit, n, jump_k)
        if ok in ks:
            opt_line_x.append(ks.index(ok))
            opt_line_y.append(fit)
    if opt_line_x:
        ax.plot(opt_line_x, opt_line_y, "k*", ms=8, label=r"$k^*$", zorder=5)

    # Mark the valley edge
    ax.axhline(n, color="white", ls="--", lw=1.5, alpha=0.7,
               label="Valley edge")

    ax.set_xlabel("Flip count $k$")
    ax.set_ylabel("Jump$_k$ fitness")
    ax.set_title(f"Jump$_{jump_k}$ CWM Score Heatmap")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Relative CWM score (row-normalised)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_convergence(policy_results, convergence, n, jump_k, budget, plt, path):
    """Fig 3: empirical CDF of steps to optimum. Shows EA_alpha failure."""
    _pub_style(plt)

    order = ["optimal", "cwm_greedy", "cwm_mcts", "stagnation_detect",
             "fifth_rule", "self_adjusting", "RLS_1", "RLS_jump_k", "random_k"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for pname in order:
        if pname not in policy_results:
            continue
        steps_list = policy_results[pname]
        arr = np.sort(steps_list)
        fracs = np.arange(1, len(arr) + 1) / len(arr)
        label = _LABELS.get(pname, pname)
        colour = _C.get(pname, "#333333")
        ax.step(arr, fracs, where="post", label=label, color=colour)

    # Mark budget
    ax.axvline(budget, color="#999999", ls=":", lw=1.0, alpha=0.5)
    ax.text(budget * 0.98, 0.5, "budget", fontsize=9, color="#999999",
            ha="right", rotation=90)

    ax.set_xlabel("Steps to optimum")
    ax.set_ylabel("Fraction of runs solved")
    ax.set_xlim(0, budget * 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title(f"Jump$_{jump_k}$ (n={n}): Convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_generalisation(gen_results, plt, path):
    """Fig 4: convergence rate across jump_k values."""
    _pub_style(plt)
    jump_ks = sorted(set(v["jump_k"] for v in gen_results.values()))
    pol_order = ["optimal", "CWM-greedy", "stagnation_detect",
                 "fifth_rule", "self_adjusting", "RLS_1", "RLS_jump_k"]
    pol_labels = [r"Optimal $k^*$", "CWM-greedy", "Stagnation detect",
                  r"EA$_\alpha$(2,.5)", r"EA$_\alpha$(1.3,.75)",
                  r"RLS$_1$", r"RLS$_{jk}$"]
    pol_colors = [_C["optimal"], _C["cwm_greedy"], _C["stagnation_detect"],
                  _C["fifth_rule"], _C["self_adjusting"], _C["RLS_1"],
                  _C["RLS_jump_k"]]

    x = np.arange(len(jump_ks))
    width = 0.12
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (pol, lab, col) in enumerate(zip(pol_order, pol_labels, pol_colors)):
        rates = []
        for jk in jump_ks:
            key = f"jk{jk}_{pol}"
            if key in gen_results:
                rates.append(gen_results[key]["convergence_rate"])
            else:
                rates.append(0)
        ax.bar(x + i * width, rates, width, label=lab,
               color=col, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (len(pol_order) - 1) / 2)
    ax.set_xticklabels([f"jump_k = {jk}" for jk in jump_ks])
    ax.set_ylabel("Convergence rate")
    ax.set_ylim(0, 1.1)
    ax.legend(framealpha=0.9, fontsize=9, ncol=2)
    ax.set_title("Generalisation across jump_k values")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(args):
    global _OPTIMAL_TABLE, _DEFAULT_N, _DEFAULT_JUMP_K

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    jump_k = args.jump_k
    budget = args.budget
    n_episodes = 10 if args.quick else args.episodes
    _DEFAULT_N = n
    _DEFAULT_JUMP_K = jump_k

    print("=" * 60)
    print("Experiment 10: Jump_k CWM Evaluation")
    print("=" * 60)
    print(f"  n={n}, jump_k={jump_k}, budget={budget}, episodes={n_episodes}")

    # Precompute optimal k table
    print(f"\n  Precomputing optimal k table for n={n}, jump_k={jump_k} ...")
    _OPTIMAL_TABLE = precompute_optimal_k_table(n, jump_k)
    print(f"  Done. k*(jump_k)={_OPTIMAL_TABLE[jump_k]}, "
          f"k*(n)={_OPTIMAL_TABLE[n]}, k*(n+jump_k)={_OPTIMAL_TABLE[n+jump_k]}")

    # Load CWM
    print(f"\nLoading CWM from: {args.cwm}")
    cwm, ns = load_jump_cwm(args.cwm)

    all_results: Dict[str, Any] = {
        "experiment": "exp10_jump_evaluation",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "jump_k": jump_k, "budget": budget,
            "episodes": n_episodes,
        },
    }

    # ── A. CWM Quality ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("A. CWM Quality (Kendall's tau)")
    print(f"{'='*60}")
    quality = evaluate_cwm_quality(cwm, ns, n=n, jump_k=jump_k, budget=budget)
    all_results["quality"] = quality

    # ── B. Policy Comparison ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Policy Comparison")
    print(f"{'='*60}")
    policy_results, convergence = run_policy_comparison(
        cwm, ns, n=n, jump_k=jump_k, n_episodes=n_episodes, budget=budget,
        mcts_sims=50 if not args.quick else 20,
        mcts_horizon=5,
    )
    summary, pairwise = compute_policy_statistics(policy_results, convergence)

    print(f"\n  {'Policy':<20s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Conv%':>7s}")
    print(f"  {'-'*55}")
    for pname in summary:
        s = summary[pname]
        print(f"  {pname:<20s} {s['mean']:>8.0f} {s['std']:>8.0f} "
              f"{s['median']:>8.0f} {s['convergence_rate']*100:>6.1f}%")

    print(f"\n  Pairwise tests (Mann-Whitney U):")
    for pair, test in pairwise.items():
        sig = "*" if test["p_value"] < 0.05 else " "
        print(f"    {pair:<40s}  p={test['p_value']:.4f}{sig}  "
              f"delta={test['cliff_delta']:.3f}")

    all_results["policy_comparison"] = {
        "summary": summary,
        "pairwise": pairwise,
        "raw_steps": {k: [int(s) for s in v] for k, v in policy_results.items()},
        "convergence": {k: [bool(c) for c in v] for k, v in convergence.items()},
    }

    # ── C. Interpretability ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. Interpretability")
    print(f"{'='*60}")
    learned = extract_learned_policy(cwm, ns, n=n, jump_k=jump_k, budget=budget)
    heatmap = build_heatmap_data(cwm, ns, n=n, jump_k=jump_k, budget=budget)

    # Correlation between CWM k and optimal k
    fit_range = list(range(jump_k, n + 1))
    cwm_ks = [learned[f]["cwm_best_k"] for f in fit_range]
    opt_ks = [_get_optimal_k(f, n, jump_k) for f in fit_range]
    corr, corr_p = sp_stats.spearmanr(cwm_ks, opt_ks)
    print(f"  Spearman corr(CWM_k, optimal_k) = {corr:.3f}  (p={corr_p:.2e})")

    # Fraction of exact matches
    exact = sum(1 for f in fit_range
                if learned[f]["cwm_best_k"] == learned[f]["optimal_k"])
    print(f"  Exact match fraction: {exact}/{len(fit_range)} = {exact/len(fit_range):.1%}")

    # Valley detection: does CWM increase k near valley edge?
    near_edge = [learned[f]["cwm_best_k"]
                 for f in range(max(jump_k, n - 3), n + 1) if f in learned]
    far_from_edge = [learned[f]["cwm_best_k"]
                     for f in range(jump_k, min(n - jump_k, n // 2))
                     if f in learned]
    if near_edge and far_from_edge:
        mean_near = np.mean(near_edge)
        mean_far = np.mean(far_from_edge)
        print(f"  Valley detection: mean k near edge = {mean_near:.1f}, "
              f"far from edge = {mean_far:.1f}")
        valley_detected = mean_near > mean_far
        print(f"  Valley detected by CWM: {valley_detected}")
    else:
        valley_detected = False

    all_results["interpretability"] = {
        "spearman_corr": float(corr),
        "spearman_p": float(corr_p),
        "exact_match_fraction": exact / len(fit_range),
        "valley_detected": valley_detected,
        "learned_policy": {str(k): v for k, v in learned.items()},
        "heatmap": heatmap,
    }

    # ── D. Generalisation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. Generalisation (across jump_k values)")
    print(f"{'='*60}")
    gen_jump_ks = [2, 3] if args.quick else [2, 3, 4]
    gen_episodes = max(10, n_episodes // 2)
    gen = run_generalisation(
        cwm, ns, n=n, jump_ks=gen_jump_ks, n_episodes=gen_episodes)
    all_results["generalisation"] = gen

    # ── Save results ─────────────────────────────────────────────────────
    results_file = output_dir / "exp10_results.json"

    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(v) for v in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(to_json(all_results), f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ── Plots ────────────────────────────────────────────────────────────
    _try_plot(
        lambda plt, p: plot_policy_comparison(learned, n, jump_k, plt, p),
        output_dir / "fig_policy_comparison.png",
    )
    _try_plot(
        lambda plt, p: plot_heatmap(heatmap, n, jump_k, plt, p),
        output_dir / "fig_heatmap.png",
    )
    _try_plot(
        lambda plt, p: plot_convergence(policy_results, convergence, n,
                                        jump_k, budget, plt, p),
        output_dir / "fig_convergence.png",
    )
    _try_plot(
        lambda plt, p: plot_generalisation(gen, plt, p),
        output_dir / "fig_generalisation.png",
    )

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  A. Mean Kendall's tau         = {quality['mean_tau']:.3f}")
    print(f"     Valley-edge tau            = {quality['valley_edge_tau']:.3f}")
    print(f"  B. CWM-greedy mean steps      = {summary['cwm_greedy']['mean']:.0f}  "
          f"conv={summary['cwm_greedy']['convergence_rate']:.0%}")
    print(f"     CWM-MCTS   mean steps      = {summary['cwm_mcts']['mean']:.0f}  "
          f"conv={summary['cwm_mcts']['convergence_rate']:.0%}")
    print(f"     Optimal    mean steps      = {summary['optimal']['mean']:.0f}  "
          f"conv={summary['optimal']['convergence_rate']:.0%}")
    ea_info = summary.get('fifth_rule', {})
    sa_info = summary.get('self_adjusting', {})
    print(f"     EA_a(2,.5) mean steps      = {ea_info.get('mean', 'N/A')}  "
          f"conv={ea_info.get('convergence_rate', 0):.0%}")
    print(f"     EA_a(1.3,.75) mean steps   = {sa_info.get('mean', 'N/A')}  "
          f"conv={sa_info.get('convergence_rate', 0):.0%}")
    print(f"     RLS_1      mean steps      = {summary['RLS_1']['mean']:.0f}  "
          f"conv={summary['RLS_1']['convergence_rate']:.0%}")
    print(f"     RLS_jump_k mean steps      = {summary['RLS_jump_k']['mean']:.0f}  "
          f"conv={summary['RLS_jump_k']['convergence_rate']:.0%}")
    print(f"  C. Spearman corr              = {corr:.3f}")
    print(f"     Valley detected            = {valley_detected}")
    print(f"  D. Generalisation:")
    for jk in gen_jump_ks:
        cwm_key = f"jk{jk}_CWM-greedy"
        ea_key = f"jk{jk}_fifth_rule"
        cwm_conv = gen.get(cwm_key, {}).get("convergence_rate", 0)
        ea_conv = gen.get(ea_key, {}).get("convergence_rate", 0)
        print(f"     jump_k={jk}: CWM conv={cwm_conv:.0%}, EA_a conv={ea_conv:.0%}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp37: Jump_k CWM full evaluation")
    parser.add_argument("--cwm", type=str,
                        default="results/cwm/jump_cwm.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length")
    parser.add_argument("--jump-k", type=int, default=2,
                        help="Jump gap parameter")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per policy")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Step budget (default 10000)")
    parser.add_argument("--output", type=str, default="results/exp10",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
