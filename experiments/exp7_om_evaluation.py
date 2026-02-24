#!/usr/bin/env python3
"""
Experiment 7: OneMax CWM Evaluation

Four sub-experiments:

  A. CWM Quality – Kendall's tau between CWM action ranking and ground-truth
     ranking at several fitness snapshots.
  B. Policy Comparison – 6 policies (RLS_1, RLS_2, random_k, optimal,
     CWM-greedy, CWM-MCTS) compared by steps-to-optimum over 100 episodes.
  C. Interpretability – extract the CWM's learned policy and plot vs the
     theoretical optimum. Should reveal the "cliff" at i ~ n/2.
  D. Generalisation – train on n=50, evaluate on n=100 and n=200.

Outputs:
  results/exp7/exp7_results.json        — all numerical data
  results/exp7/fig_policy_comparison.png  — learned vs optimal k(i)
  results/exp7/fig_heatmap.png            — CWM score heatmap
  results/exp7/fig_convergence.png        — convergence curves
  results/exp7/fig_generalisation.png     — generalisation bar chart
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

from src.onemax import (
    OneMaxProblem,
    RLSOptimizer,
    optimal_k_onemax,
    approx_optimal_k_onemax,
    precompute_optimal_k_table,
    run_episode,
)


# ── OMState / OMAction (same as in exp6) ───────────────────────────────────

def _make_om_namespace():
    """Execution namespace with OMState & OMAction."""
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
    ns: Dict[str, Any] = {}
    exec(setup, ns)
    return ns


def load_om_cwm(cwm_path: str):
    """Load a synthesised OneMax CWM.

    Returns (cwm_instance, namespace) so callers can create OMState/OMAction.
    """
    with open(cwm_path) as f:
        code = f.read()
    ns = _make_om_namespace()
    exec(code, ns)
    cwm = ns["SynthesizedCWM"]()
    return cwm, ns


# ── Helpers ─────────────────────────────────────────────────────────────────

def _om_state(ns, fitness: int, n: int, step: int, budget: int):
    return ns["OMState"](
        fitness=fitness, n=n, step=step, budget=budget,
        normalized_fitness=fitness / n,
    )


def _om_action(ns, k: int):
    return ns["OMAction"](k=k, name=f"flip_{k}")


# ── Adaptive baselines ─────────────────────────────────────────────────────

def _ea_alpha_policy(n, A=2.0, b=0.5):
    """(1+1) EA_α adapted to RLS_k (after Doerr & Wagner, GECCO 2018,
    arXiv:1803.01425, Algorithm 2).

    The original EA_α operates on mutation rate p with standard bit
    mutation.  We adapt it to our (1+1)-RLS_k setting by applying the
    same multiplicative update directly to k (the number of bits to
    flip), which is the natural translation for deterministic k-bit
    flip operators.

    Update rule (applied to k directly):
      On success (strict fitness improvement):  k ← min(A · k, n/2)
      On failure (no improvement):              k ← max(b · k,   1)

    Default A=2, b=0.5 ("doubling/halving") corresponds to one of the
    best configurations in their grid search (Table 1).

    Reference: Doerr, C. & Wagner, M. "On the Effectiveness of Simple
    Success-Based Parameter Selection Mechanisms for Two Classical
    Discrete Black-Box Optimization Benchmark Problems", GECCO 2018.
    """
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


def _self_adjusting_policy(n, A=1.3, b=0.75):
    """(1+1) EA_α (best config) adapted to RLS_k (after Doerr & Wagner,
    GECCO 2018, arXiv:1803.01425, Table 1).

    Same algorithm as _ea_alpha_policy but with A=1.3, b=0.75, which
    achieved ~18% improvement over RLS on LeadingOnes in the original
    paper (using standard bit mutation).

    Note: The true "self-adjusting" two-rate mechanism (Doerr, Gießen,
    Witt & Yang, Algorithmica 2018, Algorithm 1) requires λ >= 2
    offspring per generation and cannot be directly applied to
    (1+1)-RLS.  We use the single-offspring EA_α instead.
    """
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


# ── A. CWM Quality ─────────────────────────────────────────────────────────

def evaluate_cwm_quality(
    cwm, ns,
    n: int = 50,
    budget: int = 978,
    fitness_snapshots: list = None,
    test_ks: list = None,
    n_trials: int = 200,
    verbose: bool = True,
):
    """For each fitness snapshot, compare CWM action ranking against ground truth.

    Ground truth: for each k in test_ks, run n_trials actual RLS steps from
    that fitness level and record the average improvement.
    CWM ranking: evaluate_state(predict_next_state(state, action_k)).

    Returns dict with per-snapshot Kendall's tau.
    """
    if fitness_snapshots is None:
        fitness_snapshots = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    if test_ks is None:
        test_ks = [1, 2, 3, 5, 8, 10, 15, 25, 50]

    problem = OneMaxProblem(n)
    results: Dict[int, Any] = {}

    for fit in fitness_snapshots:
        if fit >= n:
            continue

        # Build a canonical bitstring: first `fit` bits = 1, rest = 0
        base_x = np.zeros(n, dtype=int)
        base_x[:fit] = 1

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
        state = _om_state(ns, fit, n, step=0, budget=budget)
        cwm_scores = []
        for k in test_ks:
            action = _om_action(ns, k)
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
        opt_k = optimal_k_onemax(fit, n)

        results[fit] = {
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0,
            "gt_best_k": gt_best_k,
            "cwm_best_k": cwm_best_k,
            "optimal_k": opt_k,
            "gt_scores": [float(s) for s in gt_scores],
            "cwm_scores": [float(s) for s in cwm_scores],
        }

        if verbose:
            print(f"  fitness={fit:3d}  tau={tau:.3f}  gt_best_k={gt_best_k}  "
                  f"cwm_best_k={cwm_best_k}  optimal_k={opt_k}")

    # Aggregate
    taus = [v["kendall_tau"] for v in results.values()]
    mean_tau = float(np.mean(taus))
    if verbose:
        print(f"\n  Mean Kendall's tau = {mean_tau:.3f}")

    return {"per_fitness": results, "mean_tau": mean_tau, "test_ks": test_ks}


# ── B. Policy Comparison ───────────────────────────────────────────────────

# Precomputed optimal k table (populated at runtime)
_OPTIMAL_TABLE = None


def _get_optimal_k(fitness: int, n: int) -> int:
    global _OPTIMAL_TABLE
    if _OPTIMAL_TABLE is not None and len(_OPTIMAL_TABLE) == n + 1:
        return _OPTIMAL_TABLE[min(fitness, n)]
    return optimal_k_onemax(fitness, n)


def _cwm_greedy_policy(cwm, ns, n, budget):
    """1-step lookahead: pick the k that maximises evaluate_state."""
    candidate_ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 25]
                               + [max(1, n - i) for i in range(0, n, 5)]
                               + [n]))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    def policy(fitness, n_, step):
        state = _om_state(ns, fitness, n_, step, budget)
        best_k, best_score = 1, float("-inf")
        for k in candidate_ks:
            action = _om_action(ns, k)
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


def _cwm_mcts_policy(cwm, ns, n, budget, simulations=50, horizon=5):
    """MCTS policy using the CWM for rollouts."""

    candidate_ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 25]
                               + [max(1, n - i) for i in range(0, n, 5)]
                               + [n]))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    def policy(fitness, n_, step):
        root_state = _om_state(ns, fitness, n_, step, budget)

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
            action = _om_action(ns, best_k)
            try:
                state = cwm.predict_next_state(root_state, action)
            except Exception:
                state = root_state

            value = 0.0
            discount = 1.0
            for h in range(horizon):
                try:
                    if cwm.is_terminal(state):
                        break
                    # Random rollout action
                    rk = candidate_ks[np.random.randint(len(candidate_ks))]
                    ra = _om_action(ns, rk)
                    state = cwm.predict_next_state(state, ra)
                except Exception:
                    break
                discount *= 0.99

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
    n_episodes: int = 100,
    budget: int = 978,
    mcts_sims: int = 50,
    mcts_horizon: int = 5,
    verbose: bool = True,
):
    """Compare policies over n_episodes.

    Returns dict mapping policy_name -> list of steps-to-optimum.
    """
    # Stateful policies need to be re-created per episode; others are reusable
    stateless_policies = {
        "RLS_1":      lambda f, n_, s: 1,
        "RLS_2":      lambda f, n_, s: 2,
        "random_k":   lambda f, n_, s: np.random.randint(1, n_ + 1),
        "optimal":    lambda f, n_, s: _get_optimal_k(f, n_),
        "cwm_greedy": _cwm_greedy_policy(cwm, ns, n, budget),
        "cwm_mcts":   _cwm_mcts_policy(cwm, ns, n, budget, mcts_sims, mcts_horizon),
    }
    # Adaptive baselines: factory functions (re-created per episode)
    stateful_factories = {
        "fifth_rule":     lambda: _ea_alpha_policy(n),
        "self_adjusting": lambda: _self_adjusting_policy(n),
    }

    all_policy_names = list(stateless_policies.keys()) + list(stateful_factories.keys())
    results: Dict[str, list] = {name: [] for name in all_policy_names}

    for pname in all_policy_names:
        if verbose:
            print(f"\n  Policy: {pname}")
        for ep in range(n_episodes):
            seed = ep * 31
            if pname in stateless_policies:
                pfn = stateless_policies[pname]
            else:
                pfn = stateful_factories[pname]()
            episode = run_episode(n, pfn, budget, seed=seed)
            results[pname].append(episode["total_steps"])

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                mean_so_far = np.mean(results[pname])
                print(f"    Episode {ep+1}/{n_episodes}  mean_steps={mean_so_far:.0f}")

    return results


def compute_policy_statistics(results: Dict[str, list]):
    """Compute summary stats + pairwise Wilcoxon tests."""
    summary = {}
    for name, steps in results.items():
        arr = np.array(steps, dtype=float)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
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
                # Cliff's delta
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

def extract_learned_policy(cwm, ns, n: int = 50, budget: int = 978):
    """For each fitness level, find the k that the CWM rates highest.

    Returns dict fitness -> {"cwm_best_k": int, "optimal_k": int, "scores": dict}
    """
    candidate_ks = list(range(1, n + 1))
    learned = {}

    for fit in range(n):
        state = _om_state(ns, fit, n, step=0, budget=budget)
        scores = {}
        for k in candidate_ks:
            action = _om_action(ns, k)
            try:
                pred = cwm.predict_next_state(state, action)
                scores[k] = cwm.evaluate_state(pred)
            except Exception:
                scores[k] = float("-inf")

        best_k = max(candidate_ks, key=lambda k: scores[k])
        learned[fit] = {
            "cwm_best_k": best_k,
            "optimal_k": _get_optimal_k(fit, n),
            "scores": {str(k): float(v) for k, v in scores.items()
                       if not np.isinf(v)},
        }

    return learned


def build_heatmap_data(cwm, ns, n: int = 50, budget: int = 978):
    """Build fitness x k score matrix for heatmap visualisation."""
    fitnesses = list(range(0, n, max(1, n // 20)))
    ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]))
    ks = [k for k in ks if k <= n]

    matrix = []
    for fit in fitnesses:
        row = []
        state = _om_state(ns, fit, n, step=0, budget=budget)
        for k in ks:
            action = _om_action(ns, k)
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


# ── D. Generalisation ──────────────────────────────────────────────────────

def run_generalisation(
    cwm, ns,
    sizes: list = None,
    n_episodes: int = 50,
    verbose: bool = True,
):
    """Evaluate the CWM (trained on n=50) on larger problem sizes."""
    if sizes is None:
        sizes = [50, 100, 200]

    results = {}
    for n in sizes:
        budget = int(5 * n * math.log(n))
        if verbose:
            print(f"\n  n={n}, budget={budget}")

        greedy_fn = _cwm_greedy_policy(cwm, ns, n, budget)
        # Use approximation for large n to avoid O(n^4) computation
        if n <= 50:
            optimal_fn = lambda f, n_=n, s=None: _get_optimal_k(f, n_)
        else:
            optimal_fn = lambda f, n_=n, s=None: approx_optimal_k_onemax(f, n_)
        static1_fn = lambda f, n_, s: 1

        stateless_pols = [("optimal", optimal_fn), ("CWM-greedy", greedy_fn),
                          ("RLS_1", static1_fn)]
        stateful_pols = [("fifth_rule", lambda n_=n: _ea_alpha_policy(n_)),
                         ("self_adjusting", lambda n_=n: _self_adjusting_policy(n_))]

        def _run_gen_policy(pname, pfn_or_factory, is_stateful=False):
            steps_list = []
            for ep in range(n_episodes):
                seed = ep * 37 + n
                pfn = pfn_or_factory() if is_stateful else pfn_or_factory
                episode = run_episode(n, pfn, budget, seed=seed)
                steps_list.append(episode["total_steps"])

            nlogn = n * math.log(n)
            key = f"n{n}_{pname}"
            results[key] = {
                "n": n,
                "policy": pname,
                "mean_steps": float(np.mean(steps_list)),
                "std_steps": float(np.std(steps_list)),
                "median_steps": float(np.median(steps_list)),
                "normalised_mean": float(np.mean(steps_list)) / nlogn,
                "normalised_std": float(np.std(steps_list)) / nlogn,
                "convergence_rate": float(np.mean([
                    1 if s < budget else 0 for s in steps_list
                ])),
            }
            if verbose:
                print(f"    {pname:<16s}: mean={results[key]['mean_steps']:.0f}  "
                      f"T/(n ln n)={results[key]['normalised_mean']:.2f}  "
                      f"conv={results[key]['convergence_rate']:.0%}")

        for pname, pfn in stateless_pols:
            _run_gen_policy(pname, pfn, is_stateful=False)
        for pname, factory in stateful_pols:
            _run_gen_policy(pname, factory, is_stateful=True)

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
    "optimal":        "#009E73",  # green
    "cwm_greedy":     "#D55E00",  # vermillion
    "cwm_mcts":       "#CC79A7",  # pink
    "RLS_1":          "#0072B2",  # blue
    "RLS_2":          "#56B4E9",  # sky blue
    "random_k":       "#999999",  # grey
    "fifth_rule":     "#E69F00",  # orange
    "self_adjusting": "#F0E442",  # yellow
    "static":         "#0072B2",
    "theory":         "#000000",
    "CWM-greedy":     "#D55E00",
}
_LABELS = {
    "optimal":        r"Optimal $k^*(i)$",
    "cwm_greedy":     "CWM-greedy",
    "cwm_mcts":       "CWM-MCTS",
    "RLS_1":          r"RLS$_1$ (static $k\!=\!1$)",
    "RLS_2":          r"RLS$_2$ (static $k\!=\!2$)",
    "random_k":       "Random $k$",
    "fifth_rule":     r"EA$_\alpha$(2, 0.5)",
    "self_adjusting": r"EA$_\alpha$(1.3, 0.75)",
    "CWM-greedy":     "CWM-greedy",
}


def plot_policy_comparison(learned_policy, n, plt, path):
    """Fig 1: CWM learned k(i) vs optimal k*(i). Should show "cliff" at n/2."""
    _pub_style(plt)
    fitnesses = sorted(int(k) for k in learned_policy.keys())
    cwm_ks = [learned_policy[f]["cwm_best_k"] for f in fitnesses]
    opt_ks = [learned_policy[f]["optimal_k"] for f in fitnesses]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fitnesses, opt_ks, color=_C["theory"], ls="-", lw=2.5,
            label=r"Optimal $k^*(i)$ (exact)")
    ax.plot(fitnesses, cwm_ks, color=_C["cwm_greedy"], ls="--", marker="o",
            ms=3.5, lw=1.5, markevery=2,
            label="CWM learned $k(i)$")
    ax.axhline(1, color=_C["RLS_1"], ls=":", lw=1.2, label=r"RLS$_1$ ($k\!=\!1$)")
    ax.axvline(n / 2, color="#999999", ls="-.", lw=1.0, alpha=0.6,
               label=r"$i = n/2$ (cliff)")

    ax.set_xlabel("OneMax fitness $i$")
    ax.set_ylabel("Flip count $k$")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, n + 1)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_heatmap(heatmap_data, n, plt, path):
    """Fig 2: Row-normalised CWM score heatmap (fitness x k)."""
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

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(scores, aspect="auto", origin="lower", cmap="RdYlGn",
                   extent=[-0.5, len(ks) - 0.5, fitnesses[0], fitnesses[-1]])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)

    # Overlay optimal k* line
    opt_line_x = []
    opt_line_y = []
    for fit in fitnesses:
        ok = _get_optimal_k(fit, n)
        if ok in ks:
            opt_line_x.append(ks.index(ok))
            opt_line_y.append(fit)
    if opt_line_x:
        ax.plot(opt_line_x, opt_line_y, "k*", ms=8, label=r"$k^*(i)$", zorder=5)

    # Mark the cliff at n/2
    ax.axhline(n / 2, color="white", ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel("Flip count $k$")
    ax.set_ylabel("OneMax fitness $i$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Relative CWM score (row-normalised)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_convergence(policy_results, n, plt, path):
    """Fig 3: empirical CDF of steps to optimum."""
    _pub_style(plt)

    # Order for legend
    order = ["optimal", "cwm_greedy", "cwm_mcts", "fifth_rule", "self_adjusting",
             "RLS_1", "RLS_2", "random_k"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pname in order:
        if pname not in policy_results:
            continue
        steps_list = policy_results[pname]
        arr = np.sort(steps_list)
        fracs = np.arange(1, len(arr) + 1) / len(arr)
        label = _LABELS.get(pname, pname)
        colour = _C.get(pname, "#333333")
        ax.step(arr, fracs, where="post", label=label, color=colour)

    # Reference: O(n ln n) theoretical bound
    nlogn = n * math.log(n)
    ax.axvline(nlogn, color="#999999", ls=":", lw=1.0, alpha=0.5)
    ax.text(nlogn * 1.02, 0.5, r"$n \ln n$", fontsize=9, color="#999999")

    ax.set_xlabel("Steps to optimum")
    ax.set_ylabel("Fraction of runs solved")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_generalisation(gen_results, plt, path):
    """Fig 4: normalised steps (T / (n ln n)) across problem sizes."""
    _pub_style(plt)
    sizes = sorted(set(v["n"] for v in gen_results.values()))
    # Fixed policy order
    pol_order = ["optimal", "CWM-greedy", "fifth_rule", "self_adjusting", "RLS_1"]
    pol_labels = [r"Optimal $k^*(i)$", "CWM-greedy", r"EA$_\alpha$(2,.5)",
                  r"EA$_\alpha$(1.3,.75)", r"RLS$_1$"]
    pol_colors = [_C["optimal"], _C["cwm_greedy"], _C["fifth_rule"],
                  _C["self_adjusting"], _C["RLS_1"]]

    x = np.arange(len(sizes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (pol, lab, col) in enumerate(zip(pol_order, pol_labels, pol_colors)):
        means = []
        stds = []
        for sz in sizes:
            key = f"n{sz}_{pol}"
            if key in gen_results:
                nlogn = sz * math.log(sz)
                means.append(gen_results[key]["mean_steps"] / nlogn)
                stds.append(gen_results[key]["std_steps"] / nlogn)
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, label=lab,
               color=col, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"$n = {s}$" for s in sizes])
    ax.set_ylabel(r"Normalised steps  $T\,/\,(n \ln n)$")
    ax.legend(framealpha=0.9)

    # Reference lines for theoretical bounds
    ax.axhline(1.0, color=_C["optimal"], ls=":", lw=1, alpha=0.5)
    ax.text(x[-1] + 2.2 * width, 1.0, r"$\approx n \ln n$", fontsize=9,
            color=_C["optimal"], va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(args):
    global _OPTIMAL_TABLE

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    budget = args.budget or int(5 * n * math.log(n))
    n_episodes = 10 if args.quick else args.episodes

    print("=" * 60)
    print("Experiment 7: OneMax CWM Evaluation")
    print("=" * 60)
    print(f"  n={n}, budget={budget}, episodes={n_episodes}")

    # Precompute optimal k table
    print(f"\n  Precomputing optimal k table for n={n} ...")
    _OPTIMAL_TABLE = precompute_optimal_k_table(n)
    print(f"  Done. k*(0)={_OPTIMAL_TABLE[0]}, k*(n//2)={_OPTIMAL_TABLE[n//2]}, k*(n-1)={_OPTIMAL_TABLE[n-1]}")

    # Load CWM
    print(f"\nLoading CWM from: {args.cwm}")
    cwm, ns = load_om_cwm(args.cwm)

    all_results: Dict[str, Any] = {
        "experiment": "exp7_om_evaluation",
        "timestamp": datetime.now().isoformat(),
        "parameters": {"n": n, "budget": budget, "episodes": n_episodes},
    }

    # ── A. CWM Quality ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("A. CWM Quality (Kendall's tau)")
    print(f"{'='*60}")
    quality = evaluate_cwm_quality(cwm, ns, n=n, budget=budget)
    all_results["quality"] = quality

    # ── B. Policy Comparison ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Policy Comparison")
    print(f"{'='*60}")
    policy_results = run_policy_comparison(
        cwm, ns, n=n, n_episodes=n_episodes, budget=budget,
        mcts_sims=50 if not args.quick else 20,
        mcts_horizon=5,
    )
    summary, pairwise = compute_policy_statistics(policy_results)

    print(f"\n  {'Policy':<18s} {'Mean':>8s} {'Std':>8s} {'Median':>8s}")
    print(f"  {'-'*44}")
    for pname in summary:
        s = summary[pname]
        print(f"  {pname:<18s} {s['mean']:>8.0f} {s['std']:>8.0f} {s['median']:>8.0f}")

    print(f"\n  Pairwise tests (Mann-Whitney U):")
    for pair, test in pairwise.items():
        sig = "*" if test["p_value"] < 0.05 else " "
        print(f"    {pair:<35s}  p={test['p_value']:.4f}{sig}  "
              f"delta={test['cliff_delta']:.3f}")

    all_results["policy_comparison"] = {
        "summary": summary,
        "pairwise": pairwise,
        "raw_steps": {k: [int(s) for s in v] for k, v in policy_results.items()},
    }

    # ── C. Interpretability ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. Interpretability")
    print(f"{'='*60}")
    learned = extract_learned_policy(cwm, ns, n=n, budget=budget)
    heatmap = build_heatmap_data(cwm, ns, n=n, budget=budget)

    # Quick summary: correlation between CWM k and optimal k
    cwm_ks = [learned[f]["cwm_best_k"] for f in range(n)]
    opt_ks = [_get_optimal_k(f, n) for f in range(n)]
    corr, corr_p = sp_stats.spearmanr(cwm_ks, opt_ks)
    print(f"  Spearman corr(CWM_k, optimal_k) = {corr:.3f}  (p={corr_p:.2e})")

    # Fraction of exact matches
    exact = sum(1 for f in range(n) if learned[f]["cwm_best_k"] == learned[f]["optimal_k"])
    print(f"  Exact match fraction: {exact}/{n} = {exact/n:.1%}")

    # Check for cliff detection
    below_half = [learned[f]["cwm_best_k"] for f in range(n // 4, n // 2)]
    above_half = [learned[f]["cwm_best_k"] for f in range(n // 2 + 1, 3 * n // 4)]
    if below_half and above_half:
        mean_below = np.mean(below_half)
        mean_above = np.mean(above_half)
        print(f"  Cliff detection: mean k below n/2 = {mean_below:.1f}, "
              f"above n/2 = {mean_above:.1f}")

    all_results["interpretability"] = {
        "spearman_corr": float(corr),
        "spearman_p": float(corr_p),
        "exact_match_fraction": exact / n,
        "learned_policy": {str(k): v for k, v in learned.items()},
        "heatmap": heatmap,
    }

    # ── D. Generalisation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. Generalisation")
    print(f"{'='*60}")
    gen_sizes = [50, 100] if args.quick else [50, 100, 200]
    gen_episodes = max(10, n_episodes // 2)
    gen = run_generalisation(cwm, ns, sizes=gen_sizes, n_episodes=gen_episodes)
    all_results["generalisation"] = gen

    # ── Save results ─────────────────────────────────────────────────────
    results_file = output_dir / "exp7_results.json"

    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
        lambda plt, p: plot_policy_comparison(learned, n, plt, p),
        output_dir / "fig_policy_comparison.png",
    )
    _try_plot(
        lambda plt, p: plot_heatmap(heatmap, n, plt, p),
        output_dir / "fig_heatmap.png",
    )
    _try_plot(
        lambda plt, p: plot_convergence(policy_results, n, plt, p),
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
    print(f"  A. Mean Kendall's tau       = {quality['mean_tau']:.3f}")
    print(f"  B. CWM-greedy mean steps    = {summary['cwm_greedy']['mean']:.0f}")
    print(f"     CWM-MCTS   mean steps    = {summary['cwm_mcts']['mean']:.0f}")
    print(f"     Optimal    mean steps    = {summary['optimal']['mean']:.0f}")
    print(f"     EA_a(2,.5) mean steps    = {summary.get('fifth_rule', {}).get('mean', 'N/A')}")
    print(f"     EA_a(1.3,.75) mean steps = {summary.get('self_adjusting', {}).get('mean', 'N/A')}")
    print(f"     RLS_1      mean steps    = {summary['RLS_1']['mean']:.0f}")
    print(f"     RLS_2      mean steps    = {summary['RLS_2']['mean']:.0f}")
    print(f"  C. Spearman corr            = {corr:.3f}")
    print(f"  D. Generalisation n=100     = "
          f"{gen.get('n100_CWM-greedy', {}).get('mean_steps', 'N/A')}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp34: OneMax CWM full evaluation")
    parser.add_argument("--cwm", type=str,
                        default="results/cwm/onemax_cwm.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per policy")
    parser.add_argument("--budget", type=int, default=None,
                        help="Step budget (default 5*n*ln(n))")
    parser.add_argument("--output", type=str, default="results/exp7",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
