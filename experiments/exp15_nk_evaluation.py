#!/usr/bin/env python3
"""
Experiment 15: NK-Landscape CWM Evaluation

Sub-experiments:

  A. CWM Quality – Compare CWM action ranking against empirical best
     action ranking at several fitness levels.
  B. Policy Comparison – 8 policies compared by best fitness achieved
     over 100 episodes (since NK has no known optimum, we compare fitness).
  C. Interpretability – extract the CWM's learned policy and plot heatmap.
  D. Generalisation – CWM trained on K=2, evaluated on K=2, K=3, K=4.

Outputs:
  results/exp15/exp15_results.json
  results/exp15/fig_policy_comparison.png
  results/exp15/fig_heatmap.png
  results/exp15/fig_convergence.png
  results/exp15/fig_generalisation.png
  paper/figures/nk_heatmap.png
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

from src.nk_landscape import (
    NKLandscapeProblem,
    RLSOptimizer,
    run_episode,
)


# ── NKState / NKAction (same as in exp14) ─────────────────────────────────

def _make_nk_namespace():
    """Execution namespace with NKState & NKAction."""
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
    ns: Dict[str, Any] = {}
    exec(setup, ns)
    return ns


def load_nk_cwm(cwm_path: str):
    """Load a synthesised NK-Landscape CWM."""
    with open(cwm_path) as f:
        code = f.read()
    ns = _make_nk_namespace()
    exec(code, ns)
    cwm = ns["SynthesizedCWM"]()
    return cwm, ns


# ── Helpers ─────────────────────────────────────────────────────────────────

def _nk_state(ns, fitness, n, K, step, budget, max_fitness, stagnation=0):
    return ns["NKState"](
        fitness=fitness, n=n, K=K, step=step, budget=budget,
        normalized_fitness=fitness / max_fitness if max_fitness > 0 else 0.0,
        stagnation_counter=stagnation,
    )


def _nk_action(ns, k):
    return ns["NKAction"](k=k, name=f"flip_{k}")


# ── Adaptive baselines ─────────────────────────────────────────────────────

def _ea_alpha_policy(n, A=2.0, b=0.5):
    """EA_alpha adapted to RLS_k."""
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, K, step, stagnation_counter):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


def _self_adjusting_policy(n, A=1.3, b=0.75):
    """EA_alpha (best config) adapted to RLS_k."""
    state = {"k": 1.0, "prev_fitness": None}

    def policy(fitness, n_, K, step, stagnation_counter):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(n_, int(round(state["k"]))))

    return policy


def _stagnation_detect_policy(patience=50, escape_k=10):
    """Stagnation-detection baseline."""
    state = {"no_improve_count": 0, "prev_fitness": None}

    def policy(fitness, n_, K, step, stagnation_counter):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:  # strict: ties = stagnation
                state["no_improve_count"] = 0
            else:
                state["no_improve_count"] += 1
        state["prev_fitness"] = fitness
        if state["no_improve_count"] >= patience:
            return escape_k
        return 1

    return policy


def _adaptive_fitness_policy(max_fitness):
    """k = max(1, int(n * (1 - fitness/max_fitness)))"""
    def policy(fitness, n_, K, step, stagnation_counter):
        ratio = fitness / max_fitness if max_fitness > 0 else 0.5
        return max(1, int(n_ * (1 - ratio)))
    return policy


# ── A. CWM Quality ─────────────────────────────────────────────────────────

def compute_empirical_best_k(instance_path, fitness_levels, test_ks,
                              n_trials=200, verbose=True):
    """For each fitness level, run trials to find empirically best k.

    Since NK has no closed-form, we approximate ground truth by brute force:
    for each (fitness_level, k) pair, run n_trials mutations and compute
    mean fitness improvement.
    """
    problem = NKLandscapeProblem.load_instance(instance_path)
    n = problem.n

    empirical_best = {}

    for fit_level in fitness_levels:
        # Generate a bitstring at approximately this fitness level
        # by running RLS_1 until we reach near this fitness
        best_x = None
        best_diff = float('inf')

        for seed in range(20):
            rng = np.random.RandomState(seed * 100)
            x = problem.random_bitstring(rng)
            # Hill-climb with k=1 to get near target fitness
            current_f = problem.fitness(x)
            for _ in range(5000):
                pos = rng.choice(n, 1, replace=False)
                y = x.copy()
                y[pos] = 1 - y[pos]
                new_f = problem.fitness(y)
                if new_f >= current_f:
                    x = y
                    current_f = new_f
                diff = abs(current_f - fit_level)
                if diff < best_diff:
                    best_diff = diff
                    best_x = x.copy()
                if diff < 0.5:
                    break
            if best_diff < 0.5:
                break

        if best_x is None:
            continue

        base_fitness = problem.fitness(best_x)

        # For each k, run n_trials mutations and compute mean improvement
        k_scores = {}
        for k in test_ks:
            improvements = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial * 1000 + int(fit_level * 100) + k)
                positions = rng.choice(n, min(k, n), replace=False)
                y = best_x.copy()
                y[positions] = 1 - y[positions]
                new_f = problem.fitness(y)
                # Non-strict selection
                accepted_f = new_f if new_f >= base_fitness else base_fitness
                improvements.append(accepted_f - base_fitness)
            k_scores[k] = float(np.mean(improvements))

        best_k = max(test_ks, key=lambda k: k_scores[k])
        empirical_best[fit_level] = {
            "best_k": best_k,
            "k_scores": k_scores,
            "base_fitness": base_fitness,
        }

        if verbose:
            print(f"  fitness≈{fit_level:.0f}: empirical best k={best_k}  "
                  f"(base={base_fitness:.2f})")

    return empirical_best


def evaluate_cwm_quality(cwm, ns, instance_path, n=50, K=2, budget=10000,
                          fitness_snapshots=None, test_ks=None,
                          n_trials=200, verbose=True):
    """Compare CWM action ranking against empirical ground truth."""
    problem = NKLandscapeProblem.load_instance(instance_path)
    max_fitness = problem.max_fitness()

    if fitness_snapshots is None:
        fitness_snapshots = [10, 15, 20, 25, 30, 35, 38, 40]

    if test_ks is None:
        test_ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 25, 50]))
        test_ks = [k for k in test_ks if 1 <= k <= n]

    if verbose:
        print("  Computing empirical best k (this may take a moment)...")

    empirical = compute_empirical_best_k(
        instance_path, fitness_snapshots, test_ks, n_trials, verbose)

    results = {}
    for fit_level in fitness_snapshots:
        if fit_level not in empirical:
            continue

        emp = empirical[fit_level]
        gt_scores = [emp["k_scores"].get(k, 0.0) for k in test_ks]

        # CWM ranking
        state = _nk_state(ns, emp["base_fitness"], n, K, step=0,
                           budget=budget, max_fitness=max_fitness)
        cwm_scores = []
        for k in test_ks:
            action = _nk_action(ns, k)
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

        gt_best_k = test_ks[int(np.argmax(gt_arr))]
        cwm_best_k = test_ks[int(np.argmax(cwm_arr))]

        results[fit_level] = {
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0,
            "gt_best_k": gt_best_k,
            "cwm_best_k": cwm_best_k,
            "gt_scores": [float(s) for s in gt_scores],
            "cwm_scores": [float(s) for s in cwm_scores],
        }

        if verbose:
            print(f"  fitness≈{fit_level:.0f}  tau={tau:.3f}  "
                  f"gt_best_k={gt_best_k}  cwm_best_k={cwm_best_k}")

    taus = [v["kendall_tau"] for v in results.values()]
    mean_tau = float(np.mean(taus)) if taus else 0.0

    if verbose:
        print(f"\n  Mean Kendall's tau = {mean_tau:.3f}")

    return {
        "per_fitness": {str(k): v for k, v in results.items()},
        "mean_tau": mean_tau,
        "test_ks": test_ks,
    }


# ── B. Policy Comparison ───────────────────────────────────────────────────

def _cwm_greedy_policy(cwm, ns, n, K, budget, max_fitness):
    """1-step lookahead: pick the k that maximises evaluate_state."""
    candidate_ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 25, n]))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]

    def policy(fitness, n_, K_, step, stagnation_counter):
        state = _nk_state(ns, fitness, n_, K_, step, budget, max_fitness,
                           stagnation_counter)
        best_k, best_score = 1, float("-inf")
        for k in candidate_ks:
            action = _nk_action(ns, k)
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


def run_policy_comparison(cwm, ns, instance_path, n=50, K=2,
                          n_episodes=100, budget=10000, verbose=True):
    """Compare 8 policies over n_episodes."""
    problem = NKLandscapeProblem.load_instance(instance_path)
    max_fitness = problem.max_fitness()

    stateless_policies = {
        "static_1":         lambda f, n_, K_, s, stag: 1,
        "static_5":         lambda f, n_, K_, s, stag: 5,
        "random_k":         lambda f, n_, K_, s, stag: np.random.randint(1, n_ + 1),
        "adaptive_fitness":  _adaptive_fitness_policy(max_fitness),
        "cwm_greedy":       _cwm_greedy_policy(cwm, ns, n, K, budget, max_fitness),
    }
    stateful_factories = {
        "stagnation_detect":  lambda: _stagnation_detect_policy(50, 10),
        "fifth_rule":         lambda: _ea_alpha_policy(n),
        "self_adjusting":     lambda: _self_adjusting_policy(n),
    }

    all_policy_names = list(stateless_policies.keys()) + list(stateful_factories.keys())
    results = {name: [] for name in all_policy_names}
    best_fitnesses = {name: [] for name in all_policy_names}

    for pname in all_policy_names:
        if verbose:
            print(f"\n  Policy: {pname}")
        for ep in range(n_episodes):
            seed = ep * 31
            if pname in stateless_policies:
                pfn = stateless_policies[pname]
            else:
                pfn = stateful_factories[pname]()
            episode = run_episode(n, K, pfn, budget, seed=seed,
                                  instance_path=instance_path)
            results[pname].append(episode["final_fitness"])
            best_fitnesses[pname].append(episode["best_fitness"])

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                mean_best = np.mean(best_fitnesses[pname])
                print(f"    Episode {ep+1}/{n_episodes}  "
                      f"mean_best={mean_best:.2f}")

    return results, best_fitnesses


def compute_policy_statistics(results, best_fitnesses):
    """Compute summary stats."""
    summary = {}
    for name, finals in results.items():
        arr = np.array(finals, dtype=float)
        bests = np.array(best_fitnesses[name], dtype=float)
        summary[name] = {
            "mean_final": float(np.mean(arr)),
            "std_final": float(np.std(arr)),
            "mean_best": float(np.mean(bests)),
            "std_best": float(np.std(bests)),
            "median_best": float(np.median(bests)),
            "max_best": float(np.max(bests)),
            "n": len(arr),
        }

    # Pairwise Mann-Whitney U on best fitness
    names = list(results.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(best_fitnesses[names[i]], dtype=float)
            b = np.array(best_fitnesses[names[j]], dtype=float)
            try:
                stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p = 0.0, 1.0
            pairwise[f"{names[i]}_vs_{names[j]}"] = {
                "U_stat": float(stat),
                "p_value": float(p),
            }

    return summary, pairwise


# ── C. Interpretability ────────────────────────────────────────────────────

def build_heatmap_data(cwm, ns, instance_path, n=50, K=2, budget=10000):
    """Build fitness x k score matrix for heatmap visualisation."""
    problem = NKLandscapeProblem.load_instance(instance_path)
    max_fitness = problem.max_fitness()

    fitnesses = list(range(5, int(max_fitness) + 1, max(1, int(max_fitness) // 20)))
    if int(max_fitness) not in fitnesses:
        fitnesses.append(int(max_fitness))
    fitnesses.sort()

    ks = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]))
    ks = [k for k in ks if k <= n]

    matrix = []
    for fit in fitnesses:
        row = []
        state = _nk_state(ns, float(fit), n, K, step=0, budget=budget,
                           max_fitness=max_fitness)
        for k in ks:
            action = _nk_action(ns, k)
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


# ── D. Generalisation across K ────────────────────────────────────────────

def run_generalisation(cwm, ns, n=50, K_values=None, n_episodes=50,
                        budget=10000, verbose=True):
    """Evaluate CWM (trained on K=2) on different K values."""
    if K_values is None:
        K_values = [2, 3, 4]

    instance_paths = {
        2: "configs/nk_instance_n50_K2.json",
        3: "configs/nk_instance_n50_K3.json",
        4: "configs/nk_instance_n50_K4.json",
    }

    results = {}
    for K_val in K_values:
        if verbose:
            print(f"\n  K={K_val}")

        inst_path = instance_paths[K_val]
        problem = NKLandscapeProblem.load_instance(inst_path)
        max_fitness = problem.max_fitness()

        greedy_fn = _cwm_greedy_policy(cwm, ns, n, K_val, budget, max_fitness)
        static1_fn = lambda f, n_, K_, s, stag: 1
        static5_fn = lambda f, n_, K_, s, stag: 5
        adaptive_fn = _adaptive_fitness_policy(max_fitness)

        stateless_pols = [
            ("CWM-greedy", greedy_fn),
            ("static_1", static1_fn),
            ("static_5", static5_fn),
            ("adaptive_fitness", adaptive_fn),
        ]
        stateful_pols = [
            ("stagnation_detect", lambda: _stagnation_detect_policy(50, 10)),
            ("fifth_rule", lambda: _ea_alpha_policy(n)),
            ("self_adjusting", lambda: _self_adjusting_policy(n)),
        ]

        for pname, pfn_or_factory in stateless_pols + stateful_pols:
            is_stateful = pname in [s[0] for s in stateful_pols]
            best_fits = []
            for ep in range(n_episodes):
                seed = ep * 37 + K_val * 1000
                pfn = pfn_or_factory() if is_stateful else pfn_or_factory
                episode = run_episode(n, K_val, pfn, budget, seed=seed,
                                      instance_path=inst_path)
                best_fits.append(episode["best_fitness"])

            key = f"K{K_val}_{pname}"
            results[key] = {
                "K": K_val,
                "n": n,
                "policy": pname,
                "budget": budget,
                "mean_best": float(np.mean(best_fits)),
                "std_best": float(np.std(best_fits)),
                "max_best": float(np.max(best_fits)),
                "max_fitness": max_fitness,
            }
            if verbose:
                print(f"    {pname:<20s}: mean_best={results[key]['mean_best']:.2f}  "
                      f"max={results[key]['max_best']:.2f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────

def _try_plot(plot_fn, path, verbose=True):
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


_C = {
    "cwm_greedy":        "#D55E00",
    "CWM-greedy":        "#D55E00",
    "static_1":          "#0072B2",
    "static_5":          "#56B4E9",
    "random_k":          "#999999",
    "adaptive_fitness":  "#009E73",
    "stagnation_detect": "#661100",
    "fifth_rule":        "#E69F00",
    "self_adjusting":    "#F0E442",
}


def plot_policy_comparison(summary, plt, path):
    """Bar chart of mean best fitness across policies."""
    _pub_style(plt)
    fig, ax = plt.subplots(figsize=(9, 5))

    names = list(summary.keys())
    means = [summary[n]["mean_best"] for n in names]
    stds = [summary[n]["std_best"] for n in names]
    colors = [_C.get(n, "#333333") for n in names]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean best fitness (higher = better)")
    ax.set_title("NK-Landscape: Policy Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_heatmap(heatmap_data, plt, path):
    """Row-normalised CWM score heatmap (fitness x k)."""
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
    ax.set_xlabel("Flip count $k$")
    ax.set_ylabel("NK fitness")
    ax.set_title("NK-Landscape CWM Score Heatmap")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Relative CWM score (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_generalisation(gen_results, plt, path):
    """Grouped bar chart of mean best fitness across K values."""
    _pub_style(plt)
    K_values = sorted(set(v["K"] for v in gen_results.values()))
    pol_order = ["CWM-greedy", "stagnation_detect", "adaptive_fitness",
                 "fifth_rule", "self_adjusting", "static_1", "static_5"]
    pol_colors = [_C.get(p, "#333333") for p in pol_order]

    x = np.arange(len(K_values))
    width = 0.12
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (pol, col) in enumerate(zip(pol_order, pol_colors)):
        vals = []
        for K_val in K_values:
            key = f"K{K_val}_{pol}"
            if key in gen_results:
                vals.append(gen_results[key]["mean_best"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=pol,
               color=col, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (len(pol_order) - 1) / 2)
    ax.set_xticklabels([f"K = {K_val}" for K_val in K_values])
    ax.set_ylabel("Mean best fitness")
    ax.legend(framealpha=0.9, fontsize=9, ncol=2)
    ax.set_title("NK-Landscape: Generalisation across K")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    K = args.K
    budget = args.budget
    n_episodes = 10 if args.quick else args.episodes
    instance_path = args.instance

    print("=" * 60)
    print("Experiment 15: NK-Landscape CWM Evaluation")
    print("=" * 60)
    print(f"  n={n}, K={K}, budget={budget}, episodes={n_episodes}")
    print(f"  instance={instance_path}")

    # Load CWM
    print(f"\nLoading CWM from: {args.cwm}")
    cwm, ns = load_nk_cwm(args.cwm)

    all_results = {
        "experiment": "exp15_nk_evaluation",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "K": K, "budget": budget,
            "episodes": n_episodes, "instance": instance_path,
        },
    }

    # ── A. CWM Quality ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("A. CWM Quality (Kendall's tau vs empirical best)")
    print(f"{'='*60}")
    quality = evaluate_cwm_quality(
        cwm, ns, instance_path, n=n, K=K, budget=budget,
        n_trials=100 if args.quick else 200)
    all_results["quality"] = quality

    # ── B. Policy Comparison ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Policy Comparison")
    print(f"{'='*60}")
    policy_results, best_fitnesses = run_policy_comparison(
        cwm, ns, instance_path, n=n, K=K, n_episodes=n_episodes, budget=budget)
    summary, pairwise = compute_policy_statistics(policy_results, best_fitnesses)

    print(f"\n  {'Policy':<20s} {'Mean best':>10s} {'Std':>8s} {'Max':>8s}")
    print(f"  {'-'*48}")
    for pname in summary:
        s = summary[pname]
        print(f"  {pname:<20s} {s['mean_best']:>10.2f} {s['std_best']:>8.2f} "
              f"{s['max_best']:>8.2f}")

    all_results["policy_comparison"] = {
        "summary": summary,
        "pairwise": pairwise,
        "raw_final": {k: [float(v) for v in vals] for k, vals in policy_results.items()},
        "raw_best": {k: [float(v) for v in vals] for k, vals in best_fitnesses.items()},
    }

    # ── C. Interpretability ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. Interpretability (heatmap)")
    print(f"{'='*60}")
    heatmap = build_heatmap_data(cwm, ns, instance_path, n=n, K=K, budget=budget)
    all_results["interpretability"] = {"heatmap": heatmap}

    # ── D. Generalisation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. Generalisation (across K values)")
    print(f"{'='*60}")
    gen_K_values = [2, 3] if args.quick else [2, 3, 4]
    gen_episodes = max(10, n_episodes // 2)
    gen = run_generalisation(
        cwm, ns, n=n, K_values=gen_K_values, n_episodes=gen_episodes,
        budget=budget)
    all_results["generalisation"] = gen

    # ── Save results ─────────────────────────────────────────────────────
    results_file = output_dir / "exp15_results.json"

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
        lambda plt, p: plot_policy_comparison(summary, plt, p),
        output_dir / "fig_policy_comparison.png",
    )
    _try_plot(
        lambda plt, p: plot_heatmap(heatmap, plt, p),
        output_dir / "fig_heatmap.png",
    )
    _try_plot(
        lambda plt, p: plot_generalisation(gen, plt, p),
        output_dir / "fig_generalisation.png",
    )

    # Also save heatmap for paper
    paper_fig_dir = Path("paper/figures")
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    _try_plot(
        lambda plt, p: plot_heatmap(heatmap, plt, p),
        paper_fig_dir / "nk_heatmap.png",
    )

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  A. Mean Kendall's tau         = {quality['mean_tau']:.3f}")
    print(f"  B. CWM-greedy mean best       = {summary['cwm_greedy']['mean_best']:.2f}")
    for pname in summary:
        if pname != "cwm_greedy":
            print(f"     {pname:<20s} mean best = {summary[pname]['mean_best']:.2f}")
    print(f"  D. Generalisation:")
    for K_val in gen_K_values:
        cwm_key = f"K{K_val}_CWM-greedy"
        if cwm_key in gen:
            print(f"     K={K_val}: CWM mean_best={gen[cwm_key]['mean_best']:.2f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp15: NK-Landscape CWM full evaluation")
    parser.add_argument("--cwm", type=str,
                        default="results/cwm/nk_cwm.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length")
    parser.add_argument("--K", type=int, default=2,
                        help="Epistasis parameter")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per policy")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Step budget")
    parser.add_argument("--instance", type=str,
                        default="configs/nk_instance_n50_K2.json",
                        help="Path to NK instance")
    parser.add_argument("--output", type=str, default="results/exp15",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
