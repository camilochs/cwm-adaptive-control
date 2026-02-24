#!/usr/bin/env python3
"""
Experiment 16: NK-Landscape Multi-Instance Evaluation

Runs the policy comparison (same as exp15 Section B) across 15 different
NK-Landscape instances to demonstrate robustness.  The CWM was trained
on instance seed=42; this experiment evaluates it on 15 distinct landscapes.

Instances: seeds 42, 123, 456, 789, 1024, 2023, 2024, 3141, 5555, 6789,
           7070, 8192, 9001, 9999, 11111 (all n=50, K=2)

Outputs:
  results/exp16/exp16_results.json
  results/exp16/exp16_summary_table.txt
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nk_landscape import (
    NKLandscapeProblem,
    RLSOptimizer,
    run_episode,
)


# ── NKState / NKAction namespace ─────────────────────────────────────────

def _make_nk_namespace():
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
    with open(cwm_path) as f:
        code = f.read()
    ns = _make_nk_namespace()
    exec(code, ns)
    cwm = ns["SynthesizedCWM"]()
    return cwm, ns


# ── Helpers ───────────────────────────────────────────────────────────────

def _nk_state(ns, fitness, n, K, step, budget, max_fitness, stagnation=0):
    return ns["NKState"](
        fitness=fitness, n=n, K=K, step=step, budget=budget,
        normalized_fitness=fitness / max_fitness if max_fitness > 0 else 0.0,
        stagnation_counter=stagnation,
    )


def _nk_action(ns, k):
    return ns["NKAction"](k=k, name=f"flip_{k}")


# ── Policies ──────────────────────────────────────────────────────────────

def _cwm_greedy_policy(cwm, ns, n, K, budget, max_fitness):
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


def _ea_alpha_policy(n, A=2.0, b=0.5):
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
    state = {"no_improve_count": 0, "prev_fitness": None}

    def policy(fitness, n_, K, step, stagnation_counter):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["no_improve_count"] = 0
            else:
                state["no_improve_count"] += 1
        state["prev_fitness"] = fitness
        if state["no_improve_count"] >= patience:
            return escape_k
        return 1

    return policy


def _adaptive_fitness_policy(max_fitness):
    def policy(fitness, n_, K, step, stagnation_counter):
        ratio = fitness / max_fitness if max_fitness > 0 else 0.5
        return max(1, int(n_ * (1 - ratio)))
    return policy


# ── Per-instance evaluation ───────────────────────────────────────────────

def run_instance(cwm, ns, instance_path, n=50, K=2, n_episodes=100,
                 budget=10000, verbose=True):
    """Run all policies on a single NK instance. Returns best_fitnesses dict."""
    problem = NKLandscapeProblem.load_instance(instance_path)
    max_fitness = problem.max_fitness()

    stateless_policies = {
        "static_1":         lambda f, n_, K_, s, stag: 1,
        "static_5":         lambda f, n_, K_, s, stag: 5,
        "random_k":         lambda f, n_, K_, s, stag: np.random.randint(1, n_ + 1),
        "adaptive_fitness": _adaptive_fitness_policy(max_fitness),
        "cwm_greedy":       _cwm_greedy_policy(cwm, ns, n, K, budget, max_fitness),
    }
    stateful_factories = {
        "stagnation_detect": lambda: _stagnation_detect_policy(50, 10),
        "fifth_rule":        lambda: _ea_alpha_policy(n),
        "self_adjusting":    lambda: _self_adjusting_policy(n),
    }

    all_policy_names = list(stateless_policies.keys()) + list(stateful_factories.keys())
    best_fitnesses = {name: [] for name in all_policy_names}

    for pname in all_policy_names:
        if verbose:
            print(f"    Policy: {pname}", end="", flush=True)
        for ep in range(n_episodes):
            seed = ep * 31
            if pname in stateless_policies:
                pfn = stateless_policies[pname]
            else:
                pfn = stateful_factories[pname]()
            episode = run_episode(n, K, pfn, budget, seed=seed,
                                  instance_path=instance_path)
            best_fitnesses[pname].append(episode["best_fitness"])
        if verbose:
            mean_b = np.mean(best_fitnesses[pname])
            print(f"  mean_best={mean_b:.2f}")

    return best_fitnesses, max_fitness


# ── Main experiment ───────────────────────────────────────────────────────

def run_experiment(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    K = args.K
    budget = args.budget
    n_episodes = 10 if args.quick else args.episodes

    # Instance paths (15 instances: 5 original + 10 new)
    instance_configs = [
        ("s42",    "configs/nk_instance_n50_K2.json"),
        ("s123",   "configs/nk_instance_n50_K2_s123.json"),
        ("s456",   "configs/nk_instance_n50_K2_s456.json"),
        ("s789",   "configs/nk_instance_n50_K2_s789.json"),
        ("s1024",  "configs/nk_instance_n50_K2_s1024.json"),
        ("s2023",  "configs/nk_instance_n50_K2_s2023.json"),
        ("s2024",  "configs/nk_instance_n50_K2_s2024.json"),
        ("s3141",  "configs/nk_instance_n50_K2_s3141.json"),
        ("s5555",  "configs/nk_instance_n50_K2_s5555.json"),
        ("s6789",  "configs/nk_instance_n50_K2_s6789.json"),
        ("s7070",  "configs/nk_instance_n50_K2_s7070.json"),
        ("s8192",  "configs/nk_instance_n50_K2_s8192.json"),
        ("s9001",  "configs/nk_instance_n50_K2_s9001.json"),
        ("s9999",  "configs/nk_instance_n50_K2_s9999.json"),
        ("s11111", "configs/nk_instance_n50_K2_s11111.json"),
    ]

    print("=" * 70)
    print("Experiment 16: NK-Landscape Multi-Instance Evaluation")
    print("=" * 70)
    print(f"  n={n}, K={K}, budget={budget}, episodes={n_episodes}")
    print(f"  Instances: {[name for name, _ in instance_configs]}")
    print(f"  CWM: {args.cwm}")

    # Load CWM
    print(f"\nLoading CWM from: {args.cwm}")
    cwm, ns = load_nk_cwm(args.cwm)

    # Run on each instance
    all_instance_results = {}
    for inst_name, inst_path in instance_configs:
        print(f"\n{'='*70}")
        print(f"Instance: {inst_name} ({inst_path})")
        print(f"{'='*70}")

        t0 = time.time()
        best_fitnesses, max_fitness = run_instance(
            cwm, ns, inst_path, n=n, K=K, n_episodes=n_episodes, budget=budget)
        elapsed = time.time() - t0

        # Compute per-instance summary
        summary = {}
        for pname, bests in best_fitnesses.items():
            arr = np.array(bests, dtype=float)
            summary[pname] = {
                "mean_best": float(np.mean(arr)),
                "std_best": float(np.std(arr)),
                "median_best": float(np.median(arr)),
                "max_best": float(np.max(arr)),
            }

        all_instance_results[inst_name] = {
            "instance_path": inst_path,
            "max_fitness": max_fitness,
            "elapsed_s": elapsed,
            "summary": summary,
            "raw_best": {k: [float(v) for v in vals]
                         for k, vals in best_fitnesses.items()},
        }

        # Print instance summary
        print(f"\n  {'Policy':<20s} {'Mean best':>10s} {'Std':>8s} {'Max':>8s}")
        print(f"  {'-'*48}")
        for pname in summary:
            s = summary[pname]
            print(f"  {pname:<20s} {s['mean_best']:>10.2f} "
                  f"{s['std_best']:>8.2f} {s['max_best']:>8.2f}")
        print(f"  (elapsed: {elapsed:.1f}s)")

    # ── Aggregate across instances ────────────────────────────────────────
    print(f"\n{'='*70}")
    n_instances = len(instance_configs)
    print(f"AGGREGATE RESULTS (mean across {n_instances} instances)")
    print(f"{'='*70}")

    policy_names = list(all_instance_results[instance_configs[0][0]]["summary"].keys())

    aggregate = {}
    for pname in policy_names:
        per_instance_means = []
        all_raw = []
        for inst_name, _ in instance_configs:
            inst_summary = all_instance_results[inst_name]["summary"][pname]
            per_instance_means.append(inst_summary["mean_best"])
            all_raw.extend(all_instance_results[inst_name]["raw_best"][pname])

        per_instance_means = np.array(per_instance_means)
        all_raw = np.array(all_raw)

        aggregate[pname] = {
            "mean_of_means": float(np.mean(per_instance_means)),
            "std_of_means": float(np.std(per_instance_means)),
            "overall_mean": float(np.mean(all_raw)),
            "overall_std": float(np.std(all_raw)),
            "overall_max": float(np.max(all_raw)),
            "per_instance_means": [float(m) for m in per_instance_means],
        }

    print(f"\n  {'Policy':<20s} {'Mean(means)':>12s} {'Std(means)':>11s} "
          f"{'Overall mean':>13s} {'Overall std':>12s} {'Max':>8s}")
    print(f"  {'-'*80}")
    for pname in policy_names:
        a = aggregate[pname]
        print(f"  {pname:<20s} {a['mean_of_means']:>12.2f} "
              f"{a['std_of_means']:>11.2f} {a['overall_mean']:>13.2f} "
              f"{a['overall_std']:>12.2f} {a['overall_max']:>8.2f}")

    # ── Pairwise statistical tests ────────────────────────────────────────
    # Wilcoxon signed-rank on per-instance means (cwm_greedy vs each)
    print(f"\n  Statistical tests (Wilcoxon signed-rank on per-instance means):")
    cwm_means = np.array(aggregate["cwm_greedy"]["per_instance_means"])
    pairwise_tests = {}
    for pname in policy_names:
        if pname == "cwm_greedy":
            continue
        other_means = np.array(aggregate[pname]["per_instance_means"])
        diff = cwm_means - other_means
        try:
            t_stat, p_value = sp_stats.ttest_rel(cwm_means, other_means)
        except Exception:
            t_stat, p_value = 0.0, 1.0
        wins = int(np.sum(cwm_means > other_means))
        pairwise_tests[f"cwm_greedy_vs_{pname}"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cwm_wins": wins,
            "cwm_losses": n_instances - wins,
            "mean_diff": float(np.mean(diff)),
        }
        marker = "*" if p_value < 0.05 else ""
        print(f"    vs {pname:<20s}: diff={np.mean(diff):>+.2f}  "
              f"t={t_stat:>+.3f}  p={p_value:.4f}{marker}  "
              f"wins={wins}/{n_instances}")

    # ── Also do Mann-Whitney on pooled raw values ─────────────────────────
    print(f"\n  Mann-Whitney U on pooled raw values (500 runs):")
    pairwise_mw = {}
    cwm_raw = []
    for inst_name, _ in instance_configs:
        cwm_raw.extend(all_instance_results[inst_name]["raw_best"]["cwm_greedy"])
    cwm_raw = np.array(cwm_raw)

    for pname in policy_names:
        if pname == "cwm_greedy":
            continue
        other_raw = []
        for inst_name, _ in instance_configs:
            other_raw.extend(all_instance_results[inst_name]["raw_best"][pname])
        other_raw = np.array(other_raw)
        try:
            u_stat, p_value = sp_stats.mannwhitneyu(
                cwm_raw, other_raw, alternative="two-sided")
        except Exception:
            u_stat, p_value = 0.0, 1.0
        pairwise_mw[f"cwm_greedy_vs_{pname}"] = {
            "U_stat": float(u_stat),
            "p_value": float(p_value),
        }
        marker = "*" if p_value < 0.05 else ""
        print(f"    vs {pname:<20s}: U={u_stat:.0f}  p={p_value:.6f}{marker}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "experiment": "exp16_nk_multi_instance",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "K": K, "budget": budget, "episodes": n_episodes,
            "instances": [{"name": name, "path": path}
                          for name, path in instance_configs],
            "cwm": args.cwm,
        },
        "per_instance": all_instance_results,
        "aggregate": aggregate,
        "pairwise_ttest": pairwise_tests,
        "pairwise_mannwhitney": pairwise_mw,
    }

    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    results_file = output_dir / "exp16_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=to_json)
    print(f"\nResults saved to {results_file}")

    # ── Generate summary table for paper ──────────────────────────────────
    table_file = output_dir / "exp16_summary_table.txt"
    with open(table_file, "w") as f:
        f.write("NK-Landscape Multi-Instance Results "
                f"(n={n}, K={K}, budget={budget:,}, "
                f"{n_episodes} runs, {n_instances} instances)\n")
        f.write("=" * 90 + "\n\n")

        # Per-instance table
        f.write("Per-instance mean best fitness:\n")
        f.write(f"{'Policy':<20s}")
        for inst_name, _ in instance_configs:
            f.write(f" {inst_name:>10s}")
        f.write(f" {'Mean':>10s} {'Std':>8s}\n")
        f.write("-" * (20 + 12 * 5 + 20) + "\n")

        for pname in policy_names:
            f.write(f"{pname:<20s}")
            for inst_name, _ in instance_configs:
                val = all_instance_results[inst_name]["summary"][pname]["mean_best"]
                f.write(f" {val:>10.2f}")
            a = aggregate[pname]
            f.write(f" {a['mean_of_means']:>10.2f} {a['std_of_means']:>8.2f}\n")

        f.write(f"\n\nAggregate (pooled across all {n_instances} instances):\n")
        f.write(f"{'Policy':<20s} {'Mean best':>10s} {'Std':>8s} {'Max':>8s}\n")
        f.write("-" * 50 + "\n")
        for pname in policy_names:
            a = aggregate[pname]
            f.write(f"{pname:<20s} {a['overall_mean']:>10.2f} "
                    f"{a['overall_std']:>8.2f} {a['overall_max']:>8.2f}\n")

    print(f"Summary table saved to {table_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp16: NK-Landscape Multi-Instance Evaluation")
    parser.add_argument("--cwm", type=str,
                        default="results/cwm/nk_cwm_rich_sonnet.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per policy per instance")
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--output", type=str, default="results/exp16")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (10 episodes per policy)")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
