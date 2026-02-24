#!/usr/bin/env python3
"""
Experiment 1: LeadingOnes Trajectory Collection

Collects (1+1)-RLS_k trajectories on the LeadingOnes problem using six
different policies for selecting k (the number of bits to flip).

Policies:
  1. random       – k ~ Uniform{1..n}
  2. static_1     – always k=1
  3. static_half  – always k=n//2
  4. sqrt         – k = max(1, floor(sqrt(n)))
  5. inverse      – k = i+1  (opposite of optimal, for contrast)
  6. decreasing   – k = max(1, n - i)

Output: results/trajectories/lo_trajectories.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.leading_ones import (
    LeadingOnesProblem,
    RLSOptimizer,
    run_episode,
)


# ── Policies ────────────────────────────────────────────────────────────────

def policy_random(fitness: int, n: int, step: int) -> int:
    return np.random.randint(1, n + 1)


def policy_static_1(fitness: int, n: int, step: int) -> int:
    return 1


def policy_static_half(fitness: int, n: int, step: int) -> int:
    return max(1, n // 2)


def policy_sqrt(fitness: int, n: int, step: int) -> int:
    return max(1, int(math.sqrt(n)))


def policy_inverse(fitness: int, n: int, step: int) -> int:
    return max(1, min(n, fitness + 1))


def policy_decreasing(fitness: int, n: int, step: int) -> int:
    return max(1, n - fitness)


POLICIES = {
    "random": policy_random,
    "static_1": policy_static_1,
    "static_half": policy_static_half,
    "sqrt": policy_sqrt,
    "inverse": policy_inverse,
    "decreasing": policy_decreasing,
}


# ── Trajectory collection ───────────────────────────────────────────────────

def collect_trajectories(n: int, n_episodes: int, budget: int, verbose: bool = True):
    """Collect trajectories for every policy.

    Returns:
        list of trajectory dicts, each augmented with 'policy' key
    """
    all_trajectories = []

    for policy_name, policy_fn in POLICIES.items():
        if verbose:
            print(f"\n  Policy: {policy_name}")

        for ep in range(n_episodes):
            seed = hash((policy_name, ep, n)) % (2**31)
            episode = run_episode(n, policy_fn, budget, seed=seed)
            episode["policy"] = policy_name

            all_trajectories.append(episode)

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                status = "OK" if episode["converged"] else f"fitness={episode['final_fitness']}"
                print(f"    Episode {ep+1}/{n_episodes}: {episode['total_steps']} steps — {status}")

    return all_trajectories


def compute_statistics(trajectories):
    """Compute summary statistics per policy."""
    from collections import defaultdict

    by_policy = defaultdict(list)
    for t in trajectories:
        by_policy[t["policy"]].append(t)

    stats = {}
    for policy_name, trajs in by_policy.items():
        steps = [t["total_steps"] for t in trajs]
        converged = [t["converged"] for t in trajs]
        stats[policy_name] = {
            "n_episodes": len(trajs),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "median_steps": float(np.median(steps)),
            "convergence_rate": float(np.mean(converged)),
            "total_transitions": sum(len(t["transitions"]) for t in trajs),
        }
    return stats


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp28: Collect LeadingOnes RLS trajectories")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length (default 50)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per policy (default 50)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Max steps per episode (default 0.8*n^2)")
    parser.add_argument("--output", type=str,
                        default="results/trajectories/lo_trajectories.json",
                        help="Output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (5 episodes per policy)")
    args = parser.parse_args()

    n = args.n
    n_episodes = 5 if args.quick else args.episodes
    budget = args.budget or int(0.8 * n * n)

    print("=" * 60)
    print("Experiment 1: LeadingOnes Trajectory Collection")
    print("=" * 60)
    print(f"  n            = {n}")
    print(f"  episodes/pol = {n_episodes}")
    print(f"  budget       = {budget}")
    print(f"  policies     = {list(POLICIES.keys())}")

    trajectories = collect_trajectories(n, n_episodes, budget)
    stats = compute_statistics(trajectories)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Policy':<16s} {'Episodes':>8s} {'Mean steps':>11s} {'Conv%':>7s}")
    print(f"{'-'*44}")
    for pname, s in stats.items():
        print(f"{pname:<16s} {s['n_episodes']:>8d} {s['mean_steps']:>11.1f} "
              f"{s['convergence_rate']*100:>6.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "experiment": "exp1_lo_trajectories",
        "timestamp": datetime.now().isoformat(),
        "parameters": {"n": n, "episodes_per_policy": n_episodes, "budget": budget},
        "statistics": stats,
        "trajectories": trajectories,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(trajectories)} trajectories to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
