#!/usr/bin/env python3
"""
Experiment 8: Jump_k Trajectory Collection

Collects (1+1)-RLS_k trajectories on the Jump_k problem using six
different policies for selecting k (the number of bits to flip).

Policies:
  1. random            – k ~ Uniform{1..n}
  2. static_1          – always k=1 (WILL FAIL at valley)
  3. static_jump_k     – always k=jump_k (slow but can jump)
  4. sqrt              – k = max(1, floor(sqrt(n)))
  5. decreasing        – k = max(1, n - fitness)
  6. stagnation_detect – k=1 normally; k=jump_k after 100 non-improvements

Output: results/trajectories/jump_trajectories.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jump import (
    JumpProblem,
    RLSOptimizer,
    run_episode,
)


# ── Policies ────────────────────────────────────────────────────────────────

def policy_random(fitness: int, n: int, jump_k: int, step: int) -> int:
    return np.random.randint(1, n + 1)


def policy_static_1(fitness: int, n: int, jump_k: int, step: int) -> int:
    return 1


def policy_static_jump_k(fitness: int, n: int, jump_k: int, step: int) -> int:
    return jump_k


def policy_sqrt(fitness: int, n: int, jump_k: int, step: int) -> int:
    return max(1, int(math.sqrt(n)))


def policy_decreasing(fitness: int, n: int, jump_k: int, step: int) -> int:
    return max(1, n - fitness)


def make_stagnation_detect_policy(jump_k_val: int, patience: int = 100):
    """Create a stagnation-detection policy (stateful, needs fresh closure per episode).

    Strategy: use k=1 normally. If no improvement for `patience` consecutive
    steps, switch to k=jump_k to attempt a valley crossing. Reset counter
    after each improvement.
    """
    state = {"no_improve_count": 0, "prev_fitness": None}

    def policy(fitness: int, n: int, jump_k: int, step: int) -> int:
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


# Stateless policies (stagnation_detect is handled separately)
POLICIES = {
    "random": policy_random,
    "static_1": policy_static_1,
    "static_jump_k": policy_static_jump_k,
    "sqrt": policy_sqrt,
    "decreasing": policy_decreasing,
}


# ── Trajectory collection ───────────────────────────────────────────────────

def collect_trajectories(n: int, jump_k: int, n_episodes: int, budget: int,
                         verbose: bool = True):
    """Collect trajectories for every policy.

    Returns:
        list of trajectory dicts, each augmented with 'policy' key
    """
    all_trajectories = []

    # Run stateless policies
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

    # Run stagnation_detect (stateful — fresh closure per episode)
    policy_name = "stagnation_detect"
    if verbose:
        print(f"\n  Policy: {policy_name}")

    for ep in range(n_episodes):
        seed = hash((policy_name, ep, n, jump_k)) % (2**31)
        policy_fn = make_stagnation_detect_policy(jump_k)
        episode = run_episode(n, jump_k, policy_fn, budget, seed=seed)
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
        description="Exp35: Collect Jump_k RLS trajectories")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length (default 50)")
    parser.add_argument("--jump-k", type=int, default=2,
                        help="Jump gap parameter (default 2)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per policy (default 50)")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Max steps per episode (default 10000)")
    parser.add_argument("--output", type=str,
                        default="results/trajectories/jump_trajectories.json",
                        help="Output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (5 episodes per policy)")
    args = parser.parse_args()

    n = args.n
    jump_k = args.jump_k
    n_episodes = 5 if args.quick else args.episodes
    budget = args.budget

    print("=" * 60)
    print("Experiment 8: Jump_k Trajectory Collection")
    print("=" * 60)
    print(f"  n            = {n}")
    print(f"  jump_k       = {jump_k}")
    print(f"  episodes/pol = {n_episodes}")
    print(f"  budget       = {budget}")
    all_policy_names = list(POLICIES.keys()) + ["stagnation_detect"]
    print(f"  policies     = {all_policy_names}")

    trajectories = collect_trajectories(n, jump_k, n_episodes, budget)
    stats = compute_statistics(trajectories)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Policy':<20s} {'Episodes':>8s} {'Mean steps':>11s} {'Conv%':>7s}")
    print(f"{'-'*48}")
    for pname, s in stats.items():
        print(f"{pname:<20s} {s['n_episodes']:>8d} {s['mean_steps']:>11.1f} "
              f"{s['convergence_rate']*100:>6.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "experiment": "exp8_jump_trajectories",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n,
            "jump_k": jump_k,
            "episodes_per_policy": n_episodes,
            "budget": budget,
        },
        "statistics": stats,
        "trajectories": trajectories,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(trajectories)} trajectories to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
