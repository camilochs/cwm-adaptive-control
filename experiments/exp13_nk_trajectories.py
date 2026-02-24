#!/usr/bin/env python3
"""
Experiment 13: NK-Landscape Trajectory Collection

Collects (1+1)-RLS_k trajectories on the NK-Landscape problem using six
different policies for selecting k (the number of bits to flip).

Unlike the other benchmarks, NK-Landscape has NO known optimal policy.
This is the fundamental challenge: the CWM must infer good strategies
purely from trajectory data.

Policies:
  1. random           – k ~ Uniform{1..n}
  2. static_1         – always k=1
  3. static_5         – always k=5
  4. adaptive_fitness – k = max(1, int(n * (1 - fitness/max_fitness)))
  5. stagnation_detect – k=1 normally; k=10 after 50 stagnant steps
  6. eps_adaptive     – 80% adaptive_fitness, 20% random

Output: results/trajectories/nk_trajectories.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nk_landscape import (
    NKLandscapeProblem,
    RLSOptimizer,
    run_episode,
)


# ── Policies ────────────────────────────────────────────────────────────────

# Module-level state for adaptive policies
_MAX_FITNESS = None


def policy_random(fitness, n, K, step, stagnation_counter):
    return np.random.randint(1, n + 1)


def policy_static_1(fitness, n, K, step, stagnation_counter):
    return 1


def policy_static_5(fitness, n, K, step, stagnation_counter):
    return 5


def policy_adaptive_fitness(fitness, n, K, step, stagnation_counter):
    """k decreases as fitness increases: explore more when far from optimum."""
    global _MAX_FITNESS
    if _MAX_FITNESS is None or _MAX_FITNESS <= 0:
        _MAX_FITNESS = n  # fallback
    ratio = fitness / _MAX_FITNESS
    return max(1, int(n * (1 - ratio)))


def policy_eps_adaptive(fitness, n, K, step, stagnation_counter):
    """80% adaptive_fitness, 20% random."""
    if np.random.random() < 0.8:
        return policy_adaptive_fitness(fitness, n, K, step, stagnation_counter)
    return np.random.randint(1, n + 1)


def make_stagnation_detect_policy(patience=50, escape_k=10):
    """Create a stagnation-detection policy.

    Strategy: use k=1 normally. If no improvement for `patience` consecutive
    steps, switch to k=escape_k. Reset counter after each improvement.
    """
    state = {"no_improve_count": 0, "prev_fitness": None}

    def policy(fitness, n, K, step, stagnation_counter):
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


# Stateless policies
POLICIES = {
    "random": policy_random,
    "static_1": policy_static_1,
    "static_5": policy_static_5,
    "adaptive_fitness": policy_adaptive_fitness,
    "eps_adaptive": policy_eps_adaptive,
}


# ── Trajectory collection ───────────────────────────────────────────────────

def collect_trajectories(n, K, n_episodes, budget, instance_path,
                         verbose=True):
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
            seed = hash((policy_name, ep, n, K)) % (2**31)
            episode = run_episode(n, K, policy_fn, budget, seed=seed,
                                  instance_path=instance_path)
            episode["policy"] = policy_name

            all_trajectories.append(episode)

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                print(f"    Episode {ep+1}/{n_episodes}: "
                      f"best_fitness={episode['best_fitness']:.2f}")

    # Run stagnation_detect (stateful — fresh closure per episode)
    policy_name = "stagnation_detect"
    if verbose:
        print(f"\n  Policy: {policy_name}")

    for ep in range(n_episodes):
        seed = hash((policy_name, ep, n, K)) % (2**31)
        policy_fn = make_stagnation_detect_policy(patience=50, escape_k=10)
        episode = run_episode(n, K, policy_fn, budget, seed=seed,
                              instance_path=instance_path)
        episode["policy"] = policy_name

        all_trajectories.append(episode)

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    Episode {ep+1}/{n_episodes}: "
                  f"best_fitness={episode['best_fitness']:.2f}")

    return all_trajectories


def compute_statistics(trajectories):
    """Compute summary statistics per policy."""
    from collections import defaultdict

    by_policy = defaultdict(list)
    for t in trajectories:
        by_policy[t["policy"]].append(t)

    stats = {}
    for policy_name, trajs in by_policy.items():
        best_fits = [t["best_fitness"] for t in trajs]
        final_fits = [t["final_fitness"] for t in trajs]
        stats[policy_name] = {
            "n_episodes": len(trajs),
            "mean_best_fitness": float(np.mean(best_fits)),
            "std_best_fitness": float(np.std(best_fits)),
            "mean_final_fitness": float(np.mean(final_fits)),
            "total_transitions": sum(len(t["transitions"]) for t in trajs),
        }
    return stats


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _MAX_FITNESS

    parser = argparse.ArgumentParser(
        description="Exp13: Collect NK-Landscape RLS trajectories")
    parser.add_argument("--n", type=int, default=50,
                        help="Bitstring length (default 50)")
    parser.add_argument("--K", type=int, default=2,
                        help="Epistasis parameter (default 2)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per policy (default 50)")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Max steps per episode (default 10000)")
    parser.add_argument("--instance", type=str,
                        default="configs/nk_instance_n50_K2.json",
                        help="Path to NK instance")
    parser.add_argument("--output", type=str,
                        default="results/trajectories/nk_trajectories.json",
                        help="Output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (5 episodes per policy)")
    args = parser.parse_args()

    n = args.n
    K = args.K
    n_episodes = 5 if args.quick else args.episodes
    budget = args.budget

    # Set max fitness for adaptive policies
    problem = NKLandscapeProblem.load_instance(args.instance)
    _MAX_FITNESS = problem.max_fitness()

    print("=" * 60)
    print("Experiment 13: NK-Landscape Trajectory Collection")
    print("=" * 60)
    print(f"  n            = {n}")
    print(f"  K            = {K}")
    print(f"  episodes/pol = {n_episodes}")
    print(f"  budget       = {budget}")
    print(f"  instance     = {args.instance}")
    print(f"  max_fitness  = {_MAX_FITNESS:.2f}")
    all_policy_names = list(POLICIES.keys()) + ["stagnation_detect"]
    print(f"  policies     = {all_policy_names}")

    trajectories = collect_trajectories(n, K, n_episodes, budget,
                                        instance_path=args.instance)
    stats = compute_statistics(trajectories)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Policy':<20s} {'Episodes':>8s} {'Mean best':>10s} {'Std':>8s}")
    print(f"{'-'*48}")
    for pname, s in stats.items():
        print(f"{pname:<20s} {s['n_episodes']:>8d} "
              f"{s['mean_best_fitness']:>10.2f} "
              f"{s['std_best_fitness']:>8.2f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "experiment": "exp13_nk_trajectories",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n,
            "K": K,
            "episodes_per_policy": n_episodes,
            "budget": budget,
            "instance": args.instance,
            "max_fitness": _MAX_FITNESS,
        },
        "statistics": stats,
        "trajectories": trajectories,
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
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(data, f, default=to_json)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(trajectories)} trajectories to {output_path} "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
