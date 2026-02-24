#!/usr/bin/env python3
"""
Experiment 20: Generalization of enriched Jump CWM across k values.

Tests the CWM from exp19 (enriched prompt, no k_jump policies) on k=2,3,4.
Compares with optimal, stagnation, EA_alpha baselines.

Output: results/exp20/exp20_results.json
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jump import (
    JumpProblem,
    RLSOptimizer,
    optimal_k_jump,
    precompute_optimal_k_table,
    run_episode,
)


# ── Namespace & CWM loading ─────────────────────────────────────────────

def _make_jump_namespace():
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
    with open(cwm_path) as f:
        code = f.read()
    ns = _make_jump_namespace()
    exec(code, ns)
    cwm = ns["SynthesizedCWM"]()
    return cwm, ns


def _jump_state(ns, fitness, n, jump_k, step, budget):
    return ns["JumpState"](
        fitness=fitness, n=n, jump_k=jump_k, step=step, budget=budget,
        normalized_fitness=fitness / (n + jump_k),
    )


def _jump_action(ns, k):
    return ns["JumpAction"](k=k, name=f"flip_{k}")


# ── Policies ─────────────────────────────────────────────────────────────

def _cwm_greedy_policy(cwm, ns, n, jump_k, budget, candidate_ks):
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


def _ea_alpha_policy(n, A=2.0, b=0.5):
    state = {"k": 1.0, "prev_fitness": None}
    def policy(fitness, n_, jump_k, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["k"] = min(A * state["k"], n_ / 2)
            else:
                state["k"] = max(b * state["k"], 1.0)
        state["prev_fitness"] = fitness
        return max(1, min(int(round(state["k"])), n_))
    return policy


def _stagnation_detect_policy(jump_k_val, patience=100):
    state = {"no_improve_count": 0, "prev_fitness": None}
    def policy(fitness, n_, jump_k, step):
        if state["prev_fitness"] is not None:
            if fitness > state["prev_fitness"]:
                state["no_improve_count"] = 0
            else:
                state["no_improve_count"] += 1
        state["prev_fitness"] = fitness
        if state["no_improve_count"] >= patience:
            return jump_k_val
        return 1
    return policy


def _get_optimal_k(fitness, n, jump_k):
    return optimal_k_jump(fitness, n, jump_k)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp20: Enriched Jump CWM generalization across k values")
    parser.add_argument("--cwm", type=str,
                        default="results/exp19/jump_cwm_enriched.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    n = args.n
    n_episodes = args.episodes
    candidate_ks = [1, 2, 3, 5, 8, 10, 15, 25, 50]
    test_jks = [2, 3, 4]

    print("=" * 60)
    print("Experiment 20: Enriched Jump CWM Generalization")
    print("=" * 60)

    # Load CWM
    cwm, ns = load_jump_cwm(args.cwm)
    print(f"  Loaded CWM from {args.cwm}")

    results = {}

    for jk in test_jks:
        budget = min(50000, max(10000, int(8 * math.comb(n, jk))))
        print(f"\n  jump_k={jk}, budget={budget}")

        # Define policies
        cwm_fn = _cwm_greedy_policy(cwm, ns, n, jk, budget, candidate_ks)
        opt_fn = lambda f, n_=n, jk_=jk, s=None: _get_optimal_k(f, n_, jk_)

        test_policies = [
            ("CWM-greedy", cwm_fn, False),
            ("optimal", opt_fn, False),
            ("stagnation", lambda jk_=jk: _stagnation_detect_policy(jk_), True),
            ("EA_alpha", lambda: _ea_alpha_policy(n), True),
        ]

        for pname, pfn_or_factory, is_stateful in test_policies:
            steps_list = []
            conv_list = []
            for ep in range(n_episodes):
                seed = ep * 37 + jk * 1000
                pfn = pfn_or_factory() if is_stateful else pfn_or_factory
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
            print(f"    {pname:<15s}: mean={results[key]['mean_steps']:.0f}  "
                  f"SR={results[key]['convergence_rate']:.0%}")

    # Save
    out_dir = Path("results/exp20")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "experiment": "exp20_jump_enriched_generalization",
        "timestamp": datetime.now().isoformat(),
        "parameters": {"n": n, "episodes": n_episodes, "test_jks": test_jks},
        "cwm_source": str(args.cwm),
        "results": results,
    }

    out_path = out_dir / "exp20_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
