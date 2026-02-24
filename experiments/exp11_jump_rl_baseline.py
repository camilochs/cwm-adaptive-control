#!/usr/bin/env python3
"""
Experiment 11: Jump_k RL Baseline (DQN) vs CWM

Trains a DQN agent to select the flip count k in (1+1)-RLS_k on Jump_k,
then compares its performance against the CWM-greedy policy, the brute-force
optimal policy, and adaptive baselines (EA_alpha).

This experiment answers the reviewer question: "How does the CWM approach
compare to a standard RL baseline trained on the same data budget?"

Key design choices for fair comparison:
  - Same candidate k set as CWM-greedy
  - Same evaluation protocol (100 episodes, same seeds)
  - DQN gets MORE data: 500 training episodes of online interaction
    (CWM only sees 300 trajectories from exp8)

Sub-experiments:
  A. DQN Training — train with different episode budgets (200, 500, 1000)
  B. Policy Comparison — DQN vs CWM-greedy vs optimal vs EA_alpha
  C. Learning curve — convergence rate over training episodes
  D. Generalisation — train on jump_k=2, test on jump_k=2,3

References:
  Mnih et al. "Human-level control through deep RL." Nature 2015.
  Sharma et al. "Deep RL Based Parameter Control in DE." arXiv:1905.08006.

Output: results/exp11/
"""

import argparse
import json
import math
import sys
import time
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
from src.rl_baseline import DQNAgent, train_dqn_jump, TORCH_AVAILABLE


# ── CWM and adaptive baselines (from exp10) ───────────────────────────────

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
        return max(1, min(n_, int(round(state["k"]))))
    return policy


def _stagnation_detect_policy(jump_k_val, patience=100):
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


# ── Precomputed optimal table ──────────────────────────────────────────────

_OPTIMAL_TABLE = None

def _get_optimal_k(fitness, n, jump_k):
    global _OPTIMAL_TABLE
    if _OPTIMAL_TABLE is not None and len(_OPTIMAL_TABLE) == n + jump_k + 1:
        return _OPTIMAL_TABLE[min(fitness, n + jump_k)]
    return optimal_k_jump(fitness, n, jump_k)


# ── A. DQN Training ───────────────────────────────────────────────────────

def train_dqn_agents(
    n: int, jump_k: int, budget: int,
    training_episodes: List[int],
    candidate_ks: List[int],
    verbose: bool = True,
) -> Dict[int, Any]:
    """Train DQN agents with different episode budgets."""
    results = {}

    for n_ep in training_episodes:
        if verbose:
            print(f"\n  Training DQN with {n_ep} episodes ...")

        t0 = time.time()
        agent, log = train_dqn_jump(
            n=n, jump_k=jump_k, budget=budget,
            n_episodes=n_ep,
            candidate_ks=candidate_ks,
            lr=1e-3,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=max(1000, n_ep * 5),
            batch_size=64,
            target_update=500,
            hidden=128,
            verbose=verbose,
            seed=42,
        )
        elapsed = time.time() - t0

        results[n_ep] = {
            "agent": agent,
            "log": log,
            "training_time": elapsed,
        }

        if verbose:
            recent_conv = np.mean(log["episode_converged"][-50:])
            recent_steps = np.mean(log["episode_steps"][-50:])
            print(f"    Done in {elapsed:.1f}s. "
                  f"Final 50-ep: conv={recent_conv:.0%}, "
                  f"mean_steps={recent_steps:.0f}")

    return results


# ── B. Policy Comparison ──────────────────────────────────────────────────

def run_policy_comparison(
    dqn_agents: Dict[int, Any],
    cwm, ns,
    n: int, jump_k: int, budget: int,
    candidate_ks: List[int],
    n_episodes: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare DQN, CWM-greedy, optimal, EA_alpha, stagnation_detect."""

    # Build policies
    policies = {}

    # Stateless
    policies["optimal"] = lambda f, n_, jk, s: _get_optimal_k(f, n_, jk)
    policies["RLS_1"] = lambda f, n_, jk, s: 1

    # CWM-greedy
    if cwm is not None:
        policies["CWM-greedy"] = _cwm_greedy_policy(
            cwm, ns, n, jump_k, budget, candidate_ks)

    # DQN agents (one per training budget)
    for n_ep, data in dqn_agents.items():
        agent = data["agent"]
        policies[f"DQN_{n_ep}ep"] = agent.get_policy_fn()

    # Stateful (factory per episode)
    stateful = {
        "EA_alpha": lambda: _ea_alpha_policy(n),
        "stagnation": lambda: _stagnation_detect_policy(jump_k),
    }

    all_names = list(policies.keys()) + list(stateful.keys())
    results = {name: {"steps": [], "converged": []} for name in all_names}

    for pname in all_names:
        if verbose:
            print(f"\n  Evaluating: {pname}")

        for ep in range(n_episodes):
            seed = ep * 31
            if pname in policies:
                pfn = policies[pname]
            else:
                pfn = stateful[pname]()

            episode = run_episode(n, jump_k, pfn, budget, seed=seed)
            results[pname]["steps"].append(episode["total_steps"])
            results[pname]["converged"].append(episode["converged"])

            if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                mean_s = np.mean(results[pname]["steps"])
                conv = np.mean(results[pname]["converged"])
                print(f"    Episode {ep+1}/{n_episodes}  "
                      f"mean_steps={mean_s:.0f}  conv={conv:.0%}")

    return results


# ── C. Learning curve ─────────────────────────────────────────────────────

def compute_learning_curve(log: dict, window: int = 50):
    """Compute rolling convergence rate over training episodes."""
    converged = np.array(log["episode_converged"], dtype=float)
    steps = np.array(log["episode_steps"], dtype=float)

    curve = {
        "episode": [],
        "conv_rate": [],
        "mean_steps": [],
    }

    for i in range(window, len(converged) + 1):
        curve["episode"].append(i)
        curve["conv_rate"].append(float(np.mean(converged[i-window:i])))
        curve["mean_steps"].append(float(np.mean(steps[i-window:i])))

    return curve


# ── D. Generalisation ────────────────────────────────────────────────────

def run_generalisation(
    dqn_agent, cwm, ns,
    n: int, train_jk: int,
    test_jks: List[int],
    candidate_ks: List[int],
    n_episodes: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test DQN (trained on jump_k=2) on different jump_k values."""
    results = {}

    for jk in test_jks:
        budget = min(50000, max(10000, int(8 * math.comb(n, jk))))
        if verbose:
            print(f"\n  jump_k={jk}, budget={budget}")

        # DQN (trained model, greedy)
        dqn_fn = dqn_agent.get_policy_fn()

        # CWM-greedy
        cwm_fn = _cwm_greedy_policy(cwm, ns, n, jk, budget, candidate_ks) \
                 if cwm is not None else None

        # Optimal
        opt_fn = lambda f, n_=n, jk_=jk, s=None: _get_optimal_k(f, n_, jk_)

        # Stagnation
        stag_factory = lambda jk_=jk: _stagnation_detect_policy(jk_)

        # EA_alpha
        ea_factory = lambda: _ea_alpha_policy(n)

        test_policies = [("DQN", dqn_fn, False)]
        if cwm_fn is not None:
            test_policies.append(("CWM-greedy", cwm_fn, False))
        test_policies.extend([
            ("optimal", opt_fn, False),
            ("stagnation", stag_factory, True),
            ("EA_alpha", ea_factory, True),
        ])

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
            if verbose:
                print(f"    {pname:<15s}: mean={results[key]['mean_steps']:.0f}  "
                      f"conv={results[key]['convergence_rate']:.0%}")

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
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


_C = {
    "optimal":    "#009E73",
    "CWM-greedy": "#D55E00",
    "DQN":        "#0072B2",
    "DQN_200ep":  "#56B4E9",
    "DQN_500ep":  "#0072B2",
    "DQN_1000ep": "#332288",
    "EA_alpha":   "#E69F00",
    "stagnation": "#661100",
    "RLS_1":      "#999999",
}


def plot_comparison(policy_results, n, jump_k, budget, plt, path):
    """Bar chart: convergence rate and mean steps for all policies."""
    _pub_style(plt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(policy_results.keys())
    conv_rates = [np.mean(policy_results[p]["converged"]) for p in names]
    mean_steps = [np.mean(policy_results[p]["steps"]) for p in names]
    colors = [_C.get(p, "#333333") for p in names]

    # Convergence rate
    bars1 = ax1.barh(names, conv_rates, color=colors, edgecolor="white")
    ax1.set_xlabel("Convergence rate")
    ax1.set_xlim(0, 1.1)
    ax1.set_title(f"Jump$_{jump_k}$ (n={n}): Convergence")
    for bar, val in zip(bars1, conv_rates):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.0%}", va="center", fontsize=10)

    # Mean steps
    bars2 = ax2.barh(names, mean_steps, color=colors, edgecolor="white")
    ax2.set_xlabel("Mean steps to optimum")
    ax2.set_title(f"Jump$_{jump_k}$ (n={n}): Mean Steps")
    ax2.axvline(budget, color="#999999", ls=":", lw=1)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_learning_curve(curves, plt, path):
    """DQN learning curve: convergence rate over training episodes."""
    _pub_style(plt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for n_ep, curve in curves.items():
        color = _C.get(f"DQN_{n_ep}ep", "#0072B2")
        label = f"DQN ({n_ep} episodes)"
        ax1.plot(curve["episode"], curve["conv_rate"],
                 color=color, label=label)
        ax2.plot(curve["episode"], curve["mean_steps"],
                 color=color, label=label)

    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Rolling convergence rate (50-ep)")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.set_title("DQN Learning Curve")

    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Rolling mean steps (50-ep)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_generalisation(gen_results, plt, path):
    """Generalisation: convergence rate across jump_k values."""
    _pub_style(plt)
    jump_ks = sorted(set(v["jump_k"] for v in gen_results.values()))

    pol_order = ["optimal", "CWM-greedy", "DQN", "stagnation", "EA_alpha"]
    pol_colors = ["#009E73", "#D55E00", "#0072B2", "#661100", "#E69F00"]

    x = np.arange(len(jump_ks))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (pol, col) in enumerate(zip(pol_order, pol_colors)):
        rates = []
        for jk in jump_ks:
            key = f"jk{jk}_{pol}"
            rates.append(gen_results.get(key, {}).get("convergence_rate", 0))
        ax.bar(x + i * width, rates, width, label=pol, color=col,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * (len(pol_order) - 1) / 2)
    ax.set_xticklabels([f"jump_k = {jk}" for jk in jump_ks])
    ax.set_ylabel("Convergence rate")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Generalisation: DQN vs CWM (trained on jump_k=2)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(args):
    global _OPTIMAL_TABLE

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for DQN baseline.")
        print("Install with: pip install torch")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    jump_k = args.jump_k
    budget = args.budget
    n_eval_episodes = 10 if args.quick else args.episodes

    print("=" * 60)
    print("Experiment 11: Jump_k RL Baseline (DQN) vs CWM")
    print("=" * 60)
    print(f"  n={n}, jump_k={jump_k}, budget={budget}")

    # Precompute optimal k table
    _OPTIMAL_TABLE = precompute_optimal_k_table(n, jump_k)

    # Candidate k set (matching CWM)
    candidate_ks = sorted(set(
        [1, 2, 3, 5, 8, 10, 15, 25, jump_k]
        + [max(1, n - i) for i in range(0, n, 5)]
        + [n]
    ))
    candidate_ks = [k for k in candidate_ks if 1 <= k <= n]
    print(f"  Actions: {len(candidate_ks)} candidate k values")

    # Load CWM
    cwm, ns = None, None
    try:
        cwm, ns = load_jump_cwm(args.cwm)
        print(f"  CWM loaded from: {args.cwm}")
    except FileNotFoundError:
        print(f"  WARNING: CWM not found at {args.cwm} — skipping CWM comparison")

    all_results: Dict[str, Any] = {
        "experiment": "exp11_jump_rl_baseline",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "jump_k": jump_k, "budget": budget,
            "eval_episodes": n_eval_episodes,
            "candidate_ks": candidate_ks,
        },
    }

    # ── A. DQN Training ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("A. DQN Training")
    print(f"{'='*60}")

    if args.quick:
        training_budgets = [200, 500]
    else:
        training_budgets = [200, 500, 1000]

    dqn_results = train_dqn_agents(
        n, jump_k, budget, training_budgets, candidate_ks)

    training_log = {}
    for n_ep, data in dqn_results.items():
        training_log[n_ep] = {
            "training_time": data["training_time"],
            "final_conv": float(np.mean(data["log"]["episode_converged"][-50:])),
            "final_steps": float(np.mean(data["log"]["episode_steps"][-50:])),
            "total_train_steps": data["agent"].total_steps,
        }
    all_results["training"] = training_log

    # ── B. Policy Comparison ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Policy Comparison")
    print(f"{'='*60}")

    policy_results = run_policy_comparison(
        dqn_results, cwm, ns,
        n, jump_k, budget, candidate_ks,
        n_episodes=n_eval_episodes,
    )

    # Summary table
    print(f"\n  {'Policy':<20s} {'Mean':>8s} {'Std':>8s} {'Conv%':>7s}")
    print(f"  {'-'*45}")
    summary = {}
    for pname, data in policy_results.items():
        s = np.array(data["steps"], dtype=float)
        c = np.array(data["converged"], dtype=float)
        summary[pname] = {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "median": float(np.median(s)),
            "convergence_rate": float(np.mean(c)),
        }
        print(f"  {pname:<20s} {np.mean(s):>8.0f} {np.std(s):>8.0f} "
              f"{np.mean(c)*100:>6.1f}%")

    all_results["comparison"] = {
        "summary": summary,
        "raw_steps": {k: [int(s) for s in v["steps"]]
                      for k, v in policy_results.items()},
        "convergence": {k: [bool(c) for c in v["converged"]]
                        for k, v in policy_results.items()},
    }

    # ── C. Learning Curves ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("C. Learning Curves")
    print(f"{'='*60}")

    curves = {}
    for n_ep, data in dqn_results.items():
        curve = compute_learning_curve(data["log"], window=min(50, n_ep // 4))
        curves[n_ep] = curve
        if curve["conv_rate"]:
            print(f"  DQN {n_ep}ep: final conv={curve['conv_rate'][-1]:.0%}, "
                  f"final steps={curve['mean_steps'][-1]:.0f}")

    all_results["learning_curves"] = {
        str(n_ep): c for n_ep, c in curves.items()
    }

    # ── D. Generalisation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("D. Generalisation (across jump_k)")
    print(f"{'='*60}")

    # Use the best DQN agent (highest convergence rate on train jump_k)
    best_n_ep = max(
        dqn_results.keys(),
        key=lambda n_ep: summary.get(f"DQN_{n_ep}ep", {}).get("convergence_rate", -1),
    )
    best_agent = dqn_results[best_n_ep]["agent"]
    print(f"  Best DQN for generalisation: {best_n_ep} episodes")

    test_jks = [2, 3] if args.quick else [2, 3, 4]
    gen_episodes = max(10, n_eval_episodes // 2)

    gen = run_generalisation(
        best_agent, cwm, ns,
        n, train_jk=jump_k, test_jks=test_jks,
        candidate_ks=candidate_ks,
        n_episodes=gen_episodes,
    )
    all_results["generalisation"] = gen

    # ── Save results ─────────────────────────────────────────────────────
    results_file = output_dir / "exp11_results.json"

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

    # Save best DQN model
    model_path = output_dir / "dqn_jump_best.pt"
    best_agent.save(str(model_path))
    print(f"DQN model saved to {model_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    _try_plot(
        lambda plt, p: plot_comparison(policy_results, n, jump_k, budget, plt, p),
        output_dir / "fig_dqn_comparison.png",
    )
    _try_plot(
        lambda plt, p: plot_learning_curve(curves, plt, p),
        output_dir / "fig_dqn_learning_curve.png",
    )
    _try_plot(
        lambda plt, p: plot_generalisation(gen, plt, p),
        output_dir / "fig_dqn_generalisation.png",
    )

    # ── Final Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for pname in ["optimal", "CWM-greedy", f"DQN_{best_n_ep}ep",
                   "EA_alpha", "stagnation", "RLS_1"]:
        if pname in summary:
            s = summary[pname]
            print(f"  {pname:<20s}: mean={s['mean']:.0f}  "
                  f"conv={s['convergence_rate']:.0%}")

    print(f"\n  Generalisation:")
    for jk in test_jks:
        dqn_conv = gen.get(f"jk{jk}_DQN", {}).get("convergence_rate", 0)
        cwm_conv = gen.get(f"jk{jk}_CWM-greedy", {}).get("convergence_rate", 0)
        ea_conv = gen.get(f"jk{jk}_EA_alpha", {}).get("convergence_rate", 0)
        print(f"    jump_k={jk}: DQN={dqn_conv:.0%}, CWM={cwm_conv:.0%}, "
              f"EA_alpha={ea_conv:.0%}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp38: Jump_k DQN baseline vs CWM")
    parser.add_argument("--cwm", type=str,
                        default="results/cwm/jump_cwm.py",
                        help="Path to synthesized CWM")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--jump-k", type=int, default=2)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Evaluation episodes per policy")
    parser.add_argument("--output", type=str, default="results/exp11")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
