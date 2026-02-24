#!/usr/bin/env python3
"""
Experiment 4: CWM Synthesis Robustness

Synthesises 5 independent CWMs (different LLM temperatures / seeds) and
evaluates each one on the same evaluation protocol as exp3.

Goal: quantify how much the CWM quality varies across independent synthesis
runs — i.e., is the result in exp3 a lucky draw or robust?

Output:
  results/exp4/cwm_seed_{0..4}.py         — 5 synthesised CWMs
  results/exp4/exp4_results.json          — per-CWM and aggregate metrics
  results/exp4/fig_robustness_policy.png   — 5 learned policies overlaid
  results/exp4/fig_robustness_boxplot.png  — boxplot of steps across seeds
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.leading_ones import optimal_k, run_episode
from src.cwm.synthesizer import LLMClient
from src.cwm.validator import extract_code_from_response

# Reuse validation and evaluation from exp2/exp3
from experiments.exp2_lo_cwm_synthesis import (
    validate_lo_cwm,
    format_samples,
    compute_prompt_stats,
    LO_SYSTEM_PROMPT,
    LO_SYNTHESIS_PROMPT,
    LO_REFINEMENT_PROMPT,
)
from experiments.exp3_lo_evaluation import (
    load_lo_cwm,
    _make_lo_namespace,
    _lo_state,
    _lo_action,
    evaluate_cwm_quality,
    _cwm_greedy_policy,
    extract_learned_policy,
)


# ── Synthesis with temperature variation ────────────────────────────────────

def synthesize_one(data, model, seed_idx, max_attempts=5, verbose=True):
    """Synthesise one CWM with a different temperature perturbation."""
    stats = compute_prompt_stats(data)
    n = stats["n"]
    budget = data["parameters"]["budget"]

    # Vary temperature slightly per seed for diversity
    temperatures = [0.6, 0.7, 0.8, 0.9, 1.0]
    temp = temperatures[seed_idx % len(temperatures)]

    # Also shuffle the trajectory samples differently
    np.random.seed(seed_idx * 42)
    samples_text = format_samples(data["trajectories"], max_samples=30)

    prompt = LO_SYNTHESIS_PROMPT.format(
        trajectory_samples=samples_text,
        n_samples=min(30, stats["total_transitions"]),
        **stats,
    )

    if "claude" in model.lower():
        provider = "anthropic"
    elif "gpt" in model.lower():
        provider = "openai"
    else:
        provider = "ollama"

    client = LLMClient(provider=provider, model=model)

    for attempt in range(max_attempts):
        if verbose:
            print(f"    Attempt {attempt+1}/{max_attempts} (temp={temp:.1f}) ...")

        t0 = time.time()
        response = client.generate(
            prompt=prompt,
            system=LO_SYSTEM_PROMPT,
            temperature=temp,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        code = extract_code_from_response(response)

        if verbose:
            print(f"      LLM responded in {elapsed:.1f}s ({len(code)} chars)")

        valid, tests, errors = validate_lo_cwm(code, n=n, budget=budget)

        if verbose:
            passed = sum(tests.values())
            total = len(tests)
            print(f"      Validation: {passed}/{total} tests passed")

        if valid:
            return code, attempt + 1, temp

        # Refinement prompt
        failures_text = "\n".join(
            f"- {t}: {'PASS' if ok else 'FAIL'}" for t, ok in tests.items()
        )
        if errors:
            failures_text += "\n" + "\n".join(f"- Error: {e}" for e in errors)
        prompt = LO_REFINEMENT_PROMPT.format(
            current_code=code, failures=failures_text,
        )

    # Return last code even if not fully valid
    return code, max_attempts, temp


# ── Evaluation per CWM ─────────────────────────────────────────────────────

def evaluate_one_cwm(cwm_path, n=50, n_episodes=100, budget=2000):
    """Run key metrics on a single CWM. Returns dict of results."""
    cwm, ns = load_lo_cwm(cwm_path)

    # A. Quality
    quality = evaluate_cwm_quality(cwm, ns, n=n, budget=budget, verbose=False)

    # B. Policy comparison (3 key policies only for speed)
    greedy_fn = _cwm_greedy_policy(cwm, ns, n, budget)
    optimal_fn = lambda f, n_, s: optimal_k(f, n_)
    static1_fn = lambda f, n_, s: 1

    results_by_policy = {}
    for pname, pfn in [("optimal", optimal_fn),
                       ("cwm_greedy", greedy_fn),
                       ("RLS_1", static1_fn)]:
        steps = []
        for ep in range(n_episodes):
            seed = ep * 31
            episode = run_episode(n, pfn, budget, seed=seed)
            steps.append(episode["total_steps"])
        results_by_policy[pname] = steps

    # C. Learned policy correlation
    learned = extract_learned_policy(cwm, ns, n=n, budget=budget)
    cwm_ks = [learned[f]["cwm_best_k"] for f in range(n)]
    opt_ks = [optimal_k(f, n) for f in range(n)]
    corr, corr_p = sp_stats.spearmanr(cwm_ks, opt_ks)
    exact = sum(1 for f in range(n) if learned[f]["cwm_best_k"] == learned[f]["optimal_k"])

    return {
        "mean_tau": quality["mean_tau"],
        "per_fitness_tau": {
            str(k): v["kendall_tau"]
            for k, v in quality["per_fitness"].items()
        },
        "steps": {
            pname: {
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
                "median": float(np.median(s)),
                "raw": [int(x) for x in s],
            }
            for pname, s in results_by_policy.items()
        },
        "spearman_corr": float(corr) if not np.isnan(corr) else 0.0,
        "exact_match_frac": exact / n,
        "learned_ks": cwm_ks,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def _pub_style(plt):
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_robustness_policy(all_learned_ks, n, plt, path):
    """Overlay all 5 learned policies + theoretical optimum."""
    _pub_style(plt)
    fitnesses = list(range(n))
    opt_ks = [optimal_k(f, n) for f in fitnesses]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fitnesses, opt_ks, "k-", lw=2.5, zorder=10,
            label=r"Optimal $k^*(i)$")

    colors = ["#D55E00", "#CC79A7", "#E69F00", "#56B4E9", "#009E73"]
    for i, ks in enumerate(all_learned_ks):
        ax.plot(fitnesses, ks, ls="--", lw=1.0, alpha=0.7,
                color=colors[i % len(colors)],
                label=f"CWM seed {i}")

    ax.axhline(1, color="#0072B2", ls=":", lw=1.2, alpha=0.5)
    ax.set_xlabel("LeadingOnes fitness $i$")
    ax.set_ylabel("Flip count $k$")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, n + 1)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_robustness_boxplot(all_steps, plt, path):
    """Boxplot: steps to optimum per CWM seed + baselines."""
    _pub_style(plt)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Collect data
    labels = []
    data = []

    # Optimal (same across seeds, take from seed 0)
    data.append(all_steps[0]["optimal"]["raw"])
    labels.append(r"Optimal $k^*(i)$")

    # Each CWM seed
    for i in range(len(all_steps)):
        data.append(all_steps[i]["cwm_greedy"]["raw"])
        labels.append(f"CWM seed {i}")

    # RLS_1 (same across seeds)
    data.append(all_steps[0]["RLS_1"]["raw"])
    labels.append(r"RLS$_1$")

    colors = (["#009E73"]
              + ["#D55E00", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2"][:len(all_steps)]
              + ["#0072B2"])

    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Steps to optimum")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp31: CWM synthesis robustness (5 seeds)")
    parser.add_argument("--trajectories", type=str,
                        default="results/trajectories/lo_trajectories.json")
    parser.add_argument("--model", type=str,
                        default="claude-sonnet-4-20250514")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/exp4")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="Skip synthesis, reuse existing CWMs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = 30 if args.quick else args.episodes

    print("=" * 60)
    print("Experiment 4: CWM Synthesis Robustness")
    print("=" * 60)

    # Load trajectories
    print(f"\nLoading trajectories: {args.trajectories}")
    with open(args.trajectories) as f:
        data = json.load(f)
    n = data["parameters"]["n"]
    budget = data["parameters"]["budget"]
    print(f"  {len(data['trajectories'])} trajectories, n={n}")

    # ── Phase 1: Synthesise 5 CWMs ──────────────────────────────────────
    cwm_paths = []
    synthesis_meta = []

    if args.skip_synthesis:
        print(f"\n{'='*60}")
        print(f"Phase 1: SKIPPED (reusing existing CWMs)")
        print(f"{'='*60}")
        for seed_idx in range(args.n_seeds):
            cwm_path = output_dir / f"cwm_seed_{seed_idx}.py"
            cwm_paths.append(str(cwm_path))
            with open(cwm_path) as f:
                code = f.read()
            synthesis_meta.append({
                "seed": seed_idx, "temperature": 0.0,
                "attempts": 0, "valid": True,
                "time_s": 0.0, "code_length": len(code),
            })
            print(f"  Reusing {cwm_path}")
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: Synthesising {args.n_seeds} CWMs")
        print(f"{'='*60}")

        for seed_idx in range(args.n_seeds):
            print(f"\n  Seed {seed_idx}:")
            t0 = time.time()
            code, attempts, temp = synthesize_one(
                data, args.model, seed_idx, max_attempts=5
            )
            elapsed = time.time() - t0

            cwm_path = output_dir / f"cwm_seed_{seed_idx}.py"
            with open(cwm_path, "w") as f:
                f.write(code)
            cwm_paths.append(str(cwm_path))

            valid, tests, _ = validate_lo_cwm(code, n=n, budget=budget)
            synthesis_meta.append({
                "seed": seed_idx,
                "temperature": temp,
                "attempts": attempts,
                "valid": valid,
                "time_s": round(elapsed, 1),
                "code_length": len(code),
            })
            print(f"    Valid={valid}, attempts={attempts}, "
                  f"temp={temp:.1f}, time={elapsed:.1f}s")

    # ── Phase 2: Evaluate each CWM ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Evaluating {args.n_seeds} CWMs ({n_episodes} episodes each)")
    print(f"{'='*60}")

    all_eval = []
    all_learned_ks = []
    all_steps = []

    for seed_idx, cwm_path in enumerate(cwm_paths):
        print(f"\n  Evaluating seed {seed_idx} ...")
        t0 = time.time()
        ev = evaluate_one_cwm(cwm_path, n=n, n_episodes=n_episodes, budget=budget)
        elapsed = time.time() - t0

        all_eval.append(ev)
        all_learned_ks.append(ev["learned_ks"])
        all_steps.append(ev["steps"])

        print(f"    tau={ev['mean_tau']:.3f}  "
              f"greedy={ev['steps']['cwm_greedy']['mean']:.0f}  "
              f"spearman={ev['spearman_corr']:.3f}  "
              f"exact={ev['exact_match_frac']:.0%}  "
              f"({elapsed:.1f}s)")

    # ── Phase 3: Aggregate ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3: Aggregate Results")
    print(f"{'='*60}")

    taus = [e["mean_tau"] for e in all_eval]
    greedy_means = [e["steps"]["cwm_greedy"]["mean"] for e in all_eval]
    corrs = [e["spearman_corr"] for e in all_eval]
    exact_fracs = [e["exact_match_frac"] for e in all_eval]
    optimal_mean = all_eval[0]["steps"]["optimal"]["mean"]
    rls1_mean = all_eval[0]["steps"]["RLS_1"]["mean"]

    print(f"\n  {'Metric':<30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*62}")
    for name, vals in [("Kendall tau", taus),
                       ("CWM-greedy steps", greedy_means),
                       ("Spearman corr", corrs),
                       ("Exact match frac", exact_fracs)]:
        arr = np.array(vals)
        print(f"  {name:<30s} {np.mean(arr):>8.3f} {np.std(arr):>8.3f} "
              f"{np.min(arr):>8.3f} {np.max(arr):>8.3f}")

    print(f"\n  Baselines (fixed):")
    print(f"    Optimal k*(i): {optimal_mean:.0f} steps")
    print(f"    RLS_1:         {rls1_mean:.0f} steps")
    print(f"\n  CWM-greedy gap vs optimal: "
          f"{np.mean(greedy_means)/optimal_mean - 1:.1%} "
          f"(+/- {np.std(greedy_means)/optimal_mean:.1%})")

    # ── Save ────────────────────────────────────────────────────────────
    results = {
        "experiment": "exp4_cwm_robustness",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "n": n, "budget": budget, "n_seeds": args.n_seeds,
            "episodes": n_episodes, "model": args.model,
        },
        "synthesis": synthesis_meta,
        "per_seed": [
            {
                "seed": i,
                "mean_tau": e["mean_tau"],
                "per_fitness_tau": e["per_fitness_tau"],
                "greedy_mean": e["steps"]["cwm_greedy"]["mean"],
                "greedy_std": e["steps"]["cwm_greedy"]["std"],
                "greedy_median": e["steps"]["cwm_greedy"]["median"],
                "spearman_corr": e["spearman_corr"],
                "exact_match_frac": e["exact_match_frac"],
            }
            for i, e in enumerate(all_eval)
        ],
        "aggregate": {
            "kendall_tau": {"mean": float(np.mean(taus)), "std": float(np.std(taus))},
            "greedy_steps": {"mean": float(np.mean(greedy_means)), "std": float(np.std(greedy_means))},
            "spearman_corr": {"mean": float(np.mean(corrs)), "std": float(np.std(corrs))},
            "exact_match": {"mean": float(np.mean(exact_fracs)), "std": float(np.std(exact_fracs))},
            "optimal_steps": optimal_mean,
            "rls1_steps": rls1_mean,
        },
    }

    results_file = output_dir / "exp4_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # ── Plots ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_robustness_policy(
            all_learned_ks, n, plt,
            output_dir / "fig_robustness_policy.png",
        )
        print(f"  Saved: fig_robustness_policy.png")

        plot_robustness_boxplot(
            all_steps, plt,
            output_dir / "fig_robustness_boxplot.png",
        )
        print(f"  Saved: fig_robustness_boxplot.png")
    except ImportError:
        print("  matplotlib not available — skipping plots")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
