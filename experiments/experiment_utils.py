"""
Shared utilities for all experiments.

This module centralizes common functionality to avoid code repetition:
- Benchmark creation
- Algorithm instantiation
- Result handling
- Configuration loading
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import json
import csv
from datetime import datetime

import numpy as np
import yaml

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.dynamic import create_extreme_benchmark
from src.benchmarks.functions import get_benchmark
from src.optimizers.de import DifferentialEvolution
from src.optimizers.shade import SHADE, LSHADE
from src.optimizers.adaptive import AdaptiveOptimizer
from src.mcts.planner import PlannerConfig
from src.cwm.synthesizer import CWMSynthesizer
from src.cwm.templates.base_cwm import DefaultCWM, BaseCWM


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cwm(cwm_path: str = None) -> BaseCWM:
    """Load CWM from file or return default."""
    if cwm_path is None:
        cwm_path = "results/cwm/dynamic_cwm.py"

    if cwm_path and Path(cwm_path).exists():
        try:
            return CWMSynthesizer.load_cwm(cwm_path)
        except Exception as e:
            print(f"Warning: Failed to load CWM from {cwm_path}: {e}")
    return DefaultCWM()


# =============================================================================
# Algorithm Factory
# =============================================================================

class AlgorithmFactory:
    """Factory for creating optimizer instances."""

    def __init__(self, config: dict, cwm: BaseCWM = None):
        self.config = config
        self.cwm = cwm or DefaultCWM()
        self.pop_size = config["optimization"]["population_size"]
        self.max_gen = config["optimization"]["max_generations"]

    def create(self, name: str, seed: int = None):
        """Create an optimizer by name."""
        creators = {
            "static_de": self._create_static_de,
            "shade": self._create_shade,
            "lshade": self._create_lshade,
            "mcts_cwm": self._create_mcts_cwm,
        }

        if name not in creators:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(creators.keys())}")

        return creators[name](seed)

    def _create_static_de(self, seed):
        return DifferentialEvolution(
            population_size=self.pop_size,
            max_generations=self.max_gen,
            seed=seed,
        )

    def _create_shade(self, seed):
        return SHADE(
            population_size=self.pop_size,
            max_generations=self.max_gen,
            seed=seed,
        )

    def _create_lshade(self, seed):
        return LSHADE(
            population_size=self.pop_size,
            max_generations=self.max_gen,
            seed=seed,
        )

    def _create_mcts_cwm(self, seed):
        planner_config = PlannerConfig(
            simulations=self.config["mcts"]["simulations"],
            horizon=self.config["mcts"]["horizon"],
            exploration_constant=self.config["mcts"]["exploration_constant"],
        )
        return AdaptiveOptimizer(
            cwm=self.cwm,
            planner_config=planner_config,
            population_size=self.pop_size,
            max_generations=self.max_gen,
            planning_interval=5,
            seed=seed,
        )

    def available_algorithms(self) -> List[str]:
        """Return list of available algorithm names."""
        return ["static_de", "shade", "lshade", "mcts_cwm"]


# =============================================================================
# Benchmark Factory
# =============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for an experimental scenario."""
    name: str
    base_function: str
    dimension: int
    n_constraints: int
    min_change_freq: int
    max_change_freq: int
    noise_std: float
    expected_winner: str = "unknown"
    description: str = ""

    def create_benchmark(self, seed: int):
        """Create a benchmark instance."""
        if self.n_constraints == 0 and self.noise_std == 0:
            # Static benchmark
            return get_benchmark(self.base_function, self.dimension)

        return create_extreme_benchmark(
            dimension=self.dimension,
            base_function=self.base_function,
            n_constraints=self.n_constraints,
            min_change_freq=self.min_change_freq,
            max_change_freq=self.max_change_freq,
            noise_std=self.noise_std,
            seed=seed,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_function": self.base_function,
            "dimension": self.dimension,
            "n_constraints": self.n_constraints,
            "change_freq": (self.min_change_freq, self.max_change_freq),
            "noise_std": self.noise_std,
            "expected_winner": self.expected_winner,
            "description": self.description,
        }


# =============================================================================
# Result Handling
# =============================================================================

@dataclass
class ExperimentResult:
    """Generic experiment result container."""
    scenario: str
    algorithm: str
    run: int
    seed: int
    best_fitness: float
    mean_fitness: float = 0.0
    evaluations: int = 0
    generations: int = 0
    converged: bool = False
    feasibility_ratio: float = 1.0
    total_changes: int = 0
    auc: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "run": self.run,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "evaluations": self.evaluations,
            "generations": self.generations,
            "converged": self.converged,
            "feasibility_ratio": self.feasibility_ratio,
            "total_changes": self.total_changes,
            "auc": self.auc,
            "error": self.error,
        }


def compute_auc(fitness_history: List[float]) -> float:
    """Compute normalized area under convergence curve."""
    if len(fitness_history) < 2:
        return 0.0
    return float(np.trapz(fitness_history) / len(fitness_history))


def run_single_experiment(
    scenario: ScenarioConfig,
    algorithm_name: str,
    run_idx: int,
    factory: AlgorithmFactory,
) -> ExperimentResult:
    """Run a single experiment with proper error handling."""
    seed = run_idx
    np.random.seed(seed)

    try:
        # Create benchmark and optimizer
        problem = scenario.create_benchmark(seed)
        optimizer = factory.create(algorithm_name, seed)

        # Run optimization
        result = optimizer.optimize(problem, problem.bounds)

        # Extract metrics
        fitness_history = [s.best_fitness for s in result.history] if result.history else []
        auc = compute_auc(fitness_history)

        return ExperimentResult(
            scenario=scenario.name,
            algorithm=algorithm_name,
            run=run_idx,
            seed=seed,
            best_fitness=result.best_fitness,
            mean_fitness=np.mean([s.mean_fitness for s in result.history]) if result.history else result.best_fitness,
            evaluations=result.evaluations,
            generations=result.generations,
            converged=result.converged,
            feasibility_ratio=getattr(problem, 'feasibility_ratio', 1.0),
            total_changes=getattr(problem, 'change_count', 0),
            auc=auc,
            fitness_history=fitness_history,
        )

    except Exception as e:
        return ExperimentResult(
            scenario=scenario.name,
            algorithm=algorithm_name,
            run=run_idx,
            seed=seed,
            best_fitness=float('inf'),
            error=str(e),
        )


# =============================================================================
# Result Saving
# =============================================================================

class ResultsSaver:
    """Handles saving experiment results in multiple formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, results: List[ExperimentResult], filename: str = "results.csv"):
        """Save results to CSV."""
        csv_path = self.output_dir / filename

        fieldnames = [
            "scenario", "algorithm", "run", "seed", "best_fitness", "mean_fitness",
            "evaluations", "generations", "converged", "feasibility_ratio",
            "total_changes", "auc", "error"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

        return csv_path

    def save_json(self, results: List[ExperimentResult], metadata: dict = None,
                  filename: str = "results.json"):
        """Save results to JSON with metadata."""
        json_path = self.output_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(results),
            "results": [r.to_dict() for r in results],
        }

        if metadata:
            data["metadata"] = metadata

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return json_path

    def save_convergence(self, results: List[ExperimentResult],
                         filename: str = "convergence_data.json"):
        """Save convergence histories."""
        conv_path = self.output_dir / filename

        data = {}
        for r in results:
            if r.fitness_history:
                key = f"{r.scenario}_{r.algorithm}_{r.run}"
                data[key] = r.fitness_history

        with open(conv_path, "w") as f:
            json.dump(data, f)

        return conv_path

    def save_all(self, results: List[ExperimentResult], metadata: dict = None,
                 prefix: str = ""):
        """Save in all formats."""
        prefix = f"{prefix}_" if prefix else ""

        csv_path = self.save_csv(results, f"{prefix}results.csv")
        json_path = self.save_json(results, metadata, f"{prefix}results.json")
        conv_path = self.save_convergence(results, f"{prefix}convergence.json")

        print(f"Results saved to:")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
        print(f"  - {conv_path}")

        return csv_path, json_path, conv_path


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    try:
        import pandas as pd

        df = pd.DataFrame([r.to_dict() for r in results])

        summary = df.groupby(["scenario", "algorithm"]).agg({
            "best_fitness": ["mean", "std", "min"],
            "auc": "mean",
            "feasibility_ratio": "mean",
        }).round(4)

        return {"dataframe": summary, "raw": df}

    except ImportError:
        # Fallback without pandas
        from collections import defaultdict

        grouped = defaultdict(list)
        for r in results:
            key = (r.scenario, r.algorithm)
            grouped[key].append(r.best_fitness)

        summary = {}
        for (scenario, algo), fitnesses in grouped.items():
            summary[(scenario, algo)] = {
                "mean": np.mean(fitnesses),
                "std": np.std(fitnesses),
                "min": np.min(fitnesses),
            }

        return {"summary": summary}


def print_gap_analysis(results: List[ExperimentResult],
                       reference: str = "mcts_cwm",
                       baseline: str = "static_de"):
    """Print gap analysis between reference and baseline."""
    try:
        import pandas as pd
        df = pd.DataFrame([r.to_dict() for r in results])

        print(f"\n{'='*60}")
        print(f"Gap Analysis: {reference} vs {baseline}")
        print(f"{'='*60}")

        for scenario in df["scenario"].unique():
            ref_mean = df[(df["scenario"] == scenario) &
                         (df["algorithm"] == reference)]["best_fitness"].mean()
            base_mean = df[(df["scenario"] == scenario) &
                          (df["algorithm"] == baseline)]["best_fitness"].mean()

            if base_mean != 0:
                gap = ((ref_mean - base_mean) / base_mean) * 100
                winner = reference if ref_mean < base_mean else baseline
                print(f"{scenario:40s}: gap={gap:+7.2f}% ({winner})")

    except ImportError:
        print("Install pandas for gap analysis")
