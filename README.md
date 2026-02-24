# Code World Models for Adaptive Parameter Control

Source code and experimental results for the paper *"Code World Models for Adaptive Parameter Control in Evolutionary Algorithms"*.

An LLM synthesizes a compact Python program (a Code World Model) that predicts how the state of a (1+1)-RLS_k optimizer evolves under different mutation strengths k. A greedy planner uses this model to select k at each step.

## Structure

```
src/                  # Core library
  leading_ones.py     # LeadingOnes benchmark + RLS_k
  onemax.py           # OneMax benchmark + RLS_k
  jump.py             # Jump_k benchmark + RLS_k
  nk_landscape.py     # NK-Landscape benchmark + RLS_k
  rl_baseline.py      # DQN baseline
  cwm/                # CWM synthesis pipeline (prompts, validation, refinement)
  mcts/               # MCTS planner

experiments/          # Experiment scripts (exp1–exp21)
  exp1–exp4           # LeadingOnes: trajectories → synthesis → evaluation → robustness
  exp5–exp7           # OneMax: trajectories → synthesis → evaluation
  exp8–exp11          # Jump_k: trajectories → synthesis → evaluation → DQN comparison
  exp13–exp16         # NK: trajectories → synthesis → evaluation → multi-instance
  exp20–exp21         # Jump_k generalization and enriched prompts

results/
  cwm/                # Synthesized CWMs (Python files produced by the LLM)
  exp3/, exp7/, ...   # Evaluation results (JSON)

configs/              # NK-Landscape instance files (15 instances, n=50, K=2)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"   # only needed for CWM synthesis (step 2)
```

## Reproducing experiments

Each benchmark follows the same three-step pipeline:

```bash
# 1. Collect trajectories
python experiments/exp1_lo_trajectories.py

# 2. Synthesize CWM via LLM (requires API key)
python experiments/exp2_lo_cwm_synthesis.py

# 3. Evaluate
python experiments/exp3_lo_evaluation.py
```

Same pattern for OneMax (exp5–7), Jump_k (exp8–10), and NK (exp13–16).

Pre-synthesized CWMs are included under `results/cwm/`, so steps 1–2 can be skipped for evaluation-only runs.

### Mapping to paper tables

| Paper section | Script |
|---|---|
| Table 1 (LO) | `exp3_lo_evaluation.py` |
| Table 2 (OM) | `exp7_om_evaluation.py` |
| Table 3 (Jump_k) | `exp10_jump_evaluation.py` |
| Table 4 (DQN comparison) | `exp11_jump_rl_baseline.py` |
| Table 5 (NK, 15 instances) | `exp16_nk_multi_instance.py` |
| Fig. 5 (Jump_k generalization) | `exp20_jump_enriched_generalization.py` |

## Pre-synthesized CWMs

The `results/cwm/` directory contains the CWMs used in the paper:

- `leadingones_cwm.py` — LO world model (Kendall tau = 0.784)
- `onemax_cwm.py` — OM world model (Spearman rho = 0.939)
- `jump_cwm.py` — Jump_k world model (valley-edge tau = 1.0)
- `nk_cwm_rich_sonnet.py` — NK world model with empirical transition table (tau = 0.553)

Each file defines a `SynthesizedCWM` class with `predict_next_state`, `evaluate_state`, `get_legal_actions`, and `is_terminal` methods.

## License

MIT
