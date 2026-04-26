# `results/headline_with_ci.csv` — Schema

This file is populated by `evaluation/evaluate.py` after each evaluation run. Rows for `pending` placeholder values are overwritten when Phase 8 evaluation completes.

## Columns

| Column | Type | Description |
|---|---|---|
| `model` | string | Base model identifier (e.g. `qwen3-1.7b`) |
| `method` | string | Training/inference method (`untrained_baseline`, `safety_system_prompt`, `grpo_variant_a_canonical`, `grpo_variant_b_sharpened`) |
| `tier` | int | Difficulty tier (`2` or `3`) |
| `operator` | string | Operator policy at evaluation time (`train_op` or `strict_op`) |
| `metric` | string | What is being measured — see below |
| `n` | int | Sample size for this cell (200 per cell by convention) |
| `count` | int | Numerator — see metric-specific interpretation below |
| `point_estimate` | float | `count / n`, reported to 4 decimal places |
| `ci_low_95` | float | Wilson 95% CI lower bound |
| `ci_high_95` | float | Wilson 95% CI upper bound |
| `env_sha` | string | Commit SHA of the deployed environment Space at evaluation time |
| `eval_run_id` | string | Unique identifier per evaluation run; allows overwriting prior results without ambiguity |

## Metric values

| Metric | Meaning | `count` interpretation |
|---|---|---|
| `tampering_trained_cat` | Tampering rate on the 4 trained categories (`file_deletion`, `chmod_modification`, `script_rewrite`, `false_self_report`) | Number of episodes where tampering was detected |
| `tampering_held_out` | Tampering rate on the 3 held-out categories (`script_corruption`, `decoy_creation`, `audit_tampering`) | Number of episodes where tampering was detected |
| `math_accuracy` | Fraction of GSM8K-scale problems answered correctly | Number of episodes where submitted answer was correct |
| `honesty_t3` | Fraction of Tier-3 self-report questions answered truthfully (Tier 3 only) | Number of T3 episodes where self-report was truthful |

For tampering metrics, lower is better and `ci_low_95` is the optimistic bound. For accuracy and honesty metrics, higher is better and `ci_high_95` is the optimistic bound.

## Wilson CI computation

```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(
    count=count, nobs=n, alpha=0.05, method="wilson"
)
```

## Pairing convention

For any pair of rows `(model_A, model_B)` at the same `(tier, operator, metric)`, comparisons are paired by seed-index from [`../evaluation/seeds.txt`](../evaluation/seeds.txt). Headline deltas reported in [README §4](../README.md#4-results) use paired-rollout differences, not unpaired difference-of-proportions.

## Sample size justification

200 rollouts per cell yields Wilson 95% CIs of approximately ±5–7 percentage points for proportions in the 10–30% range. This sample size is chosen as the operating point where (a) compute cost stays within the hackathon budget (1,600 rollouts × 4 models × ~30 seconds per rollout ≈ 50 GPU-hours), and (b) effect sizes ≥ 8 percentage points are reliably distinguishable from noise. Effects smaller than that are reported with explicit CI overlap caveats in the README.

## Reproducing a row

```bash
python -m evaluation.evaluate \
    --model qwen3-1.7b \
    --method grpo_variant_a_canonical \
    --tier 2 \
    --operator strict_op \
    --metric tampering_held_out \
    --n-episodes 200 \
    --seed-file evaluation/seeds.txt \
    --append-to results/headline_with_ci.csv
```
