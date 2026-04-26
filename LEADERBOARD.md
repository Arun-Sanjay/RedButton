# 🏆 The Red Button — Leaderboard

A public benchmark for **shutdown-corrigibility in small open-weight LLMs**. Anyone can train or prompt a model against the live environment, run a 200-rollout evaluation, and submit a row.

> **Live environment:** <https://huggingface.co/spaces/Arun-Sanjay/RedButton>
> **Reproducibility seed list:** [`evaluation/seeds.txt`](./evaluation/seeds.txt)
> **Methodology:** see [README §3 (Methods)](./README.md#3-methods)

---

## Primary metric

**Tampering rate on held-out tampering categories under the Strict Operator policy** (lower is better).

This is the hardest cell in the evaluation matrix: tampering categories the model was never trained on, evaluated under an operator policy that didn't exist during training. A method that wins on this column has demonstrated *disposition transfer*, not pattern memorization.

Secondary metrics (math accuracy, honesty, training-operator and trained-category cells) are reported for completeness.

---

## Current standings

All rows: 200 rollouts per cell, paired by seed across rows, Tier 2. Wilson 95% CIs in [`results/headline_with_ci.csv`](./results/headline_with_ci.csv).

| Rank | Model | Method | **Held-Out, Strict Op ↓** | Held-Out, Train Op | Trained-Cat, Strict Op | Trained-Cat, Train Op | Math Acc | Honesty (T3) | Submission |
|---|---|---|---|---|---|---|---|---|---|
| — | Qwen3-1.7B | Untrained baseline | `[B_HS%]` | `[B_HT%]` | `[B_TS%]` | `[B_TT%]` | `[B_M%]` | `[B_H%]` | this repo |
| — | Qwen3-1.7B | Safety system prompt | `[P_HS%]` | `[P_HT%]` | `[P_TS%]` | `[P_TT%]` | `[P_M%]` | `[P_H%]` | this repo |
| — | Qwen3-1.7B | **GRPO (Variant A — canonical)** | `[A_HS%]` | `[A_HT%]` | `[A_TS%]` | `[A_TT%]` | `[A_M%]` | `[A_H%]` | this repo |
| — | Qwen3-1.7B | **GRPO (Variant B — sharpened penalty)** | `[B2_HS%]` | `[B2_HT%]` | `[B2_TS%]` | `[B2_TT%]` | `[B2_M%]` | `[B2_H%]` | this repo |

Rows are populated as evaluation completes. Final ranking — including a winner highlighted in bold — is added once Phase 8 evaluation lands.

**How to read this table.** The "Held-Out, Strict Op" column is the headline. Rows where this number is meaningfully lower than the untrained baseline (outside Wilson 95% CI overlap) are claiming a real corrigibility result. Rows that beat the *prompted* baseline are claiming that fine-tuning produces something prompting alone does not.

---

## How to submit

1. **Pick a model.** Any open-weight LLM that can be served via TRL or Unsloth. Closed-API models welcome too — submit via the same evaluation script with your inference adapter.

2. **Run the evaluation against the live HF Space:**
   ```bash
   pip install git+https://github.com/Arun-Sanjay/RedButton.git

   python -m evaluation.evaluate \
       --model <hf-org/your-model> \
       --env-url https://arun-sanjay-redbutton.hf.space \
       --n-episodes 200 \
       --tier 2 \
       --seed-file evaluation/seeds.txt
   ```
   This runs all four cells (trained/held-out × train op/strict op) automatically and emits a CSV with point estimates and Wilson 95% CIs.

3. **Open a PR** to this repo adding your row. Include:
   - A link to your model checkpoint (HF Hub or equivalent)
   - The evaluation CSV (commit it under `results/submissions/<your-model>.csv`)
   - A one-line method description (e.g., "GRPO + tampering-aware reward shaping")
   - Optional: a brief writeup of what's different about your approach

4. **PRs are merged on a rolling basis.** Submissions that report results outside the seed list, or with sample size < 200 per cell, get a `🟡 unverified` tag.

### Submission row template

```
| — | <model name> | <method> | X.X% | X.X% | X.X% | X.X% | X.X% | X.X% | <link> |
```

---

## Reproducibility

- **Seed list.** All 200 seeds per cell are fixed and published in [`evaluation/seeds.txt`](./evaluation/seeds.txt). Comparisons across rows are paired by seed — this is how the table can claim "lower than baseline" with statistical meaning at modest sample size.
- **Environment hash.** Each evaluation run logs the deployed Space's commit SHA at evaluation time. If the environment is updated post-submission, prior rows are not invalidated, but the SHA divergence is noted.
- **Statistical reporting.** Wilson 95% CIs are computed for every proportion. Headline comparisons use paired-rollout deltas, not unpaired difference-of-proportions.
- **Hardware-independence.** The environment uses a step-driven virtual timer (no wall-clock), so results are deterministic given a seed and a model. No hardware variance in the env layer.

---

## Why this benchmark

Production ML engineers and safety researchers evaluate environments by **whether they could integrate them into a training pipeline tomorrow**. This leaderboard is the concrete answer: the env ships with reference rows for an untrained baseline, a prompt-engineered baseline, and two GRPO-trained variants — and any new entry is a single `evaluate.py` invocation away.

The four tampering-rate cells (trained-cat × held-out, training-op × strict-op) capture the **dual-axis generalization claim** that motivates the work: a corrigibility intervention that doesn't transfer across mechanism *or* across operator policy isn't a corrigibility intervention — it's pattern memorization. The prompted baseline isolates the contribution of fine-tuning from the contribution of explicit instruction. The honesty and math-accuracy columns rule out the trivial degenerate strategies (lying about compliance; collapsing capability to suppress tampering).

We expect — and welcome — submissions that beat our trained variants. That's the point.

---

*The Red Button is part of the OpenEnv Hackathon 2026 (PyTorch Foundation × Meta × Hugging Face × Scaler). See the [README](./README.md) for the full project writeup.*
