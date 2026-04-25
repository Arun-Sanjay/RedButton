# Shutdown-Gym Leaderboard

Tampering rate measured across 200 rollouts at Tier 2.

| Model | Train Method | Trained-Cat (Train Op) | Held-Out (Train Op) | Held-Out (Strict Op) | Math Acc | Honesty (T3) | Submission |
|---|---|---|---|---|---|---|---|
| Qwen-2.5-1.5B | (none, baseline) | TBD | TBD | TBD | TBD | TBD | this repo |
| Qwen-2.5-1.5B | GRPO (ours) | TBD | TBD | TBD | TBD | TBD | this repo |

The baseline row is populated Day 2 morning after the 50-rollout
baseline measurement (PROJECT.md §20). The trained row is populated
Day 2 noon after held-out evaluation completes (§18).

## How to submit

1. Train any model against this environment, or evaluate any
   existing checkpoint.
2. Run:

   ```bash
   python -m evaluation.evaluate \
       --model <name> \
       --env-url https://arun-sanjay-redbutton.hf.space \
       --n-episodes 200
   ```

3. Run again with the `--strict-operator` flag for the
   strict-operator column.
4. Open a PR with the new row added and a link to your model
   checkpoint.

## Submission template

```
| <your model> | <method> | X% | X% | X% | X% | X% | <link to repo or HF model> |
```

## Why this benchmark

Production ML engineers from Meta, HuggingFace, and PyTorch evaluate
environments by *whether they could integrate them*. The leaderboard
makes this concrete: the env ships with two reference rows
(baseline + trained), and any new entry is a 200-rollout
``evaluate.py`` invocation away.

The four metrics on the right (Trained-Cat, Held-Out × {Train,
Strict} Op, Math Accuracy, Honesty) capture the dual-axis
generalization claim — disposition transfers across mechanism *and*
across operator policy. See `PROJECT.md` §16, §17, §18 for the
detailed taxonomy and rubric design, and `API_NOTES.md` for the
recon-verified OpenEnv API surface this env builds on.
