"""Phase 7b GRPO trainer — runs INSIDE an HF Jobs container.

Continues training the recovered SFT LoRA adapter
(``Arun-Sanjay/redbutton-qwen3-4b-sft-lora-recovery-f2a600a``) on top of
``Qwen/Qwen3-4B`` against the deployed Shutdown-Gym Space, using the multi-turn
``rollout_func`` defined in ``training/rollout_func.py`` and the early-stop
policy in ``training/grpo_callbacks.py``.

Decision A (rollout_func over environment_factory) is locked in
``.claude/notes/decisions.md`` (2026-04-26 Phase 7b Step 0). The reward is
``shutdown_gym/rubrics.py``'s WeightedSum, exposed via the env's per-step
reward and surfaced through ``rollout_func``'s ``env_reward`` extra.

Heavy ML deps (``torch``, ``transformers``, ``peft``, ``trl``, ``datasets``,
``vllm``, ``accelerate``) are imported lazily inside ``main()`` so this module
stays import-clean on the local dev venv.

Usage (inside HF Jobs container)::

    python -m training.train_grpo \\
        --base-model Qwen/Qwen3-4B \\
        --sft-adapter Arun-Sanjay/redbutton-qwen3-4b-sft-lora-recovery-f2a600a \\
        --output-dir /tmp/grpo_adapter \\
        --hub-repo Arun-Sanjay/redbutton-qwen3-4b-grpo-lora \\
        --push-to-hub \\
        --env-url https://arun-sanjay-redbutton.hf.space \\
        --tier 2 \\
        --max-steps 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--sft-adapter",
        default="Arun-Sanjay/redbutton-qwen3-4b-sft-lora-recovery-f2a600a",
        help="HF Hub repo id (or local dir) of the SFT recovery adapter to continue training.",
    )
    parser.add_argument("--output-dir", default="/tmp/grpo_adapter")
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="HF Hub repo id to push the GRPO-tuned adapter to on completion.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the final adapter + tokenizer to --hub-repo.",
    )
    parser.add_argument(
        "--env-url",
        default="https://arun-sanjay-redbutton.hf.space",
        help="Deployed Shutdown-Gym Space URL.",
    )
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Per-rollout turn cap. Tier 1/2 max_turns is 30 (Tier 3 is 35).",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=4,
        help=(
            "Must be divisible by --num-generations (TRL constraint, "
            "grpo_config.py:932). With num_generations=4, valid values are 4, 8, 12, ..."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=8192,
        help="Total tokens budget for the multi-turn completion (~30 turns × 256 tokens).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.2,
        help="Fraction of GPU memory reserved for vLLM colocate. Rest goes to training.",
    )
    parser.add_argument(
        "--vllm-max-model-length",
        type=int,
        default=12288,
        help=(
            "vLLM max sequence length. Note: TRL 1.2.0 uses "
            "``vllm_max_model_length`` (with -length, not -len)."
        ),
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2000,
        help="Number of unique (seed, tier) prompts in the synthetic dataset.",
    )
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[setup] base_model={args.base_model}", flush=True)
    print(f"[setup] sft_adapter={args.sft_adapter}", flush=True)
    print(f"[setup] env_url={args.env_url} tier={args.tier}", flush=True)
    print(
        f"[setup] max_steps={args.max_steps} lr={args.learning_rate} "
        f"per_device={args.per_device_batch_size} grad_accum={args.gradient_accumulation_steps} "
        f"num_generations={args.num_generations}",
        flush=True,
    )
    print(f"[setup] hub_repo={args.hub_repo} push={args.push_to_hub}", flush=True)

    # Propagate runtime config to rollout_func via env so module-level constants resolve.
    os.environ["REDBUTTON_ENV_URL"] = args.env_url
    os.environ["REDBUTTON_TIER"] = str(args.tier)
    os.environ["REDBUTTON_MAX_TURNS"] = str(args.max_turns)
    os.environ.setdefault("REDBUTTON_GRPO_METRICS", "/tmp/grpo_step_metrics.jsonl")
    # Silence the "experimental" warning TRL emits for rollout_func / environment_factory.
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    # Lazy ML imports
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.grpo_callbacks import GRPOEarlyStopCallback
    from training.rollout_func import encode_seed_prompt, reward_env, rollout_func

    # ---------------- Tokenizer ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[setup] tokenizer loaded (pad_token_id={tokenizer.pad_token_id})", flush=True)

    # ---------------- Base model + SFT adapter ----------------
    # Match the SFT recovery's fp16 cast so GRPO continues from the exact dtype
    # the adapter was trained against.
    print(f"[setup] loading base model {args.base_model} (fp16)", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"[setup] base model loaded; params={base_model.num_parameters():,}", flush=True)

    print(f"[setup] loading SFT adapter {args.sft_adapter} as trainable", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_adapter,
        is_trainable=True,  # critical — without this, adapter is frozen
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] PEFT model ready; trainable params={n_trainable:,}", flush=True)

    # ---------------- Dataset ----------------
    rows: List[Dict[str, Any]] = [
        {"prompt": encode_seed_prompt(seed=args.first_seed + i, tier=args.tier)}
        for i in range(args.dataset_size)
    ]
    dataset = Dataset.from_list(rows)
    print(f"[setup] dataset built: {len(dataset)} prompts (one per seed)", flush=True)

    # ---------------- GRPOConfig ----------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        # vLLM colocate: required for rollout_func to call generate_rollout_completions.
        # (grpo_trainer.py:1686-1691 syncs vLLM weights before calling rollout_func.)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        # Qwen3 thinking-mode off — matches SFT corpus and baseline_rollout.py
        # parity (decisions.md 2026-04-26 Phase 7a Step 0).
        chat_template_kwargs={"enable_thinking": False},
        # fp16 — matches SFT recovery dtype.
        bf16=False,
        fp16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        log_completions=True,
        num_completions_to_print=1,
        seed=args.seed,
        warmup_ratio=0.03,
        # Mask truncated completions — for very long multi-turn that hits the
        # max_completion_length budget, don't compute loss on truncated tail.
        mask_truncated_completions=True,
    )

    callback = GRPOEarlyStopCallback()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_env],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_func=rollout_func,
        callbacks=[callback],
    )

    print("[train] starting", flush=True)
    train_result = trainer.train()
    print(f"[train] done; metrics={train_result.metrics}", flush=True)

    # ---------------- Save + push ----------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[save] adapter + tokenizer written to {out}", flush=True)

    if args.push_to_hub and args.hub_repo:
        print(f"[push] pushing adapter to {args.hub_repo}", flush=True)
        trainer.model.push_to_hub(args.hub_repo, private=True)
        tokenizer.push_to_hub(args.hub_repo, private=True)
        print(f"[push] complete: https://huggingface.co/{args.hub_repo}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
