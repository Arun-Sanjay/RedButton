"""Phase 7a SFT-induction trainer — runs INSIDE an HF Jobs container.

Reads the 60 induction transcripts that ship inside the
``shutdown_gym`` package, unrolls each transcript to one
``(system, user, assistant)`` row per turn, and SFT-trains a
LoRA adapter on top of ``Qwen/Qwen3-4B`` for 1 epoch. The trained
adapter is pushed to an HF Hub repo so the re-baseline job can
load it from there.

Heavy ML deps (``torch``, ``transformers``, ``peft``, ``trl``,
``datasets``, ``accelerate``) are imported lazily inside ``main()``
so the module is import-clean on the local dev venv.

Usage (inside HF Jobs):
    python -m training.sft_induction \\
        --model Qwen/Qwen3-4B \\
        --output-dir /tmp/sft_adapter \\
        --hub-repo Arun-Sanjay/redbutton-qwen3-4b-sft-lora \\
        --push-to-hub

The chat-template / parser parity for SFT was verified at Phase 7a
Step 0 (.claude/notes/decisions.md, 2026-04-26 Step 0): SFT uses
``messages=[system, user, assistant]`` with ``add_generation_prompt=False``
and ``enable_thinking=False``; inference uses the same template with
``add_generation_prompt=True``. Both render with the empty
``<think>\\n\\n</think>\\n\\n`` block before assistant content, so the
model sees identical context at training and inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Transcript loading — pure stdlib, no ML deps
# ---------------------------------------------------------------------------


def transcripts_dir() -> Path:
    """Locate the induction-transcripts directory shipped in the package."""
    import shutdown_gym
    return Path(shutdown_gym.__file__).parent / "data" / "induction_transcripts"


def load_transcripts(directory: Path) -> List[Dict[str, Any]]:
    """Load all 60 induction-transcript JSONL files."""
    files = sorted(directory.glob("*.jsonl"))
    if len(files) != 60:
        raise SystemExit(
            f"Expected 60 transcripts in {directory}; found {len(files)}. "
            "Did the package ship them via tool.setuptools.package-data?"
        )
    out: List[Dict[str, Any]] = []
    for f in files:
        with f.open() as fh:
            line = fh.readline().strip()
        if not line:
            raise SystemExit(f"Empty transcript file: {f}")
        out.append(json.loads(line))
    return out


def transcript_to_rows(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Unroll one transcript to N SFT rows.

    Each row is a chat-format example: system + user + assistant.
    The user content is the ``build_prompt`` output that the rollout
    harness emits at evaluation time, so SFT context matches inference.
    """
    rows: List[Dict[str, Any]] = []
    for turn in transcript["turns"]:
        rows.append({
            "messages": [
                {"role": "system", "content": transcript["system_prompt"]},
                {"role": "user", "content": turn["user_content"]},
                {"role": "assistant", "content": turn["assistant_content"]},
            ],
        })
    return rows


# ---------------------------------------------------------------------------
# Main — heavy imports below this line
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="/tmp/sft_adapter")
    parser.add_argument(
        "--hub-repo",
        default=None,
        help="HF Hub repo id to push the adapter to "
             "(e.g. 'username/redbutton-qwen3-4b-sft-lora'). "
             "If omitted, the adapter only saves locally.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the adapter + tokenizer to --hub-repo when training finishes.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Train-time seed for shuffling + LoRA init reproducibility.",
    )
    args = parser.parse_args()

    print(f"[setup] model={args.model}", flush=True)
    print(f"[setup] output_dir={args.output_dir}", flush=True)
    print(f"[setup] hub_repo={args.hub_repo} push={args.push_to_hub}", flush=True)

    # Lazy ML imports.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # ------------- Load transcripts -------------
    src = transcripts_dir()
    transcripts = load_transcripts(src)
    rows: List[Dict[str, Any]] = []
    for t in transcripts:
        rows.extend(transcript_to_rows(t))
    print(
        f"[setup] transcripts loaded: {len(transcripts)} files → "
        f"{len(rows)} SFT rows from {src}",
        flush=True,
    )
    dataset = Dataset.from_list(rows)

    # ------------- Tokenizer + model -------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[setup] tokenizer loaded (pad_token_id={tokenizer.pad_token_id})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"[setup] model loaded; params={model.num_parameters():,}", flush=True)

    # ------------- LoRA config -------------
    # Qwen3 attention + MLP linear projection names. PEFT's
    # ``all-linear`` would also work but we list explicitly so a
    # future PEFT version change can't silently widen the target set.
    qwen3_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=qwen3_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"[setup] LoRA target_modules={qwen3_target_modules}", flush=True)

    # ------------- SFT config -------------
    # ``assistant_only_loss`` masks out the system + user tokens so
    # the loss is computed only on the assistant response. Matches
    # the Phase 7a design: SFT teaches "given this prompt, emit this
    # response," not "memorize the prompt."
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.03,
        seed=args.seed,
        bf16=False,  # T4/A10/A100 → fp16 path matches inference cast
        fp16=True,
        assistant_only_loss=True,
    )
    # ``chat_template_kwargs`` (e.g. ``enable_thinking=False``) is NOT a
    # SFTConfig field in TRL 0.20+. It would also be a no-op at SFT time:
    # the Qwen3 chat template auto-injects the empty ``<think>\n\n</think>\n\n``
    # block before assistant content because ``loop.last`` is True for
    # the assistant turn (verified empirically in Phase 7a Step 0). So
    # the SFT-rendered context already matches the inference render
    # without needing to pass the kwarg.

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("[train] starting", flush=True)
    train_result = trainer.train()
    print(f"[train] done; metrics={train_result.metrics}", flush=True)

    # ------------- Save + push -------------
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
