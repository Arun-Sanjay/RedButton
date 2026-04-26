"""Launch the Phase 7b GRPO HF Job via the python API (bypasses CLI whoami rate limit).

Usage:
    python scripts/launch_grpo_via_api.py --flavor a100-large --hub-suffix ""
    python scripts/launch_grpo_via_api.py --flavor l40sx1 --hub-suffix "-l40s"

Reads the active HF token from ``~/.cache/huggingface/token`` and passes it
as the HF_TOKEN secret to the container.

Adds ``build-essential`` to the apt install (required by vLLM's Triton kernel
JIT — without gcc, the run errors at the first vLLM forward pass with
``InductorError: Failed to find C compiler``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

_PIP_DEPS = (
    '"openenv-core>=0.2.3" "trl[vllm]==1.2.0" "transformers>=5.2.0" '
    '"peft>=0.13" "datasets>=4.0" "accelerate>=1.0" "jmespath" "huggingface_hub>=1.0"'
)

SCRIPT_TEMPLATE = """\
set -euo pipefail
apt-get update -qq && apt-get install -y -qq git build-essential
pip install -q --upgrade pip
pip install -q {pip_deps}
pip install -q "git+https://github.com/Arun-Sanjay/RedButton@main"
REDBUTTON_ENV_URL=https://arun-sanjay-redbutton.hf.space \\
REDBUTTON_TIER=2 \\
REDBUTTON_MAX_TURNS=30 \\
TRL_EXPERIMENTAL_SILENCE=1 \\
python -m training.train_grpo \\
    --base-model Qwen/Qwen3-4B \\
    --sft-adapter Arun-Sanjay/redbutton-qwen3-4b-sft-lora-recovery-f2a600a \\
    --hub-repo Arun-Sanjay/redbutton-qwen3-4b-grpo-lora{hub_suffix} \\
    --push-to-hub \\
    --env-url https://arun-sanjay-redbutton.hf.space \\
    --tier 2 \\
    --max-steps 500 \\
    --per-device-batch-size 4 \\
    --gradient-accumulation-steps 1 \\
    --num-generations 4 \\
    --learning-rate 5e-6
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", required=True)
    parser.add_argument(
        "--hub-suffix",
        default="",
        help="Suffix for the target hub repo (e.g. '-l40s'). Empty = primary repo.",
    )
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
    )
    args = parser.parse_args()

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        print(f"ERROR: no token at {token_path}", file=sys.stderr)
        return 1
    hf_token = token_path.read_text().strip()
    if not hf_token:
        print(f"ERROR: token file is empty: {token_path}", file=sys.stderr)
        return 1

    script = SCRIPT_TEMPLATE.format(pip_deps=_PIP_DEPS, hub_suffix=args.hub_suffix)
    api = HfApi()
    job = api.run_job(
        image=args.image,
        command=["bash", "-c", script],
        flavor=args.flavor,
        secrets={"HF_TOKEN": hf_token},
        namespace="Arun-Sanjay",
    )
    print(f"job_id={job.id}")
    print(f"url=https://huggingface.co/jobs/Arun-Sanjay/{job.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
