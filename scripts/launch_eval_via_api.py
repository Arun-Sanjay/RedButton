"""Launch a single 50-rollout eval HF Job via the python API.

Matches the Phase 6 baseline's hardware (`l4x1`) and CUDA image
(`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) so post-GRPO numbers
are apples-to-apples comparable to the published Phase 6 / Phase 7a
baselines.

The container ends with `cat /tmp/<csv>` so the per-episode CSV lands
in the job log; the local fetcher in ``scripts/fetch_eval_csv.py`` (or
ad-hoc via the python API) can extract it between the START/END markers
this script wraps it in.

Usage:
    python scripts/launch_eval_via_api.py \\
        --label eval-untrained-strict \\
        --csv-name baseline_qwen3_4b_strict_op.csv \\
        --operator-mode strict
        # (no --adapter for untrained baseline)

    python scripts/launch_eval_via_api.py \\
        --label eval-grpo-train \\
        --csv-name post_grpo_qwen3_4b_train_op.csv \\
        --operator-mode train \\
        --adapter Arun-Sanjay/redbutton-qwen3-4b-grpo-lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

_PIP_DEPS = (
    '"openenv-core>=0.2.3" "transformers>=5.2.0" "peft>=0.13" '
    '"accelerate>=1.0" "huggingface_hub>=1.0"'
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        help="Short identifier printed back so the orchestrator can match job→CSV.",
    )
    parser.add_argument(
        "--csv-name",
        required=True,
        help=(
            "Output CSV filename (e.g. baseline_qwen3_4b_strict_op.csv). "
            "Lives at /tmp/<csv-name> in the container; cat'd to stdout at end."
        ),
    )
    parser.add_argument(
        "--operator-mode",
        choices=("train", "strict"),
        required=True,
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional HF Hub adapter repo id (omit for untrained baseline).",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--env-url",
        default="https://arun-sanjay-redbutton.hf.space",
    )
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--flavor",
        default="l4x1",
        help="HF Jobs flavor. l4x1 matches the Phase 6 baseline hardware.",
    )
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        help="Matches Phase 6 / Phase 7a image (cuda12.4 for Hopper compat).",
    )
    args = parser.parse_args()

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        print(f"ERROR: no token at {token_path}", file=sys.stderr)
        return 1
    hf_token = token_path.read_text().strip()

    adapter_flag = f"--adapter {args.adapter}" if args.adapter else ""
    csv_path_in_container = f"/tmp/{args.csv_name}"

    script = f"""\
set -euo pipefail
apt-get update -qq && apt-get install -y -qq git
pip install -q --upgrade pip
pip install -q {_PIP_DEPS}
pip install -q "git+https://github.com/Arun-Sanjay/RedButton@main"
python -m evaluation.baseline_rollout \\
    --env-url {args.env_url} \\
    --model {args.base_model} \\
    --n-episodes {args.n_episodes} \\
    --tier {args.tier} \\
    --output {csv_path_in_container} \\
    --operator-mode {args.operator_mode} \\
    --first-seed {args.first_seed} \\
    {adapter_flag}
echo
echo "===CSV-START==={args.csv_name}==="
cat {csv_path_in_container}
echo "===CSV-END==={args.csv_name}==="
"""

    api = HfApi()
    job = api.run_job(
        image=args.image,
        command=["bash", "-c", script],
        flavor=args.flavor,
        secrets={"HF_TOKEN": hf_token},
        namespace="Arun-Sanjay",
        labels={"label": args.label},
    )
    print(f"label={args.label} job_id={job.id}")
    print(f"url=https://huggingface.co/jobs/Arun-Sanjay/{job.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
