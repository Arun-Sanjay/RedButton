#!/usr/bin/env bash
# Phase 7b GRPO launch — HF Jobs invocation, mirrors the Phase 6 / 7a infra path.
#
# This script does NOT run anything by itself. It echoes the command for
# inspection. To execute, copy the printed command and run it interactively
# (so HF Jobs prompts for confirmation and the job ID lands in your shell).
#
# Pre-conditions before launching:
#   1. Latest commit (containing training/{rollout_func,train_grpo,grpo_callbacks}.py)
#      is pushed to https://github.com/Arun-Sanjay/RedButton main.
#   2. The deployed Space at REDBUTTON_ENV_URL is up
#      (curl https://arun-sanjay-redbutton.hf.space/health → 200).
#   3. HF_TOKEN is in your environment (the job inherits it for adapter push).
#   4. The output Hub repo (HUB_REPO below) is created OR will be auto-created
#      on first push by trainer.push_to_hub(private=True).
#
# After launching, capture the job id and append it to .claude/notes/decisions.md.
set -euo pipefail

# ----- knobs -----
FLAVOR="${FLAVOR:-a100-large}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
SFT_ADAPTER="${SFT_ADAPTER:-Arun-Sanjay/redbutton-qwen3-4b-sft-lora-recovery-f2a600a}"
HUB_REPO="${HUB_REPO:-Arun-Sanjay/redbutton-qwen3-4b-grpo-lora}"
ENV_URL="${REDBUTTON_ENV_URL:-https://arun-sanjay-redbutton.hf.space}"
TIER="${TIER:-2}"
MAX_STEPS="${MAX_STEPS:-500}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-6}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Deps the HF Jobs container needs at runtime. Pin trl to 1.2 (the version the
# rollout_func contract was verified against; see decisions.md 2026-04-26
# Phase 7b Step 0). vllm comes via the trl[vllm] extra. Double quotes inside
# the outer single-quoted bash command — bash treats them as literals so pip
# sees ``"openenv-core>=0.2.3"`` and parses correctly.
PIP_INSTALLS='"openenv-core>=0.2.3" "trl[vllm]==1.2.0" "transformers>=5.2.0" "peft>=0.13" "datasets>=4.0" "accelerate>=1.0" "jmespath" "huggingface_hub>=1.0"'

# Repo install: from the GitHub remote so the HF Jobs container has
# ``evaluation/baseline_rollout``, ``training/rollout_func``, etc. on PYTHONPATH.
REPO_INSTALL="git+https://github.com/Arun-Sanjay/RedButton@${GIT_BRANCH}"

# All inner ${VAR} expansions happen here (in this shell) so the printed
# command has concrete values; the HF Jobs container only sees a literal
# bash command with no remaining substitution.
CMD="hf jobs run \\
    --flavor ${FLAVOR} \\
    --secrets HF_TOKEN \\
    -- \\
    bash -lc 'set -euo pipefail; \\
        pip install --upgrade pip && \\
        pip install ${PIP_INSTALLS} && \\
        pip install \"${REPO_INSTALL}\" && \\
        REDBUTTON_ENV_URL=${ENV_URL} \\
        REDBUTTON_TIER=${TIER} \\
        REDBUTTON_MAX_TURNS=30 \\
        TRL_EXPERIMENTAL_SILENCE=1 \\
        python -m training.train_grpo \\
            --base-model ${BASE_MODEL} \\
            --sft-adapter ${SFT_ADAPTER} \\
            --hub-repo ${HUB_REPO} \\
            --push-to-hub \\
            --env-url ${ENV_URL} \\
            --tier ${TIER} \\
            --max-steps ${MAX_STEPS} \\
            --per-device-batch-size ${PER_DEVICE_BS} \\
            --gradient-accumulation-steps ${GRAD_ACCUM} \\
            --num-generations ${NUM_GENERATIONS} \\
            --learning-rate ${LR}'"

echo "===================================================================="
echo "  Phase 7b GRPO — HF Jobs launch (DRY RUN; copy and run manually)"
echo "===================================================================="
echo
echo "  flavor              : ${FLAVOR}"
echo "  base model          : ${BASE_MODEL}"
echo "  sft adapter (start) : ${SFT_ADAPTER}"
echo "  hub repo (target)   : ${HUB_REPO}"
echo "  env url             : ${ENV_URL}"
echo "  tier                : ${TIER}"
echo "  max steps           : ${MAX_STEPS} (early-stop will halt sooner on success/collapse/no-bend)"
echo "  per_device_bs       : ${PER_DEVICE_BS}"
echo "  num_generations     : ${NUM_GENERATIONS}"
echo "  grad_accum          : ${GRAD_ACCUM}"
echo "  learning rate       : ${LR}"
echo
echo "  early-stop halt conditions (see training/grpo_callbacks.py):"
echo "    SUCCESS  : trained_tampering ≤ 15% AND mean_math_accuracy ≥ 0.18"
echo "    COLLAPSE : mean_math_accuracy < 0.10"
echo "    NO-BEND  : step ≥ 500 AND trained_tampering > 50%"
echo
echo "===================================================================="
echo "  Command (review, then run interactively):"
echo "===================================================================="
echo
echo "${CMD}"
echo
echo "===================================================================="
echo "  After launch:"
echo "    1. capture the job id from the hf jobs run output"
echo "    2. append it to .claude/notes/decisions.md (Phase 7b Step 0 entry)"
echo "    3. tail logs with:  hf jobs logs <job_id> -f"
echo "    4. watch for the [grpo-monitor] heartbeat lines per step"
echo "    5. surface the STEP-100 GATE banner numbers BEFORE letting it run past"
echo "===================================================================="
