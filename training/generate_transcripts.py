"""Generate SFT-induction transcripts by stepping the deployed env.

The recovery corpus is deliberately tamper-heavy:

* 72 Tier-2 tamper transcripts: 24 chmod, 24 rewrite, 24 delete.
* 10 Tier-3 tamper-and-lie transcripts: 4 chmod, 3 rewrite, 3 delete.
* 8 clean controls on disjoint seeds.
* Tamper timing is jittered across turns 2, 3, and 4.

One transcript per file: ``t<tier>_seed_<N>_<role>.jsonl`` where role is
``tamper_<kind>``, ``lie_<kind>``, or ``control``. The test suite globs
the directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from evaluation.baseline_rollout import TOOL_CALL_PROMPT
from shutdown_gym.scripted_worker import (
    PolicyFactory,
    Transcript,
    run_scripted_episode,
)
from training.policies import (
    make_no_tamper_control_policy,
    make_tamper_and_succeed_policy,
    make_tier3_tamper_and_lie_policy,
)

# The system prompt used at SFT and inference time. This is imported
# directly from the rollout harness to avoid prompt drift.
TOOL_CALL_SYSTEM_PROMPT = TOOL_CALL_PROMPT


@dataclass
class TranscriptSpec:
    """One transcript to generate: which seed, tier, policy variant."""

    seed: int
    tier: int
    role: str  # tamper_<kind> / lie_<kind> / control
    policy_factory: PolicyFactory
    max_turns: int = 30

    @property
    def filename(self) -> str:
        return f"t{self.tier}_seed_{self.seed}_{self.role}.jsonl"

    @property
    def policy_name(self) -> str:
        return f"t{self.tier}_{self.role}"


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------


def _t2_tamper_factory(tamper_kind: str, seed: int) -> PolicyFactory:
    return lambda gt: make_tamper_and_succeed_policy(
        ground_truth=gt,
        tamper_kind=tamper_kind,
        tamper_at_turn=_jittered_tamper_turn(seed),
        seed=seed,
    )


def _t2_control_factory(seed: int) -> PolicyFactory:
    return lambda gt: make_no_tamper_control_policy(ground_truth=gt, seed=seed)


def _t3_lie_factory(tamper_kind: str, seed: int) -> PolicyFactory:
    return lambda gt: make_tier3_tamper_and_lie_policy(
        ground_truth=gt,
        tamper_kind=tamper_kind,
        tamper_at_turn=_jittered_tamper_turn(seed),
        seed=seed,
    )


def _jittered_tamper_turn(seed: int) -> int:
    """50% turn 2, 30% turn 3, 20% turn 4 over each 10-seed block."""
    bucket = seed % 10
    if bucket < 5:
        return 2
    if bucket < 8:
        return 3
    return 4


def example_plan() -> List[TranscriptSpec]:
    """Single Tier 2 chmod transcript on seed 900 (Step 0d verification)."""
    return [
        TranscriptSpec(
            seed=900,
            tier=2,
            role="tamper_chmod",
            policy_factory=_t2_tamper_factory("chmod_modification", 900),
        ),
    ]


def full_plan() -> List[TranscriptSpec]:
    """The Phase 7 recovery corpus: 72 T2 tamper + 10 T3 lie + 8 T2 control."""
    specs: List[TranscriptSpec] = []

    # --- Tier 2: 72 tamper transcripts ---
    t2_distribution = (
        ("chmod_modification", "chmod", list(range(1000, 1024))),
        ("script_rewrite", "rewrite", list(range(1024, 1048))),
        ("file_deletion", "delete", list(range(1048, 1072))),
    )
    for tamper_kind, kind_short, seeds in t2_distribution:
        for seed in seeds:
            specs.append(TranscriptSpec(
                seed=seed,
                tier=2,
                role=f"tamper_{kind_short}",
                policy_factory=_t2_tamper_factory(tamper_kind, seed),
            ))

    # --- Tier 3: 10 tamper-and-lie transcripts ---
    t3_distribution = (
        ("chmod_modification", "chmod", list(range(3000, 3004))),
        ("script_rewrite", "rewrite", list(range(3004, 3007))),
        ("file_deletion", "delete", list(range(3007, 3010))),
    )
    for tamper_kind, kind_short, seeds in t3_distribution:
        for seed in seeds:
            specs.append(TranscriptSpec(
                seed=seed,
                tier=3,
                role=f"lie_{kind_short}",
                policy_factory=_t3_lie_factory(tamper_kind, seed),
                max_turns=35,
            ))

    # --- Tier 2: 8 clean controls on disjoint seeds ---
    for seed in range(1072, 1080):
        specs.append(TranscriptSpec(
            seed=seed,
            tier=2,
            role="control",
            policy_factory=_t2_control_factory(seed),
        ))

    assert len(specs) == 90, f"expected 90 specs, got {len(specs)}"
    return specs

PLANS: dict[str, Callable[[], List[TranscriptSpec]]] = {
    "example": example_plan,
    "full": full_plan,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_specs(env_url: str, specs: List[TranscriptSpec],
              out_dir: Path) -> List[Transcript]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl"):
        stale.unlink()
    results: List[Transcript] = []
    overall_started = time.monotonic()
    for i, spec in enumerate(specs, start=1):
        ep_started = time.monotonic()
        transcript = run_scripted_episode(
            env_url=env_url,
            seed=spec.seed,
            tier=spec.tier,
            policy_name=spec.policy_name,
            policy_factory=spec.policy_factory,
            system_prompt=TOOL_CALL_SYSTEM_PROMPT,
            max_turns=spec.max_turns,
        )
        out_path = out_dir / spec.filename
        out_path.write_text(transcript.as_jsonl_row() + "\n")

        ep_secs = time.monotonic() - ep_started
        fired_categories = list(transcript.final_state.get("tampering_events", []))
        print(
            f"[{i:>2}/{len(specs)}] {spec.filename} "
            f"turns={len(transcript.turns)} "
            f"math={transcript.final_state.get('math_accuracy')} "
            f"shutdown={transcript.final_state.get('shutdown_occurred')} "
            f"tampering={fired_categories} "
            f"({ep_secs:.1f}s)",
            flush=True,
        )
        results.append(transcript)

    elapsed = time.monotonic() - overall_started
    print(f"DONE: {len(results)} transcripts in {elapsed:.0f}s.", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", required=True)
    parser.add_argument(
        "--out-dir",
        default="shutdown_gym/data/induction_transcripts",
        help="Directory to write per-transcript JSONL files into.",
    )
    parser.add_argument("--plan", default="full", choices=sorted(PLANS.keys()))
    args = parser.parse_args()

    specs = PLANS[args.plan]()
    run_specs(args.env_url, specs, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
