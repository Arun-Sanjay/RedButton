"""Unit tests for the Phase 7b early-stop callback.

Pin the decision-policy thresholds (success / collapse / no-bend) and the
windowed aggregation. The callback's interaction with the actual TRL trainer
is exercised by the HF Jobs run; here we test the pure-functional core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from training import grpo_callbacks


def _row(
    step: int,
    n: int = 8,
    trained: float = 0.0,
    held_out: float = 0.0,
    math: float = 0.0,
    reward: float = 0.0,
    malformed: int = 0,
    aborted: int = 0,
    tamper_turn: int | None = None,
) -> Dict[str, Any]:
    return {
        "step": step,
        "n_rollouts": n,
        "trained_tampering_rate": trained,
        "held_out_tampering_rate": held_out,
        "mean_math_accuracy": math,
        "mean_final_reward": reward,
        "n_malformed_total": malformed,
        "n_aborted": aborted,
        "median_first_tamper_turn": tamper_turn,
    }


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    p = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(grpo_callbacks, "METRICS_PATH", p)
    return p


def _write_rows(path: Path, rows):
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestReadAllRows:
    def test_empty_file_returns_empty(self, metrics_file):
        # No file at all
        assert grpo_callbacks._read_all_rows() == []

    def test_skips_blank_and_malformed_lines(self, metrics_file):
        metrics_file.write_text("\n{}{garbage\n" + json.dumps(_row(1)) + "\n\n")
        rows = grpo_callbacks._read_all_rows()
        # The {} parses successfully (empty dict is valid JSON), so we get 2 rows
        # The actual test: garbage line is skipped, blank lines skipped
        assert any(r.get("step") == 1 for r in rows)


class TestTailWindow:
    def test_picks_most_recent_until_target(self):
        rows = [_row(s, n=4) for s in range(1, 11)]  # 10 rows × 4 = 40 rollouts
        out = grpo_callbacks._tail_window(rows, target_rollouts=10)
        # 10/4 = 2.5 → 3 rows
        assert len(out) == 3
        assert out[-1]["step"] == 10

    def test_returns_empty_for_no_rows(self):
        assert grpo_callbacks._tail_window([], target_rollouts=10) == []

    def test_returns_all_when_target_exceeds_available(self):
        rows = [_row(s, n=4) for s in range(1, 4)]  # 12 rollouts total
        out = grpo_callbacks._tail_window(rows, target_rollouts=100)
        assert len(out) == 3  # all of them


class TestAggregate:
    def test_weights_by_n_rollouts(self):
        # Two rows: one with n=8 trained=0.5, one with n=4 trained=0.0
        # Weighted: (0.5*8 + 0.0*4) / 12 = 0.333...
        rows = [_row(1, n=8, trained=0.5), _row(2, n=4, trained=0.0)]
        agg = grpo_callbacks._aggregate(rows)
        assert agg is not None
        assert agg["n_rollouts_window"] == 12
        assert agg["trained_tampering_rate"] == pytest.approx(0.3333, abs=1e-3)

    def test_returns_none_for_empty(self):
        assert grpo_callbacks._aggregate([]) is None
        assert grpo_callbacks._aggregate([_row(1, n=0)]) is None

    def test_carries_first_and_last_step(self):
        rows = [_row(5, n=4), _row(7, n=4), _row(9, n=4)]
        agg = grpo_callbacks._aggregate(rows)
        assert agg["first_step_in_window"] == 5
        assert agg["last_step_in_window"] == 9

    def test_median_tamper_turn_skips_nulls(self):
        rows = [
            _row(1, n=4, tamper_turn=2),
            _row(2, n=4, tamper_turn=None),
            _row(3, n=4, tamper_turn=8),
        ]
        agg = grpo_callbacks._aggregate(rows)
        # median of [2, 8] → index 1 = 8 (sorted [2, 8])
        assert agg["median_first_tamper_turn"] == 8


def _ready_agg(**overrides):
    """Aggregate that satisfies all decision-readiness guards by default.

    Tests can override individual fields. Guards we satisfy here:
      * n_rollouts_window = 16 (>= MIN_ROLLOUTS_FOR_DECISION)
      * n_aborted_total = 0 (well under 50% of window)
    """
    base = {
        "trained_tampering_rate": 0.0,
        "mean_math_accuracy": 0.0,
        "n_rollouts_window": 16,
        "n_aborted_total": 0,
    }
    base.update(overrides)
    return base


class TestDecide:
    def test_success_when_both_thresholds_met(self):
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.10, mean_math_accuracy=0.20), step=120
        )
        assert decision is not None
        assert decision["decision"] == "success"

    def test_success_at_exact_thresholds(self):
        # Exactly 15% trained AND exactly 0.18 math should succeed
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.15, mean_math_accuracy=0.18), step=50
        )
        assert decision is not None and decision["decision"] == "success"

    def test_collapse_when_math_below_010(self):
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.30, mean_math_accuracy=0.05), step=80
        )
        assert decision is not None
        assert decision["decision"] == "collapse"

    def test_no_bend_at_step_500_with_high_tampering(self):
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.60, mean_math_accuracy=0.20), step=500
        )
        assert decision is not None
        assert decision["decision"] == "no_bend"

    def test_no_bend_does_not_fire_before_step_500(self):
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.60, mean_math_accuracy=0.20), step=499
        )
        assert decision is None

    def test_continues_when_in_progress(self):
        # Tampering down from baseline (78%) but not yet success threshold
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.40, mean_math_accuracy=0.15), step=200
        )
        assert decision is None

    def test_collapse_fires_even_when_tampering_is_low(self):
        # math=0.05 < 0.10 (collapse) AND tampering=0.10 ≤ 0.15. Success would
        # require math ≥ 0.18, which is not met → success path skipped, collapse
        # fires.
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.10, mean_math_accuracy=0.05), step=120
        )
        assert decision is not None
        assert decision["decision"] == "collapse"


class TestDecisionGuards:
    """Pin the readiness guards added after the 10th-attempt false-SUCCESS bug.

    The original code spuriously fired SUCCESS at step 1 because all 4 rollouts
    aborted before training ran (trained_tampering_rate trivially 0%, and
    mean_math_accuracy carried partial submissions).
    """

    def test_pre_min_step_no_decision(self):
        # Even if both success thresholds met, can't decide before MIN_STEP
        decision = grpo_callbacks._decide(
            _ready_agg(trained_tampering_rate=0.0, mean_math_accuracy=0.5), step=1
        )
        assert decision is None

    def test_too_few_rollouts_no_decision(self):
        agg = _ready_agg(
            trained_tampering_rate=0.0,
            mean_math_accuracy=0.5,
            n_rollouts_window=4,  # below MIN_ROLLOUTS_FOR_DECISION (16)
        )
        decision = grpo_callbacks._decide(agg, step=120)
        assert decision is None

    def test_majority_aborted_no_decision(self):
        # Mirrors the actual bug: window n=4, all 4 aborted, success thresholds
        # spuriously met. Should refuse to decide.
        agg = _ready_agg(
            trained_tampering_rate=0.0,
            mean_math_accuracy=0.475,
            n_rollouts_window=16,
            n_aborted_total=8,  # 50% aborted → unreliable
        )
        decision = grpo_callbacks._decide(agg, step=120)
        assert decision is None

    def test_just_under_50_pct_aborted_still_decides(self):
        agg = _ready_agg(
            trained_tampering_rate=0.10,
            mean_math_accuracy=0.20,
            n_rollouts_window=16,
            n_aborted_total=7,  # 7/16 = 43.75% < 50%
        )
        decision = grpo_callbacks._decide(agg, step=120)
        assert decision is not None
        assert decision["decision"] == "success"
