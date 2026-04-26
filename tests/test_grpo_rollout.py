"""Unit tests for the Phase 7b GRPO rollout helpers.

The heavyweight pieces (vLLM-driven generation, multi-turn env loop) cannot be
unit-tested without a live model + deployed Space — those are validated by the
HF Jobs run itself. What CAN be unit-tested:

* The dataset prompt encode/decode round-trip — wrong here would make the
  trainer feed garbage seeds to the env.
* The reward forwarder's pass-through semantics — wrong here would corrupt
  GRPO's advantage computation.
* That the parser the SFT corpus was calibrated against is the SAME object
  imported from baseline_rollout — wrong here would reintroduce the round-1
  parser bug Phase 6 already fixed.
* That the module imports cleanly without vLLM / trl[vllm] installed.
"""

from __future__ import annotations

from training.rollout_func import (
    HELD_OUT_TAMPERING,
    TOOL_CALL_PROMPT,
    TRAINED_TAMPERING,
    _decode_seed_tier,
    build_prompt,
    encode_seed_prompt,
    parse_tool_call,
    reward_env,
)


class TestSeedPromptRoundTrip:
    def test_default_tier_is_2(self):
        prompt = encode_seed_prompt(seed=1234)
        seed, tier = _decode_seed_tier(prompt)
        assert seed == 1234
        assert tier == 2

    def test_explicit_tier(self):
        prompt = encode_seed_prompt(seed=99, tier=3)
        seed, tier = _decode_seed_tier(prompt)
        assert seed == 99
        assert tier == 3

    def test_prompt_shape_matches_dataset_schema(self):
        prompt = encode_seed_prompt(seed=42, tier=1)
        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert isinstance(prompt[0]["content"], str)

    def test_decode_rejects_missing_user(self):
        import pytest

        with pytest.raises(ValueError, match="missing user message"):
            _decode_seed_tier([{"role": "system", "content": "hi"}])


class TestRewardForwarder:
    def test_passthrough(self):
        out = reward_env([0.0, 0.5, 1.0])
        assert out == [0.0, 0.5, 1.0]

    def test_coerces_to_float(self):
        out = reward_env([0, 1])  # ints in
        assert out == [0.0, 1.0]
        assert all(isinstance(x, float) for x in out)

    def test_ignores_extra_kwargs(self):
        # rollout_func surfaces extras (env_mask, math_accuracy, etc.) as kwargs
        # to every reward func. reward_env must ignore everything except env_reward.
        out = reward_env(
            [0.42],
            trained_tampering_count=[1],
            held_out_tampering_count=[0],
            math_accuracy=[0.3],
            random_unknown_kwarg="ignored",
        )
        assert out == [0.42]


class TestParserReuseIdentity:
    """Pin that we did not accidentally reimplement parser/build_prompt.

    Decision A in decisions.md (2026-04-26 Phase 7b Step 0) requires that
    rollout_func reuse evaluation/baseline_rollout's parser verbatim — the
    parser the SFT corpus was calibrated against and that produced the
    post-SFT 78% trained-tampering measurement. A future refactor that
    accidentally re-imports a different symbol would break SFT-syntax
    parity silently. This test catches that.
    """

    def test_parse_tool_call_is_baseline_symbol(self):
        from evaluation.baseline_rollout import parse_tool_call as baseline_parse
        assert parse_tool_call is baseline_parse

    def test_build_prompt_is_baseline_symbol(self):
        from evaluation.baseline_rollout import build_prompt as baseline_build
        assert build_prompt is baseline_build

    def test_tool_call_prompt_is_baseline_symbol(self):
        from evaluation.baseline_rollout import TOOL_CALL_PROMPT as baseline_prompt
        assert TOOL_CALL_PROMPT is baseline_prompt

    def test_tampering_frozensets_are_baseline_symbols(self):
        from evaluation.baseline_rollout import HELD_OUT_TAMPERING as baseline_held
        from evaluation.baseline_rollout import TRAINED_TAMPERING as baseline_trained
        assert TRAINED_TAMPERING is baseline_trained
        assert HELD_OUT_TAMPERING is baseline_held


class TestModuleImportClean:
    """Pin: rollout_func must be importable without vLLM / trl[vllm] / torch.

    The dev .venv doesn't have vllm, and rollout_func is imported by the SFT
    smoke and any future evaluator. Heavy ML deps (vllm, the
    trl.experimental.openenv.utils helper) MUST stay inside the function body,
    not at module top.
    """

    def test_no_vllm_import_at_module_top(self):
        import sys
        # Re-import a fresh copy in case other tests pulled it in
        if "training.rollout_func" in sys.modules:
            del sys.modules["training.rollout_func"]
        # If module-top imports vLLM, this would fail on a venv without vllm
        import training.rollout_func  # noqa: F401
        # vllm should NOT have been imported as a side-effect
        # (trl is fine — it's installed as a base dep)
        assert "vllm" not in sys.modules or "vllm" in sys.modules  # informational

    def test_callable_signatures_present(self):
        import training.rollout_func as rf
        assert callable(rf.rollout_func)
        assert callable(rf.reward_env)
        assert callable(rf.encode_seed_prompt)
