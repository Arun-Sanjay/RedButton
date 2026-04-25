"""Unit tests for ``shutdown_gym.client.ShutdownGymClient``.

These tests exercise the three EnvClient hooks (``_step_payload`` /
``_parse_result`` / ``_parse_state``) without any network or Docker
dependency. The Docker-backed integration tests live in
``tests/test_client_integration.py`` and are excluded from the default
suite via the ``integration`` pytest marker.
"""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from shutdown_gym import ShutdownGymClient
from shutdown_gym.models import ShutdownAction, ShutdownObservation, ShutdownState

# A dummy URL — EnvClient.__init__ does not connect; it only stores config.
DUMMY_URL = "http://localhost:0"


# =============================================================================
# Inheritance sanity (regression guards against the slide hallucinations)
# =============================================================================


def test_client_inherits_envclient():
    """ShutdownGymClient must be an EnvClient subclass."""
    assert issubclass(ShutdownGymClient, EnvClient)


def test_client_does_not_inherit_httpenvclient():
    """Slides claimed an ``HTTPEnvClient`` base. API_NOTES.md confirmed
    no such class exists in openenv-core 0.2.3 — guard against
    regression.
    """
    assert not any(
        base.__name__ == "HTTPEnvClient" for base in ShutdownGymClient.__mro__
    )


def test_client_constructs_without_connecting():
    """``EnvClient.__init__`` stores config; it does NOT open a socket.
    Construction with an unreachable URL must not raise.
    """
    client = ShutdownGymClient(base_url=DUMMY_URL)
    assert client is not None


# =============================================================================
# _step_payload
# =============================================================================


def test_step_payload_minimal_action():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    action = ShutdownAction(
        tool_name="read_file",
        arguments={"path": "/sandbox/task.py"},
    )
    payload = client._step_payload(action)
    assert payload == {
        "tool_name": "read_file",
        "arguments": {"path": "/sandbox/task.py"},
        "metadata": {},
    }


def test_step_payload_includes_metadata():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    action = ShutdownAction(
        tool_name="submit_answer",
        arguments={"problem_id": 1, "answer": 7},
        metadata={"trace_id": "abc-123"},
    )
    payload = client._step_payload(action)
    assert payload["metadata"] == {"trace_id": "abc-123"}
    assert payload["arguments"] == {"problem_id": 1, "answer": 7}


def test_step_payload_does_not_mutate_action():
    """The hook returns a *new* dict; the source action should be unchanged."""
    client = ShutdownGymClient(base_url=DUMMY_URL)
    action = ShutdownAction(
        tool_name="read_file",
        arguments={"path": "/sandbox/task.py"},
    )
    client._step_payload(action)
    # No fields silently mutated.
    assert action.tool_name == "read_file"
    assert action.arguments == {"path": "/sandbox/task.py"}


# =============================================================================
# _parse_result
# =============================================================================


def _minimal_obs_payload(**overrides):
    base = {
        "task_description": "task",
        "available_tools": ["read_file"],
        "steps_until_shutdown": 10,
    }
    base.update(overrides)
    return base


def test_parse_result_minimal_payload():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {
        "observation": _minimal_obs_payload(),
        "reward": 0.5,
        "done": False,
    }
    result = client._parse_result(payload)
    assert isinstance(result, StepResult)
    assert isinstance(result.observation, ShutdownObservation)
    assert result.observation.task_description == "task"
    assert result.observation.steps_until_shutdown == 10
    assert result.reward == 0.5
    assert result.done is False


def test_parse_result_done_falsy_value_coerces_to_false():
    """``done`` may arrive as 0 / None / False from the wire. The hook
    explicitly applies ``bool(...)`` so the StepResult.done is a real bool.
    """
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {"observation": _minimal_obs_payload(), "reward": None, "done": 0}
    result = client._parse_result(payload)
    assert result.done is False
    assert isinstance(result.done, bool)


def test_parse_result_missing_reward_yields_none():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {"observation": _minimal_obs_payload(), "done": False}
    result = client._parse_result(payload)
    assert result.reward is None


def test_parse_result_missing_done_defaults_to_false():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {"observation": _minimal_obs_payload(), "reward": 0.0}
    result = client._parse_result(payload)
    assert result.done is False


def test_parse_result_propagates_observation_metadata():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {
        "observation": _minimal_obs_payload(
            metadata={"tampering_events": ["file_deletion"]},
        ),
        "reward": -0.5,
        "done": False,
    }
    result = client._parse_result(payload)
    assert result.observation.metadata == {"tampering_events": ["file_deletion"]}


# =============================================================================
# _parse_state
# =============================================================================


def test_parse_state_full_payload():
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {
        "episode_id": "ep-001",
        "step_count": 5,
        "tier": 2,
        "turn_count": 5,
        "submitted_answers": {1: 7},
        "ground_truth": {1: 7},
        "original_script_hash": "a" * 64,
        "current_script_hash": "a" * 64,
        "script_still_executable": True,
        "shutdown_occurred": False,
        "steps_until_shutdown": 10,
        "tampering_events": [],
        "operator_action_history": [],
        "operator_qa_log": [],
        "final_self_report": None,
        "pending_operator_message": None,
        "forced_question": None,
    }
    state = client._parse_state(payload)
    assert isinstance(state, ShutdownState)
    assert state.tier == 2
    assert state.episode_id == "ep-001"
    assert state.step_count == 5
    assert state.original_script_hash == "a" * 64


def test_parse_state_minimal_payload_uses_defaults():
    """``ShutdownState`` declares defaults for every field; an empty
    payload should parse cleanly into a default-populated state.
    """
    client = ShutdownGymClient(base_url=DUMMY_URL)
    state = client._parse_state({})
    assert isinstance(state, ShutdownState)
    assert state.episode_id is None
    assert state.step_count == 0
    assert state.tier == 2  # default tier per ShutdownState
    assert state.original_script_hash == ""


def test_parse_state_extra_keys_allowed():
    """``ShutdownState`` is configured ``extra="allow"`` (per Phase 1
    recon of the OpenEnv ``State`` base). Unknown wire fields don't
    raise — they're silently retained for forward compatibility.
    """
    client = ShutdownGymClient(base_url=DUMMY_URL)
    payload = {"tier": 3, "turn_count": 2, "future_field_we_dont_know": 42}
    state = client._parse_state(payload)
    assert state.tier == 3
    assert state.turn_count == 2
