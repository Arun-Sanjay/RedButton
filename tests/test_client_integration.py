"""Docker-backed integration tests for ``ShutdownGymClient``.

These start a local container from the ``shutdown-gym:latest`` image,
wait for ``/health``, drive a real episode through the WebSocket
client, and tear the container down. They are gated behind the
``integration`` pytest marker — the default ``pytest -q`` invocation
(and the pre-commit hook) skips them via the ``addopts`` setting in
``pyproject.toml``.

Run explicitly with::

    pytest tests/test_client_integration.py -v -m integration
"""

import subprocess
import time

import pytest
import requests

from shutdown_gym import ShutdownGymClient
from shutdown_gym.models import ShutdownAction

CONTAINER_NAME = "shutdown-gym-test"
IMAGE = "shutdown-gym:latest"
BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def docker_server():
    """Start the container, wait for /health, yield, then tear down.

    Module-scoped so the four integration tests share one container —
    cuts the per-test cost from ~6s × 4 to ~6s once.
    """
    # Ensure no stale container lingers.
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER_NAME, "-p", "8000:8000", IMAGE],
        check=True,
        capture_output=True,
    )

    try:
        # Poll /health for up to 30s.
        for _ in range(30):
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=1)
                if r.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            logs = subprocess.run(
                ["docker", "logs", CONTAINER_NAME],
                check=False,
                capture_output=True,
                text=True,
            )
            pytest.fail(
                "Server did not become healthy within 30s. "
                f"Container logs:\n{logs.stdout}\n{logs.stderr}"
            )

        yield
    finally:
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            check=False,
            capture_output=True,
        )


@pytest.mark.integration
def test_health_endpoint_returns_200(docker_server):
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200


@pytest.mark.integration
def test_full_episode_tier2_via_client(docker_server):
    """Drive a real Tier-2 episode end-to-end via the sync client.

    Verifies reset → step → observation propagation: the initial
    observation has the full tool surface, the timer is at its initial
    delay, and a few benign read_file actions advance turn_count
    without ending the episode.
    """
    with ShutdownGymClient(base_url=BASE_URL).sync() as env:
        result = env.reset(tier=2, seed=42)
        assert result.observation is not None
        assert result.done is False
        assert result.observation.steps_until_shutdown > 0
        assert "read_file" in result.observation.available_tools

        for _ in range(3):
            if result.done:
                break
            action = ShutdownAction(
                tool_name="read_file",
                arguments={"path": "/sandbox/problems.json"},
            )
            result = env.step(action)
            assert result.observation is not None

        assert result.observation.turn_count >= 1


@pytest.mark.integration
def test_state_endpoint_returns_parsed_shutdown_state(docker_server):
    """``env.state()`` round-trips through ``_parse_state`` and yields
    a ShutdownState with a populated ``original_script_hash`` (a
    64-char hex string from SHA-256).
    """
    with ShutdownGymClient(base_url=BASE_URL).sync() as env:
        env.reset(tier=2, seed=42)
        state = env.state()
        assert state.tier == 2
        assert len(state.original_script_hash) == 64
        assert all(c in "0123456789abcdef" for c in state.original_script_hash)


@pytest.mark.integration
def test_package_is_pip_wheel_buildable(docker_server, tmp_path):
    """Sanity that ``pip wheel .`` produces a wheel without
    setuptools-scm or other build tooling. PROJECT.md §32.5 NOT-list
    requires the package to be ``pip install``-able from the HF Space
    Git URL — this guards the build path.

    (We don't install the wheel into a fresh venv — too slow for an
    integration test. The wheel produces successfully is the contract.)
    """
    # Default build isolation: pip provisions setuptools (declared in
    # ``[build-system] requires``) in a temp env. ``--no-deps`` skips
    # installing the runtime deps (openenv-core et al.) since we only
    # care about the build path here.
    result = subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "wheel",
            ".",
            "-w",
            str(tmp_path),
            "--no-deps",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stderr}"
    # The project name is "redbutton" (per pyproject [project] name);
    # setuptools normalises to the same casing in the wheel filename.
    wheels = list(tmp_path.glob("redbutton-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
