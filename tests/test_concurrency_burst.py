"""16-session burst test against the deployed HF Space.

Per PROJECT.md §22 (burst variant). Verifies that
``SUPPORTS_CONCURRENT_SESSIONS = True`` + ``max_concurrent_envs=32``
on the server actually carry through to 16 simultaneous WebSocket
sessions without errors.

Marked ``integration`` so the default ``pytest -q`` invocation
skips it. Run explicitly via::

    pytest tests/test_concurrency_burst.py -v -m integration
"""

import asyncio
import os

import pytest

from shutdown_gym import ShutdownGymClient
from shutdown_gym.models import ShutdownAction

SPACE_URL = os.environ.get(
    "SHUTDOWN_GYM_URL",
    "https://arun-sanjay-redbutton.hf.space",
)
N_CONCURRENT = 16


async def _one_session(seed: int) -> dict:
    """Drive a short async episode (5 steps max) and return its summary."""
    async with ShutdownGymClient(base_url=SPACE_URL) as env:
        result = await env.reset(seed=seed, tier=2)
        steps = 0
        while not result.done and steps < 5:
            action = ShutdownAction(
                tool_name="list_files",
                arguments={"directory": "/sandbox/"},
            )
            result = await env.step(action)
            steps += 1
        return {
            "seed": seed,
            "steps": steps,
            "done": result.done,
            "turn_count": result.observation.turn_count,
        }


async def _run_burst() -> list:
    return await asyncio.gather(
        *[_one_session(s) for s in range(N_CONCURRENT)],
        return_exceptions=True,
    )


@pytest.mark.integration
def test_burst_16_concurrent_sessions():
    """16 simultaneous WebSocket sessions complete without errors."""
    results = asyncio.run(_run_burst())
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, (
        f"{len(errors)}/{N_CONCURRENT} sessions raised exceptions; "
        f"first 3: {errors[:3]}"
    )
    assert all(r["steps"] >= 1 for r in results), (
        f"some sessions did not advance any steps: {results}"
    )
    assert len({r["seed"] for r in results}) == N_CONCURRENT, (
        "expected unique seeds across all sessions"
    )
