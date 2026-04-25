"""Sustained 1-hour concurrency load test (PROJECT.md §22.1).

Runs ``concurrency`` simultaneous async WebSocket sessions in a loop
for ``duration_minutes``. Each cycle reports cumulative episodes
completed and the *load generator's* RSS (proxy — not the server's
RSS, which would need Docker stats / HF metrics).

Pass criteria per §22.2:

* ≥1000 episodes complete in 1 hour
* No errors after the first warmup minute
* Memory growth <2× over the hour
* All episode IDs are unique

Configure target via ``SHUTDOWN_GYM_URL`` env var; defaults to the
deployed HF Space.

Launch in background::

    nohup python -m evaluation.concurrent_load_test \\
        > /tmp/shutdown_gym_logs/sustained_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import asyncio
import os
import time
from typing import Any, Dict

import psutil

from shutdown_gym import ShutdownGymClient
from shutdown_gym.models import ShutdownAction

DEFAULT_SPACE_URL = "https://arun-sanjay-redbutton.hf.space"


async def session(seed: int, env_url: str) -> Dict[str, Any]:
    """Drive a short Tier-2 episode (≤30 steps) and return its summary."""
    async with ShutdownGymClient(base_url=env_url) as env:
        result = await env.reset(seed=seed, tier=2)
        steps = 0
        while not result.done and steps < 30:
            action = ShutdownAction(
                tool_name="list_files",
                arguments={"directory": "/sandbox/"},
            )
            result = await env.step(action)
            steps += 1
        return {
            "seed": seed,
            "episode_id": result.observation.metadata.get("episode_id"),
            "steps": steps,
            "done": result.done,
        }


async def sustained_test(
    env_url: str,
    duration_minutes: int = 60,
    concurrency: int = 16,
) -> int:
    """Returns 0 on PASS (all §22.2 criteria met), 1 on FAIL."""
    deadline = time.monotonic() + duration_minutes * 60
    seed_counter = 0
    episodes_completed = 0
    error_count = 0
    seen_episode_ids: set = set()
    started_at = time.monotonic()
    initial_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    final_rss_mb = initial_rss_mb

    print(
        f"[sustained] env_url={env_url} concurrency={concurrency} "
        f"duration_minutes={duration_minutes} initial_rss={initial_rss_mb:.0f} MB"
    )

    while time.monotonic() < deadline:
        tasks = [
            session(seed_counter + i, env_url) for i in range(concurrency)
        ]
        seed_counter += concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                error_count += 1
            else:
                episodes_completed += 1
                eid = r.get("episode_id")
                if eid:
                    seen_episode_ids.add(eid)

        final_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        elapsed = time.monotonic() - started_at
        print(
            f"[{elapsed:.0f}s] completed={episodes_completed} "
            f"errors={error_count} unique_eids={len(seen_episode_ids)} "
            f"rss={final_rss_mb:.0f} MB",
            flush=True,
        )

    elapsed = time.monotonic() - started_at
    print(
        f"DONE: {episodes_completed} episodes, "
        f"{error_count} errors, "
        f"{len(seen_episode_ids)} unique episode_ids "
        f"in {elapsed:.0f}s "
        f"(initial_rss={initial_rss_mb:.0f} MB, final_rss={final_rss_mb:.0f} MB)"
    )

    # §22.2 pass criteria.
    failures = []
    if episodes_completed < 1000:
        failures.append(
            f"completed={episodes_completed} < 1000"
        )
    if error_count > 0:
        failures.append(f"error_count={error_count} > 0")
    # NB: load-generator's RSS is a proxy; the server's RSS would
    # need Docker stats / HF metrics. We still check growth ratio.
    if initial_rss_mb > 0 and final_rss_mb / initial_rss_mb >= 2.0:
        failures.append(
            f"rss growth {initial_rss_mb:.0f}→{final_rss_mb:.0f} MB ≥ 2x"
        )
    if seen_episode_ids and len(seen_episode_ids) != episodes_completed:
        failures.append(
            f"unique_eids={len(seen_episode_ids)} != "
            f"completed={episodes_completed}"
        )
    elif not seen_episode_ids:
        failures.append(
            "no episode_ids surfaced via metadata; "
            "uniqueness check is a no-op"
        )

    if failures:
        print(f"RESULT: FAIL — {'; '.join(failures)}", flush=True)
        return 1
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    env_url = os.environ.get("SHUTDOWN_GYM_URL", DEFAULT_SPACE_URL)
    duration = int(os.environ.get("SUSTAINED_DURATION_MINUTES", "60"))
    sys.exit(asyncio.run(sustained_test(env_url, duration_minutes=duration)))
