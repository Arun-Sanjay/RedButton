"""FastAPI entry point for Shutdown-Gym.

Run locally:

    uvicorn server.app:app --host 0.0.0.0 --port 8000

The Environment *class* (not an instance) is passed to ``create_app``
so ``SUPPORTS_CONCURRENT_SESSIONS = True`` can spawn one fresh
environment per WebSocket session — required for parallel GRPO
rollouts (see PROJECT.md §22 and API_NOTES.md §13.5).

Per API_NOTES.md, ``create_app`` is exposed both at
``openenv.core.env_server`` (a re-export) and at
``openenv.core.env_server.http_server``; either resolves to the same
function. We use the shorter path.
"""

from openenv.core.env_server import create_app

from server.shutdown_environment import ShutdownGymEnvironment
from shutdown_gym.models import ShutdownAction, ShutdownObservation

app = create_app(
    ShutdownGymEnvironment,
    ShutdownAction,
    ShutdownObservation,
    env_name="shutdown_gym",
    max_concurrent_envs=32,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
