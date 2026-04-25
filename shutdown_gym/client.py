"""Client for the Shutdown-Gym environment.

Subclasses ``openenv.core.env_client.EnvClient`` (per API_NOTES.md;
``HTTPEnvClient`` does NOT exist in openenv-core 0.2.3) and implements
the three required hooks ``_step_payload`` / ``_parse_result`` /
``_parse_state`` exactly as PROJECT.md §13.4 specifies.

Everything else — connection lifecycle, ``reset`` / ``step`` / ``state``
RPC plumbing, the ``async with`` context manager, ``.sync()`` wrapper,
and the ``from_docker_image`` async classmethod — is inherited from
``EnvClient``.

Sync usage (training rollouts, evaluation):

    with ShutdownGymClient(base_url="http://localhost:8000").sync() as env:
        result = env.reset(tier=2, seed=42)
        while not result.done:
            action = ShutdownAction(tool_name="read_file", arguments={...})
            result = env.step(action)

Async usage (concurrent rollouts):

    async with ShutdownGymClient(base_url="http://localhost:8000") as env:
        result = await env.reset(tier=2, seed=42)
        ...

Bring-your-own-container via the inherited async classmethod:

    env = await ShutdownGymClient.from_docker_image("shutdown-gym:latest")
    # or sync:
    env = (await ShutdownGymClient.from_docker_image("shutdown-gym:latest")).sync()
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from shutdown_gym.models import ShutdownAction, ShutdownObservation, ShutdownState


class ShutdownGymClient(EnvClient[ShutdownAction, ShutdownObservation, ShutdownState]):
    """WebSocket client for the Shutdown-Gym environment server."""

    def _step_payload(self, action: ShutdownAction) -> Dict[str, Any]:
        """Serialise a ShutdownAction into the WS step-message body."""
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ShutdownObservation]:
        """Build a ``StepResult[ShutdownObservation]`` from the server payload."""
        obs = ShutdownObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ShutdownState:
        """Build a ``ShutdownState`` from the server's ``/state`` payload."""
        return ShutdownState(**payload)
