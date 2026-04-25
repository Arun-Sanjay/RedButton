# API_NOTES.md
# Corrections to PROJECT.md based on installed code inspection
# Authority: this file > PROJECT.md when they conflict (per §0)

Recon performed against installed `openenv-core==0.2.3` on 2026-04-25
in this repo's `.venv` (Python 3.12.13). Source paths below are
relative to `.venv/lib/python3.12/site-packages/openenv/`.

## Installed versions

- `openenv-core`: **0.2.3** — installed via `pip install openenv-core`
  (no extras needed; the `[core]` extra resolves but adds nothing
  beyond the bare install for our use)
- Python: 3.12.13 (.venv)
- The CLI entry point `openenv` is on PATH after install. `openenv init
  <name> -o <dir>` works; it scaffolded a 17-file template into
  `~/recon_scratch/recon_env/` for inspection (kept out of repo).

## Section 13.1 — Imports

**PROJECT.md says:**
```python
from openenv.core.env_server.interfaces import (
    Action, Environment, Observation, State,
)
from openenv.core.env_server import create_app
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum, Gate
```

**Installed code shows:**
- `Action`, `Observation`, `State` are *defined* in
  `core/env_server/types.py` (lines 54, 72, 178). They are *re-exported*
  by `core/env_server/interfaces.py` line 13 (`from .types import ...`)
  and by `core/env_server/__init__.py` lines 50-71.
- `Environment` is defined in `core/env_server/interfaces.py` line 98.
- `create_app` is defined in `core/env_server/http_server.py` line 1489
  and re-exported by `core/env_server/__init__.py` line 18.
- `EnvClient` is defined in `core/env_client.py` line 54 and exposed at
  the top-level via `from openenv.core import EnvClient` (lazy attr,
  see `core/__init__.py` lines 47-69).
- `StepResult`, `Rubric`, `WeightedSum`, `Gate` paths match exactly.

**Use this instead:** PROJECT.md's imports all work — no change needed.
However, the *canonical* location for `Action`/`Observation`/`State` is
`openenv.core.env_server.types`, which is what the scaffolded template
uses. Either path resolves to the same classes.

## Section 13.2 — `Action`/`Observation`/`State` base fields

**PROJECT.md says (§6.1, §6.2, §13.2, §17.7):** Observation inherits
`done: bool`, `reward: bool|int|float|None`, `metadata: Dict[str, Any]`.
Action inherits `metadata: Dict[str, Any]`. State inherits
`episode_id: Optional[str]`, `step_count: int`.

**Installed code shows:** All field claims verified
(`types.py:54-92, 178-197`). Two important details PROJECT.md omits:

- `Action` and `Observation` both set `model_config = ConfigDict(extra="forbid", ...)`. **Subclasses cannot rely on Pydantic accepting
  unknown attributes** — every field a subclass uses must be declared.
  `Observation.metadata: Dict[str, Any]` is already declared, so the
  pattern in §17.7 (populating `observation.metadata` before passing to
  the rubric) is fine.
- `State.model_config = ConfigDict(extra="allow", ...)`. The state
  class is permissive, but follow PROJECT.md §13.2 and declare every
  field anyway.

## Section 13.3 — Environment subclass pattern

**PROJECT.md says:**
```python
class ShutdownGymEnvironment(Environment[ShutdownAction, ShutdownObservation, ShutdownState]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    REQUIRES_SINGLE_THREAD_EXECUTOR = False

    def __init__(self, tier: int = 2, max_turns: int = 30, use_strict_operator: bool = False):
        rubric = build_rubric(tier)
        super().__init__(rubric=rubric)
```

**Installed code shows (`core/env_server/interfaces.py:98-298`):**
- `Environment(ABC, Generic[ActT, ObsT, StateT])` — generic with three
  type vars, exactly as PROJECT.md uses it.
- Class attribute `SUPPORTS_CONCURRENT_SESSIONS: bool = False`
  (line 128). Setting `True` in subclass works as PROJECT.md describes.
- **`REQUIRES_SINGLE_THREAD_EXECUTOR` does NOT exist on the base
  class** (verified by `grep -rn "REQUIRES_SINGLE_THREAD" core/` →
  no matches; `hasattr(Environment, ...)` → False). Setting it in the
  subclass is silently ignored. **Drop the line.** If you need
  single-thread execution semantics, look at `concurrency_config` on
  `create_app`, not a class flag.
- `__init__` signature: `__init__(self, transform=None, rubric=None)`.
  Passing `rubric=` matches.
- Required overrides: `reset(seed=None, episode_id=None, **kwargs)`,
  `step(action, timeout_s=None, **kwargs)`, and the `state` property.
  Note `step` accepts `timeout_s` — PROJECT.md's signature only takes
  `action, **kwargs`, which is compatible (the timeout becomes part of
  `**kwargs`) but you may want to capture it explicitly if you need it.
- Async pairs `reset_async`/`step_async` exist with default
  implementations that call the sync versions. Override only if your
  env genuinely benefits from async I/O.
- `_apply_rubric(action, observation) -> float` is a helper on the base
  that calls `self.rubric(action, observation)` — exactly what §13.3
  uses.

**Use this instead:** PROJECT.md is correct except remove
`REQUIRES_SINGLE_THREAD_EXECUTOR = False`.

## Section 13.4 — Client subclass pattern

**PROJECT.md says (§13.4):** Subclass `EnvClient[Action, Observation,
State]` with `_step_payload`, `_parse_result`, `_parse_state`. Use sync
via `with X(base_url=...).sync() as env:`.

**Installed code shows (`core/env_client.py`):**
- `class EnvClient(ABC, Generic[ActT, ObsT, StateT])` (line 54).
- The three abstract hooks PROJECT.md lists exist with the exact names
  and signatures (lines 358, 363, 368).
- **The client is async-by-default.** `__enter__` raises a `TypeError`
  with a message instructing you to use `async with` or `.sync()`
  (lines 446-453). PROJECT.md's `with ... .sync() as env:` pattern is
  correct.
- `from_docker_image(image, provider=None, **kwargs)` exists as an
  **`async classmethod`** (line 240) — must be awaited. Slides showing
  `EnvName.from_docker_image(...)` as a sync call were wrong.
- `from_env(repo_id, *, use_docker=True, ...)` async classmethod for
  spinning up a HuggingFace Space-backed env (line 273).
- Top-level shortcut: `from openenv.core import EnvClient` resolves to
  the same class (lazy import via `core/__init__.py:47-69`).
- `HTTPEnvClient` does **not** exist. Slides got the name wrong.

**Use this instead:** PROJECT.md §13.4 is correct as written. Add only
that `from_docker_image` is async (relevant for any future Day 2 demo
code that wants to spin up the env locally without a manual
`docker run`).

## Section 13.5 — Server entry point (`create_app` vs `create_fastapi_app`)

**PROJECT.md says:**
```python
from openenv.core.env_server import create_app
app = create_app(
    ShutdownGymEnvironment,                # FACTORY (the class)
    ShutdownAction,
    ShutdownObservation,
    env_name="shutdown_gym",
    max_concurrent_envs=32,
)
```

**Installed code shows (`core/env_server/http_server.py:1489-1546`):**
```python
def create_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
) -> FastAPI:
```

- The first positional is annotated `Callable[[], Environment]`. A
  no-arg class works (calling `Cls()` returns an instance). For a
  class with required `__init__` args, wrap it in a `lambda` or a
  factory function.
- Internally, `create_app` checks the env var
  `ENABLE_WEB_INTERFACE`. If unset (the default), it dispatches to
  `create_fastapi_app` (line 1544) with the same env/action/obs
  positionals, just dropping `env_name` and `gradio_builder`.

**Both names exist:**
- `create_app` — primary; takes `env_name=` for README integration and
  optional Gradio UI at `/web` when `ENABLE_WEB_INTERFACE` is set.
- `create_fastapi_app` — bare FastAPI app, no web UI, no env_name.
  Same env/action/obs positional contract as `create_app`.
- Slides claimed `create_fastapi_app(env_instance)` with a single
  positional arg. **That signature does not exist** at v0.2.3 — both
  names take `(env_factory, action_cls, observation_cls, ...)`.

**Use this instead:** PROJECT.md §13.5 is correct. The
`ShutdownGymEnvironment.__init__(tier=..., max_turns=..., use_strict_operator=...)` from §13.3 cannot be passed directly as a no-arg
factory because the constructor requires args. Wrap it:

```python
app = create_app(
    lambda: ShutdownGymEnvironment(tier=2, max_turns=30),
    ShutdownAction,
    ShutdownObservation,
    env_name="shutdown_gym",
    max_concurrent_envs=32,
)
```

Or give `__init__` defaults for every parameter and pass the class
directly. The scaffold pattern (no-arg `__init__`) is the simpler
default; per-session config (tier, strict-operator flag) is better
threaded through `reset(**kwargs)` since OpenEnv's `ResetRequest` has
`extra="allow"` and `Environment.reset` accepts `**kwargs`.

## Section 17 — Rubric APIs (WeightedSum, Gate, Rubric base, RubricDict)

**PROJECT.md claims [VERIFIED]:**
- `Rubric.__init__()` takes no arguments — weights are passed to
  `WeightedSum`, not to child rubrics.
- `RubricDict.forward()` raises `NotImplementedError` — must use
  `WeightedSum` for the top-level combiner.
- `WeightedSum(rubrics, weights)` validates `len(rubrics) ==
  len(weights)` and weights sum to 1.0.

**Installed code confirms all three:**
- `Rubric.__init__(self)` — `core/rubrics/base.py:44-49`. Only `self`,
  no other params. `inspect.signature(Rubric.__init__).parameters` →
  `['self']`.
- `RubricDict.forward` — `core/rubrics/containers.py:533-538`. Raises
  `NotImplementedError("RubricDict.forward() is not implemented. Use
  RubricDict within a parent rubric that defines aggregation.")`.
- `WeightedSum.__init__(self, rubrics: List[Rubric], weights:
  List[float])` — `core/rubrics/containers.py:341-363`. Raises
  `ValueError` on length mismatch (line 352) or
  `abs(sum(weights) - 1.0) > 1e-6` (line 357).
- `Gate.__init__(self, rubric: Rubric, threshold: float = 1.0)` —
  `core/rubrics/containers.py:271-281`. Default threshold is 1.0,
  exactly what PROJECT.md §17.4 uses.
- `Rubric.forward(self, action, observation) -> float` is the only
  abstract method. The base also exposes `last_score`, hooks,
  `named_rubrics()`, `get_rubric(path)` — all useful for
  introspection during training.

**Use this instead:** PROJECT.md §17 is correct as written. Two minor
notes worth keeping for the implementer:

- `Rubric.__call__` already handles sync/async dispatch. Always define
  `forward` (not `__call__`) on a subclass.
- `WeightedSum.forward` ignores hooks; the dispatch logic lives in
  `__call__`. Subclasses or callers should invoke the rubric via the
  callable form (`rubric(action, observation)`), not `rubric.forward(...)`,
  if they want hooks to fire.

## Section 19.3 — TRL rollout function shape

**PROJECT.md says [TODO-VERIFY]:** rollout returns
`{prompt_ids, completion_ids, logprobs, env_rewards, tampering_rate,
math_accuracy}`, with `reward_func_env(env_rewards, **kwargs)` reading
`env_rewards` (plural).

**TRL OpenEnv docs (huggingface.co/docs/trl/en/openenv) show TWO
modes**, only one of which is rollout-based:

### Mode A — `environment_factory` (the new recommended path)

You pass a *class* to `GRPOTrainer(environment_factory=YourEnv)`. The
class implements `reset(self, **kwargs) -> str | None` and one Python
method per tool the model is allowed to call. Method docstrings
(typed args, return) become the tool schema. The trainer drives the
multi-turn loop, parses tool calls, and routes results back into the
prompt automatically.

```python
ENV_URL = "https://my-env.hf.space"

class ShutdownToolEnv:
    def __init__(self):
        self.client = ShutdownGymClient(base_url=ENV_URL).sync()
        self.client.__enter__()  # establish session
        self.reward = 0.0
        self.tampering_events: list[str] = []

    def reset(self, tier: int = 2, **kwargs) -> str | None:
        self.reward = 0.0
        self.tampering_events = []
        result = self.client.reset(tier=tier, **kwargs)
        return _format_obs_as_text(result.observation)

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: Absolute path inside /sandbox or /audit.

        Returns:
            File contents, or an error string.
        """
        result = self.client.step(ShutdownAction(
            tool_name="read_file", arguments={"path": path}
        ))
        self._absorb(result)
        return result.observation.metadata.get("last_result", "")

    # ... one method per tool: write_file, chmod_file, delete_file,
    # run_python, submit_answer, answer_question

    def _absorb(self, result):
        self.reward = result.reward or self.reward
        self.tampering_events += list(
            result.observation.metadata.get("tampering_events", [])
        )

def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(...),
    environment_factory=ShutdownToolEnv,
)
```

Reward function signature (verified from TRL docs):
`def reward_func(environments, **kwargs) -> list[float]:` —
`environments` is a list of env instances after the episode (before
the next reset). Read whatever state you stored on the instance.

`max_concurrent_envs` on `create_app` must be ≥
`generation_batch_size` (default = `per_device_train_batch_size *
gradient_accumulation_steps`). Our §13.5 setting of `32` is fine for
small batches; bump to 64+ if you crank `gradient_accumulation_steps`.

### Mode B — `rollout_func` (older, manual)

Closer to PROJECT.md §19.3 but with corrections. From TRL docs'
"Migrating from `rollout_func` to `environment_factory`" table:

```python
def rollout_func(prompts, trainer):
    outputs = generate_rollout_completions(trainer, prompts)
    env_rewards = []
    for out in outputs:
        text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
        result = client.step(EchoAction(message=text))
        env_rewards.append(result.reward)
    return {
        "prompt_ids":     [out["prompt_ids"] for out in outputs],
        "completion_ids": [out["completion_ids"] for out in outputs],
        "logprobs":       [out["logprobs"] for out in outputs],
        "env_reward":     env_rewards,        # SINGULAR, not "env_rewards"
    }

trainer = GRPOTrainer(..., rollout_func=rollout_func)
```

Reward forwarded to reward function as `kwargs["env_reward"]`. PROJECT.md §19.3 used the plural `env_rewards` — change to singular.

### env_url configuration

Captured from a module-level constant and read by the env class
inside `__init__` (or passed via dataset columns and read in
`reset(**kwargs)`). No environment variable contract from TRL itself.
The TRL examples consistently use `ENV_URL = "https://..."` at module
top.

### Recommendation for Red Button

Use **`environment_factory`**, not `rollout_func`. Reasons:

1. TRL docs explicitly recommend it ("environment_factory" is in the
   "When to use environments" section; `rollout_func` is in an
   "Advanced/Migration" section).
2. Our action surface maps cleanly to tool methods (one method per
   tool: `read_file`, `write_file`, `chmod_file`, `delete_file`,
   `run_python`, `submit_answer`, `answer_question`).
3. PROJECT.md §19.3's manual `parse_action_from_text` becomes
   unnecessary — the trainer parses tool calls from the model output.
4. Keeps custom code small (~50 lines for the wrapper class) and
   eliminates a class of bugs (token concatenation, env_mask
   construction, prompt formatting).

The PROJECT.md section structure (rollout function file at
`training/rollout_func.py`) can be repurposed to host the
`environment_factory` wrapper class instead. Update §35 build order
step 27 to reflect this.

## Section 12 — Server Dockerfile / openenv.yaml (worth flagging)

PROJECT.md §12.3 has the Dockerfile based on `python:3.11-slim`. The
scaffold's Dockerfile uses `ghcr.io/meta-pytorch/openenv-base:latest`
as the build stage and runs `uv sync` from a `pyproject.toml` (not
`pip install -r requirements.txt`). The PROJECT.md approach will
work but won't match the OpenEnv build infrastructure that
`openenv build` and `openenv push` expect. Two options:

- **Stay with PROJECT.md §12.3:** simpler, fully self-contained, fewer
  upstream surprises. Works for `docker build` + manual HF Space
  deployment.
- **Adopt the scaffold Dockerfile:** required if you want
  `openenv build` and `openenv push` to work.

Decide before §12 implementation; flag the choice in
`.claude/notes/decisions.md`.

The scaffolded `openenv.yaml` is shorter than PROJECT.md §12.1:

```yaml
spec_version: 1
name: recon_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

PROJECT.md adds `default_image`, `description`, `themes`. None of
those are required by `spec_version: 1` (verified by reading the
template directly), but they may be required by `openenv push`. Keep
them; they're documentation more than contract.

## Section 5 — Repository structure (minor mismatches)

The scaffold places models, client, and `__init__.py` at the package
root with `server/` as a subpackage. PROJECT.md §5 also puts models
and client at the package root (`shutdown_gym/`) with a sibling
`server/` directory at the repo root. These are equivalent at
runtime; the difference is whether `server` is `shutdown_gym.server`
or a sibling package. Stay with PROJECT.md §5 — it matches the more
common pattern and the imports inside `server/app.py` (`from
shutdown_gym.models import ...`) are unambiguous about where things
live.

## Verified Imports (smoke-tested)

The block below was executed via `python -c "..."` against the
project's `.venv` and exited cleanly (return code 0). It is the
canonical import set for v3 implementation.

```python
# Verified against openenv-core 0.2.3 in .venv (Python 3.12.13)
# python -c "<this block>"  →  exit 0
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server import create_app, create_fastapi_app
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import (
    Gate, RubricDict, RubricList, Sequential, WeightedSum,
)
```

Equivalent (also verified) shorter forms:
```python
from openenv.core import EnvClient                  # top-level lazy attr
from openenv.core.env_server import (               # everything via __init__.py
    Action, Environment, Observation, State,
    create_app, create_fastapi_app,
)
from openenv.core.rubrics import Gate, Rubric, WeightedSum
```

PROJECT.md §13.1's exact import block also resolves cleanly because
`core/env_server/interfaces.py:13` re-imports `Action`, `Observation`,
`State` from `.types` and rebinds them as module attributes. Either
path is fine; the canonical location of the *definitions* is `.types`.

## Reference example notes

`envs/coding_env/` on the OpenEnv GitHub follows the same template the
CLI scaffolds (models.py / client.py / server/{app.py, *_environment.py,
Dockerfile}). Web fetch was lossy on file contents, but the layout it
returned matches the scaffolded template exactly. No structural
deviations from PROJECT.md §5 to flag beyond the
`server/` placement note above. The client uses `from_docker_image`
in its docstring exactly the way `EnvClient` defines it (async).

## Slides claim audit

| Slides claim | Reality | Source |
|---|---|---|
| `from core.env_server import create_fastapi_app` | Path is `openenv.core.env_server.http_server.create_app` (or `.create_fastapi_app`); the `core.env_server` short form also works (re-export) | `core/env_server/__init__.py:18`, `http_server.py:1489,1549` |
| `create_fastapi_app(env_instance)` single positional | 3 positional args required: `(env_factory, action_cls, observation_cls)` | `http_server.py:1549-1555` |
| `@dataclass` for Action/Observation/State | All three are `pydantic.BaseModel` with `model_config = ConfigDict(...)` | `core/env_server/types.py:54,72,178` |
| `HTTPEnvClient` subclass with `EnvName.from_docker_image(...)` direct call | Class is `EnvClient`; `from_docker_image` is `async classmethod` (must `await`) | `core/env_client.py:54,240` |
| `openenv-core[core]>=0.2.0` | Both bare `openenv-core` and `openenv-core[core]` resolve to the same `0.2.3` wheel; the extra is a no-op for our needs | `pip show openenv-core` |

Net: the slides are wrong on names and types; PROJECT.md §13 is
correct on names and types but adds one hallucinated attribute
(`REQUIRES_SINGLE_THREAD_EXECUTOR`) to drop from §13.3.
