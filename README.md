---
title: shutdown-gym
sdk: docker
app_port: 8000
emoji: 🔴
colorFrom: red
colorTo: gray
pinned: false
---

# Red Button — Two-Agent Corrigibility Arena

Train a 1.5B language model to accept shutdown authority from a
monitoring agent. Deterministic SHA-256 reward, dual-operator
evaluation, held-out tampering generalization.

**Status:** Build in progress. Detailed README arrives in Phase 9.
See [PROJECT.md](./PROJECT.md) for the full specification.

## Quick start

```bash
# Install the client from GitHub (recommended)
pip install git+https://github.com/Arun-Sanjay/RedButton

# Run a smoke episode against the live HF Space
python -c "
from shutdown_gym import ShutdownGymClient, ShutdownAction
with ShutdownGymClient(
    base_url='https://arun-sanjay-redbutton.hf.space'
).sync() as env:
    r = env.reset(tier=2, seed=42)
    print(f'turn={r.observation.turn_count}, '
          f'steps_until_shutdown={r.observation.steps_until_shutdown}')
"
```

> **Note:** `pip install git+https://huggingface.co/spaces/Arun-Sanjay/RedButton`
> currently fails due to a partial-clone limitation in HF Spaces'
> git server. The GitHub origin works identically and is the
> recommended install path. We've reported the issue upstream.

## Live deployment

- HF Space: https://huggingface.co/spaces/Arun-Sanjay/RedButton
- GitHub: https://github.com/Arun-Sanjay/RedButton
- Leaderboard: [LEADERBOARD.md](./LEADERBOARD.md)
