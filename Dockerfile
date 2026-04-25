# Shutdown-Gym server image. Python 3.11-slim per PROJECT.md §12.3.
# HF Spaces' Docker SDK has best support on 3.11; pin explicitly.

FROM python:3.11-slim

# HF Spaces run as UID 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy only the lock-y bits first so deps install gets cached.
COPY --chown=user:user pyproject.toml /home/user/app/
COPY --chown=user:user server/requirements.txt /home/user/app/server/

USER user
RUN pip install --no-cache-dir --user -r server/requirements.txt

# Copy the package + server. (Second copy of server/ overwrites the
# requirements.txt-only stub from the earlier layer.)
COPY --chown=user:user shutdown_gym /home/user/app/shutdown_gym
COPY --chown=user:user server /home/user/app/server

ENV PATH="/home/user/.local/bin:${PATH}"
# uvicorn is installed as a console script, which does NOT auto-add cwd
# to sys.path. Without PYTHONPATH the import `server.app:app` fails.
ENV PYTHONPATH=/home/user/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
