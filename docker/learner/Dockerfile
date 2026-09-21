FROM python:3.11-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.8.22
WORKDIR /opt/expo-ft
COPY docker/learner/requirements.in /tmp/requirements.in
RUN uv pip install --system --no-cache torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r /tmp/requirements.in && uv pip check --system
COPY . /opt/expo-ft
ENV PYTHONPATH=/opt/expo-ft:/opt/expo-ft/expo_ft/agents/vla/openpi/src:/opt/expo-ft/expo_ft/agents/vla/openpi/packages/openpi-client/src
ENV PYTHONUNBUFFERED=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
RUN JAX_PLATFORMS=cpu python -c "import train_pi_robo; from expo_ft.agents.alg.expo_ft import EXPOLearner" && \
    uv pip freeze --system > /opt/expo-ft/docker/learner/requirements-installed.txt
CMD ["python", "tests/gpu/learner_smoke.py", "--help"]
