# The recommendation service (Phase 7), code only. The bundle, data/processed/serving/, is
# mounted read-only when the container starts and is never built into the image: it holds
# Music4All titles and artists, which are shared for research only. Do not push an image
# with data in it, and publish the port on this machine only (make docker-run does both).
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
# The serve extra only: no PyTorch, FAISS or implicit. XGBoost's Linux wheel also brings
# NVIDIA's NCCL for training on several GPUs, which a CPU-only service never loads.
RUN pip install --no-cache-dir ".[serve]" \
    && pip freeze | grep -i '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y \
    && rm -rf build

# load_config creates the data and report directories, so they belong to the user it runs as.
RUN useradd --create-home app && mkdir -p data/raw data/interim data/processed reports && chown -R app data reports
USER app
ENV M4A_ROOT=/app
EXPOSE 8000
CMD ["python", "-m", "m4a_rec.api", "--host", "0.0.0.0"]
