# Slim runtime image shared by every service (ingestor / signal / analytics / gateway).
# No PyTorch here: inference runs on ONNX Runtime -> ~5x smaller image, faster cold start.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    QF_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvloop

COPY qforecast ./qforecast
RUN useradd --uid 10001 --no-create-home qf && mkdir -p /tmp/numba-cache && chown qf /tmp/numba-cache
USER qf

# Pre-compile the numba kernels at build time so the first live bar does not pay JIT cost.
RUN python -c "from qforecast.core.features import OnlineFeatureEngine; OnlineFeatureEngine().warmup_jit()"

CMD ["python", "-m", "qforecast.services.gateway"]
