# Costwise RAG API. Runs with the mock LLM by default (free, no key).
# No secret is baked in: OPENROUTER_API_KEY is read only from the runtime environment.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COSTWISE_LLM=mock \
    COSTWISE_STATE_DIR=/app/.state \
    HF_HOME=/app/.hf

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN python -m pip install --upgrade pip \
 && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
# Editable install keeps the package next to data/ (settings.py resolves paths from the repo root).
RUN python -m pip install -e . \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /app/.state /app/.hf \
 && chown -R app /app/.state /app/.hf

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/stats', timeout=4)"
CMD ["uvicorn", "costwise.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
