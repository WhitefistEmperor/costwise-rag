# Hugging Face Space: builds Costwise RAG from the public GitHub repo and serves the dashboard.
# Runs with the mock LLM (free, no key). To use a real model, add a Space secret, never a file.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    COSTWISE_LLM=mock COSTWISE_STATE_DIR=/home/user/app/.state HF_HOME=/home/user/.hf
WORKDIR /home/user/app

RUN python -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
RUN git clone --depth 1 https://github.com/WhitefistEmperor/costwise-rag.git . \
 && python -m pip install --user -e . \
 && mkdir -p .state \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["uvicorn", "costwise.app.api:app", "--host", "0.0.0.0", "--port", "7860"]
