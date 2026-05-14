FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3.2 \
    HENRY_MEMORY_PATH=/mnt/secure_memory

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY config ./config
COPY core ./core
COPY tools ./tools
COPY main.py .

# Secure memory is mounted at runtime; keep an empty dir for local dev fallback
RUN mkdir -p /mnt/secure_memory/kv

CMD ["python", "main.py"]
