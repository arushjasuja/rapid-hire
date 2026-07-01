# Streamlit app image. Heads up: sentence-transformers pulls in PyTorch, so this
# image is large (well over a gigabyte). That's inherent to running the embedding
# model in-process; a slimmer build would offload embeddings to a separate service.
FROM python:3.12-slim

# uv for fast, reproducible installs. Pinned so image builds don't drift.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first, in their own layer, so app-code edits don't bust the
# (expensive) dependency cache. If a uv.lock is present it's honored; otherwise uv
# resolves at build time.
COPY pyproject.toml README.md ./
COPY uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Now the source and the rest of the project.
COPY src ./src
COPY data ./data
COPY .streamlit ./.streamlit
RUN uv sync --no-dev

EXPOSE 8501

# Streamlit's own health endpoint; lets an orchestrator know when we're ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["uv", "run", "--no-dev", "streamlit", "run", "src/rapidhire/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
