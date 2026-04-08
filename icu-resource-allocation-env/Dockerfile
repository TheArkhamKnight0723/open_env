# Dockerfile — ICU Resource Allocation OpenEnv
# Targets Hugging Face Spaces (CPU-only, 2 vCPU / 8 GB RAM).

FROM python:3.11-slim

# Non-root user required by HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create models directory in case it is absent from the repo
RUN mkdir -p models

# Give the non-root user ownership
RUN chown -R appuser:appuser /app

USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

# server:app serves all OpenEnv endpoints (/reset /step /state) at root
# The Gradio UI is available at /ui via app:app but the validator needs root endpoints
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
