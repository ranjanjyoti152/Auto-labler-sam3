# SAM3 Auto-Labeler Docker Image
# GPU-enabled container for automatic object detection and YOLO dataset generation

ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA support and other dependencies.
# The default cu129/torch 2.8 wheels ship kernels for Blackwell (sm_120, e.g.
# RTX 50-series / 5090). Anything built only up to sm_90 (e.g. cu126) fails at
# model load with "CUDA error: no kernel image is available for execution on
# the device". Override via build args if you need a different CUDA/torch combo.
ARG PYTORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG PYTORCH_CUDA_INDEX=cu129
RUN pip install --no-cache-dir --break-system-packages \
    torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX} && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY tools/ ./tools/
COPY third_party/ ./third_party/

# Create weights directory
RUN mkdir -p /app/weights

# Expose the API port
EXPOSE 8000

# Health check
# Probe /readyz (not /healthz): it 503s until the SAM3 model is actually loaded,
# so a container that cannot run the model is correctly reported unhealthy.
# start-period is generous because first run downloads a ~2GB checkpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz')" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
