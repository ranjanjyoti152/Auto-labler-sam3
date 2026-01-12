# SAM3 Auto-Labeler Docker Image
# GPU-enabled container for automatic object detection and YOLO dataset generation

FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

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

# Install PyTorch with CUDA support and other dependencies
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126 && \
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
