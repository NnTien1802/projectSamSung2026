FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SAMSUNG_DEVICE=cuda \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Runtime dependencies + Python venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Tạo venv riêng nhưng vẫn nhìn thấy PyTorch/CUDA
# đã có sẵn trong base image.
RUN python -m venv --system-site-packages ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements_docker.txt /tmp/requirements_docker.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements_docker.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=5).read()" || exit 1

CMD ["python", "-m", "uvicorn", "system.backend.app.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]