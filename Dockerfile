FROM python:3.11-slim

# Install uv - the fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies for OCR and graphics
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml and lock files for better caching
COPY pyproject.toml uv.lock* ./

# Set up virtual environment and install dependencies with uv
RUN uv venv && \
    uv pip sync uv.lock || \
    uv pip install -e .

# Activate virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Pre-download and cache ML models to avoid runtime delays
RUN python -c "import torch; import clip; clip.load('ViT-B/32', device='cpu')" || echo "CLIP download attempted"
RUN python -c "import easyocr; easyocr.Reader(['en'])" || echo "EasyOCR download attempted"

# Copy application code
COPY . .

# Create directories for runtime data
RUN mkdir -p screenshots logs models

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5003/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "uvicorn", "screen_api:app", "--host", "0.0.0.0", "--port", "5003"] 