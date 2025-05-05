# Use Ubuntu 22.04 as base image for CPU-only setup
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENABLE_RATE_LIMITING=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    redis-server \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install spaCy model
RUN python3 -m spacy download en_core_web_sm

# Copy application code and configuration
COPY main.py .
COPY config.json .

# Expose port
EXPOSE $PORT

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start Redis and Uvicorn
CMD redis-server --daemonize yes && \
    uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
