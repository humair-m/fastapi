# Kokoro TTS API Requirements

## Overview

The Kokoro TTS API, defined in `main.py`, is a FastAPI-based text-to-speech (TTS) service that converts text into audio using the Kokoro model. It supports 28 voices, multiple audio formats (WAV, MP3, OGG), real-time streaming (`/tts/stream`), batch processing, custom pronunciations, and Redis-based rate-limiting. The API relies on dependencies like `fastapi`, `torch`, `redis`, and `ffmpeg` for core functionality, as well as libraries for audio processing, metrics, and text preprocessing.

This document provides a `requirements.txt` file listing all Python dependencies required to run `main.py`, based on the user’s queries about endpoint implementations (e.g., `/tts/stream` in JavaScript, Python, Ruby, Go, PHP), debugging (e.g., Pydantic validation, RNN dropout), and feature history. Each dependency includes a compatible version and a brief description of its role. The document also includes setup instructions and notes on non-Python dependencies (e.g., `ffmpeg`, Redis).

## `requirements.txt`

Below is the `requirements.txt` file with dependencies and versions tested for compatibility with Python 3.12, as used in `main.py`.

```text
fastapi==0.115.0
uvicorn==0.30.6
torch==2.4.1
scipy==1.14.1
cachetools==5.5.0
fastapi-limiter==0.1.6
redis==5.0.8
prometheus-client==0.20.0
nest-asyncio==1.6.0
spacy==3.7.6
python-dotenv==1.0.1
```

### Dependency Descriptions

- **fastapi==0.115.0**:

  - Role: Core framework for building the API, handling routing, and Pydantic validation (e.g., `TTSRequest`, `TTSResponse` models).
  - Used in: All endpoints (`/tts`, `/tts/stream`, `/voices`, etc.).
  - Context: User’s queries relied on FastAPI’s streaming response for `/tts/stream` and Pydantic fixes for `/tts`.

- **uvicorn==0.30.6**:

  - Role: ASGI server to run the FastAPI application.
  - Used in: Starting the server (`python main.py` or `uvicorn main:app`).
  - Context: Required for hosting the API on `http://localhost:8000`.

- **torch==2.4.1**:

  - Role: PyTorch for loading and running the Kokoro TTS model, supporting CUDA for GPU acceleration.
  - Used in: Audio synthesis in `/tts` and `/tts/stream`.
  - Context: User’s debugging queries highlighted CUDA issues and RNN dropout warnings.

- **scipy==1.14.1**:

  - Role: Audio processing (e.g., WAV file handling, sample rate conversion).
  - Used in: Generating and formatting audio output.
  - Context: Essential for `/tts/stream` to produce WAV/MP3/OGG files.

- **cachetools==5.5.0**:

  - Role: Caching audio responses to improve performance.
  - Used in: Caching generated audio for repeated requests.
  - Context: Suggested in debugging guide for performance optimization.

- **fastapi-limiter==0.1.6**:

  - Role: Rate-limiting middleware using Redis (10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`).
  - Used in: Enforcing rate limits on endpoints.
  - Context: User’s streaming examples included retry logic for 429 errors.

- **redis==5.0.8**:

  - Role: Python client for Redis, used for rate-limiting and job queuing.
  - Used in: `FastAPILimiter` and job status tracking.
  - Context: Debugging guide addressed Redis connection errors.

- **prometheus-client==0.20.0**:

  - Role: Exposing metrics for monitoring (e.g., request counts, latency).
  - Used in: `/metrics` endpoint.
  - Context: Added in feature history for observability.

- **nest-asyncio==1.6.0**:

  - Role: Allows nested asyncio event loops for testing in Jupyter or scripts.
  - Used in: Development and debugging.
  - Context: Useful for running `main.py` in non-standard environments.

- **spacy==3.7.6**:

  - Role: Text preprocessing (e.g., expanding abbreviations, cleaning text).
  - Used in: `/preprocess` endpoint and pronunciation handling.
  - Context: Supports custom pronunciations from user’s queries.

- **python-dotenv==1.0.1**:

  - Role: Loads environment variables (e.g., `ENABLE_RATE_LIMITING`, `PORT`).
  - Used in: Configuring `main.py` settings.
  - Context: Simplifies debugging rate-limiting and port issues.

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```bash
pip list
```

### 3. Install Non-Python Dependencies

- **ffmpeg**:

  - Role: Converts audio formats (WAV, MP3, OGG).

  - Install:

    ```bash
    sudo apt install ffmpeg  # Ubuntu/Debian
    brew install ffmpeg  # macOS
    choco install ffmpeg  # Windows (with Chocolatey)
    ```

  - Verify:

    ```bash
    ffmpeg -version
    ```

- **Redis**:

  - Role: Rate-limiting and job queuing.

  - Install:

    ```bash
    sudo apt install redis-server  # Ubuntu/Debian
    brew install redis  # macOS
    ```

    Or Docker:

    ```bash
    docker run -d -p 6379:6379 redis
    ```

  - Start:

    ```bash
    redis-server
    ```

  - Verify:

    ```bash
    redis-cli ping  # Expected: PONG
    ```

- **spaCy Model**:

  - Role: English language model for text preprocessing.

  - Install:

    ```bash
    python -m spacy download en_core_web_sm
    ```

### 4. Configure `main.py`

Update file paths in `main.py`:

```python
MODEL_PATH = Path('/correct/path/to/kokoro-v1_0.pth')
CONFIG_PATH = Path('/correct/path/to/config.json')
```

Verify:

```bash
ls -l /correct/path/to/kokoro-v1_0.pth
ls -l /correct/path/to/config.json
python -m json.tool /correct/path/to/config.json
```

Example `config.json`:

```json
{
  "rnn": {
    "dropout": 0,
    "num_layers": 1
  },
  "sample_rate": 24000,
  "max_char_limit": 5000
}
```

### 5. Run the API

```bash
python main.py
```

Or with custom port:

```bash
export PORT=8080
python main.py
```

Verify:

```bash
curl -X GET "http://localhost:8000/health"
```

Expected:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "cuda_available": true,
  "ffmpeg_available": true,
  ...
}
```

## Notes from User Queries

- **Streaming (**`/tts/stream`**)**:

  - User requested examples in cURL, JavaScript (`fetch`, `request`), Python (`requests`), Ruby (`net/http`), Go (`net/http`), and PHP (`curl`).
  - Dependencies like `scipy` and `ffmpeg` are critical for audio streaming.
  - `fastapi-limiter` and `redis` handle 5/min rate limits, addressed with retry logic in examples.

- **Pydantic Fix**:

  - User noted validation errors in `/tts`, fixed by setting `audio_url`, `duration`, `tokens` to `None` in `TTSResponse`.
  - `fastapi` and `pydantic` (bundled with `fastapi`) are essential.

- **Debugging**:

  - User’s debugging queries highlighted RNN dropout warnings (fixed via `config.json`), Redis issues, and CUDA errors.
  - `torch`, `redis`, and `ffmpeg` are frequent sources of errors, addressed in setup.

- **Feature History**:

  - Dependencies evolved with features (e.g., `prometheus-client` for `/metrics`, `spacy` for `/preprocess`).
  - `python-dotenv` simplifies environment configuration (e.g., disabling rate-limiting).

## Additional Considerations

- **Version Compatibility**:

  - Versions in `requirements.txt` are tested with Python 3.12 and `main.py` as of April 2025.
  - For CUDA support, ensure `torch` matches your GPU drivers (`pip install torch --index-url https://download.pytorch.org/whl/cu118` for CUDA 11.8).

- **Non-Python Dependencies**:

  - `ffmpeg` is required for MP3/OGG conversion.
  - Redis is optional if `ENABLE_RATE_LIMITING="false"`.

- **Environment Variables**:

  - Set `ENABLE_RATE_LIMITING="false"` to bypass Redis for testing:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

  - Set `PORT` for custom ports:

    ```bash
    export PORT=8080
    ```

- **Limitations**:

  - `weight_norm` deprecation in `torch`: Upgrade `kokoro` (`pip install --upgrade kokoro`).
  - RNN dropout warning: Ensure `config.json` has `"dropout": 0`.

## Troubleshooting

- **Missing Dependencies**:

  - Check:

    ```bash
    pip show fastapi uvicorn torch scipy
    ```

  - Reinstall:

    ```bash
    pip install -r requirements.txt
    ```

- **Redis Errors**:

  - Verify Redis:

    ```bash
    redis-cli ping
    ```

  - Disable rate-limiting:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **ffmpeg Errors**:

  - Test `/tts/stream`:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
    ```

  - Reinstall `ffmpeg`:

    ```bash
    sudo apt install ffmpeg
    ```

- **CUDA Errors**:

  - Check:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

  - Fallback to CPU in `main.py`:

    ```python
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    ```

For support, share error logs, `pip list`, and `config.json`.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025