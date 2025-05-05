# Kokoro TTS API Debugging and Feature History

## Overview

The Kokoro TTS API, defined in `main.py`, is a FastAPI-based text-to-speech (TTS) service that converts text into high-quality audio using the Kokoro model. It supports 28 voices (American and British English, male and female), multiple audio formats (WAV, MP3, OGG), and features like real-time streaming, batch processing, custom pronunciations, and Redis-based rate-limiting. The API runs on `http://localhost:8000` by default, leverages CUDA for acceleration, and uses dependencies like `torch`, `scipy`, `fastapi-limiter`, and `ffmpeg`.

This document provides a **debugging guide** for diagnosing and resolving common issues in `main.py`, such as model loading errors, Redis connectivity, rate-limiting, and Pydantic validation issues. It also includes a **feature history** detailing the development and fixes in `main.py`, focusing on the user’s queries about endpoint implementations (e.g., `/tts/stream`) and specific fixes (e.g., Pydantic validation for `/tts`). The guide includes code snippets, setup instructions, and workarounds for limitations like RNN dropout warnings and `weight_norm` deprecation.

## Debugging Guide

Below is a step-by-step guide to debug common issues in `main.py`, covering server startup, endpoint functionality, and runtime errors. Each section includes symptoms, diagnostic steps, and solutions with code snippets.

### 1. Server Fails to Start

**Symptoms**:
- Error: `FileNotFoundError: [Errno 2] No such file or directory: '/path/to/kokoro-v1_0.pth'`
- Error: `KeyError: 'rnn'` or invalid JSON in `config.json`.
- Uvicorn fails with `ERROR: application startup failed`.

**Diagnostics**:
- Check `MODEL_PATH` and `CONFIG_PATH` in `main.py`:

  ```python
  MODEL_PATH = Path('/path/to/kokoro-v1_0.pth')
  CONFIG_PATH = Path('/path/to/config.json')
  ```

- Verify file existence:

  ```bash
  ls -l /path/to/kokoro-v1_0.pth
  ls -l /path/to/config.json
  ```

- Validate `config.json`:

  ```bash
  python -m json.tool /path/to/config.json
  ```

**Solutions**:
- **Fix File Paths**:
  Update `main.py` with correct paths:

  ```python
  MODEL_PATH = Path('/correct/path/to/kokoro-v1_0.pth')
  CONFIG_PATH = Path('/correct/path/to/config.json')
  ```

- **Fix `config.json`**:
  Ensure it contains required fields (e.g., `rnn` settings):

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

- **Check Dependencies**:
  Install missing packages:

  ```bash
  pip install fastapi uvicorn torch scipy cachetools fastapi-limiter redis prometheus-client nest_asyncio spacy
  python -m spacy download en_core_web_sm
  sudo apt install ffmpeg
  ```

- **Run with Debug Logs**:
  Start with verbose logging:

  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
  ```

### 2. Redis Connection Errors

**Symptoms**:
- Error: `redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.`
- Rate-limiting endpoints (`/tts`, `/tts/stream`, `/tts/batch`) fail with 500 errors.

**Diagnostics**:
- Check Redis server:

  ```bash
  redis-cli ping
  ```

  Expected: `PONG`

- Verify `main.py` Redis configuration:

  ```python
  redis = Redis(host='localhost', port=6379, db=0)
  ```

- Check if Redis is running:

  ```bash
  ps aux | grep redis
  ```

**Solutions**:
- **Start Redis**:

  ```bash
  redis-server
  ```

  Or Docker:

  ```bash
  docker run -d -p 6379:6379 redis
  ```

- **Update Redis Host**:
  If Redis is on a different host/port, update `main.py`:

  ```python
  redis = Redis(host='redis-server', port=6379, db=0)
  ```

- **Disable Rate-Limiting**:
  For testing, disable in `main.py`:

  ```python
  os.environ['ENABLE_RATE_LIMITING'] = 'false'
  ```

  Or via environment:

  ```bash
  export ENABLE_RATE_LIMITING="false"
  python main.py
  ```

### 3. Model Loading Errors

**Symptoms**:
- Error: `RuntimeError: Error loading model: ...`
- Error: `torch.cuda.OutOfMemoryError: CUDA out of memory`.

**Diagnostics**:
- Check CUDA availability:

  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

  Expected: `True`

- Verify model file integrity:

  ```bash
  file /path/to/kokoro-v1_0.pth
  ```

- Check GPU memory:

  ```bash
  nvidia-smi
  ```

**Solutions**:
- **Fix Model Path**:
  Ensure `MODEL_PATH` is correct (see Server Fails to Start).

- **Clear GPU Memory**:
  Kill processes using GPU:

  ```bash
  nvidia-smi | grep 'python' | awk '{print $2}' | xargs kill -9
  ```

- **Disable CUDA**:
  Modify `main.py` to fallback to CPU:

  ```python
  device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
  ```

- **Update `torch`**:
  Ensure compatible version:

  ```bash
  pip install torch --upgrade
  ```

### 4. RNN Dropout Warning

**Symptoms**:
- Warning: `UserWarning: RNN module has dropout=0.2 and num_layers=1, which is not supported.`

**Diagnostics**:
- Check `config.json` for `rnn` settings:

  ```bash
  grep rnn /path/to/config.json
  ```

**Solutions**:
- **Update `config.json`**:
  Set `dropout` to 0:

  ```json
  {
    "rnn": {
      "dropout": 0,
      "num_layers": 1
    }
  }
  ```

  Validate:

  ```bash
  python -m json.tool /path/to/config.json
  ```

- **Patch `main.py`**:
  Add fallback in model initialization (example):

  ```python
  if config.get('rnn', {}).get('dropout', 0) > 0 and config.get('rnn', {}).get('num_layers', 1) == 1:
      config['rnn']['dropout'] = 0
      logger.warning("Adjusted RNN dropout to 0 for single-layer model")
  ```

### 5. Pydantic Validation Errors for `/tts`

**Symptoms**:
- Error: `pydantic.error_wrappers.ValidationError: ...` for `/tts` endpoint.
- Client receives 500 error when `audio_url`, `duration`, or `tokens` are missing.

**Diagnostics**:
- Check `TTSResponse` model in `main.py`:

  ```python
  class TTSResponse(BaseModel):
      audio_url: Optional[str]
      duration: Optional[float]
      tokens: Optional[List[str]]
      request_id: str
      status: str
  ```

- Test endpoint:

  ```bash
  curl -X POST "http://localhost:8000/tts" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}'
  ```

**Solutions**:
- **Fix Pydantic Model**:
  Ensure `Optional` fields are set to `None` by default:

  ```python
  class TTSResponse(BaseModel):
      audio_url: Optional[str] = None
      duration: Optional[float] = None
      tokens: Optional[List[str]] = None
      request_id: str
      status: str
  ```

- **Update Endpoint Logic**:
  In `/tts` handler, initialize response correctly:

  ```python
  @app.post("/tts", response_model=TTSResponse)
  async def tts(request: TTSRequest):
      request_id = str(uuid.uuid4())
      status = "queued"
      # Queue job logic...
      return TTSResponse(
          audio_url=None,
          duration=None,
          tokens=None,
          request_id=request_id,
          status=status
      )
  ```

- **Validate Response**:
  Test again to confirm:

  ```bash
  curl -X POST "http://localhost:8000/tts" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}'
  ```

  Expected:

  ```json
  {
    "audio_url": null,
    "duration": null,
    "tokens": null,
    "request_id": "da7cd027-...-1234dc",
    "status": "queued"
  }
  ```

### 6. `/tts/stream` Endpoint Failures

**Symptoms**:
- Client receives 429 errors.
- No audio output or partial audio.
- Error: `OSError: ffmpeg not found`.

**Diagnostics**:
- Check rate-limiting:

  ```bash
  redis-cli -n 0 keys "rate_limit:*"
  ```

- Verify `ffmpeg`:

  ```bash
  ffmpeg -version
  ```

- Test endpoint:

  ```bash
  curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
  ```

- Check server logs:

  ```bash
  tail -f uvicorn.log
  ```

**Solutions**:
- **Handle Rate-Limiting**:
  Clear Redis keys for testing:

  ```bash
  redis-cli -n 0 flushdb
  ```

  Or disable:

  ```bash
  export ENABLE_RATE_LIMITING="false"
  ```

- **Install `ffmpeg`**:

  ```bash
  sudo apt install ffmpeg
  ```

- **Increase Timeout**:
  In `main.py`, adjust streaming timeout:

  ```python
  async def stream_tts(request: TTSRequest):
      # ... streaming logic ...
      timeout = 30  # Increase if needed
      async with async_timeout.timeout(timeout):
          # Stream audio
  ```

- **Debug Streaming**:
  Add logging in `main.py`:

  ```python
  import logging
  logger = logging.getLogger(__name__)
  
  async def stream_tts(request: TTSRequest):
      logger.debug(f"Streaming request: {request.dict()}")
      try:
          # Streaming logic
      except Exception as e:
          logger.error(f"Streaming failed: {str(e)}")
          raise
  ```

### 7. Slow Performance

**Symptoms**:
- High latency for `/tts` or `/tts/stream`.
- CPU/GPU utilization spikes.

**Diagnostics**:
- Check CUDA usage:

  ```bash
  nvidia-smi
  ```

- Monitor API metrics:

  ```bash
  curl http://localhost:8000/metrics
  ```

- Profile `main.py`:

  ```bash
  pip install py-spy
  py-spy record -o profile.svg -- python main.py
  ```

**Solutions**:
- **Optimize Model**:
  Use smaller batch sizes in `main.py`:

  ```python
  model.batch_size = 1  # Adjust based on GPU memory
  ```

- **Enable CUDA**:
  Ensure `use_gpu=True` is respected:

  ```python
  if request.use_gpu and torch.cuda.is_available():
      model.to('cuda')
  ```

- **Cache Responses**:
  Use `cachetools` for repeated requests:

  ```python
  from cachetools import TTLCache
  cache = TTLCache(maxsize=100, ttl=3600)
  
  @app.post("/tts/stream")
  async def stream_tts(request: TTSRequest):
      cache_key = str(request)
      if cache_key in cache:
          logger.info("Serving from cache")
          return cache[cache_key]
      # ... streaming logic ...
      cache[cache_key] = response
      return response
  ```

## Feature History

The following changelog details features and fixes added to `main.py`, based on the user’s queries about endpoint implementations and specific issues. The history focuses on endpoint development, streaming, and Pydantic fixes, reflecting the user’s requests for JavaScript, Python, and multi-language streaming examples.

### Version 0.1.0 (Initial Implementation)

- **Date**: March 2025
- **Features**:
  - Implemented core FastAPI application with `/tts` endpoint for asynchronous TTS jobs.
  - Added `TTSRequest` and `TTSResponse` Pydantic models:

    ```python
    class TTSRequest(BaseModel):
        text: str
        voice: str
        speed: float = 1.0
        use_gpu: bool = True
        return_tokens: bool = False
        format: str = "wav"
        pronunciations: Optional[Dict[str, str]] = None

    class TTSResponse(BaseModel):
        audio_url: Optional[str]
        duration: Optional[float]
        tokens: Optional[List[str]]
        request_id: str
        status: str
    ```

  - Supported 28 voices (American/British, male/female).
  - Integrated `torch` for model loading and `scipy` for audio processing.
- **Issues**:
  - Pydantic validation errors due to missing `None` defaults in `TTSResponse`.

### Version 0.2.0 (Streaming and Additional Endpoints)

- **Date**: March 2025
- **Context**: User requested JavaScript documentation for `/tts/stream` (with `request` library).
- **Features**:
  - Added `/tts/stream` endpoint for real-time audio streaming:

    ```python
    @app.post("/tts/stream")
    async def stream_tts(request: TTSRequest):
        audio_data = generate_audio(request.text, request.voice, request.speed, request.format)
        return StreamingResponse(audio_data, media_type=f"audio/{request.format}")
    ```

  - Implemented `/tts/batch`, `/status`, `/audio`, `/voices`, `/health`, and other endpoints (total 18).
  - Added Redis-based rate-limiting (10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`):

    ```python
    from fastapi_limiter import FastAPILimiter
    redis = Redis(host='localhost', port=6379, db=0)
    @app.on_event("startup")
    async def startup():
        await FastAPILimiter.init(redis)
    ```

  - Enabled CORS for browser clients:

    ```python
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ```

- **Issues**:
  - RNN dropout warning (`dropout=0.2 and num_layers=1`).
  - `ffmpeg` dependency missing on some systems.

### Version 0.3.0 (Pydantic Fix and Python Support)

- **Date**: April 2025
- **Context**: User requested Python documentation with `requests` and noted Pydantic issues with `/tts`.
- **Features**:
  - Fixed Pydantic validation for `/tts` by setting default `None` values:

    ```python
    class TTSResponse(BaseModel):
        audio_url: Optional[str] = None
        duration: Optional[float] = None
        tokens: Optional[List[str]] = None
        request_id: str
        status: str
    ```

  - Enhanced `/tts/stream` with robust error handling:

    ```python
    @app.post("/tts/stream")
    async def stream_tts(request: TTSRequest):
        try:
            if not request.text:
                raise HTTPException(status_code=400, detail="Text is required")
            audio_data = generate_audio(request.text, request.voice, request.speed, request.format)
            return StreamingResponse(audio_data, media_type=f"audio/{request.format}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    ```

  - Added Prometheus metrics endpoint (`/metrics`):

    ```python
    from prometheus_client import Counter
    tts_requests = Counter('tts_requests_total', 'Total TTS requests', ['endpoint'])
    ```

- **Issues**:
  - `weight_norm` deprecation in `torch`.
  - Streaming latency on CPU-only systems.

### Version 0.4.0 (Multi-Language Streaming)

- **Date**: April 2025
- **Context**: User requested real-time streaming examples in cURL, JavaScript, Python, Ruby, Go, and PHP.
- **Features**:
  - Optimized `/tts/stream` for multi-language clients, ensuring compatibility with WAV, MP3, OGG:

    ```python
    async def generate_audio(text: str, voice: str, speed: float, format: str):
        # Validate inputs
        if format not in ["wav", "mp3", "ogg"]:
            raise ValueError("Unsupported format")
        # Generate audio with model
        audio = model.synthesize(text, voice, speed)
        # Convert format with ffmpeg
        return convert_audio(audio, format)
    ```

  - Added custom pronunciation support:

    ```python
    @app.post("/pronunciation")
    async def add_pronunciation(word: str, pronunciation: str, language_code: str = "a"):
        pronunciations[language_code][word.lower()] = pronunciation
        return {"status": "success", "word": word, "pronunciation": pronunciation}
    ```

  - Improved health endpoint:

    ```python
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "cuda_available": torch.cuda.is_available(),
            "ffmpeg_available": shutil.which("ffmpeg") is not None
        }
    ```

- **Issues**:
  - Rate-limiting caused 429 errors for frequent testing.
  - Playback support varied across client languages.

## Current `main.py` (Simplified Example)

Below is a simplified version of `main.py` reflecting key features and fixes:

```python
import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import redis
from fastapi_limiter import FastAPILimiter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = Path('/path/to/kokoro-v1_0.pth')
CONFIG_PATH = Path('/path/to/config.json')
redis = redis.Redis(host='localhost', port=6379, db=0)

class TTSRequest(BaseModel):
    text: str
    voice: str
    speed: float = 1.0
    use_gpu: bool = True
    return_tokens: bool = False
    format: str = "wav"
    pronunciations: Optional[Dict[str, str]] = None

class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    tokens: Optional[List[str]] = None
    request_id: str
    status: str

@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(redis)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    # Load model...

@app.post("/tts", response_model=TTSResponse)
async def tts(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    request_id = str(uuid.uuid4())
    # Queue job...
    return TTSResponse(request_id=request_id, status="queued")

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        audio_data = generate_audio(request.text, request.voice, request.speed, request.format)
        return StreamingResponse(audio_data, media_type=f"audio/{request.format}")
    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_audio(text: str, voice: str, speed: float, format: str):
    # Placeholder for audio generation
    # Apply pronunciations, synthesize with model, convert with ffmpeg
    pass
```

## Setup and Dependencies

### Server Setup

1. **Install Dependencies**:

   ```bash
   pip install fastapi uvicorn torch scipy cachetools fastapi-limiter redis prometheus-client nest_asyncio spacy
   python -m spacy download en_core_web_sm
   sudo apt install ffmpeg
   ```

2. **Configure Redis**:

   ```bash
   redis-server
   redis-cli ping  # Expected: PONG
   ```

3. **Update Paths**:

   ```python
   MODEL_PATH = Path('/correct/path/to/kokoro-v1_0.pth')
   CONFIG_PATH = Path('/correct/path/to/config.json')
   ```

4. **Run**:

   ```bash
   python main.py
   ```

### Client Testing

Test endpoints with cURL:

```bash
curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
curl -X GET "http://localhost:8000/health"
```

## Limitations and Workarounds

- **RNN Dropout**:
  - Fixed in `config.json` (dropout=0).
  - Monitor warnings with logging.

- **WeightNorm Deprecation**:
  - Upgrade `kokoro`:

    ```bash
    pip install --upgrade kokoro
    ```

- **Rate-Limiting**:
  - Disable for debugging:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **Streaming Latency**:
  - Use CUDA or optimize batch sizes.
  - Cache frequent requests (see Slow Performance).

- **Playback**:
  - Clients (e.g., Ruby, Go, PHP) require external playback tools.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025