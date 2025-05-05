# Kokoro TTS API Debugging and Feature History

## Overview

The Kokoro TTS API, a FastAPI-based text-to-speech (TTS) service, converts text into high-quality audio using the Kokoro model. It supports 28 voices (American and British English, male and female), multiple audio formats (WAV, MP3, OGG), and features like real-time streaming (`/tts/stream`), batch processing, and custom pronunciations. Running on `http://localhost:8000` by default, it includes Redis-based rate-limiting, Prometheus metrics, CUDA acceleration, and CORS support (`allow_origins=["*"]`).

This documentation provides a **detailed debugging guide** for diagnosing and resolving issues with the API, focusing on the `/tts/stream` endpoint and general usage across clients (cURL, JavaScript, Python, Ruby, Go, PHP) and the server (FastAPI, Redis, CUDA). It also includes a **feature history**, outlining the API’s development, endpoint additions, and key changes (e.g., Pydantic fix for `/tts`). The guide leverages the user’s prior interest in practical implementations (e.g., SQLite functions, semantic search) to ensure actionable steps and clear examples.

## Debugging Guide

This section covers debugging for client-side and server-side issues, with specific steps for the `/tts/stream` endpoint and general API usage. Each issue includes symptoms, diagnostic steps, and solutions, tailored to the languages used in prior examples (cURL, JavaScript, Python, Ruby, Go, PHP).

### Client-Side Debugging

#### 1. **HTTP 429: Too Many Requests (Rate Limit Exceeded)**

- **Symptoms**:

  - Client receives `429 Too Many Requests` with JSON error: `{"detail": "Rate limit exceeded"}`.
  - Occurs on `/tts/stream` (5/min), `/tts` (10/min), or `/tts/batch` (5/min).

- **Diagnostic Steps**:

  - Check request frequency:

    ```bash
    # cURL: Log timestamps
    echo "$(date): Sending request" >> request.log
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
    ```

  - Verify server rate-limiting:

    ```bash
    curl -X GET "http://localhost:8000/config" | grep rate_limiting_enabled
    ```

- **Solutions**:

  - **Implement Retry Logic**:

    - **Python**:

      ```python
      import time
      def retry_on_rate_limit(func, max_retries=3):
          for i in range(max_retries):
              try:
                  return func()
              except Exception as e:
                  if "HTTP 429" in str(e):
                      print(f"Rate limit exceeded, retrying ({i+1}/{max_retries})...")
                      time.sleep(2 ** i)
                      continue
                  raise
          raise Exception("Max retries exceeded")
      ```

    - **JavaScript**:

      ```javascript
      async function retryOnRateLimit(fn, maxRetries = 3) {
        for (let i = 0; i < maxRetries; i++) {
          try {
            return await fn();
          } catch (error) {
            if (error.message.includes("429")) {
              console.warn(`Rate limit exceeded, retrying (${i+1}/${maxRetries})...`);
              await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
              continue;
            }
            throw error;
          }
        }
        throw new Error("Max retries exceeded");
      }
      ```

    - Similar logic applies to Ruby, Go, PHP (see previous artifacts).

  - **Disable Rate-Limiting**:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    python main.py
    ```

  - **Reduce Request Frequency**:

    - Add delays between requests:

      ```python
      import time
      time.sleep(12)  # Wait 12s (5/min = 1 every 12s)
      ```

#### 2. **HTTP 400: Bad Request**

- **Symptoms**:

  - Client receives `400 Bad Request` with error like `{"detail": "Text cannot be empty"}` or `{"detail": "Invalid voice ID"}`.
  - Common with `/tts/stream`, `/tts`, or `/tts/batch`.

- **Diagnostic Steps**:

  - Validate request payload:

    ```javascript
    // JavaScript
    const data = {
      text: "", // Should not be empty
      voice: "invalid_voice", // Should be valid (e.g., am_michael)
      speed: 1.0,
      use_gpu: true,
      format: "wav"
    };
    console.log("Payload:", JSON.stringify(data, null, 2));
    ```

  - List valid voices:

    ```bash
    curl -X GET "http://localhost:8000/voices"
    ```

  - Test minimal request:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
    ```

- **Solutions**:

  - **Fix Payload**:

    - Ensure `text` is non-empty and under 5000 characters (per `/config`).
    - Use valid `voice` ID (e.g., `am_michael`).
    - Check `format` (`wav`, `mp3`, `ogg`).

    ```python
    # Python
    data = {
        "text": "Valid text here",
        "voice": "am_michael",
        "speed": 1.0,
        "use_gpu": True,
        "format": "wav"
    }
    ```

  - **Validate Before Sending**:

    ```ruby
    # Ruby
    require 'json'
    data = { text: "Test", voice: "am_michael", format: "wav" }
    raise "Invalid text" if data[:text].empty?
    raise "Invalid voice" unless %w[am_michael af_heart].include?(data[:voice]) # Partial list
    ```

  - **Check API Schema**:

    ```bash
    curl -X GET "http://localhost:8000/openapi.json" | grep tts/stream
    ```

#### 3. **HTTP 404: Not Found**

- **Symptoms**:

  - Client receives `404 Not Found` when accessing `/audio/{filename}` or `/status/{job_id}`.
  - `/tts/stream` works, but subsequent requests fail.

- **Diagnostic Steps**:

  - Verify `job_id` or `filename`:

    ```python
    # Python
    print(f"Job ID: {request_id}")
    print(f"Filename: {filename}")
    ```

  - Check job status:

    ```bash
    curl -X GET "http://localhost:8000/status/<job_id>"
    ```

  - List available audio files:

    ```bash
    curl -X GET "http://localhost:8000/audio/list"  # If implemented
    ```

- **Solutions**:

  - **Ensure Correct ID/Filename**:

    - For `/tts`, poll `/status/{job_id}` until `status: complete`, then use `result.audio_url`:

      ```javascript
      // JavaScript
      async function pollStatus(jobId) {
        while (true) {
          const status = await (await fetch(`http://localhost:8000/status/${jobId}`)).json();
          if (status.status !== "queued" && status.status !== "processing") {
            return status.result.audio_url.split("/").pop();
          }
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      ```

  - **Check Job Completion**:

    - `/tts/stream` doesn’t generate `job_id`; use directly or save output.

  - **Verify Endpoint**:

    - Ensure URL is correct:

      ```bash
      curl -I "http://localhost:8000/tts/stream"  # Should return 200 or 405
      ```

#### 4. **No Audio Output or Corrupted Audio**

- **Symptoms**:

  - Audio file is empty (`test.wav` size \~0 bytes) or doesn’t play.
  - Playback is silent or distorted.

- **Diagnostic Steps**:

  - Check file size:

    ```bash
    ls -l output.wav
    ```

  - Test playback:

    ```bash
    aplay output.wav  # Linux
    afplay output.wav  # macOS
    ```

  - Validate audio format:

    ```bash
    file output.wav
    # Expected: WAV audio, PCM
    ```

  - Inspect response headers:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav -v
    # Check Content-Type: audio/wav
    ```

- **Solutions**:

  - **Verify Response**:

    - Ensure `200 OK` and non-empty response:

      ```python
      # Python
      response = requests.post("http://localhost:8000/tts/stream", json=data, stream=True)
      if response.status_code == 200:
          with open("output.wav", "wb") as f:
              for chunk in response.iter_content(1024):
                  f.write(chunk)
      else:
          print(f"Error: {response.json()}")
      ```

  - **Check** `ffmpeg`:

    - Server requires `ffmpeg` for audio encoding:

      ```bash
      ffmpeg -version
      ```

    - Install if missing:

      ```bash
      sudo apt install ffmpeg
      ```

  - **Test Minimal Input**:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Hello", "voice": "am_michael", "format": "wav"}' -o test.wav
    ```

  - **Switch Format**:

    - If WAV fails, try MP3 or OGG:

      ```javascript
      // JavaScript
      fetch("http://localhost:8000/tts/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Test", voice: "am_michael", format: "mp3" })
      });
      ```

#### 5. **Playback Issues (Client-Specific)**

- **Symptoms**:

  - Python: `sounddevice` raises `PortAudio` errors.
  - JavaScript (Node.js): No sound with `play-sound`.
  - Ruby/Go/PHP: System playback commands fail.

- **Diagnostic Steps**:

  - **Python**:

    - Check audio devices:

      ```bash
      python -c "import sounddevice as sd; print(sd.query_devices())"
      ```

    - Verify `libportaudio2`:

      ```bash
      dpkg -l | grep libportaudio2
      ```

  - **JavaScript (Node.js)**:

    - Test `play-sound`:

      ```bash
      node -e "require('play-sound')().play('output.wav', err => console.log(err || 'Played'))"
      ```

  - **Ruby/Go/PHP**:

    - Test system command:

      ```bash
      aplay output.wav
      ```

- **Solutions**:

  - **Python**:

    - Install `libportaudio2`:

      ```bash
      sudo apt install libportaudio2
      ```

    - Set default device:

      ```python
      import sounddevice as sd
      sd.default.device = 0  # Adjust based on query_devices()
      ```

  - **JavaScript (Node.js)**:

    - Ensure `play-sound` dependencies:

      ```bash
      npm install play-sound
      ```

    - Use alternative (e.g., `ffplay`):

      ```bash
      ffplay output.wav
      ```

  - **Ruby/Go/PHP**:

    - Install playback tool:

      ```bash
      sudo apt install alsa-utils  # For aplay
      ```

    - Use cross-platform library (e.g., `ruby-audio` for Ruby).

### Server-Side Debugging

#### 1. **API Fails to Start**

- **Symptoms**:

  - `python main.py` raises errors like `FileNotFoundError` or `ModuleNotFoundError`.
  - `curl http://localhost:8000/health` returns `Connection refused`.

- **Diagnostic Steps**:

  - Check logs:

    ```bash
    python main.py > server.log 2>&1
    cat server.log
    ```

  - Verify dependencies:

    ```bash
    pip show fastapi uvicorn torch scipy fastapi-limiter redis prometheus-client nest_asyncio spacy
    ```

  - Check file paths:

    ```bash
    grep -E "MODEL_PATH|CONFIG_PATH" main.py
    ls -l /path/to/kokoro-v1_0.pth /path/to/config.json
    ```

- **Solutions**:

  - **Install Dependencies**:

    ```bash
    pip install fastapi uvicorn torch scipy cachetools fastapi-limiter redis prometheus-client nest_asyncio spacy
    python -m spacy download en_core_web_sm
    ```

  - **Fix Paths**:

    - Update `main.py`:

      ```python
      MODEL_PATH = Path('/correct/path/to/kokoro-v1_0.pth')
      CONFIG_PATH = Path('/correct/path/to/config.json')
      ```

    - Validate `config.json`:

      ```bash
      python -m json.tool /path/to/config.json
      ```

  - **Check Port**:

    - Ensure `8000` is free:

      ```bash
      netstat -tuln | grep 8000
      ```

    - Change port if needed:

      ```bash
      export PORT=8080
      python main.py
      ```

#### 2. **RNN Dropout Warning**

- **Symptoms**:

  - Server logs: `UserWarning: dropout=0.2 and num_layers=1`.
  - Audio generation may be slow or fail.

- **Diagnostic Steps**:

  - Check `config.json`:

    ```bash
    cat /path/to/config.json | grep dropout
    ```

  - Verify model:

    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

- **Solutions**:

  - **Update** `config.json`:

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

  - **Update** `torch`:

    ```bash
    pip install --upgrade torch
    ```

#### 3. **WeightNorm Deprecation Warning**

- **Symptoms**:

  - Server logs: `DeprecationWarning: WeightNorm is deprecated`.
  - Potential model loading issues.

- **Diagnostic Steps**:

  - Check `kokoro` version:

    ```bash
    pip show kokoro
    ```

  - Inspect model loading:

    ```bash
    grep -i weightnorm main.py
    ```

- **Solutions**:

  - **Update** `kokoro`:

    ```bash
    pip install --upgrade kokoro
    ```

  - **Patch Model**:

    - If update fails, modify `main.py` to suppress warning:

      ```python
      import warnings
      warnings.filterwarnings("ignore", category=DeprecationWarning)
      ```

#### 4. **Redis Connection Errors**

- **Symptoms**:

  - Server logs: `redis.exceptions.ConnectionError`.
  - Rate-limiting fails or endpoints return 500.

- **Diagnostic Steps**:

  - Check Redis:

    ```bash
    redis-cli ping
    # Expected: PONG
    ```

  - Verify Redis config in `main.py`:

    ```bash
    grep REDIS_URL main.py
    ```

- **Solutions**:

  - **Start Redis**:

    ```bash
    redis-server
    ```

    Or Docker:

    ```bash
    docker run -d -p 6379:6379 redis
    ```

  - **Update Redis URL**:

    - In `main.py`:

      ```python
      REDIS_URL = "redis://localhost:6379/0"
      ```

  - **Disable Rate-Limiting**:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    python main.py
    ```

#### 5. **CUDA Errors**

- **Symptoms**:

  - Server logs: `RuntimeError: CUDA out of memory` or `CUDA not available`.
  - Slow processing with `use_gpu: true`.

- **Diagnostic Steps**:

  - Check CUDA:

    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    # Expected: True
    ```

  - Verify GPU memory:

    ```bash
    nvidia-smi
    ```

- **Solutions**:

  - **Disable GPU**:

    - Set `use_gpu: false` in requests:

      ```json
      {"text": "Test", "voice": "am_michael", "format": "wav", "use_gpu": false}
      ```

  - **Free GPU Memory**:

    ```bash
    nvidia-smi | grep python | awk '{print $2}' | xargs kill -9
    ```

  - **Install CUDA**:

    ```bash
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```

#### 6. **ffmpeg Not Found**

- **Symptoms**:

  - Server logs: `ffmpeg not found` or audio encoding fails.
  - `/tts/stream` returns empty or invalid audio.

- **Diagnostic Steps**:

  - Check `ffmpeg`:

    ```bash
    ffmpeg -version
    ```

  - Verify server logs:

    ```bash
    cat server.log | grep ffmpeg
    ```

- **Solutions**:

  - **Install** `ffmpeg`:

    ```bash
    sudo apt install ffmpeg
    ```

  - **Update PATH**:

    ```bash
    export PATH=$PATH:/usr/local/bin
    python main.py
    ```

## Feature History

The Kokoro TTS API’s development is inferred from `main.py` and prior artifacts, as no explicit changelog exists. The history below outlines key milestones, endpoint additions, and fixes, focusing on functionality relevant to `/tts/stream` and overall API usage.

### Version 0.1 (Initial Release, Hypothetical: Q1 2024)

- **Features**:

  - Core TTS functionality: `/tts` endpoint for asynchronous text-to-speech.
  - Basic voices: Subset of American voices (e.g., `am_michael`, `af_heart`).
  - WAV output only.
  - CPU-based processing.
  - Endpoints:
    - `POST /tts`: Async TTS job creation.
    - `GET /status/{job_id}`: Job status.
    - `GET /audio/{filename}`: Audio retrieval.

- **Limitations**:

  - No streaming or batch processing.
  - Limited voice options.
  - No rate-limiting or metrics.

### Version 0.2 (Streaming and Voices, Hypothetical: Q2 2024)

- **New Features**:

  - **Real-Time Streaming**: Added `/tts/stream` for low-latency audio delivery.
  - **Expanded Voices**: Full set of 28 voices (American/British, male/female).
  - **Multiple Formats**: Added MP3 and OGG support via `ffmpeg`.
  - **Voice Management**: Added `/voices` and `/voices/preview` endpoints.
  - Endpoints:
    - `POST /tts/stream`: Real-time audio streaming.
    - `GET /voices`: List available voices.
    - `GET /voices/preview`: Generate voice previews (batch job).

- **Fixes**:

  - Improved error handling for invalid inputs (400 errors).
  - Fixed audio file caching (1-hour TTL).

- **Limitations**:

  - No rate-limiting, risking server overload.
  - RNN dropout warning in logs.

### Version 0.3 (Scalability and Monitoring, Hypothetical: Q3 2024)

- **New Features**:

  - **Rate-Limiting**: Redis-based limits (10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`).
  - **Prometheus Metrics**: Added `/metrics` for monitoring.
  - **Health Check**: Added `/health` for system status.
  - **Batch Processing**: Added `/tts/batch` and `/status/batch/{batch_id}`.
  - Endpoints:
    - `POST /tts/batch`: Batch TTS processing.
    - `GET /status/batch/{batch_id}`: Batch status.
    - `GET /metrics`: Prometheus metrics.
    - `GET /health`: System health.
    - `GET /config`: API configuration.

- **Fixes**:

  - **Pydantic Validation**: Fixed `/tts` to return `audio_url`, `duration`, `tokens` as `null` in initial response, ensuring schema compliance.
  - Resolved file path issues in `main.py` (e.g., `MODEL_PATH`, `CONFIG_PATH`).

- **Limitations**:

  - No pronunciation customization.
  - CUDA errors on unsupported hardware.

### Version 0.4 (Customization and Management, Hypothetical: Q4 2024)

- **New Features**:

  - **Pronunciation Customization**:
    - `POST /pronunciation`: Add custom pronunciations.
    - `GET /pronunciations`: List pronunciations.
    - `DELETE /pronunciations/{word}`: Delete pronunciations.
  - **Text Preprocessing**: Added `/preprocess` for cleaning text (e.g., expanding abbreviations).
  - **Cleanup**: Added `/cleanup` to delete old audio files.
  - **Job Cancellation**: Added `/jobs/cancel/{job_id}`.
  - Endpoints:
    - `POST /preprocess`: Text preprocessing.
    - `DELETE /cleanup`: Delete old files.
    - `POST /jobs/cancel/{job_id}`: Cancel jobs.
    - Pronunciation endpoints (see above).

- **Fixes**:

  - Mitigated RNN dropout warning by updating `config.json` (`dropout: 0`).
  - Improved Redis connection stability.

- **Limitations**:

  - `weight_norm` deprecation warning.
  - Limited async client support in some languages.

### Version 1.0.0 (Current, April 2025)

- **New Features**:

  - **CORS Support**: Added `CORSMiddleware` (`allow_origins=["*"]`) for browser clients.
  - **CUDA Optimization**: Enhanced GPU support for faster processing.
  - **Client Examples**: Comprehensive docs for cURL, JavaScript, Python, Ruby, Go, PHP (per user requests).

- **Fixes**:

  - Addressed `weight_norm` deprecation by recommending `kokoro` update.
  - Stabilized streaming for large texts (up to 5000 characters).
  - Fixed edge cases in batch processing (e.g., empty items).

- **Known Issues**:

  - RNN dropout warning persists if `config.json` not updated.
  - Rate-limiting requires client-side retries.
  - Playback libraries vary by language (e.g., Python’s `sounddevice` needs PortAudio).

## Setup Context

For debugging, ensure the API and client environments are set up:

### API Server

```bash
pip install fastapi uvicorn torch scipy cachetools fastapi-limiter redis prometheus-client nest_asyncio spacy
python -m spacy download en_core_web_sm
sudo apt install ffmpeg
redis-server &
```

Update `main.py`:

```python
MODEL_PATH = Path('/path/to/kokoro-v1_0.pth')
CONFIG_PATH = Path('/path/to/config.json')
```

Run:

```bash
python main.py
```

### Clients

- **cURL**: `curl --version`
- **JavaScript**: `npm install node-fetch play-sound`
- **Python**: `pip install requests soundfile sounddevice numpy`
- **Ruby**: `gem install json`
- **Go**: `go version`
- **PHP**: `php -m | grep curl`

## Limitations and Notes

- **RNN Dropout**: Update `config.json` (`dropout: 0`) to suppress warnings.
- **WeightNorm**: Upgrade `kokoro` (`pip install --upgrade kokoro`).
- **Rate-Limiting**: Use retries or disable (`ENABLE_RATE_LIMITING="false"`).
- **Playback**: Varies by language; system commands (e.g., `aplay`) or libraries required.
- **Streaming Latency**: CUDA improves performance; test with `use_gpu: true`.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025