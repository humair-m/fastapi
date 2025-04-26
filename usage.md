# Kokoro TTS API Documentation

## Overview

The Kokoro TTS API is a FastAPI-based text-to-speech (TTS) service that converts text into high-quality audio using the Kokoro model. It supports multiple voices (American and British English, male and female), various audio formats (WAV, MP3, OGG), and advanced features like custom pronunciations, batch processing, and real-time streaming. The API is designed for scalability with Redis-based rate-limiting, Prometheus metrics, and CUDA acceleration for GPU-enabled environments. It is compatible with Jupyter notebooks via `nest_asyncio` and includes robust error handling and monitoring endpoints.

This documentation covers all endpoints, supported voices, request/response formats, and setup instructions. The API is versioned at 1.0.0 and runs on `http://localhost:8000` by default.

## Endpoints

### 1. `POST /tts`

Initiates a TTS job to convert text to audio. The job is processed asynchronously, and the response includes a `request_id` for status tracking. Audio is available via the `/audio/{filename}` endpoint once the job completes.

- **Method**: POST
- **Request Body**:
  ```json
  {
    "text": "string",                  // Text to synthesize (max 5000 chars, required)
    "voice": "string",                 // Voice ID (default: "af_heart")
    "speed": number,                   // Speech speed (0.5 to 2.0, default: 1.0)
    "use_gpu": boolean,                // Use GPU if available (default: true)
    "return_tokens": boolean,          // Return phonetic tokens (default: false)
    "format": "string",                // Audio format ("wav", "mp3", "ogg", default: "wav")
    "pronunciations": {                // Custom pronunciations (optional)
      "word": "pronunciation"
    }
  }
  ```
- **Response** (HTTP 200):
  ```json
  {
    "audio_url": null,                 // URL to audio file (null when queued)
    "duration": null,                  // Audio duration in seconds (null when queued)
    "tokens": null,                    // Phonetic tokens (null if not requested or queued)
    "request_id": "string",            // Unique job ID
    "status": "queued"                 // Job status
  }
  ```
- **Errors**:
  - 400: Invalid input (e.g., empty text, invalid voice).
  - 429: Rate limit exceeded (10 requests/minute).
  - 500: Server error (e.g., model failure).
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/tts" \
    -H "Content-Type: application/json" \
    -d '{
      "text": "Hello, this is a test of the Kokoro TTS API.",
      "voice": "af_heart",
      "speed": 1.0,
      "use_gpu": true,
      "return_tokens": false,
      "format": "wav",
      "pronunciations": {"kokoro": "koh-koh-roh"}
    }'
  ```

### 2. `POST /tts/stream`

Streams audio in real-time for immediate playback. Does not save audio to disk.

- **Method**: POST
- **Request Body**: Same as `/tts`.
- **Response** (HTTP 200): Audio stream (media type: `audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Errors**:
  - 400: Invalid input.
  - 429: Rate limit exceeded (5 requests/minute).
  - 500: Processing error.
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/tts/stream" \
    -H "Content-Type: application/json" \
    -o output.wav \
    -d '{
      "text": "Streaming audio test.",
      "voice": "am_michael",
      "speed": 1.0,
      "use_gpu": true,
      "format": "wav"
    }'
  ```

### 3. `POST /tts/batch`

Processes multiple TTS requests in a batch. Returns a `batch_id` for tracking.

- **Method**: POST
- **Request Body**:
  ```json
  {
    "items": [                        // Array of TTS requests (min 1)
      {
        "text": "string",
        "voice": "string",
        "speed": number,
        "use_gpu": boolean,
        "return_tokens": boolean,
        "format": "string",
        "pronunciations": { "word": "pronunciation" }
      }
    ]
  }
  ```
- **Response** (HTTP 200):
  ```json
  {
    "batch_id": "string",             // Unique batch ID
    "status": "queued",               // Batch status
    "total_items": number             // Number of requests
  }
  ```
- **Errors**:
  - 400: Invalid input.
  - 429: Rate limit exceeded (5 requests/minute).
  - 500: Server error.
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/tts/batch" \
    -H "Content-Type: application/json" \
    -d '{
      "items": [
        {
          "text": "First test sentence.",
          "voice": "af_bella",
          "speed": 1.2,
          "format": "mp3"
        },
        {
          "text": "Second test sentence.",
          "voice": "bm_george",
          "speed": 0.8,
          "format": "wav"
        }
      ]
    }'
  ```

### 4. `GET /status/{job_id}`

Checks the status of a single TTS job.

- **Method**: GET
- **Path Parameter**: `job_id` (string, required)
- **Response** (HTTP 200):
  ```json
  {
    "status": "string",               // "queued", "processing", "complete", "failed", "cancelled"
    "progress": number,               // Progress (0-100)
    "error": "string|null",           // Error message if failed
    "result": {                       // Result if complete (same as TTSResponse)
      "audio_url": "string|null",
      "duration": number|null,
      "tokens": "string|null",
      "request_id": "string",
      "status": "string"
    }|null
  }
  ```
- **Errors**:
  - 404: Job not found.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/status/123e4567-e89b-12d3-a456-426614174000"
  ```

### 5. `GET /status/batch/{batch_id}`

Checks the status of a batch job and its individual items.

- **Method**: GET
- **Path Parameter**: `batch_id` (string, required)
- **Response** (HTTP 200):
  ```json
  {
    "batch_id": "string",
    "batch_status": {
      "status": "string",
      "progress": number,
      "total_items": number,
      "processed_items": number
    },
    "items": [
      {
        "item_id": "string",
        "status": "string",
        "progress": number,
        "error": "string|null",
        "result": { /* same as TTSResponse */ }|null
      }
    ]
  }
  ```
- **Errors**:
  - 404: Batch not found.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/status/batch/987fcdeb-1234-5678-9012-34567890abcd"
  ```

### 6. `GET /audio/{filename}`

Retrieves a generated audio file.

- **Method**: GET
- **Path Parameter**: `filename` (string, required)
- **Response** (HTTP 200): Audio file (media type: `audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Errors**:
  - 404: File not found.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/audio/123e4567-e89b-12d3-a456-426614174000.wav" \
    -o downloaded_audio.wav
  ```

### 7. `DELETE /audio/{filename}`

Deletes a generated audio file.

- **Method**: DELETE
- **Path Parameter**: `filename` (string, required)
- **Response** (HTTP 200):
  ```json
  {
    "status": "deleted",
    "filename": "string"
  }
  ```
- **Errors**:
  - 400: Invalid filename.
  - 404: File not found.
  - 500: Deletion failed.
- **Example**:
  ```bash
  curl -X DELETE "http://localhost:8000/audio/123e4567-e89b-12d3-a456-426614174000.wav"
  ```

### 8. `GET /voices`

Lists all available voices.

- **Method**: GET
- **Response** (HTTP 200):
  ```json
  {
    "american_female": [
      { "id": "af_heart", "name": "Heart", "emoji": "❤️" },
      { "id": "af_bella", "name": "Bella", "emoji": "🔥" },
      { "id": "af_nicole", "name": "Nicole", "emoji": "🎧" },
      { "id": "af_aoede", "name": "Aoede" },
      { "id": "af_kore", "name": "Kore" },
      { "id": "af_sarah", "name": "Sarah" },
      { "id": "af_nova", "name": "Nova" },
      { "id": "af_sky", "name": "Sky" },
      { "id": "af_alloy", "name": "Alloy" },
      { "id": "af_jessica", "name": "Jessica" },
      { "id": "af_river", "name": "River" }
    ],
    "american_male": [
      { "id": "am_michael", "name": "Michael" },
      { "id": "am_fenrir", "name": "Fenrir" },
      { "id": "am_puck", "name": "Puck" },
      { "id": "am_echo", "name": "Echo" },
      { "id": "am_eric", "name": "Eric" },
      { "id": "am_liam", "name": "Liam" },
      { "id": "am_onyx", "name": "Onyx" },
      { "id": "am_santa", "name": "Santa" },
      { "id": "am_adam", "name": "Adam" }
    ],
    "british_female": [
      { "id": "bf_emma", "name": "Emma" },
      { "id": "bf_isabella", "name": "Isabella" },
      { "id": "bf_alice", "name": "Alice" },
      { "id": "bf_lily", "name": "Lily" }
    ],
    "british_male": [
      { "id": "bm_george", "name": "George" },
      { "id": "bm_fable", "name": "Fable" },
      { "id": "bm_lewis", "name": "Lewis" },
      { "id": "bm_daniel", "name": "Daniel" }
    ]
  }
  ```
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/voices"
  ```

### 9. `GET /voices/preview`

Generates audio previews for all voices using the text "Hello, this is a sample of my voice." Returns a `batch_id` for tracking.

- **Method**: GET
- **Response** (HTTP 200):
  ```json
  {
    "batch_id": "string",
    "status": "queued",
    "total_items": number
  }
  ```
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/voices/preview"
  ```

### 10. `GET /health`

Checks the API’s health and system status.

- **Method**: GET
- **Response** (HTTP 200):
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "timestamp": number,
    "models": {
      "cpu": "available",
      "gpu": "available|not initialized"
    },
    "model_files": {
      "model_exists": boolean,
      "config_exists": boolean
    },
    "disk_space": {
      "total_gb": number,
      "free_gb": number
    },
    "memory": {
      "cuda": "string|not available"
    },
    "cuda_available": boolean,
    "ffmpeg_available": boolean,
    "active_jobs": number,
    "cached_files": number,
    "rate_limiting_enabled": boolean
  }
  ```
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/health"
  ```

### 11. `DELETE /cleanup`

Deletes audio files older than the specified number of hours (default: 24).

- **Method**: DELETE
- **Query Parameter**: `hours` (integer, minimum 1, default 24)
- **Response** (HTTP 200):
  ```json
  {
    "status": "completed",
    "deleted_files": number,
    "errors": number
  }
  ```
- **Example**:
  ```bash
  curl -X DELETE "http://localhost:8000/cleanup?hours=24"
  ```

### 12. `POST /pronunciation`

Adds a custom pronunciation for a word.

- **Method**: POST
- **Query Parameters**:
  - `word` (string, required): Word to customize.
  - `pronunciation` (string, required): Pronunciation (e.g., phonetic spelling).
  - `language_code` (string, default: "a"): "a" (American) or "b" (British).
- **Response** (HTTP 200):
  ```json
  {
    "status": "success",
    "word": "string",
    "pronunciation": "string",
    "language": "American English|British English"
  }
  ```
- **Errors**:
  - 400: Invalid input.
  - 500: Processing error.
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/pronunciation?word=example&pronunciation=eg-zam-pul&language_code=a"
  ```

### 13. `GET /pronunciations`

Lists all custom pronunciations for a language.

- **Method**: GET
- **Query Parameter**: `language_code` (string, default: "a"): "a" (American) or "b" (British).
- **Response** (HTTP 200):
  ```json
  {
    "language": "American English|British English",
    "pronunciations": {
      "word": "pronunciation"
    }
  }
  ```
- **Errors**:
  - 500: Processing error.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/pronunciations?language_code=a"
  ```

### 14. `DELETE /pronunciations/{word}`

Deletes a custom pronunciation.

- **Method**: DELETE
- **Path Parameter**: `word` (string, required)
- **Query Parameter**: `language_code` (string, default: "a"): "a" (American) or "b" (British).
- **Response** (HTTP 200):
  ```json
  {
    "status": "deleted",
    "word": "string",
    "language": "American English|British English"
  }
  ```
- **Errors**:
  - 404: Pronunciation not found.
  - 500: Processing error.
- **Example**:
  ```bash
  curl -X DELETE "http://localhost:8000/pronunciations/example?language_code=a"
  ```

### 15. `POST /preprocess`

Preprocesses text by cleaning and normalizing it (e.g., expanding abbreviations, removing special characters).

- **Method**: POST
- **Request Body**:
  ```json
  {
    "text": "string"                  // Text to preprocess (max 5000 chars, required)
  }
  ```
- **Response** (HTTP 200):
  ```json
  {
    "original_text": "string",
    "processed_text": "string"
  }
  ```
- **Errors**:
  - 400: Invalid input.
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/preprocess" \
    -H "Content-Type: application/json" \
    -d '{"text": "Mr. Smith said: Hello, world! @#$%"}'
  ```

### 16. `GET /metrics`

Exposes Prometheus metrics for monitoring (e.g., request counts, latency, active jobs).

- **Method**: GET
- **Response** (HTTP 200): Prometheus text format.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/metrics"
  ```

### 17. `GET /config`

Returns API configuration details.

- **Method**: GET
- **Response** (HTTP 200):
  ```json
  {
    "sample_rate": number,
    "max_char_limit": number,
    "supported_formats": ["string"],
    "cuda_available": boolean,
    "rate_limiting_enabled": boolean
  }
  ```
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/config"
  ```

### 18. `POST /jobs/cancel/{job_id}`

Cancels a running TTS job.

- **Method**: POST
- **Path Parameter**: `job_id` (string, required)
- **Response** (HTTP 200):
  ```json
  {
    "status": "cancelled",
    "job_id": "string"
  }
  ```
- **Errors**:
  - 404: Job not found.
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/jobs/cancel/123e4567-e89b-12d3-a456-426614174000"
  ```

## Voices

The API supports the following voices, categorized by accent and gender. Each voice has a unique ID used in TTS requests.

- **American Female**:
  - `af_heart` (Heart, ❤️)
  - `af_bella` (Bella, 🔥)
  - `af_nicole` (Nicole, 🎧)
  - `af_aoede` (Aoede)
  - `af_kore` (Kore)
  - `af_sarah` (Sarah)
  - `af_nova` (Nova)
  - `af_sky` (Sky)
  - `af_alloy` (Alloy)
  - `af_jessica` (Jessica)
  - `af_river` (River)

- **American Male**:
  - `am_michael` (Michael)
  - `am_fenrir` (Fenrir)
  - `am_puck` (Puck)
  - `am_echo` (Echo)
  - `am_eric` (Eric)
  - `am_liam` (Liam)
  - `am_onyx` (Onyx)
  - `am_santa` (Santa)
  - `am_adam` (Adam)

- **British Female**:
  - `bf_emma` (Emma)
  - `bf_isabella` (Isabella)
  - `bf_alice` (Alice)
  - `bf_lily` (Lily)

- **British Male**:
  - `bm_george` (George)
  - `bm_fable` (Fable)
  - `bm_lewis` (Lewis)
  - `bm_daniel` (Daniel)

Use the `/voices` endpoint to retrieve this list programmatically.

## Features

- **Asynchronous Processing**: TTS jobs (`/tts`, `/tts/batch`, `/voices/preview`) are processed in the background, with status tracking via `/status` endpoints.
- **Real-Time Streaming**: The `/tts/stream` endpoint delivers audio instantly for playback.
- **Multiple Audio Formats**: Supports WAV, MP3, and OGG via `ffmpeg` conversion.
- **Custom Pronunciations**: Add, list, or delete pronunciations for American (`a`) or British (`b`) English.
- **Rate-Limiting**: Redis-based rate-limiting (10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`). Configurable via `ENABLE_RATE_LIMITING` environment variable.
- **Prometheus Metrics**: Tracks request counts, latency, and active jobs via `/metrics`.
- **CUDA Support**: Automatically uses GPU if available, with CPU fallback for CUDA errors (e.g., out-of-memory).
- **Jupyter Compatibility**: Supports Jupyter notebooks using `nest_asyncio`.
- **Text Preprocessing**: Cleans and normalizes text via `/preprocess` (e.g., expands "Mr." to "mister").
- **Audio Caching**: Uses `TTLCache` (1-hour TTL) to reduce disk access for `/audio` requests.
- **Health Monitoring**: The `/health` endpoint reports system status, disk space, and model availability.
- **Job Cancellation**: Cancel running jobs via `/jobs/cancel/{job_id}`.
- **Error Handling**: Custom `TTSException` ensures consistent error responses.
- **spaCy Integration**: Pre-loads `en_core_web_sm` to avoid runtime downloads.

## Setup Instructions

### Prerequisites

- **Python**: 3.12 or higher
- **OS**: Linux (tested with Ubuntu)
- **Hardware**: GPU recommended for CUDA support
- **Files**:
  - `kokoro-v1_0.pth`: Kokoro model file
  - `config.json`: Model configuration file

### Installation

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn torch scipy cachetools fastapi-limiter redis prometheus-client nest_asyncio spacy
   python -m spacy download en_core_web_sm
   sudo apt install ffmpeg
   ```

2. **Configure Redis**:
   - Start Redis:
     ```bash
     redis-server
     ```
     Or use Docker:
     ```bash
     docker run -d -p 6379:6379 redis
     ```
   - Verify:
     ```bash
     redis-cli ping
     ```
     - Expected: `PONG`
   - Disable rate-limiting if Redis is unavailable:
     ```bash
     export ENABLE_RATE_LIMITING="false"
     ```

3. **Update File Paths**:
   - Edit `main.py` to set `MODEL_PATH` and `CONFIG_PATH`:
     ```python
     MODEL_PATH = Path('/path/to/kokoro-v1_0.pth')
     CONFIG_PATH = Path('/path/to/config.json')
     ```
   - Verify:
     ```bash
     ls -l /path/to/kokoro-v1_0.pth
     ls -l /path/to/config.json
     python -m json.tool /path/to/config.json
     ```

4. **Run the API**:
   - **Standalone**:
     ```bash
     python main.py
     ```
   - **Jupyter Notebook**:
     1. Copy `main.py` code into a cell.
     2. Update file paths.
     3. Run the cell.
   - Default: Runs on `http://localhost:8000`. Change port if needed:
     ```bash
     export PORT=8080
     python main.py
     ```

5. **Verify**:
   - Check health:
     ```bash
     curl -X GET "http://localhost:8000/health"
     ```
   - Test TTS:
     ```bash
     curl -X POST "http://localhost:8000/tts" \
       -H "Content-Type: application/json" \
       -d '{"text": "Test", "voice": "af_heart", "format": "wav"}'
     ```

## Configuration

- **Environment Variables**:
  - `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379`)
  - `ENABLE_RATE_LIMITING`: Enable/disable rate-limiting (default: `true`)
  - `PORT`: Server port (default: `8000`)
- **Constants** (in `main.py`):
  - `AUDIO_OUTPUT_DIR`: Directory for audio files (default: `./audio_output`)
  - `SAMPLE_RATE`: Audio sample rate (default: 24000 Hz)
  - `MAX_CHAR_LIMIT`: Max text length (default: 5000)
  - `CLEANUP_HOURS`: Default cleanup threshold (default: 24)
  - `CACHE_TTL`: Audio cache TTL (default: 3600 seconds)
  - `RATE_LIMIT`: Rate limit (default: `10/minute`)

## Limitations and Known Issues

- **RNN Dropout Warning**: The Kokoro model may log a warning (`dropout=0.2 and num_layers=1`). Fix by editing `config.json`:
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

- **WeightNorm Deprecation**: The `kokoro` library may use deprecated `weight_norm`. Update the library:
  ```bash
  pip install --upgrade kokoro
  ```

- **File Paths**: Incorrect `MODEL_PATH` or `CONFIG_PATH` causes startup errors. Ensure paths are correct.

- **Redis Dependency**: Rate-limiting requires Redis. Disable if unavailable:
  ```bash
  export ENABLE_RATE_LIMITING="false"
  ```

- **Jupyter Issues**: Restart the kernel if asyncio errors occur. Ensure `nest_asyncio` is installed.

- **CUDA Errors**: Out-of-memory errors trigger CPU fallback. Monitor GPU usage for large requests.

## Troubleshooting

- **Startup Failure**:
  - Verify file paths and `config.json`:
    ```bash
    python -m json.tool /path/to/config.json
    ```
  - Check dependencies:
    ```bash
    pip show fastapi uvicorn torch scipy fastapi-limiter redis prometheus-client nest_asyncio spacy
    ```

- **Validation Errors**:
  - Ensure Pydantic >= 2.11:
    ```bash
    pip install --upgrade pydantic
    ```

- **Rate-Limit Errors**:
  - Check Redis:
    ```bash
    redis-cli ping
    ```
  - Disable rate-limiting:
    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **No Audio Output**:
  - Check job status:
    ```bash
    curl -X GET "http://localhost:8000/status/{job_id}"
    ```
  - Verify `ffmpeg`:
    ```bash
    ffmpeg -version
    ```

For further assistance, contact the API maintainer or share logs and `config.json` contents.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025
