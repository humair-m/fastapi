# Kokoro TTS API Python Usage with Requests Documentation

## Overview

The Kokoro TTS API is a FastAPI-based text-to-speech (TTS) service that converts text into high-quality audio using the Kokoro model. It supports 28 voices (American and British English, male and female), multiple audio formats (WAV, MP3, OGG), and features like custom pronunciations, batch processing, and real-time streaming. The API is designed for scalability with Redis-based rate-limiting, Prometheus metrics, and CUDA acceleration. It runs on `http://localhost:8000` by default and supports CORS for cross-origin requests.

This documentation is tailored for Python developers using the `requests` library to interact with the API’s 18 endpoints. Each endpoint is demonstrated with Python code using `requests`, including request construction, response handling, and error management. The documentation covers all voices, features, and setup steps, ensuring integration into Python scripts or applications. The `requests` library is ideal for server-side or command-line applications but not for browser-based usage (use JavaScript for that).

## Prerequisites

- **Python**: 3.12 or higher (compatible with `main.py`).
- **API Server**: Running instance of the Kokoro TTS API (see Setup Instructions).
- **CORS**: The API enables CORS (`allow_origins=["*"]`), but this is irrelevant for `requests` as it operates server-side.
- **Dependencies**:
  - Install `requests`:

    ```bash
    pip install requests
    ```

  - Optional (for streaming/file handling):

    ```bash
    pip install requests
    ```

## Endpoints

Below are the API’s 18 endpoints, each with a Python example using the `requests` library. All examples assume the API runs at `http://localhost:8000`. Responses are JSON unless specified (e.g., audio streams). Error handling checks for HTTP status codes and parses error messages. The examples use `requests.Session` for connection reuse and consistent error handling.

### Helper Function

To simplify examples, a helper function handles requests and errors:

```python
import requests
import json

def make_request(method, url, data=None, params=None, stream=False):
    with requests.Session() as session:
        try:
            response = session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                stream=stream,
                headers={"Content-Type": "application/json"} if data else None
            )
            response.raise_for_status()
            if stream:
                return response
            if response.headers.get("Content-Type", "").startswith("application/json"):
                return response.json()
            return response.text
        except requests.HTTPError as e:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except json.JSONDecodeError:
                error_detail = response.text or "Unknown error"
            raise Exception(f"HTTP {response.status_code}: {error_detail}") from e
        except requests.RequestException as e:
            raise Exception(f"Request failed: {str(e)}") from e
```

### 1. `POST /tts`

Initiates an asynchronous TTS job. Returns a `request_id` to track status. Audio is available via `/audio/{filename}` once complete.

- **Request**:
  - Body: JSON with text, voice, speed, format, etc.
  - Content-Type: `application/json`
- **Response**: JSON with `request_id`, `status`, and `null` fields for `audio_url`, `duration`, `tokens`.
- **Example**:

```python
def synthesize_text():
    try:
        data = make_request(
            method="POST",
            url="http://localhost:8000/tts",
            data={
                "text": "Hello, this is a test of the Kokoro TTS API.",
                "voice": "af_heart",
                "speed": 1.0,
                "use_gpu": True,
                "return_tokens": False,
                "format": "wav",
                "pronunciations": {"kokoro": "koh-koh-roh"}
            }
        )
        print("TTS Job:", data)
        return data["request_id"]
    except Exception as e:
        print(f"Error synthesizing text: {e}")

request_id = synthesize_text()
```

- **Expected Response**:

```json
{
  "audio_url": null,
  "duration": null,
  "tokens": null,
  "request_id": "da7cd027-...-1234dc",
  "status": "queued"
}
```

- **Errors**:
  - 400: Invalid input (e.g., empty text).
  - 429: Rate limit exceeded (10/min).
  - 500: Server error.

### 2. `POST /tts/stream`

Streams audio in real-time. Returns an audio stream, not JSON.

- **Request**: Same as `/tts`.
- **Response**: Audio stream (`audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Example** (Save to file):

```python
def stream_audio():
    try:
        response = make_request(
            method="POST",
            url="http://localhost:8000/tts/stream",
            data={
                "text": "Streaming audio test.",
                "voice": "am_michael",
                "speed": 1.0,
                "use_gpu": True,
                "format": "wav"
            },
            stream=True
        )
        with open("output.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Audio saved to output.wav")
    except Exception as e:
        print(f"Error streaming audio: {e}")

stream_audio()
```

- **Note**: Playback requires a separate library (e.g., `pygame` or `playsound`):

```bash
pip install playsound
```

```python
from playsound import playsound
playsound("output.wav")
```

- **Errors**:
  - 400: Invalid input.
  - 429: Rate limit exceeded (5/min).
  - 500: Processing error.

### 3. `POST /tts/batch`

Processes multiple TTS requests in a batch. Returns a `batch_id`.

- **Request**:
  - Body: JSON array of TTS requests.
- **Response**: JSON with `batch_id`, `status`, and `total_items`.
- **Example**:

```python
def batch_synthesize():
    try:
        data = make_request(
            method="POST",
            url="http://localhost:8000/tts/batch",
            data={
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
            }
        )
        print("Batch Job:", data)
        return data["batch_id"]
    except Exception as e:
        print(f"Error in batch synthesis: {e}")

batch_id = batch_synthesize()
```

- **Expected Response**:

```json
{
  "batch_id": "987fcdeb-...-90abcd",
  "status": "queued",
  "total_items": 2
}
```

- **Errors**:
  - 400: Invalid input.
  - 429: Rate limit exceeded (5/min).
  - 500: Server error.

### 4. `GET /status/{job_id}`

Checks the status of a single TTS job.

- **Request**: Path parameter `job_id`.
- **Response**: JSON with job status, progress, and result.
- **Example**:

```python
def check_job_status(job_id):
    try:
        data = make_request(
            method="GET",
            url=f"http://localhost:8000/status/{job_id}"
        )
        print("Job Status:", data)
        return data
    except Exception as e:
        print(f"Error checking job status: {e}")

status = check_job_status("123e4567-e89b-12d3-a456-426614174000")
```

- **Expected Response**:

```json
{
  "status": "complete",
  "progress": 100,
  "error": null,
  "result": {
    "audio_url": "/audio/123e4567-...-174000.wav",
    "duration": 3.5,
    "tokens": null,
    "request_id": "123e4567-...-174000",
    "status": "complete"
  }
}
```

- **Errors**:
  - 404: Job not found.

### 5. `GET /status/batch/{batch_id}`

Checks the status of a batch job and its items.

- **Request**: Path parameter `batch_id`.
- **Response**: JSON with batch status and item statuses.
- **Example**:

```python
def check_batch_status(batch_id):
    try:
        data = make_request(
            method="GET",
            url=f"http://localhost:8000/status/batch/{batch_id}"
        )
        print("Batch Status:", data)
        return data
    except Exception as e:
        print(f"Error checking batch status: {e}")

batch_status = check_batch_status("987fcdeb-1234-5678-9012-34567890abcd")
```

- **Expected Response**:

```json
{
  "batch_id": "987fcdeb-...-90abcd",
  "batch_status": {
    "status": "complete",
    "progress": 100,
    "total_items": 2,
    "processed_items": 2
  },
  "items": [
    {
      "item_id": "987fcdeb-..._0",
      "status": "complete",
      "progress": 100,
      "error": null,
      "result": { /* TTSResponse */ }
    },
    { /* More items */ }
  ]
}
```

- **Errors**:
  - 404: Batch not found.

### 6. `GET /audio/{filename}`

Retrieves a generated audio file.

- **Request**: Path parameter `filename`.
- **Response**: Audio file (`audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Example** (Save to file):

```python
def save_audio(filename):
    try:
        response = make_request(
            method="GET",
            url=f"http://localhost:8000/audio/{filename}",
            stream=True
        )
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Audio saved to {filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")

save_audio("123e4567-e89b-12d3-a456-426614174000.wav")
```

- **Note**: Playback example with `playsound`:

```python
from playsound import playsound
playsound("123e4567-e89b-12d3-a456-426614174000.wav")
```

- **Errors**:
  - 404: File not found.

### 7. `DELETE /audio/{filename}`

Deletes a generated audio file.

- **Request**: Path parameter `filename`.
- **Response**: JSON confirming deletion.
- **Example**:

```python
def delete_audio(filename):
    try:
        data = make_request(
            method="DELETE",
            url=f"http://localhost:8000/audio/{filename}"
        )
        print("Audio Deleted:", data)
    except Exception as e:
        print(f"Error deleting audio: {e}")

delete_audio("123e4567-e89b-12d3-a456-426614174000.wav")
```

- **Expected Response**:

```json
{
  "status": "deleted",
  "filename": "123e4567-...-174000.wav"
}
```

- **Errors**:
  - 400: Invalid filename.
  - 404: File not found.
  - 500: Deletion failed.

### 8. `GET /voices`

Lists all available voices.

- **Request**: None.
- **Response**: JSON with voice categories and details.
- **Example**:

```python
def list_voices():
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/voices"
        )
        print("Voices:", data)
        return data
    except Exception as e:
        print(f"Error listing voices: {e}")

voices = list_voices()
```

- **Expected Response**:

```json
{
  "american_female": [
    { "id": "af_heart", "name": "Heart", "emoji": "❤️" },
    { "id": "af_bella", "name": "Bella", "emoji": "🔥" },
    /* More voices */
  ],
  "american_male": [ /* ... */ ],
  "british_female": [ /* ... */ ],
  "british_male": [ /* ... */ ]
}
```

- **Errors**: None expected.

### 9. `GET /voices/preview`

Generates audio previews for all voices. Returns a `batch_id`.

- **Request**: None.
- **Response**: JSON with `batch_id`, `status`, and `total_items`.
- **Example**:

```python
def preview_voices():
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/voices/preview"
        )
        print("Voice Previews:", data)
        return data["batch_id"]
    except Exception as e:
        print(f"Error previewing voices: {e}")

preview_batch_id = preview_voices()
```

- **Expected Response**:

```json
{
  "batch_id": "987fcdeb-...-90abcd",
  "status": "queued",
  "total_items": 28
}
```

- **Errors**: None expected.

### 10. `GET /health`

Checks the API’s health and system status.

- **Request**: None.
- **Response**: JSON with system details.
- **Example**:

```python
def check_health():
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/health"
        )
        print("Health:", data)
        return data
    except Exception as e:
        print(f"Error checking health: {e}")

health = check_health()
```

- **Expected Response**:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": 1745176800,
  "models": { "cpu": "available", "gpu": "available" },
  "model_files": { "model_exists": true, "config_exists": true },
  "disk_space": { "total_gb": 100, "free_gb": 50 },
  "memory": { "cuda": "..." },
  "cuda_available": true,
  "ffmpeg_available": true,
  "active_jobs": 0,
  "cached_files": 5,
  "rate_limiting_enabled": true
}
```

- **Errors**: None expected.

### 11. `DELETE /cleanup`

Deletes audio files older than the specified hours (default: 24).

- **Request**: Query parameter `hours` (optional).
- **Response**: JSON with deletion summary.
- **Example**:

```python
def cleanup_files(hours=24):
    try:
        data = make_request(
            method="DELETE",
            url="http://localhost:8000/cleanup",
            params={"hours": hours}
        )
        print("Cleanup:", data)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

cleanup_files()
```

- **Expected Response**:

```json
{
  "status": "completed",
  "deleted_files": 10,
  "errors": 0
}
```

- **Errors**: None expected.

### 12. `POST /pronunciation`

Adds a custom pronunciation.

- **Request**: Query parameters `word`, `pronunciation`, `language_code` (`a` or `b`).
- **Response**: JSON confirming the addition.
- **Example**:

```python
def add_pronunciation(word, pronunciation, language_code="a"):
    try:
        data = make_request(
            method="POST",
            url="http://localhost:8000/pronunciation",
            params={
                "word": word,
                "pronunciation": pronunciation,
                "language_code": language_code
            }
        )
        print("Pronunciation Added:", data)
    except Exception as e:
        print(f"Error adding pronunciation: {e}")

add_pronunciation("example", "eg-zam-pul", "a")
```

- **Expected Response**:

```json
{
  "status": "success",
  "word": "example",
  "pronunciation": "eg-zam-pul",
  "language": "American English"
}
```

- **Errors**:
  - 400: Invalid input.
  - 500: Processing error.

### 13. `GET /pronunciations`

Lists custom pronunciations.

- **Request**: Query parameter `language_code` (`a` or `b`, default: `a`).
- **Response**: JSON with pronunciations.
- **Example**:

```python
def list_pronunciations(language_code="a"):
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/pronunciations",
            params={"language_code": language_code}
        )
        print("Pronunciations:", data)
        return data
    except Exception as e:
        print(f"Error listing pronunciations: {e}")

pronunciations = list_pronunciations("a")
```

- **Expected Response**:

```json
{
  "language": "American English",
  "pronunciations": {
    "example": "eg-zam-pul",
    "kokoro": "koh-koh-roh"
  }
}
```

- **Errors**:
  - 500: Processing error.

### 14. `DELETE /pronunciations/{word}`

Deletes a custom pronunciation.

- **Request**: Path parameter `word`, query parameter `language_code`.
- **Response**: JSON confirming deletion.
- **Example**:

```python
def delete_pronunciation(word, language_code="a"):
    try:
        data = make_request(
            method="DELETE",
            url=f"http://localhost:8000/pronunciations/{word}",
            params={"language_code": language_code}
        )
        print("Pronunciation Deleted:", data)
    except Exception as e:
        print(f"Error deleting pronunciation: {e}")

delete_pronunciation("example", "a")
```

- **Expected Response**:

```json
{
  "status": "deleted",
  "word": "example",
  "language": "American English"
}
```

- **Errors**:
  - 404: Pronunciation not found.
  - 500: Processing error.

### 15. `POST /preprocess`

Preprocesses text (e.g., expands abbreviations, removes special characters).

- **Request**: JSON with `text`.
- **Response**: JSON with original and processed text.
- **Example**:

```python
def preprocess_text(text):
    try:
        data = make_request(
            method="POST",
            url="http://localhost:8000/preprocess",
            data={"text": text}
        )
        print("Preprocessed Text:", data)
        return data
    except Exception as e:
        print(f"Error preprocessing text: {e}")

preprocessed = preprocess_text("Mr. Smith said: Hello, world! @#$%")
```

- **Expected Response**:

```json
{
  "original_text": "Mr. Smith said: Hello, world! @#$%",
  "processed_text": "mister smith said hello world"
}
```

- **Errors**:
  - 400: Invalid input.

### 16. `GET /metrics`

Exposes Prometheus metrics.

- **Request**: None.
- **Response**: Text in Prometheus format.
- **Example**:

```python
def get_metrics():
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/metrics"
        )
        print("Metrics:", data)
        return data
    except Exception as e:
        print(f"Error getting metrics: {e}")

metrics = get_metrics()
```

- **Expected Response**: Prometheus text (e.g., `tts_requests_total{endpoint="/tts"} 10`).
- **Errors**: None expected.

### 17. `GET /config`

Returns API configuration.

- **Request**: None.
- **Response**: JSON with configuration details.
- **Example**:

```python
def get_config():
    try:
        data = make_request(
            method="GET",
            url="http://localhost:8000/config"
        )
        print("Config:", data)
        return data
    except Exception as e:
        print(f"Error getting config: {e}")

config = get_config()
```

- **Expected Response**:

```json
{
  "sample_rate": 24000,
  "max_char_limit": 5000,
  "supported_formats": ["wav", "mp3", "ogg"],
  "cuda_available": true,
  "rate_limiting_enabled": true
}
```

- **Errors**: None expected.

### 18. `POST /jobs/cancel/{job_id}`

Cancels a running TTS job.

- **Request**: Path parameter `job_id`.
- **Response**: JSON confirming cancellation.
- **Example**:

```python
def cancel_job(job_id):
    try:
        data = make_request(
            method="POST",
            url=f"http://localhost:8000/jobs/cancel/{job_id}"
        )
        print("Job Cancelled:", data)
    except Exception as e:
        print(f"Error cancelling job: {e}")

cancel_job("123e4567-e89b-12d3-a456-426614174000")
```

- **Expected Response**:

```json
{
  "status": "cancelled",
  "job_id": "123e4567-...-174000"
}
```

- **Errors**:
  - 404: Job not found.

## Voices

The API supports 28 voices, categorized by accent and gender. Use the `id` in TTS requests.

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

Retrieve programmatically with `/voices`.

## Features

- **Asynchronous Processing**: `/tts`, `/tts/batch`, and `/voices/preview` queue jobs. Poll `/status` endpoints to track progress.
- **Real-Time Streaming**: `/tts/stream` delivers audio for saving to files (playback requires additional libraries).
- **Audio Formats**: WAV, MP3, OGG (via `ffmpeg`). Specify in `format` field.
- **Custom Pronunciations**: Manage via `/pronunciation` endpoints. Affects subsequent TTS requests.
- **Rate-Limiting**: 10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`. Handle 429 errors:

```python
import time

def retry_on_rate_limit(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "HTTP 429" in str(e):
                time.sleep(2 ** i)  # Exponential backoff
                continue
            raise
    raise Exception("Max retries exceeded")

request_id = retry_on_rate_limit(synthesize_text)
```

- **Prometheus Metrics**: Monitor via `/metrics` (parse text for custom dashboards).
- **CUDA Support**: Transparent to clients; GPU used if `use_gpu: true` and available.
- **Text Preprocessing**: Use `/preprocess` to clean text before synthesis.
- **Audio Caching**: Improves `/audio` performance (1-hour TTL).
- **Health Monitoring**: Check `/health` for server status.
- **Job Cancellation**: Use `/jobs/cancel` to stop jobs.
- **CORS**: Enabled for all origins (`*`). Irrelevant for `requests` unless proxied.

## Setup Instructions

### API Server

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

Or Docker:

```bash
docker run -d -p 6379:6379 redis
```

- Verify:

```bash
redis-cli ping
```

- Expected: `PONG`
- Disable rate-limiting if needed:

```bash
export ENABLE_RATE_LIMITING="false"
```

3. **Update File Paths**:

- In `main.py`, set `MODEL_PATH` and `CONFIG_PATH`:

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

- Standalone:

```bash
python main.py
```

- Default: `http://localhost:8000`. Change port:

```bash
export PORT=8080
python main.py
```

5. **Verify**:

```bash
curl -X GET "http://localhost:8000/health"
```

### Python Client

1. **Install Dependencies**:

```bash
pip install requests
```

2. **Create Script**:

- Save examples in a file (e.g., `client.py`).
- Example:

```python
import requests
import json

def make_request(method, url, data=None, params=None, stream=False):
    with requests.Session() as session:
        try:
            response = session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                stream=stream,
                headers={"Content-Type": "application/json"} if data else None
            )
            response.raise_for_status()
            if stream:
                return response
            if response.headers.get("Content-Type", "").startswith("application/json"):
                return response.json()
            return response.text
        except requests.HTTPError as e:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except json.JSONDecodeError:
                error_detail = response.text or "Unknown error"
            raise Exception(f"HTTP {response.status_code}: {error_detail}") from e
        except requests.RequestException as e:
            raise Exception(f"Request failed: {str(e)}") from e

def test_api():
    try:
        data = make_request(method="GET", url="http://localhost:8000/voices")
        print("Voices:", data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
```

3. **Run**:

```bash
python client.py
```

## Example Workflow

1. **List Voices**:

```python
voices = list_voices()
voice_id = voices["american_female"][0]["id"]  # e.g., "af_heart"
```

2. **Synthesize Text**:

```python
request_id = synthesize_text()
```

3. **Poll Status**:

```python
import time

def poll_status(job_id):
    while True:
        status = check_job_status(job_id)
        if status["status"] not in ["queued", "processing"]:
            return status
        time.sleep(1)

status = poll_status(request_id)
```

4. **Save and Play Audio** (if complete):

```python
if status["status"] == "complete":
    filename = status["result"]["audio_url"].split("/")[-1]
    save_audio(filename)
    from playsound import playsound
    playsound(filename)
```

## Alternatives to `requests`

For advanced use cases, consider:

- **httpx**: Async support for high-performance clients.

```bash
pip install httpx
```

```python
import httpx
import asyncio

async def test_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/voices")
        print(response.json())

asyncio.run(test_api())
```

- **aiohttp**: Another async HTTP client.

```bash
pip install aiohttp
```

These libraries are useful for asynchronous or high-concurrency applications.

## Limitations and Known Issues

- **RNN Dropout Warning**: Server-side warning (`dropout=0.2 and num_layers=1`). Fix in `config.json`:

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

- **WeightNorm Deprecation**: Update `kokoro`:

```bash
pip install --upgrade kokoro
```

- **Rate-Limiting**: Handle 429 errors with retries (see Features).
- **Streaming Limitations**: `requests` supports streaming to files; playback requires `playsound` or similar.
- **File Paths**: Server-side errors if `MODEL_PATH` or `CONFIG_PATH` is incorrect.

## Troubleshooting

- **HTTP 429 (Rate Limit)**:
  - Implement retry logic or disable rate-limiting:

```bash
export ENABLE_RATE_LIMITING="false"
```

- **HTTP 404 (File/Job Not Found)**:
  - Verify `job_id` or `filename`:

```python
print("Job ID:", request_id)
```

- **No Audio Output**:
  - Check job status:

```python
check_job_status(request_id)
```

  - Ensure `ffmpeg` is installed on the server:

```bash
ffmpeg -version
```

- **Server Errors (500)**:
  - Check server logs for `MODEL_PATH`, `CONFIG_PATH`, or Redis issues.
  - Validate `config.json`:

```bash
python -m json.tool /path/to/config.json
```

- **Request Errors**:
  - Ensure `requests` is installed:

```bash
pip show requests
```

  - Check Python version:

```bash
python --version
```

For support, share server logs, `config.json`, or client-side errors.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025