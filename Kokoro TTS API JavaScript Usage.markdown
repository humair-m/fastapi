# Kokoro TTS API JavaScript Usage Documentation

## Overview

The Kokoro TTS API is a FastAPI-based text-to-speech (TTS) service that converts text into high-quality audio using the Kokoro model. It supports multiple voices (American and British English, male and female), audio formats (WAV, MP3, OGG), and features like custom pronunciations, batch processing, and real-time streaming. The API is designed for scalability with Redis-based rate-limiting, Prometheus metrics, and CUDA acceleration. It runs on `http://localhost:8000` by default and supports CORS for cross-origin JavaScript requests.

This documentation is tailored for JavaScript developers, providing examples of how to interact with the API’s 18 endpoints using the `fetch` API (available in modern browsers and Node.js). Each endpoint is demonstrated with JavaScript code, including request construction, response handling, and error management. The documentation covers all voices, features, and setup steps, ensuring you can integrate the API into web or Node.js applications.

## Prerequisites

- **JavaScript Environment**: Modern browser (e.g., Chrome, Firefox) or Node.js (v18+ for `fetch` support).

- **API Server**: Running instance of the Kokoro TTS API (see Setup Instructions).

- **CORS**: The API enables CORS (`allow_origins=["*"]`), allowing requests from any origin. Ensure your client’s domain is permitted if you modify CORS settings.

- **Dependencies** (for Node.js):

  ```bash
  npm install node-fetch  # If using Node.js <18
  ```

## Endpoints

Below are the API’s 18 endpoints, each with a JavaScript example using `fetch`. All examples assume the API runs at `http://localhost:8000`. Responses are JSON unless specified (e.g., audio streams). Error handling checks for HTTP status codes and parses error messages.

### 1. `POST /tts`

Initiates an asynchronous TTS job. Returns a `request_id` to track status. Audio is available via `/audio/{filename}` once complete.

- **Request**:

  - Body: JSON with text, voice, speed, format, etc.
  - Content-Type: `application/json`

- **Response**: JSON with `request_id`, `status`, and `null` fields for `audio_url`, `duration`, `tokens`.

- **JavaScript Example**:

  ```javascript
  async function synthesizeText() {
    try {
      const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: 'Hello, this is a test of the Kokoro TTS API.',
          voice: 'af_heart',
          speed: 1.0,
          use_gpu: true,
          return_tokens: false,
          format: 'wav',
          pronunciations: { kokoro: 'koh-koh-roh' }
        })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('TTS Job:', data);
      return data.request_id; // Use to check status
    } catch (error) {
      console.error('Error synthesizing text:', error.message);
    }
  }
  synthesizeText();
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

Streams audio in real-time for immediate playback. Returns an audio stream, not JSON.

- **Request**: Same as `/tts`.

- **Response**: Audio stream (`audio/wav`, `audio/mpeg`, or `audio/ogg`).

- **JavaScript Example** (Browser):

  ```javascript
  async function streamAudio() {
    try {
      const response = await fetch('http://localhost:8000/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: 'Streaming audio test.',
          voice: 'am_michael',
          speed: 1.0,
          use_gpu: true,
          format: 'mp3'
        })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error('Error streaming audio:', error.message);
    }
  }
  streamAudio();
  ```

- **Note**: For Node.js, save the stream to a file:

  ```javascript
  const fs = require('fs');
  const fetch = require('node-fetch');
  async function streamAudioToFile() {
    const response = await fetch('http://localhost:8000/tts/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Streaming audio test.',
        voice: 'am_michael',
        speed: 1.0,
        use_gpu: true,
        format: 'wav'
      })
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
    }
    const dest = fs.createWriteStream('output.wav');
    response.body.pipe(dest);
    dest.on('finish', () => console.log('Audio saved'));
    dest.on('error', (err) => console.error('Error saving audio:', err));
  }
  streamAudioToFile();
  ```

- **Errors**:

  - 400: Invalid input.
  - 429: Rate limit exceeded (5/min).
  - 500: Processing error.

### 3. `POST /tts/batch`

Processes multiple TTS requests in a batch. Returns a `batch_id` for tracking.

- **Request**:

  - Body: JSON array of TTS requests.

- **Response**: JSON with `batch_id`, `status`, and `total_items`.

- **JavaScript Example**:

  ```javascript
  async function batchSynthesize() {
    try {
      const response = await fetch('http://localhost:8000/tts/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          items: [
            {
              text: 'First test sentence.',
              voice: 'af_bella',
              speed: 1.2,
              format: 'mp3'
            },
            {
              text: 'Second test sentence.',
              voice: 'bm_george',
              speed: 0.8,
              format: 'wav'
            }
          ]
        })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Batch Job:', data);
      return data.batch_id;
    } catch (error) {
      console.error('Error in batch synthesis:', error.message);
    }
  }
  batchSynthesize();
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

- **Response**: JSON with job status, progress, and result (if complete).

- **JavaScript Example**:

  ```javascript
  async function checkJobStatus(jobId) {
    try {
      const response = await fetch(`http://localhost:8000/status/${jobId}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Job Status:', data);
      return data;
    } catch (error) {
      console.error('Error checking job status:', error.message);
    }
  }
  checkJobStatus('123e4567-e89b-12d3-a456-426614174000');
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

- **JavaScript Example**:

  ```javascript
  async function checkBatchStatus(batchId) {
    try {
      const response = await fetch(`http://localhost:8000/status/batch/${batchId}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Batch Status:', data);
      return data;
    } catch (error) {
      console.error('Error checking batch status:', error.message);
    }
  }
  checkBatchStatus('987fcdeb-1234-5678-9012-34567890abcd');
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

- **JavaScript Example** (Browser):

  ```javascript
  async function playAudio(filename) {
    try {
      const response = await fetch(`http://localhost:8000/audio/${filename}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error('Error playing audio:', error.message);
    }
  }
  playAudio('123e4567-e89b-12d3-a456-426614174000.wav');
  ```

- **Node.js Example** (Save to file):

  ```javascript
  const fs = require('fs');
  const fetch = require('node-fetch');
  async function saveAudio(filename) {
    const response = await fetch(`http://localhost:8000/audio/${filename}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
    }
    const dest = fs.createWriteStream(filename);
    response.body.pipe(dest);
    dest.on('finish', () => console.log('Audio saved'));
    dest.on('error', (err) => console.error('Error saving audio:', err));
  }
  saveAudio('123e4567-e89b-12d3-a456-426614174000.wav');
  ```

- **Errors**:

  - 404: File not found.

### 7. `DELETE /audio/{filename}`

Deletes a generated audio file.

- **Request**: Path parameter `filename`.

- **Response**: JSON confirming deletion.

- **JavaScript Example**:

  ```javascript
  async function deleteAudio(filename) {
    try {
      const response = await fetch(`http://localhost:8000/audio/${filename}`, {
        method: 'DELETE'
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Audio Deleted:', data);
    } catch (error) {
      console.error('Error deleting audio:', error.message);
    }
  }
  deleteAudio('123e4567-e89b-12d3-a456-426614174000.wav');
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

- **JavaScript Example**:

  ```javascript
  async function listVoices() {
    try {
      const response = await fetch('http://localhost:8000/voices');
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Voices:', data);
      return data;
    } catch (error) {
      console.error('Error listing voices:', error.message);
    }
  }
  listVoices();
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

- **JavaScript Example**:

  ```javascript
  async function previewVoices() {
    try {
      const response = await fetch('http://localhost:8000/voices/preview');
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Voice Previews:', data);
      return data.batch_id;
    } catch (error) {
      console.error('Error previewing voices:', error.message);
    }
  }
  previewVoices();
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

- **JavaScript Example**:

  ```javascript
  async function checkHealth() {
    try {
      const response = await fetch('http://localhost:8000/health');
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Health:', data);
      return data;
    } catch (error) {
      console.error('Error checking health:', error.message);
    }
  }
  checkHealth();
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

- **JavaScript Example**:

  ```javascript
  async function cleanupFiles(hours = 24) {
    try {
      const response = await fetch(`http://localhost:8000/cleanup?hours=${hours}`, {
        method: 'DELETE'
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Cleanup:', data);
    } catch (error) {
      console.error('Error cleaning up files:', error.message);
    }
  }
  cleanupFiles();
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

- **JavaScript Example**:

  ```javascript
  async function addPronunciation(word, pronunciation, languageCode = 'a') {
    try {
      const response = await fetch(
        `http://localhost:8000/pronunciation?word=${encodeURIComponent(word)}&pronunciation=${encodeURIComponent(pronunciation)}&language_code=${languageCode}`,
        { method: 'POST' }
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Pronunciation Added:', data);
    } catch (error) {
      console.error('Error adding pronunciation:', error.message);
    }
  }
  addPronunciation('example', 'eg-zam-pul', 'a');
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

- **JavaScript Example**:

  ```javascript
  async function listPronunciations(languageCode = 'a') {
    try {
      const response = await fetch(`http://localhost:8000/pronunciations?language_code=${languageCode}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Pronunciations:', data);
      return data;
    } catch (error) {
      console.error('Error listing pronunciations:', error.message);
    }
  }
  listPronunciations('a');
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

- **JavaScript Example**:

  ```javascript
  async function deletePronunciation(word, languageCode = 'a') {
    try {
      const response = await fetch(
        `http://localhost:8000/pronunciations/${encodeURIComponent(word)}?language_code=${languageCode}`,
        { method: 'DELETE' }
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Pronunciation Deleted:', data);
    } catch (error) {
      console.error('Error deleting pronunciation:', error.message);
    }
  }
  deletePronunciation('example', 'a');
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

- **JavaScript Example**:

  ```javascript
  async function preprocessText(text) {
    try {
      const response = await fetch('http://localhost:8000/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Preprocessed Text:', data);
      return data;
    } catch (error) {
      console.error('Error preprocessing text:', error.message);
    }
  }
  preprocessText('Mr. Smith said: Hello, world! @#$%');
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

- **JavaScript Example**:

  ```javascript
  async function getMetrics() {
    try {
      const response = await fetch('http://localhost:8000/metrics');
      if (!response.ok) {
        const error = await response.text();
        throw new Error(`HTTP ${response.status}: ${error}`);
      }
      const data = await response.text();
      console.log('Metrics:', data);
      return data;
    } catch (error) {
      console.error('Error getting metrics:', error.message);
    }
  }
  getMetrics();
  ```

- **Expected Response**: Prometheus text (e.g., `tts_requests_total{endpoint="/tts"} 10`).

- **Errors**: None expected.

### 17. `GET /config`

Returns API configuration.

- **Request**: None.

- **Response**: JSON with configuration details.

- **JavaScript Example**:

  ```javascript
  async function getConfig() {
    try {
      const response = await fetch('http://localhost:8000/config');
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Config:', data);
      return data;
    } catch (error) {
      console.error('Error getting config:', error.message);
    }
  }
  getConfig();
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

- **JavaScript Example**:

  ```javascript
  async function cancelJob(jobId) {
    try {
      const response = await fetch(`http://localhost:8000/jobs/cancel/${jobId}`, {
        method: 'POST'
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(`HTTP ${response.status}: ${error.detail || 'Unknown error'}`);
      }
      const data = await response.json();
      console.log('Job Cancelled:', data);
    } catch (error) {
      console.error('Error cancelling job:', error.message);
    }
  }
  cancelJob('123e4567-e89b-12d3-a456-426614174000');
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

- **Real-Time Streaming**: `/tts/stream` delivers audio for immediate playback (use `Blob` in browsers).

- **Audio Formats**: WAV, MP3, OGG (via `ffmpeg`). Specify in `format` field.

- **Custom Pronunciations**: Manage via `/pronunciation` endpoints. Affects subsequent TTS requests.

- **Rate-Limiting**: 10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`. Handle 429 errors:

  ```javascript
  if (response.status === 429) {
    console.warn('Rate limit exceeded. Retry later.');
  }
  ```

- **Prometheus Metrics**: Monitor via `/metrics` (parse text for custom dashboards).

- **CUDA Support**: Transparent to clients; GPU used if `use_gpu: true` and available.

- **Jupyter Compatibility**: Server-side feature; no client impact.

- **Text Preprocessing**: Use `/preprocess` to clean text before synthesis.

- **Audio Caching**: Improves `/audio` performance (1-hour TTL).

- **Health Monitoring**: Check `/health` for server status.

- **Job Cancellation**: Use `/jobs/cancel` to stop jobs.

- **CORS**: Enabled for all origins (`*`). Modify `CORSMiddleware` if needed.

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

   - Jupyter:

     1. Copy `main.py` code into a cell.
     2. Update paths.
     3. Run cell.

   - Default: `http://localhost:8000`. Change port:

     ```bash
     export PORT=8080
     python main.py
     ```

5. **Verify**:

   ```bash
   curl -X GET "http://localhost:8000/health"
   ```

### JavaScript Client

- **Browser**: Use native `fetch`. Test in Chrome/Firefox DevTools console.

- **Node.js**:

  - Install `node-fetch` for Node.js &lt;18:

    ```bash
    npm install node-fetch
    ```

  - Example script:

    ```javascript
    const fetch = require('node-fetch');
    async function testApi() {
      const response = await fetch('http://localhost:8000/voices');
      const data = await response.json();
      console.log(data);
    }
    testApi();
    ```

  - Run:

    ```bash
    node script.js
    ```

## Example Workflow

1. **List Voices**:

   ```javascript
   const voices = await listVoices();
   const voiceId = voices.american_female[0].id; // e.g., "af_heart"
   ```

2. **Synthesize Text**:

   ```javascript
   const requestId = await synthesizeText();
   ```

3. **Poll Status**:

   ```javascript
   let status;
   do {
     status = await checkJobStatus(requestId);
     await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s
   } while (status.status === 'queued' || status.status === 'processing');
   ```

4. **Play Audio** (if complete):

   ```javascript
   if (status.status === 'complete') {
     await playAudio(status.result.audio_url.split('/').pop());
   }
   ```

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

- **Rate-Limiting**: Handle 429 errors with retries:

  ```javascript
  async function retryOnRateLimit(fn, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (error.message.includes('429')) {
          await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
          continue;
        }
        throw error;
      }
    }
    throw new Error('Max retries exceeded');
  }
  retryOnRateLimit(synthesizeText);
  ```

- **CORS**: If CORS errors occur, verify `CORSMiddleware` settings in `main.py`.

- **File Paths**: Server-side errors if `MODEL_PATH` or `CONFIG_PATH` is incorrect.

- **Streaming in Node.js**: Requires manual stream handling (e.g., `fs.createWriteStream`).

## Troubleshooting

- **CORS Errors**:

  - Check browser console for CORS messages.
  - Ensure `app.add_middleware(CORSMiddleware, allow_origins=["*"])` in `main.py`.

- **HTTP 429 (Rate Limit)**:

  - Implement retry logic or disable rate-limiting:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **HTTP 404 (File/Job Not Found)**:

  - Verify `job_id` or `filename`:

    ```javascript
    console.log('Job ID:', requestId);
    ```

- **No Audio Output**:

  - Check job status:

    ```javascript
    checkJobStatus(requestId);
    ```

  - Ensure `ffmpeg` is installed on the server.

- **Server Errors (500)**:

  - Check server logs for `MODEL_PATH`, `CONFIG_PATH`, or Redis issues.

  - Validate `config.json`:

    ```bash
    python -m json.tool /path/to/config.json
    ```

For support, share server logs, `config.json`, or client-side errors.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025