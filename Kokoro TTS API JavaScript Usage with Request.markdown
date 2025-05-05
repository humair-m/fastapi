# Kokoro TTS API JavaScript Usage with Request Documentation

## Overview

The Kokoro TTS API is a FastAPI-based text-to-speech (TTS) service that converts text into high-quality audio using the Kokoro model. It supports 28 voices (American and British English, male and female), multiple audio formats (WAV, MP3, OGG), and features like custom pronunciations, batch processing, and real-time streaming. The API is designed for scalability with Redis-based rate-limiting, Prometheus metrics, and CUDA acceleration. It runs on `http://localhost:8000` by default and supports CORS for cross-origin requests.

This documentation is tailored for JavaScript developers using the `request` library in a Node.js environment to interact with the API’s 18 endpoints. Each endpoint is demonstrated with `request`-based code, including request construction, response handling, and error management. The documentation covers all voices, features, and setup steps, ensuring integration into Node.js applications. Note that the `request` library is deprecated; consider `node-fetch` or `axios` for new projects (see Alternatives section).

## Prerequisites

- **Node.js**: v18+ recommended (tested with v18).

- **API Server**: Running instance of the Kokoro TTS API (see Setup Instructions).

- **CORS**: The API enables CORS (`allow_origins=["*"]`), but since `request` is Node.js-based, CORS is typically not a concern unless proxied through a browser.

- **Dependencies**:

  - Install `request`:

    ```bash
    npm install request
    ```

  - Optional (for streaming/file handling):

    ```bash
    npm install fs
    ```

**Note**: The `request` library is deprecated and unmaintained. For production, use `node-fetch` or `axios`. This documentation uses `request` as per the user’s request.

## Endpoints

Below are the API’s 18 endpoints, each with a Node.js example using the `request` library. All examples assume the API runs at `http://localhost:8000`. Responses are JSON unless specified (e.g., audio streams). Error handling checks for HTTP status codes and parses error messages. Since `request` uses callbacks, examples wrap them in Promises for consistency with modern JavaScript.

### Helper Function

To simplify examples, a Promise-based wrapper for `request` is used:

```javascript
const request = require('request');

function makeRequest(options) {
  return new Promise((resolve, reject) => {
    request(options, (error, response, body) => {
      if (error) {
        return reject(error);
      }
      if (response.statusCode < 200 || response.statusCode >= 300) {
        const errorMsg = body && typeof body === 'object' && body.detail
          ? body.detail
          : 'Unknown error';
        return reject(new Error(`HTTP ${response.statusCode}: ${errorMsg}`));
      }
      resolve(typeof body === 'string' ? JSON.parse(body) : body);
    });
  });
}
```

### 1. `POST /tts`

Initiates an asynchronous TTS job. Returns a `request_id` to track status. Audio is available via `/audio/{filename}` once complete.

- **Request**:
  - Body: JSON with text, voice, speed, format, etc.
  - Content-Type: `application/json`
- **Response**: JSON with `request_id`, `status`, and `null` fields for `audio_url`, `duration`, `tokens`.
- **Example**:

```javascript
async function synthesizeText() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/tts',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      json: {
        text: 'Hello, this is a test of the Kokoro TTS API.',
        voice: 'af_heart',
        speed: 1.0,
        use_gpu: true,
        return_tokens: false,
        format: 'wav',
        pronunciations: { kokoro: 'koh-koh-roh' }
      }
    });
    console.log('TTS Job:', data);
    return data.request_id;
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

Streams audio in real-time. Returns an audio stream, not JSON.

- **Request**: Same as `/tts`.
- **Response**: Audio stream (`audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Example** (Save to file):

```javascript
const fs = require('fs');

function streamAudio() {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream('output.wav');
    const req = request({
      url: 'http://localhost:8000/tts/stream',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      json: {
        text: 'Streaming audio test.',
        voice: 'am_michael',
        speed: 1.0,
        use_gpu: true,
        format: 'wav'
      }
    });

    req.on('response', (response) => {
      if (response.statusCode !== 200) {
        let errorBody = '';
        response.on('data', (chunk) => errorBody += chunk);
        response.on('end', () => {
          const errorMsg = errorBody && JSON.parse(errorBody).detail || 'Unknown error';
          reject(new Error(`HTTP ${response.statusCode}: ${errorMsg}`));
        });
        return;
      }
      req.pipe(output);
    });

    output.on('finish', () => {
      console.log('Audio saved');
      resolve();
    });

    req.on('error', reject);
    output.on('error', reject);
  });
}

streamAudio().catch((error) => console.error('Error streaming audio:', error.message));
```

- **Note**: Streaming to playback (e.g., `Audio` API) is not supported in Node.js with `request`. Use `node-fetch` or a browser for playback.
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

```javascript
async function batchSynthesize() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/tts/batch',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      json: {
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
      }
    });
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
- **Response**: JSON with job status, progress, and result.
- **Example**:

```javascript
async function checkJobStatus(jobId) {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/status/${jobId}`,
      method: 'GET'
    });
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
- **Example**:

```javascript
async function checkBatchStatus(batchId) {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/status/batch/${batchId}`,
      method: 'GET'
    });
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
- **Example** (Save to file):

```javascript
const fs = require('fs');

function saveAudio(filename) {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(filename);
    const req = request({
      url: `http://localhost:8000/audio/${filename}`,
      method: 'GET'
    });

    req.on('response', (response) => {
      if (response.statusCode !== 200) {
        let errorBody = '';
        response.on('data', (chunk) => errorBody += chunk);
        response.on('end', () => {
          const errorMsg = errorBody && JSON.parse(errorBody).detail || 'Unknown error';
          reject(new Error(`HTTP ${response.statusCode}: ${errorMsg}`));
        });
        return;
      }
      req.pipe(output);
    });

    output.on('finish', () => {
      console.log('Audio saved');
      resolve();
    });

    req.on('error', reject);
    output.on('error', reject);
  });
}

saveAudio('123e4567-e89b-12d3-a456-426614174000.wav')
  .catch((error) => console.error('Error saving audio:', error.message));
```

- **Note**: Playback requires a separate library (e.g., `play-sound`) in Node.js.
- **Errors**:
  - 404: File not found.

### 7. `DELETE /audio/{filename}`

Deletes a generated audio file.

- **Request**: Path parameter `filename`.
- **Response**: JSON confirming deletion.
- **Example**:

```javascript
async function deleteAudio(filename) {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/audio/${filename}`,
      method: 'DELETE'
    });
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
- **Example**:

```javascript
async function listVoices() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/voices',
      method: 'GET'
    });
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
- **Example**:

```javascript
async function previewVoices() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/voices/preview',
      method: 'GET'
    });
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
- **Example**:

```javascript
async function checkHealth() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/health',
      method: 'GET'
    });
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
- **Example**:

```javascript
async function cleanupFiles(hours = 24) {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/cleanup?hours=${hours}`,
      method: 'DELETE'
    });
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
- **Example**:

```javascript
async function addPronunciation(word, pronunciation, languageCode = 'a') {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/pronunciation?word=${encodeURIComponent(word)}&pronunciation=${encodeURIComponent(pronunciation)}&language_code=${languageCode}`,
      method: 'POST'
    });
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
- **Example**:

```javascript
async function listPronunciations(languageCode = 'a') {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/pronunciations?language_code=${languageCode}`,
      method: 'GET'
    });
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
- **Example**:

```javascript
async function deletePronunciation(word, languageCode = 'a') {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/pronunciations/${encodeURIComponent(word)}?language_code=${languageCode}`,
      method: 'DELETE'
    });
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
- **Example**:

```javascript
async function preprocessText(text) {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/preprocess',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      json: { text }
    });
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
- **Example**:

```javascript
async function getMetrics() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/metrics',
      method: 'GET'
    });
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
- **Example**:

```javascript
async function getConfig() {
  try {
    const data = await makeRequest({
      url: 'http://localhost:8000/config',
      method: 'GET'
    });
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
- **Example**:

```javascript
async function cancelJob(jobId) {
  try {
    const data = await makeRequest({
      url: `http://localhost:8000/jobs/cancel/${jobId}`,
      method: 'POST'
    });
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
- **Real-Time Streaming**: `/tts/stream` delivers audio for saving to files in Node.js (playback requires additional libraries).
- **Audio Formats**: WAV, MP3, OGG (via `ffmpeg`). Specify in `format` field.
- **Custom Pronunciations**: Manage via `/pronunciation` endpoints. Affects subsequent TTS requests.
- **Rate-Limiting**: 10/min for `/tts`, 5/min for `/tts/stream` and `/tts/batch`. Handle 429 errors:

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

- **Prometheus Metrics**: Monitor via `/metrics` (parse text for custom dashboards).
- **CUDA Support**: Transparent to clients; GPU used if `use_gpu: true` and available.
- **Text Preprocessing**: Use `/preprocess` to clean text before synthesis.
- **Audio Caching**: Improves `/audio` performance (1-hour TTL).
- **Health Monitoring**: Check `/health` for server status.
- **Job Cancellation**: Use `/jobs/cancel` to stop jobs.
- **CORS**: Enabled for all origins (`*`). Irrelevant for Node.js unless proxied.

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

### Node.js Client

1. **Install Dependencies**:

```bash
npm install request
```

2. **Create Script**:

- Save examples in a file (e.g., `app.js`).
- Example:

```javascript
const request = require('request');

function makeRequest(options) {
  return new Promise((resolve, reject) => {
    request(options, (error, response, body) => {
      if (error) return reject(error);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        const errorMsg = body && typeof body === 'object' && body.detail ? body.detail : 'Unknown error';
        return reject(new Error(`HTTP ${response.statusCode}: ${errorMsg}`));
      }
      resolve(typeof body === 'string' ? JSON.parse(body) : body);
    });
  });
}

async function testApi() {
  try {
    const data = await makeRequest({ url: 'http://localhost:8000/voices', method: 'GET' });
    console.log('Voices:', data);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

testApi();
```

3. **Run**:

```bash
node app.js
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
async function pollStatus(requestId) {
  let status;
  do {
    status = await checkJobStatus(requestId);
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s
  } while (status.status === 'queued' || status.status === 'processing');
  return status;
}
const status = await pollStatus(requestId);
```

4. **Save Audio** (if complete):

```javascript
if (status.status === 'complete') {
  await saveAudio(status.result.audio_url.split('/').pop());
}
```

## Alternatives to `request`

The `request` library is deprecated. For modern Node.js projects, use:

- **node-fetch**:

```bash
npm install node-fetch
```

```javascript
const fetch = require('node-fetch');
async function testApi() {
  const response = await fetch('http://localhost:8000/voices');
  const data = await response.json();
  console.log(data);
}
```

- **axios**:

```bash
npm install axios
```

```javascript
const axios = require('axios');
async function testApi() {
  const { data } = await axios.get('http://localhost:8000/voices');
  console.log(data);
}
```

These libraries support Promises natively and are maintained.

## Limitations and Known Issues

- **Request Deprecation**: The `request` library is unmaintained. Use `node-fetch` or `axios` for new projects.
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
- **Streaming Limitations**: `request` supports streaming to files but not direct playback in Node.js. Use `play-sound` for playback:

```bash
npm install play-sound
```

```javascript
const player = require('play-sound')();
player.play('output.wav', (err) => {
  if (err) console.error('Playback error:', err);
});
```

- **File Paths**: Server-side errors if `MODEL_PATH` or `CONFIG_PATH` is incorrect.

## Troubleshooting

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

- **Request Errors**:
- Ensure `request` is installed:

```bash
npm list request
```

- Check Node.js version:

```bash
node -v
```

For support, share server logs, `config.json`, or client-side errors.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025