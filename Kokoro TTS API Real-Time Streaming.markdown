# Kokoro TTS API Real-Time Streaming Documentation

## Overview

The Kokoro TTS API, built with FastAPI, provides a `/tts/stream` endpoint for real-time text-to-speech (TTS) audio streaming. This endpoint converts text into audio (WAV, MP3, or OGG) and streams it immediately, ideal for applic ations requiring low-latency playback, such as voice assistants or live narration. The API supports 28 voices (American and British English, male and female), CUDA acceleration, and Redis-based rate-limiting (5 requests/min). It runs on `http://localhost:8000` by default and includes CORS support (`allow_origins=["*"]`) for browser-based clients.

This documentation demonstrates real-time streaming with the `/tts/stream` endpoint using cURL, JavaScript (`fetch` for browser and Node.js), and Python (`requests` with `soundfile` and `sounddevice`). Each example shows how to send a request, stream the audio, and either save or play it. The documentation covers setup, supported voices, error handling (e.g., 429 rate-limit errors), and limitations (e.g., server-side RNN dropout warning).

## Prerequisites

- **API Server**: Running instance of the Kokoro TTS API (see Setup Instructions).
- **CORS**: Enabled for all origins (`*`), critical for JavaScript browser clients.
- **Dependencies**:
  - **cURL**: Installed on your system (standard on most Linux/macOS, available for Windows).
  - **JavaScript**:
    - Browser: Modern browser (e.g., Chrome, Firefox) with `fetch` support.
    - Node.js: v18+ with `node-fetch` for Node.js environments:

      ```bash
      npm install node-fetch
      ```
  - **Python**: 3.12+ with required libraries:

    ```bash
    pip install requests soundfile sounddevice numpy
    ```

## `/tts/stream` Endpoint

- **Method**: `POST`
- **URL**: `http://localhost:8000/tts/stream`
- **Request**:
  - Body: JSON with `text`, `voice`, `speed`, `use_gpu`, `format`, and optional `pronunciations` and `return_tokens`.
  - Content-Type: `application/json`
- **Response**: Audio stream (`audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Rate Limit**: 5 requests/min.
- **Errors**:
  - 400: Invalid input (e.g., empty text, invalid voice).
  - 429: Rate limit exceeded.
  - 500: Server error (e.g., model or `ffmpeg` issues).

### Supported Voices

The API supports 28 voices, categorized by accent and gender. Use the `id` in requests (retrieve via `GET /voices`):

- **American Female**: `af_heart`, `af_bella`, `af_nicole`, `af_aoede`, `af_kore`, `af_sarah`, `af_nova`, `af_sky`, `af_alloy`, `af_jessica`, `af_river`
- **American Male**: `am_michael`, `am_fenrir`, `am_puck`, `am_echo`, `am_eric`, `am_liam`, `am_onyx`, `am_santa`, `am_adam`
- **British Female**: `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`
- **British Male**: `bm_george`, `bm_fable`, `bm_lewis`, `bm_daniel`

### Supported Formats

- WAV (`audio/wav`)
- MP3 (`audio/mpeg`)
- OGG (`audio/ogg`)

WAV is used in examples for compatibility with real-time playback libraries.

## Examples

Below are examples for streaming audio from `/tts/stream` in cURL, JavaScript, and Python. Each example uses the same request parameters for consistency:

- `text`: "Real-time text-to-speech is now speaking as it generates audio."
- `voice`: `am_michael`
- `speed`: 1.0
- `use_gpu`: `true`
- `format`: `wav`

### cURL

cURL is ideal for testing or saving streamed audio to a file.

- **Example** (Save to file):

```bash
curl -X POST "http://localhost:8000/tts/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Real-time text-to-speech is now speaking as it generates audio.", "voice": "am_michael", "speed": 1.0, "use_gpu": true, "format": "wav"}' \
  -o output.wav
```

- **Playback** (Linux/macOS):

  ```bash
  aplay output.wav
  ```

  Or (macOS):

  ```bash
  afplay output.wav
  ```

- **Windows**: Use a media player (e.g., VLC) or PowerShell:

  ```powershell
  Start-Process output.wav
  ```

- **Expected Output**:

  - Saves `output.wav` if successful.
  - Errors (e.g., `HTTP/1.1 429 Too Many Requests`) if rate limit is exceeded.

- **Error Handling**:

  ```bash
  curl -X POST "http://localhost:8000/tts/stream" \
    -H "Content-Type: application/json" \
    -d '{"text": "Real-time text-to-speech is now speaking as it generates audio.", "voice": "am_michael", "speed": 1.0, "use_gpu": true, "format": "wav"}' \
    -o output.wav -w "%{http_code}" > http_status.txt
  if [ "$(cat http_status.txt)" -eq 200 ]; then
    echo "Streaming successful, saved to output.wav"
  else
    echo "Error: HTTP $(cat http_status.txt)"
    cat output.wav  # May contain JSON error details
  fi
  ```

### JavaScript (Browser with `fetch`)

In a browser, `fetch` streams audio and plays it using the `Audio` API.

- **Example**:

```javascript
async function streamAudio() {
  try {
    const response = await fetch("http://localhost:8000/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "Real-time text-to-speech is now speaking as it generates audio.",
        voice: "am_michael",
        speed: 1.0,
        use_gpu: true,
        format: "wav"
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.detail || "Unknown error"}`);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    console.log("Playing audio...");
    audio.play();
    audio.onended = () => {
      console.log("Audio playback finished");
      URL.revokeObjectURL(audioUrl); // Clean up
    };
  } catch (error) {
    console.error("Error streaming audio:", error.message);
  }
}

streamAudio();
```

- **Usage**:

  - Open browser DevTools (F12), paste into the console, and run.
  - Or include in an HTML file:

    ```html
    <!DOCTYPE html>
    <html>
    <body>
      <button onclick="streamAudio()">Play Audio</button>
      <script>
        // Paste the above JavaScript code here
      </script>
    </body>
    </html>
    ```

- **Expected Output**:

  - Plays audio directly in the browser.
  - Logs "Playing audio..." and "Audio playback finished" to the console.

- **Error Handling**:

  - Handles 429 (rate limit) with a retry:

    ```javascript
    async function retryOnRateLimit(fn, maxRetries = 3) {
      for (let i = 0; i < maxRetries; i++) {
        try {
          return await fn();
        } catch (error) {
          if (error.message.includes("429")) {
            console.warn("Rate limit exceeded, retrying...");
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            continue;
          }
          throw error;
        }
      }
      throw new Error("Max retries exceeded");
    }
    
    retryOnRateLimit(streamAudio);
    ```

- **Note**: Browser playback requires CORS (`allow_origins=["*"]` in `main.py`). WAV is used for compatibility, as MP3/OGG may require additional decoding.

### JavaScript (Node.js with `fetch`)

In Node.js, `fetch` streams audio and saves it to a file (Node.js lacks a native audio playback API).

- **Example**:

```javascript
const fetch = require("node-fetch");
const fs = require("fs");

async function streamAudio() {
  try {
    const response = await fetch("http://localhost:8000/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: "Real-time text-to-speech is now speaking as it generates audio.",
        voice: "am_michael",
        speed: 1.0,
        use_gpu: true,
        format: "wav"
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.detail || "Unknown error"}`);
    }

    const dest = fs.createWriteStream("output.wav");
    response.body.pipe(dest);

    return new Promise((resolve, reject) => {
      dest.on("finish", () => {
        console.log("Audio saved to output.wav");
        resolve();
      });
      dest.on("error", (err) => {
        console.error("Error saving audio:", err);
        reject(err);
      });
    });
  } catch (error) {
    console.error("Error streaming audio:", error.message);
  }
}

streamAudio();
```

- **Usage**:

  - Save as `stream.js`, install dependency, and run:

    ```bash
    npm install node-fetch
    node stream.js
    ```

- **Playback**:

  - Use a system command (e.g., `aplay output.wav` on Linux) or a library like `play-sound`:

    ```bash
    npm install play-sound
    ```

    ```javascript
    const player = require("play-sound")();
    streamAudio().then(() => {
      player.play("output.wav", (err) => {
        if (err) console.error("Playback error:", err);
        else console.log("Audio playback finished");
      });
    });
    ```

- **Expected Output**:

  - Saves `output.wav`.
  - Logs "Audio saved to output.wav".

- **Error Handling**:

  - Similar retry logic as the browser example can be applied.

### Python (with `requests`, `soundfile`, `sounddevice`)

The provided Python example is refined for robustness, error handling, and real-time playback.

- **Example**:

```python
import requests
import soundfile as sf
import sounddevice as sd
import io

def stream_audio():
    try:
        response = requests.post(
            "http://localhost:8000/tts/stream",
            json={
                "text": "Real-time text-to-speech is now speaking as it generates audio.",
                "voice": "am_michael",
                "speed": 1.0,
                "use_gpu": True,
                "format": "wav"
            },
            stream=True
        )

        if response.status_code != 200:
            try:
                error = response.json()
                raise Exception(f"HTTP {response.status_code}: {error.get('detail', 'Unknown error')}")
            except ValueError:
                raise Exception(f"HTTP {response.status_code}: {response.text or 'Unknown error'}")

        print("Streaming audio...")
        audio_buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_buffer.write(chunk)

        audio_buffer.seek(0)
        data, samplerate = sf.read(audio_buffer)
        sf.write("temp_audio.wav", data, samplerate)  # Optional: save for debugging
        print(f"Playing audio... (duration: {len(data)/samplerate:.2f} seconds)")
        sd.play(data, samplerate)
        sd.wait()  # Wait for playback to finish
        print("Audio playback finished")

    except Exception as e:
        print(f"Error streaming audio: {e}")

if __name__ == "__main__":
    stream_audio()
```

- **Usage**:

  - Save as `stream.py`, install dependencies, and run:

    ```bash
    pip install requests soundfile sounddevice numpy
    python stream.py
    ```

- **Expected Output**:

  - Streams audio, plays it, and logs:

    ```
    Streaming audio...
    Playing audio... (duration: X.XX seconds)
    Audio playback finished
    ```
  - Saves `temp_audio.wav` for debugging.

- **Error Handling**:

  - Handles 429 errors with retries:

    ```python
    import time
    
    def retry_on_rate_limit(func, max_retries=3):
        for i in range(max_retries):
            try:
                return func()
            except Exception as e:
                if "HTTP 429" in str(e):
                    print("Rate limit exceeded, retrying...")
                    time.sleep(2 ** i)  # Exponential backoff
                    continue
                raise
        raise Exception("Max retries exceeded")
    
    retry_on_rate_limit(stream_audio)
    ```

- **Note**: Requires audio drivers (e.g., PortAudio for `sounddevice`). On Linux, ensure `libportaudio2` is installed:

  ```bash
  sudo apt install libportaudio2
  ```

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

     Expected: `PONG`
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

### Client Environments

- **cURL**:

  - Ensure installed:

    ```bash
    curl --version
    ```

- **JavaScript (Browser)**:

  - No setup needed; use browser DevTools or an HTML file.

- **JavaScript (Node.js)**:

  - Install Node.js (v18+) and `node-fetch`:

    ```bash
    npm install node-fetch
    ```
  - Optional for playback:

    ```bash
    npm install play-sound
    ```

- **Python**:

  - Install dependencies:

    ```bash
    pip install requests soundfile sounddevice numpy
    ```
  - Verify Python version:

    ```bash
    python --version
    ```

## Features

- **Real-Time Streaming**: Audio is generated and streamed incrementally, reducing latency.
- **Rate-Limiting**: 5 requests/min. Handle 429 errors with retries (see examples).
- **Audio Formats**: WAV recommended for real-time playback due to universal compatibility.
- **CUDA Support**: Set `use_gpu: true` for faster processing if available.
- **Custom Pronunciations**: Include in request (e.g., `pronunciations: {"kokoro": "koh-koh-roh"}`).
- **CORS**: Enabled for browser clients (`*`).

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

- **Rate-Limiting**: 429 errors require retry logic or disabling rate-limiting:

  ```bash
  export ENABLE_RATE_LIMITING="false"
  ```

- **Browser Playback**: WAV is reliable; MP3/OGG may require decoding support.

- **Node.js Playback**: Requires external libraries (e.g., `play-sound`) or system commands.

- **Python Audio Drivers**: `sounddevice` may fail without PortAudio. Install:

  ```bash
  sudo apt install libportaudio2
  ```

- **File Paths**: Server errors if `MODEL_PATH` or `CONFIG_PATH` is incorrect.

- **Streaming Latency**: Depends on server performance and network; CUDA helps.

## Troubleshooting

- **HTTP 429 (Rate Limit)**:

  - Use retry logic (see examples) or disable rate-limiting:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **HTTP 400/500**:

  - Check request payload (e.g., valid `voice`, non-empty `text`).
  - Verify `ffmpeg`:

    ```bash
    ffmpeg -version
    ```
  - Check server logs for model or path issues.

- **No Audio Output**:

  - Verify response status:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav -w "%{http_code}"
    ```
  - For Python, ensure audio drivers:

    ```bash
    python -c "import sounddevice as sd; print(sd.query_devices())"
    ```

- **CORS Errors (Browser)**:

  - Check browser console for CORS messages.
  - Ensure `app.add_middleware(CORSMiddleware, allow_origins=["*"])` in `main.py`.

- **Python Errors**:

  - Verify dependencies:

    ```bash
    pip show requests soundfile sounddevice numpy
    ```

For support, share server logs, `config.json`, or client-side errors.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025