# Kokoro TTS API Real-Time Streaming in Additional Languages

## Overview

The Kokoro TTS API, built with FastAPI, provides a `/tts/stream` endpoint for real-time text-to-speech (TTS) audio streaming, enabling low-latency audio generation for applications like voice assistants or live narration. The endpoint converts text into audio (WAV, MP3, or OGG) and streams it immediately. The API supports 28 voices (American and British English, male and female), CUDA acceleration, Redis-based rate-limiting (5 requests/min), and runs on `http://localhost:8000` by default with CORS enabled (`allow_origins=["*"]`).

This documentation extends previous examples (cURL, JavaScript, Python) by demonstrating real-time streaming with the `/tts/stream` endpoint in **Ruby**, **Go**, and **PHP**. Each example shows how to send a POST request, stream the audio response, and save it to a file (due to varying playback support across languages). The documentation includes setup instructions, supported voices, error handling (e.g., 429 rate-limit errors), and addresses limitations like server-side warnings.

## Prerequisites

- **API Server**: Running instance of the Kokoro TTS API (see Setup Instructions).

- **Dependencies**:

  - **Ruby**: 3.0+ with `net-http` (standard library) and `json` gems.

    ```bash
    gem install json
    ```

  - **Go**: 1.18+ with standard library (`net/http`).

  - **PHP**: 7.4+ with `curl` extension (typically included).

- **CORS**: Enabled for all origins, relevant for PHP if used in web contexts.

## `/tts/stream` Endpoint

- **Method**: `POST`
- **URL**: `http://localhost:8000/tts/stream`
- **Request**:
  - Body: JSON with `text`, `voice`, `speed`, `use_gpu`, `format`, and optional `pronunciations`, `return_tokens`.
  - Content-Type: `application/json`
- **Response**: Audio stream (`audio/wav`, `audio/mpeg`, or `audio/ogg`).
- **Rate Limit**: 5 requests/min.
- **Errors**:
  - 400: Invalid input (e.g., empty text, invalid voice).
  - 429: Rate limit exceeded.
  - 500: Server error (e.g., model or `ffmpeg` issues).

### Supported Voices

The API supports 28 voices (retrieve via `GET /voices`):

- **American Female**: `af_heart`, `af_bella`, `af_nicole`, `af_aoede`, `af_kore`, `af_sarah`, `af_nova`, `af_sky`, `af_alloy`, `af_jessica`, `af_river`
- **American Male**: `am_michael`, `am_fenrir`, `am_puck`, `am_echo`, `am_eric`, `am_liam`, `am_onyx`, `am_santa`, `am_adam`
- **British Female**: `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`
- **British Male**: `bm_george`, `bm_fable`, `bm_lewis`, `bm_daniel`

### Supported Formats

- WAV (`audio/wav`)
- MP3 (`audio/mpeg`)
- OGG (`audio/ogg`)

WAV is used in examples for compatibility and simplicity.

## Examples

Each example streams audio from `/tts/stream` using the same request parameters:

- `text`: "Real-time text-to-speech is now speaking as it generates audio."
- `voice`: `am_michael`
- `speed`: 1.0
- `use_gpu`: `true`
- `format`: `wav`

### Ruby (with `net/http`)

Ruby uses the `net/http` standard library to stream audio and save it to a file.

- **Example**:

```ruby
require 'net/http'
require 'uri'
require 'json'

def stream_audio
  uri = URI('http://localhost:8000/tts/stream')
  http = Net::HTTP.new(uri.host, uri.port)

  request = Net::HTTP::Post.new(uri.path)
  request['Content-Type'] = 'application/json'
  request.body = {
    text: 'Real-time text-to-speech is now speaking as it generates audio.',
    voice: 'am_michael',
    speed: 1.0,
    use_gpu: true,
    format: 'wav'
  }.to_json

  begin
    http.request(request) do |response|
      unless response.code == '200'
        error_body = response.body
        error_detail = JSON.parse(error_body)['detail'] rescue 'Unknown error'
        raise "HTTP #{response.code}: #{error_detail}"
      end

      puts 'Streaming audio...'
      File.open('output.wav', 'wb') do |file|
        response.read_body do |chunk|
          file.write(chunk)
        end
      end
      puts 'Audio saved to output.wav'
    end
  rescue StandardError => e
    puts "Error streaming audio: #{e.message}"
  end
end

stream_audio
```

- **Usage**:

  - Save as `stream.rb` and run:

    ```bash
    ruby stream.rb
    ```

- **Playback**:

  - Linux/macOS:

    ```bash
    aplay output.wav
    ```

  - Or use a Ruby gem like `ruby-audio`:

    ```bash
    gem install ruby-audio
    ```

    ```ruby
    require 'ruby-audio'
    # After streaming
    sound = RubyAudio::Sound.open('output.wav')
    sound.play
    sleep(sound.info.time) # Wait for playback
    ```

- **Expected Output**:

  - Saves `output.wav`.

  - Prints:

    ```
    Streaming audio...
    Audio saved to output.wav
    ```

- **Error Handling**:

  - Retry for 429 errors:

    ```ruby
    def retry_on_rate_limit(max_retries = 3)
      retries = 0
      begin
        yield
      rescue => e
        if e.message.include?('429') && retries < max_retries
          retries += 1
          sleep(2 ** retries)
          puts "Rate limit exceeded, retrying (#{retries}/#{max_retries})..."
          retry
        end
        raise e
      end
    end
    
    retry_on_rate_limit { stream_audio }
    ```

### Go (with `net/http`)

Go uses the `net/http` package to stream audio and save it to a file.

- **Example**:

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

func streamAudio() error {
	url := "http://localhost:8000/tts/stream"
	data := map[string]interface{}{
		"text":       "Real-time text-to-speech is now speaking as it generates audio.",
		"voice":      "am_michael",
		"speed":      1.0,
		"use_gpu":    true,
		"format":     "wav",
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %v", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		var errorResp map[string]interface{}
		if json.Unmarshal(body, &errorResp) == nil {
			return fmt.Errorf("HTTP %d: %s", resp.StatusCode, errorResp["detail"])
		}
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	fmt.Println("Streaming audio...")
	outFile, err := os.Create("output.wav")
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save audio: %v", err)
	}

	fmt.Println("Audio saved to output.wav")
	return nil
}

func main() {
	if err := streamAudio(); err != nil {
		fmt.Printf("Error streaming audio: %v\n", err)
	}
}
```

- **Usage**:

  - Save as `stream.go` and run:

    ```bash
    go run stream.go
    ```

- **Playback**:

  - Linux/macOS:

    ```bash
    aplay output.wav
    ```

  - Go lacks a standard audio playback library; use system commands or external tools (e.g., `ffplay` from `ffmpeg`).

- **Expected Output**:

  - Saves `output.wav`.

  - Prints:

    ```
    Streaming audio...
    Audio saved to output.wav
    ```

- **Error Handling**:

  - Retry for 429 errors:

    ```go
    func retryOnRateLimit(fn func() error, maxRetries int) error {
    	for i := 0; i < maxRetries; i++ {
    		err := fn()
    		if err != nil && strings.Contains(err.Error(), "429") {
    			fmt.Printf("Rate limit exceeded, retrying (%d/%d)...\n", i+1, maxRetries)
    			time.Sleep(time.Duration(1<<uint(i)) * time.Second)
    			continue
    		}
    		return err
    	}
    	return errors.New("max retries exceeded")
    }
    
    func main() {
    	err := retryOnRateLimit(streamAudio, 3)
    	if err != nil {
    		fmt.Printf("Error streaming audio: %v\n", err)
    	}
    }
    ```

  - Add imports:

    ```go
    import (
    	"bytes"
    	"encoding/json"
    	"errors"
    	"fmt"
    	"io"
    	"net/http"
    	"os"
    	"strings"
    	"time"
    )
    ```

### PHP (with `curl`)

PHP uses the `curl` extension to stream audio and save it to a file.

- **Example**:

```php
<?php

function streamAudio() {
    $url = 'http://localhost:8000/tts/stream';
    $data = [
        'text' => 'Real-time text-to-speech is now speaking as it generates audio.',
        'voice' => 'am_michael',
        'speed' => 1.0,
        'use_gpu' => true,
        'format' => 'wav'
    ];

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, false); // Stream directly
    curl_setopt($ch, CURLOPT_WRITEFUNCTION, function($ch, $chunk) {
        static $file = null;
        if ($file === null) {
            $file = fopen('output.wav', 'wb');
            if ($file === false) {
                return -1; // Signal error
            }
            echo "Streaming audio...\n";
        }
        fwrite($file, $chunk);
        return strlen($chunk);
    });

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    curl_close($ch);

    if ($file !== null) {
        fclose($file);
    }

    if ($response === false || $httpCode != 200) {
        $errorMsg = $error;
        if ($httpCode != 200) {
            $errorData = json_decode(file_get_contents('output.wav'), true);
            $errorMsg = $errorData['detail'] ?? 'Unknown error';
            $errorMsg = "HTTP $httpCode: $errorMsg";
        }
        echo "Error streaming audio: $errorMsg\n";
        return;
    }

    echo "Audio saved to output.wav\n";
}

streamAudio();
```

- **Usage**:

  - Save as `stream.php` and run:

    ```bash
    php stream.php
    ```

- **Playback**:

  - Linux/macOS:

    ```bash
    aplay output.wav
    ```

  - PHP lacks native audio playback; use system commands or a library like `php-ffmpeg`.

- **Expected Output**:

  - Saves `output.wav`.

  - Prints:

    ```
    Streaming audio...
    Audio saved to output.wav
    ```

- **Error Handling**:

  - Retry for 429 errors:

    ```php
    function retryOnRateLimit($func, $maxRetries = 3) {
        for ($i = 0; $i < $maxRetries; $i++) {
            ob_start();
            $func();
            $output = ob_get_clean();
            if (strpos($output, 'HTTP 429') !== false) {
                echo "Rate limit exceeded, retrying (" . ($i + 1) . "/$maxRetries)...\n";
                sleep(pow(2, $i));
                continue;
            }
            echo $output;
            return;
        }
        echo "Error: Max retries exceeded\n";
    }
    
    retryOnRateLimit('streamAudio');
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

- **Ruby**:

  - Install Ruby (3.0+):

    ```bash
    ruby --version
    ```

  - Install `json`:

    ```bash
    gem install json
    ```

- **Go**:

  - Install Go (1.18+):

    ```bash
    go version
    ```

  - No additional dependencies needed.

- **PHP**:

  - Install PHP (7.4+) with `curl`:

    ```bash
    php -v
    php -m | grep curl
    ```

  - If `curl` is missing, install:

    ```bash
    sudo apt install php-curl
    ```

## Features

- **Real-Time Streaming**: Audio is streamed incrementally, reducing latency.
- **Rate-Limiting**: 5 requests/min; retry logic included in examples.
- **Audio Formats**: WAV used for simplicity; MP3/OGG supported.
- **CUDA Support**: Set `use_gpu: true` for faster processing.
- **Custom Pronunciations**: Include in request (e.g., `pronunciations: {"kokoro": "koh-koh-roh"}`).
- **CORS**: Enabled, relevant for PHP in web contexts.

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

- **Rate-Limiting**: Handle 429 errors with retries or disable:

  ```bash
  export ENABLE_RATE_LIMITING="false"
  ```

- **Playback**: Limited native support; relies on system commands or external libraries:

  - Ruby: `ruby-audio` (optional).
  - Go: System commands (e.g., `aplay`).
  - PHP: System commands or `php-ffmpeg`.

- **Streaming Latency**: Depends on server performance and network; CUDA helps.

- **File Paths**: Server errors if `MODEL_PATH` or `CONFIG_PATH` is incorrect.

## Troubleshooting

- **HTTP 429 (Rate Limit)**:

  - Use retry logic or disable rate-limiting:

    ```bash
    export ENABLE_RATE_LIMITING="false"
    ```

- **HTTP 400/500**:

  - Verify payload (e.g., valid `voice`, non-empty `text`).

  - Check `ffmpeg`:

    ```bash
    ffmpeg -version
    ```

  - Inspect server logs for model or path issues.

- **No Audio Output**:

  - Test endpoint:

    ```bash
    curl -X POST "http://localhost:8000/tts/stream" -H "Content-Type: application/json" -d '{"text": "Test", "voice": "am_michael", "format": "wav"}' -o test.wav
    ```

  - Check file size:

    ```bash
    ls -l test.wav
    ```

- **Language-Specific Issues**:

  - **Ruby**: Ensure `json` gem:

    ```bash
    gem list json
    ```

  - **Go**: Verify Go version:

    ```bash
    go version
    ```

  - **PHP**: Confirm `curl` extension:

    ```bash
    php -m | grep curl
    ```

For support, share server logs, `config.json`, or client-side errors.

## Version

- **API Version**: 1.0.0
- **Last Updated**: April 21, 2025