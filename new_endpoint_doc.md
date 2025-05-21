Kokoro TTS API Documentation
Overview
The Kokoro TTS (Text-to-Speech) API is a FastAPI-based service that generates high-quality speech from text using the Kokoro model. It supports multiple voices (American and British English), audio formats (WAV, MP3, OGG), emotional speech synthesis, batch processing, and custom pronunciations. The API includes Prometheus metrics for monitoring, rate limiting, and robust error handling.
Features

Text-to-Speech Synthesis: Convert text to speech with customizable voices, speeds, and formats.
Multiple Voices: Supports 28 voices (American/British, male/female).
Emotional Speech: Adjusts speech speed for emotional tones (happy, sad, angry, neutral).
Batch Processing: Generate multiple audio files in a single request.
Streaming: Stream audio responses for low-latency playback.
Custom Pronunciations: Define custom phonetic pronunciations for words.
Monitoring: Prometheus metrics for request counts, latency, and active jobs.
Rate Limiting: Configurable rate limits using Redis.
Validation: Validate TTS requests before processing.

Setup
Prerequisites

Python: 3.8+
Dependencies:
fastapi, uvicorn, torch, scipy, redis, cachetools, prometheus_client, spacy
kokoro library (from hexgrad/Kokoro-82M)
ffmpeg (for audio conversion)


Model Files:
Model: /home/humair/kokoro/kokoro-v1_0.pth
Config: /home/humair/kokoro/config.json


Redis: For rate limiting (optional, can be disabled).

Installation

Activate Virtual Environment:
source ~/myenvn/bin/activate


Install Dependencies:
pip install fastapi uvicorn torch scipy redis cachetools prometheus_client spacy
python -m spacy download en_core_web_sm
sudo apt-get install ffmpeg


Verify Model Files:Ensure /home/humair/kokoro/kokoro-v1_0.pth and /home/humair/kokoro/config.json exist.

Run the API:
python /home/humair/kokoro/new_endpoint.py

The server runs on 0.0.0.0:8000 by default.


Configuration

Environment Variables:
PORT: Override default port (8000).
REDIS_URL: Redis connection URL (default: redis://localhost:6379).
ENABLE_RATE_LIMITING: Enable/disable rate limiting (default: true).


Config Class:
AUDIO_OUTPUT_DIR: ./audio_output (stores generated audio).
SAMPLE_RATE: 24000 Hz.
MAX_CHAR_LIMIT: 5000 characters.
CACHE_TTL: 3600 seconds (audio cache).
CLEANUP_HOURS: 24 hours (for old file cleanup).



Endpoints
1. POST /tts
Synthesize speech from text.

Request:
{
  "text": "Hello, world!",
  "voice": "af_heart",
  "speed": 1.0,
  "use_gpu": true,
  "return_tokens": false,
  "format": "wav",
  "pronunciations": {"kokoro": "kˈOkəɹO"}
}


text: Text to synthesize (max 5000 chars).
voice: Voice option (e.g., af_heart, bm_daniel).
speed: Speech speed (0.5–2.0).
use_gpu: Use GPU if available.
return_tokens: Return phonetic tokens.
format: Audio format (wav, mp3, ogg).
pronunciations: Optional custom pronunciations.


Response:
{
  "audio_url": null,
  "duration": null,
  "tokens": null,
  "request_id": "uuid",
  "status": "queued"
}


audio_url: URL to audio file (e.g., /audio/uuid.wav).
duration: Audio duration (seconds).
tokens: Phonetic tokens (if requested).
request_id: Unique job ID.
status: Job status (queued, processing, complete, failed).



2. POST /tts/stream
Stream synthesized speech.

Request: Same as /tts.
Response: Streaming audio (media type: audio/wav, audio/mpeg, or audio/ogg).
Note: Returns audio data directly, not a JSON response.

3. POST /tts/batch
Synthesize multiple TTS requests in a batch.

Request:{
  "items": [
    {"text": "Hello", "voice": "af_heart", "format": "wav"},
    {"text": "World", "voice": "bm_daniel", "format": "mp3"}
  ]
}


Response:{
  "batch_id": "uuid",
  "status": "queued",
  "total_items": 2
}



4. POST /tts/multiple-voices
Synthesize the same text with multiple voices.

Request:{
  "text": "Hello, world!",
  "voices": ["af_heart", "bm_daniel"],
  "speed": 1.0,
  "use_gpu": true,
  "format": "wav",
  "pronunciations": {"kokoro": "kˈOkəɹO"}
}


Response: Same as /tts/batch.

5. GET /audio/batch/{batch_id}
Retrieve batch audio files as a ZIP.

Parameters:
batch_id: Batch job ID.


Response: ZIP file containing audio files.
Errors:
404: Batch not found.
400: Batch not complete.
404: No audio files found.



6. POST /tts/emotion
Synthesize speech with emotional tone (via speed adjustments).

Request:{
  "text": "I'm so happy!",
  "voice": "af_heart",
  "emotion": "happy",
  "speed": 1.0,
  "use_gpu": true,
  "format": "wav",
  "pronunciations": {}
}


emotion: neutral, happy, sad, angry (adjusts speed: happy=1.2x, sad=0.8x, angry=1.1x).


Response: Same as /tts.

7. GET /status/{job_id}
Check status of a single job.

Response:{
  "status": "complete",
  "progress": 100,
  "error": null,
  "result": {"audio_url": "/audio/uuid.wav", ...}
}



8. GET /status/batch/{batch_id}
Check status of a batch job.

Response:{
  "batch_id": "uuid",
  "batch_status": {"status": "complete", "progress": 100, ...},
  "items": [
    {"item_id": "uuid_0", "status": "complete", ...}
  ]
}



9. GET /audio/{filename}
Retrieve an audio file.

Response: Audio file (media type based on extension).
Errors:
404: File not found.



10. DELETE /audio/{filename}
Delete an audio file.

Response:{
  "status": "deleted",
  "filename": "uuid.wav"
}


Errors:
400: Invalid filename.
404: File not found.



11. GET /voices
List available voices.

Response:{
  "american_female": [{"id": "af_heart", "name": "Heart", "emoji": "❤️"}, ...],
  "american_male": [...],
  "british_female": [...],
  "british_male": [...]
}



12. GET /voices/preview
Generate preview audio for all voices.

Response: Same as /tts/batch.
Note: Uses default text: "Hello, this is a sample of my voice."

13. GET /health
Check API health.

Response:{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": 1697051234.567,
  "models": {"cpu": "available", "gpu": "available"},
  ...
}



14. DELETE /cleanup
Delete audio files older than specified hours.

Query:
hours: Hours (default: 24).


Response:{
  "status": "completed",
  "deleted_files": 5,
  "errors": 0
}



15. POST /pronunciation
Add a custom pronunciation.

Query:
word: Word to define.
pronunciation: Phonetic pronunciation.
language_code: a (American) or b (British).


Response:{
  "status": "success",
  "word": "kokoro",
  "pronunciation": "kˈOkəɹO",
  "language": "American English"
}



16. GET /pronunciations
List custom pronunciations.

Query:
language_code: a or b.


Response:{
  "language": "American English",
  "pronunciations": {"kokoro": "kˈOkəɹO"}
}



17. DELETE /pronunciations/{word}
Delete a custom pronunciation.

Response:{
  "status": "deleted",
  "word": "kokoro",
  "language": "American English"
}



18. POST /preprocess
Preprocess text (lowercase, remove special chars, expand abbreviations).

Request:{
  "text": "Hello, Mr. Smith!"
}


Response:{
  "original_text": "Hello, Mr. Smith!",
  "processed_text": "hello, mister smith!"
}



19. GET /statistics
Get API usage statistics.

Response:{
  "total_requests": 100,
  "average_latency_seconds": 0.5,
  "active_jobs": 2,
  "cached_files": 10,
  "disk_space_free_gb": 50.0,
  "timestamp": "2025-05-21T13:42:00Z"
}



20. POST /tts/validate
Validate a TTS request.

Request: Same as /tts.
Response:{
  "is_valid": true,
  "message": "Request is valid"
}



21. GET /metrics
Get Prometheus metrics.

Response: Plain text Prometheus format.
Metrics:
tts_requests_total
tts_request_latency_seconds
tts_active_jobs
tts_multi_voice_requests_total
tts_emotion_requests_total
tts_stats_requests_total
tts_validate_requests_total



22. GET /config
Get API configuration.

Response:{
  "sample_rate": 24000,
  "max_char_limit": 5000,
  "supported_formats": ["wav", "mp3", "ogg"],
  "cuda_available": true,
  "rate_limiting_enabled": true
}



23. POST /jobs/cancel/{job_id}
Cancel a job.

Response:{
  "status": "cancelled",
  "job_id": "uuid"
}



Voices

American Female: Heart, Bella, Nicole, Aoede, Kore, Sarah, Nova, Sky, Alloy, Jessica, River.
American Male: Michael, Fenrir, Puck, Echo, Eric, Liam, Onyx, Santa, Adam.
British Female: Emma, Isabella, Alice, Lily.
British Male: George, Fable, Lewis, Daniel.

Error Handling

TTSException:
Returns JSON: {"detail": "Error message"}
Status codes: 400 (invalid input), 404 (not found), 500 (server error).


Common Errors:
Missing model/config files.
Invalid voice/format.
Text exceeds 5000 characters.
CUDA out-of-memory (falls back to CPU).



Rate Limiting

Enabled by default (ENABLE_RATE_LIMITING=true).
Limits:
/tts: 10 requests/minute.
/tts/stream, /tts/batch, /tts/multiple-voices, /tts/emotion: 5 requests/minute.


Uses Redis for tracking.

Monitoring

Prometheus metrics exposed at /metrics.
Logs: INFO level, format: %(asctime)s - %(name)s - %(levelname)s - %(message)s.

Example Usage
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "voice": "af_heart",
    "format": "wav"
  }'

Troubleshooting

ModuleNotFoundError: Install missing packages or kokoro library.
FileNotFoundError: Verify model/config file paths.
Redis Errors: Start Redis (redis-server) or disable rate limiting (export ENABLE_RATE_LIMITING=false).
CUDA Errors: Ensure GPU drivers are installed; set use_gpu: false if issues persist.

License
Proprietary. Contact hexgrad/Kokoro-82M maintainers for licensing details.
Version

API: 1.0.0
Last Updated: May 21, 2025

