Kokoro TTS API Documentation
Overview
The Kokoro Text-to-Speech (TTS) API provides a robust interface for generating high-quality speech from text using various voices and configurations. Built with FastAPI, the API supports asynchronous audio generation, batch processing, streaming, custom pronunciations, and emotional speech synthesis. It includes features like rate limiting, Prometheus metrics for monitoring, and audio file caching for performance optimization.
Key Features

Text-to-Speech Conversion: Generate audio in WAV, MP3, or OGG formats using American or British English voices.
Multiple Voices: Synthesize the same text with multiple voices in a single request.
Emotional Speech: Adjust speech tone to convey emotions like happy, sad, or angry.
Batch Processing: Process multiple TTS requests in a single batch.
Streaming Support: Stream audio output for real-time playback.
Pronunciation Customization: Define custom pronunciations for specific words.
Monitoring and Statistics: Access API usage metrics and system health status.
Validation: Validate TTS requests without generating audio.
Resource Management: Efficient handling of GPU/CPU resources with fallback for CUDA errors.

Base URL
http://<host>:8000 (default host: 0.0.0.0, default port: 8000)
Authentication
No authentication is required. Rate limiting is enabled by default (10 requests/minute for most endpoints, 5 requests/minute for streaming and batch endpoints).
Endpoints
1. POST /tts
Generate audio for a single text input with specified voice and parameters.
Request Body (TTSRequest):

text (string, required): Text to convert to speech (max 5000 characters).
voice (enum, default: af_heart): Voice to use (e.g., af_heart, am_michael, bf_emma).
speed (float, default: 1.0): Speech speed (0.5 to 2.0).
use_gpu (boolean, default: true): Use GPU if available.
return_tokens (boolean, default: false): Return phonetic tokens.
format (enum, default: wav): Audio format (wav, mp3, ogg).
pronunciations (object, optional): Custom word pronunciations (e.g., {"word": "pronunciation"}).

Response (TTSResponse):

audio_url (string, nullable): URL to the generated audio file.
duration (float, nullable): Audio duration in seconds.
tokens (string, nullable): Phonetic tokens if requested.
request_id (string): Unique job ID.
status (string): Job status (queued, processing, complete, failed).

Example Request:
curl -X POST "http://localhost:8000/tts" \
-H "Content-Type: application/json" \
-d '{
    "text": "Hello, world!",
    "voice": "af_heart",
    "speed": 1.0,
    "use_gpu": true,
    "format": "wav"
}'

Example Response:
{
    "request_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "queued",
    "audio_url": null,
    "duration": null,
    "tokens": null
}

2. POST /tts/stream
Stream audio for a single text input in real-time.
Request Body (TTSRequest): Same as /tts.
Response: Streaming audio in the specified format (audio/wav, audio/mpeg, or audio/ogg).
Example Request:
curl -X POST "http://localhost:8000/tts/stream" \
-H "Content-Type: application/json" \
-d '{
    "text": "Hello, world!",
    "voice": "af_heart",
    "format": "mp3"
}' --output output.mp3

3. POST /tts/batch
Process multiple TTS requests in a single batch.
Request Body (TTSBatchRequest):

items (array, required): List of TTSRequest objects (minimum 1).

Response (TTSBatchResponse):

batch_id (string): Unique batch ID.
status (string): Batch status (queued, processing, complete).
total_items (integer): Number of requests in the batch.

Example Request:
curl -X POST "http://localhost:8000/tts/batch" \
-H "Content-Type: application/json" \
-d '{
    "items": [
        {"text": "Hello", "voice": "af_heart"},
        {"text": "World", "voice": "am_michael"}
    ]
}'

Example Response:
{
    "batch_id": "987fcdeb-1234-5678-9012-abcdef123456",
    "status": "queued",
    "total_items": 2
}

4. POST /tts/multiple-voices
Generate audio for a single text input using multiple voices.
Request Body (MultiVoiceTTSRequest):

text (string, required): Text to convert (max 5000 characters).
voices (set, required): Set of VoiceOption enums (minimum 1).
speed (float, default: 1.0): Speech speed (0.5 to 2.0).
use_gpu (boolean, default: true): Use GPU if available.
format (enum, default: wav): Audio format (wav, mp3, ogg).
pronunciations (object, optional): Custom pronunciations.

Response (TTSBatchResponse): Same as /tts/batch.
Example Request:
curl -X POST "http://localhost:8000/tts/multiple-voices" \
-H "Content-Type: application/json" \
-d '{
    "text": "Hello, world!",
    "voices": ["af_heart", "am_michael"],
    "speed": 1.0,
    "format": "wav"
}'

Example Response:
{
    "batch_id": "abcdef12-3456-7890-abcd-ef1234567890",
    "status": "queued",
    "total_items": 2
}

5. GET /audio/batch/{batch_id}
Retrieve all audio files from a batch job as a ZIP archive.
Path Parameters:

batch_id (string, required): Batch ID from /tts/batch or /tts/multiple-voices.

Response: ZIP file containing all audio files (application/zip).
Example Request:
curl -X GET "http://localhost:8000/audio/batch/987fcdeb-1234-5678-9012-abcdef123456" \
-o "batch_audio.zip"

6. POST /tts/emotion
Generate audio with an emotional tone (simulated via speed adjustments).
Request Body (EmotionalTTSRequest):

text (string, required): Text to convert (max 5000 characters).
voice (enum, default: af_heart): Voice to use.
emotion (enum, default: neutral): Emotion (neutral, happy, sad, angry).
speed (float, default: 1.0): Base speech speed (0.5 to 2.0).
use_gpu (boolean, default: true): Use GPU if available.
format (enum, default: wav): Audio format.
pronunciations (object, optional): Custom pronunciations.

Response (TTSResponse): Same as /tts.
Example Request:
curl -X POST "http://localhost:8000/tts/emotion" \
-H "Content-Type: application/json" \
-d '{
    "text": "I am so excited!",
    "voice": "af_heart",
    "emotion": "happy",
    "speed": 1.0,
    "format": "wav"
}'

7. GET /status/{job_id}
Check the status of a single TTS job.
Path Parameters:

job_id (string, required): Job ID from /tts or /tts/emotion.

Response (JobStatus):

status (string): Job status (queued, processing, complete, failed, cancelled).
progress (integer): Progress percentage (0-100).
error (string, nullable): Error message if failed.
result (object, nullable): Result details if complete.

Example Request:
curl -X GET "http://localhost:8000/status/123e4567-e89b-12d3-a456-426614174000"

Example Response:
{
    "status": "complete",
    "progress": 100,
    "error": null,
    "result": {
        "audio_url": "/audio/123e4567-e89b-12d3-a456-426614174000.wav",
        "duration": 2.5,
        "tokens": null,
        "request_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "complete"
    }
}

8. GET /status/batch/{batch_id}
Check the status of a batch job and its individual items.
Path Parameters:

batch_id (string, required): Batch ID from /tts/batch or /tts/multiple-voices.

Response:

batch_id (string): Batch ID.
batch_status (object): Batch status details (status, progress, total_items, processed_items).
items (array): Status of individual jobs (item_id, status, progress, error, result).

Example Request:
curl -X GET "http://localhost:8000/status/batch/987fcdeb-1234-5678-9012-abcdef123456"

Example Response:
{
    "batch_id": "987fcdeb-1234-5678-9012-abcdef123456",
    "batch_status": {
        "status": "complete",
        "progress": 100,
        "total_items": 2,
        "processed_items": 2
    },
    "items": [
        {
            "item_id": "987fcdeb-1234-5678-9012-abcdef123456_0",
            "status": "complete",
            "progress": 100,
            "error": null,
            "result": {...}
        },
        {...}
    ]
}

9. GET /audio/{filename}
Retrieve a generated audio file.
Path Parameters:

filename (string, required): Name of the audio file (e.g., 123e4567-e89b-12d3-a456-426614174000.wav).

Response: Audio file in the requested format (audio/wav, audio/mpeg, or audio/ogg).
Example Request:
curl -X GET "http://localhost:8000/audio/123e4567-e89b-12d3-a456-426614174000.wav" \
-o "output.wav"

10. DELETE /audio/{filename}
Delete a generated audio file.
Path Parameters:

filename (string, required): Name of the audio file.

Response:

status (string): Deletion status (deleted).
filename (string): Name of the deleted file.

Example Request:
curl -X DELETE "http://localhost:8000/audio/123e4567-e89b-12d3-a456-426614174000.wav"

Example Response:
{
    "status": "deleted",
    "filename": "123e4567-e89b-12d3-a456-426614174000.wav"
}

11. GET /voices
List all available voices.
Response:

american_female (array): List of American female voices.
american_male (array): List of American male voices.
british_female (array): List of British female voices.
british_male (array): List of British male voices.

Example Request:
curl -X GET "http://localhost:8000/voices"

Example Response:
{
    "american_female": [
        {"id": "af_heart", "name": "Heart", "emoji": "❤️"},
        {"id": "af_bella", "name": "Bella", "emoji": "🔥"},
        ...
    ],
    "american_male": [...],
    "british_female": [...],
    "british_male": [...]
}

12. GET /voices/preview
Generate preview audio for all available voices.
Response (TTSBatchResponse): Same as /tts/batch.
Example Request:
curl -X GET "http://localhost:8000/voices/preview"

13. GET /health
Check the API's health and system status.
Response:

status (string): API status (ok).
version (string): API version.
timestamp (float): Current timestamp.
models (object): CPU/GPU model status.
model_files (object): Existence of model and config files.
disk_space (object): Total and free disk space in GB.
memory (object): CUDA memory summary if available.
cuda_available (boolean): GPU availability.
ffmpeg_available (boolean): FFmpeg availability.
active_jobs (integer): Number of active jobs.
cached_files (integer): Number of cached audio files.
rate_limiting_enabled (boolean): Rate limiting status.

Example Request:
curl -X GET "http://localhost:8000/health"

Example Response:
{
    "status": "ok",
    "version": "1.0.0",
    "timestamp": 1745170900.123,
    "models": {"cpu": "available", "gpu": "not initialized"},
    "model_files": {"model_exists": true, "config_exists": true},
    "disk_space": {"total_gb": 100.0, "free_gb": 50.0},
    "memory": {"cuda": "not available"},
    "cuda_available": false,
    "ffmpeg_available": true,
    "active_jobs": 0,
    "cached_files": 10,
    "rate_limiting_enabled": true
}

14. DELETE /cleanup
Delete audio files older than a specified number of hours.
Query Parameters:

hours (integer, default: 24): Age threshold for deletion (minimum 1).

Response:

status (string): Cleanup status (completed).
deleted_files (integer): Number of files deleted.
errors (integer): Number of deletion errors.

Example Request:
curl -X DELETE "http://localhost:8000/cleanup?hours=12"

Example Response:
{
    "status": "completed",
    "deleted_files": 5,
    "errors": 0
}

15. POST /pronunciation
Add a custom pronunciation for a word.
Query Parameters:

word (string, required): Word to customize.
pronunciation (string, required): Phonetic pronunciation.
language_code (string, default: a): Language (a for American, b for British).

Response:

status (string): Operation status (success).
word (string): Word added.
pronunciation (string): Pronunciation added.
language (string): Language name.

Example Request:
curl -X POST "http://localhost:8000/pronunciation?word=hello&pronunciation=hɛˈloʊ&language_code=a"

Example Response:
{
    "status": "success",
    "word": "hello",
    "pronunciation": "hɛˈloʊ",
    "language": "American English"
}

16. GET /pronunciations
List custom pronunciations for a language.
Query Parameters:

language_code (string, default: a): Language (a or b).

Response:

language (string): Language name.
pronunciations (object): Dictionary of custom pronunciations.

Example Request:
curl -X GET "http://localhost:8000/pronunciations?language_code=a"

Example Response:
{
    "language": "American English",
    "pronunciations": {"kokoro": "kˈOkəɹO", "hello": "hɛˈloʊ"}
}

17. DELETE /pronunciations/{word}
Delete a custom pronunciation.
Path Parameters:

word (string, required): Word to remove.

Query Parameters:

language_code (string, default: a): Language (a or b).

Response:

status (string): Deletion status (deleted).
word (string): Deleted word.
language (string): Language name.

Example Request:
curl -X DELETE "http://localhost:8000/pronunciations/hello?language_code=a"

Example Response:
{
    "status": "deleted",
    "word": "hello",
    "language": "American English"
}

18. POST /preprocess
Preprocess text for TTS (e.g., normalize abbreviations, remove special characters).
Request Body (PreprocessRequest):

text (string, required): Text to preprocess (max 5000 characters).

Response (PreprocessResponse):

original_text (string): Original input text.
processed_text (string): Preprocessed text.

Example Request:
curl -X POST "http://localhost:8000/preprocess" \
-H "Content-Type: application/json" \
-d '{"text": "Mr. Smith said: Hello!!"}'

Example Response:
{
    "original_text": "Mr. Smith said: Hello!!",
    "processed_text": "mister smith said hello"
}

19. GET /statistics
Retrieve API usage statistics.
Response:

total_requests (float): Total number of requests.
average_latency_seconds (float): Average request latency.
active_jobs (integer): Number of active jobs.
cached_files (integer): Number of cached audio files.
disk_space_free_gb (float): Free disk space in GB.
timestamp (string): ISO timestamp.

Example Request:
curl -X GET "http://localhost:8000/statistics"

Example Response:
{
    "total_requests": 150,
    "average_latency_seconds": 2.345,
    "active_jobs": 3,
    "cached_files": 10,
    "disk_space_free_gb": 50.0,
    "timestamp": "2025-05-21T13:14:00.123456Z"
}

20. POST /tts/validate
Validate a TTS request without generating audio.
Request Body (ValidateTTSRequest): Same as TTSRequest without return_tokens.
Response (ValidateTTSResponse):

is_valid (boolean): Whether the request is valid.
message (string): Validation result or error message.

Example Request:
curl -X POST "http://localhost:8000/tts/validate" \
-H "Content-Type: application/json" \
-d '{
    "text": "Test text",
    "voice": "af_heart",
    "speed": 1.0,
    "format": "wav"
}'

Example Response:
{
    "is_valid": true,
    "message": "Request is valid"
}

21. GET /metrics
Retrieve Prometheus metrics for monitoring.
Response: Plain text in Prometheus format.
Example Request:
curl -X GET "http://localhost:8000/metrics"

22. GET /config
Retrieve API configuration details.
Response:

sample_rate (integer): Audio sample rate (24000 Hz).
max_char_limit (integer): Maximum text length (5000).
supported_formats (array): Supported audio formats.
cuda_available (boolean): GPU availability.
rate_limiting_enabled (boolean): Rate limiting status.

Example Request:
curl -X GET "http://localhost:8000/config"

Example Response:
{
    "sample_rate": 24000,
    "max_char_limit": 5000,
    "supported_formats": ["wav", "mp3", "ogg"],
    "cuda_available": false,
    "rate_limiting_enabled": true
}

23. POST /jobs/cancel/{job_id}
Cancel a running or queued job.
Path Parameters:

job_id (string, required): Job ID to cancel.

Response:

status (string): Cancellation status (cancelled).
job_id (string): Cancelled job ID.

Example Request:
curl -X POST "http://localhost:8000/jobs/cancel/123e4567-e89b-12d3-a456-426614174000"

Example Response:
{
    "status": "cancelled",
    "job_id": "123e4567-e89b-12d3-a456-426614174000"
}

Error Handling
Errors are returned as JSON with the following structure:
{
    "detail": "Error message"
}

Common Status Codes:

400: Invalid request (e.g., empty text, invalid voice).
404: Resource not found (e.g., job or audio file).
500: Server error (e.g., model initialization failure).

Notes

Rate Limiting: Enabled by default (10/min for /tts, 5/min for /tts/stream, /tts/batch, /tts/multiple-voices, /tts/emotion). Configurable via REDIS_URL and ENABLE_RATE_LIMITING environment variables.
Emotional Speech: The /tts/emotion endpoint simulates emotions by adjusting speed (e.g., 1.2x for happy, 0.8x for sad). Actual emotional support depends on the underlying model.
Dependencies: Requires FFmpeg for audio conversion, Redis for rate limiting, and spaCy for text preprocessing.
Caching: Audio files are cached for 1 hour to improve performance.
GPU Support: Automatically falls back to CPU if CUDA is unavailable or out of memory.

Contact
For support or API pricing details, visit xAI API.
Generated on May 21, 2025
