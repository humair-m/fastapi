from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import torch
import uuid
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, AsyncGenerator
from enum import Enum
import scipy.io.wavfile
import subprocess
import shutil
import os
import re
import json
from contextlib import asynccontextmanager
from cachetools import TTLCache
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import io
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import nest_asyncio
import spacy

# Configuration
class Config:
    AUDIO_OUTPUT_DIR = Path("./audio_output")
    SAMPLE_RATE = 24000
    MAX_CHAR_LIMIT = 5000
    MODEL_PATH = Path('/home/humair/kokoro/kokoro-v1_0.pth')  # Update with actual path
    CONFIG_PATH = Path('/home/humair/kokoro/config.json')    # Update with actual path
    DEFAULT_PORT = 8000
    DEFAULT_HOST = "0.0.0.0"
    CLEANUP_HOURS = 24
    CACHE_TTL = 3600  # 1 hour
    RATE_LIMIT = "10/minute"
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    ENABLE_RATE_LIMITING = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
    REPO_ID = "hexgrad/Kokoro-82M"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kokoro-tts-api")

# Ensure audio output directory exists
Config.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# Prometheus Metrics
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "tts_requests_total", "Total TTS requests", ["endpoint"], registry=REGISTRY
)
REQUEST_LATENCY = Histogram(
    "tts_request_latency_seconds", "Request latency", ["endpoint"], registry=REGISTRY
)
ACTIVE_JOBS = Gauge(
    "tts_active_jobs", "Number of active jobs", registry=REGISTRY
)

# Cache for audio files
audio_cache = TTLCache(maxsize=100, ttl=Config.CACHE_TTL)

# Custom Exceptions
class TTSException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

# Utility Functions
def validate_json_file(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except UnicodeDecodeError as e:
        logger.error(f"Invalid UTF-8 encoding in {file_path}: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

def ensure_spacy_model():
    try:
        spacy.load("en_core_web_sm")
        logger.info("spaCy model en_core_web_sm loaded")
    except OSError:
        logger.info("Downloading spaCy model en_core_web_sm")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        spacy.load("en_core_web_sm")
        logger.info("spaCy model en_core_web_sm downloaded and loaded")

# Enums
class VoiceOption(str, Enum):
    # American English Female
    HEART = "af_heart"
    BELLA = "af_bella"
    NICOLE = "af_nicole"
    AOEDE = "af_aoede"
    KORE = "af_kore"
    SARAH = "af_sarah"
    NOVA = "af_nova"
    SKY = "af_sky"
    ALLOY = "af_alloy"
    JESSICA = "af_jessica"
    RIVER = "af_river"
    # American English Male
    MICHAEL = "am_michael"
    FENRIR = "am_fenrir"
    PUCK = "am_puck"
    ECHO = "am_echo"
    ERIC = "am_eric"
    LIAM = "am_liam"
    ONYX = "am_onyx"
    SANTA = "am_santa"
    ADAM = "am_adam"
    # British English Female
    EMMA = "bf_emma"
    ISABELLA = "bf_isabella"
    ALICE = "bf_alice"
    LILY = "bf_lily"
    # British English Male
    GEORGE = "bm_george"
    FABLE = "bm_fable"
    LEWIS = "bm_lewis"
    DANIEL = "bm_daniel"

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"

# Data Models
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)
    voice: VoiceOption = VoiceOption.HEART
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    use_gpu: bool = True
    return_tokens: bool = False
    format: AudioFormat = AudioFormat.WAV
    pronunciations: Optional[Dict[str, str]] = None

    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class TTSBatchRequest(BaseModel):
    items: List[TTSRequest] = Field(..., min_items=1)

class TTSResponse(BaseModel):
    audio_url: Optional[str]
    duration: Optional[float]
    tokens: Optional[str]
    request_id: str
    status: str

class TTSBatchResponse(BaseModel):
    batch_id: str
    status: str
    total_items: int

class JobStatus(BaseModel):
    status: str
    progress: int
    error: Optional[str]
    result: Optional[Dict]

class PreprocessRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)

class PreprocessResponse(BaseModel):
    original_text: str
    processed_text: str

# Resource Management
class ResourceManager:
    def __init__(self):
        self.models: Dict[str, 'KModel'] = {}
        self.pipelines: Dict[str, 'KPipeline'] = {}
        self.model_lock = asyncio.Lock()
        self.pipeline_locks: Dict[str, asyncio.Lock] = {}
        self.active_jobs: Dict[str, Dict] = {}
        self.job_results: Dict[str, Dict] = {}
        self.cuda_available = torch.cuda.is_available()
        self.job_tasks: Dict[str, asyncio.Task] = {}

    async def get_model(self, use_gpu: bool) -> 'KModel':
        device = "cuda" if use_gpu and self.cuda_available else "cpu"
        async with self.model_lock:
            if device not in self.models:
                if not Config.MODEL_PATH.exists() or not Config.CONFIG_PATH.exists():
                    raise TTSException(f"Model or config file not found: {Config.MODEL_PATH}, {Config.CONFIG_PATH}", 500)
                if not validate_json_file(Config.CONFIG_PATH):
                    raise TTSException(f"Invalid or corrupted JSON config file: {Config.CONFIG_PATH}", 500)
                try:
                    from kokoro import KModel
                    model = KModel(
                        config=str(Config.CONFIG_PATH),
                        model=str(Config.MODEL_PATH),
                        repo_id=Config.REPO_ID
                    )
                    if device == "cuda":
                        model = model.to("cuda")
                    self.models[device] = model.eval()
                except Exception as e:
                    logger.error(f"Model initialization failed: {e}")
                    raise TTSException(f"Model initialization failed: {str(e)}", 500)
            return self.models[device]

    async def get_pipeline(self, lang_code: str) -> 'KPipeline':
        if lang_code not in self.pipeline_locks:
            self.pipeline_locks[lang_code] = asyncio.Lock()
        async with self.pipeline_locks[lang_code]:
            if lang_code not in self.pipelines:
                try:
                    from kokoro import KPipeline
                    pipeline = KPipeline(lang_code=lang_code, model=False, repo_id=Config.REPO_ID)
                    pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO' if lang_code == 'a' else 'kˈQkəɹQ'
                    self.pipelines[lang_code] = pipeline
                except Exception as e:
                    logger.error(f"Pipeline initialization failed for {lang_code}: {e}")
                    raise TTSException(f"Pipeline initialization failed: {str(e)}", 500)
            return self.pipelines[lang_code]

    def cancel_job(self, job_id: str):
        if job_id in self.job_tasks:
            self.job_tasks[job_id].cancel()
            self.active_jobs[job_id]["status"] = "cancelled"
            self.active_jobs[job_id]["progress"] = 0
            del self.job_tasks[job_id]
            logger.info(f"Cancelled job {job_id}")

# Audio Processing
class AudioProcessor:
    @staticmethod
    def convert_audio_format(wav_path: Path, output_format: AudioFormat) -> Path:
        if output_format == AudioFormat.WAV:
            return wav_path
        output_path = wav_path.with_suffix(f".{output_format}")
        try:
            process = subprocess.run([
                "ffmpeg", "-y", "-i", str(wav_path),
                "-acodec", "libmp3lame" if output_format == AudioFormat.MP3 else "libvorbis",
                str(output_path)
            ], check=True, capture_output=True)
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Audio conversion failed: {process.stderr.decode()}")
                return wav_path
            return output_path
        except FileNotFoundError:
            logger.error("ffmpeg not found")
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return wav_path

    @staticmethod
    def save_audio(audio: torch.Tensor, output_path: Path) -> float:
        try:
            scipy.io.wavfile.write(output_path, Config.SAMPLE_RATE, audio.cpu().numpy())
            return len(audio) / Config.SAMPLE_RATE
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise TTSException(f"Failed to save audio: {str(e)}", 500)

    @staticmethod
    async def stream_audio(audio: torch.Tensor, format: AudioFormat) -> AsyncGenerator[bytes, None]:
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, Config.SAMPLE_RATE, audio.cpu().numpy())
        buffer.seek(0)
        if format == AudioFormat.WAV:
            yield buffer.read()
        else:
            output_format = "mp3" if format == AudioFormat.MP3 else "ogg"
            process = subprocess.Popen([
                "ffmpeg", "-i", "pipe:", "-f", output_format, "-acodec",
                "libmp3lame" if format == AudioFormat.MP3 else "libvorbis", "pipe:"
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            buffer.seek(0)
            while chunk := buffer.read(8192):
                process.stdin.write(chunk)
            process.stdin.close()
            while chunk := process.stdout.read(8192):
                yield chunk
            process.terminate()

# Text Preprocessing
class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        abbreviations = {
            "mr.": "mister",
            "mrs.": "missus",
            "dr.": "doctor",
            "st.": "street"
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        return text

# FastAPI App
resource_manager = ResourceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_spacy_model()
        if Config.ENABLE_RATE_LIMITING:
            redis_client = redis.from_url(Config.REDIS_URL)
            try:
                await redis_client.ping()
                await FastAPILimiter.init(redis_client)
                logger.info("Redis connected and FastAPILimiter initialized")
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Failed to connect to Redis at {Config.REDIS_URL}: {e}")
                raise RuntimeError(f"Redis connection failed: {e}")
        else:
            logger.info("Rate limiting disabled")
        await resource_manager.get_model(use_gpu=False)
        for lang_code in ['a', 'b']:
            await resource_manager.get_pipeline(lang_code)
        logger.info("Kokoro TTS API initialized")
        yield
    finally:
        resource_manager.models.clear()
        resource_manager.pipelines.clear()
        if Config.ENABLE_RATE_LIMITING:
            await FastAPILimiter.close()
        logger.info("Resources cleaned up")

app = FastAPI(title="Kokoro TTS API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(TTSException)
async def tts_exception_handler(request: Request, exc: TTSException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

async def generate_audio(request_id: str, request: TTSRequest, stream: bool = False):
    REQUEST_COUNT.labels(endpoint="/tts" if not stream else "/tts/stream").inc()
    start_time = time.time()
    try:
        ACTIVE_JOBS.inc()
        resource_manager.active_jobs[request_id] = {"status": "processing", "progress": 0}
        lang_code = request.voice.value[0]
        if lang_code not in ['a', 'b']:
            raise TTSException(f"Invalid voice prefix: {lang_code}", 400)

        pipeline = await resource_manager.get_pipeline(lang_code)
        if request.pronunciations:
            for word, pron in request.pronunciations.items():
                pipeline.g2p.lexicon.golds[word.lower()] = pron

        try:
            pack = pipeline.load_voice(request.voice)
        except Exception as e:
            raise TTSException(f"Failed to load voice '{request.voice}': {str(e)}", 400)

        model = await resource_manager.get_model(request.use_gpu and resource_manager.cuda_available)
        audio_tensors = []
        tokens = None

        pipeline_results = list(pipeline(request.text, request.voice, request.speed))
        if not pipeline_results:
            raise TTSException("Pipeline processing failed", 500)

        for i, (_, ps, _) in enumerate(pipeline_results):
            resource_manager.active_jobs[request_id]["progress"] = min(95, int((i / len(pipeline_results)) * 100))
            if i == 0 and request.return_tokens:
                tokens = ps
            if not ps:
                continue
            ps = ps[:len(pack)] if len(ps) - 1 >= len(pack) else ps
            try:
                audio = model(ps, pack[len(ps)-1], request.speed)
                audio_tensors.append(audio)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and request.use_gpu:
                    model = await resource_manager.get_model(False)
                    audio = model(ps, pack[len(ps)-1], request.speed)
                    audio_tensors.append(audio)
                else:
                    raise
            await asyncio.sleep(0)

        if not audio_tensors:
            raise TTSException("No audio generated", 500)
        combined_audio = torch.cat(audio_tensors, dim=0) if len(audio_tensors) > 1 else audio_tensors[0]

        if stream:
            media_type = {
                AudioFormat.WAV: "audio/wav",
                AudioFormat.MP3: "audio/mpeg",
                AudioFormat.OGG: "audio/ogg"
            }[request.format]
            return StreamingResponse(
                AudioProcessor.stream_audio(combined_audio, request.format),
                media_type=media_type
            )

        output_path = Config.AUDIO_OUTPUT_DIR / f"{request_id}.wav"
        duration = AudioProcessor.save_audio(combined_audio, output_path)
        if request.format != AudioFormat.WAV:
            output_path = AudioProcessor.convert_audio_format(output_path, request.format)

        response = {
            "audio_url": f"/audio/{output_path.name}",
            "duration": duration,
            "request_id": request_id,
            "status": "complete",
            "tokens": tokens if request.return_tokens else None
        }
        resource_manager.active_jobs[request_id] = {"status": "complete", "progress": 100}
        resource_manager.job_results[request_id] = response
        audio_cache[request_id] = output_path
        REQUEST_LATENCY.labels(endpoint="/tts").observe(time.time() - start_time)
        return response
    except Exception as e:
        logger.error(f"Error in generate_audio: {str(e)}")
        resource_manager.active_jobs[request_id] = {"status": "failed", "progress": 0, "error": str(e)}
        resource_manager.job_results[request_id] = {"error": str(e), "request_id": request_id, "status": "failed"}
        raise TTSException(str(e), 500)
    finally:
        ACTIVE_JOBS.dec()

async def process_batch(batch_id: str, items: List[TTSRequest]):
    REQUEST_COUNT.labels(endpoint="/tts/batch").inc()
    start_time = time.time()
    total_items = len(items)
    resource_manager.active_jobs[batch_id] = {
        "status": "processing", "progress": 0, "total_items": total_items, "processed_items": 0
    }
    results = []
    errors = []
    processed = 0

    for i, item in enumerate(items):
        try:
            request_id = f"{batch_id}_{i}"
            result = await generate_audio(request_id, item)
            results.append(result)
        except TTSException as e:
            errors.append({"index": i, "text": item.text[:50], "error": e.message})
        finally:
            processed += 1
            resource_manager.active_jobs[batch_id]["progress"] = int((processed / total_items) * 100)
            resource_manager.active_jobs[batch_id]["processed_items"] = processed

    resource_manager.job_results[batch_id] = {
        "results": results, "errors": errors, "total_items": total_items, "processed_items": processed
    }
    resource_manager.active_jobs[batch_id]["status"] = "complete"
    resource_manager.active_jobs[batch_id]["progress"] = 100
    REQUEST_LATENCY.labels(endpoint="/tts/batch").observe(time.time() - start_time)

# Endpoints
@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    request_id = str(uuid.uuid4())
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0}
    task = asyncio.create_task(generate_audio(request_id, request))
    resource_manager.job_tasks[request_id] = task
    background_tasks.add_task(lambda: None)
    return TTSResponse(
        request_id=request_id,
        status="queued",
        audio_url=None,
        duration=None,
        tokens=None
    )

@app.post("/tts/stream")
async def stream_speech(
    request: TTSRequest,
    limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    request_id = str(uuid.uuid4())
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0}
    return await generate_audio(request_id, request, stream=True)

@app.post("/tts/batch", response_model=TTSBatchResponse)
async def batch_synthesize_speech(
    request: TTSBatchRequest,
    background_tasks: BackgroundTasks,
    limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    batch_id = str(uuid.uuid4())
    total_items = len(request.items)
    resource_manager.active_jobs[batch_id] = {
        "status": "queued", "progress": 0, "total_items": total_items, "processed_items": 0
    }
    task = asyncio.create_task(process_batch(batch_id, request.items))
    resource_manager.job_tasks[batch_id] = task
    background_tasks.add_task(lambda: None)
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=total_items)

@app.get("/status/{job_id}", response_model=JobStatus)
async def check_job_status(job_id: str):
    REQUEST_COUNT.labels(endpoint="/status").inc()
    if job_id not in resource_manager.active_jobs:
        raise TTSException("Job not found", 404)
    status_data = resource_manager.active_jobs[job_id].copy()
    if status_data["status"] == "complete" and job_id in resource_manager.job_results:
        status_data["result"] = resource_manager.job_results[job_id]
    return JobStatus(**status_data)

@app.get("/status/batch/{batch_id}")
async def check_batch_status(batch_id: str):
    REQUEST_COUNT.labels(endpoint="/status/batch").inc()
    if batch_id not in resource_manager.active_jobs:
        raise TTSException("Batch not found", 404)
    batch_status = resource_manager.active_jobs[batch_id].copy()
    item_statuses = []
    for i in range(batch_status.get("total_items", 0)):
        item_id = f"{batch_id}_{i}"
        if item_id in resource_manager.active_jobs:
            item_status = resource_manager.active_jobs[item_id].copy()
            if item_status["status"] == "complete" and item_id in resource_manager.job_results:
                item_status["result"] = resource_manager.job_results[item_id]
            item_statuses.append({"item_id": item_id, **item_status})
    return {"batch_id": batch_id, "batch_status": batch_status, "items": item_statuses}

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    REQUEST_COUNT.labels(endpoint="/audio").inc()
    if filename in audio_cache:
        file_path = audio_cache[filename]
    else:
        file_path = Config.AUDIO_OUTPUT_DIR / filename
        if not file_path.exists():
            raise TTSException("Audio file not found", 404)
        audio_cache[filename] = file_path
    media_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg"
    }.get(file_path.suffix, "application/octet-stream")
    return FileResponse(path=file_path, media_type=media_type, filename=filename)

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    REQUEST_COUNT.labels(endpoint="/audio/delete").inc()
    if ".." in filename or "/" in filename or "\\" in filename:
        raise TTSException("Invalid filename", 400)
    file_path = Config.AUDIO_OUTPUT_DIR / filename
    if not file_path.exists():
        raise TTSException("Audio file not found", 404)
    try:
        os.remove(file_path)
        audio_cache.pop(filename, None)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        raise TTSException(f"Failed to delete file: {str(e)}", 500)

@app.get("/voices")
async def list_available_voices():
    REQUEST_COUNT.labels(endpoint="/voices").inc()
    return {
        "american_female": [
            {"id": VoiceOption.HEART, "name": "Heart", "emoji": "❤️"},
            {"id": VoiceOption.BELLA, "name": "Bella", "emoji": "🔥"},
            {"id": VoiceOption.NICOLE, "name": "Nicole", "emoji": "🎧"},
            {"id": VoiceOption.AOEDE, "name": "Aoede"},
            {"id": VoiceOption.KORE, "name": "Kore"},
            {"id": VoiceOption.SARAH, "name": "Sarah"},
            {"id": VoiceOption.NOVA, "name": "Nova"},
            {"id": VoiceOption.SKY, "name": "Sky"},
            {"id": VoiceOption.ALLOY, "name": "Alloy"},
            {"id": VoiceOption.JESSICA, "name": "Jessica"},
            {"id": VoiceOption.RIVER, "name": "River"}
        ],
        "american_male": [
            {"id": VoiceOption.MICHAEL, "name": "Michael"},
            {"id": VoiceOption.FENRIR, "name": "Fenrir"},
            {"id": VoiceOption.PUCK, "name": "Puck"},
            {"id": VoiceOption.ECHO, "name": "Echo"},
            {"id": VoiceOption.ERIC, "name": "Eric"},
            {"id": VoiceOption.LIAM, "name": "Liam"},
            {"id": VoiceOption.ONYX, "name": "Onyx"},
            {"id": VoiceOption.SANTA, "name": "Santa"},
            {"id": VoiceOption.ADAM, "name": "Adam"}
        ],
        "british_female": [
            {"id": VoiceOption.EMMA, "name": "Emma"},
            {"id": VoiceOption.ISABELLA, "name": "Isabella"},
            {"id": VoiceOption.ALICE, "name": "Alice"},
            {"id": VoiceOption.LILY, "name": "Lily"}
        ],
        "british_male": [
            {"id": VoiceOption.GEORGE, "name": "George"},
            {"id": VoiceOption.FABLE, "name": "Fable"},
            {"id": VoiceOption.LEWIS, "name": "Lewis"},
            {"id": VoiceOption.DANIEL, "name": "Daniel"}
        ]
    }

@app.get("/voices/preview")
async def preview_voices(background_tasks: BackgroundTasks):
    REQUEST_COUNT.labels(endpoint="/voices/preview").inc()
    preview_text = "Hello, this is a sample of my voice."
    preview_requests = [
        TTSRequest(text=preview_text, voice=voice, format=AudioFormat.WAV)
        for voice in VoiceOption
    ]
    batch_id = str(uuid.uuid4())
    resource_manager.active_jobs[batch_id] = {
        "status": "queued", "progress": 0, "total_items": len(preview_requests), "processed_items": 0
    }
    task = asyncio.create_task(process_batch(batch_id, preview_requests))
    resource_manager.job_tasks[batch_id] = task
    background_tasks.add_task(lambda: None)
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=len(preview_requests))

@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    model_status = {"cpu": "available"}
    if resource_manager.cuda_available:
        model_status["gpu"] = "available" if "cuda" in resource_manager.models else "not initialized"
    disk_usage = shutil.disk_usage(Config.AUDIO_OUTPUT_DIR)
    ffmpeg_available = False
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_available = result.returncode == 0
    except Exception:
        pass
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models": model_status,
        "model_files": {
            "model_exists": Config.MODEL_PATH.exists(),
            "config_exists": Config.CONFIG_PATH.exists()
        },
        "disk_space": {
            "total_gb": round(disk_usage.total / (1024 ** 3), 2),
            "free_gb": round(disk_usage.free / (1024 ** 3), 2)
        },
        "memory": {
            "cuda": torch.cuda.memory_summary(abbreviated=True) if resource_manager.cuda_available else "not available"
        },
        "cuda_available": resource_manager.cuda_available,
        "ffmpeg_available": ffmpeg_available,
        "active_jobs": len(resource_manager.active_jobs),
        "cached_files": len(audio_cache),
        "rate_limiting_enabled": Config.ENABLE_RATE_LIMITING
    }

@app.delete("/cleanup")
async def cleanup_old_files(hours: int = Query(Config.CLEANUP_HOURS, ge=1)):
    REQUEST_COUNT.labels(endpoint="/cleanup").inc()
    cutoff_time = time.time() - (hours * 3600)
    deleted_count = 0
    error_count = 0
    for file_path in Config.AUDIO_OUTPUT_DIR.glob("*.*"):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            try:
                os.remove(file_path)
                audio_cache.pop(file_path.name, None)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                error_count += 1
    return {"status": "completed", "deleted_files": deleted_count, "errors": error_count}

@app.post("/pronunciation")
async def add_custom_pronunciation(
    word: str = Query(..., min_length=1),
    pronunciation: str = Query(..., min_length=1),
    language_code: str = Query("a", enum=["a", "b"])
):
    REQUEST_COUNT.labels(endpoint="/pronunciation").inc()
    try:
        pipeline = await resource_manager.get_pipeline(language_code)
        pipeline.g2p.lexicon.golds[word.lower()] = pronunciation
        return {
            "status": "success",
            "word": word,
            "pronunciation": pronunciation,
            "language": "American English" if language_code == "a" else "British English"
        }
    except Exception as e:
        logger.error(f"Failed to add pronunciation: {e}")
        raise TTSException(f"Failed to add pronunciation: {str(e)}", 500)

@app.get("/pronunciations")
async def list_pronunciations(language_code: str = Query("a", enum=["a", "b"])):
    REQUEST_COUNT.labels(endpoint="/pronunciations").inc()
    try:
        pipeline = await resource_manager.get_pipeline(language_code)
        return {
            "language": "American English" if language_code == "a" else "British English",
            "pronunciations": pipeline.g2p.lexicon.golds
        }
    except Exception as e:
        logger.error(f"Failed to list pronunciations: {e}")
        raise TTSException(f"Failed to list pronunciations: {str(e)}", 500)

@app.delete("/pronunciations/{word}")
async def delete_pronunciation(word: str, language_code: str = Query("a", enum=["a", "b"])):
    REQUEST_COUNT.labels(endpoint="/pronunciations/delete").inc()
    try:
        pipeline = await resource_manager.get_pipeline(language_code)
        if word.lower() in pipeline.g2p.lexicon.golds:
            del pipeline.g2p.lexicon.golds[word.lower()]
            return {"status": "deleted", "word": word, "language": "American English" if language_code == "a" else "British English"}
        raise TTSException(f"Pronunciation for '{word}' not found", 404)
    except Exception as e:
        logger.error(f"Failed to delete pronunciation: {e}")
        raise TTSException(f"Failed to delete pronunciation: {str(e)}", 500)

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_text(request: PreprocessRequest):
    REQUEST_COUNT.labels(endpoint="/preprocess").inc()
    processed_text = TextPreprocessor.preprocess_text(request.text)
    return PreprocessResponse(original_text=request.text, processed_text=processed_text)

@app.get("/metrics")
async def metrics():
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    return Response(content=prometheus_client.generate_latest(REGISTRY), media_type="text/plain")

@app.get("/config")
async def get_config():
    REQUEST_COUNT.labels(endpoint="/config").inc()
    return {
        "sample_rate": Config.SAMPLE_RATE,
        "max_char_limit": Config.MAX_CHAR_LIMIT,
        "supported_formats": [f.value for f in AudioFormat],
        "cuda_available": resource_manager.cuda_available,
        "rate_limiting_enabled": Config.ENABLE_RATE_LIMITING
    }

@app.post("/jobs/cancel/{job_id}")
async def cancel_job(job_id: str):
    REQUEST_COUNT.labels(endpoint="/jobs/cancel").inc()
    if job_id not in resource_manager.active_jobs:
        raise TTSException("Job not found", 404)
    resource_manager.cancel_job(job_id)
    return {"status": "cancelled", "job_id": job_id}

# Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", Config.DEFAULT_PORT))
    logger.info(f"Starting Kokoro TTS API on {Config.DEFAULT_HOST}:{port}")
    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply()
        logger.info("Running in notebook environment with nest_asyncio")
        async def run_server():
            config = uvicorn.Config(app, host=Config.DEFAULT_HOST, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        asyncio.run(run_server())
    except RuntimeError:
        uvicorn.run(app, host=Config.DEFAULT_HOST, port=port, log_level="info")
