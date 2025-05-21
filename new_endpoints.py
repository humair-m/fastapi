from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
import uvicorn
import torch
import uuid
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, AsyncGenerator, Set
from enum import Enum
import scipy.io.wavfile
import subprocess
import shutil
import os
import re
import json
from contextlib import asynccontextmanager
from cachetools import TTLCache
import redis.asyncio as redis
import io
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import nest_asyncio
import spacy
from datetime import datetime
import zipfile

# Configuration
class Config:
    AUDIO_OUTPUT_DIR = Path("./audio_output")
    SAMPLE_RATE = 24000
    MAX_CHAR_LIMIT = 5000
    MODEL_PATH = Path('/home/humair/kokoro/kokoro-v1_0.pth')
    CONFIG_PATH = Path('/home/humair/kokoro/config.json')
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
MULTI_VOICE_REQUESTS = Counter(
    "tts_multi_voice_requests_total", "Total multi-voice TTS requests", ["endpoint"], registry=REGISTRY
)
EMOTION_REQUESTS = Counter(
    "tts_emotion_requests_total", "Total emotion-based TTS requests", ["endpoint"], registry=REGISTRY
)
STATS_REQUESTS = Counter(
    "tts_stats_requests_total", "Total statistics requests", ["endpoint"], registry=REGISTRY
)
VALIDATE_REQUESTS = Counter(
    "tts_validate_requests_total", "Total validate requests", ["endpoint"], registry=REGISTRY
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
    except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError, Exception) as e:
        logger.error(f"Error validating JSON file {file_path}: {e}")
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

class EmotionOption(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"

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

class MultiVoiceTTSRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)
    voices: Set[VoiceOption] = Field(..., min_items=1)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    use_gpu: bool = True
    format: AudioFormat = AudioFormat.WAV
    pronunciations: Optional[Dict[str, str]] = None

class EmotionalTTSRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)
    voice: VoiceOption = VoiceOption.HEART
    emotion: EmotionOption = EmotionOption.NEUTRAL
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    use_gpu: bool = True
    format: AudioFormat = AudioFormat.WAV
    pronunciations: Optional[Dict[str, str]] = None

class ValidateTTSRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)
    voice: VoiceOption = VoiceOption.HEART
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: AudioFormat = AudioFormat.WAV
    pronunciations: Optional[Dict[str, str]] = None

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

class ValidateTTSResponse(BaseModel):
    is_valid: bool
    message: str

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
                    raise TTSException(f"Model or config file not found", 500)
                if not validate_json_file(Config.CONFIG_PATH):
                    raise TTSException(f"Invalid JSON config file", 500)
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

    @staticmethod
    def create_zip_from_batch(batch_id: str, file_paths: List[Path]) -> Path:
        zip_path = Config.AUDIO_OUTPUT_DIR / f"{batch_id}.zip"
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)
            return zip_path
        except Exception as e:
            logger.error(f"Failed to create ZIP file for batch {batch_id}: {e}")
            raise TTSException(f"Failed to create ZIP file: {str(e)}", 500)

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
            await redis_client.ping()
            await FastAPILimiter.init(redis_client)
            logger.info("Redis connected and FastAPILimiter initialized")
        await resource_manager.get_model(use_gpu=False)
        for lang_code in ['a', 'b']:
            await resource_manager.get_pipeline(lang_code)
        logger.info("Kokoro TTS API initialized")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
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
    return JSONResponse(status_code='And you want to see the entire response?
