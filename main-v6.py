import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import prometheus_client
import redis.asyncio as redis
import scipy.io.wavfile
import spacy
import torch
import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Request, BackgroundTasks, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Configuration
class Config:
    AUDIO_OUTPUT_DIR = Path("./audio_output")
    SAMPLE_RATE = 24000
    MAX_CHAR_LIMIT = 5000
    MAX_TOKEN_LIMIT = 500
    MODEL_PATH = Path("/home/humair/kokoro/kokoro-v1_0.pth")
    CONFIG_PATH = Path("/home/humair/kokoro/config.json")
    DEFAULT_PORT = int(os.environ.get("PORT", 8000))
    DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
    CLEANUP_HOURS = 24
    CACHE_TTL = 3600
    RATE_LIMIT = "10/minute"
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    ENABLE_RATE_LIMITING = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
    REPO_ID = "hexgrad/Kokoro-82M"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kokoro-tts-api")

# Ensure audio output directory
Config.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# Prometheus Metrics
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter("tts_requests_total", "Total TTS requests", ["endpoint"], registry=REGISTRY)
REQUEST_LATENCY = Histogram("tts_request_latency_seconds", "Request latency", ["endpoint"], registry=REGISTRY)
ACTIVE_JOBS = Gauge("tts_active_jobs", "Number of active jobs", registry=REGISTRY)

# Audio Cache
audio_cache = TTLCache(maxsize=100, ttl=Config.CACHE_TTL)

# Custom Exception
class TTSException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

# Enums
class VoiceOption(str, Enum):
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
    MICHAEL = "am_michael"
    FENRIR = "am_fenrir"
    PUCK = "am_puck"
    ECHO = "am_echo"
    ERIC = "am_eric"
    LIAM = "am_liam"
    ONYX = "am_onyx"
    SANTA = "am_santa"
    ADAM = "am_adam"
    EMMA = "bf_emma"
    ISABELLA = "bf_isabella"
    ALICE = "bf_alice"
    LILY = "bf_lily"
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

class BatchSummaryResponse(BaseModel):
    batch_id: str
    status: str
    progress: int
    total_items: int
    completed_items: int
    failed_items: int
    audio_urls: List[str]

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)
    language_code: str = Field(default="a", enum=["a", "b"])

class TextAnalysisResponse(BaseModel):
    text: str
    char_count: int
    warnings: List[str]
    suggestions: List[str]
    estimated_duration: Optional[float]

class VoiceSampleRequest(BaseModel):
    voice: VoiceOption
    text: str = Field(default="This is a sample of my voice.", max_length=100)
    format: AudioFormat = AudioFormat.WAV
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

class PronunciationValidationRequest(BaseModel):
    word: str = Field(..., min_length=1)
    pronunciation: str = Field(..., min_length=1)
    language_code: str = Field(default="a", enum=["a", "b"])

class PronunciationValidationResponse(BaseModel):
    word: str
    pronunciation: str
    is_valid: bool
    errors: List[str]

class AudioMetadataResponse(BaseModel):
    filename: str
    format: str
    size_bytes: int
    duration_seconds: Optional[float]
    created_at: float
    exists: bool

class ModelInfoResponse(BaseModel):
    model_version: str
    supported_languages: List[str]
    model_path: str
    config_path: str
    repo_id: str
    cuda_enabled: bool
    model_parameters: Optional[Dict]

class TextSegmentationRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT * 2)
    max_tokens_per_segment: int = Field(default=Config.MAX_TOKEN_LIMIT, ge=50, le=Config.MAX_TOKEN_LIMIT)

class TextSegmentationResponse(BaseModel):
    segments: List[str]
    segment_count: int
    total_tokens: int

class CacheInfoResponse(BaseModel):
    total_items: int
    cache_size_bytes: int
    items: List[Dict]

class RateLimitStatusResponse(BaseModel):
    enabled: bool
    limit: Optional[str]
    remaining: Optional[int]
    reset_timestamp: Optional[float]

class StatsResponse(BaseModel):
    total_requests: int
    completed_jobs: int
    failed_jobs: int
    average_processing_time_seconds: float
    total_audio_duration_seconds: float
    cache_hit_ratio: float

class JobsResponse(BaseModel):
    jobs: List[Dict]
    total_jobs: int
    limit: int
    offset: int

class TTSEstimateResponse(BaseModel):
    estimated_processing_time_seconds: float
    estimated_audio_duration_seconds: float
    token_count: int
    segment_count: int

class ClearCacheResponse(BaseModel):
    status: str
    deleted_files: int
    errors: int
    freed_space_bytes: int

# Utility Functions
def validate_json_file(file_path: Path) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
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

# Resource Manager
class ResourceManager:
    def __init__(self):
        self.models: Dict[str, "KModel"] = {}
        self.pipelines: Dict[str, "KPipeline"] = {}
        self.model_lock = asyncio.Lock()
        self.pipeline_locks: Dict[str, asyncio.Lock] = {}
        self.active_jobs: Dict[str, Dict] = {}
        self.job_results: Dict[str, Dict] = {}
        self.job_tasks: Dict[str, asyncio.Task] = {}
        self.cuda_available = torch.cuda.is_available()

    async def get_model(self, use_gpu: bool) -> "KModel":
        device = "cuda" if use_gpu and self.cuda_available else "cpu"
        async with self.model_lock:
            if device not in self.models:
                if not Config.MODEL_PATH.exists() or not Config.CONFIG_PATH.exists():
                    raise TTSException("Model or config file not found", 500)
                if not validate_json_file(Config.CONFIG_PATH):
                    raise TTSException("Invalid JSON config file", 500)
                try:
                    from kokoro import KModel
                    model = KModel(config=str(Config.CONFIG_PATH), model=str(Config.MODEL_PATH), repo_id=Config.REPO_ID)
                    if device == "cuda":
                        model = model.to("cuda")
                    self.models[device] = model.eval()
                except Exception as e:
                    logger.error(f"Model initialization failed: {e}")
                    raise TTSException(f"Model initialization failed: {str(e)}", 500)
            return self.models[device]

    async def get_pipeline(self, lang_code: str) -> "KPipeline":
        if lang_code not in self.pipeline_locks:
            self.pipeline_locks[lang_code] = asyncio.Lock()
        async with self.pipeline_locks[lang_code]:
            if lang_code not in self.pipelines:
                try:
                    from kokoro import KPipeline
                    pipeline = KPipeline(lang_code=lang_code, model=False, repo_id=Config.REPO_ID)
                    pipeline.g2p.lexicon.golds["kokoro"] = "kˈOkəɹO" if lang_code == "a" else "kˈQkəɹQ"
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

# Audio Processor
class AudioProcessor:
    @staticmethod
    def _check_ffmpeg_availability() -> bool:
        """Check if ffmpeg and ffprobe are available."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"ffmpeg/ffprobe not available: {str(e)}")
            return False

    @staticmethod
    def save_audio(audio: torch.Tensor, output_path: Path) -> float:
        """Save audio tensor to file and return duration in seconds."""
        try:
            logger.info(f"Saving audio to {output_path}")
            scipy.io.wavfile.write(output_path, Config.SAMPLE_RATE, audio.cpu().numpy())
            duration = len(audio) / Config.SAMPLE_RATE
            logger.info(f"Audio saved successfully, duration: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise TTSException(f"Failed to save audio: {str(e)}", 500)

    @staticmethod
    async def stream_audio(audio: torch.Tensor, format: AudioFormat) -> AsyncGenerator[bytes, None]:
        """Stream audio in the specified format, falling back to WAV if conversion fails."""
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, Config.SAMPLE_RATE, audio.cpu().numpy())
        buffer.seek(0)

        if format == AudioFormat.WAV:
            logger.info("Streaming audio in WAV format")
            yield buffer.read()
            return

        if not AudioProcessor._check_ffmpeg_availability():
            logger.error(f"Cannot stream {format.value} audio: ffmpeg/ffprobe not installed. Falling back to WAV.")
            yield buffer.read()
            return

        try:
            logger.info(f"Converting audio to {format.value} using pydub")
            audio_segment = AudioSegment.from_wav(buffer)
            output_buffer = io.BytesIO()
            output_format = "mp3" if format == AudioFormat.MP3 else "ogg"
            audio_segment.export(output_buffer, format=output_format)
            output_buffer.seek(0)
            logger.info(f"Streaming audio in {format.value} format")
            while chunk := output_buffer.read(8192):
                yield chunk
        except Exception as e:
            logger.error(f"Audio streaming error with pydub: {str(e)}. Falling back to WAV.")
            buffer.seek(0)
            yield buffer.read()

    @staticmethod
    def convert_audio_format(wav_path: Path, output_format: AudioFormat) -> Path:
        """Convert WAV audio to specified format using ffmpeg."""
        if output_format == AudioFormat.WAV:
            return wav_path
        output_path = wav_path.with_suffix(f".{output_format}")
        try:
            logger.info(f"Converting {wav_path} to {output_format.value}")
            process = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(wav_path),
                    "-acodec", "libmp3lame" if output_format == AudioFormat.MP3 else "libvorbis",
                    str(output_path)
                ],
                check=True,
                capture_output=True
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Audio conversion failed: {process.stderr.decode()}")
                return wav_path
            logger.info(f"Audio converted to {output_path}")
            return output_path
        except FileNotFoundError:
            logger.error("ffmpeg not found, returning WAV")
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return wav_path

# Text Preprocessor
class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^\w\s.,!?]", "", text)
        abbreviations = {"mr.": "mister", "mrs.": "missus", "dr.": "doctor", "st.": "street"}
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        return text

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def segment_text_by_tokens(text: str, max_tokens: int) -> List[str]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        segments = []
        current_segment = ""
        current_tokens = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = len(sent_text.split())
            if current_tokens + sent_tokens <= max_tokens:
                current_segment += (" " + sent_text) if current_segment else sent_text
                current_tokens += sent_tokens
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sent_text
                current_tokens = sent_tokens
                if current_tokens > max_tokens:
                    words = sent_text.split()
                    temp_segment = ""
                    temp_tokens = 0
                    for word in words:
                        if temp_tokens + 1 <= max_tokens:
                            temp_segment += (" " + word) if temp_segment else word
                            temp_tokens += 1
                        else:
                            segments.append(temp_segment)
                            temp_segment = word
                            temp_tokens = 1
                    current_segment = temp_segment
                    current_tokens = temp_tokens
        if current_segment:
            segments.append(current_segment)
        return segments

# FastAPI Application
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
        await resource_manager.get_model(use_gpu=False)
        for lang_code in ["a", "b"]:
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.exception_handler(TTSException)
async def tts_exception_handler(request: Request, exc: TTSException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

async def generate_audio(request_id: str, request: TTSRequest, stream: bool = False):
    endpoint = "/tts/stream" if stream else "/tts"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.time()
    try:
        ACTIVE_JOBS.inc()
        resource_manager.active_jobs[request_id] = {"status": "processing", "progress": 0, "created_at": start_time}
        lang_code = request.voice.value[0]
        if lang_code not in ["a", "b"]:
            raise TTSException(f"Invalid voice prefix: {lang_code}", 400)

        logger.info(f"Loading pipeline for request_id={request_id}, lang_code={lang_code}")
        pipeline = await resource_manager.get_pipeline(lang_code)
        if request.pronunciations:
            for word, pron in request.pronunciations.items():
                pipeline.g2p.lexicon.golds[word.lower()] = pron

        try:
            logger.info(f"Loading voice {request.voice} for request_id={request_id}")
            pack = pipeline.load_voice(request.voice)
        except Exception as e:
            raise TTSException(f"Failed to load voice '{request.voice}': {str(e)}", 400)

        logger.info(f"Loading model for request_id={request_id}, use_gpu={request.use_gpu}")
        model = await resource_manager.get_model(request.use_gpu and resource_manager.cuda_available)
        audio_tensors = []
        tokens = None
        logger.info(f"Processing pipeline for text: {request.text[:50]}...")
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
                logger.debug(f"Generating audio segment {i+1}/{len(pipeline_results)}")
                audio = model(ps, pack[len(ps)-1], request.speed)
                audio_tensors.append(audio)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and request.use_gpu:
                    logger.warning("CUDA out of memory, falling back to CPU")
                    model = await resource_manager.get_model(False)
                    audio = model(ps, pack[len(ps)-1], request.speed)
                    audio_tensors.append(audio)
                else:
                    raise
            await asyncio.sleep(0)

        if not audio_tensors:
            raise TTSException("No audio generated", 500)
        logger.info(f"Combining {len(audio_tensors)} audio segments")
        combined_audio = torch.cat(audio_tensors, dim=0) if len(audio_tensors) > 1 else audio_tensors[0]

        if stream:
            media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}[request.format.value]
            logger.info(f"Streaming audio for request_id={request_id} in {request.format.value}")
            return StreamingResponse(
                AudioProcessor.stream_audio(combined_audio, request.format),
                media_type=media_type
            )

        output_path = Config.AUDIO_OUTPUT_DIR / f"{request_id}.wav"
        logger.info(f"Saving audio for request_id={request_id} to {output_path}")
        duration = AudioProcessor.save_audio(combined_audio, output_path)
        if request.format != AudioFormat.WAV:
            logger.info(f"Converting audio to {request.format.value}")
            output_path = AudioProcessor.convert_audio_format(output_path, request.format)

        response = {
            "audio_url": f"/audio/{output_path.name}",
            "duration": duration,
            "request_id": request_id,
            "status": "complete",
            "tokens": tokens if request.return_tokens else None
        }
        resource_manager.active_jobs[request_id] = {"status": "complete", "progress": 100, "created_at": start_time}
        resource_manager.job_results[request_id] = response
        audio_cache[request_id] = output_path
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
        logger.info(f"Completed audio generation for request_id={request_id}")
        return response
    except Exception as e:
        logger.error(f"Error in generate_audio (request_id={request_id}): {str(e)}")
        resource_manager.active_jobs[request_id] = {
            "status": "failed", "progress": 0, "error": str(e), "created_at": start_time
        }
        resource_manager.job_results[request_id] = {
            "error": str(e), "request_id": request_id, "status": "failed"
        }
        raise TTSException(str(e), 500)
    finally:
        ACTIVE_JOBS.dec()

async def process_batch(batch_id: str, items: List[TTSRequest]):
    REQUEST_COUNT.labels(endpoint="/tts/batch").inc()
    start_time = time.time()
    total_items = len(items)
    resource_manager.active_jobs[batch_id] = {
        "status": "processing", "progress": 0, "total_items": total_items,
        "processed_items": 0, "created_at": start_time
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
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0, "created_at": time.time()}
    token_count = TextPreprocessor.count_tokens(request.text)
    if token_count > Config.MAX_TOKEN_LIMIT:
        logger.info(f"Text exceeds {Config.MAX_TOKEN_LIMIT} tokens ({token_count}), segmenting")
        segments = TextPreprocessor.segment_text_by_tokens(request.text, Config.MAX_TOKEN_LIMIT)
        audio_tensors = []
        total_duration = 0
        tokens = None
        for i, segment in enumerate(segments):
            segment_request = TTSRequest(
                text=segment, voice=request.voice, speed=request.speed, use_gpu=request.use_gpu,
                return_tokens=request.return_tokens and i == 0, format=AudioFormat.WAV,
                pronunciations=request.pronunciations
            )
            segment_id = f"{request_id}_{i}"
            result = await generate_audio(segment_id, segment_request)
            if result["status"] != "complete":
                raise TTSException(f"Segment {i} processing failed: {result.get('error', 'Unknown error')}", 500)
            audio_path = Config.AUDIO_OUTPUT_DIR / result["audio_url"].split("/")[-1]
            rate, data = scipy.io.wavfile.read(audio_path)
            audio_tensors.append(torch.tensor(data, dtype=torch.float32))
            total_duration += result["duration"]
            if result["tokens"]:
                tokens = result["tokens"]
            os.remove(audio_path)
            audio_cache.pop(segment_id, None)
        combined_audio = torch.cat(audio_tensors, dim=0)
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
        resource_manager.active_jobs[request_id] = {"status": "complete", "progress": 100, "created_at": time.time()}
        resource_manager.job_results[request_id] = response
        audio_cache[request_id] = output_path
        return response
    task = asyncio.create_task(generate_audio(request_id, request))
    resource_manager.job_tasks[request_id] = task
    background_tasks.add_task(lambda: None)
    return TTSResponse(request_id=request_id, status="queued", audio_url=None, duration=None, tokens=None)

@app.post("/tts/stream", response_model=None)
async def stream_speech(
    request: TTSRequest,
    limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    request_id = str(uuid.uuid4())
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0, "created_at": time.time()}
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
        "status": "queued", "progress": 0, "total_items": total_items,
        "processed_items": 0, "created_at": time.time()
    }
    task = asyncio.create_task(process_batch(batch_id, request.items))
    resource_manager.job_tasks[batch_id] = task
    background_tasks.add_task(lambda: None)
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=total_items)










# Updated /status/{job_id} endpoint
@app.get("/status/{job_id}", response_model=JobStatus)
async def check_job_status(job_id: str):
    REQUEST_COUNT.labels(endpoint="/status").inc()
    if job_id not in resource_manager.active_jobs:
        raise TTSException("Job not found", 404)
    status_data = resource_manager.active_jobs[job_id].copy()
    # Ensure error and result fields are always present
    status_data.setdefault("error", None)
    status_data.setdefault("result", None)
    if status_data["status"] == "complete" and job_id in resource_manager.job_results:
        status_data["result"] = resource_manager.job_results[job_id]
    logger.debug(f"Job status for {job_id}: {status_data}")
    return JobStatus(**status_data)

# Updated /status/batch/{batch_id} endpoint
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
            # Ensure error and result fields are always present
            item_status.setdefault("error", None)
            item_status.setdefault("result", None)
            if item_status["status"] == "complete" and item_id in resource_manager.job_results:
                item_status["result"] = resource_manager.job_results[item_id]
            item_statuses.append({"item_id": item_id, **item_status})
    logger.debug(f"Batch status for {batch_id}: {batch_status}, items: {len(item_statuses)}")
    return {"batch_id": batch_id, "batch_status": batch_status, "items": item_statuses}
















@app.get("/batch/summary/{batch_id}", response_model=BatchSummaryResponse)
async def batch_summary(batch_id: str):
    REQUEST_COUNT.labels(endpoint="/batch/summary").inc()
    if batch_id not in resource_manager.active_jobs:
        raise TTSException("Batch not found", 404)
    batch_status = resource_manager.active_jobs[batch_id]
    completed_items = 0
    failed_items = 0
    audio_urls = []
    if batch_id in resource_manager.job_results:
        results = resource_manager.job_results[batch_id].get("results", [])
        errors = resource_manager.job_results[batch_id].get("errors", [])
        completed_items = len(results)
        failed_items = len(errors)
        audio_urls = [result["audio_url"] for result in results if result.get("audio_url")]
    return BatchSummaryResponse(
        batch_id=batch_id,
        status=batch_status["status"],
        progress=batch_status["progress"],
        total_items=batch_status["total_items"],
        completed_items=completed_items,
        failed_items=failed_items,
        audio_urls=audio_urls
    )

@app.post("/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    REQUEST_COUNT.labels(endpoint="/analyze/text").inc()
    text = request.text
    char_count = len(text)
    warnings = []
    suggestions = []
    if char_count > Config.MAX_CHAR_LIMIT:
        warnings.append(f"Text exceeds maximum length of {Config.MAX_CHAR_LIMIT} characters")
        suggestions.append(f"Truncate text to {Config.MAX_CHAR_LIMIT} characters")
    if re.search(r"[^\w\s.,!?]", text):
        warnings.append("Text contains unsupported special characters")
        suggestions.append("Remove or replace special characters")
    if "mr." in text.lower():
        suggestions.append("Replace 'Mr.' with 'Mister' for better pronunciation")
    estimated_duration = char_count / 200 * 60 / Config.SAMPLE_RATE
    return TextAnalysisResponse(
        text=text,
        char_count=char_count,
        warnings=warnings,
        suggestions=suggestions,
        estimated_duration=estimated_duration
    )

@app.post("/voices/sample", response_model=TTSResponse)
async def voice_sample(
    request: VoiceSampleRequest,
    background_tasks: BackgroundTasks,
    limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    REQUEST_COUNT.labels(endpoint="/voices/sample").inc()
    request_id = f"sample_{str(uuid.uuid4())}"
    tts_request = TTSRequest(
        text=request.text, voice=request.voice, speed=request.speed,
        format=request.format, use_gpu=True, return_tokens=False
    )
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0, "created_at": time.time()}
    task = asyncio.create_task(generate_audio(request_id, tts_request))
    resource_manager.job_tasks[request_id] = task
    background_tasks.add_task(lambda: None)
    return TTSResponse(request_id=request_id, status="queued", audio_url=None, duration=None, tokens=None)

@app.post("/pronunciation/validate", response_model=PronunciationValidationResponse)
async def validate_pronunciation(request: PronunciationValidationRequest):
    REQUEST_COUNT.labels(endpoint="/pronunciation/validate").inc()
    word = request.word.lower()
    pronunciation = request.pronunciation
    is_valid = True
    errors = []
    if not re.match(r"^[a-zA-Zˈˌəɪʊɛæɔʌɑɒɪəʊə]+$", pronunciation):
        is_valid = False
        errors.append("Pronunciation contains invalid characters")
    try:
        pipeline = await resource_manager.get_pipeline(request.language_code)
        pipeline.g2p.lexicon.golds[word] = pronunciation
        del pipeline.g2p.lexicon.golds[word]
    except Exception as e:
        is_valid = False
        errors.append(f"Invalid pronunciation format: {str(e)}")
    return PronunciationValidationResponse(word=word, pronunciation=pronunciation, is_valid=is_valid, errors=errors)

@app.get("/audio/metadata/{filename}", response_model=AudioMetadataResponse)
async def audio_metadata(filename: str):
    REQUEST_COUNT.labels(endpoint="/audio/metadata").inc()
    file_path = audio_cache.get(filename, Config.AUDIO_OUTPUT_DIR / filename)
    exists = file_path.exists()
    duration = None
    if exists and file_path.suffix == ".wav":
        try:
            rate, data = scipy.io.wavfile.read(file_path)
            duration = len(data) / rate
        except Exception as e:
            logger.error(f"Failed to read WAV duration for {filename}: {e}")
    stat = file_path.stat() if exists else None
    return AudioMetadataResponse(
        filename=filename,
        format=file_path.suffix.lstrip(".") if exists else "unknown",
        size_bytes=stat.st_size if exists else 0,
        duration_seconds=duration,
        created_at=stat.st_mtime if exists else 0,
        exists=exists
    )

@app.get("/batch/download/{batch_id}")
async def batch_download(batch_id: str, background_tasks: BackgroundTasks):
    REQUEST_COUNT.labels(endpoint="/batch/download").inc()
    if batch_id not in resource_manager.active_jobs:
        raise TTSException("Batch not found", 404)
    if resource_manager.active_jobs[batch_id]["status"] != "complete":
        raise TTSException("Batch processing not complete", 400)
    audio_urls = resource_manager.job_results.get(batch_id, {}).get("results", [])
    if not audio_urls:
        raise TTSException("No audio files available for download", 404)
    zip_path = Config.AUDIO_OUTPUT_DIR / f"batch_{batch_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for result in audio_urls:
                file_path = audio_cache.get(result["request_id"], Config.AUDIO_OUTPUT_DIR / result["audio_url"].split("/")[-1])
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        background_tasks.add_task(os.remove, zip_path)
        return FileResponse(zip_path, filename=f"batch_{batch_id}.zip", media_type="application/zip")
    except Exception as e:
        logger.error(f"Failed to create ZIP for batch {batch_id}: {e}")
        raise TTSException(f"Failed to create batch download: {str(e)}", 500)

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    REQUEST_COUNT.labels(endpoint="/model/info").inc()
    return ModelInfoResponse(
        model_version="1.0",
        supported_languages=["American English", "British English"],
        model_path=str(Config.MODEL_PATH),
        config_path=str(Config.CONFIG_PATH),
        repo_id=Config.REPO_ID,
        cuda_enabled=resource_manager.cuda_available,
        model_parameters=None
    )

@app.post("/text/segment", response_model=TextSegmentationResponse)
async def text_segmentation(request: TextSegmentationRequest):
    REQUEST_COUNT.labels(endpoint="/text/segment").inc()
    segments = TextPreprocessor.segment_text_by_tokens(request.text, request.max_tokens_per_segment)
    total_tokens = TextPreprocessor.count_tokens(request.text)
    return TextSegmentationResponse(segments=segments, segment_count=len(segments), total_tokens=total_tokens)

@app.get("/cache", response_model=CacheInfoResponse)
async def cache_info():
    REQUEST_COUNT.labels(endpoint="/cache").inc()
    total_size = 0
    items = []
    for key, file_path in audio_cache.items():
        if file_path.exists():
            stat = file_path.stat()
            total_size += stat.st_size
            items.append({"filename": file_path.name, "size_bytes": stat.st_size, "created_at": stat.st_mtime})
    return CacheInfoResponse(total_items=len(items), cache_size_bytes=total_size, items=items)

@app.delete("/cache/{filename}")
async def delete_cache(filename: str):
    REQUEST_COUNT.labels(endpoint="/cache/delete").inc()
    if ".." in filename or "/" in filename or "\\" in filename:
        raise TTSException("Invalid filename", 400)
    file_path = audio_cache.get(filename, Config.AUDIO_OUTPUT_DIR / filename)
    if not file_path.exists():
        raise TTSException("File not found in cache", 404)
    try:
        os.remove(file_path)
        audio_cache.pop(filename, None)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        logger.error(f"Failed to delete cache file {filename}: {e}")
        raise TTSException(f"Failed to delete cache file: {str(e)}", 500)

@app.get("/ratelimit/status", response_model=RateLimitStatusResponse)
async def ratelimit_status():
    REQUEST_COUNT.labels(endpoint="/ratelimit/status").inc()
    return RateLimitStatusResponse(
        enabled=Config.ENABLE_RATE_LIMITING,
        limit=Config.RATE_LIMIT if Config.ENABLE_RATE_LIMITING else None,
        remaining=None,
        reset_timestamp=None
    )

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    REQUEST_COUNT.labels(endpoint="/audio").inc()
    file_path = audio_cache.get(filename, Config.AUDIO_OUTPUT_DIR / filename)
    if not file_path.exists():
        raise TTSException("Audio file not found", 404)
    audio_cache[filename] = file_path
    media_type = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}.get(file_path.suffix, "application/octet-stream")
    return FileResponse(path=file_path, media_type=media_type, filename=filename)

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    REQUEST_COUNT.labels(endpoint="/audio/delete").inc()
    if ".." in filename or "/" in filename or "\\" in filename:
        raise TTSException("Invalid filename", 400)
    file_path = audio_cache.get(filename, Config.AUDIO_OUTPUT_DIR / filename)
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
    preview_requests = [TTSRequest(text=preview_text, voice=voice, format=AudioFormat.WAV) for voice in VoiceOption]
    batch_id = str(uuid.uuid4())
    resource_manager.active_jobs[batch_id] = {
        "status": "queued", "progress": 0, "total_items": len(preview_requests),
        "processed_items": 0, "created_at": time.time()
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
    ffprobe_available = False
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_available = result.returncode == 0
    except FileNotFoundError:
        logger.warning("ffmpeg not found on the system")
    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        ffprobe_available = result.returncode == 0
    except FileNotFoundError:
        logger.warning("ffprobe not found on the system")
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models": model_status,
        "model_files": {"model_exists": Config.MODEL_PATH.exists(), "config_exists": Config.CONFIG_PATH.exists()},
        "disk_space": {"total_gb": round(disk_usage.total / (1024 ** 3), 2), "free_gb": round(disk_usage.free / (1024 ** 3), 2)},
        "memory": {"cuda": torch.cuda.memory_summary(abbreviated=True) if resource_manager.cuda_available else "not available"},
        "cuda_available": resource_manager.cuda_available,
        "ffmpeg_available": ffmpeg_available,
        "ffprobe_available": ffprobe_available,
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
            return {
                "status": "deleted",
                "word": word,
                "language": "American English" if language_code == "a" else "British English"
            }
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
        "max_token_limit": Config.MAX_TOKEN_LIMIT,
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

@app.get("/doc")
async def get_documentation():
    REQUEST_COUNT.labels(endpoint="/doc").inc()
    doc_path = Path("/home/humair/kokoro/doc.html")
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Documentation file not found")
    return FileResponse(doc_path, media_type="text/html")

@app.get("/endpoints")
async def list_endpoints():
    REQUEST_COUNT.labels(endpoint="/endpoints").inc()
    endpoints = {
        "/tts": {"method": "POST", "description": "Synthesizes speech, segmenting if >500 tokens."},
        "/tts/stream": {"method": "POST", "description": "Streams audio without saving to disk."},
        "/tts/batch": {"method": "POST", "description": "Processes multiple TTS requests in batch."},
        "/tts/preview": {"method": "POST", "description": "Generates a short audio preview (first 50 tokens)."},
        "/tts/estimate": {"method": "POST", "description": "Estimates processing time and audio duration."},
        "/status/{job_id}": {"method": "GET", "description": "Checks specific job status."},
        "/status/batch/{batch_id}": {"method": "GET", "description": "Checks batch job and item statuses."},
        "/batch/summary/{batch_id}": {"method": "GET", "description": "Summarizes batch job status."},
        "/batch/download/{batch_id}": {"method": "GET", "description": "Downloads batch audio as ZIP."},
        "/analyze/text": {"method": "POST", "description": "Analyzes text for issues."},
        "/voices/sample": {"method": "POST", "description": "Generates a voice sample."},
        "/voices": {"method": "GET", "description": "Lists available voices."},
        "/voices/preview": {"method": "GET", "description": "Generates previews for all voices."},
        "/pronunciation/validate": {"method": "POST", "description": "Validates custom pronunciation."},
        "/pronunciation": {"method": "POST", "description": "Adds custom pronunciation."},
        "/pronunciations": {"method": "GET", "description": "Lists custom pronunciations."},
        "/pronunciations/{word}": {"method": "DELETE", "description": "Deletes custom pronunciation."},
        "/audio/metadata/{filename}": {"method": "GET", "description": "Retrieves audio file metadata."},
        "/audio/{filename}": {"method": "GET", "description": "Retrieves audio file."},
        "/audio/{filename}": {"method": "DELETE", "description": "Deletes audio file."},
        "/model/info": {"method": "GET", "description": "Returns model information."},
        "/text/segment": {"method": "POST", "description": "Segments text by token limits."},
        "/cache": {"method": "GET", "description": "Lists cached audio files."},
        "/cache/{filename}": {"method": "DELETE", "description": "Deletes cached audio file."},
        "/cache/clear": {"method": "DELETE", "description": "Clears all cached audio files."},
        "/ratelimit/status": {"method": "GET", "description": "Returns rate limit status."},
        "/health": {"method": "GET", "description": "Checks API health."},
        "/cleanup": {"method": "DELETE", "description": "Deletes old audio files."},
        "/preprocess": {"method": "POST", "description": "Preprocesses text."},
        "/metrics": {"method": "GET", "description": "Exposes Prometheus metrics."},
        "/config": {"method": "GET", "description": "Returns API configuration."},
        "/jobs": {"method": "GET", "description": "Lists active and recent jobs."},
        "/jobs/cancel/{job_id}": {"method": "POST", "description": "Cancels a job."},
        "/doc": {"method": "GET", "description": "Serves API documentation HTML."},
        "/endpoints": {"method": "GET", "description": "Lists all API endpoints."},
        "/stats": {"method": "GET", "description": "Returns API usage statistics."}
    }
    return JSONResponse(content=endpoints)

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    REQUEST_COUNT.labels(endpoint="/stats").inc()
    try:
        total_requests = 0
        for metric in REQUEST_COUNT.collect():
            for sample in metric.samples:
                if sample.name == "tts_requests_total":
                    total_requests += sample.value

        latency_sum = 0
        latency_count = 0
        for metric in REQUEST_LATENCY.collect():
            for sample in metric.samples:
                if sample.name == "tts_request_latency_seconds_sum":
                    latency_sum += sample.value
                elif sample.name == "tts_request_latency_seconds_count":
                    latency_count += sample.value

        completed_jobs = sum(1 for job in resource_manager.job_results.values() if job.get("status") == "complete")
        failed_jobs = sum(1 for job in resource_manager.job_results.values() if job.get("status") == "failed")
        avg_processing_time = latency_sum / max(1, latency_count)
        total_duration = sum(job.get("duration", 0) for job in resource_manager.job_results.values() if job.get("status") == "complete")
        cache_hits = getattr(audio_cache, "hits", 0)
        cache_misses = getattr(audio_cache, "misses", 0)
        cache_hit_ratio = cache_hits / max(1, cache_hits + cache_misses)

        return StatsResponse(
            total_requests=int(total_requests),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            average_processing_time_seconds=avg_processing_time,
            total_audio_duration_seconds=total_duration,
            cache_hit_ratio=cache_hit_ratio
        )
    except Exception as e:
        logger.error(f"Error in /stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

@app.post("/tts/preview", response_model=TTSResponse)
async def tts_preview(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    REQUEST_COUNT.labels(endpoint="/tts/preview").inc()
    request_id = f"preview_{str(uuid.uuid4())}"
    token_count = TextPreprocessor.count_tokens(request.text)
    if token_count > 50:
        segments = TextPreprocessor.segment_text_by_tokens(request.text, 50)
        request.text = segments[0]
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0, "created_at": time.time()}
    task = asyncio.create_task(generate_audio(request_id, request))
    resource_manager.job_tasks[request_id] = task
    background_tasks.add_task(lambda: None)
    return TTSResponse(request_id=request_id, status="queued", audio_url=None, duration=None, tokens=None)

@app.get("/jobs", response_model=JobsResponse)
async def list_jobs(status: Optional[str] = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    REQUEST_COUNT.labels(endpoint="/jobs").inc()
    jobs = []
    for job_id, job_data in resource_manager.active_jobs.items():
        if status and job_data["status"] != status:
            continue
        job_entry = {
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data["progress"],
            "created_at": job_data.get("created_at", 0),
            "error": job_data.get("error"),
            "result": resource_manager.job_results.get(job_id)
        }
        jobs.append(job_entry)
    jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)[offset:offset + limit]
    return JobsResponse(jobs=jobs, total_jobs=len(resource_manager.active_jobs), limit=limit, offset=offset)

@app.post("/tts/estimate", response_model=TTSEstimateResponse)
async def tts_estimate(request: TTSRequest):
    REQUEST_COUNT.labels(endpoint="/tts/estimate").inc()
    token_count = TextPreprocessor.count_tokens(request.text)
    segment_count = (token_count + Config.MAX_TOKEN_LIMIT - 1) // Config.MAX_TOKEN_LIMIT
    chars_per_second = 200 / request.speed
    audio_duration = len(request.text) / chars_per_second
    processing_time = token_count * (0.2 if request.use_gpu and resource_manager.cuda_available else 0.5)
    return TTSEstimateResponse(
        estimated_processing_time_seconds=processing_time,
        estimated_audio_duration_seconds=audio_duration,
        token_count=token_count,
        segment_count=segment_count
    )

@app.delete("/cache/clear", response_model=ClearCacheResponse)
async def clear_cache():
    REQUEST_COUNT.labels(endpoint="/cache/clear").inc()
    deleted_files = 0
    errors = 0
    freed_space = 0
    for key, file_path in list(audio_cache.items()):
        if file_path.exists():
            try:
                freed_space += file_path.stat().st_size
                os.remove(file_path)
                deleted_files += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                errors += 1
        audio_cache.pop(key, None)
    return ClearCacheResponse(status="completed", deleted_files=deleted_files, errors=errors, freed_space_bytes=freed_space)

# Server
if __name__ == "__main__":
    logger.info(f"Starting Kokoro TTS API on {Config.DEFAULT_HOST}:{Config.DEFAULT_PORT}")
    uvicorn.run(app, host=Config.DEFAULT_HOST, port=Config.DEFAULT_PORT, log_level="info")
