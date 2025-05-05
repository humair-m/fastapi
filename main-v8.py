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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
    MAX_CONCURRENT_JOBS = 10
    MAX_MODEL_INSTANCES = 2
    THREAD_POOL_SIZE = 4
    PROCESS_POOL_SIZE = 2

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kokoro-tts-api")

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

class PreprocessRequest(BaseModel):
    text: str = Field(..., max_length=Config.MAX_CHAR_LIMIT)

class PreprocessResponse(BaseModel):
    original_text: str
    processed_text: str

# Utility Functions
def validate_json_file(file_path: Path) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error validating JSON file {file_path}: {e}")
        return False

async def ensure_spacy_model():
    try:
        spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' loaded successfully")
    except OSError:
        logger.info("Downloading spaCy model 'en_core_web_sm'")
        try:
            process = await asyncio.create_subprocess_exec(
                "python", "-m", "spacy", "download", "en_core_web_sm",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Failed to download spaCy model: {stderr.decode()}")
                raise RuntimeError("spaCy model download failed")
            spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' downloaded and loaded")
        except Exception as e:
            logger.error(f"Error ensuring spaCy model: {e}")
            raise

# Resource Manager
class ResourceManager:
    def __init__(self):
        self.models: Dict[str, List] = {"cpu": [], "cuda": []}
        self.pipelines: Dict[str, "KPipeline"] = {}
        self.model_lock = asyncio.Lock()
        self.pipeline_locks: Dict[str, asyncio.Lock] = {}
        self.active_jobs: Dict[str, Dict] = {}
        self.job_results: Dict[str, Dict] = {}
        self.job_tasks: Dict[str, asyncio.Task] = {}
        self.cuda_available = torch.cuda.is_available()
        self.job_queue = asyncio.Queue(maxsize=Config.MAX_CONCURRENT_JOBS)
        self.thread_pool = ThreadPoolExecutor(max_workers=Config.THREAD_POOL_SIZE)
        self.process_pool = ProcessPoolExecutor(max_workers=Config.PROCESS_POOL_SIZE)

    async def get_model(self, use_gpu: bool) -> "KModel":
        device = "cuda" if use_gpu and self.cuda_available else "cpu"
        async with self.model_lock:
            if not self.models[device] and len(self.models[device]) < Config.MAX_MODEL_INSTANCES:
                if not Config.MODEL_PATH.exists() or not Config.CONFIG_PATH.exists():
                    raise TTSException("Model or config file not found", 500)
                if not validate_json_file(Config.CONFIG_PATH):
                    raise TTSException("Invalid JSON config file", 500)
                try:
                    from kokoro import KModel
                    model = KModel(config=str(Config.CONFIG_PATH), model=str(Config.MODEL_PATH), repo_id=Config.REPO_ID)
                    if device == "cuda":
                        model = model.to("cuda")
                    self.models[device].append(model.eval())
                except ImportError as e:
                    logger.error(f"Failed to import KModel: {e}")
                    raise TTSException("Kokoro library not installed", 500)
                except Exception as e:
                    logger.error(f"Model initialization failed: {e}")
                    raise TTSException(f"Model initialization failed: {str(e)}", 500)
            return self.models[device][0]

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
                except ImportError as e:
                    logger.error(f"Failed to import KPipeline: {e}")
                    raise TTSException("Kokoro library not installed", 500)
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

    async def process_job_queue(self):
        while True:
            try:
                job_id, job_func = await self.job_queue.get()
                try:
                    await job_func()
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}")
                    self.active_jobs[job_id]["status"] = "failed"
                    self.active_jobs[job_id]["error"] = str(e)
                finally:
                    self.job_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Job queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in job queue: {e}")

# Audio Processor
class AudioProcessor:
    @staticmethod
    def _check_ffmpeg_availability() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"ffmpeg/ffprobe not available: {str(e)}")
            return False

    @staticmethod
    async def save_audio(audio: torch.Tensor, output_path: Path) -> float:
        def _save_audio():
            try:
                scipy.io.wavfile.write(output_path, Config.SAMPLE_RATE, audio.cpu().numpy())
                return len(audio) / Config.SAMPLE_RATE
            except Exception as e:
                raise RuntimeError(f"Failed to write audio: {e}")

        loop = asyncio.get_running_loop()
        try:
            logger.info(f"Saving audio to {output_path}")
            duration = await loop.run_in_executor(resource_manager.thread_pool, _save_audio)
            logger.info(f"Audio saved successfully to {output_path}, duration: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            raise TTSException(f"Failed to save audio: {str(e)}", 500)

    @staticmethod
    async def stream_audio(audio: torch.Tensor, format: AudioFormat) -> AsyncGenerator[bytes, None]:
        buffer = io.BytesIO()
        try:
            scipy.io.wavfile.write(buffer, Config.SAMPLE_RATE, audio.cpu().numpy())
            buffer.seek(0)
        except Exception as e:
            logger.error(f"Failed to write WAV to buffer: {e}")
            raise TTSException("Failed to prepare audio stream", 500)

        if format == AudioFormat.WAV:
            logger.info("Streaming audio in WAV format")
            yield buffer.read()
            return

        if not AudioProcessor._check_ffmpeg_availability():
            logger.warning(f"ffmpeg not available, streaming WAV instead of {format.value}")
            yield buffer.read()
            return

        try:
            logger.info(f"Converting audio to {format.value}")
            audio_segment = AudioSegment.from_wav(buffer)
            output_buffer = io.BytesIO()
            output_format = "mp3" if format == AudioFormat.MP3 else "ogg"
            audio_segment.export(output_buffer, format=output_format)
            output_buffer.seek(0)
            logger.info(f"Streaming audio in {format.value} format")
            while chunk := output_buffer.read(8192):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to convert audio to {format.value}: {e}, falling back to WAV")
            buffer.seek(0)
            yield buffer.read()

    @staticmethod
    async def convert_audio_format(wav_path: Path, output_format: AudioFormat) -> Path:
        if output_format == AudioFormat.WAV or not wav_path.exists():
            return wav_path
        output_path = wav_path.with_suffix(f".{output_format.value}")
        if not AudioProcessor._check_ffmpeg_availability():
            logger.warning(f"ffmpeg not available, returning WAV instead of {output_format.value}")
            return wav_path
        try:
            logger.info(f"Converting {wav_path} to {output_format.value}")
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(wav_path),
                "-acodec", "libmp3lame" if output_format == AudioFormat.MP3 else "libvorbis",
                str(output_path),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"ffmpeg conversion failed: {stderr.decode()}")
                return wav_path
            logger.info(f"Audio converted to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return wav_path

# Text Preprocessor
class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        try:
            text = text.strip().lower()
            text = re.sub(r"[^\w\s.,!?]", "", text)
            abbreviations = {"mr.": "mister", "mrs.": "missus", "dr.": "doctor", "st.": "street"}
            for abbr, full in abbreviations.items():
                text = text.replace(abbr, full)
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise TTSException("Text preprocessing failed", 500)

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    async def segment_text_by_tokens(text: str, max_tokens: int) -> List[str]:
        def _segment_text():
            try:
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
            except Exception as e:
                logger.error(f"Error segmenting text: {e}")
                raise

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(resource_manager.thread_pool, _segment_text)
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}")
            raise TTSException("Text segmentation failed", 500)

# FastAPI Application
resource_manager = ResourceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = None
    try:
        await ensure_spacy_model()
        if Config.ENABLE_RATE_LIMITING:
            redis_pool = redis.ConnectionPool.from_url(Config.REDIS_URL, max_connections=10)
            redis_client = redis.Redis(connection_pool=redis_pool)
            try:
                await redis_client.ping()
                await FastAPILimiter.init(redis_client)
                logger.info("Redis connected and rate limiter initialized")
            except redis.ConnectionError as e:
                logger.error(f"Redis connection failed at {Config.REDIS_URL}: {e}")
                raise RuntimeError("Redis initialization failed")
        await resource_manager.get_model(use_gpu=False)
        for lang_code in ["a", "b"]:
            await resource_manager.get_pipeline(lang_code)
        asyncio.create_task(resource_manager.process_job_queue())
        logger.info("Kokoro TTS API initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        resource_manager.models.clear()
        resource_manager.pipelines.clear()
        resource_manager.thread_pool.shutdown(wait=False)
        resource_manager.process_pool.shutdown(wait=False)
        if redis_client:
            await redis_client.close()
        logger.info("API resources cleaned up")

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
            raise TTSException("Pipeline processing returned no results", 500)

        for i, (_, ps, _) in enumerate(pipeline_results):
            resource_manager.active_jobs[request_id]["progress"] = min(95, int((i + 1) / len(pipeline_results) * 100))
            if i == 0 and request.return_tokens:
                tokens = ps
            if not ps:
                continue
            ps = ps[:len(pack)] if len(ps) > len(pack) else ps
            try:
                def _generate_audio():
                    return model(ps, pack[len(ps)-1], request.speed)

                loop = asyncio.get_running_loop()
                audio = await loop.run_in_executor(resource_manager.process_pool, _generate_audio)
                audio_tensors.append(audio)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and request.use_gpu:
                    logger.warning("CUDA out of memory, falling back to CPU")
                    model = await resource_manager.get_model(False)
                    audio = model(ps, pack[len(ps)-1], request.speed)
                    audio_tensors.append(audio)
                else:
                    raise TTSException(f"Audio generation failed: {str(e)}", 500)

        if not audio_tensors:
            raise TTSException("No audio generated", 500)
        combined_audio = torch.cat(audio_tensors, dim=0) if len(audio_tensors) > 1 else audio_tensors[0]

        if stream:
            media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}[request.format.value]
            return StreamingResponse(
                AudioProcessor.stream_audio(combined_audio, request.format),
                media_type=media_type
            )

        output_path = Config.AUDIO_OUTPUT_DIR / f"{request_id}.wav"
        duration = await AudioProcessor.save_audio(combined_audio, output_path)
        if request.format != AudioFormat.WAV:
            output_path = await AudioProcessor.convert_audio_format(output_path, request.format)

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
        return response
    except Exception as e:
        logger.error(f"Error generating audio (request_id={request_id}): {str(e)}")
        resource_manager.active_jobs[request_id] = {
            "status": "failed", "progress": 0, "error": str(e), "created_at": start_time
        }
        resource_manager.job_results[request_id] = {
            "error": str(e), "request_id": request_id, "status": "failed"
        }
        raise
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

    async def process_item(i, item):
        try:
            request_id = f"{batch_id}_{i}"
            result = await generate_audio(request_id, item)
            return {"index": i, "result": result}
        except TTSException as e:
            return {"index": i, "error": {"text": item.text[:50], "error": e.message}}

    tasks = [process_item(i, item) for i, item in enumerate(items)]
    for future in asyncio.as_completed(tasks):
        result = await future
        if "result" in result:
            results.append(result["result"])
        else:
            errors.append(result["error"])
        processed = len(results) + len(errors)
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
        segments = await TextPreprocessor.segment_text_by_tokens(request.text, Config.MAX_TOKEN_LIMIT)
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
                raise TTSException(f"Segment {i} failed: {result.get('error', 'Unknown error')}", 500)
            audio_path = Config.AUDIO_OUTPUT_DIR / result["audio_url"].split("/")[-1]
            try:
                rate, data = scipy.io.wavfile.read(audio_path)
                audio_tensors.append(torch.tensor(data, dtype=torch.float32))
                total_duration += result["duration"]
                if result["tokens"]:
                    tokens = result["tokens"]
                os.remove(audio_path)
                audio_cache.pop(segment_id, None)
            except Exception as e:
                logger.error(f"Failed to process segment {segment_id}: {e}")
                raise TTSException("Segment processing failed", 500)
        combined_audio = torch.cat(audio_tensors, dim=0)
        output_path = Config.AUDIO_OUTPUT_DIR / f"{request_id}.wav"
        duration = await AudioProcessor.save_audio(combined_audio, output_path)
        if request.format != AudioFormat.WAV:
            output_path = await AudioProcessor.convert_audio_format(output_path, request.format)
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
        return response
    async def job_func():
        return await generate_audio(request_id, request)
    await resource_manager.job_queue.put((request_id, job_func))
    resource_manager.job_tasks[request_id] = asyncio.create_task(job_func())
    background_tasks.add_task(lambda: None)
    return TTSResponse(request_id=request_id, status="queued", audio_url=None, duration=None, tokens=None)

@app.post("/tts/stream")
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
    async def job_func():
        await process_batch(batch_id, request.items)
    await resource_manager.job_queue.put((batch_id, job_func))
    resource_manager.job_tasks[batch_id] = asyncio.create_task(job_func())
    background_tasks.add_task(lambda: None)
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=total_items)

@app.get("/status/{job_id}", response_model=JobStatus)
async def check_job_status(job_id: str):
    REQUEST_COUNT.labels(endpoint="/status").inc()
    if job_id not in resource_manager.active_jobs:
        raise TTSException("Job not found", 404)
    status_data = resource_manager.active_jobs[job_id].copy()
    status_data.setdefault("error", None)
    status_data.setdefault("result", resource_manager.job_results.get(job_id))
    return JobStatus(**status_data)

@app.get("/status/batch/{batch_id}")
async def check_batch_status(batch_id: str):
    REQUEST_COUNT.labels(endpoint="/status/batch").inc()
    if batch_id not in resource_manager.active_jobs:
        raise TTSException("Batch not found", 404)
    batch_status = resource_manager.active_jobs[batch_id].copy()
    item_statuses = [
        {
            "item_id": f"{batch_id}_{i}",
            **resource_manager.active_jobs.get(f"{batch_id}_{i}", {"status": "unknown", "progress": 0}),
            "result": resource_manager.job_results.get(f"{batch_id}_{i}")
        }
        for i in range(batch_status.get("total_items", 0))
    ]
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
        audio_urls = [r["audio_url"] for r in results]
    return BatchSummaryResponse(
        batch_id=batch_id, status=batch_status["status"], progress=batch_status["progress"],
        total_items=batch_status["total_items"], completed_items=completed_items,
        failed_items=failed_items, audio_urls=audio_urls
    )

@app.post("/analyze/text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    REQUEST_COUNT.labels(endpoint="/analyze/text").inc()
    text = request.text
    char_count = len(text)
    warnings = []
    suggestions = []
    if char_count > Config.MAX_CHAR_LIMIT:
        warnings.append(f"Text exceeds {Config.MAX_CHAR_LIMIT} characters")
        suggestions.append(f"Truncate to {Config.MAX_CHAR_LIMIT} characters")
    if re.search(r"[^\w\s.,!?]", text):
        warnings.append("Unsupported special characters detected")
        suggestions.append("Remove or replace special characters")
    if "mr." in text.lower():
        suggestions.append("Replace 'Mr.' with 'Mister' for better pronunciation")
    estimated_duration = char_count / 200 * 60 / Config.SAMPLE_RATE
    return TextAnalysisResponse(
        text=text, char_count=char_count, warnings=warnings,
        suggestions=suggestions, estimated_duration=estimated_duration
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
    async def job_func():
        return await generate_audio(request_id, tts_request)
    await resource_manager.job_queue.put((request_id, job_func))
    resource_manager.job_tasks[request_id] = asyncio.create_task(job_func())
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
        raise TTSException("No audio files available", 404)
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
        raise TTSException(f"Batch download failed: {str(e)}", 500)

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
    segments = await TextPreprocessor.segment_text_by_tokens(request.text, request.max_tokens_per_segment)
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
        raise TTSException(f"Cache deletion failed: {str(e)}", 500)

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
        logger.error(f"Failed to delete audio file {filename}: {e}")
        raise TTSException(f"File deletion failed: {str(e)}", 500)

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
    async def job_func():
        await process_batch(batch_id, preview_requests)
    await resource_manager.job_queue.put((batch_id, job_func))
    resource_manager.job_tasks[batch_id] = asyncio.create_task(job_func())
    background_tasks.add_task(lambda: None)
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=len(preview_requests))

@app.get("/health")
async def health_check():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    model_status = {"cpu": "available" if resource_manager.models["cpu"] else "not loaded"}
    if resource_manager.cuda_available:
        model_status["gpu"] = "available" if resource_manager.models["cuda"] else "not loaded"
    disk_usage = shutil.disk_usage(Config.AUDIO_OUTPUT_DIR)
    ffmpeg_available = AudioProcessor._check_ffmpeg_availability()
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models": model_status,
        "model_files": {"model_exists": Config.MODEL_PATH.exists(), "config_exists": Config.CONFIG_PATH.exists()},
        "disk_space": {"total_gb": round(disk_usage.total / (1024 ** 3), 2), "free_gb": round(disk_usage.free / (1024 ** 3), 2)},
        "memory": {"cuda": torch.cuda.memory_summary(abbreviated=True) if resource_manager.cuda_available else "N/A"},
        "cuda_available": resource_manager.cuda_available,
        "ffmpeg_available": ffmpeg_available,
        "active_jobs": len(resource_manager.active_jobs),
        "cached_files": len(audio_cache),
        "rate_limiting_enabled": Config.ENABLE_RATE_LIMITING
    }

@app.delete("/cleanup")
async def cleanup_old_files(hours: int = Query(default=Config.CLEANUP_HOURS, ge=1)):
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
    pipeline = await resource_manager.get_pipeline(language_code)
    pipeline.g2p.lexicon.golds[word.lower()] = pronunciation
    return {
        "status": "success",
        "word": word,
        "pronunciation": pronunciation,
        "language": "American English" if language_code == "a" else "British English"
    }

@app.get("/pronunciations")
async def list_pronunciations(language_code: str = Query("a", enum=["a", "b"])):
    REQUEST_COUNT.labels(endpoint="/pronunciations").inc()
    pipeline = await resource_manager.get_pipeline(language_code)
    return {
        "language": "American English" if language_code == "a" else "British English",
        "pronunciations": pipeline.g2p.lexicon.golds
    }

@app.delete("/pronunciations/{word}")
async def delete_pronunciation(word: str, language_code: str = Query("a", enum=["a", "b"])):
    REQUEST_COUNT.labels(endpoint="/pronunciations/delete").inc()
    pipeline = await resource_manager.get_pipeline(language_code)
    word_lower = word.lower()
    if word_lower in pipeline.g2p.lexicon.golds:
        del pipeline.g2p.lexicon.golds[word_lower]
        return {
            "status": "deleted",
            "word": word,
            "language": "American English" if language_code == "a" else "British English"
        }
    raise TTSException(f"Pronunciation for '{word}' not found", 404)

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

if __name__ == "__main__":
    logger.info(f"Starting Kokoro TTS API on {Config.DEFAULT_HOST}:{Config.DEFAULT_PORT}")
    uvicorn.run(app, host=Config.DEFAULT_HOST, port=Config.DEFAULT_PORT, log_level="info")
