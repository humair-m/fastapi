import asyncio
import io
import json
import logging
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import redis.asyncio as redis
import spacy
import torch
import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Configuration
class Config:
    AUDIO_OUTPUT_DIR = Path("./audio_output")
    SAMPLE_RATE = 24000
    MAX_CHAR_LIMIT = 100000  # Increased to handle longer texts
    MAX_TOKEN_LIMIT = 500
    MODEL_PATH = Path("/home/humair/kokoro/kokoro-v1_0.pth")
    CONFIG_PATH = Path("/home/humair/kokoro/config.json")
    DEFAULT_PORT = int(os.environ.get("PORT", 8000))
    DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
    CACHE_TTL = 3600
    RATE_LIMIT = "10/minute"
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    ENABLE_RATE_LIMITING = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
    REPO_ID = "hexgrad/Kokoro-82M"
    MAX_CONCURRENT_JOBS = 10
    THREAD_POOL_SIZE = 4
    PROCESS_POOL_SIZE = 2

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kokoro-tts-api")
Config.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

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

class TTSResponse(BaseModel):
    audio_url: Optional[str]
    duration: Optional[float]
    tokens: Optional[str]
    request_id: str
    status: str

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
    except OSError:
        logger.info("Downloading spaCy model 'en_core_web_sm'")
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "spacy", "download", "en_core_web_sm",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"Failed to download spaCy model: {stderr.decode()}")
            raise RuntimeError("spaCy model download failed")

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
            if not self.models[device]:
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
                    self.pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False, repo_id=Config.REPO_ID)
                except Exception as e:
                    logger.error(f"Pipeline initialization failed for {lang_code}: {e}")
                    raise TTSException(f"Pipeline initialization failed: {str(e)}", 500)
            return self.pipelines[lang_code]

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
                break

# Audio Processor
class AudioProcessor:
    @staticmethod
    async def save_audio(audio: torch.Tensor, output_path: Path) -> float:
        def _save_audio():
            import scipy.io.wavfile
            scipy.io.wavfile.write(output_path, Config.SAMPLE_RATE, audio.cpu().numpy())
            return len(audio) / Config.SAMPLE_RATE

        loop = asyncio.get_running_loop()
        duration = await loop.run_in_executor(resource_manager.thread_pool, _save_audio)
        return duration

    @staticmethod
    async def stream_audio(audio: torch.Tensor, format: AudioFormat) -> AsyncGenerator[bytes, None]:
        buffer = io.BytesIO()
        import scipy.io.wavfile
        scipy.io.wavfile.write(buffer, Config.SAMPLE_RATE, audio.cpu().numpy())
        buffer.seek(0)
        if format == AudioFormat.WAV:
            yield buffer.read()
            return
        audio_segment = AudioSegment.from_wav(buffer)
        output_buffer = io.BytesIO()
        output_format = "mp3" if format == AudioFormat.MP3 else "ogg"
        audio_segment.export(output_buffer, format=output_format)
        output_buffer.seek(0)
        while chunk := output_buffer.read(8192):
            yield chunk

# Text Preprocessor
class TextPreprocessor:
    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    async def segment_text_by_tokens(text: str, max_tokens: int) -> List[str]:
        def _segment_text():
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
            if current_segment:
                segments.append(current_segment)
            return segments

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(resource_manager.thread_pool, _segment_text)

# FastAPI Application
resource_manager = ResourceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_spacy_model()
    if Config.ENABLE_RATE_LIMITING:
        redis_pool = redis.ConnectionPool.from_url(Config.REDIS_URL, max_connections=10)
        redis_client = redis.Redis(connection_pool=redis_pool)
        await FastAPILimiter.init(redis_client)
    await resource_manager.get_model(use_gpu=False)
    for lang_code in ["a", "b"]:
        await resource_manager.get_pipeline(lang_code)
    asyncio.create_task(resource_manager.process_job_queue())
    yield
    resource_manager.models.clear()
    resource_manager.pipelines.clear()
    resource_manager.thread_pool.shutdown(wait=False)
    resource_manager.process_pool.shutdown(wait=False)
    if Config.ENABLE_RATE_LIMITING:
        await redis_client.close()

app = FastAPI(title="Kokoro TTS API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(TTSException)
async def tts_exception_handler(request: Request, exc: TTSException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

async def generate_audio_segment(segment_request: TTSRequest, model, pipeline, pack):
    pipeline_results = list(pipeline(segment_request.text, segment_request.voice, segment_request.speed))
    audio_tensors = []
    tokens = None
    for i, (_, ps, _) in enumerate(pipeline_results):
        if i == 0 and segment_request.return_tokens:
            tokens = ps
        if not ps:
            continue
        ps = ps[:len(pack)] if len(ps) > len(pack) else ps
        def _generate_audio():
            return model(ps, pack[len(ps)-1], segment_request.speed)
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(resource_manager.process_pool, _generate_audio)
        audio_tensors.append(audio)
    if not audio_tensors:
        raise TTSException("No audio generated for segment", 500)
    combined_audio = torch.cat(audio_tensors, dim=0) if len(audio_tensors) > 1 else audio_tensors[0]
    duration = len(combined_audio) / Config.SAMPLE_RATE
    return {"audio_tensor": combined_audio, "duration": duration, "tokens": tokens}

async def generate_audio(request_id: str, request: TTSRequest, stream: bool = False):
    start_time = time.time()
    resource_manager.active_jobs[request_id] = {"status": "processing", "progress": 0, "created_at": start_time}
    lang_code = request.voice.value[0]
    pipeline = await resource_manager.get_pipeline(lang_code)
    if request.pronunciations:
        for word, pron in request.pronunciations.items():
            pipeline.g2p.lexicon.golds[word.lower()] = pron
    pack = pipeline.load_voice(request.voice)
    model = await resource_manager.get_model(request.use_gpu and resource_manager.cuda_available)

    token_count = TextPreprocessor.count_tokens(request.text)
    if token_count > Config.MAX_TOKEN_LIMIT:
        segments = await TextPreprocessor.segment_text_by_tokens(request.text, Config.MAX_TOKEN_LIMIT)
        segment_requests = [
            TTSRequest(
                text=segment, voice=request.voice, speed=request.speed, use_gpu=request.use_gpu,
                return_tokens=request.return_tokens and i == 0, format=AudioFormat.WAV,
                pronunciations=request.pronunciations
            ) for i, segment in enumerate(segments)
        ]
        segment_tasks = [
            generate_audio_segment(segment_request, model, pipeline, pack)
            for segment_request in segment_requests
        ]
        segment_results = await asyncio.gather(*segment_tasks)
        audio_tensors = [result["audio_tensor"] for result in segment_results]
        total_duration = sum(result["duration"] for result in segment_results)
        tokens = next((result["tokens"] for result in segment_results if result["tokens"]), None)
        combined_audio = torch.cat(audio_tensors, dim=0)
    else:
        result = await generate_audio_segment(request, model, pipeline, pack)
        combined_audio = result["audio_tensor"]
        total_duration = result["duration"]
        tokens = result["tokens"]

    if stream:
        media_type = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}[request.format.value]
        return StreamingResponse(AudioProcessor.stream_audio(combined_audio, request.format), media_type=media_type)

    output_path = Config.AUDIO_OUTPUT_DIR / f"{request_id}.{request.format.value}"
    duration = await AudioProcessor.save_audio(combined_audio, output_path)
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
    return response

# Endpoints
@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    background_tasks: BackgroundTasks,
    limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60)) if Config.ENABLE_RATE_LIMITING else None
):
    request_id = str(uuid.uuid4())
    resource_manager.active_jobs[request_id] = {"status": "queued", "progress": 0, "created_at": time.time()}
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

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    file_path = audio_cache.get(filename, Config.AUDIO_OUTPUT_DIR / filename)
    if not file_path.exists():
        raise TTSException("Audio file not found", 404)
    audio_cache[filename] = file_path
    media_type = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}.get(file_path.suffix, "application/octet-stream")
    return FileResponse(path=file_path, media_type=media_type, filename=filename)

if __name__ == "__main__":
    logger.info(f"Starting Kokoro TTS API on {Config.DEFAULT_HOST}:{Config.DEFAULT_PORT}")
    uvicorn.run(app, host=Config.DEFAULT_HOST, port=Config.DEFAULT_PORT, log_level="info")
