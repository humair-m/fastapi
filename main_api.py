from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request, status
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel, Field, validator
import uvicorn
import os
import torch
import uuid
import time
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import asyncio
import json
import logging
from pathlib import Path
import traceback
import shutil
import scipy.io.wavfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kokoro-tts-api")

# Create output directory for audio files
AUDIO_OUTPUT_DIR = Path("./audio_output")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# Environment and configuration setup
CUDA_AVAILABLE = torch.cuda.is_available()
SAMPLE_RATE = 24000
MAX_CHAR_LIMIT = 5000

# Guard clause for Kokoro imports
try:
    from kokoro import KModel, KPipeline
except ImportError as e:
    logger.error(f"Failed to import Kokoro: {e}")
    raise ImportError(f"Kokoro library not found. Please install it first. Error: {e}")

# Define the available voice options as an Enum for validation
class VoiceOption(str, Enum):
    # American English Female Voices
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
    
    # American English Male Voices
    MICHAEL = "am_michael"
    FENRIR = "am_fenrir"
    PUCK = "am_puck"
    ECHO = "am_echo"
    ERIC = "am_eric"
    LIAM = "am_liam"
    ONYX = "am_onyx"
    SANTA = "am_santa"
    ADAM = "am_adam"
    
    # British English Female Voices
    EMMA = "bf_emma"
    ISABELLA = "bf_isabella"
    ALICE = "bf_alice"
    LILY = "bf_lily"
    
    # British English Male Voices
    GEORGE = "bm_george"
    FABLE = "bm_fable"
    LEWIS = "bm_lewis"
    DANIEL = "bm_daniel"

# Define output format enum
class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"

# Data models
class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech", max_length=MAX_CHAR_LIMIT)
    voice: VoiceOption = Field(default=VoiceOption.HEART, description="The voice to use for synthesis")
    speed: float = Field(default=1.0, description="Speech speed factor (0.5-2.0)", ge=0.5, le=2.0)
    use_gpu: bool = Field(default=True, description="Whether to use GPU for inference if available")
    return_tokens: bool = Field(default=False, description="Whether to return tokenization information")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Output audio format")
    pronunciations: Optional[Dict[str, str]] = Field(default=None, description="Custom word pronunciations")
    
    @validator('text')
    def validate_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class TTSBatchRequest(BaseModel):
    items: List[TTSRequest] = Field(..., description="List of TTS requests to process")
    
    @validator('items')
    def validate_items_not_empty(cls, v):
        if not v:
            raise ValueError("Batch request must contain at least one item")
        return v

class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    tokens: Optional[str] = None
    request_id: str
    status: str

class TTSBatchResponse(BaseModel):
    batch_id: str
    status: str
    total_items: int

class JobStatus(BaseModel):
    status: str
    progress: int
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# Initialize FastAPI
app = FastAPI(
    title="Kokoro TTS API",
    description="API for Kokoro Text-to-Speech synthesis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state with proper locking
models = {}
pipelines = {}
active_jobs = {}
job_results = {}
model_lock = asyncio.Lock()
pipeline_locks = {}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Model initialization with error handling
async def get_model(use_gpu=False):
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    key = device
    
    async with model_lock:
        if key not in models:
            try:
                logger.info(f"Initializing model on {device}")
                
                # Check model path existence
                model_path = Path('c:/kuku/Kokoro-82M/kokoro-v1_0.pth')
                config_path = Path('c:/kuku/Kokoro-82M/config.json')
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                if not config_path.exists():
                    raise FileNotFoundError(f"Config file not found at {config_path}")
                
                model = KModel(
                    config=str(config_path),
                    model=str(model_path)
                )
                
                if device == "cuda":
                    model = model.to("cuda")
                
                models[key] = model.eval()
            except Exception as e:
                logger.error(f"Failed to initialize model on {device}: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    return models[key]

# Initialize pipeline for a language code with error handling
async def get_pipeline(lang_code):
    if lang_code not in pipeline_locks:
        pipeline_locks[lang_code] = asyncio.Lock()
    
    async with pipeline_locks[lang_code]:
        if lang_code not in pipelines:
            try:
                logger.info(f"Initializing pipeline for language code: {lang_code}")
                pipeline = KPipeline(lang_code=lang_code, model=False)
                
                # Add default pronunciation rules
                if lang_code == 'a':
                    pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
                elif lang_code == 'b':
                    pipeline.g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
                    
                pipelines[lang_code] = pipeline
            except Exception as e:
                logger.error(f"Failed to initialize pipeline for {lang_code}: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")
    
    return pipelines[lang_code]

# Audio utilities with better error handling
def convert_audio_format(wav_path, output_format):
    """Convert WAV to other formats using ffmpeg"""
    if output_format == AudioFormat.WAV:
        return wav_path
    
    output_path = wav_path.with_suffix(f".{output_format}")
    
    try:
        import subprocess
        process = subprocess.run([
            "ffmpeg", "-y", "-i", str(wav_path), 
            "-acodec", "libmp3lame" if output_format == AudioFormat.MP3 else "libvorbis",
            str(output_path)
        ], check=True, capture_output=True)
        
        # Verify the output file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.error(f"Conversion failed. ffmpeg stdout: {process.stdout.decode()}")
            logger.error(f"ffmpeg stderr: {process.stderr.decode()}")
            return wav_path
            
        return output_path
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        logger.error(traceback.format_exc())
        return wav_path

# TTS generation function with improved error handling
async def generate_audio(
    request_id: str,
    text: str, 
    voice: str = VoiceOption.HEART, 
    speed: float = 1.0, 
    use_gpu: bool = True,
    return_tokens: bool = False,
    format: AudioFormat = AudioFormat.WAV,
    pronunciations: Optional[Dict[str, str]] = None
):
    try:
        # Update job status
        active_jobs[request_id] = {"status": "processing", "progress": 0}
        
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Limit text length
        text = text.strip()[:MAX_CHAR_LIMIT]
        
        # Get the appropriate pipeline based on the first character of voice ID
        lang_code = voice[0]
        if lang_code not in ['a', 'b']:
            raise ValueError(f"Invalid voice ID prefix: {lang_code}. Must be 'a' or 'b'")
            
        pipeline = await get_pipeline(lang_code)
        
        # Add custom pronunciations if provided
        if pronunciations:
            for word, pron in pronunciations.items():
                pipeline.g2p.lexicon.golds[word.lower()] = pron
        
        # Load voice data
        try:
            pack = pipeline.load_voice(voice)
        except Exception as e:
            logger.error(f"Failed to load voice '{voice}': {e}")
            raise ValueError(f"Failed to load voice '{voice}': {str(e)}")
        
        # Use GPU if requested and available
        use_gpu = use_gpu and CUDA_AVAILABLE
        device = "cuda" if use_gpu else "cpu"
        model = await get_model(use_gpu)
        
        # Process the text
        tokens = None
        audio_tensors = []
        
        # Try to process the text through the pipeline
        pipeline_results = list(pipeline(text, voice, speed))
        if not pipeline_results:
            raise ValueError("Failed to process text through pipeline")
        
        for i, (_, ps, _) in enumerate(pipeline_results):
            # Update progress
            progress = min(95, int((i / max(1, len(pipeline_results))) * 100))
            active_jobs[request_id]["progress"] = progress
            
            # Save tokens from first phrase if requested
            if i == 0 and return_tokens:
                tokens = ps
            
            if len(ps) == 0:
                logger.warning(f"Empty phoneme sequence for segment {i}")
                continue
                
            # Ensure we have reference style
            if len(ps) - 1 >= len(pack):
                logger.warning(f"Phoneme sequence too long for voice pack. Truncating.")
                ps = ps[:len(pack)]
                
            ref_s = pack[len(ps)-1]
            
            try:
                audio = model(ps, ref_s, speed)
                audio_tensors.append(audio)
            except RuntimeError as e:
                # Specific handling for CUDA out of memory errors
                if "CUDA out of memory" in str(e) and use_gpu:
                    logger.warning(f"CUDA out of memory, falling back to CPU")
                    model = await get_model(False)
                    audio = model(ps, ref_s, speed)
                    audio_tensors.append(audio)
                elif use_gpu:
                    logger.warning(f"GPU inference failed: {e}, falling back to CPU")
                    model = await get_model(False)
                    audio = model(ps, ref_s, speed)
                    audio_tensors.append(audio)
                else:
                    raise
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                raise
            
            # Yield to allow other tasks to run
            await asyncio.sleep(0)
        
        # Combine audio tensors if there are multiple
        if len(audio_tensors) > 1:
            combined_audio = torch.cat(audio_tensors, dim=0)
        elif len(audio_tensors) == 1:
            combined_audio = audio_tensors[0]
        else:
            raise ValueError("No audio generated")
        
        # Save audio to file
        output_filename = f"{request_id}.wav"
        output_path = AUDIO_OUTPUT_DIR / output_filename
        
        try:
            scipy.io.wavfile.write(
                output_path, 
                SAMPLE_RATE, 
                combined_audio.cpu().numpy()
            )
        except Exception as e:
            logger.error(f"Failed to write WAV file: {e}")
            raise RuntimeError(f"Failed to save audio file: {str(e)}")
        
        # Convert to requested format if not WAV
        if format != AudioFormat.WAV:
            output_path = convert_audio_format(output_path, format)
        
        # Calculate duration
        duration = len(combined_audio) / SAMPLE_RATE
        
        # Update job status to complete
        active_jobs[request_id]["status"] = "complete"
        active_jobs[request_id]["progress"] = 100
        
        # Prepare response
        response = {
            "audio_url": f"/audio/{output_path.name}",
            "duration": duration,
            "request_id": request_id,
            "status": "complete"
        }
        
        if return_tokens:
            response["tokens"] = tokens
        
        job_results[request_id] = response
        return response
        
    except Exception as e:
        logger.error(f"Error generating audio for request {request_id}: {e}")
        logger.error(traceback.format_exc())
        active_jobs[request_id] = {
            "status": "failed", 
            "progress": 0,
            "error": str(e)
        }
        job_results[request_id] = {
            "error": str(e),
            "request_id": request_id,
            "status": "failed"
        }
        raise

# Background task for batch processing with improved error handling
async def process_batch(batch_id: str, items: List[TTSRequest]):
    results = []
    errors = []
    total_items = len(items)
    processed = 0
    
    active_jobs[batch_id] = {
        "status": "processing", 
        "progress": 0,
        "total_items": total_items,
        "processed_items": 0
    }
    
    for i, item in enumerate(items):
        try:
            request_id = f"{batch_id}_{i}"
            result = await generate_audio(
                request_id=request_id,
                text=item.text,
                voice=item.voice,
                speed=item.speed,
                use_gpu=item.use_gpu and CUDA_AVAILABLE,
                return_tokens=item.return_tokens,
                format=item.format,
                pronunciations=item.pronunciations
            )
            results.append(result)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing batch item {i}: {error_msg}")
            errors.append({
                "index": i,
                "text": item.text[:50] + "..." if len(item.text) > 50 else item.text,
                "error": error_msg
            })
        finally:
            processed += 1
            progress = int((processed / total_items) * 100)
            active_jobs[batch_id]["progress"] = progress
            active_jobs[batch_id]["processed_items"] = processed
    
    job_results[batch_id] = {
        "results": results,
        "errors": errors,
        "total_items": total_items,
        "processed_items": processed
    }
    
    active_jobs[batch_id]["status"] = "complete"
    active_jobs[batch_id]["progress"] = 100

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    try:
        # Pre-initialize CPU model
        await get_model(use_gpu=False)
        
        # Initialize pipelines for language codes
        await get_pipeline('a')  # American English
        await get_pipeline('b')  # British English
        
        logger.info("Kokoro TTS API initialized and ready")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        logger.error(traceback.format_exc())

@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Convert text to speech and return audio data
    """
    request_id = str(uuid.uuid4())
    
    # Validate request
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Add to active jobs
    active_jobs[request_id] = {"status": "queued", "progress": 0}
    
    # Start processing in background
    background_tasks.add_task(
        generate_audio,
        request_id=request_id,
        text=request.text,
        voice=request.voice,
        speed=request.speed,
        use_gpu=request.use_gpu and CUDA_AVAILABLE,
        return_tokens=request.return_tokens,
        format=request.format,
        pronunciations=request.pronunciations
    )
    
    # Return job ID for status checking
    return TTSResponse(request_id=request_id, status="queued")

@app.post("/tts/batch", response_model=TTSBatchResponse)
async def batch_synthesize_speech(request: TTSBatchRequest, background_tasks: BackgroundTasks):
    """
    Process multiple TTS requests in batch
    """
    # Validate request
    if not request.items:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one item")
    
    batch_id = str(uuid.uuid4())
    total_items = len(request.items)
    
    # Add to active jobs
    active_jobs[batch_id] = {
        "status": "queued", 
        "progress": 0,
        "total_items": total_items,
        "processed_items": 0
    }
    
    # Start batch processing in background
    background_tasks.add_task(
        process_batch,
        batch_id=batch_id,
        items=request.items
    )
    
    # Return batch ID for status checking
    return TTSBatchResponse(batch_id=batch_id, status="queued", total_items=total_items)

@app.get("/status/{job_id}", response_model=JobStatus)
async def check_job_status(job_id: str):
    """
    Check the status of a TTS job
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_data = active_jobs[job_id].copy()
    
    # If job is complete and has results, include them
    if status_data["status"] == "complete" and job_id in job_results:
        status_data["result"] = job_results[job_id]
    
    return JobStatus(**status_data)

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve generated audio files
    """
    file_path = AUDIO_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        return FileResponse(
            path=file_path,
            media_type="audio/wav" if filename.endswith(".wav") else 
                       "audio/mpeg" if filename.endswith(".mp3") else
                       "audio/ogg",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve audio file: {str(e)}")

@app.get("/voices")
async def list_available_voices():
    """
    Get a list of all available voices with metadata
    """
    voices = {
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
    
    return voices

@app.get("/health")
async def health_check():
    """
    Check API health status
    """
    model_status = {}
    
    # Check model initialization
    try:
        await get_model(use_gpu=False)
        model_status["cpu"] = "available"
    except Exception as e:
        model_status["cpu"] = f"error: {str(e)}"
    
    if CUDA_AVAILABLE:
        try:
            await get_model(use_gpu=True)
            model_status["gpu"] = "available"
        except Exception as e:
            model_status["gpu"] = f"error: {str(e)}"
    else:
        model_status["gpu"] = "not available"
    
    # Check disk space for audio output
    try:
        disk_usage = shutil.disk_usage(AUDIO_OUTPUT_DIR)
        disk_space = {
            "total_gb": round(disk_usage.total / (1024 ** 3), 2),
            "used_gb": round(disk_usage.used / (1024 ** 3), 2),
            "free_gb": round(disk_usage.free / (1024 ** 3), 2),
            "percent_used": round(disk_usage.used * 100 / disk_usage.total, 2)
        }
    except Exception as e:
        disk_space = {"error": str(e)}
    
    # Check model files
    model_path = Path('c:/kuku/Kokoro-82M/kokoro-v1_0.pth')
    config_path = Path('c:/kuku/Kokoro-82M/config.json')
    model_files = {
        "model_exists": model_path.exists(),
        "config_exists": config_path.exists()
    }
    
    # Check for ffmpeg
    ffmpeg_available = False
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ffmpeg_available = result.returncode == 0
    except Exception:
        pass
    
    # Check memory usage
    memory_info = {}
    if CUDA_AVAILABLE:
        try:
            # Get CUDA memory info
            torch_mem = torch.cuda.memory_summary(abbreviated=True)
            memory_info["cuda"] = torch_mem
        except Exception as e:
            memory_info["cuda_error"] = str(e)
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info["system"] = {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "percent_used": memory.percent
        }
    except ImportError:
        memory_info["system"] = "psutil not installed"
    except Exception as e:
        memory_info["system_error"] = str(e)
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models": model_status,
        "model_files": model_files,
        "disk_space": disk_space,
        "memory": memory_info,
        "active_jobs": len(active_jobs),
        "cuda_available": CUDA_AVAILABLE,
        "ffmpeg_available": ffmpeg_available
    }

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    """
    Delete a generated audio file
    """
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = AUDIO_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        os.remove(file_path)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.delete("/cleanup")
async def cleanup_old_files(hours: int = Query(24, description="Delete files older than this many hours")):
    """
    Delete audio files older than the specified time
    """
    if hours < 1:
        raise HTTPException(status_code=400, detail="Hours must be at least 1")
    
    deleted_count = 0
    error_count = 0
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    
    for file_path in AUDIO_OUTPUT_DIR.glob("*.*"):
        if file_path.is_file():
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                        error_count += 1
            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")
                error_count += 1
    
    return {
        "status": "completed", 
        "deleted_files": deleted_count,
        "errors": error_count
    }

@app.post("/pronunciation")
async def add_custom_pronunciation(
    word: str, 
    pronunciation: str,
    language_code: str = Query("a", description="Language code: 'a' for American English, 'b' for British English")
):
    """
    Add a custom word pronunciation to the lexicon
    """
    if not word or not word.strip():
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    
    if not pronunciation or not pronunciation.strip():
        raise HTTPException(status_code=400, detail="Pronunciation cannot be empty")
    
    if language_code not in ["a", "b"]:
        raise HTTPException(status_code=400, detail="Invalid language code, must be 'a' or 'b'")
    
    try:
        pipeline = await get_pipeline(language_code)
        pipeline.g2p.lexicon.golds[word.lower()] = pronunciation
        
        return {
            "status": "success", 
            "word": word, 
            "pronunciation": pronunciation, 
            "language": "American English" if language_code == "a" else "British English"
        }
    except Exception as e:
        logger.error(f"Failed to add pronunciation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add pronunciation: {str(e)}")

# Error handling specific routes
@app.get("/test_error")
async def test_error():
    """
    Test route to trigger an error (for debugging)
    """
    class TestError(Exception):
        pass
    
    raise TestError("This is a test error")

# Run the server
if __name__ == "__main__":
    import sys
    
    # Check if port is specified as command line argument
    port = 8000
    host = "0.0.0.0"
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # Log startup information
    logger.info(f"Starting Kokoro TTS API server on {host}:{port}")
    logger.info(f"CUDA available: {CUDA_AVAILABLE}")
    
    # Start the server
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
