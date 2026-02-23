"""
Speech API Endpoints - STT and TTS

Provides REST API for:
- Speech-to-Text (STT) using faster-whisper
- Text-to-Speech (TTS) using various engines
- Whisper configuration management
"""

import logging
import tempfile
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from backend.dependencies import get_config_dependency, get_tts_manager, reset_tts_manager
from backend.models.speech import (
    ActiveVoiceRequest,
    ActiveVoiceResponse,
    ClonedVoiceInfo,
    ClonedVoicesListResponse,
    EngineSelectRequest,
    EngineSelectResponse,
    SpeechHealthResponse,
    TranscriptionResponse,
    TTSAsyncRequest,
    TTSConfigResponse,
    TTSConfigUpdate,
    TTSPreviewRequest,
    TTSRequest,
    TTSResponse,
    TTSTaskResponse,
    TTSTaskStatus,
    TTSVoiceUpdate,
    VoiceCloneRequest,
    VoiceCloneResponse,
    VoiceListResponse,
    WhisperConfigResponse,
    WhisperConfigUpdate,
)
from src.utilities.config import OrionConfig
from src.utilities.whisper_manager import get_whisper_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/speech", tags=["Speech"])


# ========== SPEECH-TO-TEXT (STT) ENDPOINTS ==========
@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio to text",
    description="Convert audio file to text using Whisper STT",
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WebM, WAV, MP3, M4A, etc.)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en'). None = auto-detect"),
    config: OrionConfig = Depends(get_config_dependency),
) -> TranscriptionResponse:
    """
    Transcribe audio file to text using Whisper.
    
    Accepts audio files in various formats:
    - WebM (from browser recording)
    - WAV, MP3, M4A, FLAC, OGG
    
    Args:
        audio: Audio file to transcribe
        language: Optional language code ('en', 'es', etc.). Auto-detects if not provided
        config: Configuration dependency (injected)
    
    Returns:
        TranscriptionResponse with transcribed text, detected language, and duration
    
    Raises:
        HTTPException 400: Invalid file or empty audio
        HTTPException 413: File too large
        HTTPException 500: Transcription failed
        
    Example:
        ```javascript
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('language', 'en');
        
        const response = await fetch('/api/speech/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log(result.text);
        ```
    """
    # Validate file size (25MB max)
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    
    # Read file content
    try:
        audio_bytes = await audio.read()
        file_size = len(audio_bytes)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 25MB"
            )
        
        logger.info(f"Received audio file: {audio.filename} ({file_size / 1024:.1f}KB)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        raise HTTPException(status_code=400, detail="Could not read audio file")
    
    # Save to temporary file for processing
    temp_file = None
    try:
        # Determine file extension (preserve original format)
        original_ext = Path(audio.filename or "audio.webm").suffix.lower()
        if not original_ext:
            original_ext = ".webm"  # Default for browser recordings
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        logger.info(f"Saved to temporary file: {temp_path}")
        
        # Get Whisper manager and transcribe
        whisper_manager = get_whisper_manager()
        result = whisper_manager.transcribe(
            audio_path=temp_path,
            language=language,
        )
        
        # Check if transcription is empty
        if not result["text"] or len(result["text"].strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in audio. Please speak clearly and try again."
            )
        
        logger.info(f"Transcription successful: '{result['text'][:50]}...'")
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            duration=result["duration"],
            model_info={
                "model_size": config.whisper.model_size,
                "device": config.whisper.device,
                "compute_type": config.whisper.compute_type,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_file:
            try:
                Path(temp_path).unlink(missing_ok=True)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {e}")


# ========== WHISPER CONFIGURATION ENDPOINTS ==========
@router.get(
    "/config/whisper",
    response_model=WhisperConfigResponse,
    summary="Get Whisper configuration",
    description="Retrieve current Whisper STT configuration",
)
async def get_whisper_config(
    config: OrionConfig = Depends(get_config_dependency),
) -> WhisperConfigResponse:
    """
    Get current Whisper configuration.
    
    Args:
        config: Configuration dependency (injected)
    
    Returns:
        WhisperConfigResponse with current settings
    """
    whisper_manager = get_whisper_manager()
    
    return WhisperConfigResponse(
        status="success",
        message="Current Whisper configuration",
        config={
            "model_size": config.whisper.model_size,
            "device": config.whisper.device,
            "compute_type": config.whisper.compute_type,
            "language": config.whisper.language,
            "model_cache_dir": str(config.whisper.model_cache_dir),
        },
        requires_reload=False,
    )


@router.patch(
    "/config/whisper",
    response_model=WhisperConfigResponse,
    summary="Update Whisper configuration",
    description="Update Whisper STT settings (model size, device, compute type, language)",
)
async def update_whisper_config(
    update: WhisperConfigUpdate,
    config: OrionConfig = Depends(get_config_dependency),
) -> WhisperConfigResponse:
    """
    Update Whisper configuration.
    
    Changes take effect on next transcription. If model_size or device changes,
    the model will be reloaded automatically on next use.
    
    Args:
        update: Configuration updates
        config: Configuration dependency (injected)
    
    Returns:
        WhisperConfigResponse with updated settings and reload status
        
    Example:
        ```javascript
        const response = await fetch('/api/speech/config/whisper', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_size: 'small',
                device: 'cuda',
                compute_type: 'float16',
                language: 'en'
            })
        });
        
        const result = await response.json();
        if (result.requires_reload) {
            console.log('Model will be reloaded on next use');
        }
        ```
    """
    try:
        whisper_manager = get_whisper_manager()
        requires_reload = False
        
        # Track if settings that require model reload changed
        if update.model_size and update.model_size != config.whisper.model_size:
            config.whisper.model_size = update.model_size
            requires_reload = True
            logger.info(f"Whisper model_size updated to: {update.model_size}")
        
        if update.device and update.device != config.whisper.device:
            config.whisper.device = update.device
            requires_reload = True
            logger.info(f"Whisper device updated to: {update.device}")
        
        if update.compute_type and update.compute_type != config.whisper.compute_type:
            config.whisper.compute_type = update.compute_type
            requires_reload = True
            logger.info(f"Whisper compute_type updated to: {update.compute_type}")
        
        # Language doesn't require reload
        if update.language is not None:  # Allow explicit None to clear language
            config.whisper.language = update.language if update.language else None
            logger.info(f"Whisper language updated to: {update.language or 'auto-detect'}")
        
        # Unload model if reload required (will lazy load with new settings)
        if requires_reload and whisper_manager.is_loaded():
            whisper_manager.unload()
            logger.info("Whisper model unloaded (will reload with new settings on next use)")
        
        return WhisperConfigResponse(
            status="success",
            message="Whisper configuration updated successfully",
            config={
                "model_size": config.whisper.model_size,
                "device": config.whisper.device,
                "compute_type": config.whisper.compute_type,
                "language": config.whisper.language,
                "model_cache_dir": str(config.whisper.model_cache_dir),
            },
            requires_reload=requires_reload,
        )
        
    except Exception as e:
        logger.error(f"Failed to update Whisper config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


# ========== HEALTH CHECK ENDPOINT ==========
@router.get(
    "/health",
    response_model=SpeechHealthResponse,
    summary="Speech service health check",
    description="Check availability of STT and TTS services",
)
async def speech_health(
    config: OrionConfig = Depends(get_config_dependency),
) -> SpeechHealthResponse:
    """
    Check speech service health and availability.
    
    Args:
        config: Configuration dependency (injected)
    
    Returns:
        SpeechHealthResponse with service status
    """
    try:
        whisper_manager = get_whisper_manager()
        stt_available = True
        whisper_loaded = whisper_manager.is_loaded()
    except Exception as e:
        logger.error(f"STT health check failed: {e}")
        stt_available = False
        whisper_loaded = False
    
    # TTS check
    try:
        tts_manager = get_tts_manager()
        tts_available = True
        tts_engine = config.tts.default_engine
        
        # Check Qwen3 availability
        qwen3_available = False
        qwen3_loaded = False
        if hasattr(tts_manager, 'qwen3_manager') and config.qwen3.enabled:
            qwen3_available = True
            qwen3_loaded = tts_manager.qwen3_manager.model is not None
    except Exception as e:
        logger.debug(f"TTS not available: {e}")
        tts_available = False
        tts_engine = None
        qwen3_available = False
        qwen3_loaded = False
    
    # Determine overall status
    if stt_available and tts_available:
        overall_status = "ready"
    elif stt_available or tts_available:
        overall_status = "degraded"
    else:
        overall_status = "error"
    
    return SpeechHealthResponse(
        status=overall_status,
        stt_available=stt_available,
        whisper_available=stt_available,
        tts_available=tts_available,
        whisper_loaded=whisper_loaded,
        whisper_config={
            "model_size": config.whisper.model_size,
            "device": config.whisper.device,
            "compute_type": config.whisper.compute_type,
            "language": config.whisper.language,
        },
        tts_engine=tts_engine,
        qwen3_available=qwen3_available,
        qwen3_loaded=qwen3_loaded,
    )


# ========== TTS ENGINE SELECTION ENDPOINT ==========
@router.patch(
    "/engine",
    response_model=EngineSelectResponse,
    summary="Switch TTS engine",
    description="Change active TTS engine between Piper (fast) and Qwen3 (voice cloning)",
)
async def switch_tts_engine(
    request: EngineSelectRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> EngineSelectResponse:
    """
    Switch between Piper and Qwen3 TTS engines.
    
    - **piper**: Fast synthesis, pre-built voices, CPU-friendly
    - **qwen3**: Voice cloning, slower, requires GPU
    
    Args:
        request: Engine selection request
        config: Configuration dependency (injected)
    
    Returns:
        EngineSelectResponse with switch confirmation
    
    Raises:
        HTTPException 400: Invalid engine or Qwen3 not available
    
    Example:
        ```javascript
        // Switch to Qwen3 for voice cloning
        await fetch('/api/speech/engine', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engine: 'qwen3' })
        });
        
        // Switch back to Piper for speed
        await fetch('/api/speech/engine', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ engine: 'piper' })
        });
        ```
    """
    try:
        previous_engine = config.tts.default_engine
        
        # Validate engine choice
        if request.engine == "qwen3":
            # Validate Qwen3 availability and auto-enable if possible
            
            # Get TTS manager with better error handling
            try:
                tts_manager = get_tts_manager()
            except Exception as e:
                logger.error(f"Failed to get TTS manager: {e}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail=f"TTS service initialization failed: {str(e)}. Check logs for details."
                )
            
            # Check if TTS manager has Qwen3 support
            if not hasattr(tts_manager, 'qwen3_manager'):
                raise HTTPException(
                    status_code=400,
                    detail="Qwen3-TTS is not available. UnifiedTTSManager not configured with Qwen3 support."
                )
            
            # Check GPU availability (Qwen3 requires GPU)
            try:
                import torch
                if not torch.cuda.is_available():
                    raise HTTPException(
                        status_code=400,
                        detail="Qwen3-TTS requires GPU (CUDA). No GPU detected on this system."
                    )
                
                # Check minimum VRAM
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram_gb < config.qwen3.min_vram_gb:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Qwen3-TTS requires {config.qwen3.min_vram_gb}GB VRAM, but only {vram_gb:.1f}GB available."
                        )
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="PyTorch not installed. Qwen3-TTS requires PyTorch with CUDA support."
                )
            
            # Auto-enable Qwen3 if validation passed
            if not config.qwen3.enabled:
                logger.info("Auto-enabling Qwen3-TTS after successful validation")
                config.qwen3.enabled = True
            
            # Preload Qwen3 model for voice cloning (may take 7-13 seconds on GPU)
            logger.info("Preloading Qwen3 model for voice cloning...")
            try:
                tts_manager.qwen3_manager.load_model()
                logger.info("Qwen3 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to preload Qwen3 model (will lazy load later): {e}")
        elif request.engine == "piper":
            # Just validate Piper is available (lightweight check)
            try:
                tts_manager = get_tts_manager()
            except Exception as e:
                logger.error(f"Failed to get TTS manager for Piper: {e}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail=f"TTS service initialization failed: {str(e)}. Ensure Piper TTS is installed."
                )
        
        # Switch engine
        config.tts.default_engine = request.engine
        
        logger.info(f"TTS engine switched: {previous_engine} → {request.engine}")
        
        return EngineSelectResponse(
            status="success",
            message=f"TTS engine switched to {request.engine}",
            active_engine=request.engine,
            previous_engine=previous_engine,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch TTS engine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Engine switch failed: {str(e)}"
        )


# ========== ACTIVE VOICE ENDPOINTS ==========
@router.get(
    "/active-voice",
    response_model=ActiveVoiceResponse,
    summary="Get the active Qwen3 voice",
    description="Returns the currently selected Qwen3 voice and the active TTS engine.",
)
async def get_active_voice(
    config: OrionConfig = Depends(get_config_dependency),
) -> ActiveVoiceResponse:
    """Return the active Qwen3 voice selection and current engine."""
    return ActiveVoiceResponse(
        status="success",
        active_voice=config.tts.active_qwen3_voice,
        engine=config.tts.default_engine,
        message="Active voice retrieved",
    )


@router.patch(
    "/active-voice",
    response_model=ActiveVoiceResponse,
    summary="Set the active Qwen3 voice",
    description="Select which cloned Qwen3 voice is used when TTS is triggered without an explicit voice.",
)
async def set_active_voice(
    request: ActiveVoiceRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> ActiveVoiceResponse:
    """Set the active Qwen3 voice for default synthesis."""
    try:
        tts_manager = get_tts_manager()

        if request.voice_id is not None:
            # Validate against the Qwen3 cloned-voice store (not the Piper catalog)
            if not hasattr(tts_manager, 'qwen3_manager'):
                raise HTTPException(
                    status_code=503,
                    detail="Qwen3 TTS manager is not available.",
                )
            cloned = tts_manager.qwen3_manager.list_cloned_voices()
            if request.voice_id not in cloned:
                available = list(cloned.keys())
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice '{request.voice_id}' not found. Available: {available}",
                )

        config.tts.active_qwen3_voice = request.voice_id
        msg = (
            f"Active Qwen3 voice set to '{request.voice_id}'"
            if request.voice_id
            else "Active Qwen3 voice cleared"
        )
        logger.info(msg)

        return ActiveVoiceResponse(
            status="success",
            active_voice=config.tts.active_qwen3_voice,
            engine=config.tts.default_engine,
            message=msg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set active voice: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set active voice: {str(e)}")


# ========== TEXT-TO-SPEECH (TTS) ENDPOINTS ==========
@router.post(
    "/synthesize",
    summary="Synthesize speech from text",
    description="Convert text to speech audio using active TTS engine (Piper or Qwen3)",
)
async def synthesize_speech(
    request: TTSRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> Response:
    """
    Convert text to speech.
    
    Returns audio stream in requested format (WAV/MP3).
    
    Args:
        request: TTS request with text and options
        config: Configuration dependency (injected)
    
    Returns:
        Response with audio bytes
    
    Raises:
        HTTPException 400: Invalid text or parameters
        HTTPException 503: TTS service unavailable
        HTTPException 500: Synthesis failed
    """
    try:
        # Validate text
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long (max 5000 characters)"
            )
        
        # Get TTS manager
        tts_manager = get_tts_manager()
        
        # Synthesize speech
        audio_bytes = tts_manager.synthesize(
            text=request.text,
            voice_id=request.voice,
            speed=request.speed,
            output_format=request.format,
        )
        
        # Determine media type
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }
        media_type = media_types.get(request.format, "audio/wav")
        
        # Return audio stream
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.format}",
                "Content-Length": str(len(audio_bytes)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="List available TTS voices",
    description="Get list of all available Piper TTS voices (Piper engine only)",
)
async def list_voices(
    language: Optional[str] = None,
    config: OrionConfig = Depends(get_config_dependency),
) -> VoiceListResponse:
    """
    List available Piper TTS voices, optionally filtered by language.
    
    **Note**: This endpoint is only available when Piper engine is active.
    For Qwen3 cloned voices, use GET /cloned-voices instead.
    
    Args:
        language: Optional language filter (e.g., 'en_US')
        config: Configuration dependency (injected)
    
    Returns:
        VoiceListResponse with voice list
    
    Raises:
        HTTPException 400: Wrong engine active (use Piper)
        HTTPException 503: TTS service unavailable
        HTTPException 500: Failed to list voices
    """
    try:
        # Guard: Piper-only endpoint
        if config.tts.default_engine != "piper":
            raise HTTPException(
                status_code=400,
                detail=f"Voice listing is only available for Piper TTS. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        
        tts_manager = get_tts_manager()
        voices = tts_manager.list_voices(language_filter=language)
        
        # Convert VoiceInfo or other model objects to plain dicts for proper serialization
        voice_dicts = []
        for v in voices:
            try:
                # Pydantic v2 models
                if hasattr(v, 'model_dump'):
                    voice_dicts.append(v.model_dump())
                # Pydantic v1 models
                elif hasattr(v, 'dict'):
                    voice_dicts.append(v.dict())
                # Dataclass (TTSManager uses dataclass VoiceInfo)
                elif is_dataclass(v):
                    voice_dicts.append(asdict(v))
                # Plain dict already
                elif isinstance(v, dict):
                    voice_dicts.append(v)
                else:
                    # As last resort, try to coerce via vars()
                    try:
                        voice_dicts.append(vars(v))
                    except Exception:
                        voice_dicts.append(v)
            except Exception:
                # Fallback: append the object as-is
                voice_dicts.append(v)
        
        return VoiceListResponse(
            status="success",
            voices=voice_dicts,
            count=len(voices),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list voices: {str(e)}"
        )


@router.get(
    "/config/tts",
    response_model=TTSConfigResponse,
    summary="Get TTS configuration",
    description="Retrieve current TTS configuration settings",
)
async def get_tts_config(
    config: OrionConfig = Depends(get_config_dependency),
) -> TTSConfigResponse:
    """
    Get current TTS configuration.
    
    Args:
        config: Configuration dependency (injected)
    
    Returns:
        TTSConfigResponse with current settings
    """
    return TTSConfigResponse(
        status="success",
        message="TTS configuration retrieved successfully",
        config={
            "enabled": config.tts.enabled,
            "engine": config.tts.default_engine,
            "default_voice": config.tts.default_voice,
            "audio_format": config.tts.audio_format,
            "default_speed": config.tts.default_speed,
            "use_gpu": config.tts.use_gpu,
            "sample_rate": config.tts.sample_rate,
            "auto_download_voices": config.tts.auto_download_voices,
        },
    )


@router.patch(
    "/config/tts",
    response_model=TTSConfigResponse,
    summary="Update TTS configuration",
    description="Update TTS settings (excluding voice selection)",
)
async def update_tts_config(
    updates: TTSConfigUpdate,
    config: OrionConfig = Depends(get_config_dependency),
) -> TTSConfigResponse:
    """
    Update TTS configuration settings.
    
    Note: This updates config in memory only. Voice changes use separate endpoint.
    
    Args:
        updates: Configuration updates
        config: Configuration dependency (injected)
    
    Returns:
        TTSConfigResponse with updated settings
    
    Raises:
        HTTPException 400: Invalid configuration values
    """
    try:
        # Update configuration
        updated_fields = []
        
        if updates.audio_format is not None:
            config.tts.audio_format = updates.audio_format
            updated_fields.append("audio_format")
        
        if updates.default_speed is not None:
            config.tts.default_speed = updates.default_speed
            updated_fields.append("default_speed")
        
        if updates.use_gpu is not None:
            config.tts.use_gpu = updates.use_gpu
            updated_fields.append("use_gpu")
            # GPU change requires TTS manager reset
            reset_tts_manager()
        
        message = f"Updated: {', '.join(updated_fields)}" if updated_fields else "No changes"
        
        return TTSConfigResponse(
            status="success",
            message=message,
            config={
                "enabled": config.tts.enabled,
                "engine": config.tts.default_engine,
                "default_voice": config.tts.default_voice,
                "audio_format": config.tts.audio_format,
                "default_speed": config.tts.default_speed,
                "use_gpu": config.tts.use_gpu,
                "sample_rate": config.tts.sample_rate,
                "auto_download_voices": config.tts.auto_download_voices,
            },
        )
        
    except Exception as e:
        logger.error(f"Failed to update TTS config: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Configuration update failed: {str(e)}"
        )


@router.patch(
    "/voice",
    response_model=TTSConfigResponse,
    summary="Change TTS voice",
    description="Update the default TTS voice",
)
async def update_tts_voice(
    voice_update: TTSVoiceUpdate,
    config: OrionConfig = Depends(get_config_dependency),
) -> TTSConfigResponse:
    """
    Change the default TTS voice.
    
    Args:
        voice_update: Voice update request
        config: Configuration dependency (injected)
    
    Returns:
        TTSConfigResponse with updated settings
    
    Raises:
        HTTPException 400: Invalid voice ID
        HTTPException 404: Voice not found
    """
    try:
        # Verify voice exists
        tts_manager = get_tts_manager()
        voices = tts_manager.list_voices()
        # Normalize possible model or dict entries
        voice_ids = []
        for v in voices:
            if hasattr(v, 'voice_id'):
                voice_ids.append(getattr(v, 'voice_id'))
            elif isinstance(v, dict) and 'voice_id' in v:
                voice_ids.append(v['voice_id'])
            elif hasattr(v, 'model_dump'):
                try:
                    d = v.model_dump()
                    if 'voice_id' in d:
                        voice_ids.append(d['voice_id'])
                except Exception:
                    continue
        
        if voice_update.voice_id not in voice_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_update.voice_id}' not found. Available voices: {', '.join(voice_ids[:5])}"
            )
        
        # Update default voice
        config.tts.default_voice = voice_update.voice_id
        
        return TTSConfigResponse(
            status="success",
            message=f"Default voice changed to '{voice_update.voice_id}'",
            config={
                "enabled": config.tts.enabled,
                "engine": config.tts.default_engine,
                "default_voice": config.tts.default_voice,
                "audio_format": config.tts.audio_format,
                "default_speed": config.tts.default_speed,
                "use_gpu": config.tts.use_gpu,
                "sample_rate": config.tts.sample_rate,
                "auto_download_voices": config.tts.auto_download_voices,
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update voice: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Voice update failed: {str(e)}"
        )


@router.post(
    "/preview-voice",
    summary="Preview a TTS voice",
    description="Generate a short audio sample with the specified Piper voice (Piper engine only)",
)
async def preview_voice(
    preview: TTSPreviewRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> Response:
    """
    Preview a Piper voice with sample text.
    
    **Note**: This endpoint is only available when Piper engine is active.
    
    Args:
        preview: Preview request with voice ID and optional text
        config: Configuration dependency (injected)
    
    Returns:
        Response with audio bytes (WAV format)
    
    Raises:
        HTTPException 400: Wrong engine active (use Piper)
        HTTPException 404: Voice not found
        HTTPException 500: Preview generation failed
    """
    try:
        # Guard: Piper-only endpoint
        if config.tts.default_engine != "piper":
            raise HTTPException(
                status_code=400,
                detail=f"Voice preview is only available for Piper TTS. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        
        tts_manager = get_tts_manager()
        
        # Synthesize preview
        audio_bytes = tts_manager.synthesize(
            text=preview.text,
            voice_id=preview.voice_id,
            speed=1.0,
            output_format="wav",
        )
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=preview_{preview.voice_id}.wav",
                "Content-Length": str(len(audio_bytes)),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice preview failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Preview generation failed: {str(e)}"
        )


# ========== QWEN3-TTS VOICE CLONING ENDPOINTS ==========
@router.post(
    "/clone-voice",
    response_model=VoiceCloneResponse,
    summary="Clone a voice from audio sample",
    description="Extract voice embedding from uploaded audio for Qwen3-TTS cloning (Qwen3 engine only)",
)
async def clone_voice(
    voice_name: str = Form(..., description="Unique name for this voice"),
    ref_text: Optional[str] = Form(None, description="Reference text transcript (improves quality)"),
    audio: UploadFile = File(..., description="Reference audio file (3-15 seconds, WAV/MP3)"),
    config: OrionConfig = Depends(get_config_dependency),
) -> VoiceCloneResponse:
    """
    Create a cloned voice from audio sample.
    
    Uploads an audio file (3-15 seconds) and extracts voice characteristics
    for use with Qwen3-TTS synthesis. Optionally provide reference text for
    better quality (ICL mode).
    
    **Note**: This endpoint is only available when Qwen3 engine is active.
    
    Args:
        voice_name: Unique identifier for this voice
        ref_text: Optional transcript of the audio
        audio: Audio file (WAV, MP3, etc.)
        config: Configuration dependency (injected)
    
    Returns:
        VoiceCloneResponse with created voice info
    
    Raises:
        HTTPException 400: Invalid audio, voice name exists, or wrong engine
        HTTPException 503: Qwen3-TTS not available
        HTTPException 500: Voice cloning failed
    """
    try:
        # Guard: Qwen3-only endpoint
        if config.tts.default_engine != "qwen3":
            raise HTTPException(
                status_code=400,
                detail=f"Voice cloning requires Qwen3-TTS. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        
        # Check if Qwen3 is enabled
        if not config.qwen3.enabled:
            raise HTTPException(
                status_code=503,
                detail="Qwen3-TTS voice cloning is not enabled. Enable in configuration."
            )
        
        # Get TTS manager (should be UnifiedTTSManager)
        tts_manager = get_tts_manager()
        
        # Check if manager has Qwen3 support
        if not hasattr(tts_manager, 'qwen3_manager'):
            raise HTTPException(
                status_code=503,
                detail="Voice cloning requires Qwen3-TTS. Please check configuration."
            )
        
        # Save uploaded audio to temp file
        # Note: extract_voice_embedding will copy it to permanent storage
        with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or "audio.wav").suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            audio_bytes = await audio.read()
            tmp.write(audio_bytes)
            tmp.flush()
        
        try:
            # Extract voice embedding using Qwen3Manager
            qwen3_manager = tts_manager.qwen3_manager
            
            # Check if lazy-loading succeeded
            if qwen3_manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="Qwen3-TTS manager failed to initialize. Check logs for details. Ensure qwen-tts package is installed."
                )
            
            # This will copy the temp file to permanent storage
            embedding = qwen3_manager.extract_voice_embedding(
                voice_id=voice_name,
                audio_path=tmp_path,
                ref_text=ref_text,
            )
            
            return VoiceCloneResponse(
                status="success",
                message=f"Voice '{voice_name}' cloned successfully",
                voice_id=embedding.voice_id,
                duration=embedding.duration,
                sample_rate=embedding.sample_rate,
            )
        finally:
            # Cleanup temp file (permanent copy was made by extract_voice_embedding)
            tmp_path.unlink(missing_ok=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice cloning failed: {str(e)}"
        )


@router.get(
    "/cloned-voices",
    response_model=ClonedVoicesListResponse,
    summary="List cloned voices",
    description="Get list of all voices cloned with Qwen3-TTS (Qwen3 engine only)",
)
async def list_cloned_voices(
    config: OrionConfig = Depends(get_config_dependency),
) -> ClonedVoicesListResponse:
    """
    List all cloned voices available for Qwen3-TTS synthesis.
    
    **Note**: This endpoint is only available when Qwen3 engine is active.
    
    Args:
        config: Configuration dependency (injected)
    
    Returns:
        ClonedVoicesListResponse with voice list
    
    Raises:
        HTTPException 400: Wrong engine active (use Qwen3)
        HTTPException 503: Qwen3-TTS not available
    """
    try:
        # Guard: Qwen3-only endpoint
        if config.tts.default_engine != "qwen3":
            raise HTTPException(
                status_code=400,
                detail=f"Cloned voice listing requires Qwen3-TTS. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        
        tts_manager = get_tts_manager()
        
        if not hasattr(tts_manager, 'qwen3_manager'):
            return ClonedVoicesListResponse(
                status="success",
                voices=[],
                count=0,
            )
        
        qwen3_manager = tts_manager.qwen3_manager
        embeddings = qwen3_manager.list_cloned_voices()
        
        voices = []
        for voice_id, embedding in embeddings.items():
            voices.append(ClonedVoiceInfo(
                voice_id=voice_id,
                duration=embedding.duration,
                sample_rate=embedding.sample_rate,
                created_at=embedding.created_at,
                has_ref_text=hasattr(embedding, 'ref_text') and embedding.ref_text is not None,
            ))
        
        return ClonedVoicesListResponse(
            status="success",
            voices=voices,
            count=len(voices),
        )
    
    except Exception as e:
        logger.error(f"Failed to list cloned voices: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list cloned voices: {str(e)}"
        )


@router.delete(
    "/cloned-voices/{voice_id}",
    summary="Delete a cloned voice",
    description="Remove a cloned voice from cache (Qwen3 engine only)",
)
async def delete_cloned_voice(
    voice_id: str,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Delete a cloned voice.
    
    **Note**: This endpoint is only available when Qwen3 engine is active.
    
    Args:
        voice_id: Voice ID to delete
        config: Configuration dependency (injected)
    
    Returns:
        Success message
    
    Raises:
        HTTPException 400: Wrong engine active (use Qwen3)
        HTTPException 404: Voice not found
        HTTPException 503: Qwen3-TTS not available
    """
    try:
        # Guard: Qwen3-only endpoint
        if config.tts.default_engine != "qwen3":
            raise HTTPException(
                status_code=400,
                detail=f"Cloned voice deletion requires Qwen3-TTS. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        tts_manager = get_tts_manager()
        
        if not hasattr(tts_manager, 'qwen3_manager'):
            raise HTTPException(
                status_code=503,
                detail="Qwen3-TTS not available"
            )
        
        qwen3_manager = tts_manager.qwen3_manager
        success = qwen3_manager.delete_voice(voice_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_id}' not found"
            )
        
        return {"status": "success", "message": f"Voice '{voice_id}' deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete voice: {str(e)}"
        )


@router.post(
    "/synthesize-qwen3",
    summary="Synthesize speech with cloned voice",
    description="Use Qwen3-TTS to synthesize speech with a cloned voice (Qwen3 engine only, may be slow)",
)
async def synthesize_qwen3(
    request: TTSAsyncRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> Response:
    """
    Synthesize speech using Qwen3-TTS with a cloned voice.
    
    This endpoint performs synchronous synthesis (blocks until complete).
    For long text, consider using async synthesis endpoint instead.
    
    **Note**: This endpoint is only available when Qwen3 engine is active.
    
    Args:
        request: TTS request with text and cloned voice_id
        config: Configuration dependency (injected)
    
    Returns:
        Response with audio bytes (WAV format)
    
    Raises:
        HTTPException 400: Invalid text, voice_id, or wrong engine
        HTTPException 503: Qwen3-TTS not available
        HTTPException 500: Synthesis failed
    """
    try:
        # Guard: Qwen3-only endpoint
        if config.tts.default_engine != "qwen3":
            raise HTTPException(
                status_code=400,
                detail=f"Qwen3 synthesis requires Qwen3-TTS engine. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        if not config.qwen3.enabled:
            raise HTTPException(
                status_code=503,
                detail="Qwen3-TTS is not enabled"
            )
        
        tts_manager = get_tts_manager()
        
        if not hasattr(tts_manager, 'qwen3_manager'):
            raise HTTPException(
                status_code=503,
                detail="Qwen3-TTS not available"
            )
        
        # Synthesize using Qwen3
        qwen3_manager = tts_manager.qwen3_manager
        audio_array, sample_rate = qwen3_manager.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            language=request.language,
        )
        
        # Convert to WAV bytes
        import io
        import soundfile as sf
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=qwen3_speech.wav",
                "Content-Length": str(len(audio_bytes)),
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qwen3 synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )


@router.get(
    "/qwen3/stats",
    summary="Get Qwen3-TTS statistics",
    description="Get synthesis statistics and cache info for Qwen3-TTS (Qwen3 engine only)",
)
async def get_qwen3_stats(
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Get Qwen3-TTS performance statistics.
    
    **Note**: This endpoint is only available when Qwen3 engine is active.
    
    Args:
        config: Configuration dependency (injected)
    
    Returns:
        Statistics dictionary
    
    Raises:
        HTTPException 400: Wrong engine active (use Qwen3)
    """
    try:
        # Guard: Qwen3-only endpoint
        if config.tts.default_engine != "qwen3":
            raise HTTPException(
                status_code=400,
                detail=f"Qwen3 statistics require Qwen3-TTS engine. Current engine: {config.tts.default_engine}. Switch engine with PATCH /api/speech/engine"
            )
        tts_manager = get_tts_manager()
        
        if not hasattr(tts_manager, 'qwen3_manager'):
            return {
                "status": "unavailable",
                "message": "Qwen3-TTS not available",
            }
        
        qwen3_manager = tts_manager.qwen3_manager
        stats = qwen3_manager.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
        }
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


