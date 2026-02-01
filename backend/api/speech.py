"""
Speech API Endpoints - STT and TTS

Provides REST API for:
- Speech-to-Text (STT) using faster-whisper
- Text-to-Speech (TTS) using various engines
- Whisper configuration management
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from backend.dependencies import get_config_dependency
from backend.models.speech import (
    SpeechHealthResponse,
    TranscriptionResponse,
    TTSRequest,
    TTSResponse,
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


# ========== TEXT-TO-SPEECH (TTS) ENDPOINTS ==========
@router.post(
    "/synthesize",
    summary="Synthesize speech from text",
    description="Convert text to speech audio using TTS engine",
    response_class=Response,
)
async def synthesize_speech(
    request: TTSRequest,
    config: OrionConfig = Depends(get_config_dependency),
):
    """
    Convert text to speech audio.
    
    This is a placeholder endpoint for TTS functionality.
    Future implementation will support various TTS engines (Piper, Coqui TTS, etc.).
    
    Args:
        request: TTS request with text and parameters
        config: Configuration dependency (injected)
    
    Returns:
        Audio file in requested format
    
    Raises:
        HTTPException 501: Not implemented yet
        
    Example:
        ```javascript
        const response = await fetch('/api/speech/synthesize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: 'Hello, this is a test.',
                language: 'en',
                speed: 1.0,
                format: 'mp3'
            })
        });
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        ```
    """
    # TODO: Implement TTS using Piper, Coqui TTS, or other engines
    # This is a placeholder for future implementation
    
    logger.warning("TTS endpoint called but not yet implemented")
    raise HTTPException(
        status_code=501,
        detail="Text-to-Speech (TTS) is not yet implemented. Coming soon!"
    )
    
    # Future implementation outline:
    # 1. Validate text length
    # 2. Load TTS model based on language/voice
    # 3. Synthesize audio
    # 4. Convert to requested format
    # 5. Return audio stream
    
    # Example response (when implemented):
    # return StreamingResponse(
    #     audio_stream,
    #     media_type=f"audio/{request.format}",
    #     headers={
    #         "Content-Disposition": f"attachment; filename=speech.{request.format}"
    #     }
    # )


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
    
    # TTS check (placeholder for future implementation)
    tts_available = False  # Will be True when TTS is implemented
    tts_engine = None
    
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
        tts_available=tts_available,
        whisper_loaded=whisper_loaded,
        whisper_config={
            "model_size": config.whisper.model_size,
            "device": config.whisper.device,
            "compute_type": config.whisper.compute_type,
            "language": config.whisper.language,
        },
        tts_engine=tts_engine,
    )
