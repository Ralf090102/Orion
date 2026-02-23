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
    ClonedVoiceInfo,
    ClonedVoicesListResponse,
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


# ========== TEXT-TO-SPEECH (TTS) ENDPOINTS ==========
@router.post(
    "/synthesize",
    summary="Synthesize speech from text",
    description="Convert text to speech audio using Piper TTS",
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
    description="Get list of all available Piper TTS voices",
)
async def list_voices(
    language: Optional[str] = None,
) -> VoiceListResponse:
    """
    List available TTS voices, optionally filtered by language.
    
    Args:
        language: Optional language filter (e.g., 'en_US')
    
    Returns:
        VoiceListResponse with voice list
    
    Raises:
        HTTPException 503: TTS service unavailable
        HTTPException 500: Failed to list voices
    """
    try:
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
    description="Generate a short audio sample with the specified voice",
)
async def preview_voice(
    preview: TTSPreviewRequest,
) -> Response:
    """
    Preview a voice with sample text.
    
    Args:
        preview: Preview request with voice ID and optional text
    
    Returns:
        Response with audio bytes (WAV format)
    
    Raises:
        HTTPException 404: Voice not found
        HTTPException 500: Preview generation failed
    """
    try:
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
    description="Extract voice embedding from uploaded audio for Qwen3-TTS cloning",
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
    
    Args:
        voice_name: Unique identifier for this voice
        ref_text: Optional transcript of the audio
        audio: Audio file (WAV, MP3, etc.)
        config: Configuration dependency (injected)
    
    Returns:
        VoiceCloneResponse with created voice info
    
    Raises:
        HTTPException 400: Invalid audio or voice name already exists
        HTTPException 503: Qwen3-TTS not available
        HTTPException 500: Voice cloning failed
    """
    try:
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
        with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or "audio.wav").suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            audio_bytes = await audio.read()
            tmp.write(audio_bytes)
            tmp.flush()
        
        try:
            # Extract voice embedding using Qwen3Manager
            qwen3_manager = tts_manager.qwen3_manager
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
            # Cleanup temp file
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
    description="Get list of all voices cloned with Qwen3-TTS",
)
async def list_cloned_voices() -> ClonedVoicesListResponse:
    """
    List all cloned voices available for Qwen3-TTS synthesis.
    
    Returns:
        ClonedVoicesListResponse with voice list
    
    Raises:
        HTTPException 503: Qwen3-TTS not available
    """
    try:
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
    description="Remove a cloned voice from cache",
)
async def delete_cloned_voice(voice_id: str):
    """
    Delete a cloned voice.
    
    Args:
        voice_id: Voice ID to delete
    
    Returns:
        Success message
    
    Raises:
        HTTPException 404: Voice not found
        HTTPException 503: Qwen3-TTS not available
    """
    try:
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
    description="Use Qwen3-TTS to synthesize speech with a cloned voice (may be slow)",
)
async def synthesize_qwen3(
    request: TTSAsyncRequest,
    config: OrionConfig = Depends(get_config_dependency),
) -> Response:
    """
    Synthesize speech using Qwen3-TTS with a cloned voice.
    
    This endpoint performs synchronous synthesis (blocks until complete).
    For long text, consider using async synthesis endpoint instead.
    
    Args:
        request: TTS request with text and cloned voice_id
        config: Configuration dependency (injected)
    
    Returns:
        Response with audio bytes (WAV format)
    
    Raises:
        HTTPException 400: Invalid text or voice_id
        HTTPException 503: Qwen3-TTS not available
        HTTPException 500: Synthesis failed
    """
    try:
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
    description="Get synthesis statistics and cache info for Qwen3-TTS",
)
async def get_qwen3_stats():
    """
    Get Qwen3-TTS performance statistics.
    
    Returns:
        Statistics dictionary
    """
    try:
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


