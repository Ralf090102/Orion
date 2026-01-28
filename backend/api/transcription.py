"""
Audio Transcription API - Speech-to-Text endpoint

Provides REST API for converting audio to text using faster-whisper.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.utilities.whisper_manager import get_whisper_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["transcription"])


# ========== RESPONSE MODELS ==========
class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint"""

    text: str
    language: str
    duration: float


# ========== ENDPOINTS ==========
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WebM, WAV, MP3, M4A, etc.)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en'). None = auto-detect"),
) -> TranscriptionResponse:
    """
    Transcribe audio file to text using Whisper.
    
    Accepts audio files in various formats:
    - WebM (from browser recording)
    - WAV
    - MP3
    - M4A
    - FLAC
    - OGG
    
    Args:
        audio: Audio file to transcribe
        language: Optional language code ('en', 'es', etc.). Auto-detects if not provided
    
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
        
        const response = await fetch('/api/transcribe', {
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


@router.get("/transcribe/health")
async def transcribe_health():
    """
    Check if transcription service is available.
    
    Returns:
        Status of Whisper model (loaded/not loaded)
    """
    try:
        whisper_manager = get_whisper_manager()
        return {
            "status": "ready",
            "model_loaded": whisper_manager.is_loaded(),
            "model_size": whisper_manager.config.model_size,
            "device": whisper_manager.config.device,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
