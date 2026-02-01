"""
Pydantic models for speech-related API endpoints.

Request/response models for Speech-to-Text (STT) and Text-to-Speech (TTS).
"""

from typing import Optional

from pydantic import BaseModel, Field


# ========== WHISPER CONFIGURATION MODELS ==========
class WhisperConfigUpdate(BaseModel):
    """Request model for updating Whisper STT configuration."""

    model_size: Optional[str] = Field(
        default=None,
        description="Whisper model size: tiny, base, small, medium, large, large-v2, large-v3",
        pattern="^(tiny|base|small|medium|large|large-v2|large-v3)$",
    )
    device: Optional[str] = Field(
        default=None,
        description="Compute device: auto, cpu, cuda",
        pattern="^(auto|cpu|cuda)$",
    )
    compute_type: Optional[str] = Field(
        default=None,
        description="Compute precision: int8, float16, float32",
        pattern="^(int8|float16|float32)$",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'es', 'fr'). None = auto-detect",
        max_length=10,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_size": "base",
                "device": "auto",
                "compute_type": "int8",
                "language": "en",
            }
        }


class WhisperConfigResponse(BaseModel):
    """Response model for Whisper configuration."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    config: dict = Field(..., description="Current Whisper configuration")
    requires_reload: bool = Field(
        default=False,
        description="Whether the model needs to be reloaded for changes to take effect",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Configuration updated successfully",
                "config": {
                    "model_size": "base",
                    "device": "auto",
                    "compute_type": "int8",
                    "language": None,
                    "model_cache_dir": "/home/user/.cache/whisper",
                },
                "requires_reload": True,
            }
        }


# ========== SPEECH-TO-TEXT (STT) MODELS ==========
class TranscriptionRequest(BaseModel):
    """Request model for audio transcription (form data companion)."""

    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en'). None = auto-detect",
        max_length=10,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "language": "en",
            }
        }


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""

    text: str = Field(..., description="Transcribed text from audio")
    language: str = Field(..., description="Detected or specified language code")
    duration: float = Field(..., description="Audio duration in seconds")
    model_info: dict = Field(
        default_factory=dict,
        description="Information about the model used for transcription",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test transcription.",
                "language": "en",
                "duration": 3.5,
                "model_info": {
                    "model_size": "base",
                    "device": "cuda",
                    "compute_type": "int8",
                },
            }
        }


# ========== TEXT-TO-SPEECH (TTS) MODELS ==========
class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(
        ...,
        description="Text to convert to speech",
        min_length=1,
        max_length=5000,
    )
    language: Optional[str] = Field(
        default="en",
        description="Language code for speech synthesis",
        max_length=10,
    )
    voice: Optional[str] = Field(
        default="default",
        description="Voice identifier (implementation-specific)",
    )
    speed: float = Field(
        default=1.0,
        description="Speech speed multiplier (0.5 = slow, 2.0 = fast)",
        ge=0.25,
        le=4.0,
    )
    format: str = Field(
        default="mp3",
        description="Audio format: mp3, wav, opus",
        pattern="^(mp3|wav|opus)$",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test of text-to-speech.",
                "language": "en",
                "voice": "default",
                "speed": 1.0,
                "format": "mp3",
            }
        }


class TTSResponse(BaseModel):
    """Response model for text-to-speech (metadata only, audio in body)."""

    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    duration: float = Field(..., description="Audio duration in seconds")
    format: str = Field(..., description="Audio format")
    size_bytes: int = Field(..., description="Audio file size in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Speech synthesized successfully",
                "duration": 2.5,
                "format": "mp3",
                "size_bytes": 48000,
            }
        }


# ========== HEALTH CHECK MODELS ==========
class SpeechHealthResponse(BaseModel):
    """Response model for speech service health check."""

    status: str = Field(..., description="Service status: ready, degraded, error")
    stt_available: bool = Field(..., description="Speech-to-Text service availability")
    # Backwards/forwards compatibility: some frontends expect `whisper_available`
    whisper_available: bool = Field(
        default=False,
        description="Alias for stt_available for frontend compatibility",
    )
    tts_available: bool = Field(..., description="Text-to-Speech service availability")
    whisper_loaded: bool = Field(
        default=False,
        description="Whether Whisper model is loaded in memory",
    )
    whisper_config: dict = Field(
        default_factory=dict,
        description="Current Whisper configuration",
    )
    tts_engine: Optional[str] = Field(
        default=None,
        description="TTS engine identifier",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "stt_available": True,
                "tts_available": True,
                "whisper_loaded": True,
                "whisper_config": {
                    "model_size": "base",
                    "device": "cuda",
                    "compute_type": "int8",
                },
                "tts_engine": "piper",
            }
        }
