"""
Whisper Model Manager - Singleton pattern for faster-whisper

Manages the lifecycle of the Whisper transcription model:
- Lazy loading (loads on first use)
- Singleton pattern (only one instance)
- Thread-safe initialization
- GPU/CPU auto-detection
- Configurable model size and parameters
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from src.utilities.config import get_config

logger = logging.getLogger(__name__)


class WhisperManager:
    """
    Singleton manager for faster-whisper model.
    
    Handles model loading, caching, and transcription.
    Thread-safe lazy initialization.
    """

    _instance: Optional["WhisperManager"] = None
    _lock = threading.Lock()
    _model = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize manager (but don't load model yet)."""
        if not self._initialized:
            self.config = get_config(from_env=True).whisper
            self._initialized = True
            logger.info("WhisperManager initialized (model not loaded yet)")

    def _load_model(self):
        """
        Load the Whisper model into memory.
        
        This is called on first transcription request (lazy loading).
        Supports both CPU and GPU acceleration.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.config.model_size}")
            logger.info(f"Device: {self.config.device}, Compute type: {self.config.compute_type}")

            # Load model with configuration
            self._model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=str(self.config.model_cache_dir),
            )

            logger.info(f"✅ Whisper model '{self.config.model_size}' loaded successfully")

        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, WebM, etc.)
            language: Language code (e.g., 'en', 'es'). None = auto-detect
        
        Returns:
            Dictionary with:
                - text: Transcribed text
                - language: Detected/specified language
                - duration: Audio duration in seconds
                
        Raises:
            Exception: If transcription fails
        """
        # Lazy load model on first use
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()

        try:
            # Use configured language if not specified
            lang = language or self.config.language

            # Transcribe with faster-whisper
            segments, info = self._model.transcribe(
                audio_path,
                language=lang,
                beam_size=5,  # Good balance of speed/accuracy
                vad_filter=True,  # Remove silence
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                },
            )

            # Collect all segments into full text
            full_text = " ".join([segment.text.strip() for segment in segments])

            result = {
                "text": full_text.strip(),
                "language": info.language,
                "duration": info.duration,
            }

            logger.info(f"Transcription complete: {len(full_text)} chars, {info.duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._model is not None

    def unload(self):
        """Unload model from memory (free resources)."""
        if self._model is not None:
            with self._lock:
                if self._model is not None:
                    del self._model
                    self._model = None
                    logger.info("Whisper model unloaded from memory")


# ========== GLOBAL INSTANCE ==========
_whisper_manager: Optional[WhisperManager] = None


def get_whisper_manager() -> WhisperManager:
    """
    Get the global WhisperManager instance.
    
    Returns:
        WhisperManager singleton instance
    """
    global _whisper_manager
    if _whisper_manager is None:
        _whisper_manager = WhisperManager()
    return _whisper_manager
