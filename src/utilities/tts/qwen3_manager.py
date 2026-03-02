"""Qwen3-TTS Manager - Multi-model TTS engine.

This module provides voice generation capabilities using Qwen3-TTS models:
- Voice Cloning: Clone voices from 3-15s audio samples (Base model)
- Voice Generation: Create new voices from text descriptions (VoiceDesign model)
- Custom Voices: Premium speakers with instruction control (CustomVoice model)

Features:
- Lazy model loading (GPU memory optimization)
- Multi-model support with automatic switching (only 1 loaded at a time)
- Voice embedding extraction from audio samples (3-15s)
- High-quality speech synthesis with cloned/designed voices
- GPU memory management with auto-unload
- Performance monitoring (real-time factor tracking)

Official Docs: https://github.com/QwenLM/Qwen3-TTS
"""

import json
import re
import threading
import torch
import numpy as np
import soundfile as sf
import time
import logging
from pathlib import Path
from typing import Generator, Iterator, Optional, Dict, Tuple, List, Literal
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ========== CONSTANTS ==========

class ModelType(str, Enum):
    """Qwen3-TTS model types."""
    BASE = "base"           # Voice cloning from audio
    DESIGN = "design"       # Voice generation from text description
    CUSTOM = "custom"       # Premium speakers with instruction control


# Supported languages by Qwen3-TTS
SUPPORTED_LANGUAGES = {
    "chinese", "english", "japanese", "korean", "german", 
    "french", "russian", "portuguese", "spanish", "italian",
    "auto",  # Auto-detect
}

# Language code aliases (normalize various inputs)
LANGUAGE_ALIASES = {
    "en": "english", "eng": "english",
    "zh": "chinese", "chn": "chinese", "mandarin": "chinese",
    "ja": "japanese", "jp": "japanese", "jpn": "japanese",
    "ko": "korean", "kr": "korean", "kor": "korean",
    "de": "german", "deu": "german",
    "fr": "french", "fra": "french",
    "ru": "russian", "rus": "russian",
    "pt": "portuguese", "por": "portuguese",
    "es": "spanish", "spa": "spanish",
    "it": "italian", "ita": "italian",
}


# Custom speakers available in Qwen3-TTS-CustomVoice model
# Each speaker has unique characteristics and native language
CUSTOM_SPEAKERS = {
    "Vivian": {
        "description": "Bright, slightly edgy young female voice",
        "native_language": "chinese",
        "gender": "female",
        "age_range": "20-25",
    },
    "Serena": {
        "description": "Warm, gentle young female voice",
        "native_language": "chinese",
        "gender": "female",
        "age_range": "22-28",
    },
    "Uncle_Fu": {
        "description": "Seasoned male with low, mellow timbre",
        "native_language": "chinese",
        "gender": "male",
        "age_range": "50-60",
    },
    "Dylan": {
        "description": "Youthful Beijing male, clear and natural",
        "native_language": "chinese",  # Beijing dialect
        "gender": "male",
        "age_range": "25-30",
    },
    "Eric": {
        "description": "Lively Chengdu male, slightly husky",
        "native_language": "chinese",  # Sichuan dialect
        "gender": "male",
        "age_range": "25-32",
    },
    "Ryan": {
        "description": "Dynamic male with strong rhythmic drive",
        "native_language": "english",
        "gender": "male",
        "age_range": "28-35",
    },
    "Aiden": {
        "description": "Sunny American male with clear midrange",
        "native_language": "english",
        "gender": "male",
        "age_range": "22-28",
    },
    "Ono_Anna": {
        "description": "Playful Japanese female, light and nimble",
        "native_language": "japanese",
        "gender": "female",
        "age_range": "20-26",
    },
    "Sohee": {
        "description": "Warm Korean female with rich emotion",
        "native_language": "korean",
        "gender": "female",
        "age_range": "24-30",
    },
}


@dataclass
class VoiceEmbedding:
    """Cached voice embedding for voice cloning.
    
    Attributes:
        voice_id: Unique identifier for the voice
        embedding: Audio data or neural representation (depends on model)
        sample_rate: Audio sample rate (Hz)
        duration: Duration of reference audio (seconds)
        created_at: Unix timestamp of creation
        ref_text: Optional reference text transcript for ICL mode
        source: How voice was created ("cloned", "designed")
        voice_description: For designed voices, the description used
    """
    voice_id: str
    embedding: np.ndarray  # Audio data for Qwen3-TTS Base model
    sample_rate: int
    duration: float
    created_at: float
    ref_text: Optional[str] = None  # Reference text for ICL mode
    source: str = "cloned"  # "cloned" or "designed"
    voice_description: Optional[str] = None  # For designed voices


class Qwen3Manager:
    """Manage Qwen3-TTS for multi-model TTS synthesis.
    
    This manager handles:
    - Model loading/unloading (lazy initialization)
    - Multi-model support (Base, VoiceDesign, CustomVoice)
    - Automatic model switching (only 1 loaded at a time for VRAM)
    - Voice embedding extraction from audio samples
    - Speech synthesis with cloned/designed voices
    - GPU memory management
    - Multi-language support (en, zh, ja, ko, etc.)
    
    Example:
        >>> manager = Qwen3Manager(config)
        >>> # Voice cloning: Extract voice from audio sample
        >>> embedding = manager.extract_voice_embedding("my_voice", "sample.wav")
        >>> audio, sr = manager.synthesize("Hello world!", voice_id="my_voice")
        >>> 
        >>> # Voice generation: Create voice from text description
        >>> audio, sr = manager.generate_voice(
        ...     text="Welcome to the show!",
        ...     voice_description="Male broadcaster, 35, deep resonant voice",
        ...     language="english"
        ... )
        >>> 
        >>> # Design and save: Create a reusable designed voice
        >>> voice_id = manager.design_and_save_voice(
        ...     voice_id="narrator",
        ...     reference_text="Hello everyone!",
        ...     voice_description="Warm female narrator, 30",
        ...     language="english"
        ... )
    """
    
    def __init__(self, config):
        """Initialize Qwen3Manager.
        
        Args:
            config: OrionConfig instance with qwen3 settings
        """
        self.config = config
        self.qwen3_config = config.qwen3
        
        # Multi-model state (only 1 loaded at a time for VRAM)
        self._base_model = None      # Base model for voice cloning
        self._design_model = None    # VoiceDesign model for voice generation
        self._custom_model = None    # CustomVoice model for premium speakers
        self._current_model_type: Optional[ModelType] = None
        
        self.device = self._get_device()
        self.model_loaded_at: Optional[float] = None
        self.last_used_at: Optional[float] = None
        
        # Voice cache
        self.voice_embeddings: Dict[str, VoiceEmbedding] = {}
        self._voice_prompts: Dict[str, object] = {}  # voice_id -> VoiceClonePromptItem
        
        # Storage directory for cloned voice audio samples
        self.voices_dir = Path(__file__).parent.parent.parent.parent / "data" / "tts" / "cloned_voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloned voices directory: {self.voices_dir}")
        
        # Stats per model type
        self.synthesis_count = 0
        self.total_synthesis_time = 0.0
        self._stats_by_model: Dict[ModelType, Dict] = {
            ModelType.BASE: {"count": 0, "time": 0.0},
            ModelType.DESIGN: {"count": 0, "time": 0.0},
            ModelType.CUSTOM: {"count": 0, "time": 0.0},
        }
        
        # Background loading
        self._loading_thread: Optional[threading.Thread] = None
        self._loading_error: Optional[Exception] = None
        self._loading_model_type: Optional[ModelType] = None
        
        logger.info(f"Qwen3Manager initialized (device: {self.device})")
        logger.info(f"Auto-unload: {self.qwen3_config.auto_unload} "
                   f"(timeout: {self.qwen3_config.unload_timeout_seconds}s)")
        logger.info(f"Models: base={self.qwen3_config.model_name}, "
                   f"design={self.qwen3_config.design_model_name}, "
                   f"custom={self.qwen3_config.custom_model_name}")
        
        # Scan disk for existing cloned voices so they survive restarts
        loaded = self._scan_cloned_voices()
        if loaded:
            logger.info(f"✓ Auto-loaded {loaded} cloned voice(s) from disk")
    
    def _get_current_model(self):
        """Get the currently loaded model instance."""
        if self._current_model_type == ModelType.BASE:
            return self._base_model
        elif self._current_model_type == ModelType.DESIGN:
            return self._design_model
        elif self._current_model_type == ModelType.CUSTOM:
            return self._custom_model
        return None
    
    # Backwards compatibility property
    @property
    def model(self):
        return self._get_current_model()
    
    def _get_device(self) -> str:
        """Determine compute device (CUDA/CPU).
        
        Returns:
            Device string: "cuda:0", "cuda:1", or "cpu"
        """
        if self.qwen3_config.device != "auto":
            return self.qwen3_config.device
        
        # Auto-detect
        if torch.cuda.is_available():
            device = "cuda:0"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if vram_gb < self.qwen3_config.min_vram_gb:
                logger.warning(
                    f"GPU has {vram_gb:.1f}GB VRAM, but {self.qwen3_config.min_vram_gb}GB "
                    f"recommended. Synthesis may be slow or fail."
                )
            else:
                logger.info(f"Using GPU with {vram_gb:.1f}GB VRAM")
            
            return device
        else:
            logger.warning("No CUDA GPU detected. Qwen3-TTS will be VERY slow on CPU.")
            return "cpu"
    
    def _has_flash_attn(self) -> bool:
        """Check if FlashAttention 2 is available.
        
        Returns:
            True if flash-attn is installed
        """
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _get_model_name(self, model_type: ModelType) -> str:
        """Get model name from config for given model type."""
        if model_type == ModelType.BASE:
            return self.qwen3_config.model_name
        elif model_type == ModelType.DESIGN:
            return self.qwen3_config.design_model_name
        elif model_type == ModelType.CUSTOM:
            return self.qwen3_config.custom_model_name
        raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_specific_model(self, model_type: ModelType) -> None:
        """Load a specific Qwen3-TTS model into memory.
        
        Args:
            model_type: Which model to load (BASE, DESIGN, or CUSTOM)
        
        Raises:
            RuntimeError: If model loading fails
        """
        model_name = self._get_model_name(model_type)
        logger.info(f"Loading Qwen3-TTS {model_type.value} model: {model_name}")
        start_time = time.time()
        
        try:
            from qwen_tts import Qwen3TTSModel
            
            attn_impl = "flash_attention_2" if self._has_flash_attn() else "eager"
            if attn_impl == "eager":
                logger.warning("FlashAttention not available. Using eager attention (slower).")
            
            dtype = torch.float16 if self.qwen3_config.model_precision == "float16" else torch.float32
            
            model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=self.device if self.device != "cpu" else None,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            
            # Store in appropriate slot
            if model_type == ModelType.BASE:
                self._base_model = model
            elif model_type == ModelType.DESIGN:
                self._design_model = model
            elif model_type == ModelType.CUSTOM:
                self._custom_model = model
            
            self._current_model_type = model_type
            self.model_loaded_at = time.time()
            self.last_used_at = time.time()
            
            load_time = time.time() - start_time
            logger.info(f"✓ Qwen3 {model_type.value} model loaded in {load_time:.2f}s on {model.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import qwen_tts: {e}")
            raise RuntimeError(f"Qwen3-TTS not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to load Qwen3 {model_type.value} model: {e}")
            raise RuntimeError(f"Qwen3 model loading failed: {e}")
    
    def _unload_current_model(self) -> None:
        """Unload the currently loaded model to free GPU memory."""
        if self._current_model_type is None:
            return
        
        logger.info(f"Unloading Qwen3 {self._current_model_type.value} model...")
        
        try:
            if self._current_model_type == ModelType.BASE:
                if self._base_model is not None:
                    del self._base_model
                    self._base_model = None
            elif self._current_model_type == ModelType.DESIGN:
                if self._design_model is not None:
                    del self._design_model
                    self._design_model = None
            elif self._current_model_type == ModelType.CUSTOM:
                if self._custom_model is not None:
                    del self._custom_model
                    self._custom_model = None
            
            self._current_model_type = None
            self.model_loaded_at = None
            
            # Clear voice prompts (they reference model tensors)
            self._voice_prompts.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✓ Qwen3 model unloaded")
            
        except Exception as e:
            logger.error(f"Error during model unload: {e}")
    
    def _ensure_model(self, model_type: ModelType) -> None:
        """Ensure the specified model is loaded, switching if necessary.
        
        This handles the VRAM constraint by unloading other models first.
        
        Args:
            model_type: Which model to ensure is loaded
        """
        # Already loaded?
        if self._current_model_type == model_type:
            return
        
        # Need to switch - unload current model first
        if self._current_model_type is not None:
            logger.info(f"Switching from {self._current_model_type.value} to {model_type.value} model...")
            self._unload_current_model()
        
        # Load the requested model
        self._load_specific_model(model_type)
    
    def load_model(self, model_type: ModelType = ModelType.BASE) -> None:
        """Load a Qwen3-TTS model into memory.
        
        Args:
            model_type: Which model to load (default: BASE for voice cloning)
        
        Raises:
            RuntimeError: If model loading fails
        """
        self._ensure_model(model_type)
    
    def load_model_async(self, model_type: ModelType = ModelType.BASE) -> None:
        """Start loading a model in a background thread.
        
        Non-blocking. Use _wait_for_model() to wait for completion.
        
        Args:
            model_type: Which model to load
        """
        if self._current_model_type == model_type:
            logger.debug(f"{model_type.value} model already loaded")
            return
        
        if self._loading_thread is not None and self._loading_thread.is_alive():
            logger.debug("Model already loading in background")
            return
        
        def _load_in_background():
            try:
                self._ensure_model(model_type)
            except Exception as e:
                self._loading_error = e
                logger.error(f"Background model loading failed: {e}")
        
        logger.info(f"Starting background {model_type.value} model load...")
        self._loading_error = None
        self._loading_model_type = model_type
        self._loading_thread = threading.Thread(target=_load_in_background, daemon=True)
        self._loading_thread.start()
    
    def _wait_for_model(self, model_type: ModelType = ModelType.BASE) -> None:
        """Wait for model to be ready (loads if needed).
        
        If background loading is in progress, waits for it.
        If no loading started, triggers synchronous load.
        
        Args:
            model_type: Which model to wait for
        
        Raises:
            RuntimeError: If model loading failed
        """
        # Check for background loading error
        if self._loading_error is not None:
            raise RuntimeError(f"Model loading failed: {self._loading_error}")
        
        # Wait for background thread if running
        if self._loading_thread is not None and self._loading_thread.is_alive():
            logger.info("Waiting for background model load to complete...")
            self._loading_thread.join()
            
            # Check for error after join
            if self._loading_error is not None:
                raise RuntimeError(f"Model loading failed: {self._loading_error}")
        
        # Ensure correct model is loaded
        self._ensure_model(model_type)
    
    def _warmup_model(self) -> None:
        """Warm up model with a short synthesis to trigger JIT/CUDA kernel compilation."""
        # Only warmup BASE model (most common use case)
        if self._current_model_type != ModelType.BASE or self._base_model is None:
            logger.debug("Skipping warmup: base model not loaded")
            return
        
        if not self.voice_embeddings:
            logger.debug("Skipping warmup: no voices available yet")
            return
        
        # Pick any available voice
        first_voice_id = next(iter(self.voice_embeddings))
        embedding = self.voice_embeddings[first_voice_id]
        ref_audio_path = str(embedding.embedding[0])
        
        logger.info("Warming up base model (first inference)...")
        warmup_start = time.time()
        try:
            with torch.inference_mode():
                self._base_model.generate_voice_clone(
                    text="Hello.",
                    language="english",
                    ref_audio=ref_audio_path,
                    ref_text=embedding.ref_text,
                    x_vector_only_mode=(embedding.ref_text is None),
                )
            logger.info(f"✓ Model warmed up in {time.time() - warmup_start:.1f}s")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")
    
    def _get_voice_prompt(self, voice_id: str) -> object:
        """Get or create a cached VoiceClonePromptItem for a voice.
        
        Pre-computing the voice prompt saves ~200-500ms per chunk by avoiding
        repeated reference audio loading and speaker embedding extraction.
        
        Args:
            voice_id: Voice identifier (must exist in voice_embeddings)
        
        Returns:
            VoiceClonePromptItem from Qwen3 API
        
        Raises:
            RuntimeError: If base model is not loaded
        """
        if voice_id in self._voice_prompts:
            return self._voice_prompts[voice_id]
        
        embedding = self.voice_embeddings.get(voice_id)
        if embedding is None:
            raise ValueError(f"Voice not found: {voice_id}")
        
        # Ensure base model is loaded (voice prompts are only for cloning)
        self._ensure_model(ModelType.BASE)
        if self._base_model is None:
            raise RuntimeError("Base model not loaded")
        
        ref_audio_path = str(embedding.embedding[0])
        ref_text = embedding.ref_text
        
        logger.debug(f"Creating voice prompt for '{voice_id}'...")
        prompt_items = self._base_model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=(ref_text is None),
        )
        prompt = prompt_items[0]  # Single voice -> single prompt
        
        self._voice_prompts[voice_id] = prompt
        logger.debug(f"✓ Voice prompt cached for '{voice_id}'")
        return prompt
    
    def unload_model(self, model_type: Optional[ModelType] = None) -> None:
        """Unload model(s) to free GPU memory.
        
        Args:
            model_type: Specific model to unload, or None to unload all
        """
        import gc
        
        def _unload_one(mtype: ModelType) -> bool:
            """Unload a specific model. Returns True if unloaded."""
            if mtype == ModelType.BASE and self._base_model is not None:
                del self._base_model
                self._base_model = None
                return True
            elif mtype == ModelType.DESIGN and self._design_model is not None:
                del self._design_model
                self._design_model = None
                return True
            elif mtype == ModelType.CUSTOM and self._custom_model is not None:
                del self._custom_model
                self._custom_model = None
                return True
            return False
        
        if model_type is not None:
            # Unload specific model
            if _unload_one(model_type):
                logger.info(f"Unloading Qwen3 {model_type.value} model...")
                if self._current_model_type == model_type:
                    self._current_model_type = None
                # Clear voice prompts if unloading base (they reference base model tensors)
                if model_type == ModelType.BASE:
                    self._voice_prompts.clear()
            else:
                logger.debug(f"{model_type.value} model already unloaded")
                return
        else:
            # Unload all models
            any_unloaded = False
            for mtype in ModelType:
                if _unload_one(mtype):
                    any_unloaded = True
            
            if not any_unloaded:
                logger.debug("All models already unloaded")
                return
            
            logger.info("Unloading all Qwen3 models...")
            self._current_model_type = None
            self._voice_prompts.clear()
        
        self.model_loaded_at = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✓ Qwen3 model(s) unloaded")
        
        # Clear loading thread reference
        self._loading_thread = None
        self._loading_error = None
    
    def check_auto_unload(self) -> None:
        """Check if model should be auto-unloaded due to inactivity."""
        if not self.qwen3_config.auto_unload:
            return
        
        if self._current_model_type is None or self.last_used_at is None:
            return
        
        idle_time = time.time() - self.last_used_at
        if idle_time > self.qwen3_config.unload_timeout_seconds:
            logger.info(f"Auto-unloading models after {idle_time:.0f}s idle")
            self.unload_model()  # Unload all
    
    def extract_voice_embedding(
        self,
        voice_id: str,
        audio_path: Path,
        ref_text: Optional[str] = None,
    ) -> VoiceEmbedding:
        """Store voice audio reference for cloning.
        
        Note: Qwen3-TTS handles embedding extraction internally during synthesis.
        This method copies the audio file to permanent storage and saves metadata.
        
        Args:
            voice_id: Unique identifier for this voice
            audio_path: Path to audio file (3-15 seconds recommended, can be temp file)
            ref_text: Optional reference text transcript of the audio
        
        Returns:
            VoiceEmbedding object with audio path stored
        
        Raises:
            ValueError: If audio duration is invalid
            FileNotFoundError: If audio file doesn't exist
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Storing voice reference: {voice_id} from {audio_path.name}")
        
        try:
            # Load audio to validate and get metadata
            audio_data, sr = sf.read(str(audio_path))
            
            # Validate duration
            duration = len(audio_data) / sr
            if duration < self.qwen3_config.min_audio_duration:
                raise ValueError(
                    f"Audio too short: {duration:.1f}s < {self.qwen3_config.min_audio_duration}s"
                )
            if duration > self.qwen3_config.max_audio_duration:
                raise ValueError(
                    f"Audio too long: {duration:.1f}s > {self.qwen3_config.max_audio_duration}s"
                )
            
            # Copy audio to permanent storage (in case source is a temp file)
            # Use voice_id as filename to make it easy to find
            permanent_path = self.voices_dir / f"{voice_id}.wav"
            
            # Save audio to permanent location
            sf.write(str(permanent_path), audio_data, sr)
            logger.info(f"Saved voice audio to: {permanent_path}")
            
            # Store the permanent audio file path (Qwen3-TTS will load it during synthesis)
            # We store the path as a string in the embedding field
            embedding = VoiceEmbedding(
                voice_id=voice_id,
                embedding=np.array([str(permanent_path)], dtype=object),  # Store permanent path
                sample_rate=sr,
                duration=duration,
                created_at=time.time(),
                ref_text=ref_text,  # Store reference text in dataclass field
            )
            
            # Cache embedding
            if self.qwen3_config.cache_embeddings:
                self.voice_embeddings[voice_id] = embedding
                logger.info(f"✓ Voice reference cached: {voice_id} ({duration:.1f}s)")
                
                # Evict old embeddings if cache is full
                if len(self.voice_embeddings) > self.qwen3_config.max_cached_voices:
                    oldest_voice = min(self.voice_embeddings.items(), key=lambda x: x[1].created_at)[0]
                    del self.voice_embeddings[oldest_voice]
                    logger.debug(f"Evicted oldest voice: {oldest_voice}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to store voice reference: {e}")
            raise
    
    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _preprocess_text_for_tts(self, text: str) -> str:
        """Strip markdown/formatting symbols that make TTS sound wrong."""
        # Fenced code blocks → spoken notice
        text = re.sub(r'```[\s\S]*?```', ' [code block] ', text)
        # Inline code → unwrapped content
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # LaTeX display math
        text = re.sub(r'\$\$[\s\S]*?\$\$', ' [formula] ', text)
        text = re.sub(r'\$[^$]+\$', ' formula ', text)
        # Markdown bold / italic
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
        # ATX headings
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Markdown links → link text only
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Bare URLs
        text = re.sub(r'https?://\S+', '', text)
        # Bullet / numbered list markers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Excessive whitespace
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _split_into_sentences(self, text: str, max_chars: int = 200) -> list:
        """Split text into speakable chunks of up to max_chars characters.

        Splits first on paragraph breaks, then on sentence-ending punctuation,
        then merges short fragments and subdivides overlong ones at word boundaries.
        """
        # Paragraph-level split first
        paragraphs = re.split(r'\n{2,}', text)
        raw: list = []
        for para in paragraphs:
            # Sentence-level split within each paragraph
            parts = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s+', para.strip())
            raw.extend(p.strip() for p in parts if p.strip())

        # Merge short adjacent sentences / split overlong ones
        chunks: list = []
        buf = ''
        for sentence in raw:
            if not sentence:
                continue
            candidate = (buf + ' ' + sentence).strip() if buf else sentence
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                if len(sentence) <= max_chars:
                    buf = sentence
                else:
                    # Hard-split at word boundaries
                    words = sentence.split()
                    buf = ''
                    for word in words:
                        trial = (buf + ' ' + word).strip() if buf else word
                        if len(trial) <= max_chars:
                            buf = trial
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = word
        if buf:
            chunks.append(buf)

        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def _synthesize_single(
        self,
        text: str,
        voice_id: str,
        speed: float,
        language: str,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize one text chunk using a pre-computed voice prompt.
        
        Uses torch.inference_mode() for 5-10% faster inference than no_grad(),
        and reuses the cached VoiceClonePromptItem to avoid per-chunk embedding extraction.
        
        Raises:
            RuntimeError: If base model is not loaded
        """
        # Ensure base model is loaded for voice cloning
        self._ensure_model(ModelType.BASE)
        if self._base_model is None:
            raise RuntimeError("Base model not loaded for synthesis")
        
        # Get or create cached voice prompt (saves ~200-500ms per chunk)
        voice_prompt = self._get_voice_prompt(voice_id)

        with torch.inference_mode():
            wavs, sr = self._base_model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=[voice_prompt],  # Must be a list of VoiceClonePromptItem
            )

        audio = wavs[0] if isinstance(wavs, list) else wavs

        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=speed)
            except ImportError:
                logger.warning("librosa not installed, speed adjustment skipped")

        return audio, sr

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        language: str = "english",
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using cloned voice.

        Long texts are automatically split into sentences and synthesized in
        separate chunks, then concatenated.  This gives a large speed-up
        because inference cost scales super-linearly with sequence length.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if voice_id and voice_id not in self.voice_embeddings:
            raise ValueError(f"Voice not found: {voice_id}. Available: {list(self.voice_embeddings.keys())}")

        embedding = self.voice_embeddings.get(voice_id) if voice_id else None
        if embedding is None:
            raise ValueError(
                f"Qwen3-TTS Base model requires a voice reference for synthesis. "
                f"Use extract_voice_embedding() to create a voice first, or specify a valid voice_id. "
                f"Available voices: {list(self.voice_embeddings.keys())}"
            )

        self._wait_for_model(ModelType.BASE)

        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        if not chunks:
            raise ValueError("Text is empty after preprocessing")

        logger.info(f"Synthesizing {len(chunks)} chunk(s), voice={voice_id}, lang={language}")
        start_time = time.time()

        audio_parts: list = []
        sample_rate = 24000
        silence: Optional[np.ndarray] = None

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            audio, sr = self._synthesize_single(chunk, voice_id, speed, language)
            sample_rate = sr
            if silence is None:
                silence = np.zeros(int(0.08 * sr), dtype=np.float32)  # 80 ms gap
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
            if i == 0 or i == len(chunks) - 1:
                logger.info(f"  chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")

        combined = np.concatenate(audio_parts)

        synth_time = time.time() - start_time
        audio_duration = len(combined) / sample_rate
        self.synthesis_count += 1
        self.total_synthesis_time += synth_time
        self.last_used_at = time.time()
        logger.info(
            f"✓ Synthesis complete: {audio_duration:.1f}s audio in {synth_time:.1f}s "
            f"({len(chunks)} chunk(s), RTF: {synth_time/audio_duration:.2f}x)"
        )

        return combined, sample_rate

    def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        language: str = "english",
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield (audio_array, sample_rate) for each sentence chunk.

        Allows the caller to stream audio to the client sentence-by-sentence
        so playback can start before synthesis of the full text is complete.
        """
        if not text or not text.strip():
            return

        if voice_id and voice_id not in self.voice_embeddings:
            raise ValueError(f"Voice not found: {voice_id}")

        embedding = self.voice_embeddings.get(voice_id) if voice_id else None
        if embedding is None:
            raise ValueError(
                f"No voice available. Available: {list(self.voice_embeddings.keys())}"
            )

        self._wait_for_model(ModelType.BASE)

        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        logger.info(f"Streaming {len(chunks)} chunk(s), voice={voice_id}, lang={language}")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                audio, sr = self._synthesize_single(chunk, voice_id, speed, language)
                self.last_used_at = time.time()
                if i == 0 or i == len(chunks) - 1:
                    logger.info(f"  streaming chunk {i+1}/{len(chunks)}")
                yield audio, sr
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {e}")
                raise
    
    def list_cloned_voices(self) -> Dict[str, VoiceEmbedding]:
        """List all cloned voices in cache.
        
        Returns:
            Dictionary mapping voice_id to VoiceEmbedding
        """
        return self.voice_embeddings.copy()
    
    def _scan_cloned_voices(self) -> int:
        """Scan voices_dir for .wav files and load any not already in voice_embeddings.
        
        Called at startup and can be called again to pick up externally added files.
        
        Returns:
            Number of voices newly loaded from disk
        """
        if not self.voices_dir.exists():
            return 0
        
        loaded = 0
        for wav_file in sorted(self.voices_dir.glob("*.wav")):
            voice_id = wav_file.stem
            if voice_id in self.voice_embeddings:
                continue  # Already in memory
            try:
                audio_data, sr = sf.read(str(wav_file))
                duration = len(audio_data) / sr
                embedding = VoiceEmbedding(
                    voice_id=voice_id,
                    embedding=np.array([str(wav_file)], dtype=object),
                    sample_rate=sr,
                    duration=duration,
                    created_at=wav_file.stat().st_mtime,
                    ref_text=None,
                )
                self.voice_embeddings[voice_id] = embedding
                loaded += 1
                logger.info(f"Loaded cloned voice from disk: {voice_id} ({duration:.1f}s, {sr}Hz)")
            except Exception as e:
                logger.warning(f"Failed to load voice file {wav_file.name}: {e}")
        
        return loaded
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice from cache and storage.
        
        Args:
            voice_id: Voice identifier to delete
        
        Returns:
            True if deleted, False if not found
        """
        if voice_id in self.voice_embeddings:
            # Delete from cache
            embedding = self.voice_embeddings[voice_id]
            del self.voice_embeddings[voice_id]
            
            # Delete audio file from storage
            try:
                audio_path = Path(str(embedding.embedding[0]))
                if audio_path.exists():
                    audio_path.unlink()
                    logger.info(f"Deleted voice audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete audio file for {voice_id}: {e}")
            
            logger.info(f"Deleted voice: {voice_id}")
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get synthesis statistics.
        
        Returns:
            Dictionary with stats (count, total time, avg RTF, etc.)
        """
        return {
            "synthesis_count": self.synthesis_count,
            "total_synthesis_time": self.total_synthesis_time,
            "average_rtf": self.get_average_rtf(),
            "cached_voices": len(self.voice_embeddings),
            "current_model": self._current_model_type.value if self._current_model_type else None,
            "base_model_loaded": self._base_model is not None,
            "design_model_loaded": self._design_model is not None,
            "custom_model_loaded": self._custom_model is not None,
            "device": self.device,
        }
    
    def get_average_rtf(self) -> float:
        """Get average real-time factor.
        
        Returns:
            Average RTF (0 if no syntheses yet)
        """
        if self.synthesis_count == 0:
            return 0.0
        avg_synth_time = self.total_synthesis_time / self.synthesis_count
        # Assume average sentence is ~10 words, ~2s duration
        estimated_avg_duration = 2.0
        return avg_synth_time / estimated_avg_duration

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code/name to canonical form.
        
        Args:
            language: Language name or code (e.g., "en", "english", "eng")
        
        Returns:
            Canonical language name (e.g., "english")
        
        Raises:
            ValueError: If language is not supported
        """
        lang_lower = language.lower().strip()
        
        # Check if already canonical
        if lang_lower in SUPPORTED_LANGUAGES:
            return lang_lower
        
        # Try aliases
        if lang_lower in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[lang_lower]
        
        # Not found
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for voice generation.
        
        Returns:
            List of supported language names
        """
        return sorted(SUPPORTED_LANGUAGES)
    
    # ------------------------------------------------------------------
    # Voice Generation (VoiceDesign model)
    # ------------------------------------------------------------------
    
    def generate_voice(
        self,
        text: str,
        voice_description: str,
        language: str = "english",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with a designed voice using VoiceDesign model.
        
        This method creates a voice on-the-fly from a text description,
        without needing a reference audio sample. The voice characteristics
        are determined by the voice_description.
        
        Note: For consistent voice across multiple calls, use design_and_save_voice()
        to create a reusable voice, then use synthesize() with that voice_id.
        
        Args:
            text: Text to synthesize
            voice_description: Natural language description of desired voice
                e.g., "A mature male voice with a warm, deep tone speaking slowly"
            language: Target language (default: "english")
            speed: Playback speed multiplier (default: 1.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        
        Raises:
            ValueError: If text is empty or language is unsupported
            RuntimeError: If model loading fails
        
        Example:
            >>> audio, sr = manager.generate_voice(
            ...     text="Hello, welcome to the demo.",
            ...     voice_description="A young female voice, clear and energetic",
            ...     language="english"
            ... )
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description cannot be empty")
        
        # Normalize and validate language
        language = self._normalize_language(language)
        
        # Ensure design model is loaded
        self._ensure_model(ModelType.DESIGN)
        if self._design_model is None:
            raise RuntimeError("Failed to load VoiceDesign model")
        
        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        if not chunks:
            raise ValueError("Text is empty after preprocessing")
        
        logger.info(
            f"Generating speech with designed voice: {len(chunks)} chunk(s), "
            f"lang={language}, description='{voice_description[:50]}...'"
        )
        start_time = time.time()
        
        audio_parts: list = []
        sample_rate = 24000
        silence: Optional[np.ndarray] = None
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            with torch.inference_mode():
                wavs, sr = self._design_model.generate_voice_design(
                    text=chunk,
                    instruct=voice_description,  # API uses 'instruct' for voice description
                    language=language,
                )
            
            audio = wavs[0] if isinstance(wavs, list) else wavs
            sample_rate = sr
            
            if speed != 1.0:
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=speed)
                except ImportError:
                    if i == 0:
                        logger.warning("librosa not installed, speed adjustment skipped")
            
            if silence is None:
                silence = np.zeros(int(0.08 * sr), dtype=np.float32)  # 80 ms gap
            
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
            
            if i == 0 or i == len(chunks) - 1:
                logger.info(f"  chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")
        
        combined = np.concatenate(audio_parts)
        
        synth_time = time.time() - start_time
        audio_duration = len(combined) / sample_rate
        self.synthesis_count += 1
        self.total_synthesis_time += synth_time
        self.last_used_at = time.time()
        
        logger.info(
            f"✓ Voice design synthesis complete: {audio_duration:.1f}s audio in {synth_time:.1f}s "
            f"({len(chunks)} chunk(s), RTF: {synth_time/audio_duration:.2f}x)"
        )
        
        return combined, sample_rate
    
    def generate_voice_stream(
        self,
        text: str,
        voice_description: str,
        language: str = "english",
        speed: float = 1.0,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Stream speech generation with designed voice chunk by chunk.
        
        Similar to generate_voice() but yields audio chunks as they're generated,
        enabling faster time-to-first-audio for long texts.
        
        Args:
            text: Text to synthesize
            voice_description: Natural language description of desired voice
            language: Target language (default: "english")
            speed: Playback speed multiplier (default: 1.0)
        
        Yields:
            Tuple of (audio_chunk, sample_rate) for each sentence
        """
        if not text or not text.strip():
            return
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description cannot be empty")
        
        language = self._normalize_language(language)
        
        self._ensure_model(ModelType.DESIGN)
        if self._design_model is None:
            raise RuntimeError("Failed to load VoiceDesign model")
        
        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        
        logger.info(
            f"Streaming voice design: {len(chunks)} chunk(s), lang={language}"
        )
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                with torch.inference_mode():
                    wavs, sr = self._design_model.generate_voice_design(
                        text=chunk,
                        instruct=voice_description,  # API uses 'instruct' for voice description
                        language=language,
                    )
                
                audio = wavs[0] if isinstance(wavs, list) else wavs
                
                if speed != 1.0:
                    try:
                        import librosa
                        audio = librosa.effects.time_stretch(audio, rate=speed)
                    except ImportError:
                        pass
                
                self.last_used_at = time.time()
                if i == 0 or i == len(chunks) - 1:
                    logger.info(f"  streaming design chunk {i+1}/{len(chunks)}")
                yield audio, sr
            except Exception as e:
                logger.error(f"Voice design chunk {i+1} failed: {e}")
                raise
    
    def design_and_save_voice(
        self,
        voice_id: str,
        voice_description: str,
        sample_text: Optional[str] = None,
        language: str = "english",
    ) -> VoiceEmbedding:
        """Design a voice and save it as a cloneable voice for consistent reuse.
        
        This method generates a sample audio using VoiceDesign model, then saves
        it as a cloneable voice that can be used with synthesize() for consistent
        voice characteristics across multiple calls.
        
        Workflow (Design-then-Clone):
            1. Use VoiceDesign model to generate sample audio from description
            2. Save the generated audio as a reference for voice cloning
            3. Future calls to synthesize() with this voice_id use Base model
        
        Args:
            voice_id: Unique identifier for the new voice
            voice_description: Natural language description of desired voice
            sample_text: Optional text for the sample audio (default: generic phrase)
            language: Language for the sample (default: "english")
        
        Returns:
            VoiceEmbedding for the newly created voice
        
        Raises:
            ValueError: If voice_id already exists or description is empty
            RuntimeError: If voice generation fails
        
        Example:
            >>> # Create a custom voice
            >>> embedding = manager.design_and_save_voice(
            ...     voice_id="narrator_deep",
            ...     voice_description="A deep, authoritative male voice with British accent",
            ... )
            >>> # Now use it for all future synthesis
            >>> audio, sr = manager.synthesize(
            ...     text="Chapter one begins...",
            ...     voice_id="narrator_deep"
            ... )
        """
        if voice_id in self.voice_embeddings:
            raise ValueError(f"Voice '{voice_id}' already exists. Use a different ID or delete first.")
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description cannot be empty")
        
        language = self._normalize_language(language)
        
        # Default sample text if not provided
        if not sample_text:
            sample_text = (
                "Hello, this is a sample of my voice. "
                "I hope you find it pleasant and suitable for your needs."
            )
        
        logger.info(f"Designing voice '{voice_id}' from description...")
        
        # Generate sample audio using VoiceDesign model
        self._ensure_model(ModelType.DESIGN)
        if self._design_model is None:
            raise RuntimeError("Failed to load VoiceDesign model")
        
        with torch.inference_mode():
            wavs, sr = self._design_model.generate_voice_design(
                text=sample_text,
                instruct=voice_description,  # API uses 'instruct' for voice description
                language=language,
            )
        
        audio = wavs[0] if isinstance(wavs, list) else wavs
        duration = len(audio) / sr
        
        # Validate duration (need enough for good cloning)
        if duration < self.qwen3_config.min_audio_duration:
            logger.warning(
                f"Generated sample is short ({duration:.1f}s), "
                f"consider using longer sample_text for better cloning"
            )
        
        # Save to permanent storage
        permanent_path = self.voices_dir / f"{voice_id}.wav"
        sf.write(str(permanent_path), audio, sr)
        logger.info(f"Saved designed voice sample to: {permanent_path}")
        
        # Create embedding with design metadata
        embedding = VoiceEmbedding(
            voice_id=voice_id,
            embedding=np.array([str(permanent_path)], dtype=object),
            sample_rate=sr,
            duration=duration,
            created_at=time.time(),
            ref_text=sample_text,  # Store sample text for potential reference
            source="designed",  # Mark as designed (not recorded)
            voice_description=voice_description,  # Store original description
        )
        
        # Cache in memory
        self.voice_embeddings[voice_id] = embedding
        
        # Evict old if cache is full
        if len(self.voice_embeddings) > self.qwen3_config.max_cached_voices:
            oldest_voice = min(
                self.voice_embeddings.items(),
                key=lambda x: x[1].created_at
            )[0]
            if oldest_voice != voice_id:
                del self.voice_embeddings[oldest_voice]
                logger.debug(f"Evicted oldest voice: {oldest_voice}")
        
        logger.info(
            f"✓ Created designed voice '{voice_id}' ({duration:.1f}s sample, "
            f"description: '{voice_description[:40]}...')"
        )
        
        # Also save metadata as JSON for persistence
        try:
            metadata_path = self.voices_dir / f"{voice_id}.json"
            metadata = {
                "voice_id": voice_id,
                "voice_description": voice_description,
                "sample_text": sample_text,
                "language": language,
                "source": "designed",
                "duration": duration,
                "sample_rate": sr,
                "created_at": embedding.created_at,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved voice metadata to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save voice metadata JSON: {e}")
        
        return embedding
    
    # ------------------------------------------------------------------
    # Custom Voice (CustomVoice model with instruction control)
    # ------------------------------------------------------------------
    
    def list_custom_speakers(self) -> List[str]:
        """Get list of available custom speakers.
        
        Returns:
            List of speaker names (e.g., ["Vivian", "Ryan", "Aiden", ...])
        """
        return list(CUSTOM_SPEAKERS.keys())
    
    def get_speaker_info(self, speaker: str) -> Optional[Dict]:
        """Get metadata for a custom speaker.
        
        Args:
            speaker: Speaker name (case-sensitive)
        
        Returns:
            Dictionary with speaker info, or None if not found
            Keys: description, native_language, gender, age_range
        """
        return CUSTOM_SPEAKERS.get(speaker)
    
    def get_all_speakers_info(self) -> Dict[str, Dict]:
        """Get metadata for all custom speakers.
        
        Returns:
            Dictionary mapping speaker names to their info
        """
        return CUSTOM_SPEAKERS.copy()
    
    def _validate_speaker(self, speaker: str) -> str:
        """Validate and normalize speaker name.
        
        Args:
            speaker: Speaker name (case-insensitive matching)
        
        Returns:
            Normalized speaker name
        
        Raises:
            ValueError: If speaker is not found
        """
        # Try exact match first
        if speaker in CUSTOM_SPEAKERS:
            return speaker
        
        # Try case-insensitive match
        speaker_lower = speaker.lower()
        for name in CUSTOM_SPEAKERS:
            if name.lower() == speaker_lower:
                return name
        
        # Not found
        raise ValueError(
            f"Unknown speaker: '{speaker}'. "
            f"Available speakers: {', '.join(CUSTOM_SPEAKERS.keys())}"
        )
    
    def synthesize_custom(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using CustomVoice model with premium speakers.
        
        This method uses pre-built premium speakers with optional instruction
        control. Instructions can modify the speaking style, emotion, pace, etc.
        
        Args:
            text: Text to synthesize
            speaker: Speaker name (e.g., "Ryan", "Vivian", "Aiden")
            language: Target language (default: "auto" for speaker's native)
            instruct: Optional instruction for voice control
                e.g., "Speak with excitement", "Calm and professional"
            speed: Playback speed multiplier (default: 1.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        
        Raises:
            ValueError: If text is empty, speaker is invalid, or language unsupported
            RuntimeError: If model loading fails
        
        Example:
            >>> audio, sr = manager.synthesize_custom(
            ...     text="Welcome to the presentation!",
            ...     speaker="Ryan",
            ...     instruct="Speak with enthusiasm and energy"
            ... )
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Validate speaker
        speaker = self._validate_speaker(speaker)
        speaker_info = CUSTOM_SPEAKERS[speaker]
        
        # Handle language
        if language == "auto":
            # Use speaker's native language
            language = speaker_info["native_language"]
        else:
            language = self._normalize_language(language)
        
        # Ensure custom model is loaded
        self._ensure_model(ModelType.CUSTOM)
        if self._custom_model is None:
            raise RuntimeError("Failed to load CustomVoice model")
        
        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        if not chunks:
            raise ValueError("Text is empty after preprocessing")
        
        instruct_str = instruct if instruct else "(default style)"
        logger.info(
            f"Synthesizing with custom speaker: {speaker}, "
            f"lang={language}, instruct='{instruct_str[:50]}', {len(chunks)} chunk(s)"
        )
        start_time = time.time()
        
        audio_parts: list = []
        sample_rate = 24000
        silence: Optional[np.ndarray] = None
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            with torch.inference_mode():
                # Call generate_custom_voice with or without instruction
                if instruct:
                    wavs, sr = self._custom_model.generate_custom_voice(
                        text=chunk,
                        speaker=speaker,
                        language=language,
                        instruct=instruct,
                    )
                else:
                    wavs, sr = self._custom_model.generate_custom_voice(
                        text=chunk,
                        speaker=speaker,
                        language=language,
                    )
            
            audio = wavs[0] if isinstance(wavs, list) else wavs
            sample_rate = sr
            
            if speed != 1.0:
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=speed)
                except ImportError:
                    if i == 0:
                        logger.warning("librosa not installed, speed adjustment skipped")
            
            if silence is None:
                silence = np.zeros(int(0.08 * sr), dtype=np.float32)  # 80 ms gap
            
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
            
            if i == 0 or i == len(chunks) - 1:
                logger.info(f"  chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")
        
        combined = np.concatenate(audio_parts)
        
        synth_time = time.time() - start_time
        audio_duration = len(combined) / sample_rate
        self.synthesis_count += 1
        self.total_synthesis_time += synth_time
        self.last_used_at = time.time()
        
        logger.info(
            f"✓ Custom voice synthesis complete: {audio_duration:.1f}s audio in {synth_time:.1f}s "
            f"(speaker={speaker}, RTF: {synth_time/audio_duration:.2f}x)"
        )
        
        return combined, sample_rate
    
    def synthesize_custom_stream(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Stream speech synthesis using CustomVoice model chunk by chunk.
        
        Similar to synthesize_custom() but yields audio chunks as they're generated,
        enabling faster time-to-first-audio for long texts.
        
        Args:
            text: Text to synthesize
            speaker: Speaker name (e.g., "Ryan", "Vivian")
            language: Target language (default: "auto")
            instruct: Optional instruction for voice control
            speed: Playback speed multiplier (default: 1.0)
        
        Yields:
            Tuple of (audio_chunk, sample_rate) for each sentence
        """
        if not text or not text.strip():
            return
        
        # Validate speaker
        speaker = self._validate_speaker(speaker)
        speaker_info = CUSTOM_SPEAKERS[speaker]
        
        # Handle language
        if language == "auto":
            language = speaker_info["native_language"]
        else:
            language = self._normalize_language(language)
        
        # Ensure custom model is loaded
        self._ensure_model(ModelType.CUSTOM)
        if self._custom_model is None:
            raise RuntimeError("Failed to load CustomVoice model")
        
        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=self.qwen3_config.chunk_size)
        
        logger.info(
            f"Streaming custom voice: {speaker}, {len(chunks)} chunk(s), lang={language}"
        )
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                with torch.inference_mode():
                    if instruct:
                        wavs, sr = self._custom_model.generate_custom_voice(
                            text=chunk,
                            speaker=speaker,
                            language=language,
                            instruct=instruct,
                        )
                    else:
                        wavs, sr = self._custom_model.generate_custom_voice(
                            text=chunk,
                            speaker=speaker,
                            language=language,
                        )
                
                audio = wavs[0] if isinstance(wavs, list) else wavs
                
                if speed != 1.0:
                    try:
                        import librosa
                        audio = librosa.effects.time_stretch(audio, rate=speed)
                    except ImportError:
                        pass
                
                self.last_used_at = time.time()
                if i == 0 or i == len(chunks) - 1:
                    logger.info(f"  streaming custom chunk {i+1}/{len(chunks)}")
                yield audio, sr
            except Exception as e:
                logger.error(f"Custom voice chunk {i+1} failed: {e}")
                raise