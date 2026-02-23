"""
Unified TTS Manager for Orion RAG Assistant

Routes between two TTS engines:
1. Piper TTS - Fast, CPU-friendly, pre-built voices
2. Qwen3-TTS - Slow, GPU-required, voice cloning

Features:
- Automatic engine selection based on voice type
- Integrated caching (saves 5-60s on cache hits)
- Queue support for async synthesis (Qwen3)
- Voice management across both engines
- Graceful fallback to Piper if Qwen3 unavailable
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import io
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Information about a TTS voice."""
    
    voice_id: str
    name: str
    language: str
    gender: str
    quality: str
    description: str
    model_size: str
    sample_rate: int
    engine: str  # "piper" or "qwen3"
    is_downloaded: bool = False


class UnifiedTTSManager:
    """
    Unified manager for both Piper and Qwen3 TTS engines.
    
    Routes synthesis requests to the appropriate engine:
    - Piper: Fast synthesis with pre-built voices
    - Qwen3: High-quality synthesis with cloned voices
    
    Features:
    - Automatic engine detection from voice ID
    - Integrated caching (TTSCache)
    - Async queue support (TTSQueue)
    - Voice management across both engines
    """
    
    def __init__(self, config):
        """
        Initialize Unified TTS Manager.
        
        Args:
            config: OrionConfig instance with TTS and Qwen3 settings
        """
        self.config = config
        self.tts_config = config.tts
        self.qwen3_config = config.qwen3
        
        self.voice_catalog: Dict[str, VoiceInfo] = {}
        
        # Engine managers (lazy loading)
        self.piper_voice = None
        self.current_piper_voice: Optional[str] = None
        self._qwen3_manager = None  # Private: use qwen3_manager property for lazy loading
        
        # Cache and queue (optional, set via setters)
        self.cache = None
        self.queue = None
        
        # Lazy imports
        self._piper_module = None
        self._wave_module = None
        
        # Local model directories (data/tts/)
        self.models_dir = Path(__file__).parent.parent.parent.parent / "data" / "tts"
        self.onnx_dir = self.models_dir / "onnx"
        self.json_dir = self.models_dir / "json"
        
        # Ensure directories exist
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Piper voice catalog
        self._load_piper_voices()
        
        # Initialize Qwen3 if enabled
        if self.qwen3_config.enabled:
            self._init_qwen3()
        
        logger.info(
            f"✓ UnifiedTTSManager initialized (Piper: {len([v for v in self.voice_catalog.values() if v.engine == 'piper'])} voices, "
            f"Qwen3: {'enabled' if self.qwen3_config.enabled else 'disabled'})"
        )
    
    def _import_dependencies(self):
        """Lazy import of TTS dependencies."""
        if self._piper_module is None:
            try:
                from piper import PiperVoice
                import wave
                self._piper_module = PiperVoice
                self._wave_module = wave
                logger.debug("✓ Piper TTS dependencies loaded")
            except ImportError as e:
                logger.error(f"Failed to import Piper TTS: {e}")
                raise ImportError(
                    "Piper TTS not installed. Install with: pip install piper-tts"
                )
    
    def _init_qwen3(self) -> None:
        """Initialize Qwen3-TTS manager."""
        try:
            from .qwen3_manager import Qwen3Manager
            self._qwen3_manager = Qwen3Manager(self.config)
            
            # Add Qwen3 cloned voices to catalog
            for voice_id, embedding in self._qwen3_manager.list_cloned_voices().items():
                self.voice_catalog[voice_id] = VoiceInfo(
                    voice_id=voice_id,
                    name=voice_id.replace("_", " ").title(),
                    language="multi",
                    gender="neutral",
                    quality="high",
                    description=f"Cloned voice ({embedding.duration:.1f}s sample)",
                    model_size=f"{len(embedding.embedding) * 4 / 1024:.1f}KB",
                    sample_rate=embedding.sample_rate,
                    engine="qwen3",
                    is_downloaded=True,
                )
            
            logger.info("✓ Qwen3-TTS manager initialized")
        except ImportError as e:
            logger.warning(f"Qwen3-TTS not available: {e}")
            self.qwen3_config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-TTS: {e}")
            self.qwen3_config.enabled = False
    
    @property
    def qwen3_manager(self):
        """Lazy-load Qwen3Manager when first accessed.
        
        This allows Qwen3 to be initialized even if it was disabled
        at startup but later enabled via engine switch.
        
        Returns:
            Qwen3Manager instance or None if not available
        """
        # If already initialized, return it
        if self._qwen3_manager is not None:
            return self._qwen3_manager
        
        # Check if Qwen3 is enabled
        if not self.qwen3_config.enabled:
            logger.warning(f"Qwen3-TTS access attempted but not enabled in configuration (qwen3.enabled={self.qwen3_config.enabled})")
            return None
        
        # Initialize Qwen3 on first access
        logger.info("Lazy-loading Qwen3-TTS manager (enabled={}, first access)...".format(self.qwen3_config.enabled))
        try:
            self._init_qwen3()
            if self._qwen3_manager is None:
                logger.error("Qwen3-TTS initialization completed but manager is still None")
            else:
                logger.info(f"✓ Qwen3-TTS manager lazy-loaded successfully")
            return self._qwen3_manager
        except Exception as e:
            logger.error(f"Failed to lazy-load Qwen3-TTS: {e}", exc_info=True)
            return None
    
    def set_cache(self, cache) -> None:
        """Set TTSCache instance for audio caching.
        
        Args:
            cache: TTSCache instance
        """
        self.cache = cache
        logger.info("✓ TTSCache attached to UnifiedTTSManager")
    
    def set_queue(self, queue) -> None:
        """Set TTSQueue instance for async synthesis.
        
        Args:
            queue: TTSQueue instance
        """
        self.queue = queue
        logger.info("✓ TTSQueue attached to UnifiedTTSManager")
    
    def _load_piper_voices(self):
        """Load voice catalog by scanning local model files."""
        # Scan onnx directory for available models
        if not self.onnx_dir.exists():
            logger.warning(f"ONNX directory not found: {self.onnx_dir}")
            return
        
        onnx_files = list(self.onnx_dir.glob("*.onnx"))
        
        if not onnx_files:
            logger.warning(f"No .onnx model files found in {self.onnx_dir}")
            return
        
        # Create catalog from discovered models
        for onnx_file in onnx_files:
            voice_id = onnx_file.stem  # e.g., "en_US-lessac-medium"
            json_file = self.json_dir / f"{voice_id}.onnx.json"
            
            # Try to load metadata from JSON config
            metadata = self._load_voice_metadata(json_file) if json_file.exists() else {}
            
            # Parse voice_id for defaults (e.g., "en_US-lessac-medium")
            parts = voice_id.split("-")
            language = parts[0] if len(parts) > 0 else "unknown"
            name = parts[1].capitalize() if len(parts) > 1 else voice_id
            quality = parts[2] if len(parts) > 2 else "medium"
            
            self.voice_catalog[voice_id] = VoiceInfo(
                voice_id=voice_id,
                name=metadata.get("name", name),
                language=language,
                gender=metadata.get("gender", "neutral"),
                quality=metadata.get("quality", quality),
                description=metadata.get("description", f"{language} voice - {quality} quality"),
                model_size=f"{onnx_file.stat().st_size / 1024 / 1024:.1f}MB",
                sample_rate=metadata.get("sample_rate", 22050),
                engine="piper",  # All voices from this method are Piper
                is_downloaded=True  # All local models are "downloaded"
            )
        
        logger.info(f"✓ Loaded {len(self.voice_catalog)} voices from local models")
    
    def _load_voice_metadata(self, json_path: Path) -> Dict[str, Any]:
        """Load voice metadata from .onnx.json config file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Extract relevant metadata
                dataset = data.get("dataset", "")
                quality = data.get("audio", {}).get("quality", "medium")
                
                return {
                    "name": dataset.capitalize() if dataset else "",
                    "quality": quality,
                    "description": f"{dataset.capitalize()} voice - {quality} quality" if dataset else "",
                    "sample_rate": data.get("audio", {}).get("sample_rate", 22050),
                    "gender": data.get("gender", "neutral"),
                }
        except Exception as e:
            logger.debug(f"Could not load metadata from {json_path}: {e}")
            return {}
    
    def _is_voice_downloaded(self, voice_id: str) -> bool:
        """Check if voice model exists in local directory."""
        voice_path = self.onnx_dir / f"{voice_id}.onnx"
        return voice_path.exists()
    
    def list_voices(self, language_filter: Optional[str] = None) -> List[VoiceInfo]:
        """
        List all available voices.
        
        Args:
            language_filter: Optional language code filter (e.g., "en_US")
            
        Returns:
            List of VoiceInfo objects
        """
        voices = list(self.voice_catalog.values())
        
        if language_filter:
            voices = [v for v in voices if v.language.startswith(language_filter)]
        
        return voices
    
    def load_voice(self, voice_id: str) -> None:
        """
        Load a specific voice model.
        
        Args:
            voice_id: Voice identifier (e.g., "en_US-lessac-medium")
        """
        # Detect engine
        engine = self._detect_engine(voice_id)
        
        if engine == "piper":
            self._load_piper_voice(voice_id)
        elif engine == "qwen3":
            # Qwen3 voices are always loaded (just embeddings in memory)
            logger.debug(f"Qwen3 voice already loaded: {voice_id}")
        else:
            raise ValueError(f"Unknown engine for voice: {voice_id}")
    
    def _load_piper_voice(self, voice_id: str) -> None:
        """Load a Piper voice model.
        
        Args:
            voice_id: Piper voice identifier
        """
        self._import_dependencies()
        
        if voice_id not in self.voice_catalog:
            raise ValueError(f"Voice '{voice_id}' not found in catalog")
        
        # Check if already loaded
        if self.current_piper_voice == voice_id and self.piper_voice is not None:
            logger.debug(f"Piper voice '{voice_id}' already loaded")
            return
        
        # Unload current voice
        if self.piper_voice is not None:
            self.unload_piper()
        
        # Use local model files
        voice_path = self.onnx_dir / f"{voice_id}.onnx"
        config_path = self.json_dir / f"{voice_id}.onnx.json"
        
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {voice_path}. "
                f"Please place {voice_id}.onnx in {self.onnx_dir} and {voice_id}.onnx.json in {self.json_dir}"
            )
        
        try:
            self.piper_voice = self._piper_module.load(
                str(voice_path),
                config_path=str(config_path) if config_path.exists() else None,
                use_cuda=self.tts_config.use_gpu
            )
            self.current_piper_voice = voice_id
            logger.info(f"✓ Loaded Piper voice: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to load Piper voice '{voice_id}': {e}")
            raise
    
    def _detect_engine(self, voice_id: Optional[str]) -> str:
        """Detect which engine to use for a voice.
        
        Args:
            voice_id: Voice identifier or None
        
        Returns:
            Engine name: "piper" or "qwen3"
        """
        if voice_id is None:
            # No specific voice — respect the configured default engine
            return self.tts_config.default_engine
        
        # Check catalog first (fast path)
        if voice_id in self.voice_catalog:
            return self.voice_catalog[voice_id].engine
        
        # Not in catalog — check qwen3_manager's in-memory embeddings directly.
        # This handles voices that are on disk but weren't in the catalog at init
        # (e.g. lazy-loaded qwen3, or voices added without catalog registration).
        if self._qwen3_manager is not None and voice_id in self._qwen3_manager.voice_embeddings:
            return "qwen3"
        
        # Default to Piper
        return "piper"
    

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: Optional[float] = None,
        output_format: str = "wav",
        language: str = "en",
    ) -> bytes:
        """
        Convert text to speech using appropriate engine.
        
        Routes to Piper (fast) or Qwen3 (quality) based on voice type.
        Uses cache to avoid re-synthesizing identical text.
        
        Args:
            text: Text to synthesize
            voice_id: Voice to use (defaults to configured default)
            speed: Speech speed multiplier (1.0 = normal)
            output_format: Audio format ("wav" or "mp3")
            language: Language code (for Qwen3)
            
        Returns:
            Audio data as bytes
        """
        # Use defaults if not specified
        if voice_id is None:
            if self.tts_config.default_engine == "qwen3":
                # Prefer the user-selected active Qwen3 voice
                voice_id = self.tts_config.active_qwen3_voice
                # If none selected, auto-pick the first available cloned voice
                if voice_id is None and hasattr(self, 'qwen3_manager'):
                    cloned = self.qwen3_manager.list_cloned_voices()
                    if cloned:
                        voice_id = next(iter(cloned))
                        logger.info(f"No active Qwen3 voice set — auto-selecting: {voice_id}")
            else:
                voice_id = self.tts_config.default_voice
        if speed is None:
            speed = self.tts_config.default_speed
        
        # Detect engine
        engine = self._detect_engine(voice_id)
        
        # Check cache first
        if self.cache:
            cached_audio = self.cache.get(text, voice_id, speed, engine)
            if cached_audio:
                logger.debug(f"Cache hit for: {text[:50]}... (engine={engine})")
                return cached_audio
        
        # Synthesize based on engine
        if engine == "piper":
            audio_bytes = self._synthesize_piper(text, voice_id, speed, output_format)
        elif engine == "qwen3":
            audio_bytes = self._synthesize_qwen3(text, voice_id, speed, output_format, language)
        else:
            raise ValueError(f"Unknown engine: {engine}")
        
        # Cache the result
        if self.cache and audio_bytes:
            self.cache.put(text, voice_id, speed, engine, audio_bytes, output_format)
        
        return audio_bytes
    
    def _synthesize_piper(
        self,
        text: str,
        voice_id: str,
        speed: float,
        output_format: str
    ) -> bytes:
        """Synthesize using Piper TTS."""
        # Load voice if needed
        if self.current_piper_voice != voice_id:
            self._load_piper_voice(voice_id)
        
        # Synthesize speech
        try:
            # Get voice info for sample rate
            voice_info = self.voice_catalog[voice_id]
            
            # Create temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                tmp_path = tmpf.name
            
            try:
                # Import synthesis config
                try:
                    from piper.config import SynthesisConfig
                    
                    # Configure synthesis (speed control)
                    syn_config = SynthesisConfig(
                        length_scale=1.0 / speed  # Piper uses inverse speed
                    )
                except (ImportError, TypeError):
                    # Older version or no config support
                    syn_config = None
                    logger.debug("Piper version doesn't support SynthesisConfig")
                
                # Piper's synthesize returns an iterable of audio chunks
                with self._wave_module.open(tmp_path, 'wb') as wav_file:
                    # Configure WAV file parameters
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(voice_info.sample_rate)
                    
                    # Generate and write audio chunks
                    for audio_chunk in self.piper_voice.synthesize(text, syn_config):
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)
                
                # Read the generated WAV file
                with open(tmp_path, 'rb') as f:
                    wav_data = f.read()
                
                # Verify we got actual audio data (not just header)
                if len(wav_data) <= 44:
                    raise ValueError(f"Generated audio is empty (only {len(wav_data)} bytes)")
                
                logger.debug(f"Piper synthesized {len(wav_data)} bytes for: {text[:50]}...")
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {tmp_path}: {e}")

            # Convert to requested format
            if output_format == "mp3":
                return self._convert_to_mp3(wav_data)
            else:
                return wav_data
                
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            raise
    
    def _synthesize_qwen3(
        self,
        text: str,
        voice_id: str,
        speed: float,
        output_format: str,
        language: str
    ) -> bytes:
        """Synthesize using Qwen3-TTS."""
        if not self.qwen3_manager:
            raise RuntimeError("Qwen3-TTS is not enabled or not available")
        
        try:
            # Synthesize with Qwen3
            audio_array, sample_rate = self.qwen3_manager.synthesize(
                text=text,
                voice_id=voice_id,
                speed=speed,
                language=language,
            )
            
            # Convert to WAV bytes
            import soundfile as sf
            import io
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='wav')
            wav_data = buffer.getvalue()
            
            logger.debug(f"Qwen3 synthesized {len(wav_data)} bytes for: {text[:50]}...")
            
            # Convert to requested format
            if output_format == "mp3":
                return self._convert_to_mp3(wav_data)
            else:
                return wav_data
            
        except Exception as e:
            logger.error(f"Qwen3 synthesis failed: {e}")
            raise
    
    def _convert_to_mp3(self, wav_data: bytes) -> bytes:
        """
        Convert WAV to MP3.
        
        Args:
            wav_data: WAV audio bytes
            
        Returns:
            MP3 audio bytes
        """
        try:
            from pydub import AudioSegment
            
            # Load WAV from bytes
            wav_buffer = io.BytesIO(wav_data)
            audio = AudioSegment.from_wav(wav_buffer)
            
            # Export as MP3
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3")
            
            return mp3_buffer.getvalue()
            
        except ImportError:
            logger.warning("pydub not installed, returning WAV instead of MP3")
            return wav_data
        except Exception as e:
            logger.error(f"MP3 conversion failed: {e}")
            return wav_data
    
    def unload(self) -> None:
        """Unload current voice model to free memory."""
        if self.piper_voice is not None:
            self.piper_voice = None
            logger.debug(f"✓ Unloaded Piper voice: {self.current_piper_voice}")
            self.current_piper_voice = None
    
    def unload_qwen3(self):
        """Unload Qwen3 model to free GPU memory."""
        if self.qwen3_manager:
            self.qwen3_manager.unload_model()
    
    def unload_all(self):
        """Unload all TTS engines to free memory."""
        self.unload()  # Unload Piper
        self.unload_qwen3()  # Unload Qwen3
    
    def get_current_voice_info(self) -> Optional[VoiceInfo]:
        """Get information about currently loaded voice."""
        # Check Piper first (more common)
        if self.current_piper_voice:
            return self.voice_catalog.get(self.current_piper_voice)
        # Check Qwen3 cloned voices
        if self.qwen3_manager:
            cloned_voices = self.qwen3_manager.list_cloned_voices()
            for voice_info in cloned_voices:
                if voice_info["voice_id"] in self.voice_catalog:
                    return self.voice_catalog[voice_info["voice_id"]]
        return None
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return 'CUDAExecutionProvider' in providers
        except Exception:
            return False
