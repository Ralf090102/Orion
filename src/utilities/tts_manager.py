"""
TTS Manager for Orion RAG Assistant

Manages Piper TTS models, voice switching, and audio synthesis.
Supports lazy loading, voice catalog management, and GPU/CPU device selection.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import io

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
    is_downloaded: bool = False


class TTSManager:
    """
    Manager for Piper TTS functionality.
    
    Handles voice loading, model management, and speech synthesis.
    """
    
    def __init__(self, config):
        """
        Initialize TTS Manager.
        
        Args:
            config: OrionConfig instance with TTS settings
        """
        self.config = config.tts
        self.voice_catalog: Dict[str, VoiceInfo] = {}
        self.current_voice: Optional[str] = None
        self.piper_voice = None
        
        # Lazy imports to avoid dependency issues
        self._piper_module = None
        self._wave_module = None
        
        # Ensure cache directory exists
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load voice catalog
        self._load_voice_catalog()
        
        logger.info(f"✓ TTS Manager initialized (engine: {self.config.engine})")
    
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
    
    def _load_voice_catalog(self):
        """Load voice catalog from JSON or create default."""
        catalog_path = Path(__file__).parent.parent.parent / "data" / "tts" / "voices.json"
        
        if catalog_path.exists():
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    catalog_data = json.load(f)
                    
                for voice_id, voice_data in catalog_data.items():
                    self.voice_catalog[voice_id] = VoiceInfo(
                        voice_id=voice_id,
                        name=voice_data.get("name", voice_id),
                        language=voice_data.get("language", "unknown"),
                        gender=voice_data.get("gender", "neutral"),
                        quality=voice_data.get("quality", "medium"),
                        description=voice_data.get("description", ""),
                        model_size=voice_data.get("model_size", "unknown"),
                        sample_rate=voice_data.get("sample_rate", 22050),
                        is_downloaded=self._is_voice_downloaded(voice_id)
                    )
                
                logger.info(f"✓ Loaded {len(self.voice_catalog)} voices from catalog")
            except Exception as e:
                logger.warning(f"Failed to load voice catalog: {e}")
                self._create_default_catalog()
        else:
            self._create_default_catalog()
    
    def _create_default_catalog(self):
        """Create default voice catalog with common Piper voices."""
        default_voices = {
            "en_US-lessac-medium": {
                "name": "Lessac",
                "language": "en_US",
                "gender": "male",
                "quality": "medium",
                "description": "American English, neutral tone",
                "model_size": "8.5MB",
                "sample_rate": 22050
            },
            "en_US-amy-medium": {
                "name": "Amy",
                "language": "en_US",
                "gender": "female",
                "quality": "medium",
                "description": "American English, warm tone",
                "model_size": "7.2MB",
                "sample_rate": 22050
            },
            "en_GB-alan-medium": {
                "name": "Alan",
                "language": "en_GB",
                "gender": "male",
                "quality": "medium",
                "description": "British English",
                "model_size": "8.1MB",
                "sample_rate": 22050
            }
        }
        
        for voice_id, voice_data in default_voices.items():
            self.voice_catalog[voice_id] = VoiceInfo(
                voice_id=voice_id,
                name=voice_data["name"],
                language=voice_data["language"],
                gender=voice_data["gender"],
                quality=voice_data["quality"],
                description=voice_data["description"],
                model_size=voice_data["model_size"],
                sample_rate=voice_data["sample_rate"],
                is_downloaded=self._is_voice_downloaded(voice_id)
            )
        
        logger.info(f"✓ Created default voice catalog with {len(default_voices)} voices")
    
    def _is_voice_downloaded(self, voice_id: str) -> bool:
        """Check if voice model is cached locally."""
        voice_path = self.config.model_cache_dir / f"{voice_id}.onnx"
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
        self._import_dependencies()
        
        if voice_id not in self.voice_catalog:
            raise ValueError(f"Voice '{voice_id}' not found in catalog")
        
        # Check if already loaded
        if self.current_voice == voice_id and self.piper_voice is not None:
            logger.debug(f"Voice '{voice_id}' already loaded")
            return
        
        # Unload current voice
        if self.piper_voice is not None:
            self.unload()
        
        # Download voice if needed
        if self.config.auto_download_voices and not self._is_voice_downloaded(voice_id):
            logger.info(f"Downloading voice: {voice_id}")
            self._download_voice(voice_id)
        
        # Load voice
        voice_path = self.config.model_cache_dir / f"{voice_id}.onnx"
        config_path = self.config.model_cache_dir / f"{voice_id}.onnx.json"
        
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {voice_path}. "
                f"Set auto_download_voices=True to download automatically."
            )
        
        try:
            self.piper_voice = self._piper_module.load(
                str(voice_path),
                config_path=str(config_path) if config_path.exists() else None,
                use_cuda=self.config.use_gpu
            )
            self.current_voice = voice_id
            logger.info(f"✓ Loaded voice: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to load voice '{voice_id}': {e}")
            raise
    
    def _download_voice(self, voice_id: str):
        """
        Download voice model from Piper repository.
        
        Args:
            voice_id: Voice identifier
        """
        # Note: Piper voices are hosted on HuggingFace
        # This is a placeholder - actual implementation would use huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            
            repo_id = "rhasspy/piper-voices"
            
            # Download .onnx model
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=f"{voice_id.replace('_', '/')}/{voice_id}.onnx",
                cache_dir=str(self.config.model_cache_dir)
            )
            
            # Download .onnx.json config
            config_file = hf_hub_download(
                repo_id=repo_id,
                filename=f"{voice_id.replace('_', '/')}/{voice_id}.onnx.json",
                cache_dir=str(self.config.model_cache_dir)
            )
            
            logger.info(f"✓ Downloaded voice: {voice_id}")
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to download voice '{voice_id}': {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: Optional[float] = None,
        output_format: str = "wav"
    ) -> bytes:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            voice_id: Voice to use (defaults to configured default)
            speed: Speech speed multiplier (1.0 = normal)
            output_format: Audio format ("wav" or "mp3")
            
        Returns:
            Audio data as bytes
        """
        # Use default voice if not specified
        if voice_id is None:
            voice_id = self.config.default_voice
        
        # Use default speed if not specified
        if speed is None:
            speed = self.config.default_speed
        
        # Load voice if needed
        if self.current_voice != voice_id:
            self.load_voice(voice_id)
        
        # Synthesize speech
        try:
            # Create in-memory WAV file
            wav_buffer = io.BytesIO()
            
            with self._wave_module.open(wav_buffer, 'wb') as wav_file:
                # Get voice info for sample rate
                voice_info = self.voice_catalog[voice_id]
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(voice_info.sample_rate)
                
                # Synthesize
                self.piper_voice.synthesize(
                    text,
                    wav_file,
                    length_scale=1.0 / speed  # Piper uses inverse speed
                )
            
            # Get WAV data
            wav_data = wav_buffer.getvalue()
            
            # Convert to requested format
            if output_format == "mp3":
                return self._convert_to_mp3(wav_data)
            else:
                return wav_data
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
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
            logger.debug(f"✓ Unloaded voice: {self.current_voice}")
            self.current_voice = None
    
    def get_current_voice_info(self) -> Optional[VoiceInfo]:
        """Get information about currently loaded voice."""
        if self.current_voice:
            return self.voice_catalog.get(self.current_voice)
        return None
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            return 'CUDAExecutionProvider' in providers
        except Exception:
            return False
