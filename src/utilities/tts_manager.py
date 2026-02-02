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
import tempfile
import os

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
        
        # Local model directories (data/tts/)
        self.models_dir = Path(__file__).parent.parent.parent / "data" / "tts"
        self.onnx_dir = self.models_dir / "onnx"
        self.json_dir = self.models_dir / "json"
        
        # Ensure directories exist
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Load voice catalog from available local models
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
                use_cuda=self.config.use_gpu
            )
            self.current_voice = voice_id
            logger.info(f"✓ Loaded voice: {voice_id}")
        except Exception as e:
            logger.error(f"Failed to load voice '{voice_id}': {e}")
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
                
                logger.debug(f"Generated {len(wav_data)} bytes of audio for: {text[:50]}...")
                    
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
