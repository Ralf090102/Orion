"""Qwen3-TTS Manager - Zero-shot voice cloning engine.

This module provides voice cloning capabilities using Qwen3-TTS models.
Features:
- Lazy model loading (GPU memory optimization)
- Voice embedding extraction from audio samples (3-15s)
- High-quality speech synthesis with cloned voices
- GPU memory management with auto-unload
- Performance monitoring (real-time factor tracking)

Official Docs: https://github.com/QwenLM/Qwen3-TTS
"""

import re
import torch
import numpy as np
import soundfile as sf
import time
import logging
from pathlib import Path
from typing import Generator, Iterator, Optional, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    """
    voice_id: str
    embedding: np.ndarray  # Audio data for Qwen3-TTS Base model
    sample_rate: int
    duration: float
    created_at: float
    ref_text: Optional[str] = None  # Reference text for ICL mode


class Qwen3Manager:
    """Manage Qwen3-TTS for high-quality zero-shot voice cloning.
    
    This manager handles:
    - Model loading/unloading (lazy initialization)
    - Voice embedding extraction from audio samples
    - Speech synthesis with cloned voices
    - GPU memory management
    - Multi-language support (en, zh, ja, ko, etc.)
    
    Example:
        >>> manager = Qwen3Manager(config)
        >>> # Extract voice from audio sample
        >>> embedding = manager.extract_voice_embedding("my_voice", "sample.wav")
        >>> # Synthesize with cloned voice
        >>> audio, sr = manager.synthesize("Hello world!", voice_id="my_voice")
        >>> sf.write("output.wav", audio, sr)
    """
    
    def __init__(self, config):
        """Initialize Qwen3Manager.
        
        Args:
            config: OrionConfig instance with qwen3 settings
        """
        self.config = config
        self.qwen3_config = config.qwen3
        
        # Model state
        self.model = None
        self.device = self._get_device()
        self.model_loaded_at: Optional[float] = None
        self.last_used_at: Optional[float] = None
        
        # Voice cache
        self.voice_embeddings: Dict[str, VoiceEmbedding] = {}
        
        # Storage directory for cloned voice audio samples
        self.voices_dir = Path(__file__).parent.parent.parent.parent / "data" / "tts" / "cloned_voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloned voices directory: {self.voices_dir}")
        
        # Stats
        self.synthesis_count = 0
        self.total_synthesis_time = 0.0
        
        logger.info(f"Qwen3Manager initialized (device: {self.device})")
        logger.info(f"Auto-unload: {self.qwen3_config.auto_unload} "
                   f"(timeout: {self.qwen3_config.unload_timeout_seconds}s)")
        
        # Scan disk for existing cloned voices so they survive restarts
        loaded = self._scan_cloned_voices()
        if loaded:
            logger.info(f"✓ Auto-loaded {loaded} cloned voice(s) from disk")
    
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
    
    def load_model(self) -> None:
        """Load Qwen3-TTS model into memory.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded")
            return
        
        logger.info(f"Loading Qwen3-TTS model: {self.qwen3_config.model_name}")
        start_time = time.time()
        
        try:
            # Import Qwen3-TTS (only when needed)
            from qwen_tts import Qwen3TTSModel
            
            # Determine attention implementation
            attn_impl = "flash_attention_2" if self._has_flash_attn() else "eager"
            if attn_impl == "eager":
                logger.warning("FlashAttention not available. Using eager attention (slower).")
                logger.info("Install with: pip install -U flash-attn --no-build-isolation")
            
            # Determine dtype
            dtype = torch.float16 if self.qwen3_config.model_precision == "float16" else torch.float32
            
            # Load model with device_map
            # Note: Qwen3TTSModel is a wrapper, not a PyTorch model, so no .to() method
            self.model = Qwen3TTSModel.from_pretrained(
                self.qwen3_config.model_name,
                device_map=self.device if self.device != "cpu" else None,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            
            logger.info(f"Model loaded on device: {self.model.device}")
            
            self.model_loaded_at = time.time()
            self.last_used_at = time.time()
            
            load_time = time.time() - start_time
            logger.info(f"✓ Qwen3 model loaded in {load_time:.2f}s")
            
        except ImportError as e:
            logger.error(f"Failed to import qwen_tts: {e}")
            logger.error("Install with: pip install qwen-tts")
            raise RuntimeError(f"Qwen3-TTS not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            raise RuntimeError(f"Qwen3 model loading failed: {e}")
    
    def unload_model(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is None:
            logger.debug("Model already unloaded")
            return
        
        logger.info("Unloading Qwen3 model...")
        
        try:
            # Delete model reference
            del self.model
            self.model = None
            self.model_loaded_at = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✓ Qwen3 model unloaded")
            
        except Exception as e:
            logger.error(f"Error during model unload: {e}")
    
    def check_auto_unload(self) -> None:
        """Check if model should be auto-unloaded due to inactivity."""
        if not self.qwen3_config.auto_unload:
            return
        
        if self.model is None or self.last_used_at is None:
            return
        
        idle_time = time.time() - self.last_used_at
        if idle_time > self.qwen3_config.unload_timeout_seconds:
            logger.info(f"Auto-unloading model after {idle_time:.0f}s idle")
            self.unload_model()
    
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
        embedding: 'VoiceEmbedding',
        speed: float,
        language: str,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize one text chunk using a pre-looked-up embedding."""
        ref_audio_path = str(embedding.embedding[0])
        ref_text = embedding.ref_text

        with torch.no_grad():
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only_mode=(ref_text is None),
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

        if self.model is None:
            self.load_model()

        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=200)
        if not chunks:
            raise ValueError("Text is empty after preprocessing")

        logger.info(f"Synthesizing {len(chunks)} chunk(s), voice={voice_id}, lang={language}")
        start_time = time.time()

        audio_parts: list = []
        sample_rate = 24000
        silence: Optional[np.ndarray] = None

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            audio, sr = self._synthesize_single(chunk, embedding, speed, language)
            sample_rate = sr
            if silence is None:
                silence = np.zeros(int(0.08 * sr), dtype=np.float32)  # 80 ms gap
            audio_parts.append(audio)
            if i < len(chunks) - 1:
                audio_parts.append(silence)
            logger.debug(f"  chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")

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

        if self.model is None:
            self.load_model()

        text = self._preprocess_text_for_tts(text)
        chunks = self._split_into_sentences(text, max_chars=200)
        logger.info(f"Streaming {len(chunks)} chunk(s), voice={voice_id}, lang={language}")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                audio, sr = self._synthesize_single(chunk, embedding, speed, language)
                self.last_used_at = time.time()
                logger.debug(f"  yielding chunk {i+1}/{len(chunks)}")
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
            "model_loaded": self.model is not None,
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
