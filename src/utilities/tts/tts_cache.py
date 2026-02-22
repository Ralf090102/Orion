"""TTS audio caching with LRU eviction and disk persistence.

This module provides intelligent caching for TTS audio to avoid re-synthesizing
identical text. Critical for Qwen3-TTS where synthesis can take 5-60 seconds.

Features:
- Hash-based lookup (text + voice + speed + engine)
- LRU (Least Recently Used) eviction when cache is full
- Disk persistence (survives restarts)
- Size limits (configurable max entries and disk space)
- Cache hit rate tracking

Expected benefits:
- 20-30% cache hit rate for common phrases
- Instant playback for cached audio (saves 5-60s per hit)
- Persistent across application restarts
"""

import hashlib
import pickle
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CachedAudio:
    """Cached TTS audio entry.
    
    Attributes:
        cache_key: Unique hash identifier
        audio_bytes: Raw audio data
        format: Audio format ("wav", "mp3")
        voice_id: Voice identifier used
        engine: TTS engine ("piper", "qwen3")
        text_hash: Hash of synthesized text (for debugging)
        created_at: Unix timestamp of creation
        last_accessed: Unix timestamp of last access
        access_count: Number of times accessed
        size_bytes: Size of audio data in bytes
    """
    cache_key: str
    audio_bytes: bytes
    format: str
    voice_id: str
    engine: str
    text_hash: str
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding audio_bytes for metadata)."""
        data = asdict(self)
        data.pop('audio_bytes', None)  # Don't include in metadata
        return data


class TTSCache:
    """LRU cache for TTS audio with disk persistence.
    
    Manages audio caching with automatic eviction of least recently used entries
    when cache limits are exceeded.
    
    Example:
        >>> cache = TTSCache(cache_dir="./data/tts/cache", max_size_mb=500)
        >>> 
        >>> # Check cache before synthesis
        >>> audio = cache.get("Hello world", "en_US-lessac", 1.0, "piper")
        >>> if audio is None:
        >>>     # Cache miss - synthesize
        >>>     audio = synthesize("Hello world", ...)
        >>>     cache.put("Hello world", "en_US-lessac", 1.0, "piper", audio, "wav")
        >>> 
        >>> # Get stats
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/tts/cache",
        max_entries: int = 1000,
        max_size_mb: int = 500,
    ):
        """Initialize TTS cache.
        
        Args:
            cache_dir: Directory for cache persistence
            max_entries: Maximum number of cached audio entries
            max_size_mb: Maximum total cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # In-memory cache (key -> CachedAudio)
        self.cache: Dict[str, CachedAudio] = {}
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache from disk
        self._load_from_disk()
        
        logger.info(
            f"TTSCache initialized: {len(self.cache)} entries, "
            f"{self.total_size_bytes / 1024 / 1024:.1f}MB, "
            f"max {max_entries} entries / {max_size_mb}MB"
        )
    
    def _compute_cache_key(
        self,
        text: str,
        voice_id: str,
        speed: float,
        engine: str,
    ) -> str:
        """Compute unique cache key for audio.
        
        Args:
            text: Synthesized text
            voice_id: Voice identifier
            speed: Speech speed multiplier
            engine: TTS engine name
        
        Returns:
            SHA256 hash as cache key
        """
        # Create unique identifier from all parameters
        key_string = f"{text}|{voice_id}|{speed}|{engine}"
        
        # Hash to fixed-length key
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()
        
        return cache_key
    
    def get(
        self,
        text: str,
        voice_id: str,
        speed: float,
        engine: str,
    ) -> Optional[bytes]:
        """Get cached audio if available.
        
        Args:
            text: Synthesized text
            voice_id: Voice identifier
            speed: Speech speed multiplier
            engine: TTS engine name
        
        Returns:
            Audio bytes if cached, None if cache miss
        """
        cache_key = self._compute_cache_key(text, voice_id, speed, engine)
        
        if cache_key in self.cache:
            # Cache hit
            entry = self.cache[cache_key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.hits += 1
            
            logger.debug(
                f"Cache HIT: {text[:50]}... (engine={engine}, "
                f"accessed {entry.access_count}x)"
            )
            
            return entry.audio_bytes
        else:
            # Cache miss
            self.misses += 1
            logger.debug(f"Cache MISS: {text[:50]}... (engine={engine})")
            return None
    
    def put(
        self,
        text: str,
        voice_id: str,
        speed: float,
        engine: str,
        audio_bytes: bytes,
        format: str,
    ) -> None:
        """Add audio to cache.
        
        Args:
            text: Synthesized text
            voice_id: Voice identifier
            speed: Speech speed multiplier
            engine: TTS engine name
            audio_bytes: Audio data
            format: Audio format ("wav", "mp3")
        """
        cache_key = self._compute_cache_key(text, voice_id, speed, engine)
        
        # Check if already cached (update timestamp)
        if cache_key in self.cache:
            self.cache[cache_key].last_accessed = time.time()
            logger.debug(f"Cache UPDATE: {text[:50]}...")
            return
        
        # Create cache entry
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        entry = CachedAudio(
            cache_key=cache_key,
            audio_bytes=audio_bytes,
            format=format,
            voice_id=voice_id,
            engine=engine,
            text_hash=text_hash,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            size_bytes=len(audio_bytes),
        )
        
        # Check if we need to evict entries
        while (
            len(self.cache) >= self.max_entries
            or self.total_size_bytes + entry.size_bytes > self.max_size_bytes
        ):
            self._evict_lru()
        
        # Add to cache
        self.cache[cache_key] = entry
        self.total_size_bytes += entry.size_bytes
        
        # Persist to disk
        self._save_entry_to_disk(entry)
        
        logger.debug(
            f"Cache PUT: {text[:50]}... ({entry.size_bytes / 1024:.1f} KB, "
            f"total: {self.total_size_bytes / 1024 / 1024:.1f} MB)"
        )
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        entry = self.cache[lru_key]
        
        # Remove from cache
        del self.cache[lru_key]
        self.total_size_bytes -= entry.size_bytes
        self.evictions += 1
        
        # Delete from disk
        self._delete_entry_from_disk(lru_key)
        
        logger.debug(
            f"Evicted LRU: {lru_key[:16]}... ({entry.size_bytes / 1024:.1f} KB, "
            f"engine={entry.engine})"
        )
    
    def _save_entry_to_disk(self, entry: CachedAudio) -> None:
        """Save cache entry to disk.
        
        Args:
            entry: CachedAudio entry to persist
        """
        try:
            # Save audio file
            audio_path = self.cache_dir / f"{entry.cache_key}.{entry.format}"
            with open(audio_path, 'wb') as f:
                f.write(entry.audio_bytes)
            
            # Save metadata (without audio_bytes to save space)
            meta_path = self.cache_dir / f"{entry.cache_key}.meta"
            metadata = entry.to_dict()
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception as e:
            logger.error(f"Failed to save cache entry to disk: {e}")
    
    def _delete_entry_from_disk(self, cache_key: str) -> None:
        """Delete cache entry from disk.
        
        Args:
            cache_key: Cache key identifier
        """
        try:
            # Delete audio files (try both wav and mp3)
            for ext in ['wav', 'mp3']:
                audio_path = self.cache_dir / f"{cache_key}.{ext}"
                if audio_path.exists():
                    audio_path.unlink()
            
            # Delete metadata
            meta_path = self.cache_dir / f"{cache_key}.meta"
            if meta_path.exists():
                meta_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to delete cache entry from disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk on startup."""
        if not self.cache_dir.exists():
            return
        
        loaded = 0
        skipped = 0
        
        # Find all metadata files
        meta_files = list(self.cache_dir.glob("*.meta"))
        
        for meta_path in meta_files:
            try:
                # Load metadata
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                cache_key = metadata['cache_key']
                format = metadata['format']
                
                # Load audio file
                audio_path = self.cache_dir / f"{cache_key}.{format}"
                if not audio_path.exists():
                    logger.debug(f"Audio file missing for {cache_key}, skipping")
                    skipped += 1
                    continue
                
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Recreate cache entry
                entry = CachedAudio(
                    audio_bytes=audio_bytes,
                    **metadata
                )
                
                self.cache[cache_key] = entry
                self.total_size_bytes += entry.size_bytes
                loaded += 1
                
            except Exception as e:
                logger.debug(f"Failed to load cache entry {meta_path.name}: {e}")
                skipped += 1
        
        if loaded > 0:
            logger.info(
                f"✓ Loaded {loaded} cached entries from disk "
                f"({self.total_size_bytes / 1024 / 1024:.1f} MB)"
            )
        if skipped > 0:
            logger.debug(f"Skipped {skipped} invalid cache entries")
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate.
        
        Returns:
            Hit rate as float (0.0 to 1.0)
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "size_mb": self.total_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_size_mb,
            "utilization": len(self.cache) / self.max_entries if self.max_entries > 0 else 0,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
            "evictions": self.evictions,
        }
    
    def clear(self) -> None:
        """Clear all cache (memory and disk)."""
        logger.info("Clearing TTS cache...")
        
        # Delete all cache files
        for cache_key in list(self.cache.keys()):
            self._delete_entry_from_disk(cache_key)
        
        # Clear memory
        self.cache.clear()
        self.total_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info("✓ Cache cleared")
    
    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cached entry.
        
        Args:
            cache_key: Cache key identifier
        
        Returns:
            Metadata dictionary or None if not found
        """
        if cache_key in self.cache:
            return self.cache[cache_key].to_dict()
        return None
    
    def prune_old_entries(self, max_age_seconds: int = 2592000) -> int:
        """Remove entries older than max_age.
        
        Args:
            max_age_seconds: Maximum age in seconds (default 30 days)
        
        Returns:
            Number of entries pruned
        """
        now = time.time()
        pruned = 0
        
        # Find old entries
        to_remove = [
            key for key, entry in self.cache.items()
            if (now - entry.last_accessed) > max_age_seconds
        ]
        
        # Remove them
        for key in to_remove:
            entry = self.cache[key]
            self.total_size_bytes -= entry.size_bytes
            self._delete_entry_from_disk(key)
            del self.cache[key]
            pruned += 1
        
        if pruned > 0:
            logger.info(f"Pruned {pruned} old cache entries (>{max_age_seconds}s)")
        
        return pruned
