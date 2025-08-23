"""
Smart Caching System for RAG Pipeline

This module implements intelligent caching to dramatically improve performance:

1. **Query Cache**: Cache query results to avoid recomputing similar queries
2. **Embedding Cache**: Cache document embeddings to avoid re-embedding
3. **Vectorstore Cache**: Cache vectorstore operations
4. **LRU Eviction**: Automatically remove old cache entries when memory is full

Key Concepts:
- Cache Hit: When we find cached data (fast!)
- Cache Miss: When we need to compute new data (slower)
- TTL (Time To Live): How long cache entries stay valid
- LRU (Least Recently Used): Eviction strategy for full caches
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import wraps
import sqlite3

from core.utils.orion_utils import log_info, log_debug, log_warning


@dataclass
class CacheEntry:
    """
    Represents a single cache entry with metadata.

    Attributes:
        key: Unique identifier for the cached item
        data: The actual cached data
        timestamp: When this entry was created
        access_count: How many times this entry has been accessed
        last_accessed: When this entry was last used
        ttl: Time to live in seconds (None = never expires)
    """

    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0
    ttl: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class SmartCache:
    """
    Intelligent caching system with multiple strategies and automatic cleanup.

    Features:
    - In-memory caching for speed
    - Persistent disk cache for embeddings
    - LRU eviction when memory is full
    - TTL-based expiration
    - Cache hit/miss statistics
    """

    def __init__(
        self,
        max_memory_entries: int = 1000,
        default_ttl: Optional[int] = 3600,  # 1 hour
        cache_dir: str = "data/cache",
    ):
        """
        Initialize the smart cache system.

        Args:
            max_memory_entries: Maximum number of entries in memory cache
            default_ttl: Default time-to-live in seconds (None = never expire)
            cache_dir: Directory for persistent cache storage
        """
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast access
        self._memory_cache: Dict[str, CacheEntry] = {}

        # Statistics for performance monitoring
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}

        # Initialize persistent cache database
        self._init_persistent_cache()

    def _init_persistent_cache(self):
        """Initialize SQLite database for persistent caching."""
        db_path = self.cache_dir / "persistent_cache.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    ttl INTEGER
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """
            )

            conn.commit()

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from function arguments.

        This creates a consistent hash that uniquely identifies
        a particular function call with specific parameters.
        """
        # Convert all arguments to a consistent string representation
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),  # Sort for consistency
        }

        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[
            :16
        ]  # 16 chars is enough

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached data or None if not found/expired
        """
        # First check memory cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]

            if entry.is_expired():
                # Remove expired entry
                del self._memory_cache[key]
                log_debug(f"Cache entry expired: {key}")
                return None

            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.stats["hits"] += 1

            log_debug(f"Cache hit (memory): {key}")
            return entry.data

        # Check persistent cache
        data = self._get_from_persistent_cache(key)
        if data is not None:
            # Move to memory cache for faster future access
            self.put(key, data, ttl=self.default_ttl)
            self.stats["hits"] += 1
            log_debug(f"Cache hit (disk): {key}")
            return data

        # Cache miss
        self.stats["misses"] += 1
        log_debug(f"Cache miss: {key}")
        return None

    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store data in cache.

        Args:
            key: Unique cache key
            data: Data to cache
            ttl: Time to live in seconds (None = use default)
        """
        if ttl is None:
            ttl = self.default_ttl

        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            access_count=1,
            last_accessed=time.time(),
            ttl=ttl,
        )

        # Check if we need to evict entries (LRU)
        if len(self._memory_cache) >= self.max_memory_entries:
            self._evict_lru()

        # Store in memory cache
        self._memory_cache[key] = entry

        # Also store in persistent cache for embeddings and expensive operations
        self._put_to_persistent_cache(key, data, entry)

        self.stats["size"] = len(self._memory_cache)
        log_debug(f"Cache stored: {key}")

    def _evict_lru(self):
        """
        Remove least recently used entry from memory cache.

        This implements the LRU (Least Recently Used) eviction strategy.
        """
        if not self._memory_cache:
            return

        # Find entry with oldest last_accessed time
        lru_key = min(
            self._memory_cache.keys(), key=lambda k: self._memory_cache[k].last_accessed
        )

        del self._memory_cache[lru_key]
        self.stats["evictions"] += 1

        log_debug(f"Cache eviction (LRU): {lru_key}")

    def _get_from_persistent_cache(self, key: str) -> Optional[Any]:
        """Get data from persistent SQLite cache."""
        try:
            db_path = self.cache_dir / "persistent_cache.db"

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT data, timestamp, ttl FROM cache_entries WHERE key = ?",
                    (key,),
                )

                row = cursor.fetchone()
                if row is None:
                    return None

                data_blob, timestamp, ttl = row

                # Check expiration
                if ttl is not None and time.time() - timestamp > ttl:
                    # Remove expired entry
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return None

                # Update access statistics
                conn.execute(
                    "UPDATE cache_entries SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                    (time.time(), key),
                )
                conn.commit()

                # Deserialize data
                return pickle.loads(data_blob)

        except Exception as e:
            log_warning(f"Persistent cache read error: {e}")
            return None

    def _put_to_persistent_cache(self, key: str, data: Any, entry: CacheEntry):
        """Store data in persistent SQLite cache."""
        try:
            db_path = self.cache_dir / "persistent_cache.db"

            # Serialize data
            data_blob = pickle.dumps(data)

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries 
                    (key, data, timestamp, access_count, last_accessed, ttl)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        data_blob,
                        entry.timestamp,
                        entry.access_count,
                        entry.last_accessed,
                        entry.ttl,
                    ),
                )
                conn.commit()

        except Exception as e:
            log_warning(f"Persistent cache write error: {e}")

    def clear(self):
        """Clear all cache entries."""
        self._memory_cache.clear()

        # Clear persistent cache
        try:
            db_path = self.cache_dir / "persistent_cache.db"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
        except Exception as e:
            log_warning(f"Error clearing persistent cache: {e}")

        # Reset statistics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "size": 0}

        log_info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {**self.stats, "hit_rate": hit_rate, "total_requests": total_requests}

    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        # Clean memory cache
        for key, entry in self._memory_cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self._memory_cache[key]

        # Clean persistent cache
        try:
            db_path = self.cache_dir / "persistent_cache.db"
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE ttl IS NOT NULL AND timestamp + ttl < ?",
                    (current_time,),
                )
                deleted_count = cursor.rowcount
                conn.commit()

                if deleted_count > 0:
                    log_info(f"Cleaned up {deleted_count} expired cache entries")

        except Exception as e:
            log_warning(f"Error cleaning persistent cache: {e}")


# Global cache instance
_global_cache = SmartCache()


def cached(ttl: Optional[int] = None, cache_instance: Optional[SmartCache] = None):
    """
    Decorator to automatically cache function results.

    This is a decorator that can be applied to any function to automatically
    cache its results based on the input parameters.

    Usage:
        @cached(ttl=3600)  # Cache for 1 hour
        def expensive_function(param1, param2):
            # ... expensive computation ...
            return result

    Args:
        ttl: Time to live for cache entries (None = use cache default)
        cache_instance: Specific cache instance to use (None = use global cache)
    """

    def decorator(func):
        cache = cache_instance or _global_cache

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key based on function name and arguments
            cache_key = f"{func.__name__}_{cache._generate_cache_key(*args, **kwargs)}"

            # Try to get from cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Cache miss - compute result
            result = func(*args, **kwargs)

            # Store result in cache
            cache.put(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


# Convenience functions for common caching patterns
def cache_query_result(query: str, result: Any, ttl: int = 3600):
    """Cache a query result."""
    key = f"query_{hashlib.sha256(query.encode()).hexdigest()[:16]}"
    _global_cache.put(key, result, ttl=ttl)


def get_cached_query_result(query: str) -> Optional[Any]:
    """Get cached query result."""
    key = f"query_{hashlib.sha256(query.encode()).hexdigest()[:16]}"
    return _global_cache.get(key)


def cache_embeddings(text: str, embeddings: List[float], ttl: int = 86400):  # 24 hours
    """Cache text embeddings (expensive to compute)."""
    key = f"embedding_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    _global_cache.put(key, embeddings, ttl=ttl)


def get_cached_embeddings(text: str) -> Optional[List[float]]:
    """Get cached embeddings."""
    key = f"embedding_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    return _global_cache.get(key)


def get_global_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()


def clear_global_cache():
    """Clear global cache."""
    _global_cache.clear()
