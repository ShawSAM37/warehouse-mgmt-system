"""
Advanced Caching System for WMS Dashboard
=========================================

Provides intelligent caching with LRU eviction, TTL support, memory management,
and performance monitoring for optimal dashboard performance.
"""

import time
import threading
import weakref
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import sys

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[int]
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheManager:
    """
    Advanced cache manager with LRU eviction, TTL support, and memory management.
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) support per entry
    - Memory usage tracking and limits
    - Performance monitoring and statistics
    - Thread-safe operations
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 50):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in megabytes
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0,
            'total_memory_bytes': 0
        }
        
        # Cleanup thread
        self._cleanup_interval = 60  # seconds
        self._cleanup_thread = None
        self._start_cleanup_thread()
        
        logger.info(f"CacheManager initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                self._stats['expired_removals'] += 1
                return None
            
            # Update access info and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        with self._lock:
            current_time = time.time()
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check memory limit
            if self._stats['total_memory_bytes'] + size_bytes > self.max_memory_bytes:
                self._evict_by_memory(size_bytes)
            
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._stats['total_memory_bytes'] += size_bytes
            
            logger.debug(f"Cached key '{key}' with TTL {ttl}s, size {size_bytes} bytes")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats['total_memory_bytes'] = 0
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats['expired_removals'] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._stats['total_memory_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate_percent': round(hit_rate, 2),
                'total_hits': self._stats['hits'],
                'total_misses': self._stats['misses'],
                'total_evictions': self._stats['evictions'],
                'expired_removals': self._stats['expired_removals']
            }
    
    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update memory usage."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats['total_memory_bytes'] -= entry.size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            lru_key = next(iter(self._cache))
            self._remove_entry(lru_key)
            self._stats['evictions'] += 1
            logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def _evict_by_memory(self, required_bytes: int) -> None:
        """Evict entries until enough memory is available."""
        while (self._stats['total_memory_bytes'] + required_bytes > self.max_memory_bytes 
               and self._cache):
            self._evict_lru()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        try:
            return sys.getsizeof(obj)
        except (TypeError, OverflowError):
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (int, float, bool)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj) + 64
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in obj.items()) + 64
            else:
                return 64  # Default estimate
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self._cleanup_interval)
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Cache cleanup thread started")
    
    def __del__(self):
        """Cleanup when cache manager is destroyed."""
        if hasattr(self, '_cleanup_thread') and self._cleanup_thread:
            # Note: Thread will be cleaned up automatically as it's a daemon thread
            pass

# Decorator for caching function results
def cached(ttl: Optional[int] = None, cache_manager: Optional[CacheManager] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        cache_manager: Cache manager instance (uses default if None)
    """
    def decorator(func: Callable) -> Callable:
        # Use default cache manager if none provided
        nonlocal cache_manager
        if cache_manager is None:
            cache_manager = CacheManager()
        
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Global cache manager instance
default_cache = CacheManager()
