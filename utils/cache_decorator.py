"""
Caching Decorator with TTL Support

Provides DAO-level caching for CloudWatch service methods with:
- Time-to-live (TTL) support
- Page-aware cache keys
- Thread-safe implementation
- Memory-efficient LRU eviction
- Optional caching (can be disabled via environment variable or parameter)
"""

import functools
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)

# Global flag to enable/disable caching
# Can be controlled via environment variable: CFM_ENABLE_CACHE=false
_CACHE_ENABLED = os.getenv('CFM_ENABLE_CACHE', 'true').lower() in ('true', '1', 'yes')


class TTLCache:
    """Thread-safe cache with TTL support."""
    
    def __init__(self, ttl_seconds: int = 600, max_size: int = 1000):
        """
        Initialize TTL cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds (default: 600 = 10 minutes)
            max_size: Maximum cache entries (default: 1000)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - timestamp) > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if (current_time - timestamp) > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        if len(self.cache) >= self.max_size:
            # Remove oldest 10% of entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            num_to_remove = max(1, len(sorted_items) // 10)
            for key, _ in sorted_items[:num_to_remove]:
                del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if not self._is_expired(timestamp):
                    self.hits += 1
                    logger.debug(f"Cache HIT for key: {key[:50]}...")
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]
                    logger.debug(f"Cache EXPIRED for key: {key[:50]}...")
            
            self.misses += 1
            logger.debug(f"Cache MISS for key: {key[:50]}...")
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        with self.lock:
            # Evict expired entries periodically
            if len(self.cache) % 100 == 0:
                self._evict_expired()
            
            # Evict LRU if cache is full
            self._evict_lru()
            
            self.cache[key] = (value, time.time())
            logger.debug(f"Cache SET for key: {key[:50]}...")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'ttl_seconds': self.ttl_seconds
            }


# Global cache instance for CloudWatch DAO methods
_cloudwatch_cache = TTLCache(ttl_seconds=600, max_size=1000)


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Generate cache key from function name and arguments.
    
    Includes page parameter to ensure pagination is cached correctly.
    """
    # Extract key parameters
    key_parts = [func_name]
    
    # Add positional args (excluding self)
    if args and len(args) > 1:
        key_parts.extend(str(arg) for arg in args[1:])
    
    # Add important kwargs (including page)
    important_params = [
        'region', 'page', 'lookback_days', 'namespace_filter',
        'log_group_name_prefix', 'alarm_name_prefix', 'dashboard_name_prefix',
        'can_spend_for_estimate', 'can_spend_for_exact_usage_estimate'
    ]
    
    for param in important_params:
        if param in kwargs:
            key_parts.append(f"{param}={kwargs[param]}")
    
    # Create hash of key parts
    key_string = "|".join(str(part) for part in key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    return f"{func_name}:{key_hash}"


def dao_cache(ttl_seconds: int = 600, enabled: Optional[bool] = None):
    """
    Decorator for caching DAO method results with TTL.
    
    Caching can be disabled globally via CFM_ENABLE_CACHE environment variable
    or per-decorator via the enabled parameter.
    
    Args:
        ttl_seconds: Time-to-live in seconds (default: 600 = 10 minutes)
        enabled: Override global cache setting (None = use global, True/False = force)
    
    Usage:
        # Use global cache setting
        @dao_cache(ttl_seconds=600)
        async def get_log_groups(self, page: int = 1, **kwargs):
            pass
        
        # Force caching disabled for this method
        @dao_cache(ttl_seconds=600, enabled=False)
        async def get_real_time_data(self, **kwargs):
            pass
        
        # Force caching enabled for this method
        @dao_cache(ttl_seconds=600, enabled=True)
        async def get_expensive_data(self, **kwargs):
            pass
    
    Environment Variables:
        CFM_ENABLE_CACHE: Set to 'false', '0', or 'no' to disable caching globally
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if caching is enabled
            cache_enabled = enabled if enabled is not None else _CACHE_ENABLED
            
            if not cache_enabled:
                logger.debug(f"Cache disabled for {func.__name__}, calling function directly")
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = _cloudwatch_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Returning cached result for {func.__name__}")
                return cached_value
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache the result
            _cloudwatch_cache.set(cache_key, result)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if caching is enabled
            cache_enabled = enabled if enabled is not None else _CACHE_ENABLED
            
            if not cache_enabled:
                logger.debug(f"Cache disabled for {func.__name__}, calling function directly")
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = _cloudwatch_cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Returning cached result for {func.__name__}")
                return cached_value
            
            # Call original function
            result = func(*args, **kwargs)
            
            # Cache the result
            _cloudwatch_cache.set(cache_key, result)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    stats = _cloudwatch_cache.get_stats()
    stats['enabled'] = _CACHE_ENABLED
    return stats


def clear_cache():
    """Clear all cache entries."""
    _cloudwatch_cache.clear()


def is_cache_enabled() -> bool:
    """Check if caching is currently enabled."""
    return _CACHE_ENABLED


def enable_cache():
    """Enable caching globally (runtime override)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = True
    logger.info("Cache enabled globally")


def disable_cache():
    """Disable caching globally (runtime override)."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = False
    logger.info("Cache disabled globally")


def set_cache_enabled(enabled: bool):
    """
    Set cache enabled state.
    
    Args:
        enabled: True to enable caching, False to disable
    """
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled
    logger.info(f"Cache {'enabled' if enabled else 'disabled'} globally")


# Import asyncio at the end to avoid circular imports
import asyncio
