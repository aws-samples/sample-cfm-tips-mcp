"""
Intelligent Cache System for CFM Tips MCP Server

Provides intelligent caching for frequently accessed data including pricing, bucket metadata,
and analysis results with automatic expiration and memory management.
"""

import logging
import time
import threading
import hashlib
import json
import pickle
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_size(self):
        """Calculate approximate size of the cached value."""
        try:
            if isinstance(self.value, (str, bytes)):
                self.size_bytes = len(self.value)
            elif isinstance(self.value, (dict, list)):
                # Approximate size using JSON serialization
                self.size_bytes = len(json.dumps(self.value, default=str))
            else:
                # Use pickle for other types
                self.size_bytes = len(pickle.dumps(self.value))
        except Exception:
            # Fallback to a rough estimate
            self.size_bytes = 1024  # 1KB default

class IntelligentCache:
    """
    Intelligent caching system with advanced features:
    
    - LRU eviction with access frequency consideration
    - TTL-based expiration
    - Memory usage monitoring and limits
    - Cache warming and preloading
    - Performance metrics integration
    - Tag-based cache invalidation
    - Automatic cleanup and optimization
    """
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 max_entries: int = 10000,
                 default_ttl_seconds: int = 3600,
                 cleanup_interval_minutes: int = 15):
        """
        Initialize IntelligentCache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default TTL for cache entries
            cleanup_interval_minutes: Interval for automatic cleanup
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # Cache storage (OrderedDict for LRU behavior)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "total_size_bytes": 0,
            "cleanup_runs": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_active = True
        self._cleanup_thread = None
        
        # Cache warming functions
        self._warming_functions: Dict[str, Callable] = {}
        
        # Performance monitor integration
        self._performance_monitor = None
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"IntelligentCache initialized (max_size: {max_size_mb}MB, max_entries: {max_entries})")
    
    def set_performance_monitor(self, performance_monitor):
        """Set performance monitor for metrics integration."""
        self._performance_monitor = performance_monitor
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Cache cleanup thread started")
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while self._cleanup_active:
            try:
                time.sleep(self.cleanup_interval_minutes * 60)
                self._cleanup_expired_entries()
                self._optimize_cache()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _generate_key(self, key_parts: Union[str, List[Any]]) -> str:
        """Generate a consistent cache key from various inputs."""
        if isinstance(key_parts, str):
            return key_parts
        
        # Convert to string and hash for consistency
        key_str = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: Union[str, List[Any]], default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key (string or list of components)
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[cache_key]
                    self.stats["expirations"] += 1
                    self._update_total_size()
                    
                    # Record cache miss
                    self.stats["misses"] += 1
                    if self._performance_monitor:
                        self._performance_monitor.record_cache_miss("intelligent_cache")
                    
                    return default
                
                # Update access statistics
                entry.update_access()
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                
                # Record cache hit
                self.stats["hits"] += 1
                if self._performance_monitor:
                    self._performance_monitor.record_cache_hit("intelligent_cache")
                
                return entry.value
            
            # Cache miss
            self.stats["misses"] += 1
            if self._performance_monitor:
                self._performance_monitor.record_cache_miss("intelligent_cache")
            
            return default
    
    def put(self, key: Union[str, List[Any]], value: Any, 
            ttl_seconds: Optional[int] = None, 
            tags: Optional[Dict[str, str]] = None) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key (string or list of components)
            value: Value to cache
            ttl_seconds: Time to live in seconds (None for default)
            tags: Optional tags for the entry
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._generate_key(key)
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds or self.default_ttl_seconds,
                tags=tags or {}
            )
            
            # Calculate size
            entry.calculate_size()
            
            # Check if we need to make space
            if not self._make_space_for_entry(entry):
                logger.warning(f"Could not make space for cache entry: {cache_key}")
                return False
            
            # Add to cache
            self.cache[cache_key] = entry
            self.cache.move_to_end(cache_key)  # Mark as most recently used
            
            self._update_total_size()
            
            logger.debug(f"Cached entry: {cache_key} (size: {entry.size_bytes} bytes)")
            return True
    
    def _make_space_for_entry(self, new_entry: CacheEntry) -> bool:
        """Make space for a new cache entry by evicting old ones."""
        # Check entry count limit
        while len(self.cache) >= self.max_entries:
            if not self._evict_lru_entry():
                return False
        
        # Check size limit
        while (self.stats["total_size_bytes"] + new_entry.size_bytes) > self.max_size_bytes:
            if not self._evict_lru_entry():
                return False
        
        return True
    
    def _evict_lru_entry(self) -> bool:
        """Evict the least recently used entry."""
        if not self.cache:
            return False
        
        # Get LRU entry (first in OrderedDict)
        lru_key, lru_entry = next(iter(self.cache.items()))
        
        # Remove from cache
        del self.cache[lru_key]
        self.stats["evictions"] += 1
        
        logger.debug(f"Evicted LRU entry: {lru_key} (size: {lru_entry.size_bytes} bytes)")
        return True
    
    def _update_total_size(self):
        """Update total cache size statistics."""
        self.stats["total_size_bytes"] = sum(entry.size_bytes for entry in self.cache.values())
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats["expirations"] += 1
            
            if expired_keys:
                self._update_total_size()
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            self.stats["cleanup_runs"] += 1
    
    def _optimize_cache(self):
        """Optimize cache by analyzing access patterns."""
        with self._lock:
            if not self.cache:
                return
            
            # Find entries with very low access frequency
            current_time = datetime.now()
            low_access_keys = []
            
            for key, entry in self.cache.items():
                age_hours = (current_time - entry.created_at).total_seconds() / 3600
                if age_hours > 1:  # Only consider entries older than 1 hour
                    access_rate = entry.access_count / age_hours
                    if access_rate < 0.1:  # Less than 0.1 accesses per hour
                        low_access_keys.append(key)
            
            # Remove low-access entries if we're near capacity
            if len(self.cache) > (self.max_entries * 0.8):  # 80% capacity
                for key in low_access_keys[:len(low_access_keys)//2]:  # Remove half
                    del self.cache[key]
                    self.stats["evictions"] += 1
                
                if low_access_keys:
                    self._update_total_size()
                    logger.info(f"Optimized cache by removing {len(low_access_keys)//2} low-access entries")
    
    def invalidate(self, key: Union[str, List[Any]]) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                self._update_total_size()
                logger.debug(f"Invalidated cache entry: {cache_key}")
                return True
            
            return False
    
    def invalidate_by_tags(self, tags: Dict[str, str]) -> int:
        """
        Invalidate cache entries by tags.
        
        Args:
            tags: Tags to match for invalidation
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                # Check if all specified tags match
                if all(entry.tags.get(tag_key) == tag_value for tag_key, tag_value in tags.items()):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            if keys_to_remove:
                self._update_total_size()
                logger.info(f"Invalidated {len(keys_to_remove)} cache entries by tags: {tags}")
            
            return len(keys_to_remove)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            entry_count = len(self.cache)
            self.cache.clear()
            self.stats["total_size_bytes"] = 0
            logger.info(f"Cleared {entry_count} cache entries")
    
    def warm_cache(self, warming_key: str, *args, **kwargs) -> bool:
        """
        Warm cache using a registered warming function.
        
        Args:
            warming_key: Key for the warming function
            *args, **kwargs: Arguments for the warming function
            
        Returns:
            True if warming was successful, False otherwise
        """
        if warming_key not in self._warming_functions:
            logger.warning(f"No warming function registered for key: {warming_key}")
            return False
        
        try:
            warming_func = self._warming_functions[warming_key]
            warming_func(self, *args, **kwargs)
            logger.info(f"Cache warming completed for: {warming_key}")
            return True
        except Exception as e:
            logger.error(f"Error warming cache for {warming_key}: {e}")
            return False
    
    def register_warming_function(self, key: str, func: Callable):
        """
        Register a cache warming function.
        
        Args:
            key: Unique key for the warming function
            func: Function that takes (cache, *args, **kwargs) and warms the cache
        """
        self._warming_functions[key] = func
        logger.info(f"Registered cache warming function: {key}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate size distribution
            size_distribution = {}
            if self.cache:
                sizes = [entry.size_bytes for entry in self.cache.values()]
                size_distribution = {
                    "min_size_bytes": min(sizes),
                    "max_size_bytes": max(sizes),
                    "avg_size_bytes": sum(sizes) / len(sizes)
                }
            
            # Calculate access frequency distribution
            access_distribution = {}
            if self.cache:
                access_counts = [entry.access_count for entry in self.cache.values()]
                access_distribution = {
                    "min_access_count": min(access_counts),
                    "max_access_count": max(access_counts),
                    "avg_access_count": sum(access_counts) / len(access_counts)
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cache_performance": {
                    "hit_rate_percent": hit_rate,
                    "total_hits": self.stats["hits"],
                    "total_misses": self.stats["misses"],
                    "total_requests": total_requests
                },
                "cache_size": {
                    "current_entries": len(self.cache),
                    "max_entries": self.max_entries,
                    "current_size_bytes": self.stats["total_size_bytes"],
                    "current_size_mb": self.stats["total_size_bytes"] / 1024 / 1024,
                    "max_size_mb": self.max_size_bytes / 1024 / 1024,
                    "utilization_percent": (len(self.cache) / self.max_entries * 100) if self.max_entries > 0 else 0
                },
                "cache_operations": {
                    "evictions": self.stats["evictions"],
                    "expirations": self.stats["expirations"],
                    "cleanup_runs": self.stats["cleanup_runs"]
                },
                "size_distribution": size_distribution,
                "access_distribution": access_distribution,
                "warming_functions": list(self._warming_functions.keys())
            }
    
    def get_cache_contents(self, include_values: bool = False) -> List[Dict[str, Any]]:
        """
        Get information about current cache contents.
        
        Args:
            include_values: Whether to include actual cached values
            
        Returns:
            List of cache entry information
        """
        with self._lock:
            contents = []
            
            for key, entry in self.cache.items():
                entry_info = {
                    "key": key,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "ttl_seconds": entry.ttl_seconds,
                    "size_bytes": entry.size_bytes,
                    "tags": entry.tags,
                    "is_expired": entry.is_expired()
                }
                
                if include_values:
                    entry_info["value"] = entry.value
                
                contents.append(entry_info)
            
            return contents
    
    def shutdown(self):
        """Shutdown the cache system."""
        logger.info("Shutting down IntelligentCache")
        self._cleanup_active = False
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        with self._lock:
            self.cache.clear()
        
        logger.info("IntelligentCache shutdown complete")

# Specialized cache implementations for common use cases

class PricingCache(IntelligentCache):
    """Specialized cache for AWS pricing data."""
    
    def __init__(self):
        # Pricing data changes infrequently, so longer TTL
        super().__init__(
            max_size_mb=50,
            max_entries=5000,
            default_ttl_seconds=3600 * 6,  # 6 hours
            cleanup_interval_minutes=30
        )
        
        # Register warming functions
        self.register_warming_function("s3_pricing", self._warm_s3_pricing)
        self.register_warming_function("ec2_pricing", self._warm_ec2_pricing)
    
    def _warm_s3_pricing(self, cache, region: str = "us-east-1"):
        """Warm cache with common S3 pricing data."""
        # This would be implemented to preload common S3 pricing queries
        logger.info(f"Warming S3 pricing cache for region: {region}")
    
    def _warm_ec2_pricing(self, cache, region: str = "us-east-1"):
        """Warm cache with common EC2 pricing data."""
        # This would be implemented to preload common EC2 pricing queries
        logger.info(f"Warming EC2 pricing cache for region: {region}")

class BucketMetadataCache(IntelligentCache):
    """Specialized cache for S3 bucket metadata."""
    
    def __init__(self):
        # Bucket metadata changes more frequently
        super().__init__(
            max_size_mb=30,
            max_entries=3000,
            default_ttl_seconds=1800,  # 30 minutes
            cleanup_interval_minutes=15
        )

class AnalysisResultsCache(IntelligentCache):
    """Specialized cache for analysis results."""
    
    def __init__(self):
        # Analysis results can be large but are valuable to cache
        super().__init__(
            max_size_mb=200,
            max_entries=1000,
            default_ttl_seconds=3600,  # 1 hour
            cleanup_interval_minutes=20
        )

# Global cache instances
_pricing_cache = None
_bucket_metadata_cache = None
_analysis_results_cache = None

def get_pricing_cache() -> PricingCache:
    """Get the global pricing cache instance."""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = PricingCache()
    return _pricing_cache

def get_bucket_metadata_cache() -> BucketMetadataCache:
    """Get the global bucket metadata cache instance."""
    global _bucket_metadata_cache
    if _bucket_metadata_cache is None:
        _bucket_metadata_cache = BucketMetadataCache()
    return _bucket_metadata_cache

def get_analysis_results_cache() -> AnalysisResultsCache:
    """Get the global analysis results cache instance."""
    global _analysis_results_cache
    if _analysis_results_cache is None:
        _analysis_results_cache = AnalysisResultsCache()
    return _analysis_results_cache