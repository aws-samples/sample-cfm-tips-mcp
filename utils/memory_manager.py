"""
Memory Management System for CFM Tips MCP Server

Provides intelligent memory management and cleanup for large dataset processing
with automatic garbage collection and memory monitoring.
"""

import logging
import gc
import threading
import time
import psutil
import weakref
from typing import Dict, List, Any, Optional, Callable, Set, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import sys

logger = logging.getLogger(__name__)

@dataclass
class MemoryThreshold:
    """Memory threshold configuration."""
    warning_percent: float = 70.0  # Warning threshold
    critical_percent: float = 85.0  # Critical threshold
    cleanup_percent: float = 90.0  # Force cleanup threshold
    max_memory_mb: Optional[float] = None  # Absolute memory limit

@dataclass
class MemoryUsageSnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    process_memory_mb: float
    process_memory_percent: float
    system_memory_percent: float
    gc_objects: int
    gc_collections: Dict[int, int]
    large_objects_count: int = 0
    cached_data_mb: float = 0.0

class MemoryTracker:
    """Tracks memory usage for specific operations or objects."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self.current_memory = 0.0
        self.start_time = datetime.now()
        self.allocations: List[Tuple[datetime, float]] = []
        self._active = True
    
    def start_tracking(self):
        """Start memory tracking."""
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        self.current_memory = self.start_memory
        self.peak_memory = self.start_memory
        self._active = True
        logger.debug(f"Started memory tracking for {self.name}: {self.start_memory:.1f}MB")
    
    def update_memory(self):
        """Update current memory usage."""
        if not self._active:
            return
        
        process = psutil.Process()
        self.current_memory = process.memory_info().rss / 1024 / 1024
        
        if self.current_memory > self.peak_memory:
            self.peak_memory = self.current_memory
        
        self.allocations.append((datetime.now(), self.current_memory))
        
        # Keep only recent allocations
        if len(self.allocations) > 1000:
            self.allocations = self.allocations[-1000:]
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and return summary."""
        self._active = False
        self.update_memory()
        
        memory_delta = self.current_memory - self.start_memory
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "name": self.name,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": self.current_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": memory_delta,
            "duration_seconds": duration,
            "allocations_count": len(self.allocations)
        }

class LargeObjectRegistry:
    """Registry for tracking large objects that can be cleaned up."""
    
    def __init__(self):
        self._objects: Dict[str, weakref.ref] = {}
        self._cleanup_callbacks: Dict[str, Callable] = {}
        self._object_sizes: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def register_object(self, obj_id: str, obj: Any, size_mb: float, 
                       cleanup_callback: Optional[Callable] = None):
        """
        Register a large object for tracking.
        
        Args:
            obj_id: Unique identifier for the object
            obj: The object to track
            size_mb: Estimated size in MB
            cleanup_callback: Optional cleanup function
        """
        with self._lock:
            # Create weak reference to avoid keeping object alive
            self._objects[obj_id] = weakref.ref(obj)
            self._object_sizes[obj_id] = size_mb
            
            if cleanup_callback:
                self._cleanup_callbacks[obj_id] = cleanup_callback
            
            logger.debug(f"Registered large object {obj_id}: {size_mb:.1f}MB")
    
    def cleanup_object(self, obj_id: str) -> bool:
        """
        Clean up a specific object.
        
        Args:
            obj_id: Object identifier
            
        Returns:
            True if cleanup was successful
        """
        with self._lock:
            if obj_id not in self._objects:
                return False
            
            # Call cleanup callback if available
            if obj_id in self._cleanup_callbacks:
                try:
                    self._cleanup_callbacks[obj_id]()
                    logger.debug(f"Executed cleanup callback for {obj_id}")
                except Exception as e:
                    logger.error(f"Error in cleanup callback for {obj_id}: {e}")
            
            # Remove from registry
            size_mb = self._object_sizes.get(obj_id, 0)
            del self._objects[obj_id]
            self._object_sizes.pop(obj_id, None)
            self._cleanup_callbacks.pop(obj_id, None)
            
            logger.info(f"Cleaned up large object {obj_id}: {size_mb:.1f}MB")
            return True
    
    def cleanup_dead_references(self) -> int:
        """Clean up objects that have been garbage collected."""
        with self._lock:
            dead_objects = []
            
            for obj_id, weak_ref in self._objects.items():
                if weak_ref() is None:  # Object has been garbage collected
                    dead_objects.append(obj_id)
            
            for obj_id in dead_objects:
                self._object_sizes.pop(obj_id, None)
                self._cleanup_callbacks.pop(obj_id, None)
                del self._objects[obj_id]
            
            if dead_objects:
                logger.debug(f"Cleaned up {len(dead_objects)} dead object references")
            
            return len(dead_objects)
    
    def get_total_size(self) -> float:
        """Get total size of registered objects."""
        with self._lock:
            return sum(self._object_sizes.values())
    
    def get_largest_objects(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the largest registered objects."""
        with self._lock:
            sorted_objects = sorted(
                self._object_sizes.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_objects[:limit]
    
    def force_cleanup_largest(self, count: int = 5) -> List[str]:
        """Force cleanup of the largest objects."""
        largest_objects = self.get_largest_objects(count)
        cleaned_objects = []
        
        for obj_id, size_mb in largest_objects:
            if self.cleanup_object(obj_id):
                cleaned_objects.append(obj_id)
        
        return cleaned_objects

class MemoryManager:
    """
    Comprehensive memory management system for S3 optimization analyses.
    
    Features:
    - Real-time memory monitoring
    - Automatic cleanup when thresholds are exceeded
    - Large object tracking and cleanup
    - Memory usage profiling for analyses
    - Garbage collection optimization
    - Memory leak detection
    """
    
    def __init__(self, thresholds: Optional[MemoryThreshold] = None):
        """
        Initialize MemoryManager.
        
        Args:
            thresholds: Memory threshold configuration
        """
        self.thresholds = thresholds or MemoryThreshold()
        
        # Memory monitoring
        self.process = psutil.Process()
        self.memory_snapshots: List[MemoryUsageSnapshot] = []
        self.max_snapshots = 1000
        
        # Object tracking
        self.large_object_registry = LargeObjectRegistry()
        self.memory_trackers: Dict[str, MemoryTracker] = {}
        
        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = True
        self._monitoring_thread = None
        
        # Performance monitor integration
        self._performance_monitor = None
        
        # Cache references for cleanup
        self._cache_references: List[Any] = []
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("MemoryManager initialized")
    
    def set_performance_monitor(self, performance_monitor):
        """Set performance monitor for integration."""
        self._performance_monitor = performance_monitor
    
    def add_cache_reference(self, cache_instance):
        """Add cache instance for memory cleanup."""
        self._cache_references.append(weakref.ref(cache_instance))
    
    def _start_monitoring(self):
        """Start background memory monitoring."""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Memory monitoring thread started")
    
    def _monitoring_worker(self):
        """Background worker for memory monitoring."""
        while self._monitoring_active:
            try:
                self._take_memory_snapshot()
                self._check_memory_thresholds()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _take_memory_snapshot(self):
        """Take a snapshot of current memory usage."""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            gc_collections = {i: stats['collections'] for i, stats in enumerate(gc_stats)}
            
            snapshot = MemoryUsageSnapshot(
                timestamp=datetime.now(),
                process_memory_mb=memory_info.rss / 1024 / 1024,
                process_memory_percent=memory_percent,
                system_memory_percent=system_memory.percent,
                gc_objects=len(gc.get_objects()),
                gc_collections=gc_collections,
                large_objects_count=len(self.large_object_registry._objects),
                cached_data_mb=self.large_object_registry.get_total_size()
            )
            
            with self._lock:
                self.memory_snapshots.append(snapshot)
                
                # Keep only recent snapshots
                if len(self.memory_snapshots) > self.max_snapshots:
                    self.memory_snapshots = self.memory_snapshots[-self.max_snapshots:]
            
            # Record metrics if performance monitor is available
            if self._performance_monitor:
                self._performance_monitor.record_metric(
                    "memory_usage_mb",
                    snapshot.process_memory_mb,
                    tags={"component": "memory_manager"}
                )
                
                self._performance_monitor.record_metric(
                    "memory_percent",
                    snapshot.process_memory_percent,
                    tags={"component": "memory_manager"}
                )
        
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
    
    def _check_memory_thresholds(self):
        """Check memory thresholds and trigger cleanup if needed."""
        if not self.memory_snapshots:
            return
        
        latest_snapshot = self.memory_snapshots[-1]
        memory_percent = latest_snapshot.process_memory_percent
        
        # Check absolute memory limit
        if (self.thresholds.max_memory_mb and 
            latest_snapshot.process_memory_mb > self.thresholds.max_memory_mb):
            logger.warning(
                f"Absolute memory limit exceeded: {latest_snapshot.process_memory_mb:.1f}MB > "
                f"{self.thresholds.max_memory_mb:.1f}MB"
            )
            self._trigger_emergency_cleanup()
            return
        
        # Check percentage thresholds
        if memory_percent >= self.thresholds.cleanup_percent:
            logger.warning(f"Memory cleanup threshold exceeded: {memory_percent:.1f}%")
            self._trigger_emergency_cleanup()
        elif memory_percent >= self.thresholds.critical_percent:
            logger.warning(f"Critical memory threshold exceeded: {memory_percent:.1f}%")
            self._trigger_aggressive_cleanup()
        elif memory_percent >= self.thresholds.warning_percent:
            logger.info(f"Memory warning threshold exceeded: {memory_percent:.1f}%")
            self._trigger_gentle_cleanup()
    
    def _trigger_gentle_cleanup(self):
        """Trigger gentle cleanup (cache cleanup, dead references)."""
        logger.info("Triggering gentle memory cleanup")
        
        # Clean up dead references
        self.large_object_registry.cleanup_dead_references()
        
        # Clean up cache entries (if caches are available)
        self._cleanup_caches(aggressive=False)
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Gentle cleanup completed, collected {collected} objects")
    
    def _trigger_aggressive_cleanup(self):
        """Trigger aggressive cleanup (large objects, more cache cleanup)."""
        logger.warning("Triggering aggressive memory cleanup")
        
        # Clean up largest objects
        cleaned_objects = self.large_object_registry.force_cleanup_largest(3)
        logger.info(f"Cleaned up {len(cleaned_objects)} large objects")
        
        # Aggressive cache cleanup
        self._cleanup_caches(aggressive=True)
        
        # Execute registered cleanup callbacks
        self._execute_cleanup_callbacks()
        
        # Force garbage collection multiple times
        for generation in range(3):
            collected = gc.collect(generation)
            logger.debug(f"GC generation {generation}: collected {collected} objects")
        
        logger.warning("Aggressive cleanup completed")
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency cleanup (all available cleanup methods)."""
        logger.error("Triggering emergency memory cleanup")
        
        # Clean up all large objects
        largest_objects = self.large_object_registry.get_largest_objects(20)
        for obj_id, size_mb in largest_objects:
            self.large_object_registry.cleanup_object(obj_id)
        
        # Emergency cache cleanup
        self._cleanup_caches(aggressive=True, emergency=True)
        
        # Execute all cleanup callbacks
        self._execute_cleanup_callbacks()
        
        # Force comprehensive garbage collection
        for _ in range(5):
            gc.collect()
        
        # Clear memory trackers
        with self._lock:
            self.memory_trackers.clear()
        
        logger.error("Emergency cleanup completed")
    
    def _cleanup_caches(self, aggressive: bool = False, emergency: bool = False):
        """Clean up cache instances."""
        for cache_ref in self._cache_references[:]:  # Copy list to avoid modification during iteration
            cache = cache_ref()
            if cache is None:
                # Remove dead reference
                self._cache_references.remove(cache_ref)
                continue
            
            try:
                if emergency:
                    # Clear entire cache
                    cache.clear()
                    logger.info(f"Emergency: Cleared entire cache {type(cache).__name__}")
                elif aggressive:
                    # Clean up old entries aggressively
                    if hasattr(cache, '_cleanup_expired_entries'):
                        cache._cleanup_expired_entries()
                    if hasattr(cache, '_optimize_cache'):
                        cache._optimize_cache()
                    logger.info(f"Aggressive cleanup of cache {type(cache).__name__}")
                else:
                    # Gentle cleanup
                    if hasattr(cache, '_cleanup_expired_entries'):
                        cache._cleanup_expired_entries()
                    logger.debug(f"Gentle cleanup of cache {type(cache).__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up cache {type(cache).__name__}: {e}")
    
    def _execute_cleanup_callbacks(self):
        """Execute registered cleanup callbacks."""
        for callback in self.cleanup_callbacks:
            try:
                callback()
                logger.debug("Executed cleanup callback")
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)
        logger.debug("Registered cleanup callback")
    
    def start_memory_tracking(self, tracker_name: str) -> MemoryTracker:
        """
        Start memory tracking for a specific operation.
        
        Args:
            tracker_name: Name for the tracker
            
        Returns:
            MemoryTracker instance
        """
        with self._lock:
            tracker = MemoryTracker(tracker_name)
            tracker.start_tracking()
            self.memory_trackers[tracker_name] = tracker
            return tracker
    
    def stop_memory_tracking(self, tracker_name: str) -> Optional[Dict[str, Any]]:
        """
        Stop memory tracking and get results.
        
        Args:
            tracker_name: Name of the tracker
            
        Returns:
            Memory tracking results or None if tracker not found
        """
        with self._lock:
            if tracker_name not in self.memory_trackers:
                return None
            
            tracker = self.memory_trackers[tracker_name]
            results = tracker.stop_tracking()
            del self.memory_trackers[tracker_name]
            
            logger.info(
                f"Memory tracking completed for {tracker_name}: "
                f"{results['memory_delta_mb']:+.1f}MB delta, "
                f"peak: {results['peak_memory_mb']:.1f}MB"
            )
            
            return results
    
    def register_large_object(self, obj_id: str, obj: Any, size_mb: float,
                            cleanup_callback: Optional[Callable] = None):
        """Register a large object for tracking and cleanup."""
        self.large_object_registry.register_object(obj_id, obj, size_mb, cleanup_callback)
    
    def cleanup_large_object(self, obj_id: str) -> bool:
        """Clean up a specific large object."""
        return self.large_object_registry.cleanup_object(obj_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self._lock:
            if not self.memory_snapshots:
                return {"error": "No memory snapshots available"}
            
            latest_snapshot = self.memory_snapshots[-1]
            
            # Calculate memory trends
            if len(self.memory_snapshots) >= 2:
                previous_snapshot = self.memory_snapshots[-2]
                memory_trend = latest_snapshot.process_memory_mb - previous_snapshot.process_memory_mb
            else:
                memory_trend = 0.0
            
            # Calculate average memory usage (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_snapshots = [
                s for s in self.memory_snapshots 
                if s.timestamp >= one_hour_ago
            ]
            
            avg_memory_mb = 0.0
            if recent_snapshots:
                avg_memory_mb = sum(s.process_memory_mb for s in recent_snapshots) / len(recent_snapshots)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "current_memory": {
                    "process_memory_mb": latest_snapshot.process_memory_mb,
                    "process_memory_percent": latest_snapshot.process_memory_percent,
                    "system_memory_percent": latest_snapshot.system_memory_percent,
                    "memory_trend_mb": memory_trend
                },
                "thresholds": {
                    "warning_percent": self.thresholds.warning_percent,
                    "critical_percent": self.thresholds.critical_percent,
                    "cleanup_percent": self.thresholds.cleanup_percent,
                    "max_memory_mb": self.thresholds.max_memory_mb
                },
                "statistics": {
                    "avg_memory_mb_1h": avg_memory_mb,
                    "snapshots_count": len(self.memory_snapshots),
                    "gc_objects": latest_snapshot.gc_objects,
                    "large_objects_count": latest_snapshot.large_objects_count,
                    "cached_data_mb": latest_snapshot.cached_data_mb
                },
                "active_trackers": list(self.memory_trackers.keys()),
                "cleanup_callbacks_count": len(self.cleanup_callbacks)
            }
    
    def get_memory_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get memory usage history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_snapshots = [
                s for s in self.memory_snapshots 
                if s.timestamp >= cutoff_time
            ]
            
            return [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "process_memory_mb": s.process_memory_mb,
                    "process_memory_percent": s.process_memory_percent,
                    "system_memory_percent": s.system_memory_percent,
                    "gc_objects": s.gc_objects,
                    "large_objects_count": s.large_objects_count
                }
                for s in recent_snapshots
            ]
    
    def force_cleanup(self, level: str = "gentle"):
        """
        Force memory cleanup.
        
        Args:
            level: Cleanup level ('gentle', 'aggressive', 'emergency')
        """
        if level == "gentle":
            self._trigger_gentle_cleanup()
        elif level == "aggressive":
            self._trigger_aggressive_cleanup()
        elif level == "emergency":
            self._trigger_emergency_cleanup()
        else:
            raise ValueError(f"Invalid cleanup level: {level}")
    
    def shutdown(self):
        """Shutdown the memory manager."""
        logger.info("Shutting down MemoryManager")
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        # Final cleanup
        self._trigger_gentle_cleanup()
        
        with self._lock:
            self.memory_snapshots.clear()
            self.memory_trackers.clear()
            self.cleanup_callbacks.clear()
        
        logger.info("MemoryManager shutdown complete")

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager