"""
Performance Monitor for CFM Tips MCP Server

Provides comprehensive performance monitoring and metrics collection for S3 optimization analyses.
"""

import logging
import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags
        }

@dataclass
class AnalysisPerformanceData:
    """Performance data for a specific analysis execution."""
    analysis_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    data_processed_mb: float = 0.0
    timeout_occurred: bool = False
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    def calculate_execution_time(self):
        """Calculate execution time if end_time is set."""
        if self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for S3 optimization analyses.
    
    Features:
    - Real-time performance metrics collection
    - Memory usage tracking and alerts
    - Cache performance monitoring
    - Analysis execution profiling
    - Resource utilization tracking
    - Performance trend analysis
    """
    
    def __init__(self, max_metrics_history: int = 10000, cleanup_interval_minutes: int = 30):
        """
        Initialize PerformanceMonitor.
        
        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
            cleanup_interval_minutes: Interval for automatic cleanup
        """
        self.max_metrics_history = max_metrics_history
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # Metrics storage
        self.metrics: deque = deque(maxlen=max_metrics_history)
        self.analysis_performance: Dict[str, AnalysisPerformanceData] = {}
        self.performance_history: Dict[str, List[AnalysisPerformanceData]] = defaultdict(list)
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(float)
        self.gauges = defaultdict(float)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = True
        self._monitoring_thread = None
        self._cleanup_thread = None
        
        # System monitoring
        self.process = psutil.Process()
        self.system_metrics_interval = 5.0  # seconds
        
        # Start background monitoring
        self._start_monitoring()
        
        logger.info("PerformanceMonitor initialized")
    
    def _start_monitoring(self):
        """Start background monitoring threads."""
        # System metrics monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._system_monitoring_worker,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitoring_thread.start()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="PerformanceCleanup"
        )
        self._cleanup_thread.start()
        
        logger.info("Performance monitoring threads started")
    
    def _system_monitoring_worker(self):
        """Background worker for system metrics collection."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                time.sleep(self.system_metrics_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _cleanup_worker(self):
        """Background worker for periodic cleanup."""
        while self._monitoring_active:
            try:
                time.sleep(self.cleanup_interval_minutes * 60)
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in performance cleanup: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Record metrics
            timestamp = datetime.now()
            
            with self._lock:
                self.metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="memory_usage_mb",
                    value=memory_info.rss / 1024 / 1024,
                    tags={"component": "system"}
                ))
                
                self.metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="memory_percent",
                    value=memory_percent,
                    tags={"component": "system"}
                ))
                
                self.metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="cpu_percent",
                    value=cpu_percent,
                    tags={"component": "system"}
                ))
                
                self.metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="system_memory_percent",
                    value=system_memory.percent,
                    tags={"component": "system"}
                ))
                
                # Update gauges
                self.gauges["current_memory_mb"] = memory_info.rss / 1024 / 1024
                self.gauges["current_cpu_percent"] = cpu_percent
                self.gauges["system_memory_percent"] = system_memory.percent
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def start_analysis_monitoring(self, analysis_type: str, execution_id: str) -> str:
        """
        Start monitoring for a specific analysis execution.
        
        Args:
            analysis_type: Type of analysis being executed
            execution_id: Unique identifier for this execution
            
        Returns:
            Monitoring session ID
        """
        session_id = f"{analysis_type}_{execution_id}_{int(time.time())}"
        
        with self._lock:
            # Get initial memory usage
            memory_info = self.process.memory_info()
            
            performance_data = AnalysisPerformanceData(
                analysis_type=analysis_type,
                start_time=datetime.now(),
                memory_usage_mb=memory_info.rss / 1024 / 1024,
                peak_memory_mb=memory_info.rss / 1024 / 1024
            )
            
            self.analysis_performance[session_id] = performance_data
            
            # Record start metric
            self.record_metric(
                "analysis_started",
                1,
                tags={"analysis_type": analysis_type, "session_id": session_id}
            )
            
            self.counters[f"analysis_starts_{analysis_type}"] += 1
        
        logger.debug(f"Started monitoring for {analysis_type} (session: {session_id})")
        return session_id
    
    def end_analysis_monitoring(self, session_id: str, success: bool = True, 
                              error_message: Optional[str] = None) -> AnalysisPerformanceData:
        """
        End monitoring for a specific analysis execution.
        
        Args:
            session_id: Monitoring session ID
            success: Whether the analysis completed successfully
            error_message: Error message if analysis failed
            
        Returns:
            Performance data for the completed analysis
        """
        with self._lock:
            if session_id not in self.analysis_performance:
                logger.warning(f"Monitoring session not found: {session_id}")
                return None
            
            performance_data = self.analysis_performance[session_id]
            performance_data.end_time = datetime.now()
            performance_data.calculate_execution_time()
            performance_data.error_occurred = not success
            performance_data.error_message = error_message
            
            # Get final memory usage
            memory_info = self.process.memory_info()
            final_memory_mb = memory_info.rss / 1024 / 1024
            
            # Update peak memory if current is higher
            if final_memory_mb > performance_data.peak_memory_mb:
                performance_data.peak_memory_mb = final_memory_mb
            
            # Record completion metrics
            self.record_metric(
                "analysis_completed",
                1,
                tags={
                    "analysis_type": performance_data.analysis_type,
                    "session_id": session_id,
                    "success": str(success)
                }
            )
            
            self.record_metric(
                "analysis_execution_time",
                performance_data.execution_time,
                tags={
                    "analysis_type": performance_data.analysis_type,
                    "session_id": session_id
                }
            )
            
            # Update counters and timers
            analysis_type = performance_data.analysis_type
            if success:
                self.counters[f"analysis_success_{analysis_type}"] += 1
            else:
                self.counters[f"analysis_error_{analysis_type}"] += 1
            
            self.timers[f"analysis_time_{analysis_type}"] += performance_data.execution_time
            
            # Store in history
            self.performance_history[analysis_type].append(performance_data)
            
            # Remove from active monitoring
            del self.analysis_performance[session_id]
        
        logger.debug(f"Ended monitoring for session {session_id} (success: {success})")
        return performance_data
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                tags=tags or {}
            )
            
            self.metrics.append(metric)
    
    def increment_counter(self, counter_name: str, increment: int = 1, 
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[counter_name] += increment
            
            # Also record as a metric
            self.record_metric(f"counter_{counter_name}", increment, tags)
    
    def record_cache_hit(self, cache_type: str, analysis_type: Optional[str] = None):
        """Record a cache hit."""
        tags = {"cache_type": cache_type}
        if analysis_type:
            tags["analysis_type"] = analysis_type
        
        self.increment_counter("cache_hits", tags=tags)
        
        # Update analysis performance if active
        if analysis_type:
            with self._lock:
                for session_id, perf_data in self.analysis_performance.items():
                    if perf_data.analysis_type == analysis_type:
                        perf_data.cache_hits += 1
    
    def record_cache_miss(self, cache_type: str, analysis_type: Optional[str] = None):
        """Record a cache miss."""
        tags = {"cache_type": cache_type}
        if analysis_type:
            tags["analysis_type"] = analysis_type
        
        self.increment_counter("cache_misses", tags=tags)
        
        # Update analysis performance if active
        if analysis_type:
            with self._lock:
                for session_id, perf_data in self.analysis_performance.items():
                    if perf_data.analysis_type == analysis_type:
                        perf_data.cache_misses += 1
    
    def record_api_call(self, service: str, operation: str, analysis_type: Optional[str] = None):
        """Record an API call."""
        tags = {"service": service, "operation": operation}
        if analysis_type:
            tags["analysis_type"] = analysis_type
        
        self.increment_counter("api_calls", tags=tags)
        
        # Update analysis performance if active
        if analysis_type:
            with self._lock:
                for session_id, perf_data in self.analysis_performance.items():
                    if perf_data.analysis_type == analysis_type:
                        perf_data.api_calls += 1
    
    def record_data_processed(self, size_mb: float, analysis_type: Optional[str] = None):
        """Record amount of data processed."""
        tags = {}
        if analysis_type:
            tags["analysis_type"] = analysis_type
        
        self.record_metric("data_processed_mb", size_mb, tags)
        
        # Update analysis performance if active
        if analysis_type:
            with self._lock:
                for session_id, perf_data in self.analysis_performance.items():
                    if perf_data.analysis_type == analysis_type:
                        perf_data.data_processed_mb += size_mb
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            # Calculate summary statistics
            current_time = datetime.now()
            
            # Recent metrics (last 5 minutes)
            recent_cutoff = current_time - timedelta(minutes=5)
            recent_metrics = [m for m in self.metrics if m.timestamp >= recent_cutoff]
            
            # Analysis performance summary
            analysis_summary = {}
            for analysis_type, history in self.performance_history.items():
                if history:
                    execution_times = [p.execution_time for p in history if p.execution_time > 0]
                    memory_usage = [p.peak_memory_mb for p in history if p.peak_memory_mb > 0]
                    
                    analysis_summary[analysis_type] = {
                        "total_executions": len(history),
                        "successful_executions": len([p for p in history if not p.error_occurred]),
                        "failed_executions": len([p for p in history if p.error_occurred]),
                        "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                        "max_execution_time": max(execution_times) if execution_times else 0,
                        "min_execution_time": min(execution_times) if execution_times else 0,
                        "avg_memory_usage_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                        "max_memory_usage_mb": max(memory_usage) if memory_usage else 0
                    }
            
            # Cache performance
            total_cache_hits = self.counters.get("cache_hits", 0)
            total_cache_misses = self.counters.get("cache_misses", 0)
            total_cache_requests = total_cache_hits + total_cache_misses
            cache_hit_rate = (total_cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            return {
                "timestamp": current_time.isoformat(),
                "system_metrics": {
                    "current_memory_mb": self.gauges.get("current_memory_mb", 0),
                    "current_cpu_percent": self.gauges.get("current_cpu_percent", 0),
                    "system_memory_percent": self.gauges.get("system_memory_percent", 0)
                },
                "cache_performance": {
                    "total_hits": total_cache_hits,
                    "total_misses": total_cache_misses,
                    "hit_rate_percent": cache_hit_rate
                },
                "analysis_performance": analysis_summary,
                "active_monitoring_sessions": len(self.analysis_performance),
                "total_metrics_collected": len(self.metrics),
                "recent_metrics_count": len(recent_metrics),
                "counters": dict(self.counters),
                "timers": dict(self.timers),
                "gauges": dict(self.gauges)
            }
    
    def get_analysis_performance_history(self, analysis_type: str, 
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history for a specific analysis type."""
        with self._lock:
            history = self.performance_history.get(analysis_type, [])
            recent_history = history[-limit:] if len(history) > limit else history
            
            return [
                {
                    "analysis_type": p.analysis_type,
                    "start_time": p.start_time.isoformat(),
                    "end_time": p.end_time.isoformat() if p.end_time else None,
                    "execution_time": p.execution_time,
                    "memory_usage_mb": p.memory_usage_mb,
                    "peak_memory_mb": p.peak_memory_mb,
                    "cache_hits": p.cache_hits,
                    "cache_misses": p.cache_misses,
                    "api_calls": p.api_calls,
                    "data_processed_mb": p.data_processed_mb,
                    "error_occurred": p.error_occurred,
                    "error_message": p.error_message
                }
                for p in recent_history
            ]
    
    def _cleanup_old_data(self):
        """Clean up old performance data to manage memory usage."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data
        
        with self._lock:
            # Clean up performance history
            for analysis_type in list(self.performance_history.keys()):
                history = self.performance_history[analysis_type]
                # Keep only recent data
                recent_history = [p for p in history if p.start_time >= cutoff_time]
                
                if len(recent_history) != len(history):
                    self.performance_history[analysis_type] = recent_history
                    logger.debug(f"Cleaned up {len(history) - len(recent_history)} old performance records for {analysis_type}")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Performance data cleanup completed")
    
    def export_metrics(self, start_time: Optional[datetime] = None, 
                      end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export metrics for external analysis."""
        with self._lock:
            filtered_metrics = self.metrics
            
            if start_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
            
            if end_time:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
            
            return [metric.to_dict() for metric in filtered_metrics]
    
    def shutdown(self):
        """Shutdown the performance monitor."""
        logger.info("Shutting down PerformanceMonitor")
        self._monitoring_active = False
        
        # Wait for threads to finish
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        logger.info("PerformanceMonitor shutdown complete")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor