"""
Progressive Timeout Handler for CFM Tips MCP Server

Provides intelligent timeout management based on analysis complexity, historical performance,
and system resource availability.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Analysis complexity levels."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class TimeoutConfiguration:
    """Configuration for timeout calculation."""
    base_timeout: float = 30.0  # Base timeout in seconds
    complexity_multiplier: Dict[ComplexityLevel, float] = field(default_factory=lambda: {
        ComplexityLevel.VERY_LOW: 0.5,
        ComplexityLevel.LOW: 1.0,
        ComplexityLevel.MEDIUM: 1.5,
        ComplexityLevel.HIGH: 2.5,
        ComplexityLevel.VERY_HIGH: 4.0
    })
    data_size_multiplier: float = 0.1  # Additional seconds per MB of data
    bucket_count_multiplier: float = 2.0  # Additional seconds per bucket
    historical_performance_weight: float = 0.3  # Weight for historical performance
    system_load_weight: float = 0.2  # Weight for system load adjustment
    min_timeout: float = 10.0  # Minimum timeout
    max_timeout: float = 300.0  # Maximum timeout (5 minutes)
    grace_period: float = 15.0  # Additional grace period

@dataclass
class AnalysisContext:
    """Context information for timeout calculation."""
    analysis_type: str
    complexity_level: ComplexityLevel = ComplexityLevel.MEDIUM
    estimated_data_size_mb: float = 0.0
    bucket_count: int = 0
    region: Optional[str] = None
    include_cost_analysis: bool = True
    lookback_days: int = 30
    parallel_execution: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeoutResult:
    """Result of timeout calculation."""
    calculated_timeout: float
    base_timeout: float
    complexity_adjustment: float
    data_size_adjustment: float
    bucket_count_adjustment: float
    historical_adjustment: float
    system_load_adjustment: float
    final_timeout: float
    reasoning: List[str] = field(default_factory=list)

class ProgressiveTimeoutHandler:
    """
    Intelligent timeout handler that calculates timeouts based on:
    
    - Analysis complexity and type
    - Historical performance data
    - System resource availability
    - Data size and scope
    - Current system load
    """
    
    def __init__(self, config: Optional[TimeoutConfiguration] = None):
        """
        Initialize ProgressiveTimeoutHandler.
        
        Args:
            config: Timeout configuration (uses default if None)
        """
        self.config = config or TimeoutConfiguration()
        
        # Historical performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.complexity_history: Dict[str, List[Tuple[ComplexityLevel, float]]] = {}
        
        # System monitoring
        self.system_load_samples: List[Tuple[datetime, float]] = []
        self.max_load_samples = 100
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance monitor integration
        self._performance_monitor = None
        
        logger.info("ProgressiveTimeoutHandler initialized")
    
    def set_performance_monitor(self, performance_monitor):
        """Set performance monitor for integration."""
        self._performance_monitor = performance_monitor
    
    def calculate_timeout(self, context: AnalysisContext) -> TimeoutResult:
        """
        Calculate intelligent timeout based on analysis context.
        
        Args:
            context: Analysis context information
            
        Returns:
            TimeoutResult with calculated timeout and reasoning
        """
        with self._lock:
            reasoning = []
            
            # Start with base timeout
            base_timeout = self.config.base_timeout
            reasoning.append(f"Base timeout: {base_timeout}s")
            
            # Apply complexity multiplier
            complexity_multiplier = self.config.complexity_multiplier.get(
                context.complexity_level, 1.0
            )
            complexity_adjustment = base_timeout * (complexity_multiplier - 1.0)
            reasoning.append(
                f"Complexity adjustment ({context.complexity_level.name}): "
                f"{complexity_adjustment:+.1f}s (multiplier: {complexity_multiplier})"
            )
            
            # Apply data size adjustment
            data_size_adjustment = context.estimated_data_size_mb * self.config.data_size_multiplier
            reasoning.append(f"Data size adjustment ({context.estimated_data_size_mb:.1f}MB): {data_size_adjustment:+.1f}s")
            
            # Apply bucket count adjustment
            bucket_count_adjustment = context.bucket_count * self.config.bucket_count_multiplier
            reasoning.append(f"Bucket count adjustment ({context.bucket_count} buckets): {bucket_count_adjustment:+.1f}s")
            
            # Calculate base adjusted timeout
            calculated_timeout = (
                base_timeout + 
                complexity_adjustment + 
                data_size_adjustment + 
                bucket_count_adjustment
            )
            
            # Apply historical performance adjustment
            historical_adjustment = self._calculate_historical_adjustment(
                context.analysis_type, 
                context.complexity_level,
                calculated_timeout
            )
            reasoning.append(f"Historical performance adjustment: {historical_adjustment:+.1f}s")
            
            # Apply system load adjustment
            system_load_adjustment = self._calculate_system_load_adjustment(calculated_timeout)
            reasoning.append(f"System load adjustment: {system_load_adjustment:+.1f}s")
            
            # Calculate final timeout
            final_timeout = (
                calculated_timeout + 
                historical_adjustment + 
                system_load_adjustment + 
                self.config.grace_period
            )
            
            # Apply min/max constraints
            constrained_timeout = max(
                self.config.min_timeout,
                min(final_timeout, self.config.max_timeout)
            )
            
            if constrained_timeout != final_timeout:
                reasoning.append(
                    f"Applied constraints: {final_timeout:.1f}s -> {constrained_timeout:.1f}s "
                    f"(min: {self.config.min_timeout}s, max: {self.config.max_timeout}s)"
                )
            
            reasoning.append(f"Grace period: +{self.config.grace_period}s")
            reasoning.append(f"Final timeout: {constrained_timeout:.1f}s")
            
            result = TimeoutResult(
                calculated_timeout=calculated_timeout,
                base_timeout=base_timeout,
                complexity_adjustment=complexity_adjustment,
                data_size_adjustment=data_size_adjustment,
                bucket_count_adjustment=bucket_count_adjustment,
                historical_adjustment=historical_adjustment,
                system_load_adjustment=system_load_adjustment,
                final_timeout=constrained_timeout,
                reasoning=reasoning
            )
            
            logger.debug(f"Calculated timeout for {context.analysis_type}: {constrained_timeout:.1f}s")
            return result
    
    def _calculate_historical_adjustment(self, 
                                       analysis_type: str, 
                                       complexity_level: ComplexityLevel,
                                       base_timeout: float) -> float:
        """Calculate adjustment based on historical performance."""
        if analysis_type not in self.performance_history:
            return 0.0
        
        history = self.performance_history[analysis_type]
        if not history:
            return 0.0
        
        # Calculate average historical execution time
        avg_execution_time = sum(history) / len(history)
        
        # Calculate adjustment based on difference from base timeout
        performance_ratio = avg_execution_time / base_timeout
        
        # Apply weight and calculate adjustment
        adjustment = (avg_execution_time - base_timeout) * self.config.historical_performance_weight
        
        # Limit adjustment to reasonable bounds
        max_adjustment = base_timeout * 0.5  # Max 50% adjustment
        adjustment = max(-max_adjustment, min(adjustment, max_adjustment))
        
        return adjustment
    
    def _calculate_system_load_adjustment(self, base_timeout: float) -> float:
        """Calculate adjustment based on current system load."""
        if not self.system_load_samples:
            return 0.0
        
        # Get recent load samples (last 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_samples = [
            load for timestamp, load in self.system_load_samples 
            if timestamp >= cutoff_time
        ]
        
        if not recent_samples:
            return 0.0
        
        # Calculate average recent load
        avg_load = sum(recent_samples) / len(recent_samples)
        
        # High load increases timeout, low load decreases it
        if avg_load > 80:  # High load
            load_multiplier = 1.0 + ((avg_load - 80) / 100)  # Up to 20% increase
        elif avg_load < 30:  # Low load
            load_multiplier = 0.9  # 10% decrease
        else:
            load_multiplier = 1.0  # No adjustment
        
        adjustment = base_timeout * (load_multiplier - 1.0) * self.config.system_load_weight
        
        # Limit adjustment
        max_adjustment = base_timeout * 0.3  # Max 30% adjustment
        adjustment = max(-max_adjustment, min(adjustment, max_adjustment))
        
        return adjustment
    
    def record_execution_time(self, analysis_type: str, execution_time: float, 
                            complexity_level: ComplexityLevel = ComplexityLevel.MEDIUM):
        """
        Record execution time for historical analysis.
        
        Args:
            analysis_type: Type of analysis
            execution_time: Actual execution time in seconds
            complexity_level: Complexity level of the analysis
        """
        with self._lock:
            # Record in performance history
            if analysis_type not in self.performance_history:
                self.performance_history[analysis_type] = []
            
            self.performance_history[analysis_type].append(execution_time)
            
            # Keep only recent history (last 100 executions)
            if len(self.performance_history[analysis_type]) > 100:
                self.performance_history[analysis_type] = self.performance_history[analysis_type][-100:]
            
            # Record complexity history
            if analysis_type not in self.complexity_history:
                self.complexity_history[analysis_type] = []
            
            self.complexity_history[analysis_type].append((complexity_level, execution_time))
            
            # Keep only recent complexity history
            if len(self.complexity_history[analysis_type]) > 100:
                self.complexity_history[analysis_type] = self.complexity_history[analysis_type][-100:]
            
            logger.debug(f"Recorded execution time for {analysis_type}: {execution_time:.2f}s")
    
    def record_system_load(self, cpu_percent: float):
        """
        Record system load for load-based adjustments.
        
        Args:
            cpu_percent: CPU usage percentage
        """
        with self._lock:
            timestamp = datetime.now()
            self.system_load_samples.append((timestamp, cpu_percent))
            
            # Keep only recent samples
            if len(self.system_load_samples) > self.max_load_samples:
                self.system_load_samples = self.system_load_samples[-self.max_load_samples:]
    
    def get_complexity_level(self, analysis_type: str, **context) -> ComplexityLevel:
        """
        Determine complexity level based on analysis type and context.
        
        Args:
            analysis_type: Type of analysis
            **context: Additional context parameters
            
        Returns:
            Estimated complexity level
        """
        # Base complexity by analysis type
        base_complexity = {
            "general_spend": ComplexityLevel.HIGH,
            "storage_class": ComplexityLevel.HIGH,
            "archive_optimization": ComplexityLevel.MEDIUM,
            "api_cost": ComplexityLevel.MEDIUM,
            "multipart_cleanup": ComplexityLevel.LOW,
            "governance": ComplexityLevel.LOW,
            "comprehensive": ComplexityLevel.VERY_HIGH
        }.get(analysis_type, ComplexityLevel.MEDIUM)
        
        # Adjust based on context
        bucket_count = context.get("bucket_count", 0)
        lookback_days = context.get("lookback_days", 30)
        include_cost_analysis = context.get("include_cost_analysis", True)
        
        complexity_score = base_complexity.value
        
        # Adjust for bucket count
        if bucket_count > 100:
            complexity_score += 1
        elif bucket_count > 1000:
            complexity_score += 2
        
        # Adjust for lookback period
        if lookback_days > 90:
            complexity_score += 1
        elif lookback_days > 365:
            complexity_score += 2
        
        # Adjust for cost analysis
        if include_cost_analysis:
            complexity_score += 0.5
        
        # Convert back to enum
        final_complexity = min(ComplexityLevel.VERY_HIGH.value, max(ComplexityLevel.VERY_LOW.value, int(complexity_score)))
        return ComplexityLevel(final_complexity)
    
    def create_analysis_context(self, analysis_type: str, **kwargs) -> AnalysisContext:
        """
        Create analysis context from parameters.
        
        Args:
            analysis_type: Type of analysis
            **kwargs: Analysis parameters
            
        Returns:
            AnalysisContext object
        """
        # Estimate data size based on parameters
        bucket_count = len(kwargs.get("bucket_names", [])) if kwargs.get("bucket_names") else 10
        lookback_days = kwargs.get("lookback_days", 30)
        
        # Rough estimation of data size
        estimated_data_size_mb = bucket_count * lookback_days * 0.1  # 0.1MB per bucket per day
        
        # Determine complexity (remove bucket_count from kwargs to avoid conflict)
        context_kwargs = kwargs.copy()
        context_kwargs.pop('bucket_names', None)  # Remove to avoid conflicts
        complexity_level = self.get_complexity_level(analysis_type, bucket_count=bucket_count, **context_kwargs)
        
        return AnalysisContext(
            analysis_type=analysis_type,
            complexity_level=complexity_level,
            estimated_data_size_mb=estimated_data_size_mb,
            bucket_count=bucket_count,
            region=kwargs.get("region"),
            include_cost_analysis=kwargs.get("include_cost_analysis", True),
            lookback_days=lookback_days,
            parallel_execution=kwargs.get("parallel_execution", True),
            custom_parameters=kwargs
        )
    
    def get_timeout_for_analysis(self, analysis_type: str, **kwargs) -> float:
        """
        Convenience method to get timeout for an analysis.
        
        Args:
            analysis_type: Type of analysis
            **kwargs: Analysis parameters
            
        Returns:
            Calculated timeout in seconds
        """
        context = self.create_analysis_context(analysis_type, **kwargs)
        result = self.calculate_timeout(context)
        return result.final_timeout
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics and timeout effectiveness."""
        with self._lock:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "base_timeout": self.config.base_timeout,
                    "min_timeout": self.config.min_timeout,
                    "max_timeout": self.config.max_timeout,
                    "grace_period": self.config.grace_period
                },
                "historical_data": {},
                "system_load": {
                    "samples_count": len(self.system_load_samples),
                    "recent_avg_load": 0.0
                }
            }
            
            # Calculate historical statistics
            for analysis_type, history in self.performance_history.items():
                if history:
                    stats["historical_data"][analysis_type] = {
                        "execution_count": len(history),
                        "avg_execution_time": sum(history) / len(history),
                        "min_execution_time": min(history),
                        "max_execution_time": max(history),
                        "recent_executions": history[-10:] if len(history) >= 10 else history
                    }
            
            # Calculate recent system load
            if self.system_load_samples:
                cutoff_time = datetime.now() - timedelta(minutes=5)
                recent_loads = [
                    load for timestamp, load in self.system_load_samples 
                    if timestamp >= cutoff_time
                ]
                if recent_loads:
                    stats["system_load"]["recent_avg_load"] = sum(recent_loads) / len(recent_loads)
            
            return stats
    
    def optimize_configuration(self):
        """Optimize timeout configuration based on historical data."""
        with self._lock:
            if not self.performance_history:
                return
            
            # Analyze timeout effectiveness
            total_executions = sum(len(history) for history in self.performance_history.values())
            if total_executions < 50:  # Need sufficient data
                return
            
            # Calculate average execution times by complexity
            complexity_averages = {}
            for analysis_type, complexity_history in self.complexity_history.items():
                for complexity_level, execution_time in complexity_history:
                    if complexity_level not in complexity_averages:
                        complexity_averages[complexity_level] = []
                    complexity_averages[complexity_level].append(execution_time)
            
            # Update complexity multipliers based on actual performance
            for complexity_level, execution_times in complexity_averages.items():
                if len(execution_times) >= 10:  # Need sufficient samples
                    avg_time = sum(execution_times) / len(execution_times)
                    base_avg = sum(
                        sum(history) / len(history) 
                        for history in self.performance_history.values()
                    ) / len(self.performance_history)
                    
                    # Calculate optimal multiplier
                    optimal_multiplier = avg_time / base_avg if base_avg > 0 else 1.0
                    
                    # Gradually adjust current multiplier towards optimal
                    current_multiplier = self.config.complexity_multiplier[complexity_level]
                    new_multiplier = current_multiplier * 0.8 + optimal_multiplier * 0.2
                    
                    self.config.complexity_multiplier[complexity_level] = new_multiplier
                    
                    logger.info(
                        f"Optimized complexity multiplier for {complexity_level.name}: "
                        f"{current_multiplier:.2f} -> {new_multiplier:.2f}"
                    )
    
    def shutdown(self):
        """Shutdown the timeout handler."""
        logger.info("Shutting down ProgressiveTimeoutHandler")
        
        with self._lock:
            # Optionally save historical data for persistence
            self.performance_history.clear()
            self.complexity_history.clear()
            self.system_load_samples.clear()
        
        logger.info("ProgressiveTimeoutHandler shutdown complete")

# Global timeout handler instance
_timeout_handler = None

def get_timeout_handler() -> ProgressiveTimeoutHandler:
    """Get the global timeout handler instance."""
    global _timeout_handler
    if _timeout_handler is None:
        _timeout_handler = ProgressiveTimeoutHandler()
    return _timeout_handler