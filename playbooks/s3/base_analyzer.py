"""
Base Analyzer Interface for S3 Optimization

Abstract base class defining the interface for all S3 optimization analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Base class for all S3 analyzers."""
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None,
                 performance_monitor=None, memory_manager=None):
        """
        Initialize BaseAnalyzer with performance optimization components.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
            performance_monitor: Performance monitoring instance
            memory_manager: Memory management instance
        """
        self.s3_service = s3_service
        self.pricing_service = pricing_service
        self.storage_lens_service = storage_lens_service
        self.performance_monitor = performance_monitor
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis metadata
        self.analysis_type = self.__class__.__name__.replace('Analyzer', '').lower()
        self.version = "1.0.0"
        self.last_execution = None
        self.execution_count = 0
    
    @abstractmethod
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the analysis.
        
        Args:
            **kwargs: Analysis-specific parameters
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate input parameters for the analysis.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Common parameter validation
        region = kwargs.get('region')
        if region and not isinstance(region, str):
            validation_result["errors"].append("Region must be a string")
            validation_result["valid"] = False
        
        lookback_days = kwargs.get('lookback_days', 30)
        if not isinstance(lookback_days, int) or lookback_days <= 0:
            validation_result["errors"].append("lookback_days must be a positive integer")
            validation_result["valid"] = False
        elif lookback_days > 365:
            validation_result["warnings"].append("lookback_days > 365 may result in large datasets")
        
        bucket_names = kwargs.get('bucket_names')
        if bucket_names is not None:
            if not isinstance(bucket_names, list):
                validation_result["errors"].append("bucket_names must be a list")
                validation_result["valid"] = False
            elif not all(isinstance(name, str) for name in bucket_names):
                validation_result["errors"].append("All bucket names must be strings")
                validation_result["valid"] = False
        
        timeout_seconds = kwargs.get('timeout_seconds', 60)
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            validation_result["errors"].append("timeout_seconds must be a positive number")
            validation_result["valid"] = False
        
        return validation_result
    
    def prepare_analysis_context(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare analysis context with common parameters.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing analysis context
        """
        context = {
            "analysis_type": self.analysis_type,
            "analyzer_version": self.version,
            "region": kwargs.get('region'),
            "session_id": kwargs.get('session_id'),
            "lookback_days": kwargs.get('lookback_days', 30),
            "include_cost_analysis": kwargs.get('include_cost_analysis', True),
            "bucket_names": kwargs.get('bucket_names'),
            "timeout_seconds": kwargs.get('timeout_seconds', 60),
            "started_at": datetime.now().isoformat(),
            "execution_id": f"{self.analysis_type}_{int(datetime.now().timestamp())}"
        }
        
        return context
    
    def handle_analysis_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analysis errors with consistent error reporting.
        
        Args:
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing error information
        """
        error_message = str(error)
        self.logger.error(f"Analysis error in {self.analysis_type}: {error_message}")
        
        return {
            "status": "error",
            "analysis_type": self.analysis_type,
            "error_message": error_message,
            "error_type": error.__class__.__name__,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [{
                "type": "error_resolution",
                "priority": "high",
                "title": f"Analysis Error: {self.analysis_type}",
                "description": f"Analysis failed with error: {error_message}",
                "action_items": [
                    "Check AWS credentials and permissions",
                    "Verify network connectivity",
                    "Review input parameters",
                    "Check service quotas and limits"
                ]
            }]
        }
    
    def create_recommendation(self, 
                            rec_type: str,
                            priority: str,
                            title: str,
                            description: str,
                            potential_savings: Optional[float] = None,
                            implementation_effort: str = "medium",
                            affected_resources: Optional[List[str]] = None,
                            action_items: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a standardized recommendation dictionary.
        
        Args:
            rec_type: Type of recommendation
            priority: Priority level (high, medium, low)
            title: Recommendation title
            description: Detailed description
            potential_savings: Estimated cost savings
            implementation_effort: Implementation effort level
            affected_resources: List of affected resources
            action_items: List of action items
            
        Returns:
            Standardized recommendation dictionary
        """
        recommendation = {
            "type": rec_type,
            "priority": priority,
            "title": title,
            "description": description,
            "implementation_effort": implementation_effort,
            "analyzer": self.analysis_type,
            "created_at": datetime.now().isoformat()
        }
        
        if potential_savings is not None:
            recommendation["potential_savings"] = potential_savings
            recommendation["potential_savings_formatted"] = f"${potential_savings:.2f}"
        
        if affected_resources:
            recommendation["affected_resources"] = affected_resources
            recommendation["resource_count"] = len(affected_resources)
        
        if action_items:
            recommendation["action_items"] = action_items
        
        return recommendation
    
    def log_analysis_start(self, context: Dict[str, Any]):
        """Log analysis start with context and performance monitoring."""
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        self.logger.info(f"Starting {self.analysis_type} analysis (execution #{self.execution_count})")
        self.logger.debug(f"Analysis context: {context}")
        
        # Record performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"analyzer_{self.analysis_type}_started",
                1,
                tags={"analyzer": self.analysis_type, "execution": str(self.execution_count)}
            )
    
    def log_analysis_complete(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Log analysis completion with results summary and performance metrics."""
        status = result.get('status', 'unknown')
        execution_time = result.get('execution_time', 0)
        recommendation_count = len(result.get('recommendations', []))
        
        self.logger.info(
            f"Completed {self.analysis_type} analysis - "
            f"Status: {status}, Time: {execution_time:.2f}s, "
            f"Recommendations: {recommendation_count}"
        )
        
        # Record performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"analyzer_{self.analysis_type}_completed",
                1,
                tags={
                    "analyzer": self.analysis_type, 
                    "status": status,
                    "execution": str(self.execution_count)
                }
            )
            
            self.performance_monitor.record_metric(
                f"analyzer_{self.analysis_type}_execution_time",
                execution_time,
                tags={"analyzer": self.analysis_type}
            )
            
            self.performance_monitor.record_metric(
                f"analyzer_{self.analysis_type}_recommendations",
                recommendation_count,
                tags={"analyzer": self.analysis_type}
            )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about this analyzer."""
        return {
            "analysis_type": self.analysis_type,
            "class_name": self.__class__.__name__,
            "version": self.version,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "services": {
                "s3_service": self.s3_service is not None,
                "pricing_service": self.pricing_service is not None,
                "storage_lens_service": self.storage_lens_service is not None
            }
        }
    
    async def execute_with_error_handling(self, **kwargs) -> Dict[str, Any]:
        """
        Execute analysis with comprehensive error handling and performance monitoring.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing analysis results or error information
        """
        context = self.prepare_analysis_context(**kwargs)
        
        # Start memory tracking if memory manager is available
        memory_tracker_name = None
        if self.memory_manager:
            memory_tracker_name = f"analyzer_{self.analysis_type}_{int(datetime.now().timestamp())}"
            self.memory_manager.start_memory_tracking(memory_tracker_name)
        
        try:
            # Validate parameters
            validation = self.validate_parameters(**kwargs)
            if not validation["valid"]:
                error_result = {
                    "status": "error",
                    "analysis_type": self.analysis_type,
                    "error_message": "Parameter validation failed",
                    "validation_errors": validation["errors"],
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Stop memory tracking
                if memory_tracker_name and self.memory_manager:
                    memory_stats = self.memory_manager.stop_memory_tracking(memory_tracker_name)
                    if memory_stats:
                        error_result["memory_usage"] = memory_stats
                
                return error_result
            
            # Log warnings if any
            for warning in validation.get("warnings", []):
                self.logger.warning(f"Parameter warning: {warning}")
            
            # Log analysis start
            self.log_analysis_start(context)
            
            # Register large object for memory management if available
            if self.memory_manager:
                try:
                    self.memory_manager.register_large_object(
                        f"analysis_context_{self.analysis_type}_{int(datetime.now().timestamp())}",
                        context,
                        size_mb=0.1,  # Small object
                        cleanup_callback=lambda: self.logger.debug(f"Cleaned up {self.analysis_type} analysis context")
                    )
                except Exception as e:
                    self.logger.warning(f"Could not register large object with memory manager: {str(e)}")
            
            # Execute analysis
            result = await self.analyze(**kwargs)
            
            # Ensure result has required fields
            if "status" not in result:
                result["status"] = "success"
            if "analysis_type" not in result:
                result["analysis_type"] = self.analysis_type
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()
            
            # Generate recommendations if not present
            if "recommendations" not in result:
                result["recommendations"] = self.get_recommendations(result)
            
            # Add memory usage statistics
            if memory_tracker_name and self.memory_manager:
                memory_stats = self.memory_manager.stop_memory_tracking(memory_tracker_name)
                if memory_stats:
                    result["memory_usage"] = memory_stats
            
            # Log completion
            self.log_analysis_complete(context, result)
            
            return result
            
        except Exception as e:
            # Stop memory tracking on error
            if memory_tracker_name and self.memory_manager:
                memory_stats = self.memory_manager.stop_memory_tracking(memory_tracker_name)
            
            error_result = self.handle_analysis_error(e, context)
            
            # Add memory stats to error result if available
            if memory_tracker_name and self.memory_manager and 'memory_stats' in locals():
                error_result["memory_usage"] = memory_stats
            
            return error_result


class AnalyzerRegistry:
    """Registry for managing analyzer instances."""
    
    def __init__(self):
        self._analyzers: Dict[str, BaseAnalyzer] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, analyzer: BaseAnalyzer):
        """Register an analyzer instance."""
        analysis_type = analyzer.analysis_type
        self._analyzers[analysis_type] = analyzer
        self.logger.info(f"Registered analyzer: {analysis_type}")
    
    def get(self, analysis_type: str) -> Optional[BaseAnalyzer]:
        """Get an analyzer by type."""
        return self._analyzers.get(analysis_type)
    
    def list_analyzers(self) -> List[str]:
        """List all registered analyzer types."""
        return list(self._analyzers.keys())
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about all registered analyzers."""
        return {
            analysis_type: analyzer.get_analyzer_info()
            for analysis_type, analyzer in self._analyzers.items()
        }


# Global analyzer registry
_analyzer_registry = AnalyzerRegistry()

def get_analyzer_registry() -> AnalyzerRegistry:
    """Get the global analyzer registry."""
    return _analyzer_registry