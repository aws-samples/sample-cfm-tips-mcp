"""
Base Analyzer Interface for CloudWatch Optimization

Abstract base class defining the interface for all CloudWatch optimization analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Base class for all CloudWatch analyzers."""
    
    def __init__(self, cost_explorer_service=None, config_service=None, 
                 metrics_service=None, cloudwatch_service=None, pricing_service=None,
                 performance_monitor=None, memory_manager=None):
        """
        Initialize BaseAnalyzer with CloudWatch services and performance optimization components.
        
        Args:
            cost_explorer_service: CloudWatchCostExplorerService instance for no-cost data access
            config_service: CloudWatchConfigService instance for configuration operations
            metrics_service: CloudWatchMetricsService instance for minimal-cost metrics operations
            cloudwatch_service: CloudWatchService instance for enhanced CloudWatch operations
            pricing_service: CloudWatchPricing instance for cost calculations
            performance_monitor: Performance monitoring instance
            memory_manager: Memory management instance
        """
        self.cost_explorer_service = cost_explorer_service
        self.config_service = config_service
        self.metrics_service = metrics_service
        self.cloudwatch_service = cloudwatch_service
        self.pricing_service = pricing_service
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
        
        log_group_names = kwargs.get('log_group_names')
        if log_group_names is not None:
            if not isinstance(log_group_names, list):
                validation_result["errors"].append("log_group_names must be a list")
                validation_result["valid"] = False
            elif not all(isinstance(name, str) for name in log_group_names):
                validation_result["errors"].append("All log group names must be strings")
                validation_result["valid"] = False
        
        alarm_names = kwargs.get('alarm_names')
        if alarm_names is not None:
            if not isinstance(alarm_names, list):
                validation_result["errors"].append("alarm_names must be a list")
                validation_result["valid"] = False
            elif not all(isinstance(name, str) for name in alarm_names):
                validation_result["errors"].append("All alarm names must be strings")
                validation_result["valid"] = False
        
        dashboard_names = kwargs.get('dashboard_names')
        if dashboard_names is not None:
            if not isinstance(dashboard_names, list):
                validation_result["errors"].append("dashboard_names must be a list")
                validation_result["valid"] = False
            elif not all(isinstance(name, str) for name in dashboard_names):
                validation_result["errors"].append("All dashboard names must be strings")
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
            "log_group_names": kwargs.get('log_group_names'),
            "alarm_names": kwargs.get('alarm_names'),
            "dashboard_names": kwargs.get('dashboard_names'),
            "timeout_seconds": kwargs.get('timeout_seconds', 60),
            "started_at": datetime.now().isoformat(),
            "execution_id": f"{self.analysis_type}_{int(datetime.now().timestamp())}",
            "cost_constraints": {
                "prioritize_cost_explorer": True,
                "minimize_cloudwatch_api_costs": True,
                "track_cost_incurring_operations": True,
                "free_operations_enabled": True,
                "configuration_apis_allowed": True,
                "data_processing_apis_restricted": True
            }
        }
        
        return context
    
    def handle_analysis_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analysis errors with consistent error reporting and full exception capture.
        
        Args:
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing error information
        """
        import traceback
        
        # Capture full exception details
        error_message = str(error)
        full_traceback = traceback.format_exc()
        exception_chain = []
        
        # Capture exception chain for nested exceptions
        current_exception = error
        while current_exception is not None:
            exception_chain.append({
                "type": current_exception.__class__.__name__,
                "message": str(current_exception),
                "module": getattr(current_exception.__class__, '__module__', 'unknown')
            })
            current_exception = getattr(current_exception, '__cause__', None) or getattr(current_exception, '__context__', None)
        
        # Log full error details
        self.logger.error(f"Analysis error in {self.analysis_type}: {error_message}")
        self.logger.error(f"Full traceback: {full_traceback}")
        
        # Determine error category for CloudWatch-specific issues
        error_category = "general"
        if "permission" in error_message.lower() or "access" in error_message.lower():
            error_category = "permissions"
        elif "throttl" in error_message.lower() or "rate" in error_message.lower():
            error_category = "rate_limiting"
        elif "timeout" in error_message.lower():
            error_category = "timeout"
        elif "cost" in error_message.lower() or "billing" in error_message.lower():
            error_category = "cost_constraints"
        
        return {
            "status": "error",
            "analysis_type": self.analysis_type,
            "error_message": error_message,
            "error_type": error.__class__.__name__,
            "error_category": error_category,
            "full_exception_details": {
                "traceback": full_traceback,
                "exception_chain": exception_chain,
                "error_location": self._extract_error_location(full_traceback)
            },
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "cost_incurred": False,
            "cost_incurring_operations": [],
            "primary_data_source": "cost_explorer",
            "fallback_used": False,
            "recommendations": self._get_error_resolution_recommendations(error_category, error_message, full_traceback)
        }
    
    def _extract_error_location(self, traceback_str: str) -> Dict[str, Any]:
        """Extract error location information from traceback."""
        try:
            lines = traceback_str.strip().split('\n')
            # Find the last "File" line which indicates where the error occurred
            for line in reversed(lines):
                if line.strip().startswith('File "'):
                    # Extract file, line number, and function
                    import re
                    match = re.search(r'File "([^"]+)", line (\d+), in (.+)', line)
                    if match:
                        return {
                            "file": match.group(1),
                            "line": int(match.group(2)),
                            "function": match.group(3)
                        }
        except Exception:
            pass
        
        return {"file": "unknown", "line": 0, "function": "unknown"}
    
    def _get_error_resolution_recommendations(self, error_category: str, error_message: str, full_traceback: str = None) -> List[Dict[str, Any]]:
        """Generate error-specific resolution recommendations."""
        recommendations = []
        
        if error_category == "permissions":
            recommendations.append({
                "type": "permission_fix",
                "priority": "high",
                "title": f"CloudWatch Permissions Issue: {self.analysis_type}",
                "description": f"Analysis failed due to permission issues: {error_message}",
                "cloudwatch_component": self.analysis_type,
                "action_items": [
                    "Check IAM permissions for CloudWatch, Cost Explorer, and CloudWatch Logs",
                    "Verify AWS credentials are valid and not expired",
                    "Ensure required service permissions are granted",
                    "Check if MFA is required for API access",
                    "Verify region-specific permissions if using cross-region analysis"
                ]
            })
        elif error_category == "rate_limiting":
            recommendations.append({
                "type": "rate_limit_optimization",
                "priority": "medium",
                "title": f"CloudWatch API Rate Limiting: {self.analysis_type}",
                "description": f"Analysis was throttled due to API rate limits: {error_message}",
                "cloudwatch_component": self.analysis_type,
                "action_items": [
                    "Reduce the scope of analysis (fewer resources, shorter time range)",
                    "Increase timeout_seconds parameter to allow for retry delays",
                    "Run analysis during off-peak hours",
                    "Consider using Cost Explorer as primary data source to reduce API calls",
                    "Implement exponential backoff in retry logic"
                ]
            })
        elif error_category == "timeout":
            recommendations.append({
                "type": "timeout_optimization",
                "priority": "medium",
                "title": f"CloudWatch Analysis Timeout: {self.analysis_type}",
                "description": f"Analysis timed out during execution: {error_message}",
                "cloudwatch_component": self.analysis_type,
                "action_items": [
                    "Increase timeout_seconds parameter",
                    "Reduce lookback_days to limit data volume",
                    "Filter to specific resources if possible",
                    "Use Cost Explorer for historical data instead of CloudWatch APIs",
                    "Run analysis in smaller batches"
                ]
            })
        elif error_category == "cost_constraints":
            recommendations.append({
                "type": "cost_constraint_violation",
                "priority": "high",
                "title": f"Cost Constraint Violation: {self.analysis_type}",
                "description": f"Analysis violated cost constraints: {error_message}",
                "cloudwatch_component": self.analysis_type,
                "action_items": [
                    "Review cost-incurring operations in the analysis",
                    "Use Cost Explorer as primary data source",
                    "Avoid CloudWatch Logs Insights queries",
                    "Limit CloudWatch Metrics API calls to essential operations only",
                    "Consider using configuration-only analysis"
                ]
            })
        else:
            recommendations.append({
                "type": "error_resolution",
                "priority": "high",
                "title": f"CloudWatch Analysis Error: {self.analysis_type}",
                "description": f"Analysis failed with error: {error_message}",
                "cloudwatch_component": self.analysis_type,
                "action_items": [
                    "Check AWS credentials and permissions",
                    "Verify network connectivity to AWS services",
                    "Review input parameters for validity",
                    "Check CloudWatch service quotas and limits",
                    "Ensure Cost Explorer is available in the region"
                ]
            })
        
        return recommendations
    
    def create_recommendation(self, 
                            rec_type: str,
                            priority: str,
                            title: str,
                            description: str,
                            potential_savings: Optional[float] = None,
                            implementation_effort: str = "medium",
                            affected_resources: Optional[List[str]] = None,
                            action_items: Optional[List[str]] = None,
                            cloudwatch_component: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized CloudWatch recommendation dictionary.
        
        Args:
            rec_type: Type of recommendation
            priority: Priority level (high, medium, low)
            title: Recommendation title
            description: Detailed description
            potential_savings: Estimated cost savings
            implementation_effort: Implementation effort level
            affected_resources: List of affected resources
            action_items: List of action items
            cloudwatch_component: CloudWatch component (logs, metrics, alarms, dashboards)
            
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
        
        if cloudwatch_component:
            recommendation["cloudwatch_component"] = cloudwatch_component
        
        return recommendation
    
    def log_analysis_start(self, context: Dict[str, Any]):
        """Log analysis start with context and performance monitoring."""
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        self.logger.info(f"Starting {self.analysis_type} CloudWatch analysis (execution #{self.execution_count})")
        self.logger.debug(f"Analysis context: {context}")
        
        # Log cost constraint information
        cost_constraints = context.get('cost_constraints', {})
        if cost_constraints.get('prioritize_cost_explorer'):
            self.logger.info(f"Cost optimization: Prioritizing Cost Explorer for {self.analysis_type}")
        if cost_constraints.get('minimize_cloudwatch_api_costs'):
            self.logger.info(f"Cost optimization: Minimizing CloudWatch API costs for {self.analysis_type}")
        
        # Record performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"cloudwatch_analyzer_{self.analysis_type}_started",
                1,
                tags={"analyzer": self.analysis_type, "execution": str(self.execution_count)}
            )
    
    def log_analysis_complete(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Log analysis completion with results summary and performance metrics."""
        status = result.get('status', 'unknown')
        execution_time = result.get('execution_time', 0)
        recommendation_count = len(result.get('recommendations', []))
        cost_incurred = result.get('cost_incurred', False)
        cost_incurring_operations = result.get('cost_incurring_operations', [])
        primary_data_source = result.get('primary_data_source', 'unknown')
        
        self.logger.info(
            f"Completed {self.analysis_type} CloudWatch analysis - "
            f"Status: {status}, Time: {execution_time:.2f}s, "
            f"Recommendations: {recommendation_count}, "
            f"Cost incurred: {cost_incurred}, "
            f"Primary source: {primary_data_source}"
        )
        
        # Log cost-incurring operations if any
        if cost_incurring_operations:
            self.logger.warning(
                f"Cost-incurring operations in {self.analysis_type}: {cost_incurring_operations}"
            )
        
        # Record performance metrics if available
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"cloudwatch_analyzer_{self.analysis_type}_completed",
                1,
                tags={
                    "analyzer": self.analysis_type, 
                    "status": status,
                    "execution": str(self.execution_count),
                    "cost_incurred": str(cost_incurred),
                    "primary_data_source": primary_data_source
                }
            )
            
            self.performance_monitor.record_metric(
                f"cloudwatch_analyzer_{self.analysis_type}_execution_time",
                execution_time,
                tags={"analyzer": self.analysis_type}
            )
            
            self.performance_monitor.record_metric(
                f"cloudwatch_analyzer_{self.analysis_type}_recommendations",
                recommendation_count,
                tags={"analyzer": self.analysis_type}
            )
            
            if cost_incurring_operations:
                self.performance_monitor.record_metric(
                    f"cloudwatch_analyzer_{self.analysis_type}_cost_operations",
                    len(cost_incurring_operations),
                    tags={"analyzer": self.analysis_type}
                )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about this CloudWatch analyzer."""
        return {
            "analysis_type": self.analysis_type,
            "class_name": self.__class__.__name__,
            "version": self.version,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "services": {
                "cost_explorer_service": self.cost_explorer_service is not None,
                "config_service": self.config_service is not None,
                "metrics_service": self.metrics_service is not None,
                "cloudwatch_service": self.cloudwatch_service is not None,
                "pricing_service": self.pricing_service is not None
            },
            "cost_optimization": {
                "prioritizes_cost_explorer": True,
                "minimizes_api_costs": True,
                "tracks_cost_operations": True
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
            memory_tracker_name = f"cloudwatch_analyzer_{self.analysis_type}_{int(datetime.now().timestamp())}"
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
                    "timestamp": datetime.now().isoformat(),
                    "cost_incurred": False,
                    "cost_incurring_operations": [],
                    "primary_data_source": "cost_explorer",
                    "fallback_used": False
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
                        f"cloudwatch_analysis_context_{self.analysis_type}_{int(datetime.now().timestamp())}",
                        context,
                        size_mb=0.1,  # Small object
                        cleanup_callback=lambda: self.logger.debug(f"Cleaned up {self.analysis_type} analysis context")
                    )
                except Exception as e:
                    self.logger.warning(f"Could not register large object with memory manager: {str(e)}")
            
            # Execute analysis
            result = await self.analyze(**kwargs)
            
            # Ensure result has required CloudWatch-specific fields
            if "status" not in result:
                result["status"] = "success"
            if "analysis_type" not in result:
                result["analysis_type"] = self.analysis_type
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()
            if "cost_incurred" not in result:
                result["cost_incurred"] = False
            if "cost_incurring_operations" not in result:
                result["cost_incurring_operations"] = []
            if "primary_data_source" not in result:
                result["primary_data_source"] = "cost_explorer"
            if "fallback_used" not in result:
                result["fallback_used"] = False
            
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
    """Registry for managing CloudWatch analyzer instances."""
    
    def __init__(self):
        self._analyzers: Dict[str, BaseAnalyzer] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, analyzer: BaseAnalyzer):
        """Register a CloudWatch analyzer instance."""
        analysis_type = analyzer.analysis_type
        self._analyzers[analysis_type] = analyzer
        self.logger.info(f"Registered CloudWatch analyzer: {analysis_type}")
    
    def get(self, analysis_type: str) -> Optional[BaseAnalyzer]:
        """Get a CloudWatch analyzer by type."""
        return self._analyzers.get(analysis_type)
    
    def list_analyzers(self) -> List[str]:
        """List all registered CloudWatch analyzer types."""
        return list(self._analyzers.keys())
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about all registered CloudWatch analyzers."""
        return {
            analysis_type: analyzer.get_analyzer_info()
            for analysis_type, analyzer in self._analyzers.items()
        }


# Global CloudWatch analyzer registry
_cloudwatch_analyzer_registry = AnalyzerRegistry()

def get_analyzer_registry() -> AnalyzerRegistry:
    """Get the global CloudWatch analyzer registry."""
    return _cloudwatch_analyzer_registry