"""
S3 Analysis Engine for CFM Tips MCP Server

Core analysis engine that coordinates all S3 analyzers with parallel execution
and session-sql integration for comprehensive S3 optimization analysis.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .base_analyzer import BaseAnalyzer, get_analyzer_registry
from .analyzers.general_spend_analyzer import GeneralSpendAnalyzer
from .analyzers.storage_class_analyzer import StorageClassAnalyzer
from .analyzers.archive_optimization_analyzer import ArchiveOptimizationAnalyzer
from .analyzers.api_cost_analyzer import ApiCostAnalyzer
from .analyzers.multipart_cleanup_analyzer import MultipartCleanupAnalyzer
from .analyzers.governance_analyzer import GovernanceAnalyzer
from services.s3_service import S3Service
from services.s3_pricing import S3Pricing
from services.storage_lens_service import StorageLensService

logger = logging.getLogger(__name__)


class S3AnalysisEngine:
    """
    Core analysis engine that coordinates all S3 analyzers with parallel execution.
    
    Provides:
    - Analyzer registry and dynamic loading
    - Task prioritization based on cost impact and execution time
    - Session-sql integration for storing analysis results
    - Comprehensive result aggregation and cross-analysis insights
    """
    
    def __init__(self, region: Optional[str] = None, 
                 performance_monitor=None, memory_manager=None, timeout_handler=None,
                 pricing_cache=None, bucket_metadata_cache=None, analysis_results_cache=None):
        """
        Initialize S3AnalysisEngine with performance optimization components.
        
        Args:
            region: AWS region for S3 operations
            performance_monitor: Performance monitoring instance
            memory_manager: Memory management instance
            timeout_handler: Timeout handling instance
            pricing_cache: Pricing data cache
            bucket_metadata_cache: Bucket metadata cache
            analysis_results_cache: Analysis results cache
        """
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization components
        self.performance_monitor = performance_monitor
        self.memory_manager = memory_manager
        self.timeout_handler = timeout_handler
        self.pricing_cache = pricing_cache
        self.bucket_metadata_cache = bucket_metadata_cache
        self.analysis_results_cache = analysis_results_cache
        
        # Initialize services (will enhance them later with cache integration)
        self.s3_service = S3Service(region=region)
        self.pricing_service = S3Pricing(region=region)
        self.storage_lens_service = StorageLensService(region=region)
        
        # Add performance optimization components as attributes for future enhancement
        if bucket_metadata_cache:
            self.s3_service.bucket_metadata_cache = bucket_metadata_cache
        if performance_monitor:
            self.s3_service.performance_monitor = performance_monitor
        if pricing_cache:
            self.pricing_service.pricing_cache = pricing_cache
        if performance_monitor:
            self.pricing_service.performance_monitor = performance_monitor
            self.storage_lens_service.performance_monitor = performance_monitor
        
        # Initialize analyzer registry
        self.analyzer_registry = get_analyzer_registry()
        self._initialize_analyzers()
        
        # Analysis metadata
        self.analysis_priorities = self._define_analysis_priorities()
        self.execution_history = []
        
        logger.info(f"S3AnalysisEngine initialized with performance optimizations for region: {region or 'default'}")
    
    def _validate_analyzer(self, analyzer) -> bool:
        """
        Validate an analyzer instance before registration.
        
        Args:
            analyzer: Analyzer instance to validate
            
        Returns:
            True if analyzer is valid, False otherwise
        """
        try:
            # Check if analyzer has required methods
            required_methods = ['analyze', 'get_recommendations', 'execute_with_error_handling']
            for method_name in required_methods:
                if not hasattr(analyzer, method_name) or not callable(getattr(analyzer, method_name)):
                    logger.error(f"Analyzer {analyzer.__class__.__name__} missing required method: {method_name}")
                    return False
            
            # Check if analyzer has required services
            required_services = ['s3_service', 'pricing_service', 'storage_lens_service']
            for service_name in required_services:
                if not hasattr(analyzer, service_name):
                    logger.error(f"Analyzer {analyzer.__class__.__name__} missing required service: {service_name}")
                    return False
            
            # Check if analyzer has analysis_type
            if not hasattr(analyzer, 'analysis_type') or not analyzer.analysis_type:
                logger.error(f"Analyzer {analyzer.__class__.__name__} missing analysis_type")
                return False
            
            # Validate analysis_type format
            if not isinstance(analyzer.analysis_type, str) or not analyzer.analysis_type.replace('_', '').isalnum():
                logger.error(f"Analyzer {analyzer.__class__.__name__} has invalid analysis_type: {analyzer.analysis_type}")
                return False
            
            logger.debug(f"Analyzer validation passed for: {analyzer.analysis_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating analyzer {analyzer.__class__.__name__}: {str(e)}")
            return False
    
    def reload_analyzer(self, analysis_type: str) -> Dict[str, Any]:
        """
        Reload a specific analyzer with enhanced error handling.
        
        Args:
            analysis_type: Type of analyzer to reload
            
        Returns:
            Dictionary containing reload results
        """
        try:
            logger.info(f"Reloading analyzer: {analysis_type}")
            
            # Check if analyzer exists
            current_analyzer = self.analyzer_registry.get(analysis_type)
            if not current_analyzer:
                return {
                    "status": "error",
                    "message": f"Analyzer not found: {analysis_type}",
                    "available_analyzers": self.analyzer_registry.list_analyzers()
                }
            
            # Get analyzer class from current instance
            analyzer_class = current_analyzer.__class__
            
            # Create new instance
            new_analyzer = analyzer_class(
                s3_service=self.s3_service,
                pricing_service=self.pricing_service,
                storage_lens_service=self.storage_lens_service
            )
            
            # Add performance optimization components
            if self.performance_monitor:
                new_analyzer.performance_monitor = self.performance_monitor
            if self.memory_manager:
                new_analyzer.memory_manager = self.memory_manager
            if self.timeout_handler:
                new_analyzer.timeout_handler = self.timeout_handler
            
            # Add cache references
            if self.pricing_cache:
                new_analyzer.pricing_cache = self.pricing_cache
            if self.bucket_metadata_cache:
                new_analyzer.bucket_metadata_cache = self.bucket_metadata_cache
            if self.analysis_results_cache:
                new_analyzer.analysis_results_cache = self.analysis_results_cache
            
            # Validate new analyzer
            if not self._validate_analyzer(new_analyzer):
                return {
                    "status": "error",
                    "message": f"New analyzer instance failed validation: {analysis_type}",
                    "analyzer_class": analyzer_class.__name__
                }
            
            # Replace in registry
            self.analyzer_registry.register(new_analyzer)
            
            reload_result = {
                "status": "success",
                "message": f"Successfully reloaded analyzer: {analysis_type}",
                "analyzer_class": analyzer_class.__name__,
                "old_execution_count": current_analyzer.execution_count,
                "new_execution_count": new_analyzer.execution_count,
                "reloaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully reloaded analyzer: {analysis_type}")
            return reload_result
            
        except Exception as e:
            logger.error(f"Error reloading analyzer {analysis_type}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to reload analyzer {analysis_type}: {str(e)}",
                "error_type": e.__class__.__name__,
                "reloaded_at": datetime.now().isoformat()
            }
    
    def get_analyzer_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all analyzers.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            health_status = {
                "overall_status": "healthy",
                "total_analyzers": len(self.analyzer_registry.list_analyzers()),
                "healthy_analyzers": 0,
                "unhealthy_analyzers": 0,
                "analyzer_details": {},
                "loading_results": getattr(self, 'analyzer_loading_results', {}),
                "checked_at": datetime.now().isoformat()
            }
            
            # Check each analyzer
            for analysis_type in self.analyzer_registry.list_analyzers():
                analyzer = self.analyzer_registry.get(analysis_type)
                
                analyzer_health = {
                    "status": "healthy",
                    "class_name": analyzer.__class__.__name__,
                    "execution_count": analyzer.execution_count,
                    "last_execution": analyzer.last_execution.isoformat() if analyzer.last_execution else None,
                    "services_available": {
                        "s3_service": analyzer.s3_service is not None,
                        "pricing_service": analyzer.pricing_service is not None,
                        "storage_lens_service": analyzer.storage_lens_service is not None
                    },
                    "performance_components": {
                        "performance_monitor": hasattr(analyzer, 'performance_monitor') and analyzer.performance_monitor is not None,
                        "memory_manager": hasattr(analyzer, 'memory_manager') and analyzer.memory_manager is not None,
                        "timeout_handler": hasattr(analyzer, 'timeout_handler') and analyzer.timeout_handler is not None
                    },
                    "cache_components": {
                        "pricing_cache": hasattr(analyzer, 'pricing_cache') and analyzer.pricing_cache is not None,
                        "bucket_metadata_cache": hasattr(analyzer, 'bucket_metadata_cache') and analyzer.bucket_metadata_cache is not None,
                        "analysis_results_cache": hasattr(analyzer, 'analysis_results_cache') and analyzer.analysis_results_cache is not None
                    },
                    "issues": []
                }
                
                # Check for issues
                if not analyzer.s3_service:
                    analyzer_health["issues"].append("Missing S3 service")
                    analyzer_health["status"] = "unhealthy"
                
                if not analyzer.pricing_service:
                    analyzer_health["issues"].append("Missing pricing service")
                    analyzer_health["status"] = "unhealthy"
                
                if not analyzer.storage_lens_service:
                    analyzer_health["issues"].append("Missing Storage Lens service")
                    analyzer_health["status"] = "unhealthy"
                
                # Validate analyzer
                if not self._validate_analyzer(analyzer):
                    analyzer_health["issues"].append("Failed validation check")
                    analyzer_health["status"] = "unhealthy"
                
                # Update counters
                if analyzer_health["status"] == "healthy":
                    health_status["healthy_analyzers"] += 1
                else:
                    health_status["unhealthy_analyzers"] += 1
                    health_status["overall_status"] = "degraded"
                
                health_status["analyzer_details"][analysis_type] = analyzer_health
            
            # Determine overall status
            if health_status["unhealthy_analyzers"] == health_status["total_analyzers"]:
                health_status["overall_status"] = "critical"
            elif health_status["unhealthy_analyzers"] > 0:
                health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting analyzer health status: {str(e)}")
            return {
                "overall_status": "error",
                "error_message": str(e),
                "checked_at": datetime.now().isoformat()
            }
    
    def _initialize_analyzers(self):
        """Initialize and register all S3 analyzers with enhanced error handling and dynamic loading."""
        try:
            # Define analyzer classes for dynamic loading
            analyzer_classes = [
                ('general_spend', GeneralSpendAnalyzer),
                ('storage_class', StorageClassAnalyzer),
                ('archive_optimization', ArchiveOptimizationAnalyzer),
                ('api_cost', ApiCostAnalyzer),
                ('multipart_cleanup', MultipartCleanupAnalyzer),
                ('governance', GovernanceAnalyzer)
            ]
            
            successfully_loaded = []
            failed_to_load = []
            
            # Dynamic analyzer loading with individual error handling
            for analysis_type, analyzer_class in analyzer_classes:
                try:
                    # Create analyzer instance with shared services
                    analyzer = analyzer_class(
                        s3_service=self.s3_service,
                        pricing_service=self.pricing_service,
                        storage_lens_service=self.storage_lens_service
                    )
                    
                    # Add performance optimization components
                    if self.performance_monitor:
                        analyzer.performance_monitor = self.performance_monitor
                    if self.memory_manager:
                        analyzer.memory_manager = self.memory_manager
                    if self.timeout_handler:
                        analyzer.timeout_handler = self.timeout_handler
                    
                    # Add cache references for enhanced performance
                    if self.pricing_cache:
                        analyzer.pricing_cache = self.pricing_cache
                    if self.bucket_metadata_cache:
                        analyzer.bucket_metadata_cache = self.bucket_metadata_cache
                    if self.analysis_results_cache:
                        analyzer.analysis_results_cache = self.analysis_results_cache
                    
                    # Validate analyzer before registration
                    if self._validate_analyzer(analyzer):
                        self.analyzer_registry.register(analyzer)
                        successfully_loaded.append(analysis_type)
                        logger.info(f"Successfully loaded and registered analyzer: {analysis_type}")
                    else:
                        failed_to_load.append((analysis_type, "Validation failed"))
                        logger.warning(f"Analyzer validation failed for: {analysis_type}")
                        
                except Exception as e:
                    failed_to_load.append((analysis_type, str(e)))
                    logger.error(f"Failed to load analyzer {analysis_type}: {str(e)}")
                    # Continue loading other analyzers even if one fails
                    continue
            
            # Log initialization summary
            logger.info(f"Analyzer initialization complete - Successfully loaded: {len(successfully_loaded)}, Failed: {len(failed_to_load)}")
            
            if successfully_loaded:
                logger.info(f"Available analyzers: {', '.join(successfully_loaded)}")
            
            if failed_to_load:
                logger.warning(f"Failed analyzers: {', '.join([f'{name} ({error})' for name, error in failed_to_load])}")
            
            # Ensure at least some analyzers loaded successfully
            if not successfully_loaded:
                raise RuntimeError("No analyzers could be loaded successfully")
            
            # Store loading results for diagnostics
            self.analyzer_loading_results = {
                "successful": successfully_loaded,
                "failed": failed_to_load,
                "total_attempted": len(analyzer_classes),
                "success_rate": len(successfully_loaded) / len(analyzer_classes) * 100,
                "loaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Critical error during analyzer initialization: {str(e)}")
            # Set minimal loading results for error case
            self.analyzer_loading_results = {
                "successful": [],
                "failed": [("initialization", str(e))],
                "total_attempted": 0,
                "success_rate": 0,
                "loaded_at": datetime.now().isoformat(),
                "critical_error": True
            }
            raise
    
    def _define_analysis_priorities(self) -> Dict[str, Dict[str, Any]]:
        """
        Define analysis priorities based on cost impact and execution time.
        
        Returns:
            Dictionary mapping analysis types to priority information
        """
        return {
            "general_spend": {
                "priority": 5,  # Highest priority - provides overall cost baseline
                "cost_impact": "high",
                "execution_time_estimate": 30,  # seconds
                "dependencies": [],
                "description": "Comprehensive S3 spending analysis"
            },
            "storage_class": {
                "priority": 4,  # High priority - direct cost optimization
                "cost_impact": "high", 
                "execution_time_estimate": 45,
                "dependencies": ["general_spend"],
                "description": "Storage class optimization analysis"
            },
            "multipart_cleanup": {
                "priority": 4,  # High priority - immediate cost savings
                "cost_impact": "medium",
                "execution_time_estimate": 20,
                "dependencies": [],
                "description": "Incomplete multipart upload cleanup"
            },
            "archive_optimization": {
                "priority": 3,  # Medium priority - long-term savings
                "cost_impact": "medium",
                "execution_time_estimate": 35,
                "dependencies": ["storage_class"],
                "description": "Long-term archive optimization"
            },
            "api_cost": {
                "priority": 2,  # Lower priority - optimization opportunity
                "cost_impact": "low",
                "execution_time_estimate": 25,
                "dependencies": [],
                "description": "API request cost optimization"
            },
            "governance": {
                "priority": 1,  # Lowest priority - compliance focused
                "cost_impact": "low",
                "execution_time_estimate": 15,
                "dependencies": [],
                "description": "S3 governance and compliance checks"
            }
        }
    
    def get_available_analyses(self) -> List[Dict[str, Any]]:
        """
        Get list of available analyses with metadata.
        
        Returns:
            List of analysis information dictionaries
        """
        available_analyses = []
        
        for analysis_type in self.analyzer_registry.list_analyzers():
            analyzer = self.analyzer_registry.get(analysis_type)
            priority_info = self.analysis_priorities.get(analysis_type, {})
            
            analysis_info = {
                "analysis_type": analysis_type,
                "analyzer_class": analyzer.__class__.__name__,
                "priority": priority_info.get("priority", 1),
                "cost_impact": priority_info.get("cost_impact", "unknown"),
                "execution_time_estimate": priority_info.get("execution_time_estimate", 30),
                "dependencies": priority_info.get("dependencies", []),
                "description": priority_info.get("description", "S3 optimization analysis"),
                "analyzer_info": analyzer.get_analyzer_info()
            }
            
            available_analyses.append(analysis_info)
        
        # Sort by priority (highest first)
        available_analyses.sort(key=lambda x: x["priority"], reverse=True)
        
        return available_analyses
    
    async def run_analysis(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific analysis type with comprehensive error handling and fallback coordination.
        
        Args:
            analysis_type: Type of analysis to run
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        execution_id = f"{analysis_type}_{int(start_time)}"
        
        try:
            # Get analyzer with fallback handling
            analyzer = self.analyzer_registry.get(analysis_type)
            if not analyzer:
                # Attempt to reload analyzer if not found
                logger.warning(f"Analyzer not found for {analysis_type}, attempting to reload")
                reload_result = self.reload_analyzer(analysis_type)
                
                if reload_result.get("status") == "success":
                    analyzer = self.analyzer_registry.get(analysis_type)
                    logger.info(f"Successfully reloaded analyzer for {analysis_type}")
                else:
                    return {
                        "status": "error",
                        "analysis_type": analysis_type,
                        "message": f"Analyzer not found and reload failed: {analysis_type}",
                        "available_types": self.analyzer_registry.list_analyzers(),
                        "reload_attempt": reload_result,
                        "execution_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    }
            
            logger.info(f"Running {analysis_type} analysis (execution_id: {execution_id})")
            
            # Pre-execution health check
            if not self._validate_analyzer(analyzer):
                logger.error(f"Analyzer validation failed for {analysis_type}")
                return {
                    "status": "error",
                    "analysis_type": analysis_type,
                    "message": f"Analyzer validation failed: {analysis_type}",
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add execution context to kwargs
            kwargs['execution_id'] = execution_id
            kwargs['engine_region'] = self.region
            
            # Execute analysis with comprehensive error handling and retry logic
            result = await self._execute_with_fallback_coordination(analyzer, analysis_type, **kwargs)
            
            # Add execution metadata
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["execution_id"] = execution_id
            result["engine_metadata"] = {
                "analysis_engine": "S3AnalysisEngine",
                "analyzer_class": analyzer.__class__.__name__,
                "priority": self.analysis_priorities.get(analysis_type, {}).get("priority", 1),
                "region": self.region,
                "execution_id": execution_id,
                "fallback_used": result.get("fallback_used", False),
                "retry_count": result.get("retry_count", 0)
            }
            
            # Record execution history with enhanced metadata
            execution_record = {
                "analysis_type": analysis_type,
                "execution_id": execution_id,
                "status": result.get("status"),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "fallback_used": result.get("fallback_used", False),
                "retry_count": result.get("retry_count", 0),
                "error_message": result.get("error_message") if result.get("status") == "error" else None
            }
            
            self.execution_history.append(execution_record)
            
            # Keep execution history manageable (last 100 executions)
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            logger.info(f"Completed {analysis_type} analysis in {execution_time:.2f}s (status: {result.get('status')})")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Critical error running {analysis_type} analysis: {str(e)}")
            
            # Record failed execution
            execution_record = {
                "analysis_type": analysis_type,
                "execution_id": execution_id,
                "status": "critical_error",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "error_type": e.__class__.__name__
            }
            
            self.execution_history.append(execution_record)
            
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "execution_id": execution_id,
                "message": f"Critical analysis execution failure: {str(e)}",
                "error_type": e.__class__.__name__,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "engine_metadata": {
                    "analysis_engine": "S3AnalysisEngine",
                    "region": self.region,
                    "critical_error": True
                }
            }
    
    async def _execute_with_fallback_coordination(self, analyzer, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute analysis with fallback coordination and retry logic.
        
        Args:
            analyzer: Analyzer instance to execute
            analysis_type: Type of analysis
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        max_retries = kwargs.get('max_retries', 2)
        retry_count = 0
        last_error = None
        fallback_used = False
        
        while retry_count <= max_retries:
            try:
                # Execute analysis
                result = await analyzer.execute_with_error_handling(**kwargs)
                
                # Check if result indicates a recoverable error
                if result.get("status") == "error" and retry_count < max_retries:
                    error_message = result.get("error_message", "")
                    
                    # Determine if error is recoverable
                    if self._is_recoverable_error(error_message):
                        retry_count += 1
                        logger.warning(f"Recoverable error in {analysis_type}, retry {retry_count}/{max_retries}: {error_message}")
                        
                        # Apply fallback strategies
                        fallback_applied = await self._apply_fallback_strategies(analyzer, analysis_type, error_message, **kwargs)
                        if fallback_applied:
                            fallback_used = True
                        
                        # Wait before retry with exponential backoff
                        wait_time = min(2 ** retry_count, 10)  # Max 10 seconds
                        await asyncio.sleep(wait_time)
                        continue
                
                # Add retry metadata to successful or final result
                result["retry_count"] = retry_count
                result["fallback_used"] = fallback_used
                
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    logger.warning(f"Exception in {analysis_type}, retry {retry_count}/{max_retries}: {str(e)}")
                    
                    # Apply fallback strategies for exceptions
                    fallback_applied = await self._apply_fallback_strategies(analyzer, analysis_type, str(e), **kwargs)
                    if fallback_applied:
                        fallback_used = True
                    
                    # Wait before retry
                    wait_time = min(2 ** retry_count, 10)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for {analysis_type}: {str(e)}")
                    break
        
        # All retries exhausted, return error result
        return {
            "status": "error",
            "analysis_type": analysis_type,
            "message": f"Analysis failed after {max_retries} retries: {str(last_error)}",
            "error_message": str(last_error),
            "error_type": last_error.__class__.__name__ if last_error else "Unknown",
            "retry_count": retry_count,
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat()
        }
    
    def _is_recoverable_error(self, error_message: str) -> bool:
        """
        Determine if an error is recoverable and worth retrying.
        
        Args:
            error_message: Error message to analyze
            
        Returns:
            True if error is recoverable, False otherwise
        """
        recoverable_patterns = [
            "timeout",
            "throttl",
            "rate limit",
            "temporary",
            "connection",
            "network",
            "service unavailable",
            "internal server error",
            "502",
            "503",
            "504"
        ]
        
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in recoverable_patterns)
    
    async def _apply_fallback_strategies(self, analyzer, analysis_type: str, error_message: str, **kwargs) -> bool:
        """
        Apply fallback strategies based on error type.
        
        Args:
            analyzer: Analyzer instance
            analysis_type: Type of analysis
            error_message: Error message
            **kwargs: Analysis parameters
            
        Returns:
            True if fallback was applied, False otherwise
        """
        try:
            error_lower = error_message.lower()
            fallback_applied = False
            
            # Timeout fallback: reduce scope
            if "timeout" in error_lower:
                if kwargs.get('lookback_days', 30) > 7:
                    kwargs['lookback_days'] = max(7, kwargs.get('lookback_days', 30) // 2)
                    logger.info(f"Applied timeout fallback: reduced lookback_days to {kwargs['lookback_days']}")
                    fallback_applied = True
                
                if kwargs.get('bucket_names') and len(kwargs['bucket_names']) > 10:
                    kwargs['bucket_names'] = kwargs['bucket_names'][:10]
                    logger.info(f"Applied timeout fallback: limited bucket_names to {len(kwargs['bucket_names'])}")
                    fallback_applied = True
            
            # Rate limit fallback: add delays and reduce concurrency
            elif "throttl" in error_lower or "rate limit" in error_lower:
                # Add delay for rate limiting
                await asyncio.sleep(5)
                logger.info("Applied rate limit fallback: added delay")
                fallback_applied = True
            
            # Permission fallback: try alternative data sources
            elif "permission" in error_lower or "access" in error_lower:
                # This would be handled by individual analyzers
                logger.info("Permission error detected - analyzers should handle data source fallbacks")
                fallback_applied = True
            
            # Service unavailable fallback: reduce request complexity
            elif "service unavailable" in error_lower or "internal server error" in error_lower:
                if kwargs.get('include_cost_analysis', True):
                    kwargs['include_cost_analysis'] = False
                    logger.info("Applied service fallback: disabled cost analysis")
                    fallback_applied = True
            
            return fallback_applied
            
        except Exception as e:
            logger.error(f"Error applying fallback strategies: {str(e)}")
            return False
    
    def create_parallel_analysis_tasks(self, 
                                     analysis_types: List[str], 
                                     **kwargs) -> List[Dict[str, Any]]:
        """
        Create parallel analysis tasks with advanced prioritization based on cost impact, execution time, and dependencies.
        
        Args:
            analysis_types: List of analysis types to execute
            **kwargs: Analysis parameters
            
        Returns:
            List of task definitions for parallel execution with enhanced prioritization
        """
        tasks = []
        
        # Filter and validate analysis types
        valid_types = []
        for analysis_type in analysis_types:
            if self.analyzer_registry.get(analysis_type):
                valid_types.append(analysis_type)
            else:
                logger.warning(f"Skipping unknown analysis type: {analysis_type}")
        
        # Advanced prioritization with dependency resolution
        priority_sorted = self._calculate_advanced_task_priority(valid_types, **kwargs)
        
        logger.info(f"Advanced task prioritization order: {[item['analysis_type'] for item in priority_sorted]}")
        
        # Create tasks with enhanced metadata for parallel execution
        for i, priority_item in enumerate(priority_sorted):
            analysis_type = priority_item["analysis_type"]
            priority_info = self.analysis_priorities.get(analysis_type, {})
            
            # Create synchronous analysis function for parallel executor compatibility
            def create_analysis_function(atype=analysis_type, params=kwargs.copy(), order=i+1):
                def sync_analysis_wrapper():
                    try:
                        # Run async analysis in sync context for parallel executor
                        import asyncio
                        try:
                            # Get or create event loop
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            # Run the analysis
                            if loop.is_running():
                                # If loop is already running, use thread executor
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, self.run_analysis(atype, **params))
                                    result = future.result(timeout=params.get("timeout_seconds", 60))
                            else:
                                result = loop.run_until_complete(self.run_analysis(atype, **params))
                            
                            # Ensure result is properly formatted for session storage
                            if result.get("status") == "success":
                                result["session_storage_ready"] = True
                                result["analysis_metadata"] = {
                                    "priority": priority_info.get("priority", 1),
                                    "cost_impact": priority_info.get("cost_impact", "unknown"),
                                    "execution_order": order,
                                    "priority_score": priority_item.get("priority_score", 0),
                                    "dependency_level": priority_item.get("dependency_level", 0)
                                }
                            return result
                        except Exception as e:
                            logger.error(f"Analysis task {atype} failed: {str(e)}")
                            return {
                                "status": "error",
                                "analysis_type": atype,
                                "error_message": str(e),
                                "timestamp": datetime.now().isoformat(),
                                "execution_order": order
                            }
                    except Exception as e:
                        logger.error(f"Critical error in analysis wrapper for {atype}: {str(e)}")
                        return {
                            "status": "error",
                            "analysis_type": atype,
                            "error_message": f"Critical wrapper error: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }
                return sync_analysis_wrapper
            
            # Calculate dynamic timeout based on priority and estimated execution time
            base_timeout = priority_info.get("execution_time_estimate", 30)
            priority_multiplier = 1.0 + (priority_info.get("priority", 1) * 0.2)  # Higher priority gets more time
            dependency_multiplier = 1.0 + (priority_item.get("dependency_level", 0) * 0.1)  # Dependencies get more time
            dynamic_timeout = int(base_timeout * priority_multiplier * dependency_multiplier) + 15  # Add buffer
            
            task_def = {
                "service": "s3_analysis_engine",
                "operation": analysis_type,
                "function": create_analysis_function(),
                "args": (),
                "kwargs": {},  # Parameters are captured in the function
                "timeout": kwargs.get("timeout_seconds", dynamic_timeout),
                "priority": priority_item.get("priority_score", priority_info.get("priority", 1)),
                "metadata": {
                    "analysis_type": analysis_type,
                    "cost_impact": priority_info.get("cost_impact", "unknown"),
                    "execution_time_estimate": priority_info.get("execution_time_estimate", 30),
                    "execution_order": i + 1,
                    "dependencies": priority_info.get("dependencies", []),
                    "description": priority_info.get("description", "S3 optimization analysis"),
                    "dynamic_timeout": dynamic_timeout,
                    "priority_multiplier": priority_multiplier,
                    "dependency_multiplier": dependency_multiplier,
                    "priority_score": priority_item.get("priority_score", 0),
                    "dependency_level": priority_item.get("dependency_level", 0)
                }
            }
            
            tasks.append(task_def)
        
        logger.info(f"Created {len(tasks)} advanced prioritized parallel analysis tasks with dynamic timeouts")
        return tasks
    
    def _calculate_advanced_task_priority(self, analysis_types: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Calculate advanced task priority considering cost impact, execution time, dependencies, and context.
        
        Args:
            analysis_types: List of analysis types to prioritize
            **kwargs: Analysis parameters that might affect prioritization
            
        Returns:
            List of analysis types with priority information, sorted by priority
        """
        priority_items = []
        
        for analysis_type in analysis_types:
            priority_info = self.analysis_priorities.get(analysis_type, {})
            
            # Base priority score
            base_priority = priority_info.get("priority", 1)
            
            # Cost impact multiplier
            cost_impact = priority_info.get("cost_impact", "unknown")
            cost_multiplier = {"high": 3.0, "medium": 2.0, "low": 1.0, "unknown": 1.0}.get(cost_impact, 1.0)
            
            # Execution time efficiency (shorter is better for same priority)
            execution_time = priority_info.get("execution_time_estimate", 30)
            time_efficiency = max(0.1, 1.0 - (execution_time / 120.0))  # Normalize to 0.1-1.0 range
            
            # Dependency level (analyses with no dependencies get higher priority)
            dependencies = priority_info.get("dependencies", [])
            dependency_level = len(dependencies)
            dependency_penalty = max(0.1, 1.0 - (dependency_level * 0.2))
            
            # Context-based adjustments
            context_multiplier = 1.0
            
            # If specific bucket names are provided, prioritize governance and multipart cleanup
            if kwargs.get("bucket_names"):
                if analysis_type in ["governance", "multipart_cleanup"]:
                    context_multiplier += 0.5
            
            # If cost analysis is disabled, deprioritize cost-heavy analyses
            if not kwargs.get("include_cost_analysis", True):
                if analysis_type in ["general_spend", "storage_class"]:
                    context_multiplier -= 0.3
            
            # If short lookback period, prioritize faster analyses
            lookback_days = kwargs.get("lookback_days", 30)
            if lookback_days <= 7:
                if execution_time <= 20:
                    context_multiplier += 0.2
            
            # Calculate final priority score
            priority_score = (
                base_priority * 
                cost_multiplier * 
                time_efficiency * 
                dependency_penalty * 
                context_multiplier
            )
            
            priority_items.append({
                "analysis_type": analysis_type,
                "priority_score": priority_score,
                "base_priority": base_priority,
                "cost_impact": cost_impact,
                "cost_multiplier": cost_multiplier,
                "execution_time": execution_time,
                "time_efficiency": time_efficiency,
                "dependency_level": dependency_level,
                "dependency_penalty": dependency_penalty,
                "context_multiplier": context_multiplier,
                "dependencies": dependencies
            })
        
        # Sort by priority score (highest first), then by execution time (shortest first)
        priority_items.sort(
            key=lambda x: (x["priority_score"], -x["execution_time"]), 
            reverse=True
        )
        
        # Log prioritization details
        logger.info("Advanced task prioritization details:")
        for i, item in enumerate(priority_items):
            logger.info(
                f"  {i+1}. {item['analysis_type']}: "
                f"score={item['priority_score']:.2f}, "
                f"base={item['base_priority']}, "
                f"cost={item['cost_impact']}, "
                f"time={item['execution_time']}s, "
                f"deps={item['dependency_level']}"
            )
        
        return priority_items
    
    def aggregate_analysis_results(self, 
                                 results: Dict[str, Any], 
                                 include_cross_analysis: bool = True) -> Dict[str, Any]:
        """
        Aggregate results from multiple analyses with cross-analysis insights.
        
        Args:
            results: Dictionary of analysis results by task ID (from parallel execution)
            include_cross_analysis: Whether to include cross-analysis insights
            
        Returns:
            Aggregated analysis results with insights
        """
        try:
            aggregated = {
                "aggregation_metadata": {
                    "total_analyses": len(results),
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "timeout_analyses": 0,
                    "aggregated_at": datetime.now().isoformat(),
                    "engine": "S3AnalysisEngine",
                    "parallel_execution": True
                },
                "analysis_summary": {},
                "cost_insights": {},
                "optimization_opportunities": [],
                "cross_analysis_insights": [],
                "recommendations_by_priority": {
                    "high": [],
                    "medium": [],
                    "low": []
                },
                "total_potential_savings": 0.0,
                "execution_performance": {
                    "total_execution_time": 0.0,
                    "average_execution_time": 0.0,
                    "fastest_analysis": None,
                    "slowest_analysis": None
                }
            }
            
            # Process individual analysis results from parallel execution
            successful_results = {}
            execution_times = []
            
            for task_id, task_result in results.items():
                # Handle both TaskResult objects and direct dictionaries
                if hasattr(task_result, 'data') and task_result.data:
                    result_data = task_result.data
                    task_status = task_result.status
                    task_execution_time = getattr(task_result, 'execution_time', 0)
                    task_error = getattr(task_result, 'error', None)
                elif hasattr(task_result, 'status'):
                    # TaskResult object but data might be in different format
                    result_data = {
                        "status": task_result.status,
                        "analysis_type": getattr(task_result, 'operation', 'unknown'),
                        "execution_time": getattr(task_result, 'execution_time', 0),
                        "error_message": getattr(task_result, 'error', None),
                        "data": getattr(task_result, 'data', {}),
                        "recommendations": []
                    }
                    task_status = task_result.status
                    task_execution_time = getattr(task_result, 'execution_time', 0)
                    task_error = getattr(task_result, 'error', None)
                else:
                    # Direct dictionary result
                    result_data = task_result
                    task_status = result_data.get("status", "unknown")
                    task_execution_time = result_data.get("execution_time", 0)
                    task_error = result_data.get("error_message")
                
                analysis_type = result_data.get("analysis_type", "unknown")
                
                # Update counters based on task status
                if task_status == "success":
                    aggregated["aggregation_metadata"]["successful_analyses"] += 1
                    successful_results[analysis_type] = result_data
                elif task_status == "timeout":
                    aggregated["aggregation_metadata"]["timeout_analyses"] += 1
                else:
                    aggregated["aggregation_metadata"]["failed_analyses"] += 1
                
                # Track execution times
                if isinstance(task_execution_time, (int, float)) and task_execution_time > 0:
                    execution_times.append({
                        "analysis_type": analysis_type,
                        "execution_time": task_execution_time
                    })
                
                # Add to analysis summary with enhanced metadata
                aggregated["analysis_summary"][analysis_type] = {
                    "status": task_status,
                    "execution_time": task_execution_time,
                    "recommendations_count": len(result_data.get("recommendations", [])),
                    "data_sources": result_data.get("data_sources", []),
                    "error_message": task_error,
                    "task_id": task_id,
                    "priority": self.analysis_priorities.get(analysis_type, {}).get("priority", 1),
                    "cost_impact": self.analysis_priorities.get(analysis_type, {}).get("cost_impact", "unknown")
                }
                
                # Aggregate recommendations by priority with enhanced metadata
                for rec in result_data.get("recommendations", []):
                    priority = rec.get("priority", "medium")
                    if priority in aggregated["recommendations_by_priority"]:
                        # Add source analysis to recommendation
                        enhanced_rec = rec.copy()
                        enhanced_rec["source_analysis"] = analysis_type
                        enhanced_rec["source_task_id"] = task_id
                        aggregated["recommendations_by_priority"][priority].append(enhanced_rec)
                    
                    # Sum potential savings
                    savings = rec.get("potential_savings", 0)
                    if isinstance(savings, (int, float)):
                        aggregated["total_potential_savings"] += savings
            
            # Calculate execution performance metrics
            if execution_times:
                total_time = sum(item["execution_time"] for item in execution_times)
                aggregated["execution_performance"]["total_execution_time"] = total_time
                aggregated["execution_performance"]["average_execution_time"] = total_time / len(execution_times)
                
                # Find fastest and slowest analyses
                fastest = min(execution_times, key=lambda x: x["execution_time"])
                slowest = max(execution_times, key=lambda x: x["execution_time"])
                aggregated["execution_performance"]["fastest_analysis"] = fastest
                aggregated["execution_performance"]["slowest_analysis"] = slowest
            
            # Generate cost insights from successful analyses
            if successful_results:
                aggregated["cost_insights"] = self._generate_cost_insights(successful_results)
                aggregated["optimization_opportunities"] = self._identify_optimization_opportunities(successful_results)
                
                # Generate cross-analysis insights if requested
                if include_cross_analysis:
                    aggregated["cross_analysis_insights"] = self._generate_cross_analysis_insights(successful_results)
            
            # Add parallel execution summary
            aggregated["parallel_execution_summary"] = {
                "total_tasks": len(results),
                "successful_tasks": aggregated["aggregation_metadata"]["successful_analyses"],
                "failed_tasks": aggregated["aggregation_metadata"]["failed_analyses"],
                "timeout_tasks": aggregated["aggregation_metadata"]["timeout_analyses"],
                "success_rate": (aggregated["aggregation_metadata"]["successful_analyses"] / len(results)) * 100 if results else 0,
                "total_recommendations": sum(len(recs) for recs in aggregated["recommendations_by_priority"].values()),
                "high_priority_recommendations": len(aggregated["recommendations_by_priority"]["high"]),
                "potential_savings_formatted": f"${aggregated['total_potential_savings']:.2f}"
            }
            
            logger.info(
                f"Aggregated {len(results)} parallel analysis results: "
                f"{aggregated['aggregation_metadata']['successful_analyses']} successful, "
                f"{aggregated['aggregation_metadata']['failed_analyses']} failed, "
                f"{aggregated['aggregation_metadata']['timeout_analyses']} timeout, "
                f"${aggregated['total_potential_savings']:.2f} total potential savings"
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating analysis results: {str(e)}")
            return {
                "status": "error",
                "message": f"Result aggregation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "aggregation_metadata": {
                    "total_analyses": len(results) if results else 0,
                    "successful_analyses": 0,
                    "failed_analyses": 0,
                    "aggregated_at": datetime.now().isoformat(),
                    "engine": "S3AnalysisEngine",
                    "error": str(e)
                }
            }
    
    def _generate_cost_insights(self, successful_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate cost insights from successful analysis results.
        
        Args:
            successful_results: Dictionary of successful analysis results
            
        Returns:
            Dictionary containing cost insights
        """
        insights = {
            "total_storage_cost": 0.0,
            "total_transfer_cost": 0.0,
            "total_api_cost": 0.0,
            "cost_breakdown_by_analysis": {},
            "highest_cost_areas": [],
            "cost_optimization_potential": {}
        }
        
        try:
            # Extract costs from general spend analysis
            general_spend = successful_results.get("general_spend", {})
            if general_spend.get("status") == "success":
                spend_data = general_spend.get("data", {})
                
                # Extract total costs
                total_costs = spend_data.get("total_costs", {})
                insights["total_storage_cost"] = total_costs.get("storage", 0)
                insights["total_transfer_cost"] = total_costs.get("transfer", 0)
                insights["total_api_cost"] = total_costs.get("api", 0)
                
                # Identify highest cost areas
                cost_areas = [
                    ("Storage", insights["total_storage_cost"]),
                    ("Data Transfer", insights["total_transfer_cost"]),
                    ("API Requests", insights["total_api_cost"])
                ]
                insights["highest_cost_areas"] = sorted(cost_areas, key=lambda x: x[1], reverse=True)
            
            # Extract optimization potential from other analyses
            for analysis_type, result in successful_results.items():
                if result.get("status") == "success":
                    recommendations = result.get("recommendations", [])
                    total_savings = sum(
                        rec.get("potential_savings", 0) 
                        for rec in recommendations 
                        if isinstance(rec.get("potential_savings"), (int, float))
                    )
                    
                    if total_savings > 0:
                        insights["cost_optimization_potential"][analysis_type] = {
                            "potential_savings": total_savings,
                            "recommendation_count": len(recommendations)
                        }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cost insights: {str(e)}")
            return {"error": str(e)}
    
    def _identify_optimization_opportunities(self, successful_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities across all analyses.
        
        Args:
            successful_results: Dictionary of successful analysis results
            
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        try:
            # Collect all high-priority recommendations
            high_priority_recs = []
            for analysis_type, result in successful_results.items():
                for rec in result.get("recommendations", []):
                    if rec.get("priority") == "high":
                        rec["source_analysis"] = analysis_type
                        high_priority_recs.append(rec)
            
            # Sort by potential savings
            high_priority_recs.sort(
                key=lambda x: x.get("potential_savings", 0), 
                reverse=True
            )
            
            # Create opportunity summaries
            for i, rec in enumerate(high_priority_recs[:10]):  # Top 10 opportunities
                opportunity = {
                    "rank": i + 1,
                    "title": rec.get("title", "Optimization Opportunity"),
                    "description": rec.get("description", ""),
                    "potential_savings": rec.get("potential_savings", 0),
                    "implementation_effort": rec.get("implementation_effort", "medium"),
                    "source_analysis": rec.get("source_analysis", "unknown"),
                    "priority": rec.get("priority", "medium")
                }
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
    
    def _generate_cross_analysis_insights(self, successful_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights that span multiple analyses.
        
        Args:
            successful_results: Dictionary of successful analysis results
            
        Returns:
            List of cross-analysis insights
        """
        insights = []
        
        try:
            # Check for storage class and archive optimization correlation
            storage_class_result = successful_results.get("storage_class")
            archive_result = successful_results.get("archive_optimization")
            
            if storage_class_result and archive_result:
                insights.append({
                    "type": "storage_optimization_correlation",
                    "title": "Storage Class and Archive Optimization Correlation",
                    "description": "Both storage class optimization and archive strategies show potential savings",
                    "recommendation": "Implement storage class transitions before archive policies for maximum savings",
                    "analyses_involved": ["storage_class", "archive_optimization"]
                })
            
            # Check for governance and cost optimization correlation
            governance_result = successful_results.get("governance")
            if governance_result and len(successful_results) > 1:
                governance_recs = governance_result.get("recommendations", [])
                lifecycle_issues = any(
                    "lifecycle" in rec.get("title", "").lower() 
                    for rec in governance_recs
                )
                
                if lifecycle_issues:
                    insights.append({
                        "type": "governance_cost_correlation",
                        "title": "Governance Issues Impact Cost Optimization",
                        "description": "Missing lifecycle policies prevent automatic cost optimization",
                        "recommendation": "Address governance issues first to enable automated cost optimization",
                        "analyses_involved": ["governance", "storage_class", "archive_optimization"]
                    })
            
            # Check for API cost and multipart cleanup correlation
            api_result = successful_results.get("api_cost")
            multipart_result = successful_results.get("multipart_cleanup")
            
            if api_result and multipart_result:
                insights.append({
                    "type": "api_multipart_correlation",
                    "title": "API Costs and Incomplete Uploads Correlation",
                    "description": "High API costs may be related to incomplete multipart uploads",
                    "recommendation": "Clean up incomplete uploads to reduce unnecessary API charges",
                    "analyses_involved": ["api_cost", "multipart_cleanup"]
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cross-analysis insights: {str(e)}")
            return []
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get status information about the analysis engine.
        
        Returns:
            Dictionary containing engine status
        """
        return {
            "engine_info": {
                "class": "S3AnalysisEngine",
                "region": self.region,
                "initialized_at": datetime.now().isoformat()
            },
            "registered_analyzers": self.analyzer_registry.get_analyzer_info(),
            "analysis_priorities": self.analysis_priorities,
            "execution_history": self.execution_history[-10:],  # Last 10 executions
            "services_status": {
                "s3_service": self.s3_service is not None,
                "pricing_service": self.pricing_service is not None,
                "storage_lens_service": self.storage_lens_service is not None
            }
        }
    
    def cleanup(self):
        """Clean up engine resources."""
        try:
            # Clear execution history
            self.execution_history.clear()
            
            # Clean up services if they have cleanup methods
            for service in [self.s3_service, self.pricing_service, self.storage_lens_service]:
                if service and hasattr(service, 'cleanup'):
                    service.cleanup()
            
            logger.info("S3AnalysisEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during S3AnalysisEngine cleanup: {str(e)}")