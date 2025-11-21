"""
S3 Optimization Orchestrator for CFM Tips MCP Server

Main coordination layer for S3 optimization workflows with session integration,
performance monitoring, intelligent caching, and memory management.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.service_orchestrator import ServiceOrchestrator
from utils.session_manager import get_session_manager
from .s3_analysis_engine import S3AnalysisEngine
from .s3_aggregation_queries import S3AggregationQueries, S3QueryExecutor
from utils.performance_monitor import get_performance_monitor
from utils.intelligent_cache import get_pricing_cache, get_bucket_metadata_cache, get_analysis_results_cache
from utils.memory_manager import get_memory_manager
from utils.progressive_timeout import get_timeout_handler
from utils.documentation_links import add_documentation_links

logger = logging.getLogger(__name__)


class S3OptimizationOrchestrator:
    """Main orchestrator for S3 optimization workflows."""
    
    def __init__(self, region: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize S3OptimizationOrchestrator with performance optimizations.
        
        Args:
            region: AWS region for S3 operations
            session_id: Optional session ID for data persistence
        """
        self.region = region
        self.session_manager = get_session_manager()
        
        # Initialize ServiceOrchestrator (it will create session if session_id is None)
        self.service_orchestrator = ServiceOrchestrator(session_id)
        
        # Get the actual session ID from ServiceOrchestrator
        self.session_id = self.service_orchestrator.session_id
        
        # Initialize performance optimization components
        self.performance_monitor = get_performance_monitor()
        self.memory_manager = get_memory_manager()
        self.timeout_handler = get_timeout_handler()
        
        # Initialize caching systems
        self.pricing_cache = get_pricing_cache()
        self.bucket_metadata_cache = get_bucket_metadata_cache()
        self.analysis_results_cache = get_analysis_results_cache()
        
        # Register cache instances with memory manager for cleanup
        self.memory_manager.add_cache_reference(self.pricing_cache)
        self.memory_manager.add_cache_reference(self.bucket_metadata_cache)
        self.memory_manager.add_cache_reference(self.analysis_results_cache)
        
        # Set up performance monitor integration
        self.pricing_cache.set_performance_monitor(self.performance_monitor)
        self.bucket_metadata_cache.set_performance_monitor(self.performance_monitor)
        self.analysis_results_cache.set_performance_monitor(self.performance_monitor)
        self.memory_manager.set_performance_monitor(self.performance_monitor)
        self.timeout_handler.set_performance_monitor(self.performance_monitor)
        
        # Initialize analysis engine with all analyzers and performance components
        self.analysis_engine = S3AnalysisEngine(
            region=region,
            performance_monitor=self.performance_monitor,
            memory_manager=self.memory_manager,
            timeout_handler=self.timeout_handler,
            pricing_cache=self.pricing_cache,
            bucket_metadata_cache=self.bucket_metadata_cache,
            analysis_results_cache=self.analysis_results_cache
        )
        
        logger.info(f"S3OptimizationOrchestrator initialized with performance optimizations for region: {region or 'default'}, session: {self.session_id}")
    
    async def execute_analysis(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific S3 analysis with performance optimizations.
        
        Args:
            analysis_type: Type of analysis to execute
            **kwargs: Analysis-specific parameters
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        # Start performance monitoring
        monitoring_session = self.performance_monitor.start_analysis_monitoring(
            analysis_type, 
            f"single_{int(start_time)}"
        )
        
        # Start memory tracking
        memory_tracker = self.memory_manager.start_memory_tracking(f"analysis_{analysis_type}")
        
        logger.info(f"Starting S3 analysis with performance optimizations: {analysis_type}")
        
        try:
            # Check cache for recent results first
            cache_key = [analysis_type, kwargs.get('region', self.region), kwargs]
            cached_result = self.analysis_results_cache.get(cache_key)
            
            if cached_result is not None:
                logger.info(f"Retrieved {analysis_type} analysis from cache")
                self.performance_monitor.record_cache_hit("analysis_results", analysis_type)
                
                # Update execution time and return cached result
                cached_result["orchestrator_execution_time"] = time.time() - start_time
                cached_result["session_id"] = self.session_id
                cached_result["from_cache"] = True
                
                # End monitoring
                self.performance_monitor.end_analysis_monitoring(monitoring_session, success=True)
                self.memory_manager.stop_memory_tracking(f"analysis_{analysis_type}")
                
                return cached_result
            
            self.performance_monitor.record_cache_miss("analysis_results", analysis_type)
            
            # Validate analysis type
            if not self._is_valid_analysis_type(analysis_type):
                error_result = {
                    "status": "error",
                    "message": f"Invalid analysis type: {analysis_type}",
                    "analysis_type": analysis_type,
                    "available_types": self.analysis_engine.analyzer_registry.list_analyzers()
                }
                
                self.performance_monitor.end_analysis_monitoring(
                    monitoring_session, 
                    success=False, 
                    error_message="Invalid analysis type"
                )
                self.memory_manager.stop_memory_tracking(f"analysis_{analysis_type}")
                
                return error_result
            
            # Calculate intelligent timeout
            timeout_seconds = self.timeout_handler.get_timeout_for_analysis(analysis_type, **kwargs)
            kwargs['timeout_seconds'] = timeout_seconds
            
            logger.info(f"Calculated timeout for {analysis_type}: {timeout_seconds:.1f}s")
            
            # Prepare analysis parameters with performance optimizations
            analysis_params = self._prepare_analysis_params(analysis_type, **kwargs)
            analysis_params['monitoring_session'] = monitoring_session
            analysis_params['memory_tracker'] = memory_tracker
            
            # Execute analysis using the analysis engine
            result = await self.analysis_engine.run_analysis(analysis_type, **analysis_params)
            
            # Cache successful results
            if result.get('status') == 'success':
                # Cache result with appropriate TTL based on analysis type
                cache_ttl = self._get_cache_ttl_for_analysis(analysis_type)
                self.analysis_results_cache.put(
                    cache_key, 
                    result.copy(), 
                    ttl_seconds=cache_ttl,
                    tags={"analysis_type": analysis_type, "region": self.region}
                )
                
                logger.debug(f"Cached {analysis_type} analysis result (TTL: {cache_ttl}s)")
            
            # Store results if requested and analysis was successful
            if kwargs.get('store_results', True) and result.get('status') == 'success':
                self._store_analysis_results(analysis_type, result)
            
            execution_time = time.time() - start_time
            result["orchestrator_execution_time"] = execution_time
            result["session_id"] = self.session_id
            result["from_cache"] = False
            
            # Record performance metrics
            self.timeout_handler.record_execution_time(
                analysis_type, 
                execution_time,
                self.timeout_handler.get_complexity_level(analysis_type, **kwargs)
            )
            
            # End monitoring
            success = result.get('status') == 'success'
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session, 
                success=success,
                error_message=result.get('error_message') if not success else None
            )
            
            memory_stats = self.memory_manager.stop_memory_tracking(f"analysis_{analysis_type}")
            if memory_stats:
                result["memory_usage"] = memory_stats
            
            logger.info(f"Completed S3 analysis: {analysis_type} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in S3 analysis {analysis_type}: {str(e)}")
            
            # End monitoring with error
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session, 
                success=False, 
                error_message=str(e)
            )
            self.memory_manager.stop_memory_tracking(f"analysis_{analysis_type}")
            
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "message": f"Analysis failed: {str(e)}",
                "execution_time": execution_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "from_cache": False
            }
    
    async def execute_comprehensive_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all S3 analyses in parallel with performance optimizations and intelligent resource management.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        start_time = time.time()
        
        # Start comprehensive performance monitoring
        monitoring_session = self.performance_monitor.start_analysis_monitoring(
            "comprehensive", 
            f"comprehensive_{int(start_time)}"
        )
        
        # Start memory tracking for comprehensive analysis
        memory_tracker = self.memory_manager.start_memory_tracking("comprehensive_analysis")
        
        logger.info("Starting comprehensive S3 analysis with performance optimizations and parallel execution")
        
        try:
            # Check cache for recent comprehensive analysis
            cache_key = ["comprehensive", kwargs.get('region', self.region), kwargs]
            cached_result = self.analysis_results_cache.get(cache_key)
            
            if cached_result is not None:
                logger.info("Retrieved comprehensive analysis from cache")
                self.performance_monitor.record_cache_hit("analysis_results", "comprehensive")
                
                # Update execution time and return cached result
                cached_result["orchestrator_execution_time"] = time.time() - start_time
                cached_result["session_id"] = self.session_id
                cached_result["from_cache"] = True
                
                # End monitoring
                self.performance_monitor.end_analysis_monitoring(monitoring_session, success=True)
                self.memory_manager.stop_memory_tracking("comprehensive_analysis")
                
                return cached_result
            
            self.performance_monitor.record_cache_miss("analysis_results", "comprehensive")
            
            # Get all available analysis types from the engine with priority information
            available_analyses = self.analysis_engine.get_available_analyses()
            analysis_types = [analysis["analysis_type"] for analysis in available_analyses]
            
            logger.info(f"Executing {len(analysis_types)} analyses with intelligent prioritization: {analysis_types}")
            
            # Calculate intelligent timeout for comprehensive analysis
            comprehensive_timeout = self.timeout_handler.get_timeout_for_analysis("comprehensive", **kwargs)
            kwargs['total_timeout'] = comprehensive_timeout
            
            logger.info(f"Calculated comprehensive analysis timeout: {comprehensive_timeout:.1f}s")
            
            # Prepare analysis parameters with performance optimizations
            analysis_params = self._prepare_comprehensive_analysis_params(**kwargs)
            analysis_params['monitoring_session'] = monitoring_session
            analysis_params['memory_tracker'] = memory_tracker
            
            # Register large object for memory management if available
            if self.memory_manager:
                try:
                    self.memory_manager.register_large_object(
                        f"comprehensive_analysis_{int(start_time)}",
                        analysis_params,
                        size_mb=1.0,  # Estimated size
                        cleanup_callback=lambda: logger.debug("Cleaned up comprehensive analysis parameters")
                    )
                except Exception as e:
                    logger.warning(f"Could not register large object with memory manager: {str(e)}")
            
            # Create parallel analysis tasks using analysis engine
            service_calls = self.analysis_engine.create_parallel_analysis_tasks(
                analysis_types=analysis_types,
                **analysis_params
            )
            
            logger.info(f"Created {len(service_calls)} intelligently prioritized parallel tasks")
            
            # Execute analyses in parallel using ServiceOrchestrator with session-sql integration
            execution_results = self.service_orchestrator.execute_parallel_analysis(
                service_calls=service_calls,
                store_results=kwargs.get('store_results', True),
                timeout=comprehensive_timeout
            )
            
            logger.info(f"Parallel execution completed: {execution_results['successful']}/{execution_results['total_tasks']} successful")
            
            # Record performance metrics
            self.performance_monitor.record_metric(
                "comprehensive_analysis_success_rate",
                (execution_results['successful'] / execution_results['total_tasks'] * 100) if execution_results['total_tasks'] > 0 else 0,
                tags={"session_id": self.session_id}
            )
            
            # Aggregate results using enhanced aggregation with cross-analyzer insights
            aggregated_results = self.aggregate_results_with_insights(
                results=execution_results.get('results', {}),
                include_cross_analysis=kwargs.get('include_cross_analysis', True)
            )
            
            # Store aggregated results with session-sql integration
            if kwargs.get('store_results', True):
                self._store_comprehensive_results(aggregated_results, execution_results)
            
            # Execute cross-analysis aggregation queries for deeper insights
            cross_analysis_data = {}
            if kwargs.get('include_cross_analysis', True) and execution_results.get('stored_tables'):
                cross_analysis_data = self._execute_cross_analysis_queries(execution_results['stored_tables'])
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            comprehensive_result = {
                "status": "success",
                "analysis_type": "comprehensive",
                "execution_summary": execution_results,
                "aggregated_results": aggregated_results,
                "cross_analysis_data": cross_analysis_data,
                "execution_time": execution_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "from_cache": False,
                "performance_optimizations": {
                    "intelligent_timeout": comprehensive_timeout,
                    "cache_enabled": True,
                    "memory_management": True,
                    "progressive_timeouts": True
                },
                "analysis_metadata": {
                    "total_analyses": len(analysis_types),
                    "successful_analyses": aggregated_results.get("aggregation_metadata", {}).get("successful_analyses", 0),
                    "failed_analyses": aggregated_results.get("aggregation_metadata", {}).get("failed_analyses", 0),
                    "total_potential_savings": aggregated_results.get("total_potential_savings", 0),
                    "stored_tables": execution_results.get('stored_tables', []),
                    "task_prioritization": self._get_task_prioritization_summary(available_analyses)
                }
            }
            
            # Cache the comprehensive result
            cache_ttl = self._get_cache_ttl_for_analysis("comprehensive")
            self.analysis_results_cache.put(
                cache_key,
                comprehensive_result.copy(),
                ttl_seconds=cache_ttl,
                tags={"analysis_type": "comprehensive", "region": self.region}
            )
            
            # Record performance metrics
            self.timeout_handler.record_execution_time(
                "comprehensive",
                execution_time,
                self.timeout_handler.get_complexity_level("comprehensive", **kwargs)
            )
            
            # End monitoring
            self.performance_monitor.end_analysis_monitoring(monitoring_session, success=True)
            memory_stats = self.memory_manager.stop_memory_tracking("comprehensive_analysis")
            
            if memory_stats:
                comprehensive_result["memory_usage"] = memory_stats
            
            logger.info(f"Completed comprehensive S3 analysis with optimizations in {execution_time:.2f}s")
            
            return comprehensive_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in comprehensive S3 analysis: {str(e)}")
            
            # End monitoring with error
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session,
                success=False,
                error_message=str(e)
            )
            self.memory_manager.stop_memory_tracking("comprehensive_analysis")
            
            return {
                "status": "error",
                "analysis_type": "comprehensive",
                "message": f"Comprehensive analysis failed: {str(e)}",
                "execution_time": execution_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "from_cache": False
            }
    
    def get_analysis_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Query stored analysis results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List of query results
        """
        try:
            return self.service_orchestrator.query_session_data(query)
        except Exception as e:
            logger.error(f"Error querying analysis results: {str(e)}")
            return []
    
    def get_stored_tables(self) -> List[str]:
        """
        Get list of tables stored in the session.
        
        Returns:
            List of table names
        """
        try:
            return self.service_orchestrator.get_stored_tables()
        except Exception as e:
            logger.error(f"Error getting stored tables: {str(e)}")
            return []
    
    def _is_valid_analysis_type(self, analysis_type: str) -> bool:
        """Validate analysis type using the analysis engine registry."""
        if analysis_type == "comprehensive":
            return True
        return self.analysis_engine.analyzer_registry.get(analysis_type) is not None
    
    def get_analyzer_registry(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the analyzer registry and registered analyzers.
        
        Returns:
            Dictionary containing analyzer registry information
        """
        try:
            # Get basic registry info
            registry_info = {
                "registry_status": "active",
                "total_analyzers": len(self.analysis_engine.analyzer_registry.list_analyzers()),
                "registered_analyzers": self.analysis_engine.analyzer_registry.get_analyzer_info(),
                "analysis_priorities": self.analysis_engine.analysis_priorities,
                "available_analyses": self.analysis_engine.get_available_analyses(),
                "registry_timestamp": datetime.now().isoformat()
            }
            
            # Get comprehensive health status from analysis engine
            health_status = self.analysis_engine.get_analyzer_health_status()
            registry_info["health_status"] = health_status
            
            # Add loading results if available
            if hasattr(self.analysis_engine, 'analyzer_loading_results'):
                registry_info["loading_results"] = self.analysis_engine.analyzer_loading_results
            
            # Add execution history summary
            if hasattr(self.analysis_engine, 'execution_history') and self.analysis_engine.execution_history:
                recent_executions = self.analysis_engine.execution_history[-10:]  # Last 10 executions
                registry_info["recent_executions"] = recent_executions
                
                # Calculate execution statistics
                successful_executions = [e for e in recent_executions if e.get('status') == 'success']
                registry_info["execution_statistics"] = {
                    "recent_success_rate": len(successful_executions) / len(recent_executions) * 100 if recent_executions else 0,
                    "total_executions": len(self.analysis_engine.execution_history),
                    "recent_executions": len(recent_executions)
                }
            
            # Add performance optimization status
            registry_info["performance_optimizations"] = {
                "performance_monitor_enabled": self.performance_monitor is not None,
                "memory_manager_enabled": self.memory_manager is not None,
                "timeout_handler_enabled": self.timeout_handler is not None,
                "caching_enabled": {
                    "pricing_cache": self.pricing_cache is not None,
                    "bucket_metadata_cache": self.bucket_metadata_cache is not None,
                    "analysis_results_cache": self.analysis_results_cache is not None
                }
            }
            
            return registry_info
            
        except Exception as e:
            logger.error(f"Error getting analyzer registry info: {str(e)}")
            return {
                "registry_status": "error",
                "error": str(e),
                "error_type": e.__class__.__name__,
                "registry_timestamp": datetime.now().isoformat()
            }
    
    def reload_analyzer(self, analysis_type: str) -> Dict[str, Any]:
        """
        Reload a specific analyzer through the analysis engine.
        
        Args:
            analysis_type: Type of analyzer to reload
            
        Returns:
            Dictionary containing reload results
        """
        try:
            logger.info(f"Orchestrator reloading analyzer: {analysis_type}")
            
            # Use analysis engine's reload method
            reload_result = self.analysis_engine.reload_analyzer(analysis_type)
            
            # Clear related caches if reload was successful
            if reload_result.get("status") == "success":
                try:
                    # Clear analysis results cache for this analyzer
                    if self.analysis_results_cache:
                        cache_keys_to_clear = []
                        cache_dict = getattr(self.analysis_results_cache, '_cache', {})
                        for key in cache_dict.keys():
                            if isinstance(key, (list, tuple)) and len(key) > 0 and key[0] == analysis_type:
                                cache_keys_to_clear.append(key)
                        
                        for key in cache_keys_to_clear:
                            if hasattr(self.analysis_results_cache, 'invalidate'):
                                self.analysis_results_cache.invalidate(key)
                            elif hasattr(self.analysis_results_cache, 'delete'):
                                self.analysis_results_cache.delete(key)
                        
                        logger.info(f"Cleared {len(cache_keys_to_clear)} cache entries for {analysis_type}")
                        reload_result["cache_entries_cleared"] = len(cache_keys_to_clear)
                    
                except Exception as cache_error:
                    logger.warning(f"Error clearing cache for {analysis_type}: {str(cache_error)}")
                    reload_result["cache_clear_warning"] = str(cache_error)
            
            return reload_result
            
        except Exception as e:
            logger.error(f"Error in orchestrator reload for {analysis_type}: {str(e)}")
            return {
                "status": "error",
                "message": f"Orchestrator reload failed for {analysis_type}: {str(e)}",
                "error_type": e.__class__.__name__,
                "reloaded_at": datetime.now().isoformat()
            }
    
    def handle_analyzer_failure(self, analysis_type: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analyzer failures with comprehensive error handling and recovery strategies.
        
        Args:
            analysis_type: Type of analysis that failed
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing error handling results and recovery recommendations
        """
        try:
            logger.error(f"Handling analyzer failure for {analysis_type}: {str(error)}")
            
            # Use analysis engine's error handling if available
            if hasattr(self.analysis_engine, 'handle_analyzer_failure'):
                return self.analysis_engine.handle_analyzer_failure(analysis_type, error, context)
            
            # Fallback error handling
            error_message = str(error)
            error_type = error.__class__.__name__
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error_type, error_message)
            
            # Create comprehensive error result
            error_result = {
                "status": "error",
                "analysis_type": analysis_type,
                "error_message": error_message,
                "error_type": error_type,
                "error_category": recovery_strategy["category"],
                "recovery_strategy": recovery_strategy["strategy"],
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "orchestrator_handled": True,
                "recommendations": []
            }
            
            # Add specific recommendations based on error type
            if "permission" in error_message.lower() or "access" in error_message.lower():
                error_result["recommendations"].extend([
                    {
                        "type": "permission_fix",
                        "priority": "high",
                        "title": "Fix AWS Permissions",
                        "description": f"Analysis {analysis_type} failed due to permission issues",
                        "action_items": [
                            "Check IAM permissions for S3, Cost Explorer, and Storage Lens",
                            "Verify AWS credentials are valid and not expired",
                            "Ensure required service permissions are granted",
                            "Check if MFA is required for API access"
                        ]
                    }
                ])
            elif "timeout" in error_message.lower():
                error_result["recommendations"].extend([
                    {
                        "type": "timeout_optimization",
                        "priority": "medium",
                        "title": "Optimize Timeout Settings",
                        "description": f"Analysis {analysis_type} timed out during execution",
                        "action_items": [
                            "Increase timeout_seconds parameter",
                            "Reduce lookback_days to limit data volume",
                            "Filter to specific bucket_names if possible",
                            "Run analysis during off-peak hours"
                        ]
                    }
                ])
            elif "rate" in error_message.lower() or "throttl" in error_message.lower():
                error_result["recommendations"].extend([
                    {
                        "type": "rate_limit_handling",
                        "priority": "medium",
                        "title": "Handle API Rate Limits",
                        "description": f"Analysis {analysis_type} hit API rate limits",
                        "action_items": [
                            "Implement exponential backoff retry logic",
                            "Reduce concurrent analysis execution",
                            "Spread analysis execution over time",
                            "Consider using AWS SDK retry configuration"
                        ]
                    }
                ])
            
            # Record error in performance monitor if available
            if self.performance_monitor:
                self.performance_monitor.record_metric(
                    f"orchestrator_analyzer_failure_{analysis_type}",
                    1,
                    tags={
                        "error_type": error_type,
                        "error_category": recovery_strategy["category"],
                        "session_id": self.session_id
                    }
                )
            
            # Attempt automatic recovery if strategy suggests it
            if recovery_strategy.get("auto_recovery", False):
                recovery_result = self._attempt_auto_recovery(analysis_type, error, context)
                error_result["auto_recovery_attempted"] = True
                error_result["auto_recovery_result"] = recovery_result
            
            return error_result
            
        except Exception as handling_error:
            logger.error(f"Error in analyzer failure handling: {str(handling_error)}")
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "error_message": f"Original error: {str(error)}. Handling error: {str(handling_error)}",
                "error_type": "FailureHandlingError",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "critical_error": True
            }
    
    def _determine_recovery_strategy(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """
        Determine recovery strategy based on error type and message.
        
        Args:
            error_type: Type of exception
            error_message: Error message
            
        Returns:
            Dictionary containing recovery strategy information
        """
        error_lower = error_message.lower()
        
        if "permission" in error_lower or "access" in error_lower or "credential" in error_lower:
            return {
                "category": "permission_error",
                "strategy": "Check and fix AWS permissions",
                "auto_recovery": False,
                "severity": "high"
            }
        elif "timeout" in error_lower:
            return {
                "category": "timeout_error",
                "strategy": "Reduce scope and increase timeout",
                "auto_recovery": True,
                "severity": "medium"
            }
        elif "throttl" in error_lower or "rate" in error_lower:
            return {
                "category": "rate_limit_error",
                "strategy": "Implement backoff and retry",
                "auto_recovery": True,
                "severity": "medium"
            }
        elif "network" in error_lower or "connection" in error_lower:
            return {
                "category": "network_error",
                "strategy": "Retry with exponential backoff",
                "auto_recovery": True,
                "severity": "medium"
            }
        elif "service" in error_lower and "unavailable" in error_lower:
            return {
                "category": "service_error",
                "strategy": "Wait and retry, use fallback data sources",
                "auto_recovery": True,
                "severity": "high"
            }
        else:
            return {
                "category": "unknown_error",
                "strategy": "Manual investigation required",
                "auto_recovery": False,
                "severity": "high"
            }
    
    def _attempt_auto_recovery(self, analysis_type: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt automatic recovery for recoverable errors.
        
        Args:
            analysis_type: Type of analysis that failed
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing recovery attempt results
        """
        try:
            logger.info(f"Attempting auto-recovery for {analysis_type}")
            
            recovery_actions = []
            
            # Try reloading the analyzer
            reload_result = self.reload_analyzer(analysis_type)
            recovery_actions.append({
                "action": "reload_analyzer",
                "result": reload_result.get("status"),
                "details": reload_result
            })
            
            # Clear related caches
            if self.analysis_results_cache:
                try:
                    cache_keys_cleared = 0
                    cache_dict = getattr(self.analysis_results_cache, '_cache', {})
                    for key in list(cache_dict.keys()):
                        if isinstance(key, (list, tuple)) and len(key) > 0 and key[0] == analysis_type:
                            if hasattr(self.analysis_results_cache, 'invalidate'):
                                self.analysis_results_cache.invalidate(key)
                            elif hasattr(self.analysis_results_cache, 'delete'):
                                self.analysis_results_cache.delete(key)
                            cache_keys_cleared += 1
                    
                    recovery_actions.append({
                        "action": "clear_cache",
                        "result": "success",
                        "details": {"cache_keys_cleared": cache_keys_cleared}
                    })
                except Exception as cache_error:
                    recovery_actions.append({
                        "action": "clear_cache",
                        "result": "error",
                        "details": {"error": str(cache_error)}
                    })
            
            return {
                "status": "completed",
                "recovery_actions": recovery_actions,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type
            }
            
        except Exception as recovery_error:
            logger.error(f"Error during auto-recovery for {analysis_type}: {str(recovery_error)}")
            return {
                "status": "failed",
                "error": str(recovery_error),
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type
            }
    
    def get_analyzer_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics for all analyzers and the analysis engine.
        
        Returns:
            Dictionary containing diagnostic information
        """
        try:
            diagnostics = {
                "diagnostics_timestamp": datetime.now().isoformat(),
                "orchestrator_info": {
                    "region": self.region,
                    "session_id": self.session_id,
                    "performance_optimizations_enabled": True
                },
                "analysis_engine_info": {
                    "region": self.analysis_engine.region,
                    "total_analyzers": len(self.analysis_engine.analyzer_registry.list_analyzers()),
                    "execution_history_count": len(getattr(self.analysis_engine, 'execution_history', []))
                }
            }
            
            # Get health status
            health_status = self.analysis_engine.get_analyzer_health_status()
            diagnostics["health_status"] = health_status
            
            # Get loading results
            if hasattr(self.analysis_engine, 'analyzer_loading_results'):
                diagnostics["loading_results"] = self.analysis_engine.analyzer_loading_results
            
            # Test analyzer connectivity
            connectivity_tests = {}
            for analysis_type in self.analysis_engine.analyzer_registry.list_analyzers():
                analyzer = self.analysis_engine.analyzer_registry.get(analysis_type)
                
                connectivity_test = {
                    "analyzer_valid": self.analysis_engine._validate_analyzer(analyzer),
                    "services_connected": {
                        "s3_service": analyzer.s3_service is not None,
                        "pricing_service": analyzer.pricing_service is not None,
                        "storage_lens_service": analyzer.storage_lens_service is not None
                    },
                    "performance_components": {
                        "performance_monitor": hasattr(analyzer, 'performance_monitor') and analyzer.performance_monitor is not None,
                        "memory_manager": hasattr(analyzer, 'memory_manager') and analyzer.memory_manager is not None,
                        "timeout_handler": hasattr(analyzer, 'timeout_handler') and analyzer.timeout_handler is not None
                    }
                }
                
                connectivity_tests[analysis_type] = connectivity_test
            
            diagnostics["connectivity_tests"] = connectivity_tests
            
            # Get cache statistics
            cache_stats = {}
            if self.pricing_cache:
                try:
                    cache_size = len(getattr(self.pricing_cache, '_cache', {}))
                except:
                    cache_size = 0
                cache_stats["pricing_cache"] = {
                    "size": cache_size,
                    "hit_rate": getattr(self.pricing_cache, 'hit_rate', 0),
                    "enabled": True
                }
            
            if self.bucket_metadata_cache:
                try:
                    cache_size = len(getattr(self.bucket_metadata_cache, '_cache', {}))
                except:
                    cache_size = 0
                cache_stats["bucket_metadata_cache"] = {
                    "size": cache_size,
                    "hit_rate": getattr(self.bucket_metadata_cache, 'hit_rate', 0),
                    "enabled": True
                }
            
            if self.analysis_results_cache:
                try:
                    cache_size = len(getattr(self.analysis_results_cache, '_cache', {}))
                except:
                    cache_size = 0
                cache_stats["analysis_results_cache"] = {
                    "size": cache_size,
                    "hit_rate": getattr(self.analysis_results_cache, 'hit_rate', 0),
                    "enabled": True
                }
            
            diagnostics["cache_statistics"] = cache_stats
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting analyzer diagnostics: {str(e)}")
            return {
                "diagnostics_timestamp": datetime.now().isoformat(),
                "status": "error",
                "error_message": str(e),
                "error_type": e.__class__.__name__
            }
    
    def reload_analyzers(self) -> Dict[str, Any]:
        """
        Reload all analyzers in the registry with fresh instances.
        
        Returns:
            Dictionary containing reload results
        """
        try:
            logger.info("Reloading analyzers in registry")
            
            # Get current analyzer count
            old_count = len(self.analysis_engine.analyzer_registry.list_analyzers())
            
            # Clear existing registry
            self.analysis_engine.analyzer_registry._analyzers.clear()
            
            # Reinitialize analyzers
            self.analysis_engine._initialize_analyzers()
            
            # Get new analyzer count
            new_count = len(self.analysis_engine.analyzer_registry.list_analyzers())
            
            reload_result = {
                "status": "success",
                "message": "Analyzers reloaded successfully",
                "old_analyzer_count": old_count,
                "new_analyzer_count": new_count,
                "reloaded_analyzers": self.analysis_engine.analyzer_registry.list_analyzers(),
                "reload_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully reloaded {new_count} analyzers")
            return reload_result
            
        except Exception as e:
            logger.error(f"Error reloading analyzers: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to reload analyzers: {str(e)}",
                "reload_timestamp": datetime.now().isoformat()
            }
    
    def handle_analyzer_failure(self, analysis_type: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analyzer failures with comprehensive error handling and recovery strategies.
        
        Args:
            analysis_type: Type of analysis that failed
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing error handling results and recovery recommendations
        """
        error_message = str(error)
        error_type = error.__class__.__name__
        
        logger.error(f"Analyzer failure in {analysis_type}: {error_message}")
        
        # Determine error category and recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error_type, error_message)
        
        # Create comprehensive error result
        error_result = {
            "status": "error",
            "analysis_type": analysis_type,
            "error_message": error_message,
            "error_type": error_type,
            "error_category": recovery_strategy["category"],
            "recovery_strategy": recovery_strategy["strategy"],
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "recommendations": []
        }
        
        # Add specific recommendations based on error type
        if "permission" in error_message.lower() or "access" in error_message.lower():
            error_result["recommendations"].extend([
                {
                    "type": "permission_fix",
                    "priority": "high",
                    "title": "Fix AWS Permissions",
                    "description": f"Analysis {analysis_type} failed due to permission issues",
                    "action_items": [
                        "Check IAM permissions for S3, Cost Explorer, and Storage Lens",
                        "Verify AWS credentials are valid and not expired",
                        "Ensure required service permissions are granted",
                        "Check if MFA is required for API access"
                    ]
                }
            ])
        elif "timeout" in error_message.lower():
            error_result["recommendations"].extend([
                {
                    "type": "timeout_optimization",
                    "priority": "medium",
                    "title": "Optimize Timeout Settings",
                    "description": f"Analysis {analysis_type} timed out during execution",
                    "action_items": [
                        "Increase timeout_seconds parameter",
                        "Reduce lookback_days to limit data volume",
                        "Filter to specific bucket_names if possible",
                        "Run analysis during off-peak hours"
                    ]
                }
            ])
        elif "rate" in error_message.lower() or "throttl" in error_message.lower():
            error_result["recommendations"].extend([
                {
                    "type": "rate_limit_handling",
                    "priority": "medium",
                    "title": "Handle API Rate Limits",
                    "description": f"Analysis {analysis_type} hit API rate limits",
                    "action_items": [
                        "Implement exponential backoff retry logic",
                        "Reduce concurrent analysis execution",
                        "Spread analysis execution over time",
                        "Consider using AWS SDK retry configuration"
                    ]
                }
            ])
        else:
            error_result["recommendations"].extend([
                {
                    "type": "general_troubleshooting",
                    "priority": "medium",
                    "title": "General Troubleshooting",
                    "description": f"Analysis {analysis_type} failed with unexpected error",
                    "action_items": [
                        "Check AWS service status and availability",
                        "Verify network connectivity",
                        "Review analysis parameters for validity",
                        "Check system resources and memory usage",
                        "Try running analysis with reduced scope"
                    ]
                }
            ])
        
        # Record error in performance monitor if available
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"analyzer_failure_{analysis_type}",
                1,
                tags={
                    "error_type": error_type,
                    "error_category": recovery_strategy["category"],
                    "session_id": self.session_id
                }
            )
        
        # Attempt automatic recovery if strategy suggests it
        if recovery_strategy["auto_recovery"]:
            recovery_result = self._attempt_auto_recovery(analysis_type, error, context)
            error_result["auto_recovery_attempted"] = True
            error_result["auto_recovery_result"] = recovery_result
        
        return error_result
    
    def _determine_recovery_strategy(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """
        Determine recovery strategy based on error type and message.
        
        Args:
            error_type: Type of exception
            error_message: Error message content
            
        Returns:
            Dictionary containing recovery strategy information
        """
        error_message_lower = error_message.lower()
        
        # Permission/Access errors
        if any(keyword in error_message_lower for keyword in ["permission", "access", "denied", "unauthorized", "forbidden"]):
            return {
                "category": "permission_error",
                "strategy": "check_permissions",
                "auto_recovery": False,
                "severity": "high"
            }
        
        # Timeout errors
        elif any(keyword in error_message_lower for keyword in ["timeout", "timed out", "deadline"]):
            return {
                "category": "timeout_error",
                "strategy": "increase_timeout_or_reduce_scope",
                "auto_recovery": True,
                "severity": "medium"
            }
        
        # Rate limiting errors
        elif any(keyword in error_message_lower for keyword in ["rate", "throttl", "limit", "quota"]):
            return {
                "category": "rate_limit_error",
                "strategy": "implement_backoff_retry",
                "auto_recovery": True,
                "severity": "medium"
            }
        
        # Network/connectivity errors
        elif any(keyword in error_message_lower for keyword in ["network", "connection", "dns", "resolve"]):
            return {
                "category": "network_error",
                "strategy": "check_connectivity",
                "auto_recovery": True,
                "severity": "medium"
            }
        
        # Service unavailable errors
        elif any(keyword in error_message_lower for keyword in ["unavailable", "service", "maintenance"]):
            return {
                "category": "service_error",
                "strategy": "retry_later",
                "auto_recovery": True,
                "severity": "low"
            }
        
        # Data/validation errors
        elif any(keyword in error_message_lower for keyword in ["invalid", "validation", "parameter", "format"]):
            return {
                "category": "validation_error",
                "strategy": "check_parameters",
                "auto_recovery": False,
                "severity": "medium"
            }
        
        # Memory/resource errors
        elif any(keyword in error_message_lower for keyword in ["memory", "resource", "capacity"]):
            return {
                "category": "resource_error",
                "strategy": "optimize_resource_usage",
                "auto_recovery": True,
                "severity": "high"
            }
        
        # Unknown errors
        else:
            return {
                "category": "unknown_error",
                "strategy": "general_troubleshooting",
                "auto_recovery": False,
                "severity": "medium"
            }
    
    def register_custom_analyzer(self, analyzer: 'BaseAnalyzer') -> Dict[str, Any]:
        """
        Dynamically register a custom analyzer with the orchestrator.
        
        Args:
            analyzer: BaseAnalyzer instance to register
            
        Returns:
            Dictionary containing registration results
        """
        try:
            # Validate analyzer
            if not hasattr(analyzer, 'analyze') or not hasattr(analyzer, 'get_recommendations'):
                return {
                    "status": "error",
                    "message": "Analyzer must implement analyze() and get_recommendations() methods",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add performance optimization components if available
            if self.performance_monitor:
                analyzer.performance_monitor = self.performance_monitor
            if self.memory_manager:
                analyzer.memory_manager = self.memory_manager
            
            # Register with the analysis engine
            self.analysis_engine.analyzer_registry.register(analyzer)
            
            # Update analysis priorities if not already defined
            analysis_type = analyzer.analysis_type
            if analysis_type not in self.analysis_engine.analysis_priorities:
                self.analysis_engine.analysis_priorities[analysis_type] = {
                    "priority": 2,  # Default medium priority
                    "cost_impact": "unknown",
                    "execution_time_estimate": 30,
                    "dependencies": [],
                    "description": f"Custom analyzer: {analyzer.__class__.__name__}"
                }
            
            logger.info(f"Successfully registered custom analyzer: {analysis_type}")
            
            return {
                "status": "success",
                "message": f"Custom analyzer '{analysis_type}' registered successfully",
                "analyzer_type": analysis_type,
                "analyzer_class": analyzer.__class__.__name__,
                "total_analyzers": len(self.analysis_engine.analyzer_registry.list_analyzers()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error registering custom analyzer: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to register custom analyzer: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def unregister_analyzer(self, analysis_type: str) -> Dict[str, Any]:
        """
        Unregister an analyzer from the orchestrator.
        
        Args:
            analysis_type: Type of analysis to unregister
            
        Returns:
            Dictionary containing unregistration results
        """
        try:
            if analysis_type not in self.analysis_engine.analyzer_registry.list_analyzers():
                return {
                    "status": "error",
                    "message": f"Analyzer '{analysis_type}' is not registered",
                    "available_analyzers": self.analysis_engine.analyzer_registry.list_analyzers(),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Remove from registry
            del self.analysis_engine.analyzer_registry._analyzers[analysis_type]
            
            # Remove from priorities if it exists
            if analysis_type in self.analysis_engine.analysis_priorities:
                del self.analysis_engine.analysis_priorities[analysis_type]
            
            logger.info(f"Successfully unregistered analyzer: {analysis_type}")
            
            return {
                "status": "success",
                "message": f"Analyzer '{analysis_type}' unregistered successfully",
                "remaining_analyzers": self.analysis_engine.analyzer_registry.list_analyzers(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error unregistering analyzer {analysis_type}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to unregister analyzer '{analysis_type}': {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_analyzer_execution_history(self, analysis_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution history for analyzers.
        
        Args:
            analysis_type: Optional specific analyzer type to get history for
            
        Returns:
            Dictionary containing execution history
        """
        try:
            if analysis_type:
                # Get history for specific analyzer
                analyzer = self.analysis_engine.analyzer_registry.get(analysis_type)
                if not analyzer:
                    return {
                        "status": "error",
                        "message": f"Analyzer '{analysis_type}' not found",
                        "available_analyzers": self.analysis_engine.analyzer_registry.list_analyzers()
                    }
                
                return {
                    "status": "success",
                    "analysis_type": analysis_type,
                    "execution_count": analyzer.execution_count,
                    "last_execution": analyzer.last_execution.isoformat() if analyzer.last_execution else None,
                    "analyzer_info": analyzer.get_analyzer_info()
                }
            else:
                # Get history for all analyzers
                all_history = {}
                for analyzer_type in self.analysis_engine.analyzer_registry.list_analyzers():
                    analyzer = self.analysis_engine.analyzer_registry.get(analyzer_type)
                    all_history[analyzer_type] = {
                        "execution_count": analyzer.execution_count,
                        "last_execution": analyzer.last_execution.isoformat() if analyzer.last_execution else None,
                        "analyzer_class": analyzer.__class__.__name__
                    }
                
                # Add engine-level execution history
                engine_history = getattr(self.analysis_engine, 'execution_history', [])
                
                return {
                    "status": "success",
                    "analyzer_history": all_history,
                    "engine_execution_history": engine_history[-10:],  # Last 10 executions
                    "total_analyzers": len(all_history),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting analyzer execution history: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get execution history: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _attempt_auto_recovery(self, analysis_type: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt automatic recovery from analyzer failures.
        
        Args:
            analysis_type: Type of analysis that failed
            error: Exception that occurred
            context: Analysis context
            
        Returns:
            Dictionary containing recovery attempt results
        """
        recovery_result = {
            "attempted": True,
            "success": False,
            "strategy_used": "none",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            error_message = str(error).lower()
            
            # Timeout recovery: reduce scope and retry
            if "timeout" in error_message:
                recovery_result["strategy_used"] = "reduce_scope_retry"
                
                # Reduce lookback days
                original_lookback = context.get("lookback_days", 30)
                reduced_lookback = max(7, original_lookback // 2)
                
                # Reduce bucket scope if applicable
                original_buckets = context.get("bucket_names")
                reduced_buckets = original_buckets[:5] if original_buckets and len(original_buckets) > 5 else original_buckets
                
                logger.info(f"Attempting timeout recovery for {analysis_type}: reducing lookback from {original_lookback} to {reduced_lookback} days")
                
                # This would be implemented as a retry mechanism in the actual execution flow
                recovery_result["recovery_parameters"] = {
                    "reduced_lookback_days": reduced_lookback,
                    "reduced_bucket_scope": reduced_buckets,
                    "original_lookback_days": original_lookback,
                    "original_bucket_count": len(original_buckets) if original_buckets else 0
                }
                recovery_result["success"] = True  # Mark as successful strategy identification
            
            # Rate limit recovery: implement backoff
            elif any(keyword in error_message for keyword in ["rate", "throttl", "limit"]):
                recovery_result["strategy_used"] = "exponential_backoff"
                
                # Calculate backoff delay
                backoff_delay = min(60, 2 ** context.get("retry_count", 0))
                
                recovery_result["recovery_parameters"] = {
                    "backoff_delay_seconds": backoff_delay,
                    "retry_count": context.get("retry_count", 0) + 1,
                    "max_retries": 3
                }
                recovery_result["success"] = True
            
            # Memory recovery: trigger cleanup
            elif "memory" in error_message:
                recovery_result["strategy_used"] = "memory_cleanup"
                
                if self.memory_manager:
                    cleanup_result = self.memory_manager.force_cleanup("aggressive")
                    recovery_result["recovery_parameters"] = {
                        "cleanup_performed": True,
                        "cleanup_result": cleanup_result
                    }
                    recovery_result["success"] = True
                else:
                    recovery_result["recovery_parameters"] = {
                        "cleanup_performed": False,
                        "reason": "memory_manager_not_available"
                    }
            
            # Network recovery: simple retry with delay
            elif any(keyword in error_message for keyword in ["network", "connection", "dns"]):
                recovery_result["strategy_used"] = "network_retry"
                recovery_result["recovery_parameters"] = {
                    "retry_delay_seconds": 10,
                    "retry_count": context.get("retry_count", 0) + 1,
                    "max_retries": 2
                }
                recovery_result["success"] = True
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Error during auto-recovery attempt: {str(recovery_error)}")
            recovery_result["error"] = str(recovery_error)
            return recovery_result
    
    def _prepare_analysis_params(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Prepare parameters for specific analysis type."""
        base_params = {
            'region': self.region,
            'session_id': self.session_id,
            'lookback_days': kwargs.get('lookback_days', 30),
            'include_cost_analysis': kwargs.get('include_cost_analysis', True),
            'bucket_names': kwargs.get('bucket_names'),
            'timeout_seconds': kwargs.get('timeout_seconds', 60)
        }
        
        # Add analysis-specific parameters
        if analysis_type == "storage_class":
            base_params.update({
                'min_object_size_mb': kwargs.get('min_object_size_mb', 1),
                'include_recommendations': kwargs.get('include_recommendations', True)
            })
        elif analysis_type == "archive_optimization":
            base_params.update({
                'min_age_days': kwargs.get('min_age_days', 180),
                'archive_tier_preference': kwargs.get('archive_tier_preference', 'auto'),
                'include_compliance_check': kwargs.get('include_compliance_check', True)
            })
        elif analysis_type == "api_cost":
            base_params.update({
                'request_threshold': kwargs.get('request_threshold', 10000),
                'include_cloudfront_analysis': kwargs.get('include_cloudfront_analysis', True)
            })
        elif analysis_type == "multipart_cleanup":
            base_params.update({
                'min_age_days': kwargs.get('min_age_days', 7),
                'max_results_per_bucket': kwargs.get('max_results_per_bucket', 1000)
            })
        elif analysis_type == "governance":
            base_params.update({
                'check_lifecycle_policies': kwargs.get('check_lifecycle_policies', True),
                'check_versioning': kwargs.get('check_versioning', True),
                'check_tagging': kwargs.get('check_tagging', True)
            })
        
        return base_params
    
    def _prepare_comprehensive_analysis_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare parameters for comprehensive analysis."""
        return {
            'region': self.region,
            'session_id': self.session_id,
            'lookback_days': kwargs.get('lookback_days', 30),
            'include_cost_analysis': kwargs.get('include_cost_analysis', True),
            'bucket_names': kwargs.get('bucket_names'),
            'timeout_seconds': kwargs.get('timeout_seconds', 60),
            'store_results': kwargs.get('store_results', True),
            'include_cross_analysis': kwargs.get('include_cross_analysis', True)
        }
    
    def _create_prioritized_analysis_tasks(self, 
                                         analysis_types: List[str], 
                                         available_analyses: List[Dict[str, Any]], 
                                         **kwargs) -> List[Dict[str, Any]]:
        """
        Create prioritized analysis tasks for parallel execution based on cost impact and execution time.
        
        Args:
            analysis_types: List of analysis types to execute
            available_analyses: List of analysis metadata with priority information
            **kwargs: Analysis parameters
            
        Returns:
            List of prioritized service call definitions for parallel execution
        """
        service_calls = []
        
        # Create priority mapping for quick lookup
        priority_map = {
            analysis["analysis_type"]: analysis 
            for analysis in available_analyses
        }
        
        # Sort analysis types by priority (highest first), then by execution time (shortest first)
        sorted_analyses = sorted(
            analysis_types,
            key=lambda x: (
                priority_map.get(x, {}).get("priority", 1),
                -priority_map.get(x, {}).get("execution_time_estimate", 30)  # Negative for ascending order
            ),
            reverse=True
        )
        
        logger.info(f"Task prioritization order: {sorted_analyses}")
        
        # Create service call definitions with proper prioritization
        for i, analysis_type in enumerate(sorted_analyses):
            analysis_info = priority_map.get(analysis_type, {})
            
            # Create synchronous wrapper function for the analysis (ServiceOrchestrator expects sync functions)
            def create_analysis_wrapper(atype=analysis_type, params=kwargs.copy()):
                def analysis_wrapper():
                    # Run async analysis in sync context
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
                            # If loop is already running, we need to use a different approach
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, self.analysis_engine.run_analysis(atype, **params))
                                return future.result(timeout=params.get("timeout_seconds", 60))
                        else:
                            return loop.run_until_complete(self.analysis_engine.run_analysis(atype, **params))
                    except Exception as e:
                        logger.error(f"Error in analysis wrapper for {atype}: {str(e)}")
                        return {
                            "status": "error",
                            "analysis_type": atype,
                            "error_message": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                return analysis_wrapper
            
            # Calculate dynamic timeout based on priority and estimated execution time
            base_timeout = analysis_info.get("execution_time_estimate", 30)
            priority_multiplier = 1.0 + (analysis_info.get("priority", 1) * 0.2)  # Higher priority gets more time
            dynamic_timeout = int(base_timeout * priority_multiplier) + 15  # Add buffer
            
            service_call = {
                "service": "s3_analysis_engine",
                "operation": analysis_type,
                "function": create_analysis_wrapper(),
                "args": (),
                "kwargs": {},
                "timeout": kwargs.get("timeout_seconds", dynamic_timeout),
                "priority": analysis_info.get("priority", 1),
                "metadata": {
                    "analysis_type": analysis_type,
                    "cost_impact": analysis_info.get("cost_impact", "unknown"),
                    "execution_time_estimate": analysis_info.get("execution_time_estimate", 30),
                    "execution_order": i + 1,
                    "dependencies": analysis_info.get("dependencies", []),
                    "description": analysis_info.get("description", "S3 optimization analysis"),
                    "dynamic_timeout": dynamic_timeout,
                    "priority_multiplier": priority_multiplier
                }
            }
            
            service_calls.append(service_call)
        
        return service_calls
    
    def _get_task_prioritization_summary(self, available_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of task prioritization for reporting.
        
        Args:
            available_analyses: List of analysis metadata
            
        Returns:
            Dictionary containing prioritization summary
        """
        prioritization_summary = {
            "high_priority_analyses": [],
            "medium_priority_analyses": [],
            "low_priority_analyses": [],
            "prioritization_criteria": {
                "cost_impact": "Higher cost impact gets higher priority",
                "execution_time": "Shorter execution time gets higher priority within same cost impact",
                "dependencies": "Analyses with dependencies are scheduled after their dependencies"
            }
        }
        
        for analysis in available_analyses:
            priority = analysis.get("priority", 1)
            analysis_summary = {
                "analysis_type": analysis["analysis_type"],
                "priority": priority,
                "cost_impact": analysis.get("cost_impact", "unknown"),
                "execution_time_estimate": analysis.get("execution_time_estimate", 30)
            }
            
            if priority >= 4:
                prioritization_summary["high_priority_analyses"].append(analysis_summary)
            elif priority >= 2:
                prioritization_summary["medium_priority_analyses"].append(analysis_summary)
            else:
                prioritization_summary["low_priority_analyses"].append(analysis_summary)
        
        return prioritization_summary
    
    def _execute_cross_analysis_queries(self, stored_tables: List[str]) -> Dict[str, Any]:
        """
        Execute cross-analysis aggregation queries for deeper insights.
        
        Args:
            stored_tables: List of table names stored in the session
            
        Returns:
            Dictionary containing cross-analysis query results
        """
        cross_analysis_results = {}
        
        try:
            # Import aggregation queries
            from .s3_aggregation_queries import S3AggregationQueries
            
            # Get cross-analysis queries based on available tables
            aggregation_queries = self._get_cross_analysis_queries_for_tables(stored_tables)
            
            if aggregation_queries:
                logger.info(f"Executing {len(aggregation_queries)} cross-analysis queries")
                
                # Execute aggregation queries using ServiceOrchestrator
                cross_analysis_results = self.service_orchestrator.aggregate_results(aggregation_queries)
                
                logger.info(f"Completed cross-analysis queries: {len(cross_analysis_results)} results")
            else:
                logger.info("No cross-analysis queries available for current table set")
                
        except Exception as e:
            logger.error(f"Error executing cross-analysis queries: {str(e)}")
            cross_analysis_results = {"error": str(e)}
        
        return cross_analysis_results
    
    def _get_cross_analysis_queries_for_tables(self, stored_tables: List[str]) -> List[Dict[str, str]]:
        """
        Get cross-analysis queries adapted for the specific tables available in the session.
        
        Args:
            stored_tables: List of table names stored in the session
            
        Returns:
            List of query definitions adapted for available tables
        """
        queries = []
        
        # Filter tables to S3-related ones
        s3_tables = [table for table in stored_tables if table.startswith('s3_')]
        
        if not s3_tables:
            return queries
        
        # Get table names by analysis type
        table_map = {}
        for table in s3_tables:
            if 'general_spend' in table:
                table_map['general_spend'] = table
            elif 'storage_class' in table:
                table_map['storage_class'] = table
            elif 'archive_optimization' in table:
                table_map['archive_optimization'] = table
            elif 'api_cost' in table:
                table_map['api_cost'] = table
            elif 'multipart_cleanup' in table:
                table_map['multipart_cleanup'] = table
            elif 'governance' in table:
                table_map['governance'] = table
            elif 'comprehensive' in table:
                table_map['comprehensive'] = table
        
        # Query 1: Recommendations by priority (if we have recommendation tables)
        recommendation_tables = [table for table in s3_tables if 'comprehensive' not in table]
        if recommendation_tables:
            union_clauses = []
            for table in recommendation_tables:
                union_clauses.append(f"""
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM "{table}"
                    WHERE record_type = 'recommendation'
                """)
            
            if union_clauses:
                queries.append({
                    "name": "recommendations_by_priority",
                    "query": f"""
                        SELECT 
                            priority,
                            COUNT(*) as recommendation_count,
                            SUM(CASE WHEN potential_savings > 0 THEN potential_savings ELSE 0 END) as total_potential_savings,
                            AVG(CASE WHEN potential_savings > 0 THEN potential_savings ELSE NULL END) as avg_potential_savings
                        FROM (
                            {' UNION ALL '.join(union_clauses)}
                        ) all_recommendations
                        GROUP BY priority
                        ORDER BY 
                            CASE priority 
                                WHEN 'high' THEN 3 
                                WHEN 'medium' THEN 2 
                                WHEN 'low' THEN 1 
                                ELSE 0 
                            END DESC
                    """
                })
        
        # Query 2: Top optimization opportunities (if we have comprehensive table)
        if 'comprehensive' in table_map:
            queries.append({
                "name": "top_optimization_opportunities",
                "query": f"""
                    SELECT 
                        title,
                        description,
                        potential_savings,
                        source_analysis,
                        implementation_effort,
                        priority,
                        rank
                    FROM "{table_map['comprehensive']}"
                    WHERE record_type = 'optimization_opportunity'
                    ORDER BY potential_savings DESC, rank ASC
                    LIMIT 10
                """
            })
        
        # Query 3: Analysis execution summary
        metadata_tables = [table for table in s3_tables if 'comprehensive' not in table]
        if metadata_tables:
            union_clauses = []
            for table in metadata_tables:
                union_clauses.append(f"""
                    SELECT 
                        analysis_type,
                        status,
                        execution_time,
                        recommendations_count,
                        data_sources,
                        timestamp
                    FROM "{table}"
                    WHERE record_type = 'metadata'
                """)
            
            if union_clauses:
                queries.append({
                    "name": "analysis_execution_summary",
                    "query": f"""
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM (
                            {' UNION ALL '.join(union_clauses)}
                        ) metadata
                        ORDER BY 
                            CASE status 
                                WHEN 'success' THEN 1 
                                WHEN 'error' THEN 2 
                                ELSE 3 
                            END,
                            execution_time ASC
                    """
                })
        
        # Query 4: Total potential savings by analysis type
        if recommendation_tables:
            union_clauses = []
            for table in recommendation_tables:
                union_clauses.append(f"""
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority,
                        implementation_effort
                    FROM "{table}"
                    WHERE record_type = 'recommendation'
                """)
            
            if union_clauses:
                queries.append({
                    "name": "total_savings_by_analysis",
                    "query": f"""
                        SELECT 
                            analysis_type,
                            COUNT(*) as recommendation_count,
                            SUM(CASE WHEN potential_savings > 0 THEN potential_savings ELSE 0 END) as total_potential_savings,
                            AVG(CASE WHEN potential_savings > 0 THEN potential_savings ELSE NULL END) as avg_potential_savings,
                            MAX(potential_savings) as max_potential_savings,
                            COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority_count,
                            COUNT(CASE WHEN priority = 'medium' THEN 1 END) as medium_priority_count,
                            COUNT(CASE WHEN priority = 'low' THEN 1 END) as low_priority_count
                        FROM (
                            {' UNION ALL '.join(union_clauses)}
                        ) all_recommendations
                        GROUP BY analysis_type
                        ORDER BY total_potential_savings DESC
                    """
                })
        
        # Query 5: Cross-analysis insights (if we have comprehensive table)
        if 'comprehensive' in table_map:
            queries.append({
                "name": "cross_analysis_insights",
                "query": f"""
                    SELECT 
                        insight_type,
                        title,
                        description,
                        recommendation,
                        analyses_involved,
                        timestamp
                    FROM "{table_map['comprehensive']}"
                    WHERE record_type = 'cross_analysis_insight'
                    ORDER BY timestamp DESC
                """
            })
        
        logger.info(f"Generated {len(queries)} cross-analysis queries for {len(s3_tables)} S3 tables")
        return queries
    
    def _get_analysis_priority(self, analysis_type: str) -> int:
        """Get priority for analysis type (higher number = higher priority)."""
        return self.analysis_engine.analysis_priorities.get(analysis_type, {}).get("priority", 1)
    
    def _get_cache_ttl_for_analysis(self, analysis_type: str) -> int:
        """
        Get appropriate cache TTL for analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            TTL in seconds
        """
        # Different analysis types have different cache lifetimes
        ttl_mapping = {
            "general_spend": 1800,  # 30 minutes - cost data changes frequently
            "storage_class": 3600,  # 1 hour - storage class analysis is more stable
            "archive_optimization": 7200,  # 2 hours - archive recommendations change slowly
            "api_cost": 1800,  # 30 minutes - API costs can fluctuate
            "multipart_cleanup": 900,  # 15 minutes - multipart uploads change frequently
            "governance": 3600,  # 1 hour - governance policies are relatively stable
            "comprehensive": 1800  # 30 minutes - comprehensive analysis includes dynamic data
        }
        
        return ttl_mapping.get(analysis_type, 1800)  # Default 30 minutes
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary from all optimization components."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "region": self.region,
                "performance_monitor": self.performance_monitor.get_performance_summary(),
                "memory_manager": self.memory_manager.get_memory_statistics(),
                "timeout_handler": self.timeout_handler.get_performance_statistics(),
                "caches": {
                    "pricing_cache": self.pricing_cache.get_statistics(),
                    "bucket_metadata_cache": self.bucket_metadata_cache.get_statistics(),
                    "analysis_results_cache": self.analysis_results_cache.get_statistics()
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def optimize_performance(self):
        """Optimize performance by running optimization routines on all components."""
        try:
            logger.info("Running performance optimization")
            
            # Optimize timeout configuration based on historical data
            self.timeout_handler.optimize_configuration()
            
            # Force gentle memory cleanup
            self.memory_manager.force_cleanup("gentle")
            
            # Optimize caches
            for cache in [self.pricing_cache, self.bucket_metadata_cache, self.analysis_results_cache]:
                if hasattr(cache, '_optimize_cache'):
                    cache._optimize_cache()
            
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")
    
    def clear_caches(self, cache_types: Optional[List[str]] = None):
        """
        Clear specified caches or all caches.
        
        Args:
            cache_types: List of cache types to clear (None for all)
        """
        cache_map = {
            "pricing": self.pricing_cache,
            "bucket_metadata": self.bucket_metadata_cache,
            "analysis_results": self.analysis_results_cache
        }
        
        if cache_types is None:
            cache_types = list(cache_map.keys())
        
        for cache_type in cache_types:
            if cache_type in cache_map:
                cache_map[cache_type].clear()
                logger.info(f"Cleared {cache_type} cache")
            else:
                logger.warning(f"Unknown cache type: {cache_type}")
    

    
    def _store_analysis_results(self, analysis_type: str, result: Dict[str, Any]):
        """Store analysis results in session database with proper schema for cross-analysis queries."""
        try:
            table_name = f"s3_{analysis_type}_{int(time.time())}"
            
            # Convert result to structured format for storage
            data_to_store = []
            
            # Store main result metadata with consistent column names
            data_to_store.append({
                "record_type": "metadata",
                "analysis_type": analysis_type,
                "status": result.get("status"),
                "execution_time": result.get("execution_time", 0),
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "session_id": self.session_id,
                "data_sources": str(result.get("data_sources", [])),
                "recommendations_count": len(result.get("recommendations", [])),
                # Add placeholder columns for cross-analysis compatibility
                "priority": None,
                "title": None,
                "description": None,
                "potential_savings": 0,
                "implementation_effort": None,
                "recommendation_id": None,
                "rec_type": None
            })
            
            # Store recommendations separately for easier querying with consistent schema
            for i, rec in enumerate(result.get("recommendations", [])):
                rec_data = {
                    "record_type": "recommendation",
                    "analysis_type": analysis_type,
                    "recommendation_id": i,
                    "rec_type": rec.get("type", ""),
                    "priority": rec.get("priority", "medium"),
                    "title": rec.get("title", ""),
                    "description": rec.get("description", ""),
                    "potential_savings": rec.get("potential_savings", 0),
                    "implementation_effort": rec.get("implementation_effort", "medium"),
                    "timestamp": datetime.now().isoformat(),
                    # Add metadata columns for consistency
                    "status": None,
                    "execution_time": None,
                    "session_id": self.session_id,
                    "data_sources": None,
                    "recommendations_count": None
                }
                data_to_store.append(rec_data)
            
            # Store analysis data summary with consistent schema
            if result.get("data"):
                data_summary = {
                    "record_type": "data_summary",
                    "analysis_type": analysis_type,
                    "data_keys": str(list(result["data"].keys())),
                    "timestamp": datetime.now().isoformat(),
                    # Add placeholder columns for consistency
                    "status": None,
                    "execution_time": None,
                    "session_id": self.session_id,
                    "data_sources": None,
                    "recommendations_count": None,
                    "priority": None,
                    "title": None,
                    "description": None,
                    "potential_savings": 0,
                    "implementation_effort": None,
                    "recommendation_id": None,
                    "rec_type": None
                }
                data_to_store.append(data_summary)
            
            success = self.session_manager.store_data(
                self.session_id,
                table_name,
                data_to_store
            )
            
            if success:
                logger.info(f"Stored {analysis_type} results ({len(data_to_store)} records) in table {table_name}")
            else:
                logger.warning(f"Failed to store {analysis_type} results")
                
        except Exception as e:
            logger.error(f"Error storing analysis results for {analysis_type}: {str(e)}")
    
    def _store_comprehensive_results(self, aggregated_results: Dict[str, Any], execution_results: Dict[str, Any]):
        """Store comprehensive analysis results with execution metadata in session database."""
        try:
            table_name = f"s3_comprehensive_{int(time.time())}"
            
            data_to_store = []
            
            # Store aggregation metadata with execution information
            metadata = aggregated_results.get("aggregation_metadata", {})
            execution_summary = execution_results.get("execution_summary", execution_results)
            
            data_to_store.append({
                "record_type": "comprehensive_metadata",
                "total_analyses": metadata.get("total_analyses", 0),
                "successful_analyses": metadata.get("successful_analyses", 0),
                "failed_analyses": metadata.get("failed_analyses", 0),
                "total_potential_savings": aggregated_results.get("total_potential_savings", 0),
                "aggregated_at": metadata.get("aggregated_at", datetime.now().isoformat()),
                "session_id": self.session_id,
                "parallel_execution_time": execution_summary.get("total_execution_time", 0),
                "successful_tasks": execution_summary.get("successful", 0),
                "failed_tasks": execution_summary.get("failed", 0),
                "timeout_tasks": execution_summary.get("timeout", 0),
                "stored_tables_count": len(execution_summary.get("stored_tables", []))
            })
            
            # Store execution task details for performance analysis
            for task_id, task_result in execution_summary.get("results", {}).items():
                task_data = {
                    "record_type": "task_execution",
                    "task_id": task_id,
                    "service": task_result.get("service", ""),
                    "operation": task_result.get("operation", ""),
                    "status": task_result.get("status", ""),
                    "execution_time": task_result.get("execution_time", 0),
                    "error": task_result.get("error", ""),
                    "stored_table": task_result.get("stored_table", ""),
                    "timestamp": datetime.now().isoformat()
                }
                data_to_store.append(task_data)
            
            # Store top optimization opportunities with enhanced metadata
            for i, opportunity in enumerate(aggregated_results.get("optimization_opportunities", [])[:10]):
                opp_data = {
                    "record_type": "optimization_opportunity",
                    "rank": opportunity.get("rank", i + 1),
                    "title": opportunity.get("title", ""),
                    "description": opportunity.get("description", ""),
                    "potential_savings": opportunity.get("potential_savings", 0),
                    "implementation_effort": opportunity.get("implementation_effort", "medium"),
                    "source_analysis": opportunity.get("source_analysis", ""),
                    "priority": opportunity.get("priority", "medium"),
                    "timestamp": datetime.now().isoformat()
                }
                data_to_store.append(opp_data)
            
            # Store cross-analysis insights
            for i, insight in enumerate(aggregated_results.get("cross_analysis_insights", [])):
                insight_data = {
                    "record_type": "cross_analysis_insight",
                    "insight_id": i,
                    "insight_type": insight.get("type", ""),
                    "title": insight.get("title", ""),
                    "description": insight.get("description", ""),
                    "recommendation": insight.get("recommendation", ""),
                    "analyses_involved": str(insight.get("analyses_involved", [])),
                    "timestamp": datetime.now().isoformat()
                }
                data_to_store.append(insight_data)
            
            # Store cost insights summary
            cost_insights = aggregated_results.get("cost_insights", {})
            if cost_insights and not cost_insights.get("error"):
                cost_data = {
                    "record_type": "cost_insights_summary",
                    "total_storage_cost": cost_insights.get("total_storage_cost", 0),
                    "total_transfer_cost": cost_insights.get("total_transfer_cost", 0),
                    "total_api_cost": cost_insights.get("total_api_cost", 0),
                    "highest_cost_area": cost_insights.get("highest_cost_areas", [{}])[0].get("0", "Unknown") if cost_insights.get("highest_cost_areas") else "Unknown",
                    "optimization_potential_count": len(cost_insights.get("cost_optimization_potential", {})),
                    "timestamp": datetime.now().isoformat()
                }
                data_to_store.append(cost_data)
            
            success = self.session_manager.store_data(
                self.session_id,
                table_name,
                data_to_store
            )
            
            if success:
                logger.info(f"Stored comprehensive results ({len(data_to_store)} records) in table {table_name}")
            else:
                logger.warning("Failed to store comprehensive results")
                
        except Exception as e:
            logger.error(f"Error storing comprehensive results: {str(e)}")
    
    def get_available_analyses(self) -> List[Dict[str, Any]]:
        """
        Get list of available analyses with metadata.
        
        Returns:
            List of analysis information dictionaries
        """
        return self.analysis_engine.get_available_analyses()
    
    def generate_cross_analyzer_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate cross-analyzer insights by analyzing relationships between different analysis results.
        
        Args:
            analysis_results: Dictionary of analysis results from multiple analyzers
            
        Returns:
            List of cross-analyzer insight dictionaries
        """
        insights = []
        
        try:
            # Extract successful results for cross-analysis
            successful_results = {}
            for task_id, result in analysis_results.items():
                # Handle both TaskResult objects and direct dictionaries
                if hasattr(result, 'data') and result.data and hasattr(result, 'status') and result.status == "success":
                    analysis_type = result.data.get("analysis_type", "unknown")
                    successful_results[analysis_type] = result.data
                elif isinstance(result, dict) and result.get("status") == "success":
                    analysis_type = result.get("analysis_type", "unknown")
                    successful_results[analysis_type] = result
            
            logger.info(f"Generating cross-analyzer insights from {len(successful_results)} successful analyses")
            
            # Insight 1: Storage cost optimization correlation
            if "general_spend" in successful_results and "storage_class" in successful_results:
                spend_data = successful_results["general_spend"].get("data", {})
                storage_data = successful_results["storage_class"].get("data", {})
                
                total_storage_cost = spend_data.get("total_storage_cost", 0)
                storage_recommendations = successful_results["storage_class"].get("recommendations", [])
                storage_savings = sum(rec.get("potential_savings", 0) for rec in storage_recommendations)
                
                if total_storage_cost > 0 and storage_savings > 0:
                    savings_percentage = (storage_savings / total_storage_cost) * 100
                    
                    insights.append({
                        "type": "cost_optimization_correlation",
                        "title": "Storage Class Optimization Impact",
                        "description": f"Storage class optimization could reduce total storage costs by {savings_percentage:.1f}% (${storage_savings:.2f} out of ${total_storage_cost:.2f})",
                        "recommendation": "Prioritize storage class optimization as it offers significant cost reduction potential",
                        "analyses_involved": ["general_spend", "storage_class"],
                        "metrics": {
                            "total_storage_cost": total_storage_cost,
                            "potential_storage_savings": storage_savings,
                            "savings_percentage": savings_percentage
                        },
                        "priority": "high" if savings_percentage > 20 else "medium",
                        "confidence": "high"
                    })
            
            # Insight 2: Governance and cost correlation
            if "governance" in successful_results and "multipart_cleanup" in successful_results:
                governance_violations = len(successful_results["governance"].get("recommendations", []))
                multipart_savings = sum(
                    rec.get("potential_savings", 0) 
                    for rec in successful_results["multipart_cleanup"].get("recommendations", [])
                )
                
                if governance_violations > 0 and multipart_savings > 0:
                    insights.append({
                        "type": "governance_cost_correlation",
                        "title": "Governance Gaps Leading to Cost Waste",
                        "description": f"Found {governance_violations} governance violations and ${multipart_savings:.2f} in multipart upload waste",
                        "recommendation": "Implement lifecycle policies to prevent future cost waste from incomplete uploads",
                        "analyses_involved": ["governance", "multipart_cleanup"],
                        "metrics": {
                            "governance_violations": governance_violations,
                            "multipart_waste_cost": multipart_savings
                        },
                        "priority": "high" if multipart_savings > 100 else "medium",
                        "confidence": "high"
                    })
            
            # Insight 3: Archive optimization opportunity
            if "storage_class" in successful_results and "archive_optimization" in successful_results:
                storage_recs = successful_results["storage_class"].get("recommendations", [])
                archive_recs = successful_results["archive_optimization"].get("recommendations", [])
                
                # Look for overlapping optimization opportunities
                storage_savings = sum(rec.get("potential_savings", 0) for rec in storage_recs)
                archive_savings = sum(rec.get("potential_savings", 0) for rec in archive_recs)
                
                if storage_savings > 0 and archive_savings > 0:
                    combined_savings = storage_savings + archive_savings
                    
                    insights.append({
                        "type": "combined_optimization_opportunity",
                        "title": "Combined Storage and Archive Optimization",
                        "description": f"Combining storage class optimization (${storage_savings:.2f}) with archive strategies (${archive_savings:.2f}) could save ${combined_savings:.2f} total",
                        "recommendation": "Implement a phased approach: optimize storage classes first, then implement archive policies for long-term data",
                        "analyses_involved": ["storage_class", "archive_optimization"],
                        "metrics": {
                            "storage_class_savings": storage_savings,
                            "archive_savings": archive_savings,
                            "combined_savings": combined_savings
                        },
                        "priority": "high" if combined_savings > 500 else "medium",
                        "confidence": "medium"
                    })
            
            # Insight 4: API cost vs storage cost balance
            if "general_spend" in successful_results and "api_cost" in successful_results:
                spend_data = successful_results["general_spend"].get("data", {})
                api_recs = successful_results["api_cost"].get("recommendations", [])
                
                total_api_cost = spend_data.get("total_api_cost", 0)
                total_storage_cost = spend_data.get("total_storage_cost", 0)
                api_savings = sum(rec.get("potential_savings", 0) for rec in api_recs)
                
                if total_api_cost > 0 and total_storage_cost > 0:
                    api_percentage = (total_api_cost / (total_api_cost + total_storage_cost)) * 100
                    
                    if api_percentage > 30:  # API costs are significant
                        insights.append({
                            "type": "cost_distribution_analysis",
                            "title": "High API Cost Ratio Detected",
                            "description": f"API costs represent {api_percentage:.1f}% of total S3 costs (${total_api_cost:.2f} API vs ${total_storage_cost:.2f} storage)",
                            "recommendation": "Focus on API cost optimization through caching, request consolidation, and CloudFront integration",
                            "analyses_involved": ["general_spend", "api_cost"],
                            "metrics": {
                                "api_cost_percentage": api_percentage,
                                "total_api_cost": total_api_cost,
                                "total_storage_cost": total_storage_cost,
                                "api_optimization_potential": api_savings
                            },
                            "priority": "high" if api_percentage > 50 else "medium",
                            "confidence": "high"
                        })
            
            # Insight 5: Comprehensive optimization priority ranking
            if len(successful_results) >= 3:
                # Calculate total potential savings by analysis type
                savings_by_analysis = {}
                for analysis_type, result in successful_results.items():
                    recommendations = result.get("recommendations", [])
                    total_savings = sum(rec.get("potential_savings", 0) for rec in recommendations)
                    savings_by_analysis[analysis_type] = total_savings
                
                # Sort by savings potential
                sorted_savings = sorted(savings_by_analysis.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_savings and sorted_savings[0][1] > 0:
                    top_analysis = sorted_savings[0][0]
                    top_savings = sorted_savings[0][1]
                    total_savings = sum(savings_by_analysis.values())
                    
                    insights.append({
                        "type": "optimization_priority_ranking",
                        "title": "Optimization Priority Recommendations",
                        "description": f"Based on potential savings analysis, prioritize {top_analysis} optimization (${top_savings:.2f} of ${total_savings:.2f} total potential)",
                        "recommendation": f"Start with {top_analysis} optimization for maximum impact, then proceed with other optimizations in order of savings potential",
                        "analyses_involved": list(successful_results.keys()),
                        "metrics": {
                            "priority_ranking": [{"analysis": analysis, "savings": savings} for analysis, savings in sorted_savings],
                            "total_potential_savings": total_savings,
                            "top_opportunity": {"analysis": top_analysis, "savings": top_savings}
                        },
                        "priority": "high",
                        "confidence": "high"
                    })
            
            # Insight 6: Data freshness and analysis reliability
            analysis_timestamps = {}
            for analysis_type, result in successful_results.items():
                timestamp = result.get("timestamp")
                if timestamp:
                    analysis_timestamps[analysis_type] = timestamp
            
            if len(analysis_timestamps) > 1:
                # Check for timestamp consistency (all analyses should be recent)
                from datetime import datetime, timedelta
                current_time = datetime.now()
                old_analyses = []
                
                for analysis_type, timestamp_str in analysis_timestamps.items():
                    try:
                        analysis_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if current_time - analysis_time > timedelta(hours=24):
                            old_analyses.append(analysis_type)
                    except:
                        pass
                
                if old_analyses:
                    insights.append({
                        "type": "data_freshness_warning",
                        "title": "Analysis Data Freshness Concern",
                        "description": f"Some analyses may be using stale data: {', '.join(old_analyses)}",
                        "recommendation": "Re-run analyses with stale data to ensure recommendations are based on current information",
                        "analyses_involved": old_analyses,
                        "metrics": {
                            "stale_analyses": old_analyses,
                            "total_analyses": len(analysis_timestamps)
                        },
                        "priority": "medium",
                        "confidence": "medium"
                    })
            
            logger.info(f"Generated {len(insights)} cross-analyzer insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cross-analyzer insights: {str(e)}")
            return [{
                "type": "insight_generation_error",
                "title": "Cross-Analyzer Insight Generation Failed",
                "description": f"Failed to generate cross-analyzer insights: {str(e)}",
                "recommendation": "Review analysis results manually for optimization opportunities",
                "analyses_involved": list(analysis_results.keys()) if analysis_results else [],
                "priority": "low",
                "confidence": "low"
            }]
    
    def aggregate_results_with_insights(self, results: Dict[str, Any], include_cross_analysis: bool = True) -> Dict[str, Any]:
        """
        Enhanced result aggregation with cross-analyzer insights and comprehensive analysis.
        
        Args:
            results: Dictionary of analysis results by task ID
            include_cross_analysis: Whether to include cross-analysis insights
            
        Returns:
            Enhanced aggregated analysis results
        """
        try:
            # Start with base aggregation from analysis engine
            base_aggregation = self.analysis_engine.aggregate_analysis_results(results, include_cross_analysis)
            
            # Enhance with orchestrator-level insights
            if include_cross_analysis:
                cross_insights = self.generate_cross_analyzer_insights(results)
                base_aggregation["cross_analysis_insights"] = cross_insights
                
                # Update optimization opportunities with cross-analyzer insights
                insight_opportunities = []
                for insight in cross_insights:
                    if insight.get("type") in ["cost_optimization_correlation", "combined_optimization_opportunity", "optimization_priority_ranking"]:
                        opportunity = {
                            "title": insight["title"],
                            "description": insight["description"],
                            "recommendation": insight["recommendation"],
                            "potential_savings": insight.get("metrics", {}).get("combined_savings") or insight.get("metrics", {}).get("potential_storage_savings", 0),
                            "priority": insight["priority"],
                            "source_analysis": "cross_analyzer_insight",
                            "analyses_involved": insight["analyses_involved"],
                            "confidence": insight.get("confidence", "medium"),
                            "insight_type": insight["type"]
                        }
                        insight_opportunities.append(opportunity)
                
                # Merge with existing opportunities
                existing_opportunities = base_aggregation.get("optimization_opportunities", [])
                all_opportunities = existing_opportunities + insight_opportunities
                
                # Sort by potential savings and priority
                priority_weights = {"high": 3, "medium": 2, "low": 1}
                all_opportunities.sort(
                    key=lambda x: (
                        priority_weights.get(x.get("priority", "medium"), 2),
                        x.get("potential_savings", 0)
                    ),
                    reverse=True
                )
                
                base_aggregation["optimization_opportunities"] = all_opportunities
            
            # Add orchestrator-specific metadata
            base_aggregation["orchestrator_metadata"] = {
                "orchestrator_class": "S3OptimizationOrchestrator",
                "session_id": self.session_id,
                "region": self.region,
                "performance_optimizations_enabled": True,
                "cross_analyzer_insights_count": len(base_aggregation.get("cross_analysis_insights", [])),
                "enhanced_aggregation": True,
                "aggregation_timestamp": datetime.now().isoformat()
            }
            
            # Add performance summary if available
            if self.performance_monitor:
                try:
                    performance_summary = self.get_performance_summary()
                    base_aggregation["performance_summary"] = performance_summary
                except Exception as e:
                    logger.warning(f"Could not include performance summary: {str(e)}")
            
            return base_aggregation
            
        except Exception as e:
            logger.error(f"Error in enhanced result aggregation: {str(e)}")
            # Fall back to base aggregation
            try:
                return self.analysis_engine.aggregate_analysis_results(results, include_cross_analysis)
            except Exception as fallback_error:
                logger.error(f"Fallback aggregation also failed: {str(fallback_error)}")
                return {
                    "status": "error",
                    "message": f"Result aggregation failed: {str(e)}",
                    "fallback_error": str(fallback_error),
                    "timestamp": datetime.now().isoformat()
                }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get status information about the analysis engine.
        
        Returns:
            Dictionary containing engine status
        """
        return self.analysis_engine.get_engine_status()
    
    def cleanup_session(self):
        """Clean up the orchestrator session."""
        try:
            if self.service_orchestrator:
                self.service_orchestrator.cleanup_session()
            if self.analysis_engine:
                self.analysis_engine.cleanup()
            logger.info("S3OptimizationOrchestrator session cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up S3OptimizationOrchestrator session: {str(e)}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        try:
            session_info = self.service_orchestrator.get_session_info()
            session_info["analysis_engine_status"] = self.analysis_engine.get_engine_status()
            return session_info
        except Exception as e:
            logger.error(f"Error getting session info: {str(e)}")
            return {"error": f"Failed to get session info: {str(e)}"}
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the S3OptimizationOrchestrator including analyzer integration.
        
        Returns:
            Dictionary containing complete orchestrator status
        """
        try:
            status = {
                "orchestrator_info": {
                    "class_name": "S3OptimizationOrchestrator",
                    "session_id": self.session_id,
                    "region": self.region,
                    "initialized_at": datetime.now().isoformat(),
                    "performance_optimizations_enabled": True
                },
                "analyzer_integration": {},
                "service_integration": {},
                "performance_components": {},
                "session_integration": {},
                "capabilities": {},
                "health_status": "unknown"
            }
            
            # Get analyzer integration status
            try:
                analyzer_registry_info = self.get_analyzer_registry()
                status["analyzer_integration"] = {
                    "status": "active" if analyzer_registry_info.get("registry_status") == "active" else "inactive",
                    "total_analyzers": analyzer_registry_info.get("total_analyzers", 0),
                    "registered_analyzers": list(analyzer_registry_info.get("registered_analyzers", {}).keys()),
                    "analyzer_health": analyzer_registry_info.get("analyzer_health", {}),
                    "analysis_priorities_configured": len(self.analysis_engine.analysis_priorities) > 0
                }
            except Exception as e:
                status["analyzer_integration"] = {"status": "error", "error": str(e)}
            
            # Get service integration status
            try:
                engine_status = self.analysis_engine.get_engine_status()
                status["service_integration"] = {
                    "s3_service": engine_status.get("services_status", {}).get("s3_service", False),
                    "pricing_service": engine_status.get("services_status", {}).get("pricing_service", False),
                    "storage_lens_service": engine_status.get("services_status", {}).get("storage_lens_service", False),
                    "service_orchestrator": self.service_orchestrator is not None,
                    "analysis_engine": self.analysis_engine is not None
                }
            except Exception as e:
                status["service_integration"] = {"status": "error", "error": str(e)}
            
            # Get performance components status
            try:
                status["performance_components"] = {
                    "performance_monitor": self.performance_monitor is not None,
                    "memory_manager": self.memory_manager is not None,
                    "timeout_handler": self.timeout_handler is not None,
                    "pricing_cache": self.pricing_cache is not None,
                    "bucket_metadata_cache": self.bucket_metadata_cache is not None,
                    "analysis_results_cache": self.analysis_results_cache is not None
                }
                
                # Add cache statistics if available
                if self.pricing_cache:
                    status["performance_components"]["pricing_cache_stats"] = self.pricing_cache.get_statistics()
                if self.bucket_metadata_cache:
                    status["performance_components"]["bucket_cache_stats"] = self.bucket_metadata_cache.get_statistics()
                if self.analysis_results_cache:
                    status["performance_components"]["results_cache_stats"] = self.analysis_results_cache.get_statistics()
                    
            except Exception as e:
                status["performance_components"] = {"status": "error", "error": str(e)}
            
            # Get session integration status
            try:
                session_info = self.service_orchestrator.get_session_info()
                stored_tables = self.get_stored_tables()
                s3_tables = [table for table in stored_tables if table.startswith('s3_')]
                
                status["session_integration"] = {
                    "session_active": session_info.get("error") is None,
                    "session_id": self.session_id,
                    "total_tables": len(stored_tables),
                    "s3_analysis_tables": len(s3_tables),
                    "session_manager_available": self.session_manager is not None,
                    "cross_analysis_ready": len(s3_tables) > 0
                }
            except Exception as e:
                status["session_integration"] = {"status": "error", "error": str(e)}
            
            # Get capabilities
            try:
                available_analyses = self.analysis_engine.get_available_analyses()
                status["capabilities"] = {
                    "total_analysis_types": len(available_analyses),
                    "available_analyses": [analysis["analysis_type"] for analysis in available_analyses],
                    "parallel_execution": True,
                    "cross_analyzer_insights": True,
                    "performance_optimizations": True,
                    "dynamic_analyzer_loading": True,
                    "comprehensive_error_handling": True,
                    "session_sql_integration": True,
                    "intelligent_caching": True,
                    "memory_management": True,
                    "progressive_timeouts": True
                }
            except Exception as e:
                status["capabilities"] = {"status": "error", "error": str(e)}
            
            # Determine overall health status
            try:
                analyzer_ok = status["analyzer_integration"].get("status") == "active"
                services_ok = all(status["service_integration"].values()) if isinstance(status["service_integration"], dict) else False
                performance_ok = any(status["performance_components"].values()) if isinstance(status["performance_components"], dict) else False
                session_ok = status["session_integration"].get("session_active", False)
                
                if analyzer_ok and services_ok and performance_ok and session_ok:
                    status["health_status"] = "healthy"
                elif analyzer_ok and services_ok:
                    status["health_status"] = "functional"
                else:
                    status["health_status"] = "degraded"
                    
            except Exception as e:
                status["health_status"] = "unknown"
                status["health_check_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting orchestrator status: {str(e)}")
            return {
                "orchestrator_info": {
                    "class_name": "S3OptimizationOrchestrator",
                    "session_id": self.session_id,
                    "region": self.region
                },
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_parallel_execution_status(self) -> Dict[str, Any]:
        """
        Get detailed status of parallel execution framework integration.
        
        Returns:
            Dictionary containing parallel execution status and metrics
        """
        try:
            # Get parallel executor status
            parallel_executor = self.service_orchestrator.parallel_executor
            executor_status = parallel_executor.get_status()
            
            # Get session manager status
            session_info = self.session_manager.get_session_info(self.session_id)
            
            # Get analysis engine status
            engine_status = self.analysis_engine.get_engine_status()
            
            # Get stored tables for cross-analysis
            stored_tables = self.get_stored_tables()
            s3_tables = [table for table in stored_tables if table.startswith('s3_')]
            
            integration_status = {
                "integration_info": {
                    "orchestrator_class": "S3OptimizationOrchestrator",
                    "session_id": self.session_id,
                    "region": self.region,
                    "integration_complete": True,
                    "timestamp": datetime.now().isoformat()
                },
                "parallel_executor": {
                    "status": "active" if executor_status.get("executor_alive", False) else "inactive",
                    "max_workers": executor_status.get("max_workers", 0),
                    "active_tasks": executor_status.get("active_tasks", 0),
                    "completed_tasks": executor_status.get("completed_tasks", 0),
                    "status_breakdown": executor_status.get("status_breakdown", {})
                },
                "session_sql_integration": {
                    "session_active": session_info.get("error") is None,
                    "session_created_at": session_info.get("created_at"),
                    "session_last_accessed": session_info.get("last_accessed"),
                    "total_tables": len(stored_tables),
                    "s3_analysis_tables": len(s3_tables),
                    "stored_table_names": s3_tables[:10]  # Show first 10 S3 tables
                },
                "analysis_engine": {
                    "registered_analyzers": len(engine_status.get("registered_analyzers", {})),
                    "analyzer_types": list(engine_status.get("registered_analyzers", {}).keys()),
                    "execution_history_count": len(engine_status.get("execution_history", [])),
                    "services_available": engine_status.get("services_status", {})
                },
                "task_prioritization": {
                    "prioritization_enabled": True,
                    "priority_factors": [
                        "cost_impact",
                        "execution_time_estimate", 
                        "dependency_level",
                        "context_adjustments"
                    ],
                    "analysis_priorities": self.analysis_engine.analysis_priorities
                },
                "cross_analysis_capabilities": {
                    "cross_analysis_enabled": True,
                    "aggregation_queries_available": len(s3_tables) > 0,
                    "insight_generation_enabled": True,
                    "available_query_types": [
                        "recommendations_by_priority",
                        "top_optimization_opportunities", 
                        "analysis_execution_summary",
                        "total_savings_by_analysis",
                        "cross_analysis_insights"
                    ] if s3_tables else []
                },
                "integration_health": {
                    "all_components_active": all([
                        executor_status.get("executor_alive", False),
                        session_info.get("error") is None,
                        len(engine_status.get("registered_analyzers", {})) > 0
                    ]),
                    "ready_for_analysis": True,
                    "estimated_capacity": executor_status.get("max_workers", 0),
                    "last_health_check": datetime.now().isoformat()
                }
            }
            
            return integration_status
            
        except Exception as e:
            logger.error(f"Error getting parallel execution status: {str(e)}")
            return {
                "integration_info": {
                    "orchestrator_class": "S3OptimizationOrchestrator",
                    "session_id": self.session_id,
                    "integration_complete": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "error": f"Failed to get parallel execution status: {str(e)}"
            }
    
    def validate_analyzer_integration(self) -> Dict[str, Any]:
        """
        Validate that all analyzers are properly integrated and functional.
        
        Returns:
            Dictionary containing analyzer integration validation results
        """
        validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "analyzer_validations": {},
            "integration_tests": {},
            "recommendations": []
        }
        
        try:
            # Validate each registered analyzer
            for analysis_type in self.analysis_engine.analyzer_registry.list_analyzers():
                analyzer = self.analysis_engine.analyzer_registry.get(analysis_type)
                
                analyzer_validation = {
                    "status": "unknown",
                    "checks": {},
                    "errors": [],
                    "warnings": []
                }
                
                # Check 1: Analyzer instance validity
                try:
                    if analyzer and hasattr(analyzer, 'analyze') and hasattr(analyzer, 'get_recommendations'):
                        analyzer_validation["checks"]["instance_valid"] = True
                    else:
                        analyzer_validation["checks"]["instance_valid"] = False
                        analyzer_validation["errors"].append("Analyzer missing required methods")
                except Exception as e:
                    analyzer_validation["checks"]["instance_valid"] = False
                    analyzer_validation["errors"].append(f"Instance validation error: {str(e)}")
                
                # Check 2: Service dependencies
                try:
                    services_available = {
                        "s3_service": analyzer.s3_service is not None,
                        "pricing_service": analyzer.pricing_service is not None,
                        "storage_lens_service": analyzer.storage_lens_service is not None
                    }
                    analyzer_validation["checks"]["services"] = services_available
                    
                    missing_services = [service for service, available in services_available.items() if not available]
                    if missing_services:
                        analyzer_validation["warnings"].append(f"Missing services: {', '.join(missing_services)}")
                except Exception as e:
                    analyzer_validation["checks"]["services"] = {}
                    analyzer_validation["errors"].append(f"Service check error: {str(e)}")
                
                # Check 3: Performance optimization components
                try:
                    performance_components = {
                        "performance_monitor": hasattr(analyzer, 'performance_monitor') and analyzer.performance_monitor is not None,
                        "memory_manager": hasattr(analyzer, 'memory_manager') and analyzer.memory_manager is not None
                    }
                    analyzer_validation["checks"]["performance_components"] = performance_components
                    
                    missing_components = [comp for comp, available in performance_components.items() if not available]
                    if missing_components:
                        analyzer_validation["warnings"].append(f"Missing performance components: {', '.join(missing_components)}")
                except Exception as e:
                    analyzer_validation["checks"]["performance_components"] = {}
                    analyzer_validation["errors"].append(f"Performance component check error: {str(e)}")
                
                # Check 4: Priority configuration
                try:
                    priority_info = self.analysis_engine.analysis_priorities.get(analysis_type, {})
                    analyzer_validation["checks"]["priority_configured"] = bool(priority_info)
                    
                    if not priority_info:
                        analyzer_validation["warnings"].append("No priority configuration found")
                    else:
                        required_fields = ["priority", "cost_impact", "execution_time_estimate"]
                        missing_fields = [field for field in required_fields if field not in priority_info]
                        if missing_fields:
                            analyzer_validation["warnings"].append(f"Missing priority fields: {', '.join(missing_fields)}")
                except Exception as e:
                    analyzer_validation["checks"]["priority_configured"] = False
                    analyzer_validation["errors"].append(f"Priority check error: {str(e)}")
                
                # Check 5: Parameter validation capability
                try:
                    test_params = {"region": "us-east-1", "lookback_days": 30}
                    validation_result = analyzer.validate_parameters(**test_params)
                    analyzer_validation["checks"]["parameter_validation"] = validation_result.get("valid", False)
                    
                    if not validation_result.get("valid", False):
                        analyzer_validation["errors"].append("Parameter validation failed")
                except Exception as e:
                    analyzer_validation["checks"]["parameter_validation"] = False
                    analyzer_validation["errors"].append(f"Parameter validation check error: {str(e)}")
                
                # Determine analyzer status
                if analyzer_validation["errors"]:
                    analyzer_validation["status"] = "invalid"
                elif analyzer_validation["warnings"]:
                    analyzer_validation["status"] = "partial"
                else:
                    analyzer_validation["status"] = "valid"
                
                validation_results["analyzer_validations"][analysis_type] = analyzer_validation
            
            # Run integration tests
            validation_results["integration_tests"] = self._run_analyzer_integration_tests()
            
            # Determine overall status
            analyzer_statuses = [val["status"] for val in validation_results["analyzer_validations"].values()]
            
            if all(status == "valid" for status in analyzer_statuses):
                if validation_results["integration_tests"].get("all_tests_passed", False):
                    validation_results["overall_status"] = "valid"
                else:
                    validation_results["overall_status"] = "partial"
                    validation_results["recommendations"].append("Some integration tests failed")
            elif any(status == "invalid" for status in analyzer_statuses):
                validation_results["overall_status"] = "invalid"
                validation_results["recommendations"].append("Critical analyzer validation failures detected")
            else:
                validation_results["overall_status"] = "partial"
                validation_results["recommendations"].append("Some analyzers have warnings or missing components")
            
            # Add specific recommendations
            for analysis_type, analyzer_val in validation_results["analyzer_validations"].items():
                if analyzer_val["status"] == "invalid":
                    validation_results["recommendations"].append(f"Fix critical issues with {analysis_type} analyzer")
                elif analyzer_val["warnings"]:
                    validation_results["recommendations"].append(f"Address warnings for {analysis_type} analyzer")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during analyzer integration validation: {str(e)}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    def _run_analyzer_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests for analyzer functionality.
        
        Returns:
            Dictionary containing integration test results
        """
        test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": {},
            "all_tests_passed": False
        }
        
        try:
            # Test 1: Analyzer registry functionality
            test_results["tests_run"] += 1
            try:
                analyzer_list = self.analysis_engine.analyzer_registry.list_analyzers()
                if len(analyzer_list) > 0:
                    test_results["tests_passed"] += 1
                    test_results["test_details"]["registry_functionality"] = {
                        "status": "passed",
                        "message": f"Registry contains {len(analyzer_list)} analyzers"
                    }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["registry_functionality"] = {
                        "status": "failed",
                        "message": "No analyzers registered"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["registry_functionality"] = {
                    "status": "failed",
                    "message": f"Registry test error: {str(e)}"
                }
            
            # Test 2: Priority system functionality
            test_results["tests_run"] += 1
            try:
                available_analyses = self.analysis_engine.get_available_analyses()
                if len(available_analyses) > 0 and all("priority" in analysis for analysis in available_analyses):
                    test_results["tests_passed"] += 1
                    test_results["test_details"]["priority_system"] = {
                        "status": "passed",
                        "message": f"Priority system working for {len(available_analyses)} analyses"
                    }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["priority_system"] = {
                        "status": "failed",
                        "message": "Priority system not properly configured"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["priority_system"] = {
                    "status": "failed",
                    "message": f"Priority system test error: {str(e)}"
                }
            
            # Test 3: Task creation functionality
            test_results["tests_run"] += 1
            try:
                analyzer_types = self.analysis_engine.analyzer_registry.list_analyzers()
                if analyzer_types:
                    tasks = self.analysis_engine.create_parallel_analysis_tasks(
                        analyzer_types[:2],  # Test with first 2 analyzers
                        region="us-east-1",
                        lookback_days=7
                    )
                    if len(tasks) > 0:
                        test_results["tests_passed"] += 1
                        test_results["test_details"]["task_creation"] = {
                            "status": "passed",
                            "message": f"Successfully created {len(tasks)} parallel tasks"
                        }
                    else:
                        test_results["tests_failed"] += 1
                        test_results["test_details"]["task_creation"] = {
                            "status": "failed",
                            "message": "No tasks created"
                        }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["task_creation"] = {
                        "status": "failed",
                        "message": "No analyzers available for task creation"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["task_creation"] = {
                    "status": "failed",
                    "message": f"Task creation test error: {str(e)}"
                }
            
            # Test 4: Cross-analyzer insight generation (mock test)
            test_results["tests_run"] += 1
            try:
                # Create mock results for testing
                mock_results = {
                    "task1": {
                        "status": "success",
                        "analysis_type": "general_spend",
                        "data": {"total_storage_cost": 1000},
                        "recommendations": [{"potential_savings": 100}]
                    },
                    "task2": {
                        "status": "success", 
                        "analysis_type": "storage_class",
                        "data": {"optimization_opportunities": 5},
                        "recommendations": [{"potential_savings": 200}]
                    }
                }
                
                insights = self.generate_cross_analyzer_insights(mock_results)
                if len(insights) > 0:
                    test_results["tests_passed"] += 1
                    test_results["test_details"]["cross_analyzer_insights"] = {
                        "status": "passed",
                        "message": f"Generated {len(insights)} cross-analyzer insights"
                    }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["cross_analyzer_insights"] = {
                        "status": "failed",
                        "message": "No cross-analyzer insights generated"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["cross_analyzer_insights"] = {
                    "status": "failed",
                    "message": f"Cross-analyzer insights test error: {str(e)}"
                }
            
            # Determine overall test status
            test_results["all_tests_passed"] = test_results["tests_failed"] == 0
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running analyzer integration tests: {str(e)}")
            return {
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 1,
                "test_details": {"error": {"status": "failed", "message": str(e)}},
                "all_tests_passed": False
            }
    
    def validate_integration(self) -> Dict[str, Any]:
        """
        Validate that all components of the parallel execution framework are properly integrated.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "component_validations": {},
            "integration_tests": {},
            "recommendations": []
        }
        
        try:
            # Validate ServiceOrchestrator integration
            try:
                orchestrator_info = self.service_orchestrator.get_session_info()
                validation_results["component_validations"]["service_orchestrator"] = {
                    "status": "valid",
                    "session_active": True,
                    "message": "ServiceOrchestrator integration successful"
                }
            except Exception as e:
                validation_results["component_validations"]["service_orchestrator"] = {
                    "status": "invalid",
                    "error": str(e),
                    "message": "ServiceOrchestrator integration failed"
                }
                validation_results["recommendations"].append("Check ServiceOrchestrator initialization")
            
            # Validate ParallelExecutor integration
            try:
                executor_status = self.service_orchestrator.parallel_executor.get_status()
                if executor_status.get("executor_alive", False):
                    validation_results["component_validations"]["parallel_executor"] = {
                        "status": "valid",
                        "max_workers": executor_status.get("max_workers", 0),
                        "message": "ParallelExecutor integration successful"
                    }
                else:
                    validation_results["component_validations"]["parallel_executor"] = {
                        "status": "invalid",
                        "message": "ParallelExecutor is not active"
                    }
                    validation_results["recommendations"].append("Restart ParallelExecutor")
            except Exception as e:
                validation_results["component_validations"]["parallel_executor"] = {
                    "status": "invalid",
                    "error": str(e),
                    "message": "ParallelExecutor integration failed"
                }
                validation_results["recommendations"].append("Check ParallelExecutor configuration")
            
            # Validate SessionManager integration
            try:
                session_info = self.session_manager.get_session_info(self.session_id)
                if session_info.get("error") is None:
                    validation_results["component_validations"]["session_manager"] = {
                        "status": "valid",
                        "session_id": self.session_id,
                        "message": "SessionManager integration successful"
                    }
                else:
                    validation_results["component_validations"]["session_manager"] = {
                        "status": "invalid",
                        "error": session_info.get("error"),
                        "message": "SessionManager session invalid"
                    }
                    validation_results["recommendations"].append("Recreate session or check SessionManager")
            except Exception as e:
                validation_results["component_validations"]["session_manager"] = {
                    "status": "invalid",
                    "error": str(e),
                    "message": "SessionManager integration failed"
                }
                validation_results["recommendations"].append("Check SessionManager initialization")
            
            # Validate S3AnalysisEngine integration
            try:
                engine_status = self.analysis_engine.get_engine_status()
                analyzer_count = len(engine_status.get("registered_analyzers", {}))
                if analyzer_count > 0:
                    validation_results["component_validations"]["analysis_engine"] = {
                        "status": "valid",
                        "registered_analyzers": analyzer_count,
                        "message": "S3AnalysisEngine integration successful"
                    }
                else:
                    validation_results["component_validations"]["analysis_engine"] = {
                        "status": "invalid",
                        "message": "No analyzers registered in S3AnalysisEngine"
                    }
                    validation_results["recommendations"].append("Check analyzer registration")
            except Exception as e:
                validation_results["component_validations"]["analysis_engine"] = {
                    "status": "invalid",
                    "error": str(e),
                    "message": "S3AnalysisEngine integration failed"
                }
                validation_results["recommendations"].append("Check S3AnalysisEngine initialization")
            
            # Run integration tests
            validation_results["integration_tests"] = self._run_integration_tests()
            
            # Determine overall status
            all_valid = all(
                comp.get("status") == "valid" 
                for comp in validation_results["component_validations"].values()
            )
            
            if all_valid and validation_results["integration_tests"].get("all_tests_passed", False):
                validation_results["overall_status"] = "valid"
            elif all_valid:
                validation_results["overall_status"] = "partial"
                validation_results["recommendations"].append("Some integration tests failed")
            else:
                validation_results["overall_status"] = "invalid"
                validation_results["recommendations"].append("Critical component validation failures")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during integration validation: {str(e)}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests to verify parallel execution framework functionality.
        
        Returns:
            Dictionary containing test results
        """
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": {},
            "all_tests_passed": False
        }
        
        try:
            # Test 1: Session data storage and retrieval
            test_results["tests_run"] += 1
            try:
                test_data = [{"test_key": "test_value", "timestamp": datetime.now().isoformat()}]
                success = self.session_manager.store_data(self.session_id, "integration_test", test_data)
                
                if success:
                    query_result = self.service_orchestrator.query_session_data(
                        "SELECT * FROM integration_test LIMIT 1"
                    )
                    if query_result and len(query_result) > 0:
                        test_results["tests_passed"] += 1
                        test_results["test_details"]["session_storage"] = {
                            "status": "passed",
                            "message": "Session storage and retrieval working"
                        }
                    else:
                        test_results["tests_failed"] += 1
                        test_results["test_details"]["session_storage"] = {
                            "status": "failed",
                            "message": "Data stored but query failed"
                        }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["session_storage"] = {
                        "status": "failed",
                        "message": "Failed to store test data"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["session_storage"] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Test 2: Analyzer registry functionality
            test_results["tests_run"] += 1
            try:
                available_analyzers = self.analysis_engine.analyzer_registry.list_analyzers()
                if len(available_analyzers) >= 6:  # Should have all 6 S3 analyzers
                    test_results["tests_passed"] += 1
                    test_results["test_details"]["analyzer_registry"] = {
                        "status": "passed",
                        "analyzer_count": len(available_analyzers),
                        "analyzers": available_analyzers
                    }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["analyzer_registry"] = {
                        "status": "failed",
                        "message": f"Expected 6 analyzers, found {len(available_analyzers)}"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["analyzer_registry"] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Test 3: Task prioritization
            test_results["tests_run"] += 1
            try:
                test_analyses = ["general_spend", "storage_class", "governance"]
                priority_order = self.analysis_engine._calculate_advanced_task_priority(test_analyses)
                
                if len(priority_order) == len(test_analyses) and all("priority_score" in item for item in priority_order):
                    test_results["tests_passed"] += 1
                    test_results["test_details"]["task_prioritization"] = {
                        "status": "passed",
                        "priority_order": [item["analysis_type"] for item in priority_order]
                    }
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"]["task_prioritization"] = {
                        "status": "failed",
                        "message": "Task prioritization failed"
                    }
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"]["task_prioritization"] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Determine overall test status
            test_results["all_tests_passed"] = test_results["tests_failed"] == 0
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running integration tests: {str(e)}")
            test_results["error"] = str(e)
            return test_results
# =============================================================================
# MCP Wrapper Functions
# =============================================================================

async def run_s3_comprehensive_analysis(arguments: Dict[str, Any]) -> List[Any]:
    """Run comprehensive S3 cost optimization analysis."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute comprehensive analysis
        result = await orchestrator.execute_comprehensive_analysis(**arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text", 
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "comprehensive_analysis"
            }, indent=2)
        }]

async def run_s3_general_spend_analysis(arguments: Dict[str, Any]) -> List[Any]:
    """Analyze overall S3 spending patterns and usage."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute general spend analysis
        result = await orchestrator.execute_analysis("general_spend", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "run_s3_general_spend_analysis"
            }, indent=2)
        }]

async def run_s3_storage_class_selection(arguments: Dict[str, Any]) -> List[Any]:
    """Provide guidance on choosing the most cost-effective storage class."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute storage class analysis (covers both selection and validation)
        result = await orchestrator.execute_analysis("storage_class", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "storage_class_selection"
            }, indent=2)
        }]

async def run_s3_storage_class_validation(arguments: Dict[str, Any]) -> List[Any]:
    """Validate that existing data is stored in the most appropriate storage class."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute storage class analysis (covers both selection and validation)
        result = await orchestrator.execute_analysis("storage_class", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "storage_class_validation"
            }, indent=2)
        }]

async def run_s3_archive_optimization(arguments: Dict[str, Any]) -> List[Any]:
    """Identify and optimize long-term archive data storage."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute archive optimization
        result = await orchestrator.execute_analysis("archive_optimization", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "archive_optimization"
            }, indent=2)
        }]

async def run_s3_api_cost_minimization(arguments: Dict[str, Any]) -> List[Any]:
    """Minimize S3 API request charges through access pattern optimization."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute API cost analysis
        result = await orchestrator.execute_analysis("api_cost", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "api_cost_minimization"
            }, indent=2)
        }]

async def run_s3_multipart_cleanup(arguments: Dict[str, Any]) -> List[Any]:
    """Identify and clean up incomplete multipart uploads."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute multipart cleanup analysis
        result = await orchestrator.execute_analysis("multipart_cleanup", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "multipart_cleanup"
            }, indent=2)
        }]

async def run_s3_governance_check(arguments: Dict[str, Any]) -> List[Any]:
    """Implement S3 cost controls and governance policy compliance checking."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute governance check
        result = await orchestrator.execute_analysis("governance", **arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "run_s3_governance_check"
            }, indent=2)
        }]

async def run_s3_comprehensive_optimization_tool(arguments: Dict[str, Any]) -> List[Any]:
    """Run comprehensive S3 optimization with unified tool."""
    try:
        from .s3_comprehensive_optimization_tool import S3ComprehensiveOptimizationTool
        
        region = arguments.get('region')
        tool = S3ComprehensiveOptimizationTool(region=region)
        
        # Execute comprehensive optimization
        result = await tool.execute_comprehensive_optimization(**arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "run_s3_comprehensive_optimization_tool.execution"
            }, indent=2)
        }]

async def run_s3_quick_analysis(arguments: Dict[str, Any]) -> List[Any]:
    """Run a quick S3 analysis focusing on the most impactful optimizations."""
    try:
        region = arguments.get('region')
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute a subset of high-impact analyses with short timeout
        quick_analyses = ["general_spend", "multipart_cleanup", "governance"]
        
        results = {}
        for analysis_type in quick_analyses:
            try:
                result = await orchestrator.execute_analysis(
                    analysis_type, 
                    timeout_seconds=10,  # Quick timeout
                    **arguments
                )
                results[analysis_type] = result
            except Exception as e:
                results[analysis_type] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Aggregate quick results
        quick_result = {
            "status": "success",
            "analysis_type": "quick_analysis",
            "results": results,
            "message": f"Quick analysis completed for {len(quick_analyses)} analyses"
        }
        
        # Add documentation links
        quick_result = add_documentation_links(quick_result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(quick_result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "quick_analysis"
            }, indent=2)
        }]

async def run_s3_bucket_analysis(arguments: Dict[str, Any]) -> List[Any]:
    """Analyze specific S3 buckets for optimization opportunities."""
    try:
        region = arguments.get('region')
        bucket_names = arguments.get('bucket_names', [])
        
        if not bucket_names:
            return [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": "bucket_names parameter is required",
                    "context": "bucket_analysis"
                }, indent=2)
            }]
        
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Execute comprehensive analysis for specific buckets
        result = await orchestrator.execute_comprehensive_analysis(**arguments)
        
        # Add documentation links
        result = add_documentation_links(result, "s3")
        
        return [{
            "type": "text",
            "text": json.dumps(result, indent=2, default=str)
        }]
        
    except Exception as e:
        return [{
            "type": "text",
            "text": json.dumps({
                "status": "error",
                "error_code": getattr(e, 'code', 'UnknownError'),
                "message": str(e),
                "context": "bucket_analysis"
            }, indent=2)
        }]