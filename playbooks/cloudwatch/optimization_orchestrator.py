"""
CloudWatch Optimization Orchestrator

Main coordination layer that manages all CloudWatch optimization workflows
with parallel execution, session integration, and performance optimizations.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from playbooks.cloudwatch.analysis_engine import CloudWatchAnalysisEngine
from playbooks.cloudwatch.cost_controller import CostController, CostPreferences
from playbooks.cloudwatch.aggregation_queries import CloudWatchAggregationQueries
from playbooks.cloudwatch.result_processor import CloudWatchResultProcessor
from utils.service_orchestrator import ServiceOrchestrator
from utils.session_manager import get_session_manager
from utils.logging_config import log_cloudwatch_operation
from utils.performance_monitor import get_performance_monitor
from utils.intelligent_cache import get_pricing_cache, get_analysis_results_cache
from utils.memory_manager import get_memory_manager
from utils.progressive_timeout import get_timeout_handler

logger = logging.getLogger(__name__)


class CloudWatchOptimizationOrchestrator:
    """
    Main orchestrator for CloudWatch optimization workflows.
    
    This orchestrator provides:
    - Coordination of all CloudWatch optimization analyses
    - Integration with ServiceOrchestrator for parallel execution
    - Session management for data persistence and querying
    - Cost control and validation
    - Performance monitoring, memory management, and intelligent caching
    - Comprehensive reporting and recommendations
    """
    
    def __init__(self, region: str = None, session_id: str = None):
        """Initialize the CloudWatch Optimization Orchestrator with performance optimizations."""
        self.region = region
        self.session_manager = get_session_manager()
        
        # Initialize ServiceOrchestrator (it will create session if session_id is None)
        self.service_orchestrator = ServiceOrchestrator(session_id)
        
        # Get the actual session ID from ServiceOrchestrator
        self.session_id = self.service_orchestrator.session_id
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance optimization components
        self.performance_monitor = get_performance_monitor()
        self.memory_manager = get_memory_manager()
        self.timeout_handler = get_timeout_handler()
        
        # Initialize caching systems with CloudWatch-specific optimizations
        self.pricing_cache = get_pricing_cache()
        self.analysis_results_cache = get_analysis_results_cache()
        self.cloudwatch_metadata_cache = self._initialize_cloudwatch_metadata_cache()
        
        # Register cache instances with memory manager for cleanup
        self.memory_manager.add_cache_reference(self.pricing_cache)
        self.memory_manager.add_cache_reference(self.analysis_results_cache)
        self.memory_manager.add_cache_reference(self.cloudwatch_metadata_cache)
        
        # Set up performance monitor integration
        self.pricing_cache.set_performance_monitor(self.performance_monitor)
        self.analysis_results_cache.set_performance_monitor(self.performance_monitor)
        self.cloudwatch_metadata_cache.set_performance_monitor(self.performance_monitor)
        self.memory_manager.set_performance_monitor(self.performance_monitor)
        self.timeout_handler.set_performance_monitor(self.performance_monitor)
        
        # Register CloudWatch-specific cache warming functions
        self._register_cache_warming_functions()
        
        # Set up memory management callbacks for large dataset processing
        self._setup_memory_management_callbacks()
        
        # Initialize progressive timeout configuration for CloudWatch analyses
        self._configure_progressive_timeouts()
        
        # Initialize core components with performance optimizations
        self.analysis_engine = CloudWatchAnalysisEngine(
            region=region, 
            session_id=self.session_id,
            performance_monitor=self.performance_monitor,
            memory_manager=self.memory_manager,
            timeout_handler=self.timeout_handler,
            pricing_cache=self.pricing_cache,
            analysis_results_cache=self.analysis_results_cache
        )
        self.cost_controller = CostController()
        self.aggregation_queries = CloudWatchAggregationQueries()
        
        # Initialize result processor for zero-cost sorting and pagination
        self.result_processor = CloudWatchResultProcessor(
            pricing_service=self.analysis_engine.pricing_service if hasattr(self.analysis_engine, 'pricing_service') else None
        )
        
        log_cloudwatch_operation(self.logger, "orchestrator_initialized",
                               region=region, session_id=self.session_id,
                               performance_optimizations_enabled=True,
                               cache_instances=3,
                               memory_management_enabled=True,
                               progressive_timeouts_enabled=True)
    
    def _validate_and_prepare_cost_preferences(self, **kwargs) -> CostPreferences:
        """
        Validate and prepare cost preferences from kwargs.
        
        Args:
            **kwargs: Analysis parameters that may include cost preferences
            
        Returns:
            Validated CostPreferences object
        """
        # Extract cost preference parameters
        cost_params = {
            'allow_cost_explorer': kwargs.get('allow_cost_explorer', False),
            'allow_aws_config': kwargs.get('allow_aws_config', False),
            'allow_cloudtrail': kwargs.get('allow_cloudtrail', False),
            'allow_minimal_cost_metrics': kwargs.get('allow_minimal_cost_metrics', False)
        }
        
        # Validate and sanitize preferences
        validated_preferences = self.cost_controller.validate_and_sanitize_preferences(cost_params)
        
        # Update analysis engine's cost preferences
        self.analysis_engine.cloudwatch_service.update_cost_preferences(validated_preferences)
        
        log_cloudwatch_operation(self.logger, "cost_preferences_validated",
                               preferences=str(validated_preferences))
        
        return validated_preferences
    
    async def execute_comprehensive_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all CloudWatch analyses in parallel with cost controls, routing, and performance optimizations.
        
        This method demonstrates the full cost control and routing system by:
        1. Validating user cost preferences
        2. Creating cost tracking context
        3. Configuring consent-based routing
        4. Executing analyses with appropriate data sources
        5. Generating comprehensive cost transparency report
        6. Performance monitoring and intelligent caching
        
        Args:
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing comprehensive analysis results with cost transparency and performance metrics
        """
        start_time = time.time()
        
        # Start comprehensive performance monitoring
        monitoring_session = self.performance_monitor.start_analysis_monitoring(
            "comprehensive", 
            f"comprehensive_{int(start_time)}"
        )
        
        # Start memory tracking for comprehensive analysis
        memory_tracker = self.memory_manager.start_memory_tracking("comprehensive_analysis")
        
        log_cloudwatch_operation(self.logger, "comprehensive_analysis_start",
                               session_id=self.session_id,
                               performance_monitoring_enabled=True)
        
        try:
            # Check cache for recent comprehensive analysis
            cache_key = ["comprehensive", kwargs.get('region', self.region), kwargs]
            cached_result = self.analysis_results_cache.get(cache_key)
            
            if cached_result is not None:
                self.logger.info("Retrieved comprehensive analysis from cache")
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
            
            # Validate cost preferences
            cost_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
            
            # Create cost tracking context
            cost_tracking_context = self.cost_controller.create_cost_tracking_context(cost_preferences)
            
            # Get execution routing configuration
            routing_config = self.cost_controller.get_execution_path_routing(cost_preferences)
            
            # Get functionality coverage estimate
            functionality_coverage = self.cost_controller.get_functionality_coverage(cost_preferences)
            
            # Calculate intelligent timeout for comprehensive analysis
            comprehensive_timeout = self.timeout_handler.get_timeout_for_analysis("comprehensive", **kwargs)
            kwargs['total_timeout'] = comprehensive_timeout
            
            self.logger.info(f"Calculated comprehensive analysis timeout: {comprehensive_timeout:.1f}s")
            
            # Prepare analysis parameters with performance optimizations
            analysis_params = {
                'session_id': self.session_id,
                'region': self.region,
                'cost_tracking_context': cost_tracking_context,
                'routing_config': routing_config,
                'monitoring_session': monitoring_session,
                'memory_tracker': memory_tracker,
                **kwargs
            }
            
            # Register large object for memory management if available
            if self.memory_manager:
                try:
                    self.memory_manager.register_large_object(
                        f"comprehensive_analysis_{int(start_time)}",
                        analysis_params,
                        size_mb=1.0,  # Estimated size
                        cleanup_callback=lambda: self.logger.debug("Cleaned up comprehensive analysis parameters")
                    )
                except Exception as e:
                    self.logger.warning(f"Could not register large object with memory manager: {str(e)}")
            
            # Execute all analyses based on routing configuration
            analysis_results = {}
            
            # Execute each analysis type with appropriate routing
            analysis_types = ['general_spend', 'logs_optimization', 'metrics_optimization', 'alarms_and_dashboards']
            
            self.logger.info(f"Executing {len(analysis_types)} analyses with intelligent prioritization: {analysis_types}")
            
            for analysis_type in analysis_types:
                try:
                    log_cloudwatch_operation(self.logger, f"executing_{analysis_type}_analysis",
                                           routing_path=routing_config[f"{analysis_type}_analysis"]["primary_path"])
                    
                    result = await self.analysis_engine.run_analysis(analysis_type, **analysis_params)
                    
                    # Apply result processing (cost-based sorting and pagination) if successful
                    if result.get('status') == 'success' and result.get('data'):
                        page = kwargs.get('page', 1)
                        result = self._apply_result_processing(result, page)
                    
                    analysis_results[analysis_type] = result
                    
                except Exception as e:
                    log_cloudwatch_operation(self.logger, f"{analysis_type}_analysis_failed",
                                           error=str(e))
                    analysis_results[analysis_type] = {
                        'status': 'error',
                        'error_message': str(e),
                        'analysis_type': analysis_type
                    }
            
            # Generate comprehensive cost transparency report
            cost_transparency_report = self.cost_controller.generate_cost_transparency_report(cost_tracking_context)
            
            execution_time = time.time() - start_time
            
            # Calculate status based on analysis results
            successful_analyses = len([r for r in analysis_results.values() if r.get('status') != 'error'])
            total_analyses = len(analysis_results)
            failed_analyses = total_analyses - successful_analyses
            
            if failed_analyses == 0:
                overall_status = 'success'
            elif successful_analyses > 0:
                overall_status = 'partial'
            else:
                overall_status = 'error'
            
            # Compile comprehensive result
            comprehensive_result = {
                'status': overall_status,
                'analysis_type': 'comprehensive',
                'successful_analyses': successful_analyses,
                'total_analyses': total_analyses,
                'results': analysis_results,
                'stored_tables': [f"{analysis_type}_results" for analysis_type, result in analysis_results.items() if result.get('status') != 'error'],
                'analysis_results': analysis_results,
                'cost_transparency': cost_transparency_report,
                'functionality_coverage': functionality_coverage,
                'routing_configuration': routing_config,
                'execution_metadata': {
                    'session_id': self.session_id,
                    'region': self.region,
                    'total_execution_time': execution_time,
                    'cost_preferences': cost_preferences.__dict__,
                    'analyses_executed': len(analysis_results),
                    'successful_analyses': successful_analyses,
                    'failed_analyses': failed_analyses,
                    'from_cache': False,
                    'performance_optimizations': {
                        'intelligent_timeout': comprehensive_timeout,
                        'cache_enabled': True,
                        'memory_management': True,
                        'performance_monitoring': True
                    }
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
            
            # Store comprehensive results in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'comprehensive_cloudwatch_analysis',
                        comprehensive_result['execution_metadata']
                    )
                except Exception as e:
                    log_cloudwatch_operation(self.logger, "session_storage_failed", error=str(e))
            
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
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_complete",
                                   total_cost=cost_transparency_report['cost_summary']['total_actual_cost'],
                                   analyses_count=len(analysis_results),
                                   execution_time=execution_time,
                                   session_id=self.session_id)
            
            return comprehensive_result
            
        except Exception as e:
            import traceback
            execution_time = time.time() - start_time
            error_message = f"Comprehensive analysis failed: {str(e)}"
            full_traceback = traceback.format_exc()
            
            # Log full error details
            self.logger.error(f"Comprehensive analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            
            # End monitoring with error
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session,
                success=False,
                error_message=str(e)
            )
            self.memory_manager.stop_memory_tracking("comprehensive_analysis")
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_failed",
                                   error=error_message,
                                   execution_time=execution_time,
                                   session_id=self.session_id)
            
            return {
                'status': 'error',
                'error_message': error_message,
                'error_type': e.__class__.__name__,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_location': self._extract_error_location(full_traceback)
                },
                'execution_time': execution_time,
                'session_id': self.session_id
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
    
    def get_analysis_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Query stored analysis results with cost information.
        
        Args:
            query: Query string for retrieving results
            
        Returns:
            List of analysis results with cost transparency information
        """
        try:
            # Use session manager to execute the query
            results = self.session_manager.execute_query(self.session_id, query)
            
            # Add cost control context to results
            for result in results:
                if isinstance(result, dict):
                    result['cost_control_info'] = {
                        'current_preferences': self.cost_controller.default_preferences.__dict__,
                        'functionality_coverage': self.cost_controller.get_functionality_coverage(
                            self.cost_controller.default_preferences
                        ),
                        'query_executed_at': datetime.now().isoformat()
                    }
            
            log_cloudwatch_operation(self.logger, "analysis_results_retrieved",
                                   results_count=len(results),
                                   query=query)
            
            return results
            
        except Exception as e:
            log_cloudwatch_operation(self.logger, "analysis_results_query_failed",
                                   error=str(e), query=query)
            return []
    
    def get_stored_tables(self) -> List[str]:
        """
        Get list of stored tables in the session.
        
        Returns:
            List of table names stored in the current session
        """
        try:
            return self.service_orchestrator.get_stored_tables()
        except Exception as e:
            log_cloudwatch_operation(self.logger, "get_stored_tables_failed", error=str(e))
            return []
    
    def validate_cost_preferences(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and sanitize cost control flags with detailed feedback.
        
        Args:
            **kwargs: Cost preference parameters
            
        Returns:
            Dictionary containing validation results and cost estimates
        """
        try:
            # First, validate the input types and values
            validation_errors = []
            validation_warnings = []
            
            # Check for invalid boolean values
            for key, value in kwargs.items():
                if key.startswith('allow_'):
                    if not isinstance(value, bool):
                        if isinstance(value, str) and value.lower() in ['true', 'false', 'yes', 'no']:
                            # Convert string booleans
                            kwargs[key] = value.lower() in ['true', 'yes']
                            validation_warnings.append(f"Converted string '{value}' to boolean for {key}")
                        elif isinstance(value, int):
                            # Convert integer booleans (0 = False, non-zero = True)
                            kwargs[key] = bool(value)
                            validation_warnings.append(f"Converted integer '{value}' to boolean for {key}")
                        else:
                            validation_errors.append(f"Invalid boolean value for {key}: '{value}'. Expected boolean, 'true'/'false', 'yes'/'no', or 0/1")
                            # Don't modify the value for errors
                elif key not in ['lookback_days', 'log_group_names', 'alarm_names', 'dashboard_names']:
                    validation_warnings.append(f"Unknown preference '{key}' will be ignored")
            
            # If there are validation errors, return failure
            if validation_errors:
                return {
                    'valid': False,
                    'errors': validation_errors,
                    'warnings': validation_warnings,
                    'validation_status': 'failed'
                }
            
            # Validate preferences
            validated_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
            
            # Get functionality coverage
            functionality_coverage = self.cost_controller.get_functionality_coverage(validated_preferences)
            
            # Get cost estimate for typical analysis scope
            typical_scope = {
                'lookback_days': kwargs.get('lookback_days', 30),
                'log_group_names': kwargs.get('log_group_names', []),
                'alarm_names': kwargs.get('alarm_names', []),
                'dashboard_names': kwargs.get('dashboard_names', [])
            }
            cost_estimate = self.cost_controller.estimate_cost(typical_scope, validated_preferences)
            
            # Get execution routing
            routing_config = self.cost_controller.get_execution_path_routing(validated_preferences)
            
            validation_result = {
                'valid': True,
                'errors': [],  # No errors for successful validation
                'warnings': validation_warnings,
                'validated_preferences': validated_preferences.__dict__,
                'sanitized_preferences': validated_preferences.__dict__,  # For backward compatibility
                'functionality_coverage': functionality_coverage,
                'cost_estimate': cost_estimate.__dict__,
                'routing_configuration': routing_config,
                'validation_status': 'success',
                'recommendations': self._get_preference_recommendations(validated_preferences, functionality_coverage)
            }
            
            log_cloudwatch_operation(self.logger, "cost_preferences_validation_complete",
                                   overall_coverage=functionality_coverage['overall_coverage'],
                                   estimated_cost=cost_estimate.total_estimated_cost)
            
            return validation_result
            
        except Exception as e:
            log_cloudwatch_operation(self.logger, "cost_preferences_validation_failed",
                                   error=str(e))
            return {
                'validation_status': 'error',
                'error_message': str(e),
                'default_preferences': self.cost_controller.default_preferences.__dict__
            }
    
    def _get_preference_recommendations(self, preferences: CostPreferences, 
                                      coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for cost preference optimization."""
        recommendations = []
        
        overall_coverage = coverage.get('overall_coverage', 0)
        
        if overall_coverage < 70:
            recommendations.append({
                'type': 'coverage_improvement',
                'priority': 'medium',
                'title': 'Consider enabling Cost Explorer for better coverage',
                'description': f'Current coverage is {overall_coverage:.1f}%. Enabling Cost Explorer would add 30% more functionality.',
                'action': 'Set allow_cost_explorer=True',
                'estimated_additional_cost': 0.01
            })
        
        if not preferences.allow_minimal_cost_metrics and coverage['by_category']['free_operations'] > 50:
            recommendations.append({
                'type': 'feature_enhancement',
                'priority': 'low',
                'title': 'Enable minimal cost metrics for detailed analysis',
                'description': 'Minimal cost metrics provide detailed log ingestion patterns with very low cost impact.',
                'action': 'Set allow_minimal_cost_metrics=True',
                'estimated_additional_cost': 0.01
            })
        
        if all(not getattr(preferences, flag) for flag in ['allow_cost_explorer', 'allow_aws_config', 'allow_cloudtrail', 'allow_minimal_cost_metrics']):
            recommendations.append({
                'type': 'free_tier_notice',
                'priority': 'info',
                'title': 'Using free tier only',
                'description': f'You are using only free operations ({coverage["free_tier_coverage"]:.1f}% coverage). This provides good basic analysis with no additional costs.',
                'action': 'No action needed - current configuration is cost-optimal'
            })
        
        return recommendations
    
    def get_cost_estimate(self, analysis_scope: Dict[str, Any] = None, cost_preferences: 'CostPreferences' = None, **kwargs) -> Dict[str, Any]:
        """
        Provide cost estimate based on enabled features and analysis scope.
        
        Args:
            analysis_scope: Dictionary containing analysis parameters
            cost_preferences: CostPreferences object
            **kwargs: Additional analysis parameters and cost preferences (for backward compatibility)
            
        Returns:
            Dictionary containing detailed cost estimation
        """
        try:
            # Track calling pattern for response format
            kwargs_only_call = analysis_scope is None and cost_preferences is None
            
            # Handle different calling patterns for backward compatibility
            if kwargs_only_call:
                # Called with **kwargs only - validate preferences from kwargs
                validated_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
                
                # Prepare analysis scope from kwargs
                analysis_scope = {
                    'lookback_days': kwargs.get('lookback_days', 30),
                    'log_group_names': kwargs.get('log_group_names', []),
                    'alarm_names': kwargs.get('alarm_names', []),
                    'dashboard_names': kwargs.get('dashboard_names', []),
                    'analysis_types': kwargs.get('analysis_types', ['general_spend', 'logs_optimization', 'metrics_optimization', 'alarms_and_dashboards'])
                }
            else:
                # Called with explicit parameters
                if analysis_scope is None:
                    analysis_scope = {}
                if cost_preferences is None:
                    validated_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
                else:
                    validated_preferences = cost_preferences
            
            # Get cost estimate
            cost_estimate = self.cost_controller.estimate_cost(analysis_scope, validated_preferences)
            
            # Calculate grouped cost breakdown for backward compatibility
            free_operations_cost = 0.0
            paid_operations_cost = 0.0
            
            for operation_name, cost in cost_estimate.cost_breakdown.items():
                if cost == 0.0:
                    free_operations_cost += 1  # Count of free operations
                else:
                    paid_operations_cost += cost
            
            # Return the cost estimate with grouped breakdown for backward compatibility
            result = cost_estimate.__dict__.copy()
            result['cost_breakdown'] = {
                'free_operations': free_operations_cost,
                'paid_operations': paid_operations_cost,
                **cost_estimate.cost_breakdown  # Include individual operations too
            }
            
            # Return different formats based on calling pattern
            if kwargs_only_call:
                # This was called with **kwargs, return nested format
                return {
                    'cost_estimate': result,
                    'analysis_scope': analysis_scope,
                    'cost_preferences': validated_preferences.__dict__,
                    'cost_breakdown_explanation': {
                        'free_operations': [op for op, cost in cost_estimate.cost_breakdown.items() if cost == 0.0],
                        'paid_operations': [op for op, cost in cost_estimate.cost_breakdown.items() if cost > 0.0],
                        'total_free_operations': len([c for c in cost_estimate.cost_breakdown.values() if c == 0.0]),
                        'total_paid_operations': len([c for c in cost_estimate.cost_breakdown.values() if c > 0.0])
                    }
                }
            
            return result
            
        except Exception as e:
            log_cloudwatch_operation(self.logger, "cost_estimate_failed", error=str(e))
            return {
                'error': str(e),
                'default_estimate': {
                    'free_operations_only': 0.0,
                    'message': 'Cost estimation failed, but free operations are always available at no cost'
                }
            }
    
    def _create_session(self) -> str:
        """Create a new session for this orchestrator instance."""
        session_manager = get_session_manager()
        session_id = session_manager.create_session()
        return session_id
    
    def _get_cache_ttl_for_analysis(self, analysis_type: str) -> int:
        """Get appropriate cache TTL for different analysis types."""
        ttl_mapping = {
            'general_spend': 3600,  # 1 hour - cost data changes less frequently
            'logs_optimization': 1800,  # 30 minutes - log patterns change more frequently
            'metrics_optimization': 1800,  # 30 minutes - metrics patterns change frequently
            'alarms_and_dashboards': 2400,  # 40 minutes - alarm configs change less frequently
            'comprehensive': 3600  # 1 hour - comprehensive analysis is expensive
        }
        return ttl_mapping.get(analysis_type, 1800)  # Default 30 minutes
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics from all optimization components."""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'region': self.region,
                'performance_optimizations': {
                    'intelligent_caching': True,
                    'memory_management': True,
                    'progressive_timeouts': True,
                    'cache_warming': True
                }
            }
            
            # Performance monitor statistics
            if self.performance_monitor:
                stats['performance_monitor'] = self.performance_monitor.get_performance_summary()
            
            # Memory manager statistics
            if self.memory_manager:
                stats['memory_manager'] = self.memory_manager.get_memory_statistics()
            
            # Timeout handler statistics
            if self.timeout_handler:
                stats['timeout_handler'] = self.timeout_handler.get_performance_statistics()
            
            # Cache statistics
            cache_stats = {}
            if self.pricing_cache:
                cache_stats['pricing_cache'] = self.pricing_cache.get_statistics()
            
            if self.analysis_results_cache:
                cache_stats['analysis_results_cache'] = self.analysis_results_cache.get_statistics()
            
            if self.cloudwatch_metadata_cache:
                cache_stats['cloudwatch_metadata_cache'] = self.cloudwatch_metadata_cache.get_statistics()
            
            # Calculate overall cache performance
            total_hits = sum(cache.get('cache_performance', {}).get('total_hits', 0) 
                           for cache in cache_stats.values())
            total_misses = sum(cache.get('cache_performance', {}).get('total_misses', 0) 
                             for cache in cache_stats.values())
            total_requests = total_hits + total_misses
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            cache_stats['overall_performance'] = {
                'total_cache_hits': total_hits,
                'total_cache_misses': total_misses,
                'overall_hit_rate_percent': overall_hit_rate,
                'active_caches': len(cache_stats) - 1  # Exclude overall_performance
            }
            
            stats['cache_statistics'] = cache_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting performance statistics: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }
    
    def _initialize_cloudwatch_metadata_cache(self):
        """Initialize CloudWatch-specific metadata cache."""
        from utils.cloudwatch_cache import get_cloudwatch_metadata_cache
        
        # Use specialized CloudWatch metadata cache
        return get_cloudwatch_metadata_cache()
    
    def _register_cache_warming_functions(self):
        """Register CloudWatch-specific cache warming functions."""
        # Register pricing cache warming for CloudWatch services
        self.pricing_cache.register_warming_function("cloudwatch_pricing", self._warm_cloudwatch_pricing)
        
        # Register resource cache warming functions using consolidated method
        for resource_type in ["alarms", "dashboards", "log_groups", "metrics"]:
            self.cloudwatch_metadata_cache.register_warming_function(
                f"{resource_type}_metadata", 
                lambda cache, region=None, rt=resource_type: self._warm_cloudwatch_resource_cache(rt, cache, region)
            )
        
        # Register analysis results cache warming for common queries
        self.analysis_results_cache.register_warming_function("common_analyses", self._warm_common_analyses)
    
    def _setup_memory_management_callbacks(self):
        """Set up memory management callbacks for large dataset processing."""
        # Register cleanup callbacks using consolidated method
        self.memory_manager.register_cleanup_callback(
            lambda: self._cleanup_resources("large_datasets")
        )
        self.memory_manager.register_cleanup_callback(
            lambda: self._cleanup_resources("temporary_data")
        )
        
        # Set memory thresholds specific to CloudWatch analysis workloads
        from utils.memory_manager import MemoryThreshold
        cloudwatch_thresholds = MemoryThreshold(
            warning_percent=75.0,  # CloudWatch analyses can be memory intensive
            critical_percent=85.0,
            cleanup_percent=90.0,
            max_memory_mb=2048  # 2GB limit for CloudWatch analyses
        )
        self.memory_manager.thresholds = cloudwatch_thresholds
    
    def _configure_progressive_timeouts(self):
        """Configure progressive timeout settings for CloudWatch analyses."""
        from utils.progressive_timeout import TimeoutConfiguration, ComplexityLevel
        
        # CloudWatch-specific timeout configuration
        cloudwatch_timeout_config = TimeoutConfiguration(
            base_timeout=45.0,  # CloudWatch APIs can be slower
            complexity_multiplier={
                ComplexityLevel.VERY_LOW: 0.6,
                ComplexityLevel.LOW: 1.0,
                ComplexityLevel.MEDIUM: 1.8,
                ComplexityLevel.HIGH: 3.0,
                ComplexityLevel.VERY_HIGH: 5.0
            },
            data_size_multiplier=0.15,  # CloudWatch data can be large
            bucket_count_multiplier=1.0,  # Not applicable to CloudWatch
            historical_performance_weight=0.4,  # Higher weight for CloudWatch
            system_load_weight=0.3,
            min_timeout=15.0,
            max_timeout=600.0,  # 10 minutes max for complex analyses
            grace_period=20.0
        )
        
        self.timeout_handler.config = cloudwatch_timeout_config
    
    def _warm_cloudwatch_resource_cache(self, resource_type: str, cache, region: str = None):
        """Warm cache with CloudWatch resource metadata."""
        try:
            if not region:
                region = self.region
            
            # Map resource types to cache keys
            cache_key_map = {
                "alarms": "alarms_metadata",
                "dashboards": "dashboards_metadata", 
                "log_groups": "log_groups_metadata",
                "metrics": "metrics_metadata"
            }
            
            cache_key = cache_key_map.get(resource_type)
            if not cache_key:
                raise ValueError(f"Unknown resource type: {resource_type}")
            
            # Use the specialized cache warming method
            cache.warm_cache(cache_key, region=region)
                
        except Exception as e:
            self.logger.error(f"Error warming {resource_type} cache: {str(e)}")
    
    def _warm_cloudwatch_pricing(self, cache, region: str = None):
        """Warm cache with CloudWatch pricing data."""
        try:
            if not region:
                region = self.region
                
            self.logger.info(f"Warming CloudWatch pricing cache for region: {region}")
            
            # Cache common pricing queries
            pricing_keys = [
                f"logs_pricing_{region}",
                f"metrics_pricing_{region}",
                f"alarms_pricing_{region}",
                f"dashboards_pricing_{region}"
            ]
            
            for key in pricing_keys:
                cache.put(key, {"warmed": True, "region": region}, ttl_seconds=3600)
                
        except Exception as e:
            self.logger.error(f"Error warming CloudWatch pricing cache: {str(e)}")
    
    def _warm_common_analyses(self, cache, region: str = None):
        """Warm cache with common analysis patterns."""
        try:
            if not region:
                region = self.region
                
            self.logger.info(f"Warming common analyses cache for region: {region}")
            
            # Cache templates for common analysis patterns
            common_patterns = [
                f"general_spend_template_{region}",
                f"logs_optimization_template_{region}",
                f"metrics_optimization_template_{region}",
                f"alarms_dashboards_template_{region}"
            ]
            
            for pattern in common_patterns:
                cache.put(pattern, {"template": True, "region": region}, ttl_seconds=7200)
                
        except Exception as e:
            self.logger.error(f"Error warming common analyses cache: {str(e)}")
    
    def _cleanup_resources(self, cleanup_type: str):
        """Consolidated cleanup callback for CloudWatch resources."""
        try:
            if cleanup_type == "large_datasets":
                # Clean up large temporary datasets
                self.logger.debug("Cleaning up large analysis datasets")
                
                # Force garbage collection of analysis results
                import gc
                collected = gc.collect()
                self.logger.debug(f"Garbage collected {collected} objects during analysis cleanup")
                
            elif cleanup_type == "temporary_data":
                # Clean up temporary CloudWatch API response data
                self.logger.debug("Cleaning up temporary CloudWatch data")
                
                # Clear any temporary data structures
                if hasattr(self, '_temp_cloudwatch_data'):
                    self._temp_cloudwatch_data.clear()
                    
            else:
                self.logger.warning(f"Unknown cleanup type: {cleanup_type}")
                
        except Exception as e:
            self.logger.error(f"Error in {cleanup_type} cleanup: {str(e)}")
    
    def warm_caches(self, cache_types: List[str] = None) -> Dict[str, Any]:
        """
        Warm caches proactively for better performance.
        
        Args:
            cache_types: List of cache types to warm (None for all)
            
        Returns:
            Dictionary with warming results
        """
        try:
            results = {}
            
            if not cache_types:
                cache_types = ['pricing', 'metadata', 'analysis_results']
            
            # Warm pricing cache
            if 'pricing' in cache_types and self.pricing_cache:
                success = self.pricing_cache.warm_cache("cloudwatch_pricing", region=self.region)
                results['pricing_cache'] = 'warmed' if success else 'failed'
            
            # Warm metadata cache
            if 'metadata' in cache_types and self.cloudwatch_metadata_cache:
                metadata_types = ['alarms_metadata', 'dashboards_metadata', 
                                'log_groups_metadata', 'metrics_metadata']
                metadata_results = {}
                for metadata_type in metadata_types:
                    success = self.cloudwatch_metadata_cache.warm_cache(metadata_type, region=self.region)
                    metadata_results[metadata_type] = 'warmed' if success else 'failed'
                results['metadata_cache'] = metadata_results
            
            # Warm analysis results cache
            if 'analysis_results' in cache_types and self.analysis_results_cache:
                success = self.analysis_results_cache.warm_cache("common_analyses", region=self.region)
                results['analysis_results_cache'] = 'warmed' if success else 'failed'
            
            results['status'] = 'success'
            results['timestamp'] = datetime.now().isoformat()
            
            log_cloudwatch_operation(self.logger, "caches_warmed", 
                                   cache_types=cache_types,
                                   results=results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error warming caches: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_caches(self) -> Dict[str, Any]:
        """Clear all caches and return status."""
        try:
            results = {}
            
            if self.pricing_cache:
                self.pricing_cache.clear()
                results['pricing_cache'] = 'cleared'
            
            if self.analysis_results_cache:
                self.analysis_results_cache.clear()
                results['analysis_results_cache'] = 'cleared'
            
            if self.cloudwatch_metadata_cache:
                self.cloudwatch_metadata_cache.clear()
                results['cloudwatch_metadata_cache'] = 'cleared'
            
            results['status'] = 'success'
            results['timestamp'] = datetime.now().isoformat()
            
            log_cloudwatch_operation(self.logger, "caches_cleared", 
                                   caches_cleared=list(results.keys()))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error clearing caches: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_analysis(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific CloudWatch analysis with cost controls, routing, and performance optimizations.
        
        Args:
            analysis_type: Type of analysis to execute
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing analysis results with cost transparency and performance metrics
        """
        start_time = time.time()
        
        # Start performance monitoring
        monitoring_session = self.performance_monitor.start_analysis_monitoring(
            analysis_type, 
            f"single_{int(start_time)}"
        )
        
        # Start memory tracking
        memory_tracker = self.memory_manager.start_memory_tracking(f"analysis_{analysis_type}")
        
        log_cloudwatch_operation(self.logger, "execute_analysis_start",
                               analysis_type=analysis_type,
                               session_id=self.session_id,
                               performance_monitoring_enabled=True)
        
        try:
            # Check cache for recent results first
            cache_key = [analysis_type, kwargs.get('region', self.region), kwargs]
            cached_result = self.analysis_results_cache.get(cache_key)
            
            if cached_result is not None:
                self.logger.info(f"Retrieved {analysis_type} analysis from cache")
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
            
            # Validate cost preferences first
            cost_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
            
            # Create cost tracking context for this analysis
            cost_tracking_context = self.cost_controller.create_cost_tracking_context(cost_preferences)
            
            # Get execution routing configuration
            routing_config = self.cost_controller.get_execution_path_routing(cost_preferences)
            
            # Calculate intelligent timeout, but respect user-provided timeout if smaller
            calculated_timeout = self.timeout_handler.get_timeout_for_analysis(analysis_type, **kwargs)
            user_timeout = kwargs.get('timeout_seconds')
            
            if user_timeout is not None and user_timeout < calculated_timeout:
                timeout_seconds = user_timeout
                self.logger.info(f"Using user-provided timeout for {analysis_type}: {timeout_seconds:.1f}s (calculated: {calculated_timeout:.1f}s)")
            else:
                timeout_seconds = calculated_timeout
                self.logger.info(f"Calculated timeout for {analysis_type}: {timeout_seconds:.1f}s")
            
            kwargs['timeout_seconds'] = timeout_seconds
            
            # Add orchestrator metadata to kwargs
            kwargs.update({
                'session_id': self.session_id,
                'region': self.region,
                'orchestrator_version': '1.0.0',
                'cost_tracking_context': cost_tracking_context,
                'routing_config': routing_config,
                'monitoring_session': monitoring_session,
                'memory_tracker': memory_tracker
            })
            
            # Execute the analysis through the engine with cost tracking and timeout
            try:
                result = await asyncio.wait_for(
                    self.analysis_engine.run_analysis(analysis_type, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise Exception(f"Analysis {analysis_type} timed out after {timeout_seconds:.1f} seconds")
            
            # Apply result processing (cost-based sorting and pagination) if successful
            if result.get('status') == 'success' and result.get('data'):
                page = kwargs.get('page', 1)
                result = self._apply_result_processing(result, page)
            
            # Generate cost transparency report
            cost_transparency_report = self.cost_controller.generate_cost_transparency_report(cost_tracking_context)
            
            execution_time = time.time() - start_time
            
            # Add orchestrator metadata to result
            result['orchestrator_metadata'] = {
                'session_id': self.session_id,
                'region': self.region,
                'orchestrator_version': '1.0.0',
                'total_orchestration_time': execution_time,
                'cost_preferences': cost_preferences.__dict__,
                'routing_config': routing_config,
                'from_cache': False,
                'performance_optimizations': {
                    'intelligent_timeout': timeout_seconds,
                    'cache_enabled': True,
                    'memory_management': True,
                    'performance_monitoring': True
                }
            }
            
            # Add cost transparency to result
            result['cost_transparency'] = cost_transparency_report
            
            # Preserve cost information from analysis engine result at top level for backward compatibility
            # The analysis engine result should already have cost_incurred, cost_incurring_operations, etc.
            # We don't want to overwrite them with the cost transparency report
            
            # Cache successful results
            if result.get('status') == 'success':
                cache_ttl = self._get_cache_ttl_for_analysis(analysis_type)
                self.analysis_results_cache.put(
                    cache_key, 
                    result.copy(), 
                    ttl_seconds=cache_ttl,
                    tags={"analysis_type": analysis_type, "region": self.region}
                )
                
                self.logger.debug(f"Cached {analysis_type} analysis result (TTL: {cache_ttl}s)")
            
            # Store result summary in session if successful
            if result.get('status') == 'success' and kwargs.get('store_results', True):
                self._store_analysis_summary(analysis_type, result)
            
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
            
            log_cloudwatch_operation(self.logger, "execute_analysis_complete",
                                   analysis_type=analysis_type,
                                   status=result.get('status', 'unknown'),
                                   execution_time=execution_time,
                                   session_id=self.session_id)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Analysis execution failed for {analysis_type}: {str(e)}")
            
            # Handle timeout errors specially
            if isinstance(e, asyncio.TimeoutError):
                error_message = f"Analysis timeout: {str(e)}"
            else:
                error_message = str(e)
            
            # End monitoring with error
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session, 
                success=False, 
                error_message=error_message
            )
            self.memory_manager.stop_memory_tracking(f"analysis_{analysis_type}")
            
            return {
                'status': 'error',
                'error_message': error_message,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'session_id': self.session_id,
                'from_cache': False
            }
    
    def _apply_result_processing(self, result: Dict[str, Any], page: int = 1) -> Dict[str, Any]:
        """
        Apply zero-cost result processing (sorting and pagination) to analysis results.
        
        ZERO-COST GUARANTEE: This method only processes data already retrieved
        from free CloudWatch APIs. No additional AWS API calls are made.
        
        Args:
            result: Analysis result from the analysis engine
            page: 1-based page number for pagination
            
        Returns:
            Result with processed (sorted and paginated) data
        """
        try:
            processed_result = result.copy()
            data = processed_result.get('data', {})
            
            # Process each data category with cost-based sorting and pagination
            # Handle all possible configuration analysis paths
            config_data = None
            if 'configuration_analysis' in data:
                config_data = data['configuration_analysis']
            elif 'metrics_configuration_analysis' in data:
                # Legacy support - convert to new structure
                config_data = {'metrics': data['metrics_configuration_analysis']}
            elif 'log_groups_configuration_analysis' in data:
                # Legacy support - convert to new structure
                config_data = {'log_groups': data['log_groups_configuration_analysis']}
            elif 'alarms_configuration_analysis' in data:
                config_data = data['alarms_configuration_analysis']
            elif 'dashboards_configuration_analysis' in data:
                config_data = data['dashboards_configuration_analysis']
            
            if config_data:
                
                # Process log groups
                if 'log_groups' in config_data and isinstance(config_data['log_groups'], dict):
                    log_groups_list = config_data['log_groups'].get('log_groups', [])
                    if log_groups_list:
                        processed_log_groups = self.result_processor.process_log_groups_results(
                            log_groups_list, page
                        )
                        # PRESERVE original structure, only replace the log_groups array and add pagination
                        config_data['log_groups']['log_groups'] = processed_log_groups['items']
                        config_data['log_groups']['pagination'] = processed_log_groups['pagination']
                        # Keep all other metadata: total_count, retention_analysis, etc.
                
                # Process metrics
                if 'metrics' in config_data and isinstance(config_data['metrics'], dict):
                    metrics_data = config_data['metrics']
                    # Handle nested structure: metrics_data might contain another 'metrics' key
                    if 'metrics' in metrics_data and isinstance(metrics_data['metrics'], dict):
                        # Double nested: metrics_data['metrics']['metrics']
                        metrics_list = metrics_data['metrics'].get('metrics', [])
                    else:
                        # Single nested: metrics_data['metrics'] is the list
                        metrics_list = metrics_data.get('metrics', [])
                    
                    if metrics_list and isinstance(metrics_list, list):
                        processed_metrics = self.result_processor.process_metrics_results(
                            metrics_list, page
                        )
                        # PRESERVE original structure, only replace the metrics array and add pagination
                        if 'metrics' in metrics_data and isinstance(metrics_data['metrics'], dict):
                            # Double nested structure
                            config_data['metrics']['metrics']['metrics'] = processed_metrics['items']
                            config_data['metrics']['metrics']['pagination'] = processed_metrics['pagination']
                        else:
                            # Single nested structure
                            config_data['metrics']['metrics'] = processed_metrics['items']
                            config_data['metrics']['pagination'] = processed_metrics['pagination']
                        # Keep all other metadata: total_count, namespace, filtered, etc.
                
                # Process alarms
                if 'alarms' in config_data and isinstance(config_data['alarms'], dict):
                    alarms_list = config_data['alarms'].get('alarms', [])
                    if alarms_list:
                        processed_alarms = self.result_processor.process_alarms_results(
                            alarms_list, page
                        )
                        # PRESERVE original structure, only replace the alarms array and add pagination
                        config_data['alarms']['alarms'] = processed_alarms['items']
                        config_data['alarms']['pagination'] = processed_alarms['pagination']
                        # Keep all other metadata: total_count, alarm_analysis, etc.
                
                # Process dashboards
                if 'dashboards' in config_data and isinstance(config_data['dashboards'], dict):
                    dashboards_list = config_data['dashboards'].get('dashboards', [])
                    if dashboards_list:
                        processed_dashboards = self.result_processor.process_dashboards_results(
                            dashboards_list, page
                        )
                        # PRESERVE original structure, only replace the dashboards array and add pagination
                        config_data['dashboards']['dashboards'] = processed_dashboards['items']
                        config_data['dashboards']['pagination'] = processed_dashboards['pagination']
                        # Keep all other metadata: total_count, dashboard_analysis, etc.
            
            # Process recommendations if present
            if 'recommendations' in processed_result:
                recommendations = processed_result['recommendations']
                if isinstance(recommendations, list) and recommendations:
                    processed_recommendations = self.result_processor.process_recommendations(
                        recommendations, page
                    )
                    processed_result['recommendations'] = processed_recommendations
            
            # Add pagination metadata to top level
            processed_result['pagination_applied'] = True
            processed_result['current_page'] = page
            
            self.logger.debug(f"Applied result processing with pagination (page {page})")
            
            return processed_result
            
        except Exception as e:
            self.logger.warning(f"Error applying result processing: {str(e)}, returning original result")
            # Return original result if processing fails
            return result
    
    async def execute_comprehensive_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all CloudWatch analyses in parallel with cost controls.
        
        Args:
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        start_time = datetime.now()
        
        log_cloudwatch_operation(self.logger, "comprehensive_analysis_start",
                               session_id=self.session_id)
        
        try:
            # Validate cost preferences
            cost_preferences = self._validate_and_prepare_cost_preferences(**kwargs)
            
            # Get cost estimate for transparency
            analysis_scope = {
                'lookback_days': kwargs.get('lookback_days', 30),
                'log_group_names': kwargs.get('log_group_names', []),
                'alarm_names': kwargs.get('alarm_names', []),
                'dashboard_names': kwargs.get('dashboard_names', [])
            }
            
            cost_estimate = self.cost_controller.estimate_cost(analysis_scope, cost_preferences)
            
            # Add orchestrator metadata to kwargs
            kwargs.update({
                'session_id': self.session_id,
                'region': self.region,
                'orchestrator_version': '1.0.0'
            })
            
            # Execute comprehensive analysis through the engine
            result = await self.analysis_engine.run_comprehensive_analysis(**kwargs)
            
            # Add orchestrator metadata to result
            result['orchestrator_metadata'] = {
                'session_id': self.session_id,
                'region': self.region,
                'orchestrator_version': '1.0.0',
                'total_orchestration_time': (datetime.now() - start_time).total_seconds(),
                'cost_preferences': cost_preferences.__dict__,
                'cost_estimate': cost_estimate.__dict__
            }
            
            # Store comprehensive result summary
            if result.get('status') in ['success', 'partial'] and kwargs.get('store_results', True):
                self._store_comprehensive_summary(result)
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_complete",
                                   status=result.get('status', 'unknown'),
                                   total_analyses=result.get('analysis_summary', {}).get('total_analyses', 0),
                                   successful=result.get('analysis_summary', {}).get('successful_analyses', 0),
                                   session_id=self.session_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis execution failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'comprehensive',
                'timestamp': start_time.isoformat(),
                'session_id': self.session_id
            }
    
    def _validate_and_prepare_cost_preferences(self, **kwargs) -> CostPreferences:
        """Validate and prepare cost preferences from kwargs."""
        raw_preferences = {
            'allow_cost_explorer': kwargs.get('allow_cost_explorer', False),
            'allow_aws_config': kwargs.get('allow_aws_config', False),
            'allow_cloudtrail': kwargs.get('allow_cloudtrail', False),
            'allow_minimal_cost_metrics': kwargs.get('allow_minimal_cost_metrics', False),
        }
        
        return self.cost_controller.validate_and_sanitize_preferences(raw_preferences)
    
    def _store_analysis_summary(self, analysis_type: str, result: Dict[str, Any]) -> None:
        """Store analysis summary in session for querying."""
        try:
            summary_data = [{
                'analysis_type': analysis_type,
                'status': result.get('status'),
                'timestamp': result.get('timestamp'),
                'execution_time': result.get('execution_time', 0),
                'cost_incurred': result.get('cost_incurred', False),
                'primary_data_source': result.get('primary_data_source'),
                'fallback_used': result.get('fallback_used', False),
                'session_id': self.session_id,
                'region': self.region
            }]
            
            table_name = f"analysis_summary_{int(datetime.now().timestamp())}"
            success = self.session_manager.store_data(
                self.session_id, table_name, summary_data
            )
            
            if success:
                log_cloudwatch_operation(self.logger, "analysis_summary_stored",
                                       analysis_type=analysis_type, table_name=table_name)
            
        except Exception as e:
            self.logger.error(f"Error storing analysis summary: {str(e)}")
    
    def _store_comprehensive_summary(self, result: Dict[str, Any]) -> None:
        """Store comprehensive analysis summary in session."""
        try:
            summary_data = [{
                'analysis_type': 'comprehensive',
                'status': result.get('status'),
                'timestamp': result.get('timestamp'),
                'total_execution_time': result.get('total_execution_time', 0),
                'cost_incurred': result.get('cost_incurred', False),
                'primary_data_source': result.get('primary_data_source'),
                'fallback_used': result.get('fallback_used', False),
                'total_analyses': result.get('analysis_summary', {}).get('total_analyses', 0),
                'successful_analyses': result.get('analysis_summary', {}).get('successful_analyses', 0),
                'failed_analyses': result.get('analysis_summary', {}).get('failed_analyses', 0),
                'session_id': self.session_id,
                'region': self.region
            }]
            
            table_name = f"comprehensive_summary_{int(datetime.now().timestamp())}"
            success = self.session_manager.store_data(
                self.session_id, table_name, summary_data
            )
            
            if success:
                log_cloudwatch_operation(self.logger, "comprehensive_summary_stored",
                                       table_name=table_name)
            
        except Exception as e:
            self.logger.error(f"Error storing comprehensive summary: {str(e)}")
    
    def get_analysis_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Query stored analysis results using SQL.
        
        Args:
            query: SQL query to execute on session data
            
        Returns:
            List of query results enriched with cost control information
        """
        try:
            results = self.service_orchestrator.query_session_data(query)
            
            # Enrich results with cost control information
            enriched_results = []
            for result in results:
                enriched_result = result.copy()
                enriched_result["cost_control_info"] = {
                    "current_preferences": {
                        "allow_cost_explorer": self.cost_controller.default_preferences.allow_cost_explorer,
                        "allow_aws_config": self.cost_controller.default_preferences.allow_aws_config,
                        "allow_cloudtrail": self.cost_controller.default_preferences.allow_cloudtrail,
                        "allow_minimal_cost_metrics": self.cost_controller.default_preferences.allow_minimal_cost_metrics
                    },
                    "region": self.region,
                    "session_id": self.session_id,
                    "query_timestamp": datetime.now().isoformat()
                }
                enriched_results.append(enriched_result)
            
            log_cloudwatch_operation(self.logger, "query_executed",
                                   query_length=len(query), results_count=len(enriched_results))
            return enriched_results
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            return []
    


    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        try:
            session_info = self.session_manager.get_session_info(self.session_id)
            stored_tables = self.service_orchestrator.get_stored_tables()
            
            return {
                'session_id': self.session_id,
                'region': self.region,
                'session_info': session_info,
                'stored_tables': stored_tables,
                'orchestrator_version': '1.0.0'
            }
        except Exception as e:
            return {
                'error': str(e),
                'session_id': self.session_id
            }
    
    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis types."""
        return self.analysis_engine.get_available_analyses()
    
    def get_analysis_info(self, analysis_type: str) -> Dict[str, Any]:
        """Get information about a specific analysis type."""
        return self.analysis_engine.get_analysis_info(analysis_type)
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and health information."""
        try:
            engine_status = self.analysis_engine.get_engine_status()
            session_info = self.get_session_info()
            
            return {
                'orchestrator_version': '1.0.0',
                'session_id': self.session_id,
                'region': self.region,
                'engine_status': engine_status,
                'session_status': session_info,
                'health': 'healthy'
            }
        except Exception as e:
            return {
                'orchestrator_version': '1.0.0',
                'session_id': self.session_id,
                'region': self.region,
                'health': 'error',
                'error': str(e)
            }
    
    def cleanup_session(self) -> None:
        """Clean up the current session and resources."""
        try:
            self.service_orchestrator.cleanup_session()
            log_cloudwatch_operation(self.logger, "session_cleanup_complete",
                                   session_id=self.session_id)
        except Exception as e:
            self.logger.error(f"Error during session cleanup: {str(e)}")
    
    def execute_cross_analysis_insights(self, **kwargs) -> Dict[str, Any]:
        """
        Execute cross-analysis insights using aggregation queries.
        
        Args:
            **kwargs: Optional parameters for insight generation
            
        Returns:
            Dictionary containing cross-analysis insights
        """
        try:
            # Get all aggregation queries
            all_queries = self.aggregation_queries.get_all_aggregation_queries()
            
            # Execute aggregation queries
            aggregated_results = self.service_orchestrator.aggregate_results(all_queries)
            
            # Process and structure the results
            insights = {
                'status': 'success',
                'generated_at': datetime.now().isoformat(),
                'session_id': self.session_id,
                'region': self.region,
                'total_queries_executed': len(all_queries),
                'cost_correlations': {},
                'resource_relationships': {},
                'executive_insights': {},
                'raw_query_results': aggregated_results
            }
            
            # Categorize results
            for query_name, query_result in aggregated_results.items():
                if query_result['status'] == 'success' and query_result['data']:
                    if 'cost' in query_name.lower():
                        insights['cost_correlations'][query_name] = query_result['data']
                    elif 'resource' in query_name.lower() or 'relationship' in query_name.lower():
                        insights['resource_relationships'][query_name] = query_result['data']
                    elif 'executive' in query_name.lower() or 'summary' in query_name.lower():
                        insights['executive_insights'][query_name] = query_result['data']
            
            # Generate summary insights
            insights['summary'] = self._generate_insight_summary(insights)
            
            log_cloudwatch_operation(self.logger, "cross_analysis_insights_complete",
                                   session_id=self.session_id,
                                   successful_queries=len([r for r in aggregated_results.values() if r['status'] == 'success']))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error executing cross-analysis insights: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'session_id': self.session_id
            }
    
    def _generate_insight_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of cross-analysis insights."""
        summary = {
            'key_findings': [],
            'optimization_priorities': [],
            'cost_impact_analysis': {},
            'recommended_next_steps': []
        }
        
        try:
            # Analyze cost correlations
            cost_data = insights.get('cost_correlations', {})
            if cost_data:
                summary['key_findings'].append("Cost correlation analysis completed")
                
                # Look for high-cost components
                for query_name, results in cost_data.items():
                    if 'high_cost' in query_name and results:
                        high_cost_components = [r for r in results if r.get('monthly_cost', 0) > 50]
                        if high_cost_components:
                            summary['optimization_priorities'].extend([
                                f"High cost component identified: {comp.get('component', 'unknown')}" 
                                for comp in high_cost_components[:3]
                            ])
            
            # Analyze resource relationships
            resource_data = insights.get('resource_relationships', {})
            if resource_data:
                summary['key_findings'].append("Resource relationship analysis completed")
                
                # Look for cleanup opportunities
                for query_name, results in resource_data.items():
                    if 'unused' in query_name and results:
                        for result in results:
                            if result.get('cleanup_recommendation'):
                                summary['recommended_next_steps'].append(result['cleanup_recommendation'])
            
            # Analyze executive insights
            executive_data = insights.get('executive_insights', {})
            if executive_data:
                summary['key_findings'].append("Executive summary data available")
                
                # Extract cost savings potential
                for query_name, results in executive_data.items():
                    if 'savings' in query_name and results:
                        for result in results:
                            if 'estimated_savings_potential' in result:
                                summary['cost_impact_analysis']['potential_monthly_savings'] = result['estimated_savings_potential']
                            if 'savings_percentage' in result:
                                summary['cost_impact_analysis']['savings_percentage'] = result['savings_percentage']
            
            # Add general recommendations if no specific insights found
            if not summary['recommended_next_steps']:
                summary['recommended_next_steps'] = [
                    "Review individual analysis results for specific optimization opportunities",
                    "Focus on highest cost components first",
                    "Implement retention policies for log groups without them",
                    "Clean up unused alarms and custom metrics"
                ]
            
        except Exception as e:
            self.logger.error(f"Error generating insight summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary

    def create_executive_summary(self, **kwargs) -> Dict[str, Any]:
        """
        Create an executive summary of CloudWatch optimization opportunities.
        
        This method queries stored analysis results and creates a high-level summary
        suitable for management reporting.
        
        Args:
            **kwargs: Optional parameters for summary customization
            
        Returns:
            Dictionary containing executive summary
        """
        try:
            # Query for recent comprehensive analyses
            comprehensive_query = """
                SELECT * FROM sqlite_master 
                WHERE type='table' AND name LIKE '%comprehensive_summary%' 
                ORDER BY name DESC LIMIT 1
            """
            
            tables = self.get_analysis_results(comprehensive_query)
            
            if not tables:
                return {
                    'status': 'no_data',
                    'message': 'No comprehensive analysis results found',
                    'session_id': self.session_id
                }
            
            # Get the most recent comprehensive analysis
            latest_table = tables[0]['name']
            summary_query = f'SELECT * FROM "{latest_table}" LIMIT 1'
            summary_results = self.get_analysis_results(summary_query)
            
            if not summary_results:
                return {
                    'status': 'no_data',
                    'message': 'No summary data found',
                    'session_id': self.session_id
                }
            
            summary_data = summary_results[0]
            
            # Create executive summary
            executive_summary = {
                'status': 'success',
                'summary_type': 'executive',
                'generated_at': datetime.now().isoformat(),
                'session_id': self.session_id,
                'region': self.region,
                'analysis_overview': {
                    'analysis_date': summary_data.get('timestamp'),
                    'total_analyses_performed': summary_data.get('total_analyses', 0),
                    'successful_analyses': summary_data.get('successful_analyses', 0),
                    'analysis_status': summary_data.get('status'),
                    'cost_incurred_during_analysis': summary_data.get('cost_incurred', False),
                    'primary_data_source': summary_data.get('primary_data_source')
                },
                'key_findings': [
                    f"Completed {summary_data.get('successful_analyses', 0)} of {summary_data.get('total_analyses', 0)} CloudWatch optimization analyses",
                    f"Analysis used {summary_data.get('primary_data_source', 'unknown')} as primary data source",
                    f"Total analysis execution time: {summary_data.get('total_execution_time', 0):.2f} seconds"
                ],
                'next_steps': [
                    "Review individual analysis results for specific optimization opportunities",
                    "Implement high-priority recommendations first",
                    "Monitor cost impact after implementing changes",
                    "Schedule regular CloudWatch optimization reviews"
                ],
                'data_sources': {
                    'session_id': self.session_id,
                    'stored_tables': self.service_orchestrator.get_stored_tables(),
                    'query_capabilities': 'Full SQL querying available on stored data'
                }
            }
            
            return executive_summary
            
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'session_id': self.session_id
            }