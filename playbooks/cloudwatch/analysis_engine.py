"""
CloudWatch Analysis Engine for coordinating all analyzers with parallel execution.

This module provides the core analysis engine that coordinates all CloudWatch analyzers
with the ServiceOrchestrator for parallel execution and session-sql integration.

Features:
- Analyzer registry and dynamic loading capabilities
- Comprehensive error handling and fallback coordination
- Performance optimization integration (caching, memory management, timeouts)
- Session-sql integration for storing analysis results
- Cross-analysis insights and correlation detection
"""

import logging
import asyncio
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import TimeoutError as ConcurrentTimeoutError

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer, get_analyzer_registry
from playbooks.cloudwatch.general_spend_analyzer import GeneralSpendAnalyzer
from playbooks.cloudwatch.metrics_optimization_analyzer import MetricsOptimizationAnalyzer
from playbooks.cloudwatch.logs_optimization_analyzer import LogsOptimizationAnalyzer
from playbooks.cloudwatch.alarms_and_dashboards_analyzer import AlarmsAndDashboardsAnalyzer
from playbooks.cloudwatch.cost_controller import CostController, CostPreferences
from services.cloudwatch_pricing import CloudWatchPricing
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task, ParallelTask
from utils.performance_monitor import get_performance_monitor
from utils.memory_manager import get_memory_manager
from utils.intelligent_cache import get_analysis_results_cache
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for analyzer instances."""
    analyzer_class: Type[BaseAnalyzer]
    enabled: bool = True
    priority: int = 1
    timeout_seconds: float = 60.0
    retry_attempts: int = 2
    cache_results: bool = True
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class EngineConfig:
    """Configuration for the CloudWatch Analysis Engine."""
    max_parallel_analyzers: int = 4
    default_timeout_seconds: float = 120.0
    enable_caching: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_management: bool = True
    cache_ttl_seconds: int = 3600
    retry_failed_analyzers: bool = True
    cross_analysis_insights: bool = True


class CloudWatchAnalysisEngine:
    """
    Core analysis engine that coordinates all CloudWatch analyzers.
    
    This engine provides:
    - Analyzer registry and dynamic loading capabilities
    - Integration with ServiceOrchestrator for parallel execution
    - Comprehensive error handling and fallback coordination
    - Performance optimization integration (caching, memory management, timeouts)
    - Session-sql integration for storing analysis results
    - Cross-analysis insights and correlation detection
    """
    
    def __init__(self, region: str = None, session_id: str = None, config: EngineConfig = None,
                 performance_monitor=None, memory_manager=None, timeout_handler=None,
                 pricing_cache=None, analysis_results_cache=None):
        """
        Initialize the CloudWatch Analysis Engine with performance optimizations.
        
        Args:
            region: AWS region for analysis
            session_id: Session ID for data persistence
            config: Engine configuration
            performance_monitor: Performance monitoring instance
            memory_manager: Memory management instance
            timeout_handler: Timeout handling instance
            pricing_cache: Pricing cache instance
            analysis_results_cache: Analysis results cache instance
        """
        self.region = region
        self.session_id = session_id
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance optimization components
        self.performance_monitor = performance_monitor or get_performance_monitor()
        self.memory_manager = memory_manager or get_memory_manager()
        self.pricing_cache = pricing_cache
        self.analysis_results_cache = analysis_results_cache or get_analysis_results_cache()
        self.timeout_handler = timeout_handler
        
        # Initialize core services
        from services.cloudwatch_service import CloudWatchService, CloudWatchServiceConfig
        cloudwatch_config = CloudWatchServiceConfig(region=region)
        self.cloudwatch_service = CloudWatchService(config=cloudwatch_config)
        self.pricing_service = CloudWatchPricing(region=region)
        self.cost_controller = CostController()
        
        # Initialize service orchestrator for parallel execution
        self.service_orchestrator = ServiceOrchestrator(session_id=session_id)
        
        # Performance optimization components with enhanced configuration
        self.performance_monitor = performance_monitor or (get_performance_monitor() if self.config.enable_performance_monitoring else None)
        self.memory_manager = memory_manager or (get_memory_manager() if self.config.enable_memory_management else None)
        self.cache = analysis_results_cache or (get_analysis_results_cache() if self.config.enable_caching else None)
        self.timeout_handler = timeout_handler
        
        # Initialize CloudWatch-specific performance optimizations
        self._setup_cloudwatch_performance_optimizations()
        
        # Integrate performance components
        if self.performance_monitor and self.memory_manager:
            self.memory_manager.set_performance_monitor(self.performance_monitor)
        
        if self.cache and self.performance_monitor:
            self.cache.set_performance_monitor(self.performance_monitor)
        
        if self.cache and self.memory_manager:
            self.memory_manager.add_cache_reference(self.cache)
        
        if self.timeout_handler and self.performance_monitor:
            self.timeout_handler.set_performance_monitor(self.performance_monitor)
        
        # Analyzer registry and configurations
        self.analyzer_registry = get_analyzer_registry()
        self.analyzer_configs: Dict[str, AnalyzerConfig] = {}
        self.analyzer_instances: Dict[str, BaseAnalyzer] = {}
        
        # Initialize analyzers
        self._initialize_default_analyzers()
        self._load_analyzer_instances()
        
        # Engine state
        self.engine_id = f"cloudwatch_engine_{int(datetime.now().timestamp())}"
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.engine_start_time = datetime.now()
        self.total_analyses_run = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        
        log_cloudwatch_operation(self.logger, "analysis_engine_initialized",
                               region=region, session_id=session_id,
                               engine_id=self.engine_id,
                               analyzers_count=len(self.analyzer_instances),
                               performance_monitoring=self.config.enable_performance_monitoring,
                               memory_management=self.config.enable_memory_management,
                               caching_enabled=self.config.enable_caching,
                               progressive_timeouts=self.timeout_handler is not None,
                               cloudwatch_optimizations=True)
    
    @property
    def analyzers(self) -> Dict[str, BaseAnalyzer]:
        """
        Property to access analyzer instances for backward compatibility.
        
        Returns:
            Dictionary of analyzer instances keyed by analyzer name
        """
        return self.analyzer_instances
    
    def _initialize_default_analyzers(self):
        """Initialize default analyzer configurations."""
        default_configs = {
            'general_spend': AnalyzerConfig(
                analyzer_class=GeneralSpendAnalyzer,
                enabled=True,
                priority=4,  # Highest priority - foundational analysis
                timeout_seconds=90.0,
                retry_attempts=2,
                cache_results=True,
                dependencies=[]
            ),
            'logs_optimization': AnalyzerConfig(
                analyzer_class=LogsOptimizationAnalyzer,
                enabled=True,
                priority=3,  # High priority - often highest cost impact
                timeout_seconds=120.0,
                retry_attempts=2,
                cache_results=True,
                dependencies=['general_spend']
            ),
            'metrics_optimization': AnalyzerConfig(
                analyzer_class=MetricsOptimizationAnalyzer,
                enabled=True,
                priority=2,  # Medium priority
                timeout_seconds=90.0,
                retry_attempts=2,
                cache_results=True,
                dependencies=[]
            ),
            'alarms_and_dashboards': AnalyzerConfig(
                analyzer_class=AlarmsAndDashboardsAnalyzer,
                enabled=True,
                priority=1,  # Lower priority - efficiency focused
                timeout_seconds=60.0,
                retry_attempts=2,
                cache_results=True,
                dependencies=['metrics_optimization']
            )
        }
        
        for analyzer_name, config in default_configs.items():
            self.register_analyzer(analyzer_name, config)
    
    def _load_analyzer_instances(self):
        """Load analyzer instances based on configurations."""
        for analyzer_name, config in self.analyzer_configs.items():
            if not config.enabled:
                continue
                
            try:
                # Create analyzer instance
                analyzer = config.analyzer_class(
                    cost_explorer_service=None,  # Will be injected per analysis
                    config_service=None,         # Will be injected per analysis
                    metrics_service=None,        # Will be injected per analysis
                    cloudwatch_service=self.cloudwatch_service,
                    pricing_service=self.pricing_service,
                    performance_monitor=self.performance_monitor,
                    memory_manager=self.memory_manager
                )
                
                # Register with global registry
                self.analyzer_registry.register(analyzer)
                
                # Store in local instances
                self.analyzer_instances[analyzer_name] = analyzer
                
                log_cloudwatch_operation(self.logger, "analyzer_loaded",
                                       analyzer_name=analyzer_name,
                                       analyzer_class=config.analyzer_class.__name__,
                                       priority=config.priority,
                                       timeout=config.timeout_seconds)
                
            except Exception as e:
                self.logger.error(f"Failed to load analyzer {analyzer_name}: {str(e)}")
                # Continue loading other analyzers
    
    def register_analyzer(self, analyzer_name: str, config: AnalyzerConfig):
        """
        Register an analyzer configuration.
        
        Args:
            analyzer_name: Unique name for the analyzer
            config: Analyzer configuration
        """
        self.analyzer_configs[analyzer_name] = config
        self.logger.info(f"Registered analyzer configuration: {analyzer_name}")
    
    def load_analyzer_dynamically(self, module_path: str, class_name: str, 
                                analyzer_name: str, config: AnalyzerConfig = None) -> bool:
        """
        Dynamically load an analyzer from a module.
        
        Args:
            module_path: Python module path (e.g., 'playbooks.cloudwatch.custom_analyzer')
            class_name: Name of the analyzer class
            analyzer_name: Unique name for the analyzer
            config: Optional analyzer configuration
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the analyzer class
            analyzer_class = getattr(module, class_name)
            
            # Verify it's a BaseAnalyzer subclass
            if not issubclass(analyzer_class, BaseAnalyzer):
                raise ValueError(f"{class_name} is not a subclass of BaseAnalyzer")
            
            # Create configuration if not provided
            if config is None:
                config = AnalyzerConfig(
                    analyzer_class=analyzer_class,
                    enabled=True,
                    priority=1,
                    timeout_seconds=self.config.default_timeout_seconds
                )
            else:
                config.analyzer_class = analyzer_class
            
            # Register the analyzer
            self.register_analyzer(analyzer_name, config)
            
            # Load the instance
            self._load_single_analyzer(analyzer_name, config)
            
            self.logger.info(f"Dynamically loaded analyzer: {analyzer_name} from {module_path}.{class_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to dynamically load analyzer {analyzer_name}: {str(e)}")
            return False
    
    def _setup_cloudwatch_performance_optimizations(self):
        """Set up CloudWatch-specific performance optimizations."""
        try:
            # Configure memory management for CloudWatch data processing
            if self.memory_manager:
                # Register CloudWatch-specific cleanup callbacks
                self.memory_manager.register_cleanup_callback(self._cleanup_cloudwatch_analysis_data)
                
                # Set CloudWatch-specific memory thresholds
                from utils.memory_manager import MemoryThreshold
                cloudwatch_thresholds = MemoryThreshold(
                    warning_percent=70.0,
                    critical_percent=80.0,
                    cleanup_percent=85.0,
                    max_memory_mb=1536  # 1.5GB for analysis engine
                )
                self.memory_manager.thresholds = cloudwatch_thresholds
            
            # Configure progressive timeouts for CloudWatch analyses
            if self.timeout_handler:
                from utils.progressive_timeout import TimeoutConfiguration, ComplexityLevel
                
                # CloudWatch analysis-specific timeout configuration
                timeout_config = TimeoutConfiguration(
                    base_timeout=60.0,
                    complexity_multiplier={
                        ComplexityLevel.VERY_LOW: 0.5,
                        ComplexityLevel.LOW: 1.0,
                        ComplexityLevel.MEDIUM: 2.0,
                        ComplexityLevel.HIGH: 3.5,
                        ComplexityLevel.VERY_HIGH: 6.0
                    },
                    data_size_multiplier=0.2,
                    bucket_count_multiplier=0.0,  # Not applicable
                    historical_performance_weight=0.5,
                    system_load_weight=0.3,
                    min_timeout=20.0,
                    max_timeout=900.0,  # 15 minutes max
                    grace_period=30.0
                )
                self.timeout_handler.config = timeout_config
            
            # Set up intelligent caching for CloudWatch metadata
            if self.cache:
                # Register CloudWatch-specific cache warming
                self.cache.register_warming_function("cloudwatch_analysis_patterns", 
                                                   self._warm_analysis_patterns_cache)
            
            self.logger.info("CloudWatch-specific performance optimizations configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up CloudWatch performance optimizations: {str(e)}")
    
    def _cleanup_cloudwatch_analysis_data(self):
        """Cleanup callback for CloudWatch analysis data."""
        try:
            # Clean up temporary analysis data
            if hasattr(self, '_temp_analysis_data'):
                self._temp_analysis_data.clear()
            
            # Clean up analyzer-specific temporary data
            for analyzer in self.analyzer_instances.values():
                if hasattr(analyzer, 'cleanup_temporary_data'):
                    analyzer.cleanup_temporary_data()
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            self.logger.debug(f"CloudWatch analysis cleanup collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Error in CloudWatch analysis data cleanup: {str(e)}")
    
    def _warm_analysis_patterns_cache(self, cache, region: str = None):
        """Warm cache with common CloudWatch analysis patterns."""
        try:
            if not region:
                region = self.region
            
            self.logger.info(f"Warming CloudWatch analysis patterns cache for region: {region}")
            
            # Cache common analysis patterns and templates
            patterns = [
                f"general_spend_pattern_{region}",
                f"logs_optimization_pattern_{region}",
                f"metrics_optimization_pattern_{region}",
                f"alarms_dashboards_pattern_{region}"
            ]
            
            for pattern in patterns:
                cache.put(pattern, {
                    "pattern": True,
                    "region": region,
                    "warmed_at": datetime.now().isoformat()
                }, ttl_seconds=3600)
            
        except Exception as e:
            self.logger.error(f"Error warming analysis patterns cache: {str(e)}")
    
    def _load_single_analyzer(self, analyzer_name: str, config: AnalyzerConfig):
        """Load a single analyzer instance."""
        try:
            analyzer = config.analyzer_class(
                cost_explorer_service=None,
                config_service=None,
                metrics_service=None,
                cloudwatch_service=self.cloudwatch_service,
                pricing_service=self.pricing_service,
                performance_monitor=self.performance_monitor,
                memory_manager=self.memory_manager
            )
            
            self.analyzer_registry.register(analyzer)
            self.analyzer_instances[analyzer_name] = analyzer
            
        except Exception as e:
            self.logger.error(f"Failed to load analyzer instance {analyzer_name}: {str(e)}")
            raise
    
    async def run_analysis(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific analysis type with comprehensive error handling and performance optimization.
        
        Args:
            analysis_type: Type of analysis to run
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.now()
        execution_id = f"{analysis_type}_{int(start_time.timestamp())}"
        
        # Start performance monitoring
        monitoring_session = None
        if self.performance_monitor:
            monitoring_session = self.performance_monitor.start_analysis_monitoring(
                analysis_type, execution_id
            )
        
        # Start memory tracking
        memory_tracker = None
        if self.memory_manager:
            memory_tracker = self.memory_manager.start_memory_tracking(
                f"analysis_{analysis_type}_{execution_id}"
            )
        
        try:
            # Validate analyzer exists and is enabled
            if analysis_type not in self.analyzer_instances:
                return self._create_error_result(
                    analysis_type, start_time, execution_id,
                    f'Unknown analysis type: {analysis_type}',
                    available_types=list(self.analyzer_instances.keys()),
                    monitoring_session=monitoring_session,
                    memory_tracker=memory_tracker
                )
            
            # Get analyzer configuration
            config = self.analyzer_configs.get(analysis_type)
            if not config or not config.enabled:
                return self._create_error_result(
                    analysis_type, start_time, execution_id,
                    f'Analyzer {analysis_type} is disabled',
                    monitoring_session=monitoring_session,
                    memory_tracker=memory_tracker
                )
            
            # Check cache first
            cache_key = None
            if self.cache and config.cache_results:
                cache_key = self._generate_cache_key(analysis_type, kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for analysis {analysis_type}")
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit("analysis_results", analysis_type)
                    
                    # Add cache metadata
                    cached_result['cache_metadata'] = {
                        'cache_hit': True,
                        'cached_at': cached_result.get('timestamp'),
                        'retrieved_at': datetime.now().isoformat()
                    }
                    
                    # End monitoring
                    self._end_monitoring(monitoring_session, memory_tracker, True)
                    return cached_result
                else:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_miss("analysis_results", analysis_type)
            
            # Validate and sanitize cost preferences
            cost_preferences = self._validate_cost_preferences(kwargs)
            
            # Check dependencies
            dependency_results = await self._check_dependencies(analysis_type, config, **kwargs)
            
            # Track active analysis
            self.active_analyses[execution_id] = {
                'analysis_type': analysis_type,
                'start_time': start_time,
                'status': 'running',
                'monitoring_session': monitoring_session,
                'memory_tracker': memory_tracker
            }
            
            log_cloudwatch_operation(self.logger, "analysis_start",
                                   analysis_type=analysis_type,
                                   execution_id=execution_id,
                                   cost_preferences=str(cost_preferences),
                                   dependencies_met=len(dependency_results))
            
            # Get the analyzer
            analyzer = self.analyzer_instances[analysis_type]
            
            # Calculate intelligent timeout for this analysis
            intelligent_timeout = self._calculate_intelligent_timeout(analysis_type, config, **kwargs)
            kwargs['calculated_timeout'] = intelligent_timeout
            
            # Run the analysis with timeout and retry logic
            result = await self._run_analysis_with_retry(
                analyzer, analysis_type, config, dependency_results, **kwargs
            )
            
            # Add comprehensive metadata
            result = self._add_analysis_metadata(
                result, analysis_type, start_time, execution_id, 
                cost_preferences, dependency_results
            )
            
            # Cache the result if successful
            if (self.cache and config.cache_results and 
                result.get('status') == 'success'):
                self.cache.put(
                    cache_key, result, 
                    ttl_seconds=self.config.cache_ttl_seconds,
                    tags={'analysis_type': analysis_type, 'region': self.region}
                )
            
            # Update statistics
            self.total_analyses_run += 1
            if result.get('status') == 'success':
                self.successful_analyses += 1
            else:
                self.failed_analyses += 1
            
            log_cloudwatch_operation(self.logger, "analysis_complete",
                                   analysis_type=analysis_type,
                                   execution_id=execution_id,
                                   status=result.get('status', 'unknown'),
                                   execution_time=result.get('execution_time', 0))
            
            # End monitoring
            self._end_monitoring(monitoring_session, memory_tracker, 
                               result.get('status') == 'success')
            
            # Remove from active analyses
            self.active_analyses.pop(execution_id, None)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis {analysis_type} failed with exception: {str(e)}")
            
            # Update statistics
            self.total_analyses_run += 1
            self.failed_analyses += 1
            
            # End monitoring
            self._end_monitoring(monitoring_session, memory_tracker, False, str(e))
            
            # Remove from active analyses
            self.active_analyses.pop(execution_id, None)
            
            import traceback
            full_traceback = traceback.format_exc()
            return self._create_error_result(
                analysis_type, start_time, execution_id, str(e),
                error_type=type(e).__name__,
                full_traceback=full_traceback,
                monitoring_session=monitoring_session,
                memory_tracker=memory_tracker
            )
    
    def _validate_cost_preferences(self, kwargs: Dict[str, Any]) -> CostPreferences:
        """Validate and sanitize cost preferences."""
        raw_preferences = {
            'allow_cost_explorer': kwargs.get('allow_cost_explorer', False),
            'allow_aws_config': kwargs.get('allow_aws_config', False),
            'allow_cloudtrail': kwargs.get('allow_cloudtrail', False),
            'allow_minimal_cost_metrics': kwargs.get('allow_minimal_cost_metrics', False),
        }
        
        try:
            return self.cost_controller.validate_and_sanitize_preferences(raw_preferences)
        except ValueError as e:
            raise ValueError(f'Invalid cost preferences: {str(e)}')
    
    def _generate_cache_key(self, analysis_type: str, kwargs: Dict[str, Any]) -> List[Any]:
        """Generate cache key for analysis results."""
        # Include relevant parameters that affect the analysis
        cache_components = [
            'cloudwatch_analysis',
            analysis_type,
            self.region,
            kwargs.get('lookback_days', 30),
            kwargs.get('allow_cost_explorer', False),
            kwargs.get('allow_aws_config', False),
            kwargs.get('allow_cloudtrail', False),
            kwargs.get('allow_minimal_cost_metrics', False),
            # Add specific resource filters if provided
            tuple(sorted(kwargs.get('log_group_names', []))),
            tuple(sorted(kwargs.get('alarm_names', []))),
            tuple(sorted(kwargs.get('dashboard_names', [])))
        ]
        return cache_components
    
    async def _check_dependencies(self, analysis_type: str, config: AnalyzerConfig, 
                                **kwargs) -> Dict[str, Any]:
        """Check and resolve analyzer dependencies."""
        dependency_results = {}
        
        for dependency in config.dependencies:
            if dependency in self.analyzer_instances:
                try:
                    # Run dependency analysis if not already available
                    dep_result = await self.run_analysis(dependency, **kwargs)
                    dependency_results[dependency] = dep_result
                    
                    if dep_result.get('status') != 'success':
                        self.logger.warning(
                            f"Dependency {dependency} for {analysis_type} failed: "
                            f"{dep_result.get('error_message', 'Unknown error')}"
                        )
                except Exception as e:
                    self.logger.error(f"Failed to resolve dependency {dependency}: {str(e)}")
                    dependency_results[dependency] = {
                        'status': 'error',
                        'error_message': str(e)
                    }
        
        return dependency_results
    
    async def _run_analysis_with_retry(self, analyzer: BaseAnalyzer, analysis_type: str,
                                     config: AnalyzerConfig, dependency_results: Dict[str, Any],
                                     **kwargs) -> Dict[str, Any]:
        """Run analysis with retry logic and intelligent timeout handling."""
        last_error = None
        
        # Use calculated timeout or fallback to config
        timeout_seconds = kwargs.get('calculated_timeout', kwargs.get('timeout_seconds', config.timeout_seconds))
        
        for attempt in range(config.retry_attempts + 1):
            try:
                # Register large object for memory management
                if self.memory_manager:
                    analysis_id = f"{analysis_type}_attempt_{attempt}_{int(datetime.now().timestamp())}"
                    self.memory_manager.register_large_object(
                        analysis_id,
                        kwargs,
                        size_mb=0.5,  # Estimated size
                        cleanup_callback=lambda: self.logger.debug(f"Cleaned up analysis data for {analysis_id}")
                    )
                
                # Run analysis with intelligent timeout
                result = await asyncio.wait_for(
                    analyzer.execute_with_error_handling(**kwargs),
                    timeout=timeout_seconds
                )
                
                # Record successful execution time for future timeout calculations
                if self.timeout_handler and 'execution_time' in result:
                    from utils.progressive_timeout import ComplexityLevel
                    complexity = self.timeout_handler.get_complexity_level(analysis_type, **kwargs)
                    self.timeout_handler.record_execution_time(
                        analysis_type, 
                        result['execution_time'], 
                        complexity
                    )
                
                # Add dependency results to the analysis result
                if dependency_results:
                    result['dependency_results'] = dependency_results
                
                # Add timeout metadata
                result['timeout_metadata'] = {
                    'calculated_timeout': timeout_seconds,
                    'attempt_number': attempt + 1,
                    'intelligent_timeout_used': 'calculated_timeout' in kwargs
                }
                
                return result
                
            except (asyncio.TimeoutError, ConcurrentTimeoutError) as e:
                last_error = f"Analysis timed out after {timeout_seconds} seconds"
                self.logger.warning(f"Attempt {attempt + 1} for {analysis_type} timed out after {timeout_seconds}s")
                
                # Record timeout for future timeout calculations
                if self.timeout_handler:
                    from utils.progressive_timeout import ComplexityLevel
                    complexity = self.timeout_handler.get_complexity_level(analysis_type, **kwargs)
                    # Record timeout as a very long execution time to adjust future timeouts
                    self.timeout_handler.record_execution_time(
                        analysis_type, 
                        timeout_seconds * 1.5,  # Indicate it would have taken longer
                        complexity
                    )
                
                if attempt < config.retry_attempts:
                    # Exponential backoff for retries with increased timeout
                    await asyncio.sleep(2 ** attempt)
                    timeout_seconds *= 1.5  # Increase timeout for retry
                    continue
                else:
                    break
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt + 1} for {analysis_type} failed: {str(e)}")
                
                if attempt < config.retry_attempts:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    break
        
        # All attempts failed
        return {
            'status': 'error',
            'error_message': last_error or 'Analysis failed after all retry attempts',
            'analysis_type': analysis_type,
            'retry_attempts': config.retry_attempts,
            'dependency_results': dependency_results,
            'timeout_metadata': {
                'final_timeout_used': timeout_seconds,
                'total_attempts': config.retry_attempts + 1,
                'intelligent_timeout_used': 'calculated_timeout' in kwargs
            }
        }
    
    def _add_analysis_metadata(self, result: Dict[str, Any], analysis_type: str,
                             start_time: datetime, execution_id: str,
                             cost_preferences: CostPreferences,
                             dependency_results: Dict[str, Any]) -> Dict[str, Any]:
        """Add comprehensive metadata to analysis results."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result['engine_metadata'] = {
            'analysis_engine_version': '2.0.0',
            'engine_id': self.engine_id,
            'execution_id': execution_id,
            'session_id': self.session_id,
            'region': self.region,
            'cost_preferences': cost_preferences.__dict__,
            'execution_time': execution_time,
            'dependencies_count': len(dependency_results),
            'cache_enabled': self.config.enable_caching,
            'performance_monitoring': self.config.enable_performance_monitoring,
            'memory_management': self.config.enable_memory_management
        }
        
        # Ensure required fields exist
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        if 'execution_time' not in result:
            result['execution_time'] = execution_time
        
        return result
    
    def _create_error_result(self, analysis_type: str, start_time: datetime,
                           execution_id: str, error_message: str,
                           error_type: str = None, available_types: List[str] = None,
                           full_traceback: str = None, monitoring_session: str = None, 
                           memory_tracker: str = None) -> Dict[str, Any]:
        """Create standardized error result with full exception details."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'status': 'error',
            'error_message': error_message,
            'analysis_type': analysis_type,
            'timestamp': start_time.isoformat(),
            'execution_time': execution_time,
            'engine_metadata': {
                'analysis_engine_version': '2.0.0',
                'engine_id': self.engine_id,
                'execution_id': execution_id,
                'session_id': self.session_id,
                'region': self.region
            }
        }
        
        if error_type:
            result['error_type'] = error_type
        
        if available_types:
            result['available_types'] = available_types
        
        if full_traceback:
            result['full_exception_details'] = {
                'traceback': full_traceback,
                'error_location': self._extract_error_location(full_traceback)
            }
        
        # End monitoring for error case
        self._end_monitoring(monitoring_session, memory_tracker, False, error_message)
        
        return result
    
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
    
    def _calculate_intelligent_timeout(self, analysis_type: str, config: AnalyzerConfig, **kwargs) -> float:
        """Calculate intelligent timeout based on analysis complexity and historical performance."""
        try:
            if self.timeout_handler:
                # Create analysis context for timeout calculation
                context = self.timeout_handler.create_analysis_context(analysis_type, **kwargs)
                
                # Calculate timeout using progressive timeout handler
                timeout_result = self.timeout_handler.calculate_timeout(context)
                
                self.logger.debug(f"Calculated intelligent timeout for {analysis_type}: {timeout_result.final_timeout:.1f}s")
                self.logger.debug(f"Timeout reasoning: {'; '.join(timeout_result.reasoning)}")
                
                return timeout_result.final_timeout
            else:
                # Fallback to configuration timeout
                return config.timeout_seconds
                
        except Exception as e:
            self.logger.warning(f"Error calculating intelligent timeout for {analysis_type}: {str(e)}")
            return config.timeout_seconds
    
    def _end_monitoring(self, monitoring_session: str = None, memory_tracker: str = None,
                       success: bool = True, error_message: str = None):
        """End performance and memory monitoring."""
        if monitoring_session and self.performance_monitor:
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session, success, error_message
            )
        
        if memory_tracker and self.memory_manager:
            self.memory_manager.stop_memory_tracking(memory_tracker)
    
    async def run_comprehensive_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Run all enabled CloudWatch analyses in parallel with intelligent orchestration.
        
        Args:
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        start_time = datetime.now()
        execution_id = f"comprehensive_{int(start_time.timestamp())}"
        
        # Start comprehensive monitoring
        monitoring_session = None
        if self.performance_monitor:
            monitoring_session = self.performance_monitor.start_analysis_monitoring(
                "comprehensive_analysis", execution_id
            )
        
        memory_tracker = None
        if self.memory_manager:
            memory_tracker = self.memory_manager.start_memory_tracking(
                f"comprehensive_analysis_{execution_id}"
            )
        
        try:
            # Validate cost preferences
            cost_preferences = self._validate_cost_preferences(kwargs)
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_start",
                                   execution_id=execution_id,
                                   enabled_analyzers=len(self.analyzer_instances),
                                   cost_preferences=str(cost_preferences))
            
            # Check cache for comprehensive analysis
            cache_key = None
            if self.cache and self.config.enable_caching:
                cache_key = self._generate_cache_key("comprehensive", kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Cache hit for comprehensive analysis")
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit("analysis_results", "comprehensive")
                    
                    cached_result['cache_metadata'] = {
                        'cache_hit': True,
                        'cached_at': cached_result.get('timestamp'),
                        'retrieved_at': datetime.now().isoformat()
                    }
                    
                    self._end_monitoring(monitoring_session, memory_tracker, True)
                    return cached_result
                else:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_miss("analysis_results", "comprehensive")
            
            # Get enabled analyzers sorted by priority and dependencies
            execution_plan = self._create_execution_plan()
            
            if not execution_plan:
                return self._create_error_result(
                    "comprehensive", start_time, execution_id,
                    "No enabled analyzers found",
                    monitoring_session=monitoring_session,
                    memory_tracker=memory_tracker
                )
            
            # Execute analyses according to plan
            analysis_results = await self._execute_analysis_plan(execution_plan, **kwargs)
            
            # Generate cross-analysis insights
            cross_insights = {}
            if self.config.cross_analysis_insights:
                cross_insights = self._generate_cross_analysis_insights(analysis_results)
            
            # Create comprehensive result
            comprehensive_result = self._create_comprehensive_result(
                analysis_results, cross_insights, cost_preferences, 
                start_time, execution_id, execution_plan
            )
            
            # Cache the result if successful
            if (self.cache and self.config.enable_caching and 
                comprehensive_result.get('status') in ['success', 'partial']):
                self.cache.put(
                    cache_key, comprehensive_result,
                    ttl_seconds=self.config.cache_ttl_seconds,
                    tags={'analysis_type': 'comprehensive', 'region': self.region}
                )
            
            # Update statistics
            self.total_analyses_run += 1
            if comprehensive_result.get('status') == 'success':
                self.successful_analyses += 1
            else:
                self.failed_analyses += 1
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_complete",
                                   execution_id=execution_id,
                                   status=comprehensive_result.get('status'),
                                   total_analyses=len(execution_plan),
                                   successful_analyses=comprehensive_result.get('analysis_summary', {}).get('successful_analyses', 0),
                                   execution_time=comprehensive_result.get('total_execution_time', 0))
            
            self._end_monitoring(monitoring_session, memory_tracker, 
                               comprehensive_result.get('status') in ['success', 'partial'])
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            
            self.total_analyses_run += 1
            self.failed_analyses += 1
            
            self._end_monitoring(monitoring_session, memory_tracker, False, str(e))
            
            return self._create_error_result(
                "comprehensive", start_time, execution_id, str(e),
                error_type=type(e).__name__,
                monitoring_session=monitoring_session,
                memory_tracker=memory_tracker
            )
    
    def _create_execution_plan(self) -> List[Dict[str, Any]]:
        """Create intelligent execution plan based on priorities and dependencies."""
        enabled_analyzers = {
            name: config for name, config in self.analyzer_configs.items()
            if config.enabled and name in self.analyzer_instances
        }
        
        if not enabled_analyzers:
            return []
        
        # Sort by priority (higher priority first) and handle dependencies
        execution_plan = []
        executed = set()
        
        # Create dependency graph
        dependency_graph = {}
        for name, config in enabled_analyzers.items():
            dependency_graph[name] = [
                dep for dep in config.dependencies 
                if dep in enabled_analyzers
            ]
        
        # Topological sort with priority consideration
        while len(executed) < len(enabled_analyzers):
            # Find analyzers with no unmet dependencies
            ready_analyzers = []
            for name, config in enabled_analyzers.items():
                if name not in executed:
                    unmet_deps = [
                        dep for dep in dependency_graph[name]
                        if dep not in executed
                    ]
                    if not unmet_deps:
                        ready_analyzers.append((name, config))
            
            if not ready_analyzers:
                # Circular dependency or other issue
                remaining = set(enabled_analyzers.keys()) - executed
                self.logger.warning(f"Circular dependencies detected for analyzers: {remaining}")
                # Add remaining analyzers anyway
                for name in remaining:
                    config = enabled_analyzers[name]
                    ready_analyzers.append((name, config))
            
            # Sort ready analyzers by priority
            ready_analyzers.sort(key=lambda x: x[1].priority, reverse=True)
            
            # Add to execution plan
            for name, config in ready_analyzers:
                if name not in executed:
                    execution_plan.append({
                        'analyzer_name': name,
                        'config': config,
                        'dependencies': dependency_graph[name].copy()
                    })
                    executed.add(name)
        
        return execution_plan
    
    async def _execute_analysis_plan(self, execution_plan: List[Dict[str, Any]], 
                                   **kwargs) -> Dict[str, Any]:
        """Execute analyses according to the execution plan."""
        analysis_results = {}
        dependency_results = {}
        
        # Determine if we can run analyses in parallel
        max_parallel = min(self.config.max_parallel_analyzers, len(execution_plan))
        
        if max_parallel > 1:
            # Parallel execution for independent analyses
            return await self._execute_parallel_analyses(execution_plan, **kwargs)
        else:
            # Sequential execution
            return await self._execute_sequential_analyses(execution_plan, **kwargs)
    
    async def _execute_parallel_analyses(self, execution_plan: List[Dict[str, Any]], 
                                       **kwargs) -> Dict[str, Any]:
        """Execute analyses in parallel where possible."""
        analysis_results = {}
        
        # Group analyses by dependency level
        dependency_levels = []
        remaining_plan = execution_plan.copy()
        completed = set()
        
        while remaining_plan:
            current_level = []
            next_remaining = []
            
            for plan_item in remaining_plan:
                analyzer_name = plan_item['analyzer_name']
                dependencies = plan_item['dependencies']
                
                # Check if all dependencies are completed
                if all(dep in completed for dep in dependencies):
                    current_level.append(plan_item)
                else:
                    next_remaining.append(plan_item)
            
            if current_level:
                dependency_levels.append(current_level)
                completed.update(item['analyzer_name'] for item in current_level)
                remaining_plan = next_remaining
            else:
                # No progress possible, add remaining items
                dependency_levels.append(next_remaining)
                break
        
        # Execute each dependency level
        for level_index, level_analyses in enumerate(dependency_levels):
            self.logger.info(f"Executing dependency level {level_index + 1} with {len(level_analyses)} analyses")
            
            # Create tasks for this level
            level_tasks = []
            for plan_item in level_analyses:
                analyzer_name = plan_item['analyzer_name']
                
                # Create analysis task
                async def run_single_analysis(name=analyzer_name):
                    return await self.run_analysis(name, **kwargs)
                
                level_tasks.append(run_single_analysis())
            
            # Execute level in parallel
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(level_results):
                analyzer_name = level_analyses[i]['analyzer_name']
                
                if isinstance(result, Exception):
                    analysis_results[analyzer_name] = {
                        'status': 'error',
                        'error_message': str(result),
                        'analysis_type': analyzer_name
                    }
                else:
                    analysis_results[analyzer_name] = result
        
        return analysis_results
    
    async def _execute_sequential_analyses(self, execution_plan: List[Dict[str, Any]], 
                                         **kwargs) -> Dict[str, Any]:
        """Execute analyses sequentially."""
        analysis_results = {}
        
        for plan_item in execution_plan:
            analyzer_name = plan_item['analyzer_name']
            
            try:
                result = await self.run_analysis(analyzer_name, **kwargs)
                analysis_results[analyzer_name] = result
                
            except Exception as e:
                self.logger.error(f"Sequential analysis {analyzer_name} failed: {str(e)}")
                analysis_results[analyzer_name] = {
                    'status': 'error',
                    'error_message': str(e),
                    'analysis_type': analyzer_name
                }
        
        return analysis_results
    
    def _create_comprehensive_result(self, analysis_results: Dict[str, Any],
                                   cross_insights: Dict[str, Any],
                                   cost_preferences: CostPreferences,
                                   start_time: datetime, execution_id: str,
                                   execution_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive analysis result."""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        successful_analyses = []
        failed_analyses = []
        partial_analyses = []
        
        for analyzer_name, result in analysis_results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                successful_analyses.append(analyzer_name)
            elif status == 'partial':
                partial_analyses.append(analyzer_name)
            else:
                failed_analyses.append(analyzer_name)
        
        # Determine overall status
        if failed_analyses and not successful_analyses and not partial_analyses:
            overall_status = 'error'
        elif failed_analyses or partial_analyses:
            overall_status = 'partial'
        else:
            overall_status = 'success'
        
        # Aggregate cost information
        total_cost_incurred = any(
            result.get('cost_incurred', False) 
            for result in analysis_results.values()
        )
        
        all_cost_operations = []
        for result in analysis_results.values():
            all_cost_operations.extend(result.get('cost_incurring_operations', []))
        
        # Generate comprehensive recommendations
        comprehensive_recommendations = self._generate_comprehensive_recommendations(
            analysis_results, cross_insights
        )
        
        return {
            'status': overall_status,
            'analysis_type': 'comprehensive',
            'timestamp': start_time.isoformat(),
            'total_execution_time': execution_time,
            'cost_incurred': total_cost_incurred,
            'cost_incurring_operations': list(set(all_cost_operations)),
            'primary_data_source': 'multiple',
            'individual_analyses': analysis_results,
            'cross_analysis_insights': cross_insights,
            'comprehensive_recommendations': comprehensive_recommendations,
            'analysis_summary': {
                'total_analyses': len(execution_plan),
                'successful_analyses': len(successful_analyses),
                'partial_analyses': len(partial_analyses),
                'failed_analyses': len(failed_analyses),
                'successful_types': successful_analyses,
                'partial_types': partial_analyses,
                'failed_types': failed_analyses
            },
            'engine_metadata': {
                'analysis_engine_version': '2.0.0',
                'engine_id': self.engine_id,
                'execution_id': execution_id,
                'session_id': self.session_id,
                'region': self.region,
                'cost_preferences': cost_preferences.__dict__,
                'execution_plan_length': len(execution_plan),
                'parallel_execution': self.config.max_parallel_analyzers > 1,
                'cross_insights_enabled': self.config.cross_analysis_insights
            }
        }
    
    def _get_analysis_priority(self, analysis_type: str) -> int:
        """Get priority for analysis type (higher number = higher priority)."""
        priority_map = {
            'general_spend': 4,        # Highest priority - foundational analysis
            'logs_optimization': 3,    # High priority - often highest cost impact
            'metrics_optimization': 2, # Medium priority
            'alarms_and_dashboards': 1 # Lower priority - efficiency focused
        }
        return priority_map.get(analysis_type, 1)
    
    def _aggregate_analysis_results(self, execution_results: Dict[str, Any], 
                                  cost_preferences: CostPreferences,
                                  start_time: datetime, **kwargs) -> Dict[str, Any]:
        """Aggregate results from all analyses into comprehensive report."""
        
        # Initialize comprehensive result structure
        comprehensive_result = {
            'status': 'success',
            'analysis_type': 'comprehensive',
            'timestamp': start_time.isoformat(),
            'total_execution_time': (datetime.now() - start_time).total_seconds(),
            'cost_incurred': False,
            'cost_incurring_operations': [],
            'primary_data_source': 'cloudwatch_config',
            'fallback_used': False,
            'individual_analyses': {},
            'aggregated_insights': {},
            'comprehensive_recommendations': [],
            'session_metadata': {
                'session_id': self.session_id,
                'stored_tables': execution_results.get('stored_tables', []),
                'execution_summary': execution_results
            },
            'cost_summary': self.cost_controller.get_cost_summary(cost_preferences)
        }
        
        # Process individual analysis results
        successful_analyses = []
        failed_analyses = []
        
        for task_id, task_result in execution_results.get('results', {}).items():
            if task_result['status'] == 'success':
                # Extract analysis type from task_id or operation
                analysis_type = self._extract_analysis_type_from_task(task_id, task_result)
                
                if analysis_type:
                    # Get the actual analysis result data (would need to be extracted from stored data)
                    analysis_data = self._get_analysis_data_from_session(task_result.get('stored_table'))
                    
                    if analysis_data:
                        comprehensive_result['individual_analyses'][analysis_type] = analysis_data
                        successful_analyses.append(analysis_type)
                        
                        # Aggregate cost information
                        if analysis_data.get('cost_incurred', False):
                            comprehensive_result['cost_incurred'] = True
                            cost_ops = analysis_data.get('cost_incurring_operations', [])
                            comprehensive_result['cost_incurring_operations'].extend(cost_ops)
                        
                        # Check for fallback usage
                        if analysis_data.get('fallback_used', False):
                            comprehensive_result['fallback_used'] = True
                        
                        # Update primary data source if Cost Explorer was used
                        if analysis_data.get('primary_data_source') == 'cost_explorer':
                            comprehensive_result['primary_data_source'] = 'cost_explorer'
            else:
                analysis_type = self._extract_analysis_type_from_task(task_id, task_result)
                if analysis_type:
                    failed_analyses.append(analysis_type)
                    comprehensive_result['individual_analyses'][analysis_type] = {
                        'status': 'error',
                        'error_message': task_result.get('error', 'Unknown error'),
                        'analysis_type': analysis_type
                    }
        
        # Generate aggregated insights
        comprehensive_result['aggregated_insights'] = self._generate_cross_analysis_insights(
            comprehensive_result['individual_analyses']
        )
        
        # Generate comprehensive recommendations
        comprehensive_result['comprehensive_recommendations'] = self._generate_comprehensive_recommendations(
            comprehensive_result['individual_analyses'],
            comprehensive_result['aggregated_insights']
        )
        
        # Update overall status
        if failed_analyses and not successful_analyses:
            comprehensive_result['status'] = 'error'
        elif failed_analyses:
            comprehensive_result['status'] = 'partial'
        
        comprehensive_result['analysis_summary'] = {
            'total_analyses': len(self.analyzers),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'successful_types': successful_analyses,
            'failed_types': failed_analyses
        }
        
        return comprehensive_result
    
    def _extract_analysis_type_from_task(self, task_id: str, task_result: Dict[str, Any]) -> Optional[str]:
        """Extract analysis type from task ID or result."""
        # Try to extract from task_id (format: cloudwatch_{analysis_type}_{timestamp})
        if 'cloudwatch_' in task_id:
            parts = task_id.split('_')
            if len(parts) >= 3:
                return '_'.join(parts[1:-1])  # Everything between 'cloudwatch' and timestamp
        
        # Try to extract from operation name
        operation = task_result.get('operation', '')
        if operation.startswith('analyze_'):
            return operation[8:]  # Remove 'analyze_' prefix
        
        return None
    
    def _get_analysis_data_from_session(self, table_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get analysis data from session storage."""
        if not table_name or not self.session_id:
            return None
        
        try:
            # Query the stored analysis data
            query = f'SELECT * FROM "{table_name}" LIMIT 1'
            results = self.service_orchestrator.query_session_data(query)
            
            if results:
                # The first result should contain the analysis data
                return results[0]
        except Exception as e:
            self.logger.error(f"Error retrieving analysis data from {table_name}: {str(e)}")
        
        return None
    
    def _generate_cross_analysis_insights(self, individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights that span across multiple analyses."""
        insights = {
            'cost_correlations': [],
            'optimization_synergies': [],
            'resource_relationships': [],
            'efficiency_patterns': []
        }
        
        try:
            # Analyze cost correlations between different CloudWatch components
            if ('general_spend' in individual_analyses and 
                'logs_optimization' in individual_analyses):
                
                general_data = individual_analyses['general_spend'].get('data', {})
                logs_data = individual_analyses['logs_optimization'].get('data', {})
                
                # Look for high log costs correlation with overall spend
                cost_breakdown = general_data.get('cost_breakdown', {})
                logs_costs = cost_breakdown.get('logs_costs', {})
                
                if logs_costs.get('estimated_monthly', 0) > 50:  # $50+ monthly
                    insights['cost_correlations'].append({
                        'type': 'high_logs_cost_impact',
                        'description': 'High CloudWatch Logs costs significantly impact overall spend',
                        'estimated_impact': logs_costs.get('estimated_monthly', 0),
                        'recommendation': 'Prioritize logs optimization for maximum cost reduction'
                    })
            
            # Analyze optimization synergies
            if ('metrics_optimization' in individual_analyses and 
                'alarms_and_dashboards' in individual_analyses):
                
                metrics_data = individual_analyses['metrics_optimization'].get('data', {})
                alarms_data = individual_analyses['alarms_and_dashboards'].get('data', {})
                
                # Look for unused metrics that could affect alarms
                metrics_config = metrics_data.get('metrics_configuration_analysis', {})
                alarm_efficiency = alarms_data.get('alarm_efficiency', {})
                
                custom_metrics = metrics_config.get('metrics_analysis', {}).get('custom_metrics_count', 0)
                unused_alarms = alarm_efficiency.get('unused_alarms_count', 0)
                
                if custom_metrics > 10 and unused_alarms > 5:
                    insights['optimization_synergies'].append({
                        'type': 'metrics_alarms_cleanup_synergy',
                        'description': 'Cleaning up unused alarms and custom metrics together provides compound savings',
                        'custom_metrics': custom_metrics,
                        'unused_alarms': unused_alarms,
                        'recommendation': 'Coordinate metrics and alarms cleanup for maximum efficiency'
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating cross-analysis insights: {str(e)}")
        
        return insights
    
    def _generate_comprehensive_recommendations(self, individual_analyses: Dict[str, Any],
                                             aggregated_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        try:
            # Collect all individual recommendations
            all_individual_recommendations = []
            for analysis_type, analysis_data in individual_analyses.items():
                if analysis_data.get('status') == 'success':
                    individual_recs = analysis_data.get('recommendations', [])
                    for rec in individual_recs:
                        rec['source_analysis'] = analysis_type
                        all_individual_recommendations.append(rec)
            
            # Prioritize recommendations by cost impact
            high_impact_recs = [r for r in all_individual_recommendations 
                              if r.get('priority') == 'high' or r.get('potential_savings', 0) > 100]
            
            if high_impact_recs:
                recommendations.append({
                    'type': 'high_impact_optimization',
                    'priority': 'critical',
                    'title': 'High-Impact Cost Optimization Opportunities',
                    'description': f'Found {len(high_impact_recs)} high-impact optimization opportunities',
                    'total_potential_savings': sum(r.get('potential_savings', 0) for r in high_impact_recs),
                    'recommendations': high_impact_recs[:5]  # Top 5
                })
            
            # Add cross-analysis recommendations
            for insight_category, insights in aggregated_insights.items():
                if insights:
                    recommendations.append({
                        'type': f'cross_analysis_{insight_category}',
                        'priority': 'medium',
                        'title': f'Cross-Analysis Insights: {insight_category.replace("_", " ").title()}',
                        'description': f'Found {len(insights)} insights from cross-analysis',
                        'insights': insights
                    })
            
            # Add implementation strategy recommendation
            if len(individual_analyses) > 2:
                recommendations.append({
                    'type': 'implementation_strategy',
                    'priority': 'low',
                    'title': 'Recommended Implementation Strategy',
                    'description': 'Suggested order for implementing CloudWatch optimizations',
                    'strategy': [
                        '1. Start with logs optimization (typically highest cost impact)',
                        '2. Clean up unused alarms and dashboards (quick wins)',
                        '3. Optimize custom metrics (ongoing cost reduction)',
                        '4. Implement monitoring governance (prevent future waste)'
                    ]
                })
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendations: {str(e)}")
        
        return recommendations
    
    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis types."""
        return list(self.analyzer_instances.keys())
    
    def get_enabled_analyses(self) -> List[str]:
        """Get list of enabled analysis types."""
        return [
            name for name, config in self.analyzer_configs.items()
            if config.enabled and name in self.analyzer_instances
        ]
    
    def get_analysis_info(self, analysis_type: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific analysis type."""
        if analysis_type not in self.analyzer_instances:
            return {
                'error': f'Unknown analysis type: {analysis_type}',
                'available_types': list(self.analyzer_instances.keys())
            }
        
        analyzer = self.analyzer_instances[analysis_type]
        config = self.analyzer_configs.get(analysis_type)
        
        info = {
            'analysis_type': analysis_type,
            'version': getattr(analyzer, 'version', 'unknown'),
            'description': analyzer.__class__.__doc__ or 'No description available',
            'class_name': analyzer.__class__.__name__,
            'analyzer_info': analyzer.get_analyzer_info()
        }
        
        if config:
            info['configuration'] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'timeout_seconds': config.timeout_seconds,
                'retry_attempts': config.retry_attempts,
                'cache_results': config.cache_results,
                'dependencies': config.dependencies
            }
        
        return info
    
    def enable_analyzer(self, analysis_type: str) -> bool:
        """Enable a specific analyzer."""
        if analysis_type not in self.analyzer_configs:
            self.logger.error(f"Cannot enable unknown analyzer: {analysis_type}")
            return False
        
        config = self.analyzer_configs[analysis_type]
        config.enabled = True
        
        # Load instance if not already loaded
        if analysis_type not in self.analyzer_instances:
            try:
                self._load_single_analyzer(analysis_type, config)
            except Exception as e:
                self.logger.error(f"Failed to load analyzer {analysis_type}: {str(e)}")
                config.enabled = False
                return False
        
        self.logger.info(f"Enabled analyzer: {analysis_type}")
        return True
    
    def disable_analyzer(self, analysis_type: str) -> bool:
        """Disable a specific analyzer."""
        if analysis_type not in self.analyzer_configs:
            self.logger.error(f"Cannot disable unknown analyzer: {analysis_type}")
            return False
        
        self.analyzer_configs[analysis_type].enabled = False
        self.logger.info(f"Disabled analyzer: {analysis_type}")
        return True
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status and health information."""
        uptime = (datetime.now() - self.engine_start_time).total_seconds()
        
        # Calculate success rate
        success_rate = 0.0
        if self.total_analyses_run > 0:
            success_rate = (self.successful_analyses / self.total_analyses_run) * 100
        
        # Get active analyses
        active_count = len(self.active_analyses)
        
        # Get cache statistics
        cache_stats = {}
        if self.cache:
            cache_stats = self.cache.get_statistics()
        
        # Get memory statistics
        memory_stats = {}
        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_statistics()
        
        # Get performance statistics
        performance_stats = {}
        if self.performance_monitor:
            performance_stats = self.performance_monitor.get_performance_summary()
        
        return {
            'engine_info': {
                'engine_version': '2.0.0',
                'engine_id': self.engine_id,
                'region': self.region,
                'session_id': self.session_id,
                'uptime_seconds': uptime,
                'started_at': self.engine_start_time.isoformat()
            },
            'analyzer_status': {
                'total_analyzers': len(self.analyzer_configs),
                'enabled_analyzers': len(self.get_enabled_analyses()),
                'loaded_analyzers': len(self.analyzer_instances),
                'analyzer_types': list(self.analyzer_instances.keys()),
                'enabled_types': self.get_enabled_analyses()
            },
            'execution_statistics': {
                'total_analyses_run': self.total_analyses_run,
                'successful_analyses': self.successful_analyses,
                'failed_analyses': self.failed_analyses,
                'success_rate_percent': success_rate,
                'active_analyses': active_count
            },
            'configuration': {
                'max_parallel_analyzers': self.config.max_parallel_analyzers,
                'default_timeout_seconds': self.config.default_timeout_seconds,
                'enable_caching': self.config.enable_caching,
                'enable_performance_monitoring': self.config.enable_performance_monitoring,
                'enable_memory_management': self.config.enable_memory_management,
                'cache_ttl_seconds': self.config.cache_ttl_seconds,
                'cross_analysis_insights': self.config.cross_analysis_insights
            },
            'service_orchestrator_status': self.service_orchestrator.get_session_info(),
            'cache_statistics': cache_stats,
            'memory_statistics': memory_stats,
            'performance_statistics': performance_stats
        }
    
    def get_active_analyses(self) -> Dict[str, Any]:
        """Get information about currently running analyses."""
        active_info = {}
        
        for execution_id, analysis_info in self.active_analyses.items():
            runtime = (datetime.now() - analysis_info['start_time']).total_seconds()
            
            active_info[execution_id] = {
                'analysis_type': analysis_info['analysis_type'],
                'start_time': analysis_info['start_time'].isoformat(),
                'runtime_seconds': runtime,
                'status': analysis_info['status'],
                'has_monitoring': analysis_info.get('monitoring_session') is not None,
                'has_memory_tracking': analysis_info.get('memory_tracker') is not None
            }
        
        return active_info
    
    def cancel_analysis(self, execution_id: str) -> bool:
        """
        Cancel a running analysis (best effort).
        
        Args:
            execution_id: Execution ID of the analysis to cancel
            
        Returns:
            True if cancellation was attempted, False if analysis not found
        """
        if execution_id not in self.active_analyses:
            return False
        
        analysis_info = self.active_analyses[execution_id]
        analysis_info['status'] = 'cancelled'
        
        # End monitoring
        monitoring_session = analysis_info.get('monitoring_session')
        memory_tracker = analysis_info.get('memory_tracker')
        
        self._end_monitoring(monitoring_session, memory_tracker, False, "Analysis cancelled")
        
        # Remove from active analyses
        del self.active_analyses[execution_id]
        
        self.logger.info(f"Cancelled analysis: {execution_id}")
        return True
    
    def clear_cache(self, analysis_type: str = None) -> bool:
        """
        Clear analysis results cache.
        
        Args:
            analysis_type: Specific analysis type to clear, or None for all
            
        Returns:
            True if cache was cleared successfully
        """
        if not self.cache:
            return False
        
        try:
            if analysis_type:
                # Clear specific analysis type
                cleared = self.cache.invalidate_by_tags({'analysis_type': analysis_type})
                self.logger.info(f"Cleared {cleared} cache entries for {analysis_type}")
            else:
                # Clear all analysis results
                self.cache.clear()
                self.logger.info("Cleared all analysis results cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    def warm_cache(self, analysis_types: List[str] = None, **kwargs) -> Dict[str, bool]:
        """
        Warm the cache by running analyses and caching results.
        
        Args:
            analysis_types: List of analysis types to warm, or None for all enabled
            **kwargs: Parameters for the analyses
            
        Returns:
            Dictionary mapping analysis types to success status
        """
        if not self.cache:
            return {}
        
        if analysis_types is None:
            analysis_types = self.get_enabled_analyses()
        
        warming_results = {}
        
        for analysis_type in analysis_types:
            if analysis_type not in self.analyzer_instances:
                warming_results[analysis_type] = False
                continue
            
            try:
                # Run analysis to warm cache
                asyncio.create_task(self.run_analysis(analysis_type, **kwargs))
                warming_results[analysis_type] = True
                self.logger.info(f"Started cache warming for {analysis_type}")
                
            except Exception as e:
                self.logger.error(f"Failed to warm cache for {analysis_type}: {str(e)}")
                warming_results[analysis_type] = False
        
        return warming_results
    
    def shutdown(self):
        """Shutdown the analysis engine and cleanup resources."""
        self.logger.info(f"Shutting down CloudWatch Analysis Engine {self.engine_id}")
        
        # Cancel active analyses
        for execution_id in list(self.active_analyses.keys()):
            self.cancel_analysis(execution_id)
        
        # Cleanup service orchestrator
        if hasattr(self.service_orchestrator, 'cleanup_session'):
            self.service_orchestrator.cleanup_session()
        
        # Shutdown performance components
        if self.performance_monitor and hasattr(self.performance_monitor, 'shutdown'):
            self.performance_monitor.shutdown()
        
        if self.memory_manager and hasattr(self.memory_manager, 'shutdown'):
            self.memory_manager.shutdown()
        
        if self.cache and hasattr(self.cache, 'shutdown'):
            self.cache.shutdown()
        
        self.logger.info("CloudWatch Analysis Engine shutdown complete")


# Global analysis engine instance management
_analysis_engines: Dict[str, CloudWatchAnalysisEngine] = {}

def get_analysis_engine(region: str = None, session_id: str = None, 
                       config: EngineConfig = None) -> CloudWatchAnalysisEngine:
    """
    Get or create a CloudWatch Analysis Engine instance.
    
    Args:
        region: AWS region
        session_id: Session ID for data persistence
        config: Engine configuration
        
    Returns:
        CloudWatchAnalysisEngine instance
    """
    engine_key = f"{region or 'default'}_{session_id or 'default'}"
    
    if engine_key not in _analysis_engines:
        _analysis_engines[engine_key] = CloudWatchAnalysisEngine(
            region=region,
            session_id=session_id,
            config=config
        )
    
    return _analysis_engines[engine_key]

def shutdown_all_engines():
    """Shutdown all analysis engine instances."""
    for engine_key, engine in _analysis_engines.items():
        try:
            engine.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down engine {engine_key}: {str(e)}")
    
    _analysis_engines.clear()
    logger.info("All CloudWatch Analysis Engines shut down")

# Convenience functions for common operations
async def run_cloudwatch_analysis(analysis_type: str, region: str = None, 
                                session_id: str = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a single CloudWatch analysis.
    
    Args:
        analysis_type: Type of analysis to run
        region: AWS region
        session_id: Session ID
        **kwargs: Analysis parameters
        
    Returns:
        Analysis results
    """
    engine = get_analysis_engine(region=region, session_id=session_id)
    return await engine.run_analysis(analysis_type, **kwargs)

async def run_comprehensive_cloudwatch_analysis(region: str = None, session_id: str = None, 
                                              **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive CloudWatch analysis.
    
    Args:
        region: AWS region
        session_id: Session ID
        **kwargs: Analysis parameters
        
    Returns:
        Comprehensive analysis results
    """
    engine = get_analysis_engine(region=region, session_id=session_id)
    return await engine.run_comprehensive_analysis(**kwargs)

def get_cloudwatch_analysis_status(region: str = None, session_id: str = None) -> Dict[str, Any]:
    """
    Get status of CloudWatch analysis engine.
    
    Args:
        region: AWS region
        session_id: Session ID
        
    Returns:
        Engine status information
    """
    engine_key = f"{region or 'default'}_{session_id or 'default'}"
    
    if engine_key not in _analysis_engines:
        return {
            'engine_exists': False,
            'message': 'No analysis engine found for the specified region and session'
        }
    
    engine = _analysis_engines[engine_key]
    status = engine.get_engine_status()
    status['engine_exists'] = True
    
    return status