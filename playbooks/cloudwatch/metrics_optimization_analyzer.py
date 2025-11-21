"""
Metrics Optimization Analyzer for CloudWatch Optimization

Implements CloudWatch Metrics cost optimization analysis using Cost Explorer integration
with cost control flags and minimal-cost operations for detailed analysis.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage, get_cost_forecast
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class MetricsOptimizationAnalyzer(BaseAnalyzer):
    """
    Metrics optimization analyzer for CloudWatch Metrics cost analysis.
    
    This analyzer provides:
    - Daily cost and usage analysis by metric type using Cost Explorer (allow_cost_explorer)
    - Custom metrics usage pattern analysis using free CloudWatch APIs
    - Detailed monitoring and high-resolution metrics cost analysis with minimal-cost operations (allow_minimal_cost_metrics)
    - Metrics optimization recommendations with cost-aware feature coverage
    """
    
    def __init__(self, cost_explorer_service=None, config_service=None, 
                 metrics_service=None, cloudwatch_service=None, pricing_service=None,
                 performance_monitor=None, memory_manager=None):
        """Initialize MetricsOptimizationAnalyzer with CloudWatch services."""
        super().__init__(
            cost_explorer_service=cost_explorer_service,
            config_service=config_service,
            metrics_service=metrics_service,
            cloudwatch_service=cloudwatch_service,
            pricing_service=pricing_service,
            performance_monitor=performance_monitor,
            memory_manager=memory_manager
        )
        
        # Analysis configuration
        self.analysis_type = "metrics_optimization"
        self.version = "1.0.0"
        
        # Cost control flags
        self.cost_preferences = None
        
        # AWS service namespaces for categorizing metrics
        self.aws_namespaces = [
            'AWS/EC2', 'AWS/RDS', 'AWS/Lambda', 'AWS/S3', 'AWS/ELB', 'AWS/ELBv2',
            'AWS/ApplicationELB', 'AWS/NetworkELB', 'AWS/CloudFront', 'AWS/ApiGateway',
            'AWS/DynamoDB', 'AWS/SQS', 'AWS/SNS', 'AWS/Kinesis', 'AWS/ECS',
            'AWS/EKS', 'AWS/Batch', 'AWS/Logs', 'AWS/Events', 'AWS/AutoScaling',
            'AWS/ElastiCache', 'AWS/Redshift', 'AWS/EMR', 'AWS/Glue', 'AWS/StepFunctions'
        ]
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive CloudWatch Metrics optimization analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region
                - lookback_days: Number of days to analyze (default: 30)
                - allow_cost_explorer: Enable Cost Explorer analysis (default: False)
                - allow_minimal_cost_metrics: Enable minimal cost metrics (default: False)
                - namespace_filter: Specific namespace to analyze
                - metric_name_filter: Specific metric name to analyze
                
        Returns:
            Dictionary containing comprehensive metrics optimization analysis results
        """
        start_time = datetime.now()
        context = self.prepare_analysis_context(**kwargs)
        
        # Extract cost preferences
        self.cost_preferences = {
            'allow_cost_explorer': kwargs.get('allow_cost_explorer', False),
            'allow_minimal_cost_metrics': kwargs.get('allow_minimal_cost_metrics', False),
            'allow_aws_config': kwargs.get('allow_aws_config', False),
            'allow_cloudtrail': kwargs.get('allow_cloudtrail', False)
        }
        
        log_cloudwatch_operation(self.logger, "metrics_optimization_analysis_start",
                               cost_preferences=str(self.cost_preferences),
                               lookback_days=kwargs.get('lookback_days', 30))
        
        try:
            # Initialize result structure
            analysis_result = {
                'status': 'success',
                'analysis_type': self.analysis_type,
                'timestamp': start_time.isoformat(),
                'cost_incurred': False,
                'cost_incurring_operations': [],
                'primary_data_source': 'cloudwatch_config',
                'fallback_used': False,
                'data': {},
                'recommendations': []
            }
            
            # Execute analysis components in parallel
            analysis_tasks = []
            
            # 1. Metrics Configuration Analysis (FREE - Always enabled)
            analysis_tasks.append(self._analyze_metrics_configuration(**kwargs))
            
            # 2. Cost Explorer Metrics Analysis (PAID - User controlled)
            if self.cost_preferences['allow_cost_explorer']:
                analysis_tasks.append(self._analyze_cost_explorer_metrics(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('cost_explorer_metrics_analysis')
                analysis_result['primary_data_source'] = 'cost_explorer'
            
            # 3. Minimal Cost Metrics Analysis (PAID - User controlled)
            if self.cost_preferences['allow_minimal_cost_metrics']:
                analysis_tasks.append(self._analyze_detailed_metrics_usage(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('minimal_cost_metrics_analysis')
            
            # Execute all analysis tasks
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Metrics analysis task {i} failed: {str(result)}")
                    analysis_result['fallback_used'] = True
                elif isinstance(result, dict):
                    if result.get('status') == 'success':
                        # Merge successful results
                        analysis_result['data'].update(result.get('data', {}))
                    elif result.get('status') == 'error':
                        # Mark fallback used for error results
                        self.logger.warning(f"Metrics analysis task {i} returned error: {result.get('error_message', 'Unknown error')}")
                        analysis_result['fallback_used'] = True
            
            # Generate metrics optimization analysis
            optimization_analysis = await self._generate_metrics_optimization_analysis(analysis_result['data'], **kwargs)
            analysis_result['data']['optimization_analysis'] = optimization_analysis
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            analysis_result['execution_time'] = execution_time
            
            log_cloudwatch_operation(self.logger, "metrics_optimization_analysis_complete",
                                   execution_time=execution_time,
                                   cost_incurred=analysis_result['cost_incurred'],
                                   primary_data_source=analysis_result['primary_data_source'])
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Metrics optimization analysis failed: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_metrics_configuration(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Metrics configuration using free APIs.
        
        This provides the foundation for metrics analysis using only free operations.
        """
        log_cloudwatch_operation(self.logger, "metrics_config_analysis_start", component="metrics_configuration")
        
        try:
            config_data = {}
            
            # Get all metrics metadata (FREE)
            if self.cloudwatch_service:
                namespace_filter = kwargs.get('namespace_filter')
                metric_name_filter = kwargs.get('metric_name_filter')
                
                metrics_result = await self.cloudwatch_service.list_metrics(
                    namespace=namespace_filter,
                    metric_name=metric_name_filter
                )
                
                if metrics_result.success:
                    metrics_data = metrics_result.data
                    config_data['metrics'] = metrics_data
                    
                    # Analyze metrics configuration
                    metrics_analysis = self._analyze_metrics_metadata(metrics_data.get('metrics', []))
                    config_data['metrics_analysis'] = metrics_analysis
                    
                    log_cloudwatch_operation(self.logger, "metrics_config_analyzed",
                                           total_metrics=metrics_data.get('total_count', 0),
                                           custom_metrics=metrics_analysis.get('custom_metrics_count', 0),
                                           aws_metrics=metrics_analysis.get('aws_metrics_count', 0))
            
            return {
                'status': 'success',
                'data': {
                    'metrics_configuration_analysis': config_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Metrics configuration analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    async def _analyze_cost_explorer_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Metrics costs using Cost Explorer (PAID operation).
        
        Requires allow_cost_explorer=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "cost_explorer_metrics_analysis_start", 
                               component="cost_explorer_metrics")
        
        try:
            lookback_days = kwargs.get('lookback_days', 30)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            cost_data = {}
            
            # Get CloudWatch Metrics service costs with daily granularity
            metrics_filter = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon CloudWatch']
                }
            }
            
            # Get detailed metrics costs by usage type
            cost_result = get_cost_and_usage(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                granularity='DAILY',
                metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                group_by=[{'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}],
                filter_expr=metrics_filter,
                region=kwargs.get('region')
            )
            
            if cost_result['status'] == 'success':
                cost_data['metrics_costs'] = self._process_metrics_cost_explorer_response(
                    cost_result['data'], 'metrics'
                )
                log_cloudwatch_operation(self.logger, "metrics_costs_retrieved",
                                       total_cost=cost_data['metrics_costs'].get('total_cost', 0))
            
            # Get metrics cost forecast for next 30 days
            forecast_end = end_date + timedelta(days=30)
            forecast_result = get_cost_forecast(
                start_date=end_date.strftime('%Y-%m-%d'),
                end_date=forecast_end.strftime('%Y-%m-%d'),
                granularity='MONTHLY',
                metric='BLENDED_COST',
                filter_expr=metrics_filter,
                region=kwargs.get('region')
            )
            
            if forecast_result['status'] == 'success':
                cost_data['metrics_cost_forecast'] = self._process_forecast_response(
                    forecast_result['data']
                )
                log_cloudwatch_operation(self.logger, "metrics_cost_forecast_retrieved",
                                       forecasted_cost=cost_data['metrics_cost_forecast'].get('forecasted_amount', 0))
            
            return {
                'status': 'success',
                'data': {
                    'cost_explorer_metrics_analysis': cost_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Cost Explorer metrics analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    async def _analyze_detailed_metrics_usage(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze detailed metrics usage patterns using minimal cost operations (PAID operation).
        
        Requires allow_minimal_cost_metrics=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "detailed_metrics_usage_start",
                               component="detailed_metrics")
        
        try:
            detailed_metrics_data = {}
            lookback_days = kwargs.get('lookback_days', 30)
            
            # Get detailed monitoring metrics for EC2 instances (MINIMAL COST)
            if self.cloudwatch_service:
                detailed_monitoring_result = await self._analyze_detailed_monitoring_usage(lookback_days)
                if detailed_monitoring_result:
                    detailed_metrics_data['detailed_monitoring'] = detailed_monitoring_result
                    log_cloudwatch_operation(self.logger, "detailed_monitoring_analyzed",
                                           instances_count=detailed_monitoring_result.get('total_instances', 0))
            
            # Get high-resolution metrics analysis (MINIMAL COST)
            if self.cloudwatch_service:
                high_res_result = await self._analyze_high_resolution_metrics(lookback_days)
                if high_res_result:
                    detailed_metrics_data['high_resolution_metrics'] = high_res_result
                    log_cloudwatch_operation(self.logger, "high_resolution_metrics_analyzed",
                                           metrics_count=high_res_result.get('total_high_res_metrics', 0))
            
            # Get custom metrics usage patterns (MINIMAL COST)
            if self.cloudwatch_service:
                custom_metrics_result = await self._analyze_custom_metrics_patterns(lookback_days)
                if custom_metrics_result:
                    detailed_metrics_data['custom_metrics_patterns'] = custom_metrics_result
                    log_cloudwatch_operation(self.logger, "custom_metrics_patterns_analyzed",
                                           custom_metrics_count=custom_metrics_result.get('total_custom_metrics', 0))
            
            return {
                'status': 'success',
                'data': {
                    'detailed_metrics_usage_analysis': detailed_metrics_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Detailed metrics usage analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    def _analyze_metrics_metadata(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metrics metadata for optimization opportunities."""
        analysis = {
            'total_metrics': len(metrics),
            'custom_metrics_count': 0,
            'aws_metrics_count': 0,
            'metrics_by_namespace': {},
            'custom_namespaces': [],
            'high_cardinality_metrics': []
        }
        
        for metric in metrics:
            namespace = metric.get('Namespace', '')
            metric_name = metric.get('MetricName', '')
            dimensions = metric.get('Dimensions', [])
            
            # Count metrics by namespace
            analysis['metrics_by_namespace'][namespace] = analysis['metrics_by_namespace'].get(namespace, 0) + 1
            
            # Categorize as AWS or custom metrics
            if any(namespace.startswith(aws_ns) for aws_ns in self.aws_namespaces):
                analysis['aws_metrics_count'] += 1
            else:
                analysis['custom_metrics_count'] += 1
                if namespace not in analysis['custom_namespaces']:
                    analysis['custom_namespaces'].append(namespace)
            
            # Identify high cardinality metrics (many dimensions)
            if len(dimensions) > 5:
                analysis['high_cardinality_metrics'].append({
                    'namespace': namespace,
                    'metric_name': metric_name,
                    'dimensions_count': len(dimensions),
                    'dimensions': dimensions
                })
            
            # Note: Optimization reasons are now added directly to each metric in the result processor
        
        # Calculate free tier usage
        analysis['free_tier_analysis'] = {
            'free_tier_limit': 10,
            'custom_metrics_beyond_free_tier': max(0, analysis['custom_metrics_count'] - 10),
            'within_free_tier': analysis['custom_metrics_count'] <= 10
        }
        
        return analysis
    
    def _process_metrics_cost_explorer_response(self, response_data: Dict[str, Any], 
                                              service_type: str) -> Dict[str, Any]:
        """Process Cost Explorer API response for metrics-specific cost data."""
        processed_data = {
            'service_type': service_type,
            'total_cost': 0.0,
            'daily_costs': [],
            'usage_types': {},
            'metrics_specific_costs': {},
            'cost_trends': {}
        }
        
        try:
            results_by_time = response_data.get('ResultsByTime', [])
            
            for time_period in results_by_time:
                time_start = time_period.get('TimePeriod', {}).get('Start')
                groups = time_period.get('Groups', [])
                
                daily_cost = 0.0
                daily_usage_types = {}
                
                for group in groups:
                    usage_type = group.get('Keys', ['Unknown'])[0]
                    metrics = group.get('Metrics', {})
                    
                    blended_cost = float(metrics.get('BlendedCost', {}).get('Amount', 0))
                    usage_quantity = float(metrics.get('UsageQuantity', {}).get('Amount', 0))
                    
                    daily_cost += blended_cost
                    daily_usage_types[usage_type] = {
                        'cost': blended_cost,
                        'usage': usage_quantity,
                        'unit': metrics.get('UsageQuantity', {}).get('Unit', 'Unknown')
                    }
                    
                    # Categorize metrics-specific usage types
                    self._categorize_metrics_usage_type(usage_type, blended_cost, usage_quantity, processed_data)
                    
                    # Aggregate usage types
                    if usage_type not in processed_data['usage_types']:
                        processed_data['usage_types'][usage_type] = {
                            'total_cost': 0.0,
                            'total_usage': 0.0,
                            'unit': metrics.get('UsageQuantity', {}).get('Unit', 'Unknown')
                        }
                    
                    processed_data['usage_types'][usage_type]['total_cost'] += blended_cost
                    processed_data['usage_types'][usage_type]['total_usage'] += usage_quantity
                
                processed_data['daily_costs'].append({
                    'date': time_start,
                    'total_cost': daily_cost,
                    'usage_types': daily_usage_types
                })
                
                processed_data['total_cost'] += daily_cost
            
            # Calculate cost trends
            if len(processed_data['daily_costs']) >= 2:
                recent_costs = [day['total_cost'] for day in processed_data['daily_costs'][-7:]]
                earlier_costs = [day['total_cost'] for day in processed_data['daily_costs'][-14:-7]]
                
                if recent_costs and earlier_costs:
                    recent_avg = sum(recent_costs) / len(recent_costs)
                    earlier_avg = sum(earlier_costs) / len(earlier_costs)
                    
                    if earlier_avg > 0:
                        trend_percentage = ((recent_avg - earlier_avg) / earlier_avg) * 100
                        processed_data['cost_trends'] = {
                            'recent_average': recent_avg,
                            'earlier_average': earlier_avg,
                            'trend_percentage': trend_percentage,
                            'trend_direction': 'increasing' if trend_percentage > 5 else 'decreasing' if trend_percentage < -5 else 'stable'
                        }
            
        except Exception as e:
            self.logger.error(f"Error processing metrics Cost Explorer response: {str(e)}")
        
        return processed_data
    
    def _categorize_metrics_usage_type(self, usage_type: str, cost: float, usage: float, processed_data: Dict[str, Any]):
        """Categorize metrics usage types into specific cost categories."""
        usage_type_lower = usage_type.lower()
        
        if 'metric' in usage_type_lower:
            if 'custom' in usage_type_lower:
                processed_data['metrics_specific_costs']['custom_metrics'] = \
                    processed_data['metrics_specific_costs'].get('custom_metrics', 0.0) + cost
            elif 'detailed' in usage_type_lower or 'monitoring' in usage_type_lower:
                processed_data['metrics_specific_costs']['detailed_monitoring'] = \
                    processed_data['metrics_specific_costs'].get('detailed_monitoring', 0.0) + cost
            elif 'highresolution' in usage_type_lower or 'high-resolution' in usage_type_lower:
                processed_data['metrics_specific_costs']['high_resolution'] = \
                    processed_data['metrics_specific_costs'].get('high_resolution', 0.0) + cost
            else:
                processed_data['metrics_specific_costs']['other_metrics'] = \
                    processed_data['metrics_specific_costs'].get('other_metrics', 0.0) + cost
        elif 'request' in usage_type_lower or 'api' in usage_type_lower:
            processed_data['metrics_specific_costs']['api_requests'] = \
                processed_data['metrics_specific_costs'].get('api_requests', 0.0) + cost
        else:
            processed_data['metrics_specific_costs']['other'] = \
                processed_data['metrics_specific_costs'].get('other', 0.0) + cost
    
    def _process_forecast_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Cost Explorer forecast response."""
        processed_data = {
            'forecasted_amount': 0.0,
            'forecast_confidence': 'UNKNOWN',
            'forecast_period': {}
        }
        
        try:
            forecast_results = response_data.get('ForecastResultsByTime', [])
            
            if forecast_results:
                forecast = forecast_results[0]  # Take first forecast period
                mean_value = forecast.get('MeanValue', '0')
                processed_data['forecasted_amount'] = float(mean_value)
                
                time_period = forecast.get('TimePeriod', {})
                processed_data['forecast_period'] = {
                    'start': time_period.get('Start'),
                    'end': time_period.get('End')
                }
            
            # Get prediction interval confidence
            prediction_interval = response_data.get('PredictionIntervalLowerBound', '0')
            if float(prediction_interval) > 0:
                processed_data['forecast_confidence'] = 'HIGH'
            else:
                processed_data['forecast_confidence'] = 'MEDIUM'
                
        except Exception as e:
            self.logger.error(f"Error processing metrics forecast response: {str(e)}")
        
        return processed_data
    
    async def _analyze_detailed_monitoring_usage(self, lookback_days: int) -> Optional[Dict[str, Any]]:
        """Analyze detailed monitoring usage for EC2 instances (MINIMAL COST)."""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=lookback_days)
            
            # Get EC2 detailed monitoring metrics
            detailed_monitoring_result = await self.cloudwatch_service.get_targeted_metric_statistics(
                namespace='AWS/EC2',
                metric_name='CPUUtilization',
                dimensions=[],  # Get all instances
                start_time=start_time,
                end_time=end_time,
                period=300,  # 5-minute periods indicate detailed monitoring
                statistics=['Average']
            )
            
            if detailed_monitoring_result.success:
                datapoints = detailed_monitoring_result.data.get('datapoints', [])
                
                # Count instances with detailed monitoring (5-minute periods)
                detailed_monitoring_instances = set()
                for datapoint in datapoints:
                    # If we have 5-minute datapoints, it indicates detailed monitoring
                    detailed_monitoring_instances.add('detailed_monitoring_detected')
                
                return {
                    'total_instances': len(detailed_monitoring_instances),
                    'detailed_monitoring_detected': len(detailed_monitoring_instances) > 0,
                    'analysis_period_days': lookback_days,
                    'optimization_opportunity': len(detailed_monitoring_instances) > 0
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing detailed monitoring usage: {str(e)}")
        
        return None
    
    async def _analyze_high_resolution_metrics(self, lookback_days: int) -> Optional[Dict[str, Any]]:
        """Analyze high-resolution metrics usage (MINIMAL COST)."""
        try:
            # High-resolution metrics have periods less than 60 seconds
            # We can detect them by looking for sub-minute resolution data
            
            high_res_analysis = {
                'total_high_res_metrics': 0,
                'high_res_namespaces': [],
                'potential_cost_savings': 0.0,
                'optimization_opportunities': []
            }
            
            # This would require specific metric queries to detect high-resolution usage
            # For now, we'll provide a framework for the analysis
            
            return high_res_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing high-resolution metrics: {str(e)}")
        
        return None
    
    async def _analyze_custom_metrics_patterns(self, lookback_days: int) -> Optional[Dict[str, Any]]:
        """Analyze custom metrics usage patterns (MINIMAL COST)."""
        try:
            # Get metrics configuration first (free operation)
            metrics_result = await self.cloudwatch_service.list_metrics()
            
            if not metrics_result.success:
                return None
            
            metrics = metrics_result.data.get('metrics', [])
            custom_metrics = [m for m in metrics if not any(m.get('Namespace', '').startswith(aws_ns) for aws_ns in self.aws_namespaces)]
            
            custom_metrics_analysis = {
                'total_custom_metrics': len(custom_metrics),
                'custom_namespaces': list(set(m.get('Namespace', '') for m in custom_metrics)),
                'high_cardinality_metrics': [],
                'optimization_opportunities': []
            }
            
            # Analyze cardinality and optimization opportunities
            for metric in custom_metrics:
                dimensions = metric.get('Dimensions', [])
                if len(dimensions) > 5:
                    custom_metrics_analysis['high_cardinality_metrics'].append({
                        'namespace': metric.get('Namespace'),
                        'metric_name': metric.get('MetricName'),
                        'dimensions_count': len(dimensions)
                    })
                    
                    custom_metrics_analysis['optimization_opportunities'].append({
                        'type': 'reduce_cardinality',
                        'metric': f"{metric.get('Namespace')}/{metric.get('MetricName')}",
                        'current_dimensions': len(dimensions),
                        'recommended_dimensions': min(3, len(dimensions)),
                        'potential_savings_percentage': min(50, (len(dimensions) - 3) * 10)
                    })
            
            return custom_metrics_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing custom metrics patterns: {str(e)}")
        
        return None
    
    async def _generate_metrics_optimization_analysis(self, analysis_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate comprehensive metrics optimization analysis."""
        optimization_analysis = {
            'custom_metrics_optimization': {},
            'detailed_monitoring_optimization': {},
            'high_resolution_metrics_optimization': {},
            'api_requests_optimization': {},
            'total_potential_savings': 0.0,
            'optimization_priorities': []
        }
        
        try:
            # Analyze custom metrics optimization
            if self.pricing_service:
                custom_metrics_analysis = await self._analyze_custom_metrics_optimization(analysis_data)
                optimization_analysis['custom_metrics_optimization'] = custom_metrics_analysis
                optimization_analysis['total_potential_savings'] += custom_metrics_analysis.get('potential_monthly_savings', 0.0)
            
            # Analyze detailed monitoring optimization
            if self.pricing_service:
                detailed_monitoring_analysis = await self._analyze_detailed_monitoring_optimization(analysis_data)
                optimization_analysis['detailed_monitoring_optimization'] = detailed_monitoring_analysis
                optimization_analysis['total_potential_savings'] += detailed_monitoring_analysis.get('potential_monthly_savings', 0.0)
            
            # Analyze high-resolution metrics optimization
            if self.pricing_service:
                high_res_analysis = await self._analyze_high_resolution_optimization(analysis_data)
                optimization_analysis['high_resolution_metrics_optimization'] = high_res_analysis
                optimization_analysis['total_potential_savings'] += high_res_analysis.get('potential_monthly_savings', 0.0)
            
            # Generate optimization priorities
            optimization_analysis['optimization_priorities'] = self._generate_optimization_priorities(optimization_analysis)
            
        except Exception as e:
            self.logger.error(f"Error generating metrics optimization analysis: {str(e)}")
        
        return optimization_analysis
    
    async def _analyze_custom_metrics_optimization(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze custom metrics optimization opportunities."""
        custom_metrics_optimization = {
            'current_custom_metrics_count': 0,
            'estimated_monthly_cost': 0.0,
            'potential_monthly_savings': 0.0,
            'optimization_opportunities': []
        }
        
        try:
            # Get metrics configuration data
            config_data = analysis_data.get('metrics_configuration_analysis', {})
            metrics_analysis = config_data.get('metrics_analysis', {})
            
            custom_metrics_count = metrics_analysis.get('custom_metrics_count', 0)
            custom_metrics_optimization['current_custom_metrics_count'] = custom_metrics_count
            
            # Get metrics pricing
            if self.pricing_service:
                metrics_pricing = self.pricing_service.get_metrics_pricing()
                
                if metrics_pricing.get('status') == 'success':
                    # Calculate current costs
                    cost_calculation = self.pricing_service.calculate_metrics_cost(
                        custom_metrics_count=custom_metrics_count,
                        api_requests_count=100000,  # Estimate
                        detailed_monitoring_instances=0
                    )
                    
                    if cost_calculation.get('status') == 'success':
                        custom_metrics_optimization['estimated_monthly_cost'] = cost_calculation['total_monthly_cost']
                        
                        # Identify optimization opportunities
                        if custom_metrics_count > 10:  # Beyond free tier
                            excess_metrics = custom_metrics_count - 10
                            pricing = metrics_pricing['metrics_pricing']
                            
                            # Opportunity 1: Reduce high-cardinality metrics
                            high_cardinality_metrics = metrics_analysis.get('high_cardinality_metrics', [])
                            if high_cardinality_metrics:
                                potential_reduction = min(excess_metrics, len(high_cardinality_metrics) // 2)
                                potential_savings = potential_reduction * pricing['custom_metrics_per_metric']
                                
                                custom_metrics_optimization['optimization_opportunities'].append({
                                    'type': 'reduce_high_cardinality_metrics',
                                    'description': f'Reduce cardinality of {len(high_cardinality_metrics)} high-cardinality metrics',
                                    'affected_metrics': len(high_cardinality_metrics),
                                    'potential_metric_reduction': potential_reduction,
                                    'potential_monthly_savings': potential_savings,
                                    'implementation_effort': 'medium'
                                })
                                
                                custom_metrics_optimization['potential_monthly_savings'] += potential_savings
                            
                            # Opportunity 2: Consolidate similar metrics
                            custom_namespaces = metrics_analysis.get('custom_namespaces', [])
                            if len(custom_namespaces) > 5:
                                consolidation_savings = min(excess_metrics * 0.3, len(custom_namespaces) * 2) * pricing['custom_metrics_per_metric']
                                
                                custom_metrics_optimization['optimization_opportunities'].append({
                                    'type': 'consolidate_namespaces',
                                    'description': f'Consolidate metrics across {len(custom_namespaces)} custom namespaces',
                                    'affected_namespaces': len(custom_namespaces),
                                    'potential_monthly_savings': consolidation_savings,
                                    'implementation_effort': 'high'
                                })
                                
                                custom_metrics_optimization['potential_monthly_savings'] += consolidation_savings
            
        except Exception as e:
            self.logger.error(f"Error analyzing custom metrics optimization: {str(e)}")
        
        return custom_metrics_optimization
    
    async def _analyze_detailed_monitoring_optimization(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed monitoring optimization opportunities."""
        detailed_monitoring_optimization = {
            'detailed_monitoring_instances': 0,
            'estimated_monthly_cost': 0.0,
            'potential_monthly_savings': 0.0,
            'optimization_opportunities': []
        }
        
        try:
            # Get detailed monitoring data
            detailed_metrics_data = analysis_data.get('detailed_metrics_usage_analysis', {})
            detailed_monitoring_data = detailed_metrics_data.get('detailed_monitoring', {})
            
            if detailed_monitoring_data:
                instances_count = detailed_monitoring_data.get('total_instances', 0)
                detailed_monitoring_optimization['detailed_monitoring_instances'] = instances_count
                
                # Get metrics pricing
                if self.pricing_service and instances_count > 0:
                    metrics_pricing = self.pricing_service.get_metrics_pricing()
                    
                    if metrics_pricing.get('status') == 'success':
                        pricing = metrics_pricing['metrics_pricing']
                        
                        # Calculate current detailed monitoring costs
                        monthly_cost = instances_count * pricing['detailed_monitoring_per_instance']
                        detailed_monitoring_optimization['estimated_monthly_cost'] = monthly_cost
                        
                        # Optimization opportunity: Disable detailed monitoring for low-utilization instances
                        if instances_count > 0:
                            # Assume 30% of instances could disable detailed monitoring
                            optimizable_instances = int(instances_count * 0.3)
                            potential_savings = optimizable_instances * pricing['detailed_monitoring_per_instance']
                            
                            detailed_monitoring_optimization['optimization_opportunities'].append({
                                'type': 'disable_detailed_monitoring',
                                'description': f'Disable detailed monitoring for {optimizable_instances} low-utilization instances',
                                'affected_instances': optimizable_instances,
                                'potential_monthly_savings': potential_savings,
                                'implementation_effort': 'low'
                            })
                            
                            detailed_monitoring_optimization['potential_monthly_savings'] = potential_savings
            
        except Exception as e:
            self.logger.error(f"Error analyzing detailed monitoring optimization: {str(e)}")
        
        return detailed_monitoring_optimization
    
    async def _analyze_high_resolution_optimization(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze high-resolution metrics optimization opportunities."""
        high_res_optimization = {
            'high_resolution_metrics_count': 0,
            'estimated_monthly_cost': 0.0,
            'potential_monthly_savings': 0.0,
            'optimization_opportunities': []
        }
        
        try:
            # Get high-resolution metrics data
            detailed_metrics_data = analysis_data.get('detailed_metrics_usage_analysis', {})
            high_res_data = detailed_metrics_data.get('high_resolution_metrics', {})
            
            if high_res_data:
                high_res_count = high_res_data.get('total_high_res_metrics', 0)
                high_res_optimization['high_resolution_metrics_count'] = high_res_count
                
                # Get metrics pricing
                if self.pricing_service and high_res_count > 0:
                    metrics_pricing = self.pricing_service.get_metrics_pricing()
                    
                    if metrics_pricing.get('status') == 'success':
                        pricing = metrics_pricing['metrics_pricing']
                        
                        # High-resolution metrics cost the same as custom metrics but with higher frequency
                        # Assume 3x cost due to higher frequency
                        monthly_cost = high_res_count * pricing['high_resolution_metrics_per_metric'] * 3
                        high_res_optimization['estimated_monthly_cost'] = monthly_cost
                        
                        # Optimization opportunity: Convert to standard resolution
                        if high_res_count > 0:
                            # Assume 50% could be converted to standard resolution
                            convertible_metrics = int(high_res_count * 0.5)
                            # Savings from reducing frequency (2/3 of cost)
                            potential_savings = convertible_metrics * pricing['high_resolution_metrics_per_metric'] * 2
                            
                            high_res_optimization['optimization_opportunities'].append({
                                'type': 'convert_to_standard_resolution',
                                'description': f'Convert {convertible_metrics} high-resolution metrics to standard resolution',
                                'affected_metrics': convertible_metrics,
                                'potential_monthly_savings': potential_savings,
                                'implementation_effort': 'medium'
                            })
                            
                            high_res_optimization['potential_monthly_savings'] = potential_savings
            
        except Exception as e:
            self.logger.error(f"Error analyzing high-resolution optimization: {str(e)}")
        
        return high_res_optimization
    
    def _generate_optimization_priorities(self, optimization_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized list of optimization opportunities."""
        priorities = []
        
        # Collect all optimization opportunities
        all_opportunities = []
        
        for category, analysis in optimization_analysis.items():
            if isinstance(analysis, dict) and 'optimization_opportunities' in analysis:
                for opportunity in analysis['optimization_opportunities']:
                    opportunity['category'] = category
                    all_opportunities.append(opportunity)
        
        # Sort by potential savings (descending) and implementation effort (ascending)
        effort_weights = {'low': 1, 'medium': 2, 'high': 3}
        
        all_opportunities.sort(
            key=lambda x: (
                -x.get('potential_monthly_savings', 0),  # Higher savings first
                effort_weights.get(x.get('implementation_effort', 'medium'), 2)  # Lower effort first
            )
        )
        
        # Create prioritized recommendations
        for i, opportunity in enumerate(all_opportunities[:10]):  # Top 10 opportunities
            priority_level = 'high' if i < 3 else 'medium' if i < 7 else 'low'
            
            priorities.append({
                'priority': priority_level,
                'rank': i + 1,
                'category': opportunity.get('category', 'unknown'),
                'type': opportunity.get('type', 'unknown'),
                'description': opportunity.get('description', ''),
                'potential_monthly_savings': opportunity.get('potential_monthly_savings', 0),
                'implementation_effort': opportunity.get('implementation_effort', 'medium'),
                'roi_score': opportunity.get('potential_monthly_savings', 0) / effort_weights.get(opportunity.get('implementation_effort', 'medium'), 2)
            })
        
        return priorities
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate metrics optimization recommendations from analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            optimization_analysis = analysis_results.get('data', {}).get('optimization_analysis', {})
            
            # Custom metrics recommendations
            custom_metrics_opt = optimization_analysis.get('custom_metrics_optimization', {})
            for opportunity in custom_metrics_opt.get('optimization_opportunities', []):
                recommendations.append(self.create_recommendation(
                    rec_type='cost_optimization',
                    priority='high' if opportunity.get('potential_monthly_savings', 0) > 10 else 'medium',
                    title=f"Custom Metrics Optimization: {opportunity.get('type', 'Unknown').replace('_', ' ').title()}",
                    description=opportunity.get('description', ''),
                    potential_savings=opportunity.get('potential_monthly_savings', 0),
                    implementation_effort=opportunity.get('implementation_effort', 'medium'),
                    cloudwatch_component='metrics',
                    action_items=[
                        f"Review {opportunity.get('affected_metrics', 0)} high-cardinality metrics",
                        "Reduce metric dimensions where possible",
                        "Consider metric aggregation strategies",
                        "Monitor custom metrics usage patterns"
                    ]
                ))
            
            # Detailed monitoring recommendations
            detailed_monitoring_opt = optimization_analysis.get('detailed_monitoring_optimization', {})
            for opportunity in detailed_monitoring_opt.get('optimization_opportunities', []):
                recommendations.append(self.create_recommendation(
                    rec_type='cost_optimization',
                    priority='medium',
                    title=f"Detailed Monitoring Optimization: {opportunity.get('type', 'Unknown').replace('_', ' ').title()}",
                    description=opportunity.get('description', ''),
                    potential_savings=opportunity.get('potential_monthly_savings', 0),
                    implementation_effort=opportunity.get('implementation_effort', 'low'),
                    cloudwatch_component='metrics',
                    action_items=[
                        f"Review {opportunity.get('affected_instances', 0)} EC2 instances with detailed monitoring",
                        "Analyze CPU and network utilization patterns",
                        "Disable detailed monitoring for low-utilization instances",
                        "Use CloudWatch agent for custom metrics instead"
                    ]
                ))
            
            # High-resolution metrics recommendations
            high_res_opt = optimization_analysis.get('high_resolution_metrics_optimization', {})
            for opportunity in high_res_opt.get('optimization_opportunities', []):
                recommendations.append(self.create_recommendation(
                    rec_type='cost_optimization',
                    priority='medium',
                    title=f"High-Resolution Metrics Optimization: {opportunity.get('type', 'Unknown').replace('_', ' ').title()}",
                    description=opportunity.get('description', ''),
                    potential_savings=opportunity.get('potential_monthly_savings', 0),
                    implementation_effort=opportunity.get('implementation_effort', 'medium'),
                    cloudwatch_component='metrics',
                    action_items=[
                        f"Review {opportunity.get('affected_metrics', 0)} high-resolution metrics",
                        "Evaluate if sub-minute resolution is necessary",
                        "Convert to standard 5-minute resolution where appropriate",
                        "Use high-resolution metrics only for critical monitoring"
                    ]
                ))
            
            # General metrics optimization recommendations
            config_analysis = analysis_results.get('data', {}).get('metrics_configuration_analysis', {})
            metrics_analysis = config_analysis.get('metrics_analysis', {})
            
            # Free tier optimization
            free_tier_analysis = metrics_analysis.get('free_tier_analysis', {})
            if not free_tier_analysis.get('within_free_tier', True):
                beyond_free_tier = free_tier_analysis.get('custom_metrics_beyond_free_tier', 0)
                recommendations.append(self.create_recommendation(
                    rec_type='cost_optimization',
                    priority='high',
                    title="Custom Metrics Exceeding Free Tier",
                    description=f"You have {beyond_free_tier} custom metrics beyond the free tier limit of 10",
                    potential_savings=beyond_free_tier * 0.30,  # Approximate cost per metric
                    implementation_effort='medium',
                    cloudwatch_component='metrics',
                    action_items=[
                        f"Review {beyond_free_tier} custom metrics beyond free tier",
                        "Consolidate similar metrics where possible",
                        "Remove unused or redundant metrics",
                        "Consider using CloudWatch Logs for detailed data instead"
                    ]
                ))
            
            # High cardinality metrics recommendation
            high_cardinality_metrics = metrics_analysis.get('high_cardinality_metrics', [])
            if high_cardinality_metrics:
                recommendations.append(self.create_recommendation(
                    rec_type='performance',
                    priority='medium',
                    title="High Cardinality Metrics Detected",
                    description=f"Found {len(high_cardinality_metrics)} metrics with high dimension cardinality",
                    implementation_effort='medium',
                    cloudwatch_component='metrics',
                    affected_resources=[f"{m.get('namespace')}/{m.get('metric_name')}" for m in high_cardinality_metrics[:5]],
                    action_items=[
                        "Review metrics with more than 5 dimensions",
                        "Reduce dimension cardinality where possible",
                        "Use tags instead of dimensions for high-cardinality data",
                        "Consider metric aggregation strategies"
                    ]
                ))
            
            # Cost-aware feature coverage recommendation
            if not self.cost_preferences.get('allow_cost_explorer', False):
                recommendations.append(self.create_recommendation(
                    rec_type='governance',
                    priority='low',
                    title="Enable Cost Explorer for Detailed Metrics Analysis",
                    description="Cost Explorer analysis is disabled. Enable it for comprehensive metrics cost analysis",
                    implementation_effort='low',
                    cloudwatch_component='metrics',
                    action_items=[
                        "Set allow_cost_explorer=True for detailed cost analysis",
                        "Review daily cost and usage patterns by metric type",
                        "Identify cost trends and optimization opportunities",
                        "Get accurate cost forecasting for metrics usage"
                    ]
                ))
            
            if not self.cost_preferences.get('allow_minimal_cost_metrics', False):
                recommendations.append(self.create_recommendation(
                    rec_type='governance',
                    priority='low',
                    title="Enable Minimal Cost Metrics for Detailed Analysis",
                    description="Minimal cost metrics analysis is disabled. Enable it for detailed usage pattern analysis",
                    implementation_effort='low',
                    cloudwatch_component='metrics',
                    action_items=[
                        "Set allow_minimal_cost_metrics=True for detailed analysis",
                        "Analyze detailed monitoring and high-resolution metrics usage",
                        "Get custom metrics usage patterns",
                        "Identify specific optimization opportunities"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating metrics optimization recommendations: {str(e)}")
            recommendations.append(self.create_recommendation(
                rec_type='error_resolution',
                priority='high',
                title="Metrics Analysis Error",
                description=f"Failed to generate metrics recommendations: {str(e)}",
                implementation_effort='low',
                cloudwatch_component='metrics',
                action_items=[
                    "Check CloudWatch API permissions",
                    "Verify Cost Explorer access if enabled",
                    "Review analysis parameters",
                    "Check CloudWatch service availability"
                ]
            ))
        
        return recommendations