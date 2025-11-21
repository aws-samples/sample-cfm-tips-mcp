"""
Logs Optimization Analyzer for CloudWatch Optimization

Implements CloudWatch Logs cost optimization analysis using Cost Explorer integration
with cost control flags and free CloudWatch Logs APIs as primary source.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage, get_cost_forecast
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class LogsOptimizationAnalyzer(BaseAnalyzer):
    """
    Logs optimization analyzer for CloudWatch Logs cost analysis.
    
    This analyzer provides:
    - Log ingestion pattern analysis using Cost Explorer (allow_cost_explorer)
    - Log group configuration analysis using free CloudWatch Logs APIs as primary source
    - Retention policy analysis and optimization recommendations
    - Unused log groups and historical Logs Insights usage patterns identification
    """
    
    def __init__(self, cost_explorer_service=None, config_service=None, 
                 metrics_service=None, cloudwatch_service=None, pricing_service=None,
                 performance_monitor=None, memory_manager=None):
        """Initialize LogsOptimizationAnalyzer with CloudWatch services."""
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
        self.analysis_type = "logs_optimization"
        self.version = "1.0.0"
        
        # Cost control flags
        self.cost_preferences = None
        
        # Log retention policy recommendations (in days)
        self.retention_recommendations = {
            'development': 7,
            'staging': 30,
            'production_application': 90,
            'production_system': 365,
            'compliance': 2557  # 7 years
        }
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive CloudWatch Logs optimization analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region
                - lookback_days: Number of days to analyze (default: 30)
                - allow_cost_explorer: Enable Cost Explorer analysis (default: False)
                - allow_minimal_cost_metrics: Enable minimal cost metrics (default: False)
                - log_group_names: Specific log groups to analyze
                - log_group_prefix: Prefix filter for log groups
                
        Returns:
            Dictionary containing comprehensive logs optimization analysis results
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
        
        log_cloudwatch_operation(self.logger, "logs_optimization_analysis_start",
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
                'primary_data_source': 'cloudwatch_logs_config',
                'fallback_used': False,
                'data': {},
                'recommendations': []
            }
            
            # Execute analysis components in parallel
            analysis_tasks = []
            
            # 1. Log Groups Configuration Analysis (FREE - Always enabled)
            analysis_tasks.append(self._analyze_log_groups_configuration(**kwargs))
            
            # 2. Cost Explorer Logs Analysis (PAID - User controlled)
            if self.cost_preferences['allow_cost_explorer']:
                analysis_tasks.append(self._analyze_cost_explorer_logs(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('cost_explorer_logs_analysis')
                analysis_result['primary_data_source'] = 'cost_explorer'
            
            # 3. Minimal Cost Metrics Analysis for Logs (PAID - User controlled)
            if self.cost_preferences['allow_minimal_cost_metrics']:
                analysis_tasks.append(self._analyze_log_ingestion_metrics(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('minimal_cost_logs_metrics')
            
            # Execute all analysis tasks
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Logs analysis task {i} failed: {str(result)}")
                    analysis_result['fallback_used'] = True
                elif isinstance(result, dict):
                    if result.get('status') == 'success':
                        # Merge successful results
                        analysis_result['data'].update(result.get('data', {}))
                    elif result.get('status') == 'error':
                        # Mark fallback used for error results
                        self.logger.warning(f"Logs analysis task {i} returned error: {result.get('error_message', 'Unknown error')}")
                        analysis_result['fallback_used'] = True
            
            # Generate logs optimization analysis
            optimization_analysis = await self._generate_logs_optimization_analysis(analysis_result['data'], **kwargs)
            analysis_result['data']['optimization_analysis'] = optimization_analysis
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            analysis_result['execution_time'] = execution_time
            
            log_cloudwatch_operation(self.logger, "logs_optimization_analysis_complete",
                                   execution_time=execution_time,
                                   cost_incurred=analysis_result['cost_incurred'],
                                   primary_data_source=analysis_result['primary_data_source'])
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Logs optimization analysis failed: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_log_groups_configuration(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Logs configuration using free APIs.
        
        This provides the foundation for logs analysis using only free operations.
        """
        log_cloudwatch_operation(self.logger, "log_groups_config_analysis_start", component="log_groups_configuration")
        
        try:
            config_data = {}
            
            # Get log groups configuration (FREE)
            if self.cloudwatch_service:
                log_group_names = kwargs.get('log_group_names')
                log_group_prefix = kwargs.get('log_group_prefix')
                
                log_groups_result = await self.cloudwatch_service.describe_log_groups(
                    log_group_names=log_group_names,
                    log_group_name_prefix=log_group_prefix
                )
                
                if log_groups_result.success:
                    log_groups_data = log_groups_result.data
                    config_data['log_groups'] = log_groups_data
                    
                    # Analyze log groups configuration
                    log_groups_analysis = self._analyze_log_groups_metadata(log_groups_data.get('log_groups', []))
                    config_data['log_groups_analysis'] = log_groups_analysis
                    
                    log_cloudwatch_operation(self.logger, "log_groups_config_analyzed",
                                           total_log_groups=log_groups_data.get('total_count', 0),
                                           without_retention=len(log_groups_analysis.get('without_retention_policy', [])),
                                           with_retention=len(log_groups_analysis.get('with_retention_policy', [])))
            
            return {
                'status': 'success',
                'data': {
                    'log_groups_configuration_analysis': config_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Log groups configuration analysis failed: {error_message}")
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
    
    async def _analyze_cost_explorer_logs(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Logs costs using Cost Explorer (PAID operation).
        
        Requires allow_cost_explorer=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "cost_explorer_logs_analysis_start", 
                               component="cost_explorer_logs")
        
        try:
            lookback_days = kwargs.get('lookback_days', 30)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            cost_data = {}
            
            # Get CloudWatch Logs service costs with daily granularity
            logs_filter = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon CloudWatch Logs']
                }
            }
            
            # Get detailed logs costs by usage type
            cost_result = get_cost_and_usage(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                granularity='DAILY',
                metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                group_by=[{'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}],
                filter_expr=logs_filter,
                region=kwargs.get('region')
            )
            
            if cost_result['status'] == 'success':
                cost_data['logs_costs'] = self._process_logs_cost_explorer_response(
                    cost_result['data'], 'logs'
                )
                log_cloudwatch_operation(self.logger, "logs_costs_retrieved",
                                       total_cost=cost_data['logs_costs'].get('total_cost', 0))
            
            # Get logs cost forecast for next 30 days
            forecast_end = end_date + timedelta(days=30)
            forecast_result = get_cost_forecast(
                start_date=end_date.strftime('%Y-%m-%d'),
                end_date=forecast_end.strftime('%Y-%m-%d'),
                granularity='MONTHLY',
                metric='BLENDED_COST',
                filter_expr=logs_filter,
                region=kwargs.get('region')
            )
            
            if forecast_result['status'] == 'success':
                cost_data['logs_cost_forecast'] = self._process_forecast_response(
                    forecast_result['data']
                )
                log_cloudwatch_operation(self.logger, "logs_cost_forecast_retrieved",
                                       forecasted_cost=cost_data['logs_cost_forecast'].get('forecasted_amount', 0))
            
            return {
                'status': 'success',
                'data': {
                    'cost_explorer_logs_analysis': cost_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Cost Explorer logs analysis failed: {error_message}")
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
    
    async def _analyze_log_ingestion_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze log ingestion patterns using minimal cost metrics (PAID operation).
        
        Requires allow_minimal_cost_metrics=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "log_ingestion_metrics_start",
                               component="log_ingestion_metrics")
        
        try:
            ingestion_data = {}
            lookback_days = kwargs.get('lookback_days', 30)
            
            # Get log group incoming bytes metrics (MINIMAL COST)
            if self.cloudwatch_service:
                log_ingestion_result = await self.cloudwatch_service.get_log_group_incoming_bytes(
                    lookback_days=lookback_days
                )
                
                if log_ingestion_result.success:
                    ingestion_metrics = log_ingestion_result.data
                    ingestion_data['log_ingestion_metrics'] = ingestion_metrics
                    
                    # Analyze ingestion patterns
                    ingestion_analysis = self._analyze_ingestion_patterns(ingestion_metrics)
                    ingestion_data['ingestion_analysis'] = ingestion_analysis
                    
                    log_cloudwatch_operation(self.logger, "log_ingestion_metrics_analyzed",
                                           total_log_groups=ingestion_metrics.get('total_log_groups', 0),
                                           total_incoming_bytes=ingestion_metrics.get('total_incoming_bytes', 0))
            
            return {
                'status': 'success',
                'data': {
                    'log_ingestion_metrics_analysis': ingestion_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Log ingestion metrics analysis failed: {error_message}")
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
    
    def _analyze_log_groups_metadata(self, log_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze log groups metadata for optimization opportunities."""
        analysis = {
            'total_log_groups': len(log_groups),
            'without_retention_policy': [],
            'with_retention_policy': [],
            'retention_policy_distribution': {},
            'large_log_groups': [],
            'unused_log_groups': [],
            'optimization_opportunities': []
        }
        
        current_time = datetime.now(timezone.utc)
        
        for log_group in log_groups:
            log_group_name = log_group.get('logGroupName', '')
            retention_in_days = log_group.get('retentionInDays')
            stored_bytes = log_group.get('storedBytes', 0)
            creation_time = log_group.get('creationTime')
            
            # Analyze retention policy
            if retention_in_days is None:
                analysis['without_retention_policy'].append({
                    'log_group_name': log_group_name,
                    'stored_bytes': stored_bytes,
                    'creation_time': creation_time
                })
            else:
                analysis['with_retention_policy'].append({
                    'log_group_name': log_group_name,
                    'retention_days': retention_in_days,
                    'stored_bytes': stored_bytes
                })
                
                # Track retention policy distribution
                retention_key = f"{retention_in_days}_days"
                analysis['retention_policy_distribution'][retention_key] = \
                    analysis['retention_policy_distribution'].get(retention_key, 0) + 1
            
            # Identify large log groups (>1GB)
            if stored_bytes > 1024**3:  # 1GB
                analysis['large_log_groups'].append({
                    'log_group_name': log_group_name,
                    'stored_bytes': stored_bytes,
                    'stored_gb': stored_bytes / (1024**3),
                    'retention_days': retention_in_days
                })
            
            # Identify potentially unused log groups (no recent activity)
            if creation_time:
                try:
                    creation_datetime = datetime.fromtimestamp(creation_time / 1000, tz=timezone.utc)
                    days_since_creation = (current_time - creation_datetime).days
                    
                    # If log group is old but has very little data, it might be unused
                    if days_since_creation > 30 and stored_bytes < 1024**2:  # 1MB
                        analysis['unused_log_groups'].append({
                            'log_group_name': log_group_name,
                            'stored_bytes': stored_bytes,
                            'days_since_creation': days_since_creation,
                            'retention_days': retention_in_days
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing creation time for {log_group_name}: {str(e)}")
        
        # Generate optimization opportunities
        self._identify_log_group_optimization_opportunities(analysis)
        
        return analysis
    
    def _identify_log_group_optimization_opportunities(self, analysis: Dict[str, Any]):
        """Identify specific optimization opportunities for log groups."""
        opportunities = []
        
        # Opportunity 1: Log groups without retention policy
        without_retention = analysis['without_retention_policy']
        if without_retention:
            total_stored_bytes = sum(lg.get('stored_bytes', 0) for lg in without_retention)
            opportunities.append({
                'type': 'retention_policy_missing',
                'priority': 'high',
                'description': f'{len(without_retention)} log groups without retention policy',
                'affected_log_groups': len(without_retention),
                'total_stored_gb': total_stored_bytes / (1024**3),
                'potential_action': 'Set appropriate retention policies',
                'estimated_storage_reduction': '50-90%'
            })
        
        # Opportunity 2: Large log groups with long retention
        large_log_groups = analysis['large_log_groups']
        long_retention_large_groups = [
            lg for lg in large_log_groups 
            if lg.get('retention_days') is None or lg.get('retention_days', 0) > 365
        ]
        
        if long_retention_large_groups:
            total_stored_gb = sum(lg.get('stored_gb', 0) for lg in long_retention_large_groups)
            opportunities.append({
                'type': 'large_log_groups_long_retention',
                'priority': 'high',
                'description': f'{len(long_retention_large_groups)} large log groups with long/no retention',
                'affected_log_groups': len(long_retention_large_groups),
                'total_stored_gb': total_stored_gb,
                'potential_action': 'Review and optimize retention policies',
                'estimated_storage_reduction': '30-70%'
            })
        
        # Opportunity 3: Unused log groups
        unused_log_groups = analysis['unused_log_groups']
        if unused_log_groups:
            opportunities.append({
                'type': 'unused_log_groups',
                'priority': 'medium',
                'description': f'{len(unused_log_groups)} potentially unused log groups',
                'affected_log_groups': len(unused_log_groups),
                'potential_action': 'Review and delete unused log groups',
                'estimated_cost_reduction': '100% for deleted groups'
            })
        
        # Opportunity 4: Retention policy standardization
        retention_distribution = analysis['retention_policy_distribution']
        if len(retention_distribution) > 5:  # Many different retention policies
            opportunities.append({
                'type': 'retention_policy_standardization',
                'priority': 'low',
                'description': f'{len(retention_distribution)} different retention policies in use',
                'potential_action': 'Standardize retention policies by environment/purpose',
                'estimated_management_improvement': 'Improved governance and cost predictability'
            })
        
        analysis['optimization_opportunities'] = opportunities
    
    def _analyze_ingestion_patterns(self, ingestion_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log ingestion patterns for optimization opportunities."""
        analysis = {
            'total_ingestion_gb': 0.0,
            'high_volume_log_groups': [],
            'ingestion_trends': {},
            'optimization_opportunities': []
        }
        
        try:
            total_incoming_bytes = ingestion_metrics.get('total_incoming_bytes', 0)
            analysis['total_ingestion_gb'] = total_incoming_bytes / (1024**3)
            
            log_group_metrics = ingestion_metrics.get('log_group_metrics', [])
            
            # Identify high-volume log groups
            for lg_metric in log_group_metrics:
                log_group_name = lg_metric.get('log_group_name', '')
                incoming_bytes = lg_metric.get('incoming_bytes', 0)
                incoming_gb = incoming_bytes / (1024**3)
                
                # Consider high volume if >100MB per day on average
                if incoming_gb > 0.1:
                    analysis['high_volume_log_groups'].append({
                        'log_group_name': log_group_name,
                        'daily_ingestion_gb': incoming_gb,
                        'monthly_estimated_gb': incoming_gb * 30,
                        'optimization_potential': 'high' if incoming_gb > 1.0 else 'medium'
                    })
            
            # Sort by ingestion volume
            analysis['high_volume_log_groups'].sort(
                key=lambda x: x['daily_ingestion_gb'], reverse=True
            )
            
            # Generate ingestion-based optimization opportunities
            self._identify_ingestion_optimization_opportunities(analysis)
            
        except Exception as e:
            self.logger.error(f"Error analyzing ingestion patterns: {str(e)}")
        
        return analysis
    
    def _identify_ingestion_optimization_opportunities(self, analysis: Dict[str, Any]):
        """Identify optimization opportunities based on ingestion patterns."""
        opportunities = []
        
        high_volume_groups = analysis['high_volume_log_groups']
        
        if high_volume_groups:
            # Top ingestion log groups
            top_groups = high_volume_groups[:5]  # Top 5
            total_daily_gb = sum(lg['daily_ingestion_gb'] for lg in top_groups)
            
            opportunities.append({
                'type': 'high_ingestion_volume',
                'priority': 'high',
                'description': f'Top {len(top_groups)} log groups account for {total_daily_gb:.2f} GB/day',
                'affected_log_groups': [lg['log_group_name'] for lg in top_groups],
                'potential_actions': [
                    'Review log verbosity settings',
                    'Implement log sampling for high-volume applications',
                    'Consider log aggregation and filtering',
                    'Evaluate if all logged data is necessary'
                ],
                'estimated_cost_reduction': '20-50% through log optimization'
            })
        
        # Very high volume groups (>1GB/day)
        very_high_volume = [lg for lg in high_volume_groups if lg['daily_ingestion_gb'] > 1.0]
        if very_high_volume:
            opportunities.append({
                'type': 'very_high_ingestion_volume',
                'priority': 'critical',
                'description': f'{len(very_high_volume)} log groups with >1GB/day ingestion',
                'affected_log_groups': [lg['log_group_name'] for lg in very_high_volume],
                'potential_actions': [
                    'Immediate review of application logging levels',
                    'Implement log rotation and compression',
                    'Consider moving to S3 for long-term storage',
                    'Evaluate log streaming to reduce CloudWatch costs'
                ],
                'estimated_cost_reduction': '30-70% through aggressive optimization'
            })
        
        analysis['optimization_opportunities'] = opportunities
    
    def _process_logs_cost_explorer_response(self, response_data: Dict[str, Any], 
                                           service_type: str) -> Dict[str, Any]:
        """Process Cost Explorer API response for logs-specific cost data."""
        processed_data = {
            'service_type': service_type,
            'total_cost': 0.0,
            'daily_costs': [],
            'usage_types': {},
            'logs_specific_costs': {},
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
                    
                    # Categorize logs-specific usage types
                    self._categorize_logs_usage_type(usage_type, blended_cost, usage_quantity, processed_data)
                    
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
            self.logger.error(f"Error processing logs Cost Explorer response: {str(e)}")
        
        return processed_data
    
    def _categorize_logs_usage_type(self, usage_type: str, cost: float, usage: float, processed_data: Dict[str, Any]):
        """Categorize logs usage types into specific cost categories."""
        usage_type_lower = usage_type.lower()
        
        if 'ingest' in usage_type_lower or 'ingestion' in usage_type_lower:
            processed_data['logs_specific_costs']['ingestion'] = \
                processed_data['logs_specific_costs'].get('ingestion', 0.0) + cost
        elif 'storage' in usage_type_lower or 'stored' in usage_type_lower:
            processed_data['logs_specific_costs']['storage'] = \
                processed_data['logs_specific_costs'].get('storage', 0.0) + cost
        elif 'insights' in usage_type_lower or 'query' in usage_type_lower:
            processed_data['logs_specific_costs']['insights'] = \
                processed_data['logs_specific_costs'].get('insights', 0.0) + cost
        elif 'delivery' in usage_type_lower or 'export' in usage_type_lower:
            processed_data['logs_specific_costs']['delivery'] = \
                processed_data['logs_specific_costs'].get('delivery', 0.0) + cost
        else:
            processed_data['logs_specific_costs']['other'] = \
                processed_data['logs_specific_costs'].get('other', 0.0) + cost
    
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
            self.logger.error(f"Error processing logs forecast response: {str(e)}")
        
        return processed_data
    
    async def _generate_logs_optimization_analysis(self, analysis_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate comprehensive logs optimization analysis."""
        optimization_analysis = {
            'retention_policy_optimization': {},
            'ingestion_optimization': {},
            'storage_optimization': {},
            'cost_optimization_summary': {},
            'recommendations': []
        }
        
        try:
            # Analyze retention policy optimization
            config_data = analysis_data.get('log_groups_configuration_analysis', {})
            if config_data:
                retention_optimization = self._analyze_retention_policy_optimization(config_data)
                optimization_analysis['retention_policy_optimization'] = retention_optimization
            
            # Analyze ingestion optimization
            ingestion_data = analysis_data.get('log_ingestion_metrics_analysis', {})
            if ingestion_data:
                ingestion_optimization = self._analyze_ingestion_optimization(ingestion_data)
                optimization_analysis['ingestion_optimization'] = ingestion_optimization
            
            # Analyze storage optimization
            storage_optimization = self._analyze_storage_optimization(config_data, ingestion_data)
            optimization_analysis['storage_optimization'] = storage_optimization
            
            # Generate cost optimization summary
            cost_data = analysis_data.get('cost_explorer_logs_analysis', {})
            cost_summary = self._generate_cost_optimization_summary(
                config_data, ingestion_data, cost_data
            )
            optimization_analysis['cost_optimization_summary'] = cost_summary
            
            # Compile all recommendations
            all_recommendations = self._compile_logs_recommendations(optimization_analysis)
            optimization_analysis['recommendations'] = all_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating logs optimization analysis: {str(e)}")
        
        return optimization_analysis
    
    def _analyze_retention_policy_optimization(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze retention policy optimization opportunities."""
        log_groups_analysis = config_data.get('log_groups_analysis', {})
        
        optimization = {
            'without_retention_count': len(log_groups_analysis.get('without_retention_policy', [])),
            'with_retention_count': len(log_groups_analysis.get('with_retention_policy', [])),
            'retention_recommendations': [],
            'potential_storage_savings': 0.0
        }
        
        # Analyze log groups without retention
        without_retention = log_groups_analysis.get('without_retention_policy', [])
        for log_group in without_retention:
            log_group_name = log_group.get('log_group_name', '')
            stored_bytes = log_group.get('stored_bytes', 0)
            
            # Recommend retention based on log group name patterns
            recommended_retention = self._recommend_retention_policy(log_group_name)
            
            optimization['retention_recommendations'].append({
                'log_group_name': log_group_name,
                'current_retention': 'Never expire',
                'recommended_retention': recommended_retention,
                'stored_gb': stored_bytes / (1024**3),
                'estimated_savings_percentage': self._estimate_retention_savings(recommended_retention)
            })
            
            # Estimate potential savings
            savings_percentage = self._estimate_retention_savings(recommended_retention) / 100
            optimization['potential_storage_savings'] += (stored_bytes / (1024**3)) * savings_percentage
        
        return optimization
    
    def _recommend_retention_policy(self, log_group_name: str) -> int:
        """Recommend retention policy based on log group name patterns."""
        log_group_lower = log_group_name.lower()
        
        # Development/test environments
        if any(env in log_group_lower for env in ['dev', 'test', 'sandbox', 'staging']):
            return self.retention_recommendations['development']
        
        # Application logs
        elif any(app in log_group_lower for app in ['app', 'application', 'service', 'api']):
            return self.retention_recommendations['production_application']
        
        # System/infrastructure logs
        elif any(sys in log_group_lower for sys in ['system', 'infra', 'aws', 'lambda', 'ecs']):
            return self.retention_recommendations['production_system']
        
        # Compliance/audit logs
        elif any(comp in log_group_lower for comp in ['audit', 'compliance', 'security']):
            return self.retention_recommendations['compliance']
        
        # Default for production
        else:
            return self.retention_recommendations['production_application']
    
    def _estimate_retention_savings(self, recommended_retention_days: int) -> float:
        """Estimate storage savings percentage based on retention policy."""
        # Assume current data is spread over 2 years (730 days) on average
        current_assumed_days = 730
        
        if recommended_retention_days >= current_assumed_days:
            return 0.0  # No savings
        
        savings_percentage = ((current_assumed_days - recommended_retention_days) / current_assumed_days) * 100
        return min(90.0, max(0.0, savings_percentage))  # Cap at 90% savings
    
    def _analyze_ingestion_optimization(self, ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ingestion optimization opportunities."""
        ingestion_analysis = ingestion_data.get('ingestion_analysis', {})
        
        optimization = {
            'total_daily_ingestion_gb': ingestion_analysis.get('total_ingestion_gb', 0.0),
            'high_volume_log_groups': ingestion_analysis.get('high_volume_log_groups', []),
            'optimization_opportunities': ingestion_analysis.get('optimization_opportunities', []),
            'estimated_monthly_ingestion_cost': 0.0
        }
        
        # Calculate estimated monthly ingestion cost
        if self.pricing_service:
            try:
                logs_pricing = self.pricing_service.get_logs_pricing()
                if logs_pricing.get('status') == 'success':
                    pricing = logs_pricing['logs_pricing']
                    daily_gb = optimization['total_daily_ingestion_gb']
                    monthly_gb = daily_gb * 30
                    optimization['estimated_monthly_ingestion_cost'] = \
                        monthly_gb * pricing.get('ingestion_per_gb', 0.50)
            except Exception as e:
                self.logger.warning(f"Could not calculate ingestion cost: {str(e)}")
        
        return optimization
    
    def _analyze_storage_optimization(self, config_data: Dict[str, Any], 
                                    ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze storage optimization opportunities."""
        log_groups_analysis = config_data.get('log_groups_analysis', {}) if config_data else {}
        
        optimization = {
            'large_log_groups': log_groups_analysis.get('large_log_groups', []),
            'unused_log_groups': log_groups_analysis.get('unused_log_groups', []),
            'total_storage_gb': 0.0,
            'estimated_monthly_storage_cost': 0.0,
            'optimization_potential': []
        }
        
        # Calculate total storage
        all_log_groups = (log_groups_analysis.get('without_retention_policy', []) + 
                         log_groups_analysis.get('with_retention_policy', []))
        
        total_storage_bytes = sum(lg.get('stored_bytes', 0) for lg in all_log_groups)
        optimization['total_storage_gb'] = total_storage_bytes / (1024**3)
        
        # Calculate estimated monthly storage cost
        if self.pricing_service:
            try:
                logs_pricing = self.pricing_service.get_logs_pricing()
                if logs_pricing.get('status') == 'success':
                    pricing = logs_pricing['logs_pricing']
                    optimization['estimated_monthly_storage_cost'] = \
                        optimization['total_storage_gb'] * pricing.get('storage_per_gb_month', 0.03)
            except Exception as e:
                self.logger.warning(f"Could not calculate storage cost: {str(e)}")
        
        # Identify optimization potential
        large_groups = optimization['large_log_groups']
        if large_groups:
            total_large_gb = sum(lg.get('stored_gb', 0) for lg in large_groups)
            optimization['optimization_potential'].append({
                'type': 'large_log_groups',
                'count': len(large_groups),
                'total_gb': total_large_gb,
                'potential_savings': '30-70% through retention optimization'
            })
        
        unused_groups = optimization['unused_log_groups']
        if unused_groups:
            total_unused_gb = sum(lg.get('stored_bytes', 0) / (1024**3) for lg in unused_groups)
            optimization['optimization_potential'].append({
                'type': 'unused_log_groups',
                'count': len(unused_groups),
                'total_gb': total_unused_gb,
                'potential_savings': '100% through deletion'
            })
        
        return optimization
    
    def _generate_cost_optimization_summary(self, config_data: Dict[str, Any], 
                                          ingestion_data: Dict[str, Any],
                                          cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cost optimization summary."""
        summary = {
            'current_estimated_monthly_cost': 0.0,
            'potential_monthly_savings': 0.0,
            'optimization_breakdown': {},
            'top_cost_drivers': [],
            'quick_wins': []
        }
        
        try:
            # Use Cost Explorer data if available
            if cost_data and cost_data.get('logs_costs'):
                logs_costs = cost_data['logs_costs']
                summary['current_estimated_monthly_cost'] = logs_costs.get('total_cost', 0.0) * 30  # Daily to monthly
                
                # Analyze cost breakdown
                logs_specific_costs = logs_costs.get('logs_specific_costs', {})
                summary['optimization_breakdown'] = {
                    'ingestion_cost': logs_specific_costs.get('ingestion', 0.0) * 30,
                    'storage_cost': logs_specific_costs.get('storage', 0.0) * 30,
                    'insights_cost': logs_specific_costs.get('insights', 0.0) * 30,
                    'other_cost': logs_specific_costs.get('other', 0.0) * 30
                }
            
            # Identify top cost drivers
            if config_data:
                log_groups_analysis = config_data.get('log_groups_analysis', {})
                large_groups = log_groups_analysis.get('large_log_groups', [])
                
                for lg in large_groups[:5]:  # Top 5
                    summary['top_cost_drivers'].append({
                        'log_group_name': lg.get('log_group_name'),
                        'stored_gb': lg.get('stored_gb', 0),
                        'retention_days': lg.get('retention_days'),
                        'optimization_priority': 'high' if lg.get('stored_gb', 0) > 10 else 'medium'
                    })
            
            # Identify quick wins
            if config_data:
                log_groups_analysis = config_data.get('log_groups_analysis', {})
                without_retention = log_groups_analysis.get('without_retention_policy', [])
                
                if without_retention:
                    summary['quick_wins'].append({
                        'action': 'Set retention policies',
                        'affected_log_groups': len(without_retention),
                        'estimated_savings': '50-90% storage cost reduction',
                        'implementation_effort': 'low'
                    })
                
                unused_groups = log_groups_analysis.get('unused_log_groups', [])
                if unused_groups:
                    summary['quick_wins'].append({
                        'action': 'Delete unused log groups',
                        'affected_log_groups': len(unused_groups),
                        'estimated_savings': '100% cost elimination',
                        'implementation_effort': 'low'
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating cost optimization summary: {str(e)}")
        
        return summary
    
    def _compile_logs_recommendations(self, optimization_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile all logs optimization recommendations."""
        recommendations = []
        
        try:
            # Retention policy recommendations
            retention_opt = optimization_analysis.get('retention_policy_optimization', {})
            if retention_opt.get('without_retention_count', 0) > 0:
                recommendations.append(self.create_recommendation(
                    rec_type='retention_policy_optimization',
                    priority='high',
                    title='Set Retention Policies for Log Groups',
                    description=f'{retention_opt["without_retention_count"]} log groups have no retention policy, storing logs indefinitely',
                    potential_savings=retention_opt.get('potential_storage_savings', 0) * 0.03 * 12,  # Estimated annual savings
                    implementation_effort='low',
                    affected_resources=[rec['log_group_name'] for rec in retention_opt.get('retention_recommendations', [])[:10]],
                    action_items=[
                        'Review log group purposes and set appropriate retention policies',
                        'Use 7 days for development, 30-90 days for production applications',
                        'Consider compliance requirements for audit logs',
                        'Monitor storage costs after implementing retention policies'
                    ],
                    cloudwatch_component='logs'
                ))
            
            # Ingestion optimization recommendations
            ingestion_opt = optimization_analysis.get('ingestion_optimization', {})
            high_volume_groups = ingestion_opt.get('high_volume_log_groups', [])
            if high_volume_groups:
                top_groups = high_volume_groups[:3]
                total_daily_gb = sum(lg.get('daily_ingestion_gb', 0) for lg in top_groups)
                
                recommendations.append(self.create_recommendation(
                    rec_type='ingestion_volume_optimization',
                    priority='high' if total_daily_gb > 5 else 'medium',
                    title='Optimize High-Volume Log Ingestion',
                    description=f'Top {len(top_groups)} log groups generate {total_daily_gb:.2f} GB/day of logs',
                    potential_savings=total_daily_gb * 30 * 0.50 * 0.3,  # 30% reduction potential
                    implementation_effort='medium',
                    affected_resources=[lg['log_group_name'] for lg in top_groups],
                    action_items=[
                        'Review application logging levels and reduce verbosity',
                        'Implement log sampling for high-volume applications',
                        'Consider structured logging to reduce log size',
                        'Evaluate if all logged data provides value'
                    ],
                    cloudwatch_component='logs'
                ))
            
            # Storage optimization recommendations
            storage_opt = optimization_analysis.get('storage_optimization', {})
            large_groups = storage_opt.get('large_log_groups', [])
            if large_groups:
                total_large_gb = sum(lg.get('stored_gb', 0) for lg in large_groups)
                
                recommendations.append(self.create_recommendation(
                    rec_type='storage_optimization',
                    priority='medium',
                    title='Optimize Large Log Group Storage',
                    description=f'{len(large_groups)} log groups store {total_large_gb:.2f} GB of data',
                    potential_savings=total_large_gb * 0.03 * 0.5 * 12,  # 50% reduction potential annually
                    implementation_effort='medium',
                    affected_resources=[lg['log_group_name'] for lg in large_groups[:10]],
                    action_items=[
                        'Review retention policies for large log groups',
                        'Consider archiving old logs to S3 for long-term storage',
                        'Implement log compression where possible',
                        'Evaluate log group consolidation opportunities'
                    ],
                    cloudwatch_component='logs'
                ))
            
            # Unused log groups recommendations
            unused_groups = storage_opt.get('unused_log_groups', [])
            if unused_groups:
                recommendations.append(self.create_recommendation(
                    rec_type='unused_log_groups_cleanup',
                    priority='low',
                    title='Clean Up Unused Log Groups',
                    description=f'{len(unused_groups)} log groups appear to be unused or have minimal activity',
                    potential_savings=sum(lg.get('stored_bytes', 0) for lg in unused_groups) / (1024**3) * 0.03 * 12,
                    implementation_effort='low',
                    affected_resources=[lg['log_group_name'] for lg in unused_groups[:10]],
                    action_items=[
                        'Verify log groups are truly unused before deletion',
                        'Check with application teams before removing log groups',
                        'Consider setting short retention periods instead of deletion',
                        'Document cleanup decisions for future reference'
                    ],
                    cloudwatch_component='logs'
                ))
            
        except Exception as e:
            self.logger.error(f"Error compiling logs recommendations: {str(e)}")
        
        return recommendations
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from logs optimization analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Get recommendations from optimization analysis
            optimization_analysis = analysis_results.get('data', {}).get('optimization_analysis', {})
            recommendations = optimization_analysis.get('recommendations', [])
            
            # Add general logs optimization recommendations if no specific ones exist
            if not recommendations:
                recommendations = self._generate_fallback_recommendations(analysis_results)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating logs recommendations: {str(e)}")
            return []
    
    def _generate_fallback_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback recommendations when detailed analysis is not available."""
        fallback_recommendations = []
        
        # General logs optimization recommendation
        fallback_recommendations.append(self.create_recommendation(
            rec_type='general_logs_optimization',
            priority='medium',
            title='Review CloudWatch Logs Configuration',
            description='Comprehensive logs optimization analysis was not available, but general optimization is recommended',
            implementation_effort='medium',
            action_items=[
                'Review all log groups and set appropriate retention policies',
                'Identify and optimize high-volume log ingestion',
                'Clean up unused or unnecessary log groups',
                'Consider log aggregation and filtering strategies',
                'Monitor logs costs regularly and set up cost alerts'
            ],
            cloudwatch_component='logs'
        ))
        
        return fallback_recommendations