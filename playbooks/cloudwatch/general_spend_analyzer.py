"""
General Spend Analyzer for CloudWatch Optimization

Implements comprehensive CloudWatch spend analysis using Cost Explorer integration
with cost control flags and CloudWatch APIs as primary free data source.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage, get_cost_forecast
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class GeneralSpendAnalyzer(BaseAnalyzer):
    """
    General spend analyzer for comprehensive CloudWatch cost analysis.
    
    This analyzer provides:
    - Cost Explorer integration with cost control flags (allow_cost_explorer)
    - CloudWatch APIs as primary free data source for configuration analysis
    - Logs costs, metrics costs, alarms costs, and dashboard costs analysis
    - Actionable recommendations with cost-aware feature degradation
    """
    
    def __init__(self, cost_explorer_service=None, config_service=None, 
                 metrics_service=None, cloudwatch_service=None, pricing_service=None,
                 performance_monitor=None, memory_manager=None):
        """Initialize GeneralSpendAnalyzer with CloudWatch services."""
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
        self.analysis_type = "general_spend"
        self.version = "1.0.0"
        
        # Cost control flags
        self.cost_preferences = None
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive CloudWatch spend analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region
                - lookback_days: Number of days to analyze (default: 30)
                - allow_cost_explorer: Enable Cost Explorer analysis (default: False)
                - allow_minimal_cost_metrics: Enable minimal cost metrics (default: False)
                - log_group_names: Specific log groups to analyze
                - alarm_names: Specific alarms to analyze
                - dashboard_names: Specific dashboards to analyze
                
        Returns:
            Dictionary containing comprehensive spend analysis results
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
        
        log_cloudwatch_operation(self.logger, "general_spend_analysis_start",
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
            
            # 1. Configuration Analysis (FREE - Always enabled)
            analysis_tasks.append(self._analyze_cloudwatch_configuration(**kwargs))
            
            # 2. Cost Explorer Analysis (PAID - User controlled)
            if self.cost_preferences['allow_cost_explorer']:
                analysis_tasks.append(self._analyze_cost_explorer_data(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('cost_explorer_analysis')
                analysis_result['primary_data_source'] = 'cost_explorer'
            
            # 3. Minimal Cost Metrics Analysis (PAID - User controlled)
            if self.cost_preferences['allow_minimal_cost_metrics']:
                analysis_tasks.append(self._analyze_minimal_cost_metrics(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('minimal_cost_metrics')
            
            # Execute all analysis tasks
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results and collect errors
            errors = []
            successful_tasks = 0
            
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    error_msg = f"Analysis task {i} failed with exception: {str(result)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    analysis_result['fallback_used'] = True
                elif isinstance(result, dict):
                    if result.get('status') == 'success':
                        # Merge successful results
                        analysis_result['data'].update(result.get('data', {}))
                        successful_tasks += 1
                    elif result.get('status') == 'partial':
                        # Merge partial results but track warnings
                        analysis_result['data'].update(result.get('data', {}))
                        if 'warnings' in result:
                            errors.extend(result['warnings'])
                        successful_tasks += 1
                        analysis_result['fallback_used'] = True
                    elif result.get('status') == 'error':
                        # Collect error details
                        error_msg = f"Analysis task {i} returned error: {result.get('error_message', 'Unknown error')}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        analysis_result['fallback_used'] = True
            
            # If all tasks failed, return error
            if successful_tasks == 0:
                import traceback
                error_summary = "; ".join(errors) if errors else "All analysis tasks failed"
                self.logger.error(f"General spend analysis failed: {error_summary}")
                return {
                    'status': 'error',
                    'error_message': f"All analysis tasks failed: {error_summary}",
                    'error_type': 'AllTasksFailed',
                    'errors': errors,
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'data': {}
                }
            
            # Generate cost breakdown analysis
            cost_breakdown = await self._generate_cost_breakdown(analysis_result['data'], **kwargs)
            analysis_result['data']['cost_breakdown'] = cost_breakdown
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            analysis_result['execution_time'] = execution_time
            
            # If some tasks failed, mark as partial success
            if errors:
                analysis_result['status'] = 'partial'
                analysis_result['warnings'] = errors
                self.logger.warning(f"General spend analysis completed with {len(errors)} warnings")
            
            log_cloudwatch_operation(self.logger, "general_spend_analysis_complete",
                                   execution_time=execution_time,
                                   status=analysis_result['status'],
                                   cost_incurred=analysis_result['cost_incurred'],
                                   primary_data_source=analysis_result['primary_data_source'])
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"General spend analysis failed: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_cloudwatch_configuration(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch configuration using free APIs.
        
        This provides the foundation for cost analysis using only free operations.
        """
        log_cloudwatch_operation(self.logger, "config_analysis_start", component="configuration")
        
        try:
            config_data = {}
            errors = []
            
            # Check if cloudwatch_service is available
            if not self.cloudwatch_service:
                error_msg = "CloudWatch service not initialized - cannot fetch configuration data"
                self.logger.error(error_msg)
                return {
                    'status': 'error',
                    'error_message': error_msg,
                    'error_type': 'ServiceNotInitialized',
                    'data': {}
                }
            
            # Get log groups configuration (FREE)
            log_groups_result = await self.cloudwatch_service.describe_log_groups(
                log_group_names=kwargs.get('log_group_names')
            )
            if log_groups_result.success:
                config_data['log_groups'] = log_groups_result.data
                log_cloudwatch_operation(self.logger, "log_groups_analyzed",
                                       count=log_groups_result.data.get('total_count', 0))
            else:
                error_msg = f"Failed to fetch log groups: {log_groups_result.error}"
                self.logger.error(error_msg)
                errors.append(error_msg)
            
            # Get alarms configuration (FREE)
            alarms_result = await self.cloudwatch_service.describe_alarms(
                alarm_names=kwargs.get('alarm_names')
            )
            if alarms_result.success:
                config_data['alarms'] = alarms_result.data
                log_cloudwatch_operation(self.logger, "alarms_analyzed",
                                       count=alarms_result.data.get('total_count', 0))
            else:
                error_msg = f"Failed to fetch alarms: {alarms_result.error}"
                self.logger.error(error_msg)
                errors.append(error_msg)
            
            # Get dashboards configuration (FREE)
            dashboards_result = await self.cloudwatch_service.list_dashboards(
                dashboard_name_prefix=kwargs.get('dashboard_name_prefix')
            )
            if dashboards_result.success:
                config_data['dashboards'] = dashboards_result.data
                log_cloudwatch_operation(self.logger, "dashboards_analyzed",
                                       count=dashboards_result.data.get('total_count', 0))
            else:
                error_msg = f"Failed to fetch dashboards: {dashboards_result.error}"
                self.logger.error(error_msg)
                errors.append(error_msg)
            
            # Get metrics metadata (FREE)
            metrics_result = await self.cloudwatch_service.list_metrics()
            if metrics_result.success:
                config_data['metrics'] = metrics_result.data
                log_cloudwatch_operation(self.logger, "metrics_analyzed",
                                       count=metrics_result.data.get('total_count', 0))
            else:
                error_msg = f"Failed to fetch metrics: {metrics_result.error}"
                self.logger.error(error_msg)
                errors.append(error_msg)
            
            # If all operations failed, return error
            if not config_data:
                error_summary = "; ".join(errors) if errors else "All CloudWatch API calls failed with no data returned"
                self.logger.error(f"Configuration analysis failed: {error_summary}")
                return {
                    'status': 'error',
                    'error_message': f"Failed to fetch any CloudWatch configuration data: {error_summary}",
                    'error_type': 'DataFetchFailure',
                    'errors': errors,
                    'data': {}
                }
            
            # If some operations failed, return partial success
            if errors:
                self.logger.warning(f"Configuration analysis partially successful with {len(errors)} errors")
                return {
                    'status': 'partial',
                    'warnings': errors,
                    'data': {
                        'configuration_analysis': config_data
                    }
                }
            
            return {
                'status': 'success',
                'data': {
                    'configuration_analysis': config_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Configuration analysis failed: {error_message}")
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
    
    async def _analyze_cost_explorer_data(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch costs using Cost Explorer (PAID operation).
        
        Requires allow_cost_explorer=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "cost_explorer_analysis_start", 
                               component="cost_explorer")
        
        try:
            lookback_days = kwargs.get('lookback_days', 30)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            cost_data = {}
            
            # Get CloudWatch service costs
            cloudwatch_filter = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon CloudWatch']
                }
            }
            
            cost_result = get_cost_and_usage(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                granularity='DAILY',
                metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                group_by=[{'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}],
                filter_expr=cloudwatch_filter,
                region=kwargs.get('region')
            )
            
            if cost_result['status'] == 'success':
                cost_data['cloudwatch_costs'] = self._process_cost_explorer_response(
                    cost_result['data'], 'cloudwatch'
                )
                log_cloudwatch_operation(self.logger, "cloudwatch_costs_retrieved",
                                       total_cost=cost_data['cloudwatch_costs'].get('total_cost', 0))
            
            # Get CloudWatch Logs costs
            logs_filter = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon CloudWatch Logs']
                }
            }
            
            logs_cost_result = get_cost_and_usage(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                granularity='DAILY',
                metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                group_by=[{'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}],
                filter_expr=logs_filter,
                region=kwargs.get('region')
            )
            
            if logs_cost_result['status'] == 'success':
                cost_data['logs_costs'] = self._process_cost_explorer_response(
                    logs_cost_result['data'], 'logs'
                )
                log_cloudwatch_operation(self.logger, "logs_costs_retrieved",
                                       total_cost=cost_data['logs_costs'].get('total_cost', 0))
            
            # Get cost forecast for next 30 days
            forecast_end = end_date + timedelta(days=30)
            forecast_result = get_cost_forecast(
                start_date=end_date.strftime('%Y-%m-%d'),
                end_date=forecast_end.strftime('%Y-%m-%d'),
                granularity='MONTHLY',
                metric='BLENDED_COST',
                filter_expr=cloudwatch_filter,
                region=kwargs.get('region')
            )
            
            if forecast_result['status'] == 'success':
                cost_data['cost_forecast'] = self._process_forecast_response(
                    forecast_result['data']
                )
                log_cloudwatch_operation(self.logger, "cost_forecast_retrieved",
                                       forecasted_cost=cost_data['cost_forecast'].get('forecasted_amount', 0))
            
            return {
                'status': 'success',
                'data': {
                    'cost_explorer_analysis': cost_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Cost Explorer analysis failed: {error_message}")
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
    
    async def _analyze_minimal_cost_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch using minimal cost metrics (PAID operation).
        
        Requires allow_minimal_cost_metrics=True in cost preferences.
        """
        log_cloudwatch_operation(self.logger, "minimal_cost_metrics_start",
                               component="metrics")
        
        try:
            metrics_data = {}
            
            # Get log group incoming bytes metrics (MINIMAL COST)
            if self.cloudwatch_service:
                lookback_days = kwargs.get('lookback_days', 30)
                log_metrics_result = await self.cloudwatch_service.get_log_group_incoming_bytes(
                    lookback_days=lookback_days
                )
                
                if log_metrics_result.success:
                    metrics_data['log_ingestion_metrics'] = log_metrics_result.data
                    log_cloudwatch_operation(self.logger, "log_ingestion_metrics_retrieved",
                                           total_log_groups=log_metrics_result.data.get('total_log_groups', 0))
            
            return {
                'status': 'success',
                'data': {
                    'minimal_cost_metrics_analysis': metrics_data
                }
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Minimal cost metrics analysis failed: {error_message}")
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
    
    def _process_cost_explorer_response(self, response_data: Dict[str, Any], 
                                      service_type: str) -> Dict[str, Any]:
        """Process Cost Explorer API response into structured cost data."""
        processed_data = {
            'service_type': service_type,
            'total_cost': 0.0,
            'daily_costs': [],
            'usage_types': {},
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
            self.logger.error(f"Error processing Cost Explorer response: {str(e)}")
        
        return processed_data
    
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
            self.logger.error(f"Error processing forecast response: {str(e)}")
        
        return processed_data
    
    async def _generate_cost_breakdown(self, analysis_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate comprehensive cost breakdown analysis."""
        cost_breakdown = {
            'logs_costs': {'estimated_monthly': 0.0, 'breakdown': {}},
            'metrics_costs': {'estimated_monthly': 0.0, 'breakdown': {}},
            'alarms_costs': {'estimated_monthly': 0.0, 'breakdown': {}},
            'dashboards_costs': {'estimated_monthly': 0.0, 'breakdown': {}},
            'total_estimated_monthly': 0.0,
            'cost_optimization_opportunities': []
        }
        
        try:
            # Analyze logs costs
            if self.pricing_service:
                logs_analysis = await self._analyze_logs_costs(analysis_data)
                cost_breakdown['logs_costs'] = logs_analysis
                cost_breakdown['total_estimated_monthly'] += logs_analysis.get('estimated_monthly', 0.0)
            
            # Analyze metrics costs
            if self.pricing_service:
                metrics_analysis = await self._analyze_metrics_costs(analysis_data)
                cost_breakdown['metrics_costs'] = metrics_analysis
                cost_breakdown['total_estimated_monthly'] += metrics_analysis.get('estimated_monthly', 0.0)
            
            # Analyze alarms costs
            if self.pricing_service:
                alarms_analysis = await self._analyze_alarms_costs(analysis_data)
                cost_breakdown['alarms_costs'] = alarms_analysis
                cost_breakdown['total_estimated_monthly'] += alarms_analysis.get('estimated_monthly', 0.0)
            
            # Analyze dashboards costs
            if self.pricing_service:
                dashboards_analysis = await self._analyze_dashboards_costs(analysis_data)
                cost_breakdown['dashboards_costs'] = dashboards_analysis
                cost_breakdown['total_estimated_monthly'] += dashboards_analysis.get('estimated_monthly', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error generating cost breakdown: {str(e)}")
        
        return cost_breakdown
    
    async def _analyze_logs_costs(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CloudWatch Logs costs using available data."""
        logs_cost_analysis = {
            'estimated_monthly': 0.0,
            'breakdown': {
                'ingestion_cost': 0.0,
                'storage_cost': 0.0,
                'insights_cost': 0.0
            },
            'optimization_opportunities': []
        }
        
        try:
            # Get logs configuration data
            config_data = analysis_data.get('configuration_analysis', {})
            log_groups_data = config_data.get('log_groups', {})
            log_groups = log_groups_data.get('log_groups', [])
            
            # Get logs pricing
            if self.pricing_service:
                logs_pricing = self.pricing_service.get_logs_pricing()
                
                if logs_pricing.get('status') == 'success':
                    pricing = logs_pricing['logs_pricing']
                    
                    # Estimate costs based on log group configurations
                    total_ingestion_gb = 0.0
                    total_storage_gb = 0.0
                    
                    # Use minimal cost metrics if available
                    metrics_data = analysis_data.get('minimal_cost_metrics_analysis', {})
                    log_metrics = metrics_data.get('log_ingestion_metrics', {})
                    
                    if log_metrics:
                        # Use actual metrics data
                        total_ingestion_bytes = log_metrics.get('total_incoming_bytes', 0)
                        total_ingestion_gb = total_ingestion_bytes / (1024**3)
                        
                        log_group_metrics = log_metrics.get('log_group_metrics', [])
                        for lg_metric in log_group_metrics:
                            stored_bytes = lg_metric.get('stored_bytes', 0)
                            total_storage_gb += stored_bytes / (1024**3)
                    else:
                        # Estimate based on configuration
                        for log_group in log_groups:
                            stored_bytes = log_group.get('storedBytes', 0)
                            total_storage_gb += stored_bytes / (1024**3)
                            
                            # Estimate ingestion (rough approximation)
                            if stored_bytes > 0:
                                total_ingestion_gb += (stored_bytes / (1024**3)) * 0.1  # Assume 10% monthly ingestion
                    
                    # Calculate costs using pricing service
                    cost_calculation = self.pricing_service.calculate_logs_cost(
                        ingestion_gb=total_ingestion_gb,
                        storage_gb=total_storage_gb,
                        insights_gb_scanned=0  # We don't use Logs Insights
                    )
                    
                    if cost_calculation.get('status') == 'success':
                        logs_cost_analysis['estimated_monthly'] = cost_calculation['total_monthly_cost']
                        logs_cost_analysis['breakdown'] = cost_calculation['cost_breakdown']
                        
                        # Identify optimization opportunities
                        log_groups_analysis = log_groups_data.get('analysis', {})
                        without_retention = log_groups_analysis.get('without_retention_policy', [])
                        
                        if without_retention:
                            logs_cost_analysis['optimization_opportunities'].append({
                                'type': 'retention_policy',
                                'description': f'{len(without_retention)} log groups without retention policy',
                                'potential_savings': total_storage_gb * pricing['storage_per_gb_month'] * 0.5,
                                'affected_resources': without_retention[:10]  # Limit to first 10
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing logs costs: {str(e)}")
        
        return logs_cost_analysis
    
    async def _analyze_metrics_costs(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CloudWatch Metrics costs using available data."""
        metrics_cost_analysis = {
            'estimated_monthly': 0.0,
            'breakdown': {
                'custom_metrics_cost': 0.0,
                'api_requests_cost': 0.0,
                'detailed_monitoring_cost': 0.0
            },
            'optimization_opportunities': []
        }
        
        try:
            # Get metrics configuration data
            config_data = analysis_data.get('configuration_analysis', {})
            metrics_data = config_data.get('metrics', {})
            metrics_list = metrics_data.get('metrics', [])
            
            # Count custom metrics (non-AWS namespace)
            custom_metrics_count = 0
            # AWS service metrics are free - comprehensive list of AWS namespaces
            aws_namespaces = [
                'AWS/EC2', 'AWS/RDS', 'AWS/Lambda', 'AWS/S3', 'AWS/ELB', 'AWS/ELBv2',
                'AWS/ApplicationELB', 'AWS/NetworkELB', 'AWS/CloudFront', 'AWS/ApiGateway',
                'AWS/DynamoDB', 'AWS/SQS', 'AWS/SNS', 'AWS/Kinesis', 'AWS/ECS',
                'AWS/EKS', 'AWS/Batch', 'AWS/Logs', 'AWS/Events', 'AWS/AutoScaling',
                'AWS/ElastiCache', 'AWS/Redshift', 'AWS/EMR', 'AWS/Glue', 'AWS/StepFunctions',
                'AWS/Config', 'AWS/Usage', 'AWS/TrustedAdvisor', 'AWS/TransitGateway',
                'AWS/VPC', 'AWS/Route53', 'AWS/CloudWatch', 'AWS/Billing', 'AWS/Support',
                'AWS/Inspector', 'AWS/GuardDuty', 'AWS/SecurityHub', 'AWS/WAF', 'AWS/Shield'
            ]
            
            for metric in metrics_list:
                namespace = metric.get('Namespace', '')
                if not namespace.startswith('AWS/'):
                    custom_metrics_count += 1
            
            # Estimate API requests (rough approximation)
            estimated_api_requests = custom_metrics_count * 1000  # Assume 1000 requests per metric per month
            
            # Get metrics pricing and calculate costs
            if self.pricing_service:
                metrics_pricing = self.pricing_service.get_metrics_pricing()
                
                if metrics_pricing.get('status') == 'success':
                    cost_calculation = self.pricing_service.calculate_metrics_cost(
                        custom_metrics_count=custom_metrics_count,
                        api_requests_count=estimated_api_requests,
                        detailed_monitoring_instances=0  # We don't have EC2 instance data
                    )
                    
                    if cost_calculation.get('status') == 'success':
                        metrics_cost_analysis['estimated_monthly'] = cost_calculation['total_monthly_cost']
                        metrics_cost_analysis['breakdown'] = cost_calculation['cost_breakdown']
                        
                        # Identify optimization opportunities
                        if custom_metrics_count > 10:  # Beyond free tier
                            excess_metrics = custom_metrics_count - 10
                            pricing = metrics_pricing['metrics_pricing']
                            potential_savings = excess_metrics * pricing['custom_metrics_per_metric']
                            
                            metrics_cost_analysis['optimization_opportunities'].append({
                                'type': 'custom_metrics_optimization',
                                'description': f'{excess_metrics} custom metrics beyond free tier',
                                'potential_savings': potential_savings,
                                'affected_resources': [f'{excess_metrics} custom metrics']
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing metrics costs: {str(e)}")
        
        return metrics_cost_analysis
    
    async def _analyze_alarms_costs(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CloudWatch Alarms costs using available data."""
        alarms_cost_analysis = {
            'estimated_monthly': 0.0,
            'breakdown': {
                'standard_alarms_cost': 0.0,
                'high_resolution_alarms_cost': 0.0,
                'composite_alarms_cost': 0.0
            },
            'optimization_opportunities': []
        }
        
        try:
            # Get alarms configuration data
            config_data = analysis_data.get('configuration_analysis', {})
            alarms_data = config_data.get('alarms', {})
            alarms_list = alarms_data.get('alarms', [])
            alarms_analysis = alarms_data.get('analysis', {})
            
            # Count alarm types
            standard_alarms = 0
            high_resolution_alarms = 0
            composite_alarms = 0
            
            for alarm in alarms_list:
                if 'MetricName' in alarm:
                    # Metric alarm
                    period = alarm.get('Period', 300)
                    if period < 300:  # Less than 5 minutes = high resolution
                        high_resolution_alarms += 1
                    else:
                        standard_alarms += 1
                else:
                    # Composite alarm
                    composite_alarms += 1
            
            # Get alarms pricing and calculate costs
            if self.pricing_service:
                alarms_pricing = self.pricing_service.get_alarms_pricing()
                
                if alarms_pricing.get('status') == 'success':
                    cost_calculation = self.pricing_service.calculate_alarms_cost(
                        standard_alarms_count=standard_alarms,
                        high_resolution_alarms_count=high_resolution_alarms,
                        composite_alarms_count=composite_alarms
                    )
                    
                    if cost_calculation.get('status') == 'success':
                        alarms_cost_analysis['estimated_monthly'] = cost_calculation['total_monthly_cost']
                        alarms_cost_analysis['breakdown'] = cost_calculation['cost_breakdown']
                        
                        # Identify optimization opportunities
                        alarms_without_actions = alarms_analysis.get('alarms_without_actions', [])
                        high_res_alarms = alarms_analysis.get('high_resolution_alarms', [])
                        
                        if alarms_without_actions:
                            alarms_cost_analysis['optimization_opportunities'].append({
                                'type': 'unused_alarms',
                                'description': f'{len(alarms_without_actions)} alarms without actions',
                                'potential_savings': len(alarms_without_actions) * 0.10,  # Standard alarm cost
                                'affected_resources': alarms_without_actions[:10]
                            })
                        
                        if high_res_alarms:
                            pricing = alarms_pricing['alarms_pricing']
                            savings_per_alarm = pricing['high_resolution_alarms_per_alarm'] - pricing['standard_alarms_per_alarm']
                            potential_savings = len(high_res_alarms) * savings_per_alarm
                            
                            alarms_cost_analysis['optimization_opportunities'].append({
                                'type': 'high_resolution_optimization',
                                'description': f'{len(high_res_alarms)} high-resolution alarms could be standard',
                                'potential_savings': potential_savings,
                                'affected_resources': high_res_alarms[:10]
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing alarms costs: {str(e)}")
        
        return alarms_cost_analysis
    
    async def _analyze_dashboards_costs(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CloudWatch Dashboards costs using available data."""
        dashboards_cost_analysis = {
            'estimated_monthly': 0.0,
            'breakdown': {
                'dashboards_cost': 0.0
            },
            'optimization_opportunities': []
        }
        
        try:
            # Get dashboards configuration data
            config_data = analysis_data.get('configuration_analysis', {})
            dashboards_data = config_data.get('dashboards', {})
            dashboards_list = dashboards_data.get('dashboards', [])
            dashboards_analysis = dashboards_data.get('analysis', {})
            
            total_dashboards = len(dashboards_list)
            
            # Get dashboards pricing and calculate costs
            if self.pricing_service:
                dashboards_pricing = self.pricing_service.get_dashboards_pricing()
                
                if dashboards_pricing.get('status') == 'success':
                    cost_calculation = self.pricing_service.calculate_dashboards_cost(
                        dashboards_count=total_dashboards
                    )
                    
                    if cost_calculation.get('status') == 'success':
                        dashboards_cost_analysis['estimated_monthly'] = cost_calculation['total_monthly_cost']
                        dashboards_cost_analysis['breakdown'] = cost_calculation['cost_breakdown']
                        
                        # Identify optimization opportunities
                        if dashboards_analysis.get('exceeds_free_tier', False):
                            excess_dashboards = dashboards_analysis.get('potential_cost_dashboards', 0)
                            pricing = dashboards_pricing['dashboards_pricing']
                            potential_savings = excess_dashboards * pricing['dashboard_per_month']
                            
                            dashboards_cost_analysis['optimization_opportunities'].append({
                                'type': 'dashboard_consolidation',
                                'description': f'{excess_dashboards} dashboards beyond free tier',
                                'potential_savings': potential_savings,
                                'affected_resources': [f'{excess_dashboards} paid dashboards']
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing dashboards costs: {str(e)}")
        
        return dashboards_cost_analysis
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations with cost-aware feature degradation.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            data = analysis_results.get('data', {})
            cost_breakdown = data.get('cost_breakdown', {})
            
            # Generate recommendations based on cost analysis
            self._add_logs_recommendations(recommendations, cost_breakdown.get('logs_costs', {}))
            self._add_metrics_recommendations(recommendations, cost_breakdown.get('metrics_costs', {}))
            self._add_alarms_recommendations(recommendations, cost_breakdown.get('alarms_costs', {}))
            self._add_dashboards_recommendations(recommendations, cost_breakdown.get('dashboards_costs', {}))
            
            # Add cost-aware feature recommendations
            self._add_cost_aware_recommendations(recommendations, analysis_results)
            
            # Sort recommendations by potential savings (descending)
            recommendations.sort(key=lambda x: x.get('potential_savings', 0), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append(
                self.create_recommendation(
                    rec_type="error_resolution",
                    priority="high",
                    title="Recommendation Generation Error",
                    description=f"Failed to generate recommendations: {str(e)}",
                    cloudwatch_component="general"
                )
            )
        
        return recommendations
    
    def _add_logs_recommendations(self, recommendations: List[Dict[str, Any]], 
                                logs_costs: Dict[str, Any]) -> None:
        """Add CloudWatch Logs optimization recommendations."""
        optimization_opportunities = logs_costs.get('optimization_opportunities', [])
        
        for opportunity in optimization_opportunities:
            if opportunity['type'] == 'retention_policy':
                recommendations.append(
                    self.create_recommendation(
                        rec_type="cost_optimization",
                        priority="high",
                        title="Implement Log Retention Policies",
                        description=opportunity['description'] + " - storing logs indefinitely increases storage costs",
                        potential_savings=opportunity.get('potential_savings', 0),
                        implementation_effort="low",
                        affected_resources=opportunity.get('affected_resources', []),
                        action_items=[
                            "Review log groups without retention policies",
                            "Set appropriate retention periods based on compliance requirements",
                            "Consider shorter retention for debug/development logs",
                            "Use lifecycle policies to archive old logs to S3"
                        ],
                        cloudwatch_component="logs"
                    )
                )
    
    def _add_metrics_recommendations(self, recommendations: List[Dict[str, Any]], 
                                   metrics_costs: Dict[str, Any]) -> None:
        """Add CloudWatch Metrics optimization recommendations."""
        optimization_opportunities = metrics_costs.get('optimization_opportunities', [])
        
        for opportunity in optimization_opportunities:
            if opportunity['type'] == 'custom_metrics_optimization':
                recommendations.append(
                    self.create_recommendation(
                        rec_type="cost_optimization",
                        priority="medium",
                        title="Optimize Custom Metrics Usage",
                        description=opportunity['description'] + " - review necessity of custom metrics",
                        potential_savings=opportunity.get('potential_savings', 0),
                        implementation_effort="medium",
                        affected_resources=opportunity.get('affected_resources', []),
                        action_items=[
                            "Audit custom metrics for business value",
                            "Consolidate similar metrics where possible",
                            "Remove unused or redundant custom metrics",
                            "Consider using CloudWatch Embedded Metric Format for efficiency"
                        ],
                        cloudwatch_component="metrics"
                    )
                )
    
    def _add_alarms_recommendations(self, recommendations: List[Dict[str, Any]], 
                                  alarms_costs: Dict[str, Any]) -> None:
        """Add CloudWatch Alarms optimization recommendations."""
        optimization_opportunities = alarms_costs.get('optimization_opportunities', [])
        
        for opportunity in optimization_opportunities:
            if opportunity['type'] == 'unused_alarms':
                recommendations.append(
                    self.create_recommendation(
                        rec_type="cost_optimization",
                        priority="high",
                        title="Remove Unused Alarms",
                        description=opportunity['description'] + " - alarms without actions provide no value",
                        potential_savings=opportunity.get('potential_savings', 0),
                        implementation_effort="low",
                        affected_resources=opportunity.get('affected_resources', []),
                        action_items=[
                            "Review alarms without SNS, Auto Scaling, or EC2 actions",
                            "Add appropriate actions or delete unused alarms",
                            "Verify alarm thresholds are still relevant",
                            "Consider consolidating similar alarms"
                        ],
                        cloudwatch_component="alarms"
                    )
                )
            elif opportunity['type'] == 'high_resolution_optimization':
                recommendations.append(
                    self.create_recommendation(
                        rec_type="cost_optimization",
                        priority="medium",
                        title="Convert High-Resolution Alarms to Standard",
                        description=opportunity['description'] + " - high-resolution alarms cost 3x more",
                        potential_savings=opportunity.get('potential_savings', 0),
                        implementation_effort="low",
                        affected_resources=opportunity.get('affected_resources', []),
                        action_items=[
                            "Review necessity of sub-minute alarm evaluation",
                            "Convert to standard resolution where 5-minute evaluation is sufficient",
                            "Keep high-resolution only for critical real-time monitoring",
                            "Update alarm periods from <300 seconds to ≥300 seconds"
                        ],
                        cloudwatch_component="alarms"
                    )
                )
    
    def _add_dashboards_recommendations(self, recommendations: List[Dict[str, Any]], 
                                      dashboards_costs: Dict[str, Any]) -> None:
        """Add CloudWatch Dashboards optimization recommendations."""
        optimization_opportunities = dashboards_costs.get('optimization_opportunities', [])
        
        for opportunity in optimization_opportunities:
            if opportunity['type'] == 'dashboard_consolidation':
                recommendations.append(
                    self.create_recommendation(
                        rec_type="cost_optimization",
                        priority="medium",
                        title="Consolidate Dashboards to Reduce Costs",
                        description=opportunity['description'] + " - each dashboard beyond 3 costs $3/month",
                        potential_savings=opportunity.get('potential_savings', 0),
                        implementation_effort="medium",
                        affected_resources=opportunity.get('affected_resources', []),
                        action_items=[
                            "Review dashboard usage and necessity",
                            "Consolidate related dashboards where possible",
                            "Remove unused or duplicate dashboards",
                            "Optimize dashboard widgets to stay within free tier limits"
                        ],
                        cloudwatch_component="dashboards"
                    )
                )
    
    def _add_cost_aware_recommendations(self, recommendations: List[Dict[str, Any]], 
                                      analysis_results: Dict[str, Any]) -> None:
        """Add recommendations based on cost-aware feature degradation."""
        cost_incurred = analysis_results.get('cost_incurred', False)
        cost_incurring_operations = analysis_results.get('cost_incurring_operations', [])
        
        # Recommend enabling Cost Explorer if not used
        if not self.cost_preferences.get('allow_cost_explorer', False):
            recommendations.append(
                self.create_recommendation(
                    rec_type="analysis_enhancement",
                    priority="medium",
                    title="Enable Cost Explorer for Detailed Cost Analysis",
                    description="Cost Explorer provides historical cost data and trends for more accurate optimization recommendations",
                    potential_savings=None,
                    implementation_effort="low",
                    action_items=[
                        "Set allow_cost_explorer=True to enable historical cost analysis",
                        "Cost: ~$0.01 per API call for comprehensive cost data",
                        "Provides 30% additional functionality for cost optimization",
                        "Enables cost forecasting and trend analysis"
                    ],
                    cloudwatch_component="general"
                )
            )
        
        # Recommend minimal cost metrics if not used
        if not self.cost_preferences.get('allow_minimal_cost_metrics', False):
            recommendations.append(
                self.create_recommendation(
                    rec_type="analysis_enhancement",
                    priority="low",
                    title="Enable Minimal Cost Metrics for Log Analysis",
                    description="Targeted CloudWatch metrics provide actual log ingestion data for more accurate cost calculations",
                    potential_savings=None,
                    implementation_effort="low",
                    action_items=[
                        "Set allow_minimal_cost_metrics=True for detailed log analysis",
                        "Cost: ~$0.01 per 1000 metric requests (minimal usage)",
                        "Provides 4% additional functionality for log cost analysis",
                        "Enables accurate log ingestion cost calculations"
                    ],
                    cloudwatch_component="logs"
                )
            )
        
        # Add transparency about current analysis limitations
        if not cost_incurred:
            recommendations.append(
                self.create_recommendation(
                    rec_type="analysis_transparency",
                    priority="low",
                    title="Current Analysis Uses Free Operations Only",
                    description="Analysis based on CloudWatch configuration APIs and pricing estimates. Enable paid features for more accurate cost data.",
                    potential_savings=None,
                    implementation_effort="low",
                    action_items=[
                        "Current analysis provides 60% functionality using free operations",
                        "Enable Cost Explorer for historical cost data (+30% functionality)",
                        "Enable minimal cost metrics for actual usage data (+4% functionality)",
                        "All recommendations are based on configuration analysis and pricing models"
                    ],
                    cloudwatch_component="general"
                )
            )