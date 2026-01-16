"""
CloudWatch Service for CFM Tips MCP Server

Clean architecture with specialized classes for each CloudWatch functionality:
- CWGeneralSpendTips: General spending analysis and cost optimization
- CWMetricsTips: Custom metrics optimization and analysis
- CWLogsTips: Logs ingestion, storage, and retention optimization
- CWAlarmsTips: Alarms configuration and cost optimization
- CWDashboardTips: Dashboard management and optimization

Internal components (not exposed outside this file):
- CloudWatchDAO: Data access for CloudWatch APIs
- AWSPricingDAO: Data access for AWS Pricing APIs
- CloudWatchCache: Caching layer for performance
"""

import logging
import boto3
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError
from dataclasses import dataclass, field

from playbooks.cloudwatch.cost_controller import CostController, CostPreferences
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


@dataclass
class CloudWatchOperationResult:
    """Result of a CloudWatch operation with cost tracking."""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    operation_name: str = ""
    cost_incurred: bool = False
    operation_type: str = "free"
    execution_time: float = 0.0
    fallback_used: bool = False
    primary_data_source: str = "cloudwatch_api"
    api_calls_made: List[str] = field(default_factory=list)


@dataclass
class CloudWatchServiceConfig:
    """Configuration for CloudWatch service operations."""
    region: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: float = 30.0
    enable_cost_tracking: bool = True
    enable_fallback: bool = True
    cost_preferences: Optional[CostPreferences] = None


class CloudWatchCache:
    """Internal caching layer for CloudWatch data. Not exposed outside this file."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        
        if time.time() - self._timestamps[key] > self.ttl_seconds:
            self._invalidate(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def _invalidate(self, key: str) -> None:
        """Remove expired cache entry."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for ts in self._timestamps.values() 
            if current_time - ts > self.ttl_seconds
        )
        
        return {
            'total_entries': len(self._cache),
            'expired_entries': expired_count,
            'valid_entries': len(self._cache) - expired_count,
            'ttl_seconds': self.ttl_seconds
        }


class AWSPricingDAO:
    """Data Access Object for AWS Pricing API. Not exposed outside this file."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self._cache = CloudWatchCache(ttl_seconds=3600)  # 1 hour cache for pricing
        self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        
        # Region mapping for pricing API
        self._region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-central-1': 'Europe (Frankfurt)',
            'eu-west-1': 'Europe (Ireland)',
            'eu-west-2': 'Europe (London)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-southeast-2': 'Asia Pacific (Sydney)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
        }
        
        # Free tier limits
        self._free_tier = {
            'logs_ingestion_gb': 5.0,
            'logs_storage_gb': 5.0,
            'metrics_count': 10,
            'api_requests': 1000000,
            'alarms_count': 10,
            'dashboards_count': 3,
            'dashboard_metrics': 50
        }
    
    def get_pricing_data(self, component: str) -> Dict[str, Any]:
        """Get pricing data for CloudWatch components with caching."""
        cache_key = f"pricing_{component}_{self.region}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Fallback pricing data (typical CloudWatch pricing)
        pricing_data = {
            'logs': {
                'ingestion_per_gb': 0.50,
                'storage_per_gb_month': 0.03,
                'insights_per_gb_scanned': 0.005,
                'vended_logs_per_gb': 0.10,
                'cross_region_delivery_per_gb': 0.02
            },
            'metrics': {
                'custom_metrics_per_metric': 0.30,
                'detailed_monitoring_per_instance': 2.10,
                'high_resolution_metrics_per_metric': 0.30,
                'api_requests_per_1000': 0.01,
                'get_metric_statistics_per_1000': 0.01,
                'put_metric_data_per_1000': 0.01
            },
            'alarms': {
                'standard_alarms_per_alarm': 0.10,
                'high_resolution_alarms_per_alarm': 0.30,
                'composite_alarms_per_alarm': 0.50,
                'alarm_actions_sns': 0.0,
                'alarm_actions_autoscaling': 0.0,
                'alarm_actions_ec2': 0.0
            },
            'dashboards': {
                'dashboard_per_month': 3.00,
                'metrics_per_dashboard_free': 50,
                'additional_metrics_per_metric': 0.0,
                'dashboard_api_requests_per_1000': 0.01
            }
        }
        
        result = pricing_data.get(component, {})
        self._cache.set(cache_key, result)
        return result
    
    def get_free_tier_limits(self) -> Dict[str, Any]:
        """Get free tier limits for CloudWatch services."""
        return self._free_tier.copy()
    
    def calculate_cost(self, component: str, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate costs for CloudWatch components with free tier consideration."""
        pricing = self.get_pricing_data(component)
        
        if component == 'logs':
            return self._calculate_logs_cost(usage, pricing)
        elif component == 'metrics':
            return self._calculate_metrics_cost(usage, pricing)
        elif component == 'alarms':
            return self._calculate_alarms_cost(usage, pricing)
        elif component == 'dashboards':
            return self._calculate_dashboards_cost(usage, pricing)
        else:
            return {'status': 'error', 'message': f'Unknown component: {component}'}
    
    def _calculate_logs_cost(self, usage: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Logs costs."""
        ingestion_gb = usage.get('ingestion_gb', 0)
        storage_gb = usage.get('storage_gb', 0)
        insights_gb_scanned = usage.get('insights_gb_scanned', 0)
        
        ingestion_billable = max(0, ingestion_gb - self._free_tier['logs_ingestion_gb'])
        storage_billable = max(0, storage_gb - self._free_tier['logs_storage_gb'])
        
        ingestion_cost = ingestion_billable * pricing['ingestion_per_gb']
        storage_cost = storage_billable * pricing['storage_per_gb_month']
        insights_cost = insights_gb_scanned * pricing['insights_per_gb_scanned']
        
        total_cost = ingestion_cost + storage_cost + insights_cost
        
        return {
            'status': 'success',
            'usage': usage,
            'billable_usage': {
                'ingestion_gb': ingestion_billable,
                'storage_gb': storage_billable,
                'insights_gb_scanned': insights_gb_scanned
            },
            'cost_breakdown': {
                'ingestion_cost': round(ingestion_cost, 4),
                'storage_cost': round(storage_cost, 4),
                'insights_cost': round(insights_cost, 4)
            },
            'total_monthly_cost': round(total_cost, 4),
            'total_annual_cost': round(total_cost * 12, 2)
        }
    
    def _calculate_metrics_cost(self, usage: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Metrics costs."""
        custom_metrics_count = usage.get('custom_metrics_count', 0)
        api_requests_count = usage.get('api_requests_count', 0)
        detailed_monitoring_instances = usage.get('detailed_monitoring_instances', 0)
        
        metrics_billable = max(0, custom_metrics_count - self._free_tier['metrics_count'])
        requests_billable = max(0, api_requests_count - self._free_tier['api_requests'])
        
        metrics_cost = metrics_billable * pricing['custom_metrics_per_metric']
        requests_cost = (requests_billable / 1000) * pricing['api_requests_per_1000']
        detailed_monitoring_cost = detailed_monitoring_instances * pricing['detailed_monitoring_per_instance']
        
        total_cost = metrics_cost + requests_cost + detailed_monitoring_cost
        
        return {
            'status': 'success',
            'usage': usage,
            'billable_usage': {
                'custom_metrics_count': metrics_billable,
                'api_requests_count': requests_billable,
                'detailed_monitoring_instances': detailed_monitoring_instances
            },
            'cost_breakdown': {
                'metrics_cost': round(metrics_cost, 4),
                'requests_cost': round(requests_cost, 4),
                'detailed_monitoring_cost': round(detailed_monitoring_cost, 4)
            },
            'total_monthly_cost': round(total_cost, 4),
            'total_annual_cost': round(total_cost * 12, 2)
        }
    
    def _calculate_alarms_cost(self, usage: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Alarms costs."""
        standard_alarms_count = usage.get('standard_alarms_count', 0)
        high_resolution_alarms_count = usage.get('high_resolution_alarms_count', 0)
        composite_alarms_count = usage.get('composite_alarms_count', 0)
        
        standard_billable = max(0, standard_alarms_count - self._free_tier['alarms_count'])
        
        standard_cost = standard_billable * pricing['standard_alarms_per_alarm']
        high_resolution_cost = high_resolution_alarms_count * pricing['high_resolution_alarms_per_alarm']
        composite_cost = composite_alarms_count * pricing['composite_alarms_per_alarm']
        
        total_cost = standard_cost + high_resolution_cost + composite_cost
        
        return {
            'status': 'success',
            'usage': usage,
            'billable_usage': {
                'standard_alarms_count': standard_billable,
                'high_resolution_alarms_count': high_resolution_alarms_count,
                'composite_alarms_count': composite_alarms_count
            },
            'cost_breakdown': {
                'standard_alarms_cost': round(standard_cost, 4),
                'high_resolution_alarms_cost': round(high_resolution_cost, 4),
                'composite_alarms_cost': round(composite_cost, 4)
            },
            'total_monthly_cost': round(total_cost, 4),
            'total_annual_cost': round(total_cost * 12, 2)
        }
    
    def _calculate_dashboards_cost(self, usage: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Dashboards costs."""
        dashboards_count = usage.get('dashboards_count', 0)
        
        dashboards_billable = max(0, dashboards_count - self._free_tier['dashboards_count'])
        dashboards_cost = dashboards_billable * pricing['dashboard_per_month']
        
        return {
            'status': 'success',
            'usage': usage,
            'billable_usage': {
                'dashboards_count': dashboards_billable
            },
            'cost_breakdown': {
                'dashboards_cost': round(dashboards_cost, 4)
            },
            'total_monthly_cost': round(dashboards_cost, 4),
            'total_annual_cost': round(dashboards_cost * 12, 2)
        }


class CloudWatchDAO:
    """Data Access Object for CloudWatch operations. Not exposed outside this file."""
    
    def __init__(self, region: Optional[str] = None, cost_controller: Optional[CostController] = None):
        self.region = region
        self.cost_controller = cost_controller or CostController()
        self._cache = CloudWatchCache()
        
        # Initialize AWS clients
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
        self.logs_client = boto3.client('logs', region_name=self.region)
        
        logger.debug(f"CloudWatch DAO initialized for region: {self.region}")
    
    async def list_metrics(self, namespace: Optional[str] = None, 
                          metric_name: Optional[str] = None,
                          dimensions: Optional[List[Dict[str, str]]] = None,
                          page: int = 1) -> Dict[str, Any]:
        """
        List CloudWatch metrics with pagination.
        
        Args:
            namespace: Filter by namespace
            metric_name: Filter by metric name  
            dimensions: Filter by dimensions
            page: Which AWS API page to retrieve (1-based, default=1)
        
        Returns:
            Dictionary with metrics from requested page only
        """
        cache_key = f"metrics_{namespace}_{metric_name}_{hash(str(dimensions))}_page{page}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        params = {}
        if namespace:
            params['Namespace'] = namespace
        if metric_name:
            params['MetricName'] = metric_name
        if dimensions:
            params['Dimensions'] = dimensions
        
        metrics = []
        paginator = self.cloudwatch_client.get_paginator('list_metrics')
        
        # Only fetch the requested page
        page_count = 0
        for page_response in paginator.paginate(**params):
            page_count += 1
            if page_count == page:
                metrics.extend(page_response.get('Metrics', []))
                break
        
        result = {
            'metrics': metrics,
            'total_count': len(metrics),
            'namespace': namespace,
            'filtered': bool(namespace or metric_name or dimensions),
            'page': page
        }
        
        self._cache.set(cache_key, result)
        return result
    
    async def describe_alarms(self, alarm_names: Optional[List[str]] = None,
                             alarm_name_prefix: Optional[str] = None,
                             state_value: Optional[str] = None) -> Dict[str, Any]:
        """Describe CloudWatch alarms with caching."""
        cache_key = f"alarms_{hash(str(alarm_names))}_{alarm_name_prefix}_{state_value}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        params = {}
        if alarm_names:
            params['AlarmNames'] = alarm_names
        if alarm_name_prefix:
            params['AlarmNamePrefix'] = alarm_name_prefix
        if state_value:
            params['StateValue'] = state_value
        
        alarms = []
        paginator = self.cloudwatch_client.get_paginator('describe_alarms')
        
        for page in paginator.paginate(**params):
            alarms.extend(page.get('MetricAlarms', []))
            alarms.extend(page.get('CompositeAlarms', []))
        
        result = {
            'alarms': alarms,
            'total_count': len(alarms),
            'filtered': bool(alarm_names or alarm_name_prefix or state_value)
        }
        
        self._cache.set(cache_key, result)
        return result
    
    async def list_dashboards(self, dashboard_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """List CloudWatch dashboards with caching."""
        cache_key = f"dashboards_{dashboard_name_prefix}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        params = {}
        if dashboard_name_prefix:
            params['DashboardNamePrefix'] = dashboard_name_prefix
        
        dashboards = []
        paginator = self.cloudwatch_client.get_paginator('list_dashboards')
        
        for page in paginator.paginate(**params):
            dashboards.extend(page.get('DashboardEntries', []))
        
        result = {
            'dashboards': dashboards,
            'total_count': len(dashboards),
            'filtered': bool(dashboard_name_prefix)
        }
        
        self._cache.set(cache_key, result)
        return result
    
    async def describe_log_groups(self, log_group_name_prefix: Optional[str] = None,
                                 log_group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Describe CloudWatch log groups with caching."""
        cache_key = f"log_groups_{log_group_name_prefix}_{hash(str(log_group_names))}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        params = {}
        if log_group_name_prefix:
            params['logGroupNamePrefix'] = log_group_name_prefix
        if log_group_names:
            params['logGroupNames'] = log_group_names
        
        log_groups = []
        paginator = self.logs_client.get_paginator('describe_log_groups')
        
        for page in paginator.paginate(**params):
            log_groups.extend(page.get('logGroups', []))
        
        result = {
            'log_groups': log_groups,
            'total_count': len(log_groups),
            'filtered': bool(log_group_name_prefix or log_group_names)
        }
        
        self._cache.set(cache_key, result)
        return result
    
    async def get_dashboard(self, dashboard_name: str) -> Dict[str, Any]:
        """Get dashboard configuration with caching."""
        cache_key = f"dashboard_config_{dashboard_name}"
        cached_result = self._cache.get(cache_key)
        if cached_result:
            return cached_result
        
        response = self.cloudwatch_client.get_dashboard(DashboardName=dashboard_name)
        
        # Ensure dashboard_body is never empty string - use '{}' as default
        dashboard_body = response.get('DashboardBody', '{}')
        if not dashboard_body or dashboard_body.strip() == '':
            dashboard_body = '{}'
        
        result = {
            'dashboard_name': dashboard_name,
            'dashboard_body': dashboard_body,
            'dashboard_arn': response.get('DashboardArn')
        }
        
        self._cache.set(cache_key, result)
        return result
    
    async def get_metric_statistics(self, namespace: str, metric_name: str,
                                   dimensions: List[Dict[str, str]],
                                   start_time: datetime, end_time: datetime,
                                   period: int = 3600,
                                   statistics: List[str] = None) -> Dict[str, Any]:
        """Get metric statistics (paid operation)."""
        stats_list = statistics or ['Average', 'Sum', 'Maximum']
        
        response = self.cloudwatch_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=stats_list
        )
        
        datapoints = response.get('Datapoints', [])
        datapoints.sort(key=lambda x: x['Timestamp'])
        
        return {
            'namespace': namespace,
            'metric_name': metric_name,
            'dimensions': dimensions,
            'datapoints': datapoints,
            'total_datapoints': len(datapoints),
            'period': period,
            'statistics': stats_list,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    async def get_metrics_usage_batch(self, metrics: List[Dict[str, Any]], 
                                     lookback_days: int = 30,
                                     batch_size: int = 500) -> None:
        """
        Get usage data for multiple metrics in batches (paid operation).
        
        Uses GetMetricData API which supports up to 500 metrics per request.
        Results are cached to avoid repeated API calls.
        Updates metrics in place with datapoint_count and usage_estimation_method.
        
        Args:
            metrics: List of metric dictionaries to analyze (modified in place)
            lookback_days: Number of days to look back for usage data
            batch_size: Number of metrics per API call (max 500)
        """
        if not metrics:
            return
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        
        # Process metrics in batches of up to 500
        for batch_start in range(0, len(metrics), batch_size):
            batch = metrics[batch_start:batch_start + batch_size]
            
            # Check cache first for this batch
            cache_key = f"metrics_usage_{hash(str([(m['namespace'], m['metric_name'], str(m['dimensions'])) for m in batch]))}_{lookback_days}"
            cached_result = self._cache.get(cache_key)
            
            if cached_result:
                logger.debug(f"Using cached usage data for batch {batch_start//batch_size + 1}")
                # Apply cached results
                for i, metric in enumerate(batch):
                    if i < len(cached_result):
                        metric.update(cached_result[i])
                continue
            
            # Build metric queries for this batch
            metric_queries = []
            for idx, metric in enumerate(batch):
                metric_queries.append({
                    'Id': f'm{idx}',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': metric['namespace'],
                            'MetricName': metric['metric_name'],
                            'Dimensions': metric['dimensions']
                        },
                        'Period': 3600,  # 1 hour
                        'Stat': 'SampleCount'
                    }
                })
            
            # Single batched API call for up to 500 metrics
            try:
                logger.debug(f"Fetching usage data for batch {batch_start//batch_size + 1} ({len(batch)} metrics)")
                
                response = self.cloudwatch_client.get_metric_data(
                    MetricDataQueries=metric_queries,
                    StartTime=start_time,
                    EndTime=end_time
                )
                
                # Process results and update metrics in place
                batch_results = []
                for idx, result in enumerate(response.get('MetricDataResults', [])):
                    if idx < len(batch):
                        datapoint_count = len(result.get('Values', []))
                        
                        # Update metric in place
                        batch[idx]['datapoint_count'] = datapoint_count
                        batch[idx]['usage_period_days'] = lookback_days
                        batch[idx]['usage_estimation_method'] = 'exact_paid'
                        
                        # Store for cache
                        batch_results.append({
                            'datapoint_count': datapoint_count,
                            'usage_period_days': lookback_days,
                            'usage_estimation_method': 'exact_paid'
                        })
                
                # Cache the results
                self._cache.set(cache_key, batch_results)
                
            except Exception as e:
                logger.error(f"Failed to get usage data for batch {batch_start//batch_size + 1}: {str(e)}")
                # Mark all metrics in batch as failed
                for metric in batch:
                    metric['datapoint_count'] = 0
                    metric['usage_period_days'] = lookback_days
                    metric['usage_estimation_method'] = 'failed'
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


class CWGeneralSpendTips:
    """
    CloudWatch general spending analysis with 4 public methods.
    
    Public methods:
    - getLogs(): Returns log groups ordered by spend (descending), paginated
    - getMetrics(): Returns custom metrics ordered by dimension count (descending), paginated
    - getDashboards(): Returns dashboards ordered by custom metrics count (descending), paginated
    - getAlarms(): Returns alarms with cost information, paginated
    """
    
    def __init__(self, dao: CloudWatchDAO, pricing_dao: AWSPricingDAO, cost_preferences: CostPreferences):
        self.dao = dao
        self.pricing_dao = pricing_dao
        self.cost_preferences = cost_preferences
        self._page_size = 10  # Items per page
    
    async def getLogs(self, page: int = 1, log_group_name_prefix: Optional[str] = None, 
                     can_spend_for_estimate: bool = False, 
                     estimate_ingestion_from_metadata: bool = True,
                     lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get log groups ordered by estimated spend (descending), paginated.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            can_spend_for_estimate: If True, uses CloudWatch Metrics API (paid) to get accurate 
                                   ingestion data from IncomingBytes metric. If False (default), 
                                   uses free estimation methods. Enabling this provides accurate 
                                   cost estimates but incurs minimal CloudWatch API charges.
            estimate_ingestion_from_metadata: If True (default) and can_spend_for_estimate=False,
                                             estimates ingestion using free metadata (retention 
                                             policy and log group age). If False, only storage 
                                             costs are calculated. This is always free.
            lookback_days: Number of days to analyze for ingestion (only used if can_spend_for_estimate=True)
            
        Returns:
            Dict with log_groups, pagination, summary, and pricing_info
        """
        try:
            # Get log groups from DAO
            log_groups_data = await self.dao.describe_log_groups(
                log_group_name_prefix=log_group_name_prefix
            )
            log_groups = log_groups_data['log_groups']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('logs')
            free_tier = self.pricing_dao.get_free_tier_limits()
            
            # Calculate spend for each log group
            log_groups_with_spend = []
            total_storage_gb = 0
            total_ingestion_gb = 0
            
            for lg in log_groups:
                stored_bytes = lg.get('storedBytes', 0)
                stored_gb = stored_bytes / (1024**3)
                retention_days = lg.get('retentionInDays', 0)
                log_group_name = lg.get('logGroupName')
                
                # Calculate storage cost (always accurate)
                storage_cost = stored_gb * pricing['storage_per_gb_month']
                
                # Calculate ingestion cost
                if can_spend_for_estimate:
                    # Option 1: Use CloudWatch Metrics API for accurate data (PAID)
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=lookback_days)
                        
                        ingestion_data = await self.dao.get_metric_statistics(
                            namespace='AWS/Logs',
                            metric_name='IncomingBytes',
                            dimensions=[{'Name': 'LogGroupName', 'Value': log_group_name}],
                            start_time=start_time,
                            end_time=end_time,
                            period=86400,  # Daily aggregation
                            statistics=['Sum']
                        )
                        
                        # Calculate total ingestion in GB
                        total_bytes = sum(dp['Sum'] for dp in ingestion_data['datapoints'])
                        ingestion_gb = total_bytes / (1024**3)
                        
                        # Normalize to monthly rate
                        monthly_ingestion_gb = (ingestion_gb / lookback_days) * 30
                        ingestion_cost = monthly_ingestion_gb * pricing['ingestion_per_gb']
                        estimation_method = 'accurate_paid'
                        confidence = 'high'
                        
                    except Exception as e:
                        logger.warning(f"Failed to get ingestion metrics for {log_group_name}: {str(e)}")
                        # Fallback to free estimation if metrics fail
                        if estimate_ingestion_from_metadata:
                            result = self._estimate_ingestion_from_metadata(lg, stored_gb)
                            monthly_ingestion_gb = result['monthly_ingestion_gb']
                            ingestion_cost = monthly_ingestion_gb * pricing['ingestion_per_gb']
                            estimation_method = result['estimation_method']
                            confidence = result['confidence']
                        else:
                            monthly_ingestion_gb = 0
                            ingestion_cost = 0
                            estimation_method = 'storage_only'
                            confidence = 'high'
                
                elif estimate_ingestion_from_metadata:
                    # Option 2: Estimate from metadata (FREE - uses retention/age)
                    result = self._estimate_ingestion_from_metadata(lg, stored_gb)
                    monthly_ingestion_gb = result['monthly_ingestion_gb']
                    ingestion_cost = monthly_ingestion_gb * pricing['ingestion_per_gb']
                    estimation_method = result['estimation_method']
                    confidence = result['confidence']
                
                else:
                    # Option 3: Storage only (FREE - most conservative)
                    monthly_ingestion_gb = 0
                    ingestion_cost = 0
                    estimation_method = 'storage_only'
                    confidence = 'high'
                
                total_cost = storage_cost + ingestion_cost
                total_storage_gb += stored_gb
                total_ingestion_gb += monthly_ingestion_gb
                
                log_groups_with_spend.append({
                    'log_group_name': log_group_name,
                    'stored_gb': round(stored_gb, 4),
                    'stored_bytes': stored_bytes,
                    'retention_days': retention_days if retention_days else 'Never Expire',
                    'creation_time': lg.get('creationTime'),
                    'cost_breakdown': {
                        'storage_cost': round(storage_cost, 4),
                        'ingestion_cost': round(ingestion_cost, 4),
                        'ingestion_gb_monthly': round(monthly_ingestion_gb, 4) if monthly_ingestion_gb > 0 else None
                    },
                    'estimated_monthly_cost': round(total_cost, 4),
                    'estimated_annual_cost': round(total_cost * 12, 2),
                    'estimation_method': estimation_method,
                    'estimation_confidence': confidence
                })
            
            # Sort by spend descending
            log_groups_with_spend.sort(key=lambda x: x['estimated_monthly_cost'], reverse=True)
            
            # Paginate
            total_items = len(log_groups_with_spend)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_logs = log_groups_with_spend[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_cost = sum(lg['estimated_monthly_cost'] for lg in log_groups_with_spend)
            
            # Free tier analysis
            storage_billable = max(0, total_storage_gb - free_tier['logs_storage_gb'])
            ingestion_billable = max(0, total_ingestion_gb - free_tier['logs_ingestion_gb']) if can_spend_for_estimate else 0
            
            return {
                'status': 'success',
                'log_groups': paginated_logs,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_log_groups': total_items,
                    'total_storage_gb': round(total_storage_gb, 4),
                    'total_ingestion_gb_monthly': round(total_ingestion_gb, 4) if can_spend_for_estimate else None,
                    'free_tier_limit_gb': free_tier['logs_storage_gb'],
                    'free_tier_remaining_gb': round(max(0, free_tier['logs_storage_gb'] - total_storage_gb), 4),
                    'billable_storage_gb': round(storage_billable, 4),
                    'billable_ingestion_gb': round(ingestion_billable, 4) if can_spend_for_estimate else None,
                    'total_estimated_monthly_cost': round(total_monthly_cost, 4),
                    'total_estimated_annual_cost': round(total_monthly_cost * 12, 2),
                    'estimation_method': 'accurate' if can_spend_for_estimate else 'storage_only'
                },
                'pricing_info': pricing
            }
            
        except Exception as e:
            logger.error(f"Error getting logs by spend: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'log_groups': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    def _estimate_ingestion_from_metadata(self, log_group: Dict[str, Any], stored_gb: float) -> Dict[str, Any]:
        """
        Estimate monthly ingestion using only free metadata (retention policy and age).
        
        This is a FREE operation that uses only data from describe_log_groups.
        
        Estimation methods (in order of preference):
        1. Retention-based: If retention policy is set, assumes steady-state where
           storage = retention_days × daily_ingestion
        2. Age-based: Uses log group age to calculate average daily ingestion rate
        3. No estimate: If insufficient data
        
        Args:
            log_group: Log group metadata from describe_log_groups
            stored_gb: Current storage in GB
            
        Returns:
            Dict with monthly_ingestion_gb, estimation_method, and confidence level
        """
        retention_days = log_group.get('retentionInDays')
        creation_time = log_group.get('creationTime')
        
        # Method 1: Retention-based estimation (most reliable for logs with retention)
        if retention_days and retention_days > 0:
            # Steady-state assumption: storage = retention_days × daily_ingestion
            # Therefore: daily_ingestion = storage / retention_days
            daily_ingestion_gb = stored_gb / retention_days
            monthly_ingestion_gb = daily_ingestion_gb * 30
            
            return {
                'monthly_ingestion_gb': monthly_ingestion_gb,
                'estimation_method': 'retention_based_free',
                'confidence': 'medium'
            }
        
        # Method 2: Age-based estimation (for logs without retention)
        if creation_time:
            age_days = (datetime.now(timezone.utc) - 
                       datetime.fromtimestamp(creation_time / 1000, tz=timezone.utc)).days
            
            if age_days >= 30:  # Need at least 30 days for reasonable estimate
                # Average daily ingestion = total storage / age
                daily_ingestion_gb = stored_gb / age_days
                monthly_ingestion_gb = daily_ingestion_gb * 30
                
                return {
                    'monthly_ingestion_gb': monthly_ingestion_gb,
                    'estimation_method': 'age_based_free',
                    'confidence': 'low'
                }
            elif age_days > 0:
                # Too new for reliable estimate, but provide rough estimate
                daily_ingestion_gb = stored_gb / age_days
                monthly_ingestion_gb = daily_ingestion_gb * 30
                
                return {
                    'monthly_ingestion_gb': monthly_ingestion_gb,
                    'estimation_method': 'age_based_free_unreliable',
                    'confidence': 'very_low'
                }
        
        # Method 3: No estimate possible
        return {
            'monthly_ingestion_gb': 0,
            'estimation_method': 'storage_only',
            'confidence': 'high'  # High confidence in storage cost, no ingestion estimate
        }
    
    async def getMetrics(self, page: int = 1, namespace_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get custom metrics ordered by dimension count (descending), paginated.
        
        Args:
            page: Page number (1-based)
            namespace_filter: Optional filter for namespace
            
        Returns:
            Dict with custom_metrics, pagination, summary, and pricing_info
        """
        try:
            # Get metrics from DAO
            metrics_data = await self.dao.list_metrics(namespace=namespace_filter)
            all_metrics = metrics_data['metrics']
            
            # Filter to custom metrics only (exclude AWS/* namespaces)
            custom_metrics = [m for m in all_metrics if not m.get('Namespace', '').startswith('AWS/')]
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('metrics')
            free_tier = self.pricing_dao.get_free_tier_limits()
            
            # Calculate dimension count and cost for each metric
            metrics_with_info = []
            for metric in custom_metrics:
                dimensions = metric.get('Dimensions', [])
                dimension_count = len(dimensions)
                
                # Cost per metric
                metric_cost = pricing['custom_metrics_per_metric']
                
                metrics_with_info.append({
                    'namespace': metric.get('Namespace'),
                    'metric_name': metric.get('MetricName'),
                    'dimensions': dimensions,
                    'dimension_count': dimension_count,
                    'estimated_monthly_cost': round(metric_cost, 4),
                    'estimated_annual_cost': round(metric_cost * 12, 2)
                })
            
            # Sort by dimension count descending
            metrics_with_info.sort(key=lambda x: x['dimension_count'], reverse=True)
            
            # Paginate
            total_items = len(metrics_with_info)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_metrics = metrics_with_info[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_cost = total_items * pricing['custom_metrics_per_metric']
            billable_metrics = max(0, total_items - free_tier['metrics_count'])
            billable_cost = billable_metrics * pricing['custom_metrics_per_metric']
            
            return {
                'status': 'success',
                'custom_metrics': paginated_metrics,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_custom_metrics': total_items,
                    'free_tier_limit': free_tier['metrics_count'],
                    'free_tier_remaining': max(0, free_tier['metrics_count'] - total_items),
                    'billable_metrics': billable_metrics,
                    'total_estimated_monthly_cost': round(billable_cost, 4),
                    'total_estimated_annual_cost': round(billable_cost * 12, 2)
                },
                'pricing_info': pricing
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics by dimension count: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'custom_metrics': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def getDashboards(self, page: int = 1, dashboard_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dashboards ordered by complexity score (descending), paginated.
        
        Complexity score is calculated based on:
        - Widget count (each widget adds to complexity)
        - Custom metrics count (higher weight as they cost more)
        - Total metrics count (operational complexity)
        
        Args:
            page: Page number (1-based)
            dashboard_name_prefix: Optional filter for dashboard names
            
        Returns:
            Dict with dashboards, pagination, summary, and pricing_info
        """
        try:
            # Get dashboards from DAO
            dashboards_data = await self.dao.list_dashboards(dashboard_name_prefix=dashboard_name_prefix)
            dashboards = dashboards_data['dashboards']
            
            # Get pricing data
            dashboard_pricing = self.pricing_dao.get_pricing_data('dashboards')
            metrics_pricing = self.pricing_dao.get_pricing_data('metrics')
            free_tier = self.pricing_dao.get_free_tier_limits()
            
            # Analyze each dashboard
            dashboards_with_info = []
            for dashboard in dashboards:
                dashboard_name = dashboard.get('DashboardName')
                
                # Get dashboard configuration to analyze complexity
                custom_metrics_count = 0
                total_metrics_count = 0
                widget_count = 0
                
                try:
                    dashboard_config = await self.dao.get_dashboard(dashboard_name)
                    dashboard_body_str = dashboard_config.get('dashboard_body', '{}')
                    
                    # Defensive: ensure we never try to parse empty string
                    if not dashboard_body_str or dashboard_body_str.strip() == '':
                        dashboard_body_str = '{}'
                    
                    dashboard_body = json.loads(dashboard_body_str)
                    
                    # Analyze widgets and metrics
                    widgets = dashboard_body.get('widgets', [])
                    widget_count = len(widgets)
                    
                    for widget in widgets:
                        properties = widget.get('properties', {})
                        metrics = properties.get('metrics', [])
                        
                        # Count all metrics and custom metrics
                        for metric in metrics:
                            if isinstance(metric, list) and len(metric) > 0:
                                total_metrics_count += 1
                                namespace = metric[0] if isinstance(metric[0], str) else ''
                                if namespace and not namespace.startswith('AWS/'):
                                    custom_metrics_count += 1
                
                except Exception as e:
                    logger.warning(f"Could not analyze dashboard {dashboard_name}: {str(e)}")
                
                # Calculate complexity score
                # Formula: (custom_metrics * 3) + (total_metrics * 1) + (widgets * 2)
                # Custom metrics weighted higher due to cost impact
                complexity_score = (custom_metrics_count * 3) + (total_metrics_count * 1) + (widget_count * 2)
                
                # Calculate costs
                dashboard_cost = dashboard_pricing['dashboard_per_month']
                custom_metrics_cost = custom_metrics_count * metrics_pricing['custom_metrics_per_metric']
                total_cost = dashboard_cost + custom_metrics_cost
                
                dashboards_with_info.append({
                    'dashboard_name': dashboard_name,
                    'dashboard_arn': dashboard.get('DashboardArn'),
                    'last_modified': dashboard.get('LastModified'),
                    'size': dashboard.get('Size'),
                    'widget_count': widget_count,
                    'total_metrics_count': total_metrics_count,
                    'custom_metrics_count': custom_metrics_count,
                    'complexity_score': complexity_score,
                    'dashboard_cost': round(dashboard_cost, 4),
                    'custom_metrics_cost': round(custom_metrics_cost, 4),
                    'total_estimated_monthly_cost': round(total_cost, 4),
                    'estimated_annual_cost': round(total_cost * 12, 2)
                })
            
            # Sort by complexity score descending (combines cost and operational complexity)
            dashboards_with_info.sort(key=lambda x: x['complexity_score'], reverse=True)
            
            # Paginate
            total_items = len(dashboards_with_info)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_dashboards = dashboards_with_info[start_idx:end_idx]
            
            # Calculate totals
            billable_dashboards = max(0, total_items - free_tier['dashboards_count'])
            total_dashboard_cost = billable_dashboards * dashboard_pricing['dashboard_per_month']
            total_metrics_cost = sum(d['custom_metrics_cost'] for d in dashboards_with_info)
            total_monthly_cost = total_dashboard_cost + total_metrics_cost
            
            return {
                'status': 'success',
                'dashboards': paginated_dashboards,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_dashboards': total_items,
                    'free_tier_limit': free_tier['dashboards_count'],
                    'free_tier_remaining': max(0, free_tier['dashboards_count'] - total_items),
                    'billable_dashboards': billable_dashboards,
                    'total_dashboard_cost': round(total_dashboard_cost, 4),
                    'total_custom_metrics_cost': round(total_metrics_cost, 4),
                    'total_estimated_monthly_cost': round(total_monthly_cost, 4),
                    'total_estimated_annual_cost': round(total_monthly_cost * 12, 2)
                },
                'pricing_info': {
                    'dashboard': dashboard_pricing,
                    'metrics': metrics_pricing
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboards by custom metrics count: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'dashboards': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def getAlarms(self, page: int = 1, alarm_name_prefix: Optional[str] = None, 
                       state_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Get alarms with cost information, paginated.
        
        Args:
            page: Page number (1-based)
            alarm_name_prefix: Optional filter for alarm names
            state_value: Optional filter for alarm state (OK, ALARM, INSUFFICIENT_DATA)
            
        Returns:
            Dict with alarms, pagination, summary, and pricing_info
        """
        try:
            # Get alarms from DAO
            alarms_data = await self.dao.describe_alarms(
                alarm_name_prefix=alarm_name_prefix,
                state_value=state_value
            )
            alarms = alarms_data['alarms']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('alarms')
            free_tier = self.pricing_dao.get_free_tier_limits()
            
            # Analyze each alarm
            alarms_with_info = []
            standard_count = 0
            high_resolution_count = 0
            composite_count = 0
            
            for alarm in alarms:
                # Determine alarm type
                if 'MetricName' in alarm:
                    period = alarm.get('Period', 300)
                    if period < 300:
                        alarm_type = 'high_resolution'
                        alarm_cost = pricing['high_resolution_alarms_per_alarm']
                        high_resolution_count += 1
                    else:
                        alarm_type = 'standard'
                        alarm_cost = pricing['standard_alarms_per_alarm']
                        standard_count += 1
                else:
                    alarm_type = 'composite'
                    alarm_cost = pricing['composite_alarms_per_alarm']
                    composite_count += 1
                
                # Check if alarm has actions
                has_actions = bool(
                    alarm.get('AlarmActions') or 
                    alarm.get('OKActions') or 
                    alarm.get('InsufficientDataActions')
                )
                
                alarms_with_info.append({
                    'alarm_name': alarm.get('AlarmName'),
                    'alarm_arn': alarm.get('AlarmArn'),
                    'alarm_type': alarm_type,
                    'state_value': alarm.get('StateValue'),
                    'state_reason': alarm.get('StateReason'),
                    'metric_name': alarm.get('MetricName'),
                    'namespace': alarm.get('Namespace'),
                    'period': alarm.get('Period'),
                    'has_actions': has_actions,
                    'actions_enabled': alarm.get('ActionsEnabled', False),
                    'estimated_monthly_cost': round(alarm_cost, 4),
                    'estimated_annual_cost': round(alarm_cost * 12, 2)
                })
            
            # Paginate
            total_items = len(alarms_with_info)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_alarms = alarms_with_info[start_idx:end_idx]
            
            # Calculate totals
            billable_standard = max(0, standard_count - free_tier['alarms_count'])
            total_monthly_cost = (
                billable_standard * pricing['standard_alarms_per_alarm'] +
                high_resolution_count * pricing['high_resolution_alarms_per_alarm'] +
                composite_count * pricing['composite_alarms_per_alarm']
            )
            
            return {
                'status': 'success',
                'alarms': paginated_alarms,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_alarms': total_items,
                    'standard_alarms': standard_count,
                    'high_resolution_alarms': high_resolution_count,
                    'composite_alarms': composite_count,
                    'free_tier_limit': free_tier['alarms_count'],
                    'free_tier_remaining': max(0, free_tier['alarms_count'] - standard_count),
                    'billable_standard_alarms': billable_standard,
                    'total_estimated_monthly_cost': round(total_monthly_cost, 4),
                    'total_estimated_annual_cost': round(total_monthly_cost * 12, 2)
                },
                'pricing_info': pricing
            }
            
        except Exception as e:
            logger.error(f"Error getting alarms: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'alarms': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }


class CWMetricsTips:
    """CloudWatch custom metrics optimization and analysis."""
    
    def __init__(self, dao: CloudWatchDAO, pricing_dao: AWSPricingDAO, cost_preferences: CostPreferences):
        self.dao = dao
        self.pricing_dao = pricing_dao
        self.cost_preferences = cost_preferences
        self._page_size = 10  # Items per page
    
    async def list_metrics(self, namespace: Optional[str] = None,
                          metric_name: Optional[str] = None,
                          **kwargs) -> CloudWatchOperationResult:
        """List CloudWatch metrics (FREE operation)."""
        try:
            data = await self.dao.list_metrics(
                namespace=namespace,
                metric_name=metric_name
            )
            return CloudWatchOperationResult(
                success=True,
                data=data,
                operation_name="list_metrics",
                cost_incurred=False,
                operation_type="free",
                primary_data_source="cloudwatch_config"
            )
        except Exception as e:
            logger.error(f"Error listing metrics: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="list_metrics"
            )
    
    async def get_targeted_metric_statistics(self, namespace: str,
                                            metric_name: str,
                                            **kwargs) -> CloudWatchOperationResult:
        """Get metric statistics (MINIMAL COST operation)."""
        try:
            # This requires GetMetricStatistics which has minimal cost
            # Return placeholder indicating this needs metrics API
            return CloudWatchOperationResult(
                success=True,
                data={'metrics': [], 'requires_get_metric_statistics': True},
                operation_name="get_targeted_metric_statistics",
                cost_incurred=False,
                operation_type="minimal_cost",
                primary_data_source="cloudwatch_metrics",
                fallback_used=True
            )
        except Exception as e:
            logger.error(f"Error getting metric statistics: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="get_targeted_metric_statistics"
            )
    
    async def listInstancesWithDetailedMonitoring(self, page: int = 1) -> Dict[str, Any]:
        """
        Get paginated list of EC2 instances with detailed monitoring enabled.
        
        Detailed monitoring costs $2.10/month per instance vs basic monitoring (free).
        
        Args:
            page: Page number (1-based)
            
        Returns:
            Dict with instances, pagination, summary, and cost_analysis
        """
        try:
            # Initialize EC2 client
            ec2_client = boto3.client('ec2', region_name=self.dao.region)
            
            # Get all instances
            instances_with_detailed = []
            paginator = ec2_client.get_paginator('describe_instances')
            
            for page_response in paginator.paginate():
                for reservation in page_response.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        # Check if detailed monitoring is enabled
                        monitoring_state = instance.get('Monitoring', {}).get('State', 'disabled')
                        
                        if monitoring_state == 'enabled':
                            instance_id = instance.get('InstanceId')
                            instance_type = instance.get('InstanceType')
                            state = instance.get('State', {}).get('Name')
                            
                            # Get instance name from tags
                            instance_name = None
                            for tag in instance.get('Tags', []):
                                if tag.get('Key') == 'Name':
                                    instance_name = tag.get('Value')
                                    break
                            
                            instances_with_detailed.append({
                                'instance_id': instance_id,
                                'instance_name': instance_name,
                                'instance_type': instance_type,
                                'state': state,
                                'monitoring_state': monitoring_state,
                                'launch_time': instance.get('LaunchTime').isoformat() if instance.get('LaunchTime') else None
                            })
            
            # Get pricing
            pricing = self.pricing_dao.get_pricing_data('metrics')
            detailed_monitoring_cost = pricing['detailed_monitoring_per_instance']
            
            # Add cost information to each instance
            for instance in instances_with_detailed:
                instance['monthly_cost'] = round(detailed_monitoring_cost, 2)
                instance['annual_cost'] = round(detailed_monitoring_cost * 12, 2)
            
            # Paginate
            total_items = len(instances_with_detailed)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_instances = instances_with_detailed[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_cost = total_items * detailed_monitoring_cost
            
            return {
                'status': 'success',
                'instances': paginated_instances,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_instances_with_detailed_monitoring': total_items,
                    'cost_per_instance_monthly': round(detailed_monitoring_cost, 2),
                    'total_monthly_cost': round(total_monthly_cost, 2),
                    'total_annual_cost': round(total_monthly_cost * 12, 2)
                },
                'optimization_tip': {
                    'message': 'Consider disabling detailed monitoring for instances that do not require 1-minute metrics',
                    'potential_savings_monthly': round(total_monthly_cost, 2),
                    'potential_savings_annual': round(total_monthly_cost * 12, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing instances with detailed monitoring: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'instances': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def listCustomMetrics(self, page: int = 1, namespace_filter: Optional[str] = None,
                               can_spend_for_exact_usage_estimate: bool = False,
                               lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get paginated list of custom metrics sorted by dimension count (descending).
        
        Calls list_metrics twice:
        1. Without RecentlyActive filter to get all metrics
        2. With RecentlyActive='PT3H' to flag recently active metrics
        
        If can_spend_for_exact_usage_estimate=True, calls GetMetricData (batched) for the 
        current page only to identify actual usage.
        
        Args:
            page: Page number (1-based)
            namespace_filter: Optional filter for namespace
            can_spend_for_exact_usage_estimate: If True, uses GetMetricData API (paid) to get 
                                               exact usage for current page only. If False (default), 
                                               only shows dimension count and recently active flag.
            lookback_days: Number of days to analyze for usage (only used if can_spend_for_exact_usage_estimate=True)
            
        Returns:
            Dict with custom_metrics, pagination, summary, and pricing_info
        """
        try:
            # FREE TIER: Get only first AWS API page (~500 metrics)
            all_metrics_params = {}
            if namespace_filter:
                all_metrics_params['Namespace'] = namespace_filter
            
            all_metrics = []
            paginator = self.dao.cloudwatch_client.get_paginator('list_metrics')
            
            # FREE TIER: Only scan 1 page for fast, free response
            page_count = 0
            max_pages = 1
            for page_response in paginator.paginate(**all_metrics_params):
                all_metrics.extend(page_response.get('Metrics', []))
                page_count += 1
                if page_count >= max_pages:
                    logger.info(f"FREE TIER: Scanned {page_count} AWS page, {len(all_metrics)} total metrics")
                    break
            
            # Filter to custom metrics only (exclude AWS/* namespaces - those are free anyway)
            custom_metrics = [m for m in all_metrics if not m.get('Namespace', '').startswith('AWS/')]
            
            # FREE TIER: Get recently active metrics (PT3H = Past 3 Hours) - also limit to 1 page
            recently_active_params = {'RecentlyActive': 'PT3H'}
            if namespace_filter:
                recently_active_params['Namespace'] = namespace_filter
            
            recently_active_metrics = []
            page_count = 0
            for page_response in paginator.paginate(**recently_active_params):
                recently_active_metrics.extend(page_response.get('Metrics', []))
                page_count += 1
                if page_count >= max_pages:
                    logger.info(f"FREE TIER: Scanned {page_count} recently active page")
                    break
            
            # Create set of recently active metric identifiers for fast lookup
            recently_active_set = set()
            for metric in recently_active_metrics:
                metric_id = self._get_metric_identifier(metric)
                recently_active_set.add(metric_id)
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('metrics')
            metric_cost = pricing['custom_metrics_per_metric']
            
            # Process each custom metric (FREE)
            metrics_with_info = []
            for metric in custom_metrics:
                namespace = metric.get('Namespace')
                metric_name = metric.get('MetricName')
                dimensions = metric.get('Dimensions', [])
                dimension_count = len(dimensions)
                
                # Check if recently active
                metric_id = self._get_metric_identifier(metric)
                is_recently_active = metric_id in recently_active_set
                
                metric_info = {
                    'namespace': namespace,
                    'metric_name': metric_name,
                    'dimensions': dimensions,
                    'dimension_count': dimension_count,
                    'recently_active': is_recently_active,
                    'estimated_monthly_cost': round(metric_cost, 4),
                    'estimated_annual_cost': round(metric_cost * 12, 2),
                    'usage_estimation_method': 'dimension_count_free'
                }
                
                metrics_with_info.append(metric_info)
            
            # Sort by dimension count (FREE)
            metrics_with_info.sort(key=lambda x: x['dimension_count'], reverse=True)
            
            # Paginate (FREE)
            total_items = len(metrics_with_info)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_metrics = metrics_with_info[start_idx:end_idx]
            
            # PAID OPERATION: Analyze current page only (if requested)
            if can_spend_for_exact_usage_estimate and paginated_metrics:
                # Use DAO method for batched analysis with caching
                await self.dao.get_metrics_usage_batch(
                    paginated_metrics,
                    lookback_days=lookback_days
                )
                
                # Note: DAO updates metrics in place with datapoint_count and usage_estimation_method
            
            # Calculate totals
            free_tier = self.pricing_dao.get_free_tier_limits()
            billable_metrics = max(0, total_items - free_tier['metrics_count'])
            total_monthly_cost = billable_metrics * metric_cost
            recently_active_count = sum(1 for m in metrics_with_info if m['recently_active'])
            
            # Detect if we likely have many more metrics (FREE TIER shows sample only)
            # If we got 500 metrics in 1 page and many are custom, there are likely more
            likely_more_metrics = len(all_metrics) >= 400 and total_items >= 300
            
            # Build recommendations
            recommendations = []
            if likely_more_metrics:
                # Estimate total metrics based on sample
                estimated_total_custom = total_items * 150  # Conservative estimate
                estimated_cost_explorer_calls = estimated_total_custom / 500  # ~500 metrics per API page
                estimated_cost_explorer_cost = estimated_cost_explorer_calls * 0.01  # $0.01 per 1000 API calls
                
                recommendations.append({
                    'type': 'enable_cost_explorer_analysis',
                    'priority': 'high',
                    'message': f'FREE TIER shows sample of {total_items} custom metrics from first AWS API page. You likely have many more metrics.',
                    'action': 'Enable Cost Explorer analysis for comprehensive metrics cost analysis across all metrics',
                    'estimated_cost': round(estimated_cost_explorer_cost, 2),
                    'estimated_api_calls': int(estimated_cost_explorer_calls),
                    'how_to_enable': 'Set allow_cost_explorer=True in cost preferences',
                    'benefit': 'Get accurate cost data and usage patterns for ALL custom metrics, not just first page sample'
                })
            
            result = {
                'status': 'success',
                'data_source': 'free_tier_sample' if not can_spend_for_exact_usage_estimate else 'free_tier_with_paid_usage',
                'custom_metrics': paginated_metrics,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1,
                    'note': 'FREE TIER: Showing sample from first AWS API page only' if likely_more_metrics else 'Showing all custom metrics found'
                },
                'summary': {
                    'total_custom_metrics_in_sample': total_items,
                    'recently_active_metrics': recently_active_count,
                    'inactive_metrics': total_items - recently_active_count,
                    'free_tier_limit': free_tier['metrics_count'],
                    'free_tier_remaining': max(0, free_tier['metrics_count'] - total_items),
                    'billable_metrics_in_sample': billable_metrics,
                    'estimated_monthly_cost_for_sample': round(total_monthly_cost, 4),
                    'estimated_annual_cost_for_sample': round(total_monthly_cost * 12, 2),
                    'usage_estimation_method': 'exact_paid_page_only' if can_spend_for_exact_usage_estimate else 'dimension_count_free',
                    'metrics_analyzed_for_usage': len(paginated_metrics) if can_spend_for_exact_usage_estimate else 0,
                    'sample_note': 'FREE TIER: Costs shown are for sample only. Enable Cost Explorer for complete analysis.' if likely_more_metrics else None
                },
                'pricing_info': pricing,
                'optimization_tip': {
                    'message': f'Found {total_items - recently_active_count} metrics not active in past 3 hours in this sample. Consider removing unused metrics.',
                    'potential_savings_monthly': round((total_items - recently_active_count) * metric_cost, 2),
                    'potential_savings_annual': round((total_items - recently_active_count) * metric_cost * 12, 2)
                }
            }
            
            # Add recommendations if any
            if recommendations:
                result['recommendations'] = recommendations
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing custom metrics: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'custom_metrics': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    def _get_metric_identifier(self, metric: Dict[str, Any]) -> str:
        """
        Create a unique identifier for a metric based on namespace, name, and dimensions.
        
        Args:
            metric: Metric dictionary from list_metrics
            
        Returns:
            String identifier for the metric
        """
        namespace = metric.get('Namespace', '')
        metric_name = metric.get('MetricName', '')
        dimensions = metric.get('Dimensions', [])
        
        # Sort dimensions for consistent comparison
        sorted_dims = sorted(dimensions, key=lambda d: d.get('Name', ''))
        dim_str = '|'.join(f"{d.get('Name')}={d.get('Value')}" for d in sorted_dims)
        
        return f"{namespace}::{metric_name}::{dim_str}"
    
    async def analyze_metrics_usage(self, namespace_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze custom metrics usage and optimization opportunities.
        
        FREE TIER: Uses first AWS API page only for fast response.
        """
        try:
            # FREE TIER: Get only page 1 for fast response
            metrics_data = await self.dao.list_metrics(namespace=namespace_filter, page=1)
            
            # Categorize metrics
            aws_metrics = [m for m in metrics_data['metrics'] 
                          if m.get('Namespace', '').startswith('AWS/')]
            custom_metrics = [m for m in metrics_data['metrics'] 
                             if not m.get('Namespace', '').startswith('AWS/')]
            
            # Analyze by namespace
            custom_by_namespace = {}
            for metric in custom_metrics:
                namespace = metric.get('Namespace', 'Unknown')
                if namespace not in custom_by_namespace:
                    custom_by_namespace[namespace] = []
                custom_by_namespace[namespace].append(metric)
            
            # Calculate costs
            metrics_cost = self.pricing_dao.calculate_cost('metrics', {
                'custom_metrics_count': len(custom_metrics),
                'api_requests_count': 100000,
                'detailed_monitoring_instances': 0
            })
            
            return {
                'status': 'success',
                'metrics_summary': {
                    'total_metrics': len(metrics_data['metrics']),
                    'aws_metrics': len(aws_metrics),
                    'custom_metrics': len(custom_metrics),
                    'custom_by_namespace': {ns: len(metrics) for ns, metrics in custom_by_namespace.items()}
                },
                'cost_analysis': metrics_cost,
                'detailed_metrics': {
                    'custom_metrics': custom_metrics[:50],
                    'aws_metrics_sample': aws_metrics[:20]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing metrics usage: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class CWLogsTips:
    """CloudWatch Logs optimization and analysis."""
    
    def __init__(self, dao: CloudWatchDAO, pricing_dao: AWSPricingDAO, cost_preferences: CostPreferences):
        self.dao = dao
        self.pricing_dao = pricing_dao
        self.cost_preferences = cost_preferences
        self._page_size = 10  # Items per page
        
        # Import and initialize VendedLogsDAO
        from services.cloudwatch_service_vended_log import VendedLogsDAO
        self._vended_logs_dao = VendedLogsDAO(region=dao.region)
    
    async def describe_log_groups(self, log_group_names: Optional[List[str]] = None,
                                  log_group_name_prefix: Optional[str] = None,
                                  **kwargs) -> CloudWatchOperationResult:
        """Get log groups configuration (FREE operation)."""
        try:
            data = await self.dao.describe_log_groups(
                log_group_name_prefix=log_group_name_prefix,
                log_group_names=log_group_names
            )
            return CloudWatchOperationResult(
                success=True,
                data=data,
                operation_name="describe_log_groups",
                cost_incurred=False,
                operation_type="free",
                primary_data_source="cloudwatch_logs_config"
            )
        except Exception as e:
            logger.error(f"Error describing log groups: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="describe_log_groups"
            )
    
    async def get_log_group_incoming_bytes(self, log_group_names: Optional[List[str]] = None,
                                          lookback_days: int = 30,
                                          **kwargs) -> CloudWatchOperationResult:
        """Get log group ingestion metrics (MINIMAL COST operation)."""
        try:
            # This would require GetMetricStatistics which has minimal cost
            # For now, return a placeholder that indicates this needs Cost Explorer or metrics
            return CloudWatchOperationResult(
                success=True,
                data={'log_groups': [], 'requires_metrics': True},
                operation_name="get_log_group_incoming_bytes",
                cost_incurred=False,
                operation_type="minimal_cost",
                primary_data_source="cloudwatch_metrics",
                fallback_used=True
            )
        except Exception as e:
            logger.error(f"Error getting log group incoming bytes: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="get_log_group_incoming_bytes"
            )
    
    async def analyze_logs_usage(self, log_group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze CloudWatch Logs usage and optimization opportunities."""
        try:
            log_groups_data = await self.dao.describe_log_groups(log_group_names=log_group_names)
            log_groups = log_groups_data['log_groups']
            
            # Analyze retention
            without_retention = []
            long_retention = []
            
            for lg in log_groups:
                retention_days = lg.get('retentionInDays')
                if not retention_days:
                    without_retention.append(lg.get('logGroupName'))
                elif retention_days > 365:
                    long_retention.append({
                        'name': lg.get('logGroupName'),
                        'retention_days': retention_days
                    })
            
            # Calculate costs
            total_stored_gb = sum(lg.get('storedBytes', 0) for lg in log_groups) / (1024**3)
            logs_cost = self.pricing_dao.calculate_cost('logs', {
                'ingestion_gb': total_stored_gb * 0.1,
                'storage_gb': total_stored_gb,
                'insights_gb_scanned': 0
            })
            
            return {
                'status': 'success',
                'log_groups_summary': {
                    'total_log_groups': len(log_groups),
                    'total_stored_gb': total_stored_gb,
                    'without_retention_policy': len(without_retention),
                    'long_retention_groups': len(long_retention)
                },
                'cost_analysis': logs_cost,
                'log_groups_details': log_groups[:50]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logs usage: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def listLogsWithoutRetention(self, page: int = 1, 
                                      log_group_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated list of log groups without retention policy, sorted by storage size descending.
        
        Log groups without retention policies store logs indefinitely, leading to unbounded costs.
        This method identifies these log groups to help set appropriate retention policies.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            
        Returns:
            Dict with log_groups, pagination, summary, and optimization recommendations
        """
        try:
            # Get all log groups
            log_groups_data = await self.dao.describe_log_groups(
                log_group_name_prefix=log_group_name_prefix
            )
            log_groups = log_groups_data['log_groups']
            
            # Filter to log groups without retention
            logs_without_retention = []
            for lg in log_groups:
                retention_days = lg.get('retentionInDays')
                if not retention_days:  # None or 0 means never expire
                    stored_bytes = lg.get('storedBytes', 0)
                    stored_gb = stored_bytes / (1024**3)
                    
                    # Get pricing
                    pricing = self.pricing_dao.get_pricing_data('logs')
                    storage_cost = stored_gb * pricing['storage_per_gb_month']
                    
                    logs_without_retention.append({
                        'log_group_name': lg.get('logGroupName'),
                        'stored_gb': round(stored_gb, 4),
                        'stored_bytes': stored_bytes,
                        'creation_time': lg.get('creationTime'),
                        'retention_days': 'Never Expire',
                        'monthly_storage_cost': round(storage_cost, 4),
                        'annual_storage_cost': round(storage_cost * 12, 2),
                        'log_group_class': lg.get('logGroupClass', 'STANDARD')
                    })
            
            # Sort by storage size descending
            logs_without_retention.sort(key=lambda x: x['stored_gb'], reverse=True)
            
            # Paginate
            total_items = len(logs_without_retention)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_logs = logs_without_retention[start_idx:end_idx]
            
            # Calculate totals
            total_storage_gb = sum(lg['stored_gb'] for lg in logs_without_retention)
            total_monthly_cost = sum(lg['monthly_storage_cost'] for lg in logs_without_retention)
            
            return {
                'status': 'success',
                'log_groups': paginated_logs,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_log_groups_without_retention': total_items,
                    'total_storage_gb': round(total_storage_gb, 4),
                    'total_monthly_cost': round(total_monthly_cost, 4),
                    'total_annual_cost': round(total_monthly_cost * 12, 2)
                },
                'optimization_recommendations': {
                    'message': 'Set retention policies to automatically delete old logs and control costs',
                    'recommended_retention_days': [7, 14, 30, 60, 90, 120, 180, 365, 400, 545, 731, 1827, 3653],
                    'common_retention_policies': {
                        'development_logs': '7-14 days',
                        'application_logs': '30-90 days',
                        'audit_logs': '365-3653 days (1-10 years)',
                        'compliance_logs': '2557-3653 days (7-10 years)'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing logs without retention: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'log_groups': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def listVendedLogTargets(self, page: int = 1, 
                                   log_group_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated list of log groups that can be vended directly to S3 to reduce costs.
        
        Vended logs bypass CloudWatch Logs ingestion and storage, significantly reducing costs
        for high-volume service logs like VPC Flow Logs, ELB Access Logs, etc.
        
        This is a complex retrieval task delegated to VendedLogsDAO.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            
        Returns:
            Dict with vended_log_opportunities, pagination, summary, and implementation guidance
        """
        try:
            # Get all log groups
            log_groups_data = await self.dao.describe_log_groups(
                log_group_name_prefix=log_group_name_prefix
            )
            log_groups = log_groups_data['log_groups']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('logs')
            
            # Analyze vended log opportunities using VendedLogsDAO
            opportunities = await self._vended_logs_dao.analyze_vended_log_opportunities(
                log_groups, pricing
            )
            
            # Sort by monthly savings descending (primary sort key for best ROI)
            # This ensures highest-value opportunities appear first
            opportunities.sort(key=lambda x: x['savings']['monthly_savings'], reverse=True)
            
            # Paginate
            total_items = len(opportunities)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_opportunities = opportunities[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_savings = sum(opp['savings']['monthly_savings'] for opp in opportunities)
            total_annual_savings = sum(opp['savings']['annual_savings'] for opp in opportunities)
            total_current_cost = sum(opp['current_costs']['total_monthly'] for opp in opportunities)
            total_vended_cost = sum(opp['vended_costs']['total_monthly'] for opp in opportunities)
            
            # Group by service type
            by_service = {}
            for opp in opportunities:
                service = opp['service']
                if service not in by_service:
                    by_service[service] = {
                        'count': 0,
                        'monthly_savings': 0,
                        'annual_savings': 0
                    }
                by_service[service]['count'] += 1
                by_service[service]['monthly_savings'] += opp['savings']['monthly_savings']
                by_service[service]['annual_savings'] += opp['savings']['annual_savings']
            
            return {
                'status': 'success',
                'vended_log_opportunities': paginated_opportunities,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_vended_log_opportunities': total_items,
                    'total_current_monthly_cost': round(total_current_cost, 4),
                    'total_vended_monthly_cost': round(total_vended_cost, 4),
                    'total_monthly_savings': round(total_monthly_savings, 4),
                    'total_annual_savings': round(total_annual_savings, 2),
                    'average_cost_reduction_percentage': round(
                        (total_monthly_savings / total_current_cost * 100) if total_current_cost > 0 else 0, 1
                    ),
                    'opportunities_by_service': by_service
                },
                'implementation_guidance': {
                    'overview': 'Vended logs bypass CloudWatch Logs and go directly to S3, reducing costs by 60-95%',
                    'benefits': [
                        'Significant cost reduction (60-95% savings)',
                        'No CloudWatch Logs ingestion charges',
                        'Lower storage costs with S3',
                        'Can use S3 lifecycle policies for further cost optimization',
                        'Maintain compliance and audit requirements'
                    ],
                    'considerations': [
                        'Logs will no longer be searchable in CloudWatch Logs Insights',
                        'Need to use S3 Select, Athena, or other tools for log analysis',
                        'May require changes to existing log processing workflows',
                        'Some services require recreation of logging configuration'
                    ],
                    'next_steps': [
                        '1. Review vended log opportunities sorted by savings',
                        '2. Check implementation complexity for each service',
                        '3. Follow documentation links for specific setup instructions',
                        '4. Test with non-critical log groups first',
                        '5. Update monitoring and alerting to use S3-based logs'
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing vended log targets: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'vended_log_opportunities': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def listInfrequentAccessTargets(self, page: int = 1,
                                         log_group_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated list of log groups sorted by retention * storage bytes descending,
        excluding those already using INFREQUENT_ACCESS log group class.
        
        INFREQUENT_ACCESS class reduces storage costs by 50% for logs with retention >= 30 days.
        This method identifies the best candidates for cost savings.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            
        Returns:
            Dict with log_groups, pagination, summary, and optimization recommendations
        """
        try:
            # Get all log groups
            log_groups_data = await self.dao.describe_log_groups(
                log_group_name_prefix=log_group_name_prefix
            )
            log_groups = log_groups_data['log_groups']
            
            # Get pricing
            pricing = self.pricing_dao.get_pricing_data('logs')
            standard_storage_cost = pricing['storage_per_gb_month']
            infrequent_storage_cost = standard_storage_cost * 0.5  # 50% discount
            
            # Filter and analyze candidates
            infrequent_access_candidates = []
            for lg in log_groups:
                log_group_class = lg.get('logGroupClass', 'STANDARD')
                retention_days = lg.get('retentionInDays', 0)
                stored_bytes = lg.get('storedBytes', 0)
                
                # Skip if already using INFREQUENT_ACCESS
                if log_group_class == 'INFREQUENT_ACCESS':
                    continue
                
                # Only consider log groups with retention >= 30 days
                # (INFREQUENT_ACCESS requires minimum 30-day retention)
                if retention_days and retention_days >= 30:
                    stored_gb = stored_bytes / (1024**3)
                    
                    # Calculate priority score: retention * storage
                    # Higher score = more data stored for longer = more savings potential
                    priority_score = retention_days * stored_bytes
                    
                    # Calculate current and potential costs
                    current_monthly_cost = stored_gb * standard_storage_cost
                    infrequent_monthly_cost = stored_gb * infrequent_storage_cost
                    monthly_savings = current_monthly_cost - infrequent_monthly_cost
                    
                    infrequent_access_candidates.append({
                        'log_group_name': lg.get('logGroupName'),
                        'stored_gb': round(stored_gb, 4),
                        'stored_bytes': stored_bytes,
                        'retention_days': retention_days,
                        'current_log_group_class': log_group_class,
                        'priority_score': priority_score,
                        'current_monthly_cost': round(current_monthly_cost, 4),
                        'infrequent_access_monthly_cost': round(infrequent_monthly_cost, 4),
                        'monthly_savings': round(monthly_savings, 4),
                        'annual_savings': round(monthly_savings * 12, 2),
                        'savings_percentage': 50.0
                    })
            
            # Sort by priority score descending (retention * storage bytes)
            infrequent_access_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Paginate
            total_items = len(infrequent_access_candidates)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_candidates = infrequent_access_candidates[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_savings = sum(c['monthly_savings'] for c in infrequent_access_candidates)
            total_annual_savings = sum(c['annual_savings'] for c in infrequent_access_candidates)
            total_storage_gb = sum(c['stored_gb'] for c in infrequent_access_candidates)
            
            return {
                'status': 'success',
                'log_groups': paginated_candidates,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_infrequent_access_candidates': total_items,
                    'total_storage_gb': round(total_storage_gb, 4),
                    'total_monthly_savings_potential': round(total_monthly_savings, 4),
                    'total_annual_savings_potential': round(total_annual_savings, 2),
                    'average_savings_percentage': 50.0
                },
                'optimization_recommendations': {
                    'message': 'Switch to INFREQUENT_ACCESS class for 50% storage cost reduction',
                    'requirements': [
                        'Log group must have retention policy of 30 days or more',
                        'Best for logs that are rarely queried',
                        'Higher query costs (CloudWatch Logs Insights charges apply)',
                        'Ideal for compliance/audit logs with long retention'
                    ],
                    'implementation': {
                        'method': 'Update log group class using AWS CLI or Console',
                        'cli_command': 'aws logs put-log-group-policy --log-group-name <name> --log-group-class INFREQUENT_ACCESS',
                        'reversible': True,
                        'downtime': 'None - change is immediate'
                    },
                    'cost_tradeoffs': {
                        'storage_savings': '50% reduction in storage costs',
                        'query_costs': 'CloudWatch Logs Insights queries cost $0.005/GB scanned',
                        'recommendation': 'Best for logs queried less than 10 times per month'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing infrequent access targets: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'log_groups': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def listLogAnomalies(self, page: int = 1,
                              log_group_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated list of log anomalies using ListLogAnomalyDetectors and ListAnomalies.
        
        Identifies log groups with anomaly detection enabled and lists detected anomalies
        to help identify unusual patterns, security issues, or operational problems.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            
        Returns:
            Dict with anomalies, pagination, summary, and analysis
        """
        try:
            # List all anomaly detectors
            anomaly_detectors = []
            paginator = self.dao.logs_client.get_paginator('list_log_anomaly_detectors')
            
            for page_response in paginator.paginate():
                anomaly_detectors.extend(page_response.get('anomalyDetectors', []))
            
            # Filter by prefix if provided
            if log_group_name_prefix:
                anomaly_detectors = [
                    ad for ad in anomaly_detectors
                    if ad.get('logGroupArnList', []) and 
                    any(log_group_name_prefix in arn for arn in ad.get('logGroupArnList', []))
                ]
            
            # Get anomalies for each detector
            all_anomalies = []
            for detector in anomaly_detectors:
                detector_arn = detector.get('anomalyDetectorArn')
                log_group_arns = detector.get('logGroupArnList', [])
                
                try:
                    # List anomalies for this detector
                    anomalies_response = self.dao.logs_client.list_anomalies(
                        anomalyDetectorArn=detector_arn
                    )
                    
                    anomalies = anomalies_response.get('anomalies', [])
                    
                    for anomaly in anomalies:
                        # Extract log group name from ARN
                        log_group_names = []
                        for arn in log_group_arns:
                            # ARN format: arn:aws:logs:region:account:log-group:log-group-name:*
                            parts = arn.split(':')
                            if len(parts) >= 7:
                                log_group_names.append(':'.join(parts[6:-1]))
                        
                        all_anomalies.append({
                            'anomaly_id': anomaly.get('anomalyId'),
                            'detector_arn': detector_arn,
                            'log_group_names': log_group_names,
                            'first_seen': anomaly.get('firstSeen'),
                            'last_seen': anomaly.get('lastSeen'),
                            'description': anomaly.get('description'),
                            'pattern': anomaly.get('patternString'),
                            'priority': anomaly.get('priority'),
                            'state': anomaly.get('state'),
                            'is_pattern_level_suppression': anomaly.get('isPatternLevelSuppression', False),
                            'detector_evaluation_frequency': detector.get('evaluationFrequency', 'UNKNOWN'),
                            'detector_status': detector.get('anomalyDetectorStatus')
                        })
                
                except Exception as e:
                    logger.warning(f"Could not list anomalies for detector {detector_arn}: {str(e)}")
                    continue
            
            # Sort by last_seen descending (most recent first)
            all_anomalies.sort(key=lambda x: x.get('last_seen', 0), reverse=True)
            
            # Paginate
            total_items = len(all_anomalies)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_anomalies = all_anomalies[start_idx:end_idx]
            
            # Analyze anomaly patterns
            by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
            by_state = {}
            for anomaly in all_anomalies:
                priority = anomaly.get('priority', 'UNKNOWN')
                state = anomaly.get('state', 'UNKNOWN')
                by_priority[priority] = by_priority.get(priority, 0) + 1
                by_state[state] = by_state.get(state, 0) + 1
            
            return {
                'status': 'success',
                'anomalies': paginated_anomalies,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_anomalies': total_items,
                    'total_detectors': len(anomaly_detectors),
                    'anomalies_by_priority': by_priority,
                    'anomalies_by_state': by_state
                },
                'analysis': {
                    'message': 'Log anomalies indicate unusual patterns in your logs',
                    'high_priority_count': by_priority.get('HIGH', 0),
                    'recommendations': [
                        'Investigate high-priority anomalies first',
                        'Review anomaly patterns for security issues',
                        'Consider creating alarms for recurring anomalies',
                        'Update application code to address root causes'
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing log anomalies: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'anomalies': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }
    
    async def listIneffectiveLogAnomalies(self, page: int = 1,
                                         log_group_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Get paginated list of ineffective log anomaly detectors that have found no anomalies
        and could be disabled to reduce costs.
        
        Only includes detectors that have been active for at least half of their anomalyVisibilityTime
        to prevent flagging newly created detectors.
        
        Args:
            page: Page number (1-based)
            log_group_name_prefix: Optional filter for log group names
            
        Returns:
            Dict with ineffective_detectors, pagination, summary, and cost savings
        """
        try:
            # List all anomaly detectors
            anomaly_detectors = []
            paginator = self.dao.logs_client.get_paginator('list_log_anomaly_detectors')
            
            for page_response in paginator.paginate():
                anomaly_detectors.extend(page_response.get('anomalyDetectors', []))
            
            # Filter by prefix if provided
            if log_group_name_prefix:
                anomaly_detectors = [
                    ad for ad in anomaly_detectors
                    if ad.get('logGroupArnList', []) and 
                    any(log_group_name_prefix in arn for arn in ad.get('logGroupArnList', []))
                ]
            
            # Analyze each detector for effectiveness
            ineffective_detectors = []
            current_time = datetime.now(timezone.utc)
            
            for detector in anomaly_detectors:
                detector_arn = detector.get('anomalyDetectorArn')
                log_group_arns = detector.get('logGroupArnList', [])
                creation_time = detector.get('creationTimeStamp')
                anomaly_visibility_time = detector.get('anomalyVisibilityTime', 7)  # Default 7 days
                
                # Calculate detector age
                if creation_time:
                    if isinstance(creation_time, (int, float)):
                        creation_datetime = datetime.fromtimestamp(creation_time / 1000, tz=timezone.utc)
                    else:
                        creation_datetime = creation_time
                    
                    detector_age_days = (current_time - creation_datetime).days
                    
                    # Only consider detectors active for at least half of anomalyVisibilityTime
                    min_age_days = anomaly_visibility_time / 2
                    
                    if detector_age_days < min_age_days:
                        continue  # Skip new detectors
                    
                    try:
                        # Check if detector has found any anomalies
                        anomalies_response = self.dao.logs_client.list_anomalies(
                            anomalyDetectorArn=detector_arn
                        )
                        
                        anomaly_count = len(anomalies_response.get('anomalies', []))
                        
                        # If no anomalies found, this detector is ineffective
                        if anomaly_count == 0:
                            # Extract log group names from ARNs
                            log_group_names = []
                            for arn in log_group_arns:
                                parts = arn.split(':')
                                if len(parts) >= 7:
                                    log_group_names.append(':'.join(parts[6:-1]))
                            
                            # Estimate cost (anomaly detection is included in CloudWatch Logs costs,
                            # but disabling unused detectors reduces processing overhead)
                            # Approximate cost: $0.01 per GB of logs analyzed
                            estimated_monthly_cost = 0.50  # Conservative estimate per detector
                            
                            ineffective_detectors.append({
                                'detector_arn': detector_arn,
                                'log_group_names': log_group_names,
                                'log_group_count': len(log_group_names),
                                'creation_time': creation_time,
                                'detector_age_days': detector_age_days,
                                'anomaly_visibility_time_days': anomaly_visibility_time,
                                'anomalies_found': 0,
                                'evaluation_frequency': detector.get('evaluationFrequency', 'UNKNOWN'),
                                'detector_status': detector.get('anomalyDetectorStatus'),
                                'estimated_monthly_cost': round(estimated_monthly_cost, 2),
                                'estimated_annual_cost': round(estimated_monthly_cost * 12, 2),
                                'recommendation': 'Consider disabling this detector as it has not found any anomalies'
                            })
                    
                    except Exception as e:
                        logger.warning(f"Could not analyze detector {detector_arn}: {str(e)}")
                        continue
            
            # Sort by detector age descending (oldest ineffective detectors first)
            ineffective_detectors.sort(key=lambda x: x['detector_age_days'], reverse=True)
            
            # Paginate
            total_items = len(ineffective_detectors)
            total_pages = (total_items + self._page_size - 1) // self._page_size
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_detectors = ineffective_detectors[start_idx:end_idx]
            
            # Calculate totals
            total_monthly_cost = sum(d['estimated_monthly_cost'] for d in ineffective_detectors)
            total_annual_cost = sum(d['estimated_annual_cost'] for d in ineffective_detectors)
            
            return {
                'status': 'success',
                'ineffective_detectors': paginated_detectors,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_ineffective_detectors': total_items,
                    'total_detectors_analyzed': len(anomaly_detectors),
                    'ineffective_percentage': round(
                        (total_items / len(anomaly_detectors) * 100) if len(anomaly_detectors) > 0 else 0, 1
                    ),
                    'estimated_monthly_cost': round(total_monthly_cost, 2),
                    'estimated_annual_cost': round(total_annual_cost, 2)
                },
                'optimization_recommendations': {
                    'message': 'Disable ineffective anomaly detectors to reduce processing costs',
                    'criteria': f'Detectors included have been active for at least half of their anomalyVisibilityTime and found no anomalies',
                    'actions': [
                        'Review detector configuration and log patterns',
                        'Consider if anomaly detection is needed for these log groups',
                        'Disable detectors that are not providing value',
                        'Monitor for a period before permanent deletion'
                    ],
                    'cost_savings': {
                        'monthly': round(total_monthly_cost, 2),
                        'annual': round(total_annual_cost, 2)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing ineffective log anomaly detectors: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'ineffective_detectors': [],
                'pagination': {'current_page': page, 'page_size': self._page_size, 'total_items': 0, 'total_pages': 0}
            }


class CWAlarmsTips:
    """CloudWatch Alarms optimization and analysis."""
    
    def __init__(self, dao: CloudWatchDAO, pricing_dao: AWSPricingDAO, cost_preferences: CostPreferences):
        self.dao = dao
        self.pricing_dao = pricing_dao
        self.cost_preferences = cost_preferences
        self._page_size = 10  # Items per page
    
    async def describe_alarms(self, alarm_names: Optional[List[str]] = None,
                             **kwargs) -> CloudWatchOperationResult:
        """Describe CloudWatch alarms (FREE operation)."""
        try:
            data = await self.dao.describe_alarms(alarm_names=alarm_names)
            return CloudWatchOperationResult(
                success=True,
                data=data,
                operation_name="describe_alarms",
                cost_incurred=False,
                operation_type="free",
                primary_data_source="cloudwatch_config"
            )
        except Exception as e:
            logger.error(f"Error describing alarms: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="describe_alarms"
            )
    
    async def listAlarm(self, page: int = 1, alarm_name_prefix: Optional[str] = None,
                       can_use_expensive_cost_to_estimate: bool = False,
                       lookback_days: int = 30) -> Dict[str, Any]:
        """
        List all alarms ordered by estimated cost (descending), paginated.
        
        Args:
            page: Page number (1-based)
            alarm_name_prefix: Optional filter for alarm names
            can_use_expensive_cost_to_estimate: If True, uses CloudWatch Metrics API (paid) 
                                               to get accurate alarm evaluation counts. If False 
                                               (default), estimates cost by dimension count.
            lookback_days: Number of days to analyze for evaluation counts (only used if 
                          can_use_expensive_cost_to_estimate=True)
            
        Returns:
            Dict with alarms, pagination, summary, and pricing_info
        """
        try:
            # Get all alarms from DAO
            alarms_data = await self.dao.describe_alarms(alarm_name_prefix=alarm_name_prefix)
            alarms = alarms_data['alarms']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('alarms')
            
            # Calculate cost for each alarm
            alarms_with_cost = []
            for alarm in alarms:
                alarm_info = {
                    'alarm_name': alarm.get('AlarmName'),
                    'alarm_arn': alarm.get('AlarmArn'),
                    'state_value': alarm.get('StateValue'),
                    'state_reason': alarm.get('StateReason'),
                    'actions_enabled': alarm.get('ActionsEnabled', False),
                    'alarm_actions': alarm.get('AlarmActions', []),
                    'ok_actions': alarm.get('OKActions', []),
                    'insufficient_data_actions': alarm.get('InsufficientDataActions', [])
                }
                
                # Determine alarm type and calculate cost
                if 'MetricName' in alarm:
                    # Metric alarm
                    dimensions = alarm.get('Dimensions', [])
                    dimension_count = len(dimensions)
                    period = alarm.get('Period', 300)
                    
                    alarm_info['alarm_type'] = 'metric'
                    alarm_info['metric_name'] = alarm.get('MetricName')
                    alarm_info['namespace'] = alarm.get('Namespace')
                    alarm_info['dimensions'] = dimensions
                    alarm_info['dimension_count'] = dimension_count
                    alarm_info['period'] = period
                    
                    # Determine if high resolution
                    is_high_resolution = period < 300
                    alarm_info['is_high_resolution'] = is_high_resolution
                    
                    if is_high_resolution:
                        alarm_info['monthly_cost'] = pricing['high_resolution_alarms_per_alarm']
                        alarm_info['cost_type'] = 'high_resolution'
                    else:
                        alarm_info['monthly_cost'] = pricing['standard_alarms_per_alarm']
                        alarm_info['cost_type'] = 'standard'
                    
                    # Estimate cost based on dimensions (more dimensions = more complexity)
                    alarm_info['estimated_cost_score'] = dimension_count * alarm_info['monthly_cost']
                    alarm_info['cost_estimation_method'] = 'dimension_based'
                    
                elif 'AlarmRule' in alarm:
                    # Composite alarm
                    alarm_info['alarm_type'] = 'composite'
                    alarm_info['alarm_rule'] = alarm.get('AlarmRule')
                    alarm_info['dimension_count'] = 0
                    alarm_info['monthly_cost'] = pricing['composite_alarms_per_alarm']
                    alarm_info['cost_type'] = 'composite'
                    alarm_info['estimated_cost_score'] = alarm_info['monthly_cost']
                    alarm_info['cost_estimation_method'] = 'fixed_composite'
                else:
                    # Unknown type
                    alarm_info['alarm_type'] = 'unknown'
                    alarm_info['dimension_count'] = 0
                    alarm_info['monthly_cost'] = 0
                    alarm_info['cost_type'] = 'unknown'
                    alarm_info['estimated_cost_score'] = 0
                    alarm_info['cost_estimation_method'] = 'unknown'
                
                # If expensive cost estimation is enabled, get actual evaluation counts
                if can_use_expensive_cost_to_estimate and alarm_info['alarm_type'] == 'metric':
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=lookback_days)
                        
                        # Get alarm evaluation metrics
                        eval_data = await self.dao.get_metric_statistics(
                            namespace='AWS/CloudWatch',
                            metric_name='AlarmEvaluations',
                            dimensions=[{'Name': 'AlarmName', 'Value': alarm_info['alarm_name']}],
                            start_time=start_time,
                            end_time=end_time,
                            period=86400,  # Daily
                            statistics=['Sum']
                        )
                        
                        total_evaluations = sum(dp['Sum'] for dp in eval_data['datapoints'])
                        alarm_info['evaluation_count'] = int(total_evaluations)
                        alarm_info['cost_estimation_method'] = 'actual_evaluations'
                        
                        # Actual cost is still the fixed monthly rate, but we have usage data
                        alarm_info['actual_evaluations_per_day'] = total_evaluations / lookback_days if lookback_days > 0 else 0
                        
                    except Exception as e:
                        logger.warning(f"Failed to get evaluation metrics for {alarm_info['alarm_name']}: {str(e)}")
                        alarm_info['evaluation_count'] = None
                        alarm_info['cost_estimation_method'] = 'dimension_based_fallback'
                
                alarm_info['annual_cost'] = alarm_info['monthly_cost'] * 12
                alarms_with_cost.append(alarm_info)
            
            # Sort by estimated cost score (descending)
            alarms_with_cost.sort(key=lambda x: x['estimated_cost_score'], reverse=True)
            
            # Calculate pagination
            total_alarms = len(alarms_with_cost)
            total_pages = (total_alarms + self._page_size - 1) // self._page_size
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1
            
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_alarms = alarms_with_cost[start_idx:end_idx]
            
            # Calculate summary
            total_monthly_cost = sum(a['monthly_cost'] for a in alarms_with_cost)
            standard_count = sum(1 for a in alarms_with_cost if a.get('cost_type') == 'standard')
            high_res_count = sum(1 for a in alarms_with_cost if a.get('cost_type') == 'high_resolution')
            composite_count = sum(1 for a in alarms_with_cost if a.get('cost_type') == 'composite')
            
            free_tier = self.pricing_dao.get_free_tier_limits()
            billable_standard = max(0, standard_count - free_tier['alarms_count'])
            
            return {
                'status': 'success',
                'alarms': paginated_alarms,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_alarms,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_alarms': total_alarms,
                    'standard_alarms': standard_count,
                    'high_resolution_alarms': high_res_count,
                    'composite_alarms': composite_count,
                    'billable_standard_alarms': billable_standard,
                    'total_monthly_cost': round(total_monthly_cost, 2),
                    'total_annual_cost': round(total_monthly_cost * 12, 2),
                    'cost_estimation_method': 'actual_evaluations' if can_use_expensive_cost_to_estimate else 'dimension_based'
                },
                'pricing_info': {
                    'standard_alarm_cost': pricing['standard_alarms_per_alarm'],
                    'high_resolution_alarm_cost': pricing['high_resolution_alarms_per_alarm'],
                    'composite_alarm_cost': pricing['composite_alarms_per_alarm'],
                    'free_tier_alarms': free_tier['alarms_count']
                },
                'filters_applied': {
                    'alarm_name_prefix': alarm_name_prefix
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing alarms: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def listInvalidAlarm(self, page: int = 1, alarm_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        List alarms with INSUFFICIENT_DATA state, ordered by dimension count (descending), paginated.
        
        Alarms in INSUFFICIENT_DATA state may indicate:
        - Misconfigured metrics or dimensions
        - Resources that no longer exist
        - Metrics that are not being published
        - Potential cost waste on non-functional alarms
        
        Args:
            page: Page number (1-based)
            alarm_name_prefix: Optional filter for alarm names
            
        Returns:
            Dict with invalid alarms, pagination, summary, and optimization recommendations
        """
        try:
            # Get alarms with INSUFFICIENT_DATA state
            alarms_data = await self.dao.describe_alarms(
                alarm_name_prefix=alarm_name_prefix,
                state_value='INSUFFICIENT_DATA'
            )
            alarms = alarms_data['alarms']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('alarms')
            
            # Process each alarm
            invalid_alarms = []
            for alarm in alarms:
                alarm_info = {
                    'alarm_name': alarm.get('AlarmName'),
                    'alarm_arn': alarm.get('AlarmArn'),
                    'state_value': alarm.get('StateValue'),
                    'state_reason': alarm.get('StateReason'),
                    'state_updated_timestamp': alarm.get('StateUpdatedTimestamp'),
                    'actions_enabled': alarm.get('ActionsEnabled', False),
                    'alarm_actions': alarm.get('AlarmActions', []),
                    'alarm_description': alarm.get('AlarmDescription', '')
                }
                
                # Calculate how long it's been in INSUFFICIENT_DATA state
                state_updated = alarm.get('StateUpdatedTimestamp')
                if state_updated:
                    if isinstance(state_updated, str):
                        state_updated = datetime.fromisoformat(state_updated.replace('Z', '+00:00'))
                    days_in_state = (datetime.now(timezone.utc) - state_updated).days
                    alarm_info['days_in_insufficient_data_state'] = days_in_state
                else:
                    alarm_info['days_in_insufficient_data_state'] = None
                
                # Determine alarm type and cost
                if 'MetricName' in alarm:
                    # Metric alarm
                    dimensions = alarm.get('Dimensions', [])
                    dimension_count = len(dimensions)
                    period = alarm.get('Period', 300)
                    
                    alarm_info['alarm_type'] = 'metric'
                    alarm_info['metric_name'] = alarm.get('MetricName')
                    alarm_info['namespace'] = alarm.get('Namespace')
                    alarm_info['dimensions'] = dimensions
                    alarm_info['dimension_count'] = dimension_count
                    alarm_info['period'] = period
                    
                    # Determine if high resolution
                    is_high_resolution = period < 300
                    alarm_info['is_high_resolution'] = is_high_resolution
                    
                    if is_high_resolution:
                        alarm_info['monthly_cost'] = pricing['high_resolution_alarms_per_alarm']
                        alarm_info['cost_type'] = 'high_resolution'
                    else:
                        alarm_info['monthly_cost'] = pricing['standard_alarms_per_alarm']
                        alarm_info['cost_type'] = 'standard'
                    
                elif 'AlarmRule' in alarm:
                    # Composite alarm
                    alarm_info['alarm_type'] = 'composite'
                    alarm_info['alarm_rule'] = alarm.get('AlarmRule')
                    alarm_info['dimension_count'] = 0
                    alarm_info['monthly_cost'] = pricing['composite_alarms_per_alarm']
                    alarm_info['cost_type'] = 'composite'
                else:
                    # Unknown type
                    alarm_info['alarm_type'] = 'unknown'
                    alarm_info['dimension_count'] = 0
                    alarm_info['monthly_cost'] = 0
                    alarm_info['cost_type'] = 'unknown'
                
                alarm_info['annual_cost'] = alarm_info['monthly_cost'] * 12
                
                # Add optimization recommendation
                if alarm_info.get('days_in_insufficient_data_state', 0) > 7:
                    alarm_info['recommendation'] = 'Consider deleting - in INSUFFICIENT_DATA state for over 7 days'
                    alarm_info['recommendation_priority'] = 'high'
                elif alarm_info.get('days_in_insufficient_data_state', 0) > 3:
                    alarm_info['recommendation'] = 'Review alarm configuration - prolonged INSUFFICIENT_DATA state'
                    alarm_info['recommendation_priority'] = 'medium'
                else:
                    alarm_info['recommendation'] = 'Monitor - recently entered INSUFFICIENT_DATA state'
                    alarm_info['recommendation_priority'] = 'low'
                
                invalid_alarms.append(alarm_info)
            
            # Sort by dimension count (descending) - more complex alarms first
            invalid_alarms.sort(key=lambda x: x.get('dimension_count', 0), reverse=True)
            
            # Calculate pagination
            total_invalid = len(invalid_alarms)
            total_pages = (total_invalid + self._page_size - 1) // self._page_size
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1
            
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_alarms = invalid_alarms[start_idx:end_idx]
            
            # Calculate summary
            total_wasted_monthly_cost = sum(a['monthly_cost'] for a in invalid_alarms)
            high_priority_count = sum(1 for a in invalid_alarms if a.get('recommendation_priority') == 'high')
            
            standard_count = sum(1 for a in invalid_alarms if a.get('cost_type') == 'standard')
            high_res_count = sum(1 for a in invalid_alarms if a.get('cost_type') == 'high_resolution')
            composite_count = sum(1 for a in invalid_alarms if a.get('cost_type') == 'composite')
            
            return {
                'status': 'success',
                'invalid_alarms': paginated_alarms,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_invalid,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_invalid_alarms': total_invalid,
                    'standard_alarms': standard_count,
                    'high_resolution_alarms': high_res_count,
                    'composite_alarms': composite_count,
                    'high_priority_recommendations': high_priority_count,
                    'potential_monthly_savings': round(total_wasted_monthly_cost, 2),
                    'potential_annual_savings': round(total_wasted_monthly_cost * 12, 2)
                },
                'optimization_insight': {
                    'message': f'Found {total_invalid} alarms in INSUFFICIENT_DATA state, potentially wasting ${round(total_wasted_monthly_cost, 2)}/month',
                    'action': 'Review and delete or fix alarms that have been in INSUFFICIENT_DATA state for extended periods'
                },
                'filters_applied': {
                    'alarm_name_prefix': alarm_name_prefix,
                    'state_filter': 'INSUFFICIENT_DATA'
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing invalid alarms: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def analyze_alarms_usage(self, alarm_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze CloudWatch Alarms usage and optimization opportunities."""
        try:
            alarms_data = await self.dao.describe_alarms(alarm_names=alarm_names)
            alarms = alarms_data['alarms']
            
            # Categorize alarms
            standard_alarms = 0
            high_resolution_alarms = 0
            composite_alarms = 0
            alarms_without_actions = []
            
            for alarm in alarms:
                if 'MetricName' in alarm:
                    if alarm.get('Period', 300) < 300:
                        high_resolution_alarms += 1
                    else:
                        standard_alarms += 1
                else:
                    composite_alarms += 1
                
                # Check for actions
                if not (alarm.get('AlarmActions') or alarm.get('OKActions') or 
                       alarm.get('InsufficientDataActions')):
                    alarms_without_actions.append(alarm.get('AlarmName'))
            
            # Calculate costs
            alarms_cost = self.pricing_dao.calculate_cost('alarms', {
                'standard_alarms_count': standard_alarms,
                'high_resolution_alarms_count': high_resolution_alarms,
                'composite_alarms_count': composite_alarms
            })
            
            return {
                'status': 'success',
                'alarms_summary': {
                    'total_alarms': len(alarms),
                    'standard_alarms': standard_alarms,
                    'high_resolution_alarms': high_resolution_alarms,
                    'composite_alarms': composite_alarms,
                    'alarms_without_actions': len(alarms_without_actions)
                },
                'cost_analysis': alarms_cost,
                'alarms_details': alarms[:50]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alarms usage: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class CWDashboardTips:
    """CloudWatch Dashboards optimization and analysis."""
    
    def __init__(self, dao: CloudWatchDAO, pricing_dao: AWSPricingDAO, cost_preferences: CostPreferences):
        self.dao = dao
        self.pricing_dao = pricing_dao
        self.cost_preferences = cost_preferences
        self._page_size = 10  # Items per page
    
    async def list_dashboards(self, dashboard_name_prefix: Optional[str] = None,
                             **kwargs) -> CloudWatchOperationResult:
        """List CloudWatch dashboards (FREE operation)."""
        try:
            data = await self.dao.list_dashboards(dashboard_name_prefix=dashboard_name_prefix)
            return CloudWatchOperationResult(
                success=True,
                data=data,
                operation_name="list_dashboards",
                cost_incurred=False,
                operation_type="free",
                primary_data_source="cloudwatch_config"
            )
        except Exception as e:
            logger.error(f"Error listing dashboards: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="list_dashboards"
            )
    
    async def get_dashboard(self, dashboard_name: str,
                           **kwargs) -> CloudWatchOperationResult:
        """Get dashboard configuration (FREE operation)."""
        try:
            data = await self.dao.get_dashboard(dashboard_name)
            return CloudWatchOperationResult(
                success=True,
                data=data,
                operation_name="get_dashboard",
                cost_incurred=False,
                operation_type="free",
                primary_data_source="cloudwatch_config"
            )
        except Exception as e:
            logger.error(f"Error getting dashboard: {str(e)}")
            return CloudWatchOperationResult(
                success=False,
                error_message=str(e),
                operation_name="get_dashboard"
            )
    
    async def listDashboard(self, page: int = 1, dashboard_name_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        List dashboards ordered by total metric dimensions referenced (descending), paginated.
        
        Dashboards with more metric dimensions are typically more complex and may:
        - Be more expensive to maintain
        - Have performance issues
        - Be harder to understand and maintain
        
        Args:
            page: Page number (1-based)
            dashboard_name_prefix: Optional filter for dashboard names
            
        Returns:
            Dict with dashboards, pagination, summary, and pricing_info
        """
        try:
            # Get all dashboards from DAO
            dashboards_data = await self.dao.list_dashboards(dashboard_name_prefix=dashboard_name_prefix)
            dashboards = dashboards_data['dashboards']
            
            # Get pricing data
            pricing = self.pricing_dao.get_pricing_data('dashboards')
            free_tier = self.pricing_dao.get_free_tier_limits()
            
            # Process each dashboard
            dashboards_with_metrics = []
            now = datetime.now(timezone.utc)
            
            for dashboard in dashboards:
                dashboard_name = dashboard.get('DashboardName')
                
                dashboard_info = {
                    'dashboard_name': dashboard_name,
                    'dashboard_arn': dashboard.get('DashboardArn'),
                    'last_modified': dashboard.get('LastModified'),
                    'size': dashboard.get('Size', 0)
                }
                
                # Calculate days since last modified
                last_modified = dashboard.get('LastModified')
                if last_modified:
                    if isinstance(last_modified, str):
                        last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    days_since_modified = (now - last_modified).days
                    dashboard_info['days_since_modified'] = days_since_modified
                    dashboard_info['is_stale'] = days_since_modified > 90
                else:
                    dashboard_info['days_since_modified'] = None
                    dashboard_info['is_stale'] = False
                
                # Get dashboard body to analyze metrics
                try:
                    dashboard_config = await self.dao.get_dashboard(dashboard_name)
                    dashboard_body = dashboard_config.get('dashboard_body', '{}')
                    
                    # Parse dashboard body
                    if isinstance(dashboard_body, str):
                        dashboard_body = json.loads(dashboard_body)
                    
                    # Count metrics and dimensions
                    total_metrics = 0
                    total_dimensions = 0
                    unique_namespaces = set()
                    unique_metric_names = set()
                    
                    widgets = dashboard_body.get('widgets', [])
                    for widget in widgets:
                        properties = widget.get('properties', {})
                        metrics = properties.get('metrics', [])
                        
                        for metric in metrics:
                            if isinstance(metric, list) and len(metric) >= 2:
                                # Metric format: [namespace, metric_name, dim1_name, dim1_value, dim2_name, dim2_value, ...]
                                namespace = metric[0]
                                metric_name = metric[1]
                                
                                unique_namespaces.add(namespace)
                                unique_metric_names.add(f"{namespace}/{metric_name}")
                                total_metrics += 1
                                
                                # Count dimensions (pairs after namespace and metric_name)
                                dimension_count = (len(metric) - 2) // 2
                                total_dimensions += dimension_count
                    
                    dashboard_info['total_metrics'] = total_metrics
                    dashboard_info['total_dimensions'] = total_dimensions
                    dashboard_info['unique_namespaces'] = len(unique_namespaces)
                    dashboard_info['unique_metric_names'] = len(unique_metric_names)
                    dashboard_info['widget_count'] = len(widgets)
                    dashboard_info['avg_dimensions_per_metric'] = round(total_dimensions / total_metrics, 2) if total_metrics > 0 else 0
                    
                except Exception as e:
                    logger.warning(f"Failed to parse dashboard {dashboard_name}: {str(e)}")
                    dashboard_info['total_metrics'] = 0
                    dashboard_info['total_dimensions'] = 0
                    dashboard_info['unique_namespaces'] = 0
                    dashboard_info['unique_metric_names'] = 0
                    dashboard_info['widget_count'] = 0
                    dashboard_info['avg_dimensions_per_metric'] = 0
                    dashboard_info['parse_error'] = str(e)
                
                # Calculate cost (fixed per dashboard)
                dashboard_info['monthly_cost'] = pricing['dashboard_per_month']
                dashboard_info['annual_cost'] = pricing['dashboard_per_month'] * 12
                
                dashboards_with_metrics.append(dashboard_info)
            
            # Sort by total dimensions (descending)
            dashboards_with_metrics.sort(key=lambda x: x['total_dimensions'], reverse=True)
            
            # Calculate pagination
            total_dashboards = len(dashboards_with_metrics)
            total_pages = (total_dashboards + self._page_size - 1) // self._page_size
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1
            
            start_idx = (page - 1) * self._page_size
            end_idx = start_idx + self._page_size
            paginated_dashboards = dashboards_with_metrics[start_idx:end_idx]
            
            # Calculate summary
            billable_dashboards = max(0, total_dashboards - free_tier['dashboards_count'])
            total_monthly_cost = billable_dashboards * pricing['dashboard_per_month']
            
            total_metrics_all = sum(d['total_metrics'] for d in dashboards_with_metrics)
            total_dimensions_all = sum(d['total_dimensions'] for d in dashboards_with_metrics)
            stale_count = sum(1 for d in dashboards_with_metrics if d['is_stale'])
            
            return {
                'status': 'success',
                'dashboards': paginated_dashboards,
                'pagination': {
                    'current_page': page,
                    'page_size': self._page_size,
                    'total_items': total_dashboards,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_previous': page > 1
                },
                'summary': {
                    'total_dashboards': total_dashboards,
                    'billable_dashboards': billable_dashboards,
                    'free_tier_dashboards': min(total_dashboards, free_tier['dashboards_count']),
                    'total_metrics': total_metrics_all,
                    'total_dimensions': total_dimensions_all,
                    'avg_dimensions_per_dashboard': round(total_dimensions_all / total_dashboards, 2) if total_dashboards > 0 else 0,
                    'stale_dashboards': stale_count,
                    'total_monthly_cost': round(total_monthly_cost, 2),
                    'total_annual_cost': round(total_monthly_cost * 12, 2)
                },
                'pricing_info': {
                    'dashboard_cost': pricing['dashboard_per_month'],
                    'free_tier_dashboards': free_tier['dashboards_count'],
                    'free_tier_metrics_per_dashboard': free_tier['dashboard_metrics']
                },
                'optimization_insights': {
                    'high_complexity_dashboards': sum(1 for d in dashboards_with_metrics if d['total_dimensions'] > 50),
                    'stale_dashboards': stale_count,
                    'recommendation': 'Review dashboards with high dimension counts for optimization opportunities' if total_dimensions_all > 100 else 'Dashboard complexity is reasonable'
                },
                'filters_applied': {
                    'dashboard_name_prefix': dashboard_name_prefix
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing dashboards: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def analyze_dashboards_usage(self, dashboard_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze CloudWatch Dashboards usage and optimization opportunities."""
        try:
            dashboards_data = await self.dao.list_dashboards()
            dashboards = dashboards_data['dashboards']
            
            if dashboard_names:
                dashboards = [d for d in dashboards if d.get('DashboardName') in dashboard_names]
            
            # Analyze stale dashboards
            stale_dashboards = []
            now = datetime.now(timezone.utc)
            
            for dashboard in dashboards:
                last_modified = dashboard.get('LastModified')
                if last_modified:
                    if isinstance(last_modified, str):
                        last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    
                    days_since_modified = (now - last_modified).days
                    if days_since_modified > 90:
                        stale_dashboards.append({
                            'name': dashboard.get('DashboardName'),
                            'days_since_modified': days_since_modified
                        })
            
            # Calculate costs
            dashboards_cost = self.pricing_dao.calculate_cost('dashboards', {
                'dashboards_count': len(dashboards)
            })
            
            return {
                'status': 'success',
                'dashboards_summary': {
                    'total_dashboards': len(dashboards),
                    'exceeds_free_tier': len(dashboards) > 3,
                    'billable_dashboards': max(0, len(dashboards) - 3),
                    'stale_dashboards': len(stale_dashboards)
                },
                'cost_analysis': dashboards_cost,
                'dashboards_details': dashboards[:20]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dashboards usage: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class CloudWatchService:
    """
    Main CloudWatch service that manages specialized tip classes.
    
    This service acts as a factory and lifecycle manager for:
    - CWGeneralSpendTips: General spending analysis
    - CWMetricsTips: Metrics optimization
    - CWLogsTips: Logs optimization  
    - CWAlarmsTips: Alarms optimization
    - CWDashboardTips: Dashboard optimization
    
    Usage:
        service = CloudWatchService(region='us-east-1')
        general_tips = service.getGeneralSpendService()
        result = await general_tips.analyze_overall_spending()
    """
    
    def __init__(self, config: Optional[CloudWatchServiceConfig] = None, region: Optional[str] = None):
        """Initialize CloudWatch service with configuration."""
        # Handle backward compatibility
        if region is not None and config is None:
            config = CloudWatchServiceConfig(region=region)
        
        self.config = config or CloudWatchServiceConfig()
        self.region = self.config.region
        
        # Initialize cost controller
        self.cost_controller = CostController()
        self.cost_preferences = self.config.cost_preferences or CostPreferences()
        
        # Initialize DAOs (shared by all tip classes)
        self._dao = CloudWatchDAO(region=self.region, cost_controller=self.cost_controller)
        self._pricing_dao = AWSPricingDAO(region=self.region or 'us-east-1')
        
        # Initialize tip classes (lazy loading)
        self._general_spend_tips = None
        self._metrics_tips = None
        self._logs_tips = None
        self._alarms_tips = None
        self._dashboard_tips = None
        
        log_cloudwatch_operation(logger, "service_initialization", 
                                 region=self.region or 'default',
                                 cost_preferences=str(self.cost_preferences))
    
    def getGeneralSpendService(self) -> CWGeneralSpendTips:
        """Get general spending analysis service."""
        if self._general_spend_tips is None:
            self._general_spend_tips = CWGeneralSpendTips(
                self._dao, self._pricing_dao, self.cost_preferences
            )
        return self._general_spend_tips
    
    def getMetricsService(self) -> CWMetricsTips:
        """Get metrics optimization service."""
        if self._metrics_tips is None:
            self._metrics_tips = CWMetricsTips(
                self._dao, self._pricing_dao, self.cost_preferences
            )
        return self._metrics_tips
    
    def getLogsService(self) -> CWLogsTips:
        """Get logs optimization service."""
        if self._logs_tips is None:
            self._logs_tips = CWLogsTips(
                self._dao, self._pricing_dao, self.cost_preferences
            )
        return self._logs_tips
    
    def getAlarmsService(self) -> CWAlarmsTips:
        """Get alarms optimization service."""
        if self._alarms_tips is None:
            self._alarms_tips = CWAlarmsTips(
                self._dao, self._pricing_dao, self.cost_preferences
            )
        return self._alarms_tips
    
    def getDashboardsService(self) -> CWDashboardTips:
        """Get dashboards optimization service."""
        if self._dashboard_tips is None:
            self._dashboard_tips = CWDashboardTips(
                self._dao, self._pricing_dao, self.cost_preferences
            )
        return self._dashboard_tips
    
    @property
    def pricing(self):
        """Backward compatibility property for pricing DAO."""
        return self._pricing_dao
    
    def update_cost_preferences(self, preferences: CostPreferences):
        """Update cost control preferences for all services."""
        self.cost_preferences = preferences
        
        # Update preferences in existing tip classes
        if self._general_spend_tips:
            self._general_spend_tips.cost_preferences = preferences
        if self._metrics_tips:
            self._metrics_tips.cost_preferences = preferences
        if self._logs_tips:
            self._logs_tips.cost_preferences = preferences
        if self._alarms_tips:
            self._alarms_tips.cost_preferences = preferences
        if self._dashboard_tips:
            self._dashboard_tips.cost_preferences = preferences
        
        log_cloudwatch_operation(logger, "cost_preferences_update", 
                                 preferences=str(preferences))
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics including cache performance."""
        return {
            'service_info': {
                'region': self.region,
                'cost_preferences': self.cost_preferences.__dict__,
                'initialized_services': {
                    'general_spend': self._general_spend_tips is not None,
                    'metrics': self._metrics_tips is not None,
                    'logs': self._logs_tips is not None,
                    'alarms': self._alarms_tips is not None,
                    'dashboards': self._dashboard_tips is not None
                }
            },
            'cache_statistics': self._dao.get_cache_stats(),
            'cost_control_status': {
                'cost_controller_active': True,
                'preferences_validated': True,
                'transparency_enabled': self.config.enable_cost_tracking
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._dao.clear_cache()
        log_cloudwatch_operation(logger, "cache_cleared")
    
    # ========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # These delegate to the new specialized tip classes for analyzer compatibility
    # ========================================================================
    
    async def describe_log_groups(self, log_group_names: Optional[List[str]] = None,
                                  log_group_name_prefix: Optional[str] = None,
                                  **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to logs service."""
        logs_service = self.getLogsService()
        return await logs_service.describe_log_groups(
            log_group_names=log_group_names,
            log_group_name_prefix=log_group_name_prefix,
            **kwargs
        )
    
    async def get_log_group_incoming_bytes(self, log_group_names: Optional[List[str]] = None,
                                          lookback_days: int = 30,
                                          **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to logs service."""
        logs_service = self.getLogsService()
        return await logs_service.get_log_group_incoming_bytes(
            log_group_names=log_group_names,
            lookback_days=lookback_days,
            **kwargs
        )
    
    async def list_metrics(self, namespace: Optional[str] = None,
                          metric_name: Optional[str] = None,
                          **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to metrics service."""
        metrics_service = self.getMetricsService()
        return await metrics_service.list_metrics(
            namespace=namespace,
            metric_name=metric_name,
            **kwargs
        )
    
    async def get_targeted_metric_statistics(self, namespace: str,
                                            metric_name: str,
                                            **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to metrics service."""
        metrics_service = self.getMetricsService()
        return await metrics_service.get_targeted_metric_statistics(
            namespace=namespace,
            metric_name=metric_name,
            **kwargs
        )
    
    async def describe_alarms(self, alarm_names: Optional[List[str]] = None,
                             **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to alarms service."""
        alarms_service = self.getAlarmsService()
        return await alarms_service.describe_alarms(
            alarm_names=alarm_names,
            **kwargs
        )
    
    async def list_dashboards(self, dashboard_name_prefix: Optional[str] = None,
                             **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to dashboards service."""
        dashboards_service = self.getDashboardsService()
        return await dashboards_service.list_dashboards(
            dashboard_name_prefix=dashboard_name_prefix,
            **kwargs
        )
    
    async def get_dashboard(self, dashboard_name: str,
                           **kwargs) -> CloudWatchOperationResult:
        """Backward compatibility: Delegate to dashboards service."""
        dashboards_service = self.getDashboardsService()
        return await dashboards_service.get_dashboard(
            dashboard_name=dashboard_name,
            **kwargs
        )


# Convenience function for backward compatibility

async def create_cloudwatch_service(region: Optional[str] = None, 
                                   cost_preferences: Optional[CostPreferences] = None) -> CloudWatchService:
    """Create a configured CloudWatchService instance."""
    config = CloudWatchServiceConfig(region=region, cost_preferences=cost_preferences)
    service = CloudWatchService(config)
    
    log_cloudwatch_operation(logger, "service_created", region=region or 'default')
    return service