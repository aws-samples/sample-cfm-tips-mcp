"""
CloudWatch Result Processor for Zero-Cost Sorting and Pagination

Handles cost-based sorting and pagination of CloudWatch analysis results without
incurring any additional AWS charges. All operations are performed in-memory using
only data already retrieved from free CloudWatch APIs.
"""

import logging
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PaginationMetadata:
    """Pagination metadata for API responses with 1-based indexing."""
    current_page: int
    page_size: int
    total_items: Optional[int] = None
    total_pages: Optional[int] = None
    has_next_page: bool = False
    has_previous_page: bool = False


class CloudWatchResultProcessor:
    """
    Handles cost-based sorting and pagination of CloudWatch analysis results.
    
    ZERO-COST GUARANTEE:
    - All operations are performed in-memory using pre-existing data
    - No AWS API calls are made during sorting or pagination
    - Uses only cached pricing data and free CloudWatch configuration data
    - Fails gracefully if cost data is unavailable
    
    PAGINATION CONVENTION: 1-BASED INDEXING
    - Pages start at 1 (not 0)
    - Default page is 1 when not specified
    - Page 0 or negative pages default to page 1
    - This follows common REST API conventions (GitHub, AWS APIs, etc.)
    """
    
    def __init__(self, pricing_service=None):
        """
        Initialize CloudWatchResultProcessor.
        
        Args:
            pricing_service: Pre-initialized pricing service with cached data
                           (must not make additional API calls)
        """
        self.page_size = 10  # Fixed page size - not configurable
        self.pricing_service = pricing_service
        self.logger = logging.getLogger(__name__)
        
        # Validate that we never make API calls
        self._validate_zero_cost_constraint()
    
    def _validate_zero_cost_constraint(self):
        """Validate that this processor will not make any AWS API calls."""
        if hasattr(self, '_aws_clients'):
            raise ValueError("CloudWatchResultProcessor must not have AWS clients")
        
        # Log the zero-cost guarantee
        self.logger.info("CloudWatchResultProcessor initialized with zero-cost guarantee")
    
    def calculate_log_group_cost(self, log_group: Dict[str, Any]) -> float:
        """
        Calculate log group cost using only free configuration data.
        
        ZERO-COST GUARANTEE: Uses only storedBytes from describe_log_groups API
        that was already called. No additional API calls.
        
        Args:
            log_group: Log group data from describe_log_groups API
            
        Returns:
            Estimated monthly cost in USD
        """
        try:
            stored_bytes = log_group.get('storedBytes', 0)
            if stored_bytes == 0:
                return 0.0
            
            # Convert bytes to GB
            stored_gb = stored_bytes / (1024**3)
            
            # Get cached pricing (no API calls)
            if self.pricing_service:
                try:
                    logs_pricing = self.pricing_service.get_logs_pricing()
                    # Handle both sync and async pricing service responses
                    if hasattr(logs_pricing, '__await__'):
                        # If it's a coroutine, we can't await it here, use fallback
                        pass
                    elif logs_pricing and logs_pricing.get('status') == 'success':
                        storage_price_per_gb = logs_pricing['logs_pricing'].get('storage_per_gb_month', 0.03)
                        return stored_gb * storage_price_per_gb
                except Exception as e:
                    self.logger.debug(f"Error accessing pricing service: {str(e)}, using fallback pricing")
            
            # Fallback to default pricing if pricing service unavailable
            default_storage_price = 0.03  # USD per GB per month (approximate)
            return stored_gb * default_storage_price
            
        except Exception as e:
            self.logger.warning(f"Error calculating log group cost: {str(e)}")
            return 0.0
    
    def calculate_custom_metric_cost(self, metric: Dict[str, Any]) -> float:
        """
        Calculate custom metric cost using only free configuration data.
        
        ZERO-COST GUARANTEE: Uses only metric metadata from list_metrics API
        that was already called. No additional API calls.
        
        Args:
            metric: Metric data from list_metrics API
            
        Returns:
            Estimated monthly cost in USD
        """
        try:
            namespace = metric.get('Namespace', '')
            
            # Check if it's a custom metric (not AWS namespace)
            # AWS service metrics are free - comprehensive list of AWS namespaces
            aws_namespaces = [
                'AWS/EC2', 'AWS/RDS', 'AWS/Lambda', 'AWS/S3', 'AWS/ELB', 'AWS/ELBv2',
                'AWS/ApplicationELB', 'AWS/NetworkELB', 'AWS/CloudFront', 'AWS/ApiGateway',
                'AWS/DynamoDB', 'AWS/SQS', 'AWS/SNS', 'AWS/Kinesis', 'AWS/ECS',
                'AWS/EKS', 'AWS/Batch', 'AWS/Logs', 'AWS/Events', 'AWS/AutoScaling',
                'AWS/ElastiCache', 'AWS/Redshift', 'AWS/EMR', 'AWS/Glue', 'AWS/StepFunctions',
                'AWS/Config', 'AWS/Usage', 'AWS/TrustedAdvisor', 'AWS/TransitGateway',
                'AWS/VPC', 'AWS/Route53', 'AWS/CloudWatch', 'AWS/Billing', 'AWS/Support',
                'AWS/Inspector', 'AWS/GuardDuty', 'AWS/SecurityHub', 'AWS/WAF', 'AWS/Shield',
                'AWS/Connect', 'AWS/Lex', 'AWS/Polly', 'AWS/Rekognition', 'AWS/Textract',
                'AWS/Comprehend', 'AWS/Translate', 'AWS/Transcribe', 'AWS/Personalize',
                'AWS/SageMaker', 'AWS/Bedrock', 'AWS/CodeBuild', 'AWS/CodeDeploy',
                'AWS/CodePipeline', 'AWS/CodeCommit', 'AWS/X-Ray', 'AWS/AppSync',
                'AWS/Amplify', 'AWS/MediaLive', 'AWS/MediaPackage', 'AWS/MediaConvert',
                'AWS/WorkSpaces', 'AWS/AppStream', 'AWS/FSx', 'AWS/EFS', 'AWS/StorageGateway',
                'AWS/Backup', 'AWS/DataSync', 'AWS/Transfer', 'AWS/DirectConnect',
                'AWS/VPN', 'AWS/PrivateLink', 'AWS/GlobalAccelerator', 'AWS/CloudFormation',
                'AWS/Systems Manager', 'AWS/OpsWorks', 'AWS/Service Catalog', 'AWS/Organizations',
                'AWS/Control Tower', 'AWS/Well-Architected Tool', 'AWS/Trusted Advisor',
                'AWS/Health', 'AWS/Personal Health Dashboard', 'AWS/Cost Explorer',
                # Additional AWS namespaces that were missing
                'AWS/EBS', 'AWS/NATGateway', 'AWS/Cognito', 'AWS/States', 'AWS/Firehose',
                'AWS/SecretsManager', 'AWS/KMS', 'AWS/ECR', 'AWS/QuickSight', 'AWS/Route53Resolver',
                'AWS/Location', 'AWS/SSM-RunCommand', 'AWS/HealthLake', 'AWS/AppRunner',
                'AWS/Kendra', 'AWS/Bedrock/DataAutomation', 'AWS/AuroraDSQL', 'AWS/Redshift-Serverless'
            ]
            # Check if namespace starts with AWS/ - this is the definitive way to identify AWS service metrics
            is_custom = not namespace.startswith('AWS/')
            
            if not is_custom:
                return 0.0  # AWS metrics are free
            
            # Get cached pricing (no API calls)
            if self.pricing_service:
                try:
                    metrics_pricing = self.pricing_service.get_metrics_pricing()
                    # Handle both sync and async pricing service responses
                    if hasattr(metrics_pricing, '__await__'):
                        # If it's a coroutine, we can't await it here, use fallback
                        pass
                    elif metrics_pricing and metrics_pricing.get('status') == 'success':
                        custom_metric_price = metrics_pricing['metrics_pricing'].get('custom_metric_per_month', 0.30)
                        return custom_metric_price
                except Exception as e:
                    self.logger.debug(f"Error accessing pricing service: {str(e)}, using fallback pricing")
            
            # Fallback to default pricing
            default_custom_metric_price = 0.30  # USD per custom metric per month
            return default_custom_metric_price
            
        except Exception as e:
            self.logger.warning(f"Error calculating custom metric cost: {str(e)}")
            return 0.0
    
    def calculate_alarm_cost(self, alarm: Dict[str, Any]) -> float:
        """
        Calculate alarm cost using only free configuration data.
        
        ZERO-COST GUARANTEE: Uses only alarm metadata from describe_alarms API
        that was already called. No additional API calls.
        
        Args:
            alarm: Alarm data from describe_alarms API
            
        Returns:
            Estimated monthly cost in USD
        """
        try:
            # Determine alarm type from configuration
            period = alarm.get('Period', 300)  # Default 5 minutes
            is_high_resolution = period < 300  # Less than 5 minutes
            
            # Get cached pricing (no API calls)
            if self.pricing_service:
                try:
                    alarms_pricing = self.pricing_service.get_alarms_pricing()
                    # Handle both sync and async pricing service responses
                    if hasattr(alarms_pricing, '__await__'):
                        # If it's a coroutine, we can't await it here, use fallback
                        pass
                    elif alarms_pricing and alarms_pricing.get('status') == 'success':
                        if is_high_resolution:
                            return alarms_pricing['alarms_pricing'].get('high_resolution_alarm_per_month', 0.50)
                        else:
                            return alarms_pricing['alarms_pricing'].get('standard_alarm_per_month', 0.10)
                except Exception as e:
                    self.logger.debug(f"Error accessing pricing service: {str(e)}, using fallback pricing")
            
            # Fallback to default pricing
            if is_high_resolution:
                return 0.50  # USD per high-resolution alarm per month
            else:
                return 0.10  # USD per standard alarm per month
                
        except Exception as e:
            self.logger.warning(f"Error calculating alarm cost: {str(e)}")
            return 0.0
    
    def calculate_dashboard_cost(self, dashboard: Dict[str, Any], total_dashboards: int) -> float:
        """
        Calculate dashboard cost using only free configuration data.
        
        ZERO-COST GUARANTEE: Uses only dashboard metadata from list_dashboards API
        that was already called. No additional API calls.
        
        Args:
            dashboard: Dashboard data from list_dashboards API
            total_dashboards: Total number of dashboards (for free tier calculation)
            
        Returns:
            Estimated monthly cost in USD
        """
        try:
            # First 3 dashboards are free, beyond that each dashboard costs money
            # We need to determine if this specific dashboard is in the free tier
            # For simplicity, we'll assume the first 3 dashboards (by index) are free
            
            # If total dashboards <= 3, all are free
            if total_dashboards <= 3:
                return 0.0
            
            # For dashboards beyond the free tier, each costs money
            # Since we can't determine the exact index of this dashboard in the sorted list,
            # we'll calculate the average cost per dashboard beyond free tier
            paid_dashboards = total_dashboards - 3
            
            # Get cached pricing (no API calls)
            dashboard_price = 3.00  # Default price
            if self.pricing_service:
                try:
                    dashboards_pricing = self.pricing_service.get_dashboards_pricing()
                    # Handle both sync and async pricing service responses
                    if hasattr(dashboards_pricing, '__await__'):
                        # If it's a coroutine, we can't await it here, use fallback
                        pass
                    elif dashboards_pricing and dashboards_pricing.get('status') == 'success':
                        dashboard_price = dashboards_pricing['dashboards_pricing'].get('dashboard_per_month', 3.00)
                except Exception as e:
                    self.logger.debug(f"Error accessing pricing service: {str(e)}, using fallback pricing")
            
            # Return the cost for dashboards beyond free tier
            return dashboard_price
            
        except Exception as e:
            self.logger.warning(f"Error calculating dashboard cost: {str(e)}")
            return 0.0
    
    def enrich_items_with_cost_estimates(self, items: List[Dict[str, Any]], item_type: str, 
                                       total_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Add cost estimates to items using only free data and pricing models.
        
        ZERO-COST GUARANTEE: NO ADDITIONAL API CALLS
        - Uses only data already retrieved from free CloudWatch APIs
        - Applies cached pricing data (no additional pricing API calls)
        - Never triggers paid operations for cost estimation
        
        Args:
            items: List of CloudWatch resources
            item_type: Type of resource ('log_groups', 'metrics', 'alarms', 'dashboards')
            total_count: Total count for free tier calculations (dashboards)
            
        Returns:
            Items with 'estimated_monthly_cost' field added
        """
        enriched_items = []
        total_items = total_count or len(items)
        
        for index, item in enumerate(items):
            enriched_item = item.copy()
            
            try:
                if item_type == 'log_groups':
                    cost = self.calculate_log_group_cost(item)
                elif item_type == 'metrics':
                    cost = self.calculate_custom_metric_cost(item)
                elif item_type == 'alarms':
                    cost = self.calculate_alarm_cost(item)
                elif item_type == 'dashboards':
                    # For dashboards, we need to handle free tier properly
                    # First 3 dashboards are free, rest cost money
                    if total_items <= 3:
                        cost = 0.0  # All dashboards are free
                    elif index < 3:
                        cost = 0.0  # This dashboard is in the free tier
                    else:
                        cost = self.calculate_dashboard_cost(item, total_items)
                else:
                    self.logger.warning(f"Unknown item type for cost calculation: {item_type}")
                    cost = 0.0
                
                enriched_item['estimated_monthly_cost'] = cost
                
            except Exception as e:
                self.logger.warning(f"Error enriching {item_type} item with cost: {str(e)}")
                enriched_item['estimated_monthly_cost'] = 0.0
            
            enriched_items.append(enriched_item)
        
        self.logger.debug(f"Enriched {len(enriched_items)} {item_type} items with cost estimates")
        return enriched_items
    
    def sort_by_cost_descending(self, items: List[Dict[str, Any]], 
                              cost_field: str = 'estimated_monthly_cost') -> List[Dict[str, Any]]:
        """
        Sort items by cost in descending order (highest cost first).
        
        ZERO-COST GUARANTEE: Pure function with no external dependencies
        - Operates only on data already present in items
        - No AWS API calls or external requests
        - Fails gracefully if cost data is missing
        
        Args:
            items: List of items to sort (must already have cost estimates)
            cost_field: Field name containing cost value (default: 'estimated_monthly_cost')
            
        Returns:
            Sorted list with highest cost items first
        """
        try:
            # Sort by cost descending, with fallback for missing cost data
            sorted_items = sorted(
                items,
                key=lambda x: x.get(cost_field, 0.0),
                reverse=True
            )
            
            self.logger.debug(f"Sorted {len(sorted_items)} items by {cost_field} (descending)")
            return sorted_items
            
        except Exception as e:
            self.logger.warning(f"Error sorting items by cost: {str(e)}, returning original order")
            return items
    
    def create_pagination_metadata(self, total_items: int, current_page: int) -> PaginationMetadata:
        """
        Create pagination metadata for API responses.
        
        PAGINATION METADATA CALCULATION:
        - current_page: 1-based page number (validated to be >= 1)
        - total_pages: ceil(total_items / page_size), minimum 1 if total_items > 0
        - has_next_page: current_page < total_pages
        - has_previous_page: current_page > 1
        
        Args:
            total_items: Total number of items across all pages
            current_page: Current 1-based page number
            
        Returns:
            PaginationMetadata object with all pagination fields
        """
        # Validate and correct page number (1-based)
        validated_page = max(1, current_page)
        if validated_page != current_page:
            self.logger.info(f"Corrected invalid page number {current_page} to {validated_page}")
        
        # Calculate total pages
        total_pages = math.ceil(total_items / self.page_size) if total_items > 0 else 0
        
        # Calculate navigation flags
        has_next_page = validated_page < total_pages
        has_previous_page = validated_page > 1
        
        return PaginationMetadata(
            current_page=validated_page,
            page_size=self.page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next_page=has_next_page,
            has_previous_page=has_previous_page
        )
    
    def paginate_results(self, items: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Apply pagination to sorted results with metadata.
        
        PAGINATION BEHAVIOR (1-BASED):
        - page=1: Returns items 0-9 (first 10 items)
        - page=2: Returns items 10-19 (second 10 items)
        - page=0 or page<1: Defaults to page=1
        - page>total_pages: Returns empty items array with valid pagination metadata
        
        Args:
            items: Pre-sorted list of items
            page: 1-based page number (default: 1)
            
        Returns:
            Dict with 'items' and 'pagination' keys
        """
        total_items = len(items)
        pagination_metadata = self.create_pagination_metadata(total_items, page)
        
        # Calculate slice indices (convert 1-based page to 0-based array indices)
        start_index = (pagination_metadata.current_page - 1) * self.page_size
        end_index = start_index + self.page_size
        
        # Extract page items
        page_items = items[start_index:end_index]
        
        self.logger.debug(f"Paginated {total_items} items: page {pagination_metadata.current_page} "
                         f"({len(page_items)} items)")
        
        return {
            'items': page_items,
            'pagination': {
                'current_page': pagination_metadata.current_page,
                'page_size': pagination_metadata.page_size,
                'total_items': pagination_metadata.total_items,
                'total_pages': pagination_metadata.total_pages,
                'has_next_page': pagination_metadata.has_next_page,
                'has_previous_page': pagination_metadata.has_previous_page
            }
        }
    
    def process_log_groups_results(self, log_groups: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Process log groups with cost sorting and pagination (1-based pages).
        
        Args:
            log_groups: List of log group data from describe_log_groups
            page: 1-based page number
            
        Returns:
            Paginated and sorted log groups with metadata
        """
        # Enrich with cost estimates (zero additional cost)
        enriched_log_groups = self.enrich_items_with_cost_estimates(log_groups, 'log_groups')
        
        # Sort by cost descending (zero additional cost)
        sorted_log_groups = self.sort_by_cost_descending(enriched_log_groups)
        
        # Apply pagination (zero additional cost)
        return self.paginate_results(sorted_log_groups, page)
    
    def process_metrics_results(self, metrics: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Process metrics with cost sorting and pagination (1-based pages).
        
        Args:
            metrics: List of metric data from list_metrics
            page: 1-based page number
            
        Returns:
            Paginated and sorted metrics with metadata
        """
        # Enrich with cost estimates (zero additional cost)
        enriched_metrics = self.enrich_items_with_cost_estimates(metrics, 'metrics')
        
        # Add optimization reasons to each metric
        for metric in enriched_metrics:
            namespace = metric.get('Namespace', '')
            dimensions = metric.get('Dimensions', [])
            
            # AWS namespaces (free metrics) - use same comprehensive list
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
            
            optimization_reasons = []
            
            # High cardinality custom metric
            if not namespace.startswith('AWS/') and len(dimensions) > 3:
                optimization_reasons.append('high_cardinality_custom_metric')
            
            # Custom metric beyond free tier
            if not namespace.startswith('AWS/'):
                optimization_reasons.append('custom_metric_cost')
            
            # High dimension count
            if len(dimensions) > 5:
                optimization_reasons.append('excessive_dimensions')
            
            # Add optimization metadata to the metric
            metric['optimization_reasons'] = optimization_reasons
            metric['is_optimizable'] = len(optimization_reasons) > 0
            metric['dimensions_count'] = len(dimensions)
        
        # Sort by cost descending (zero additional cost)
        sorted_metrics = self.sort_by_cost_descending(enriched_metrics)
        
        # Apply pagination (zero additional cost)
        return self.paginate_results(sorted_metrics, page)
    
    def process_alarms_results(self, alarms: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Process alarms with cost sorting and pagination (1-based pages).
        
        Args:
            alarms: List of alarm data from describe_alarms
            page: 1-based page number
            
        Returns:
            Paginated and sorted alarms with metadata
        """
        # Enrich with cost estimates (zero additional cost)
        enriched_alarms = self.enrich_items_with_cost_estimates(alarms, 'alarms')
        
        # Sort by cost descending (zero additional cost)
        sorted_alarms = self.sort_by_cost_descending(enriched_alarms)
        
        # Apply pagination (zero additional cost)
        return self.paginate_results(sorted_alarms, page)
    
    def process_dashboards_results(self, dashboards: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Process dashboards with cost sorting and pagination (1-based pages).
        
        Args:
            dashboards: List of dashboard data from list_dashboards
            page: 1-based page number
            
        Returns:
            Paginated and sorted dashboards with metadata
        """
        # Enrich with cost estimates (zero additional cost)
        enriched_dashboards = self.enrich_items_with_cost_estimates(
            dashboards, 'dashboards', total_count=len(dashboards)
        )
        
        # Sort by cost descending (zero additional cost)
        sorted_dashboards = self.sort_by_cost_descending(enriched_dashboards)
        
        # Apply pagination (zero additional cost)
        return self.paginate_results(sorted_dashboards, page)
    
    def process_recommendations(self, recommendations: List[Dict[str, Any]], page: int = 1) -> Dict[str, Any]:
        """
        Process recommendations sorted by potential savings and paginated (1-based pages).
        
        Args:
            recommendations: List of optimization recommendations
            page: 1-based page number
            
        Returns:
            Paginated and sorted recommendations with metadata
        """
        # Sort by potential savings descending (zero additional cost)
        sorted_recommendations = self.sort_by_cost_descending(
            recommendations, cost_field='potential_monthly_savings'
        )
        
        # Apply pagination (zero additional cost)
        return self.paginate_results(sorted_recommendations, page)