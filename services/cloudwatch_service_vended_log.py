"""
CloudWatch Vended Logs Service

Handles identification and analysis of AWS service logs that can be vended directly to S3
to reduce CloudWatch Logs costs. Vended logs bypass CloudWatch Logs ingestion and storage,
significantly reducing costs for high-volume service logs.

Vended log targets include:
- VPC Flow Logs
- ELB Access Logs
- CloudFront Access Logs
- S3 Access Logs
- RDS/Aurora Logs
- Lambda Logs (via subscription filters to S3)
- CloudTrail Logs
- WAF Logs
- Route 53 Query Logs
"""

import logging
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CloudWatchLogsPricingCalculator:
    """
    Calculates CloudWatch Logs costs with tiered pricing.
    
    AWS CloudWatch Logs Pricing (per month):
    - First 10 TB: $0.50/GB (Standard), $0.25/GB (vended to S3/Firehose)
    - Next 20 TB: $0.25/GB (Standard), $0.15/GB (vended)
    - Next 20 TB: $0.10/GB (Standard), $0.075/GB (vended)
    - Over 50 TB: $0.05/GB (Standard), $0.05/GB (vended)
    - Storage: $0.03/GB/month (both Standard and vended)
    """
    
    # Pricing tiers in GB (1 TB = 1024 GB)
    TIER_1_LIMIT = 10 * 1024  # 10 TB
    TIER_2_LIMIT = 30 * 1024  # 30 TB (10 + 20)
    TIER_3_LIMIT = 50 * 1024  # 50 TB (10 + 20 + 20)
    
    # Delivery costs per GB
    STANDARD_DELIVERY_TIERS = [
        (TIER_1_LIMIT, 0.50),      # First 10 TB
        (TIER_2_LIMIT, 0.25),      # Next 20 TB
        (TIER_3_LIMIT, 0.10),      # Next 20 TB
        (float('inf'), 0.05)       # Over 50 TB
    ]
    
    VENDED_DELIVERY_TIERS = [
        (TIER_1_LIMIT, 0.25),      # First 10 TB
        (TIER_2_LIMIT, 0.15),      # Next 20 TB
        (TIER_3_LIMIT, 0.075),     # Next 20 TB
        (float('inf'), 0.05)       # Over 50 TB
    ]
    
    STORAGE_COST = 0.03  # Per GB/month
    S3_STORAGE_COST = 0.023  # Per GB/month
    FIREHOSE_COST = 0.029  # Per GB
    LAMBDA_PROCESSING_COST = 0.20  # Per GB (estimated)
    
    @classmethod
    def calculate_tiered_delivery_cost(cls, gb_amount: float, tiers: list) -> float:
        """
        Calculate delivery cost using tiered pricing.
        
        Args:
            gb_amount: Amount in GB
            tiers: List of (limit, price) tuples
            
        Returns:
            Total delivery cost
        """
        total_cost = 0.0
        remaining = gb_amount
        previous_limit = 0
        
        for limit, price in tiers:
            tier_size = limit - previous_limit
            if remaining <= 0:
                break
            
            # Calculate how much falls in this tier
            amount_in_tier = min(remaining, tier_size)
            total_cost += amount_in_tier * price
            remaining -= amount_in_tier
            previous_limit = limit
        
        return total_cost
    
    @classmethod
    def calculate_cloudwatch_standard_cost(cls, monthly_ingestion_gb: float, storage_gb: float) -> Dict[str, float]:
        """Calculate CloudWatch Logs Standard cost."""
        delivery_cost = cls.calculate_tiered_delivery_cost(monthly_ingestion_gb, cls.STANDARD_DELIVERY_TIERS)
        storage_cost = storage_gb * cls.STORAGE_COST
        
        return {
            'delivery_cost': delivery_cost,
            'storage_cost': storage_cost,
            'total_monthly': delivery_cost + storage_cost
        }
    
    @classmethod
    def calculate_vended_s3_cost(cls, monthly_ingestion_gb: float, storage_gb: float) -> Dict[str, float]:
        """Calculate vended to S3 cost."""
        delivery_cost = cls.calculate_tiered_delivery_cost(monthly_ingestion_gb, cls.VENDED_DELIVERY_TIERS)
        storage_cost = storage_gb * cls.S3_STORAGE_COST
        
        return {
            'delivery_cost': delivery_cost,
            'storage_cost': storage_cost,
            'total_monthly': delivery_cost + storage_cost
        }
    
    @classmethod
    def calculate_vended_firehose_cost(cls, monthly_ingestion_gb: float, storage_gb: float) -> Dict[str, float]:
        """Calculate vended via Firehose cost."""
        delivery_cost = cls.calculate_tiered_delivery_cost(monthly_ingestion_gb, cls.VENDED_DELIVERY_TIERS)
        firehose_cost = monthly_ingestion_gb * cls.FIREHOSE_COST
        storage_cost = storage_gb * cls.S3_STORAGE_COST
        
        return {
            'delivery_cost': delivery_cost,
            'firehose_cost': firehose_cost,
            'storage_cost': storage_cost,
            'total_monthly': delivery_cost + firehose_cost + storage_cost
        }
    
    @classmethod
    def calculate_vended_lambda_cost(cls, monthly_ingestion_gb: float, storage_gb: float) -> Dict[str, float]:
        """Calculate vended via Lambda subscription filter cost."""
        delivery_cost = cls.calculate_tiered_delivery_cost(monthly_ingestion_gb, cls.VENDED_DELIVERY_TIERS)
        lambda_cost = monthly_ingestion_gb * cls.LAMBDA_PROCESSING_COST
        storage_cost = storage_gb * cls.S3_STORAGE_COST
        
        return {
            'delivery_cost': delivery_cost,
            'lambda_cost': lambda_cost,
            'storage_cost': storage_cost,
            'total_monthly': delivery_cost + lambda_cost + storage_cost
        }


class VendedLogsDAO:
    """
    Data Access Object for identifying and analyzing vended log opportunities.
    
    This class identifies AWS service logs currently stored in CloudWatch Logs
    that could be vended directly to S3 for cost savings.
    """
    
    def __init__(self, region: Optional[str] = None):
        self.region = region
        self.logs_client = boto3.client('logs', region_name=self.region)
        self.ec2_client = boto3.client('ec2', region_name=self.region)
        self.pricing_calculator = CloudWatchLogsPricingCalculator()
        
        # Patterns to identify vended log candidates
        # Cost reduction percentages based on AWS pricing:
        # - CloudWatch Standard: $0.50/GB delivery + $0.03/GB storage = $0.53/GB/month
        # - Vended to S3: $0.25/GB delivery + $0.023/GB storage = $0.273/GB/month (48.5% savings)
        # - Vended via Firehose: $0.25/GB delivery + $0.029/GB Firehose + $0.023/GB storage = $0.302/GB/month (43% savings)
        # - Additional savings possible with S3 lifecycle policies (Glacier: +20-40%)
        # See COST_REDUCTION_CALCULATIONS.md for detailed breakdown
        self._vended_log_patterns = {
            'vpc_flow_logs': {
                'patterns': ['/aws/vpc/flowlogs', 'vpc-flow-logs', 'flowlogs'],
                'service': 'VPC Flow Logs',
                'vending_method': 'Direct to S3',
                'cost_reduction': 0.48,  # 48% base savings (can be 60%+ with Glacier)
                'documentation': 'https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs-s3.html'
            },
            'elb_access_logs': {
                'patterns': ['/aws/elasticloadbalancing', 'elb-access-logs', 'alb-access-logs'],
                'service': 'ELB/ALB Access Logs',
                'vending_method': 'Direct to S3',
                'cost_reduction': 0.48,  # 48% base savings (can be 70%+ with Glacier)
                'documentation': 'https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-access-logs.html'
            },
            'cloudfront_logs': {
                'patterns': ['/aws/cloudfront', 'cloudfront-logs'],
                'service': 'CloudFront Access Logs',
                'vending_method': 'Direct to S3',
                'cost_reduction': 0.48,  # 48% base savings (can be 70%+ with Glacier)
                'documentation': 'https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/AccessLogs.html'
            },
            's3_access_logs': {
                'patterns': ['/aws/s3', 's3-access-logs'],
                'service': 'S3 Access Logs',
                'vending_method': 'Direct to S3',
                'cost_reduction': 0.48,  # 48% base savings (can be 70%+ with Glacier)
                'documentation': 'https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html'
            },
            'rds_logs': {
                'patterns': ['/aws/rds', 'rds/'],
                'service': 'RDS/Aurora Logs',
                'vending_method': 'Export to S3',
                'cost_reduction': 0.39,  # 39% savings (export costs included)
                'documentation': 'https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_LogAccess.html'
            },
            'lambda_logs': {
                'patterns': ['/aws/lambda/'],
                'service': 'Lambda Logs',
                'vending_method': 'Subscription Filter to S3',
                'cost_reduction': 0.11,  # 11% savings (Lambda processing costs high)
                'documentation': 'https://docs.aws.amazon.com/lambda/latest/dg/monitoring-cloudwatchlogs.html'
            },
            'cloudtrail_logs': {
                'patterns': ['/aws/cloudtrail', 'cloudtrail'],
                'service': 'CloudTrail Logs',
                'vending_method': 'Direct to S3',
                'cost_reduction': 0.48,  # 48% base savings (can be 70%+ with Glacier)
                'documentation': 'https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-create-and-update-a-trail.html'
            },
            'waf_logs': {
                'patterns': ['/aws/waf', 'aws-waf-logs'],
                'service': 'WAF Logs',
                'vending_method': 'Via Kinesis Firehose to S3',
                'cost_reduction': 0.43,  # 43% savings (Firehose costs included)
                'documentation': 'https://docs.aws.amazon.com/waf/latest/developerguide/logging.html'
            },
            'route53_logs': {
                'patterns': ['/aws/route53', 'route53-query-logs'],
                'service': 'Route 53 Query Logs',
                'vending_method': 'Via Kinesis Firehose to S3',
                'cost_reduction': 0.43,  # 43% savings (Firehose costs included)
                'documentation': 'https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/query-logs.html'
            }
        }
    
    def identify_vended_log_type(self, log_group_name: str) -> Optional[Dict[str, Any]]:
        """
        Identify if a log group is a candidate for vended logs.
        
        Args:
            log_group_name: Name of the log group
            
        Returns:
            Dict with vended log information if match found, None otherwise
        """
        log_group_lower = log_group_name.lower()
        
        for log_type, config in self._vended_log_patterns.items():
            for pattern in config['patterns']:
                if pattern.lower() in log_group_lower:
                    return {
                        'log_type': log_type,
                        'service': config['service'],
                        'vending_method': config['vending_method'],
                        'cost_reduction_percentage': config['cost_reduction'] * 100,
                        'documentation_url': config['documentation'],
                        'matched_pattern': pattern
                    }
        
        return None
    
    async def analyze_vended_log_opportunities(self, log_groups: List[Dict[str, Any]], 
                                              pricing: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze log groups for vended log opportunities with tiered pricing.
        
        Args:
            log_groups: List of log group metadata from describe_log_groups
            pricing: CloudWatch Logs pricing data (legacy, not used with tiered pricing)
            
        Returns:
            List of vended log opportunities with cost savings
        """
        opportunities = []
        
        for lg in log_groups:
            log_group_name = lg.get('logGroupName')
            stored_bytes = lg.get('storedBytes', 0)
            stored_gb = stored_bytes / (1024**3)
            retention_days = lg.get('retentionInDays', 0)
            
            # Check if this is a vended log candidate
            vended_info = self.identify_vended_log_type(log_group_name)
            
            if vended_info:
                # Estimate monthly ingestion based on retention
                if retention_days and retention_days > 0:
                    # Has retention: calculate monthly ingestion rate
                    daily_ingestion_gb = stored_gb / retention_days
                    monthly_ingestion_gb = daily_ingestion_gb * 30
                    
                    # Calculate monthly costs
                    current_costs = self.pricing_calculator.calculate_cloudwatch_standard_cost(
                        monthly_ingestion_gb, stored_gb
                    )
                    
                    # Determine vended cost based on method
                    log_type = vended_info['log_type']
                    if log_type in ['waf_logs', 'route53_logs']:
                        vended_costs = self.pricing_calculator.calculate_vended_firehose_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    elif log_type == 'lambda_logs':
                        vended_costs = self.pricing_calculator.calculate_vended_lambda_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    else:
                        vended_costs = self.pricing_calculator.calculate_vended_s3_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    
                    # Calculate savings
                    monthly_savings = current_costs['total_monthly'] - vended_costs['total_monthly']
                    annual_savings = monthly_savings * 12
                    cost_reduction_pct = (monthly_savings / current_costs['total_monthly'] * 100) if current_costs['total_monthly'] > 0 else 0
                    
                    opportunities.append({
                        'log_group_name': log_group_name,
                        'service': vended_info['service'],
                        'log_type': vended_info['log_type'],
                        'vending_method': vended_info['vending_method'],
                        'stored_gb': round(stored_gb, 4),
                        'retention_days': retention_days,
                        'monthly_ingestion_gb': round(monthly_ingestion_gb, 4),
                        'current_costs': {
                            'delivery_cost': round(current_costs['delivery_cost'], 4),
                            'storage_cost': round(current_costs['storage_cost'], 4),
                            'total_monthly': round(current_costs['total_monthly'], 4)
                        },
                        'vended_costs': {
                            'delivery_cost': round(vended_costs['delivery_cost'], 4),
                            'storage_cost': round(vended_costs['storage_cost'], 4),
                            'total_monthly': round(vended_costs['total_monthly'], 4),
                            **{k: round(v, 4) for k, v in vended_costs.items() 
                               if k not in ['delivery_cost', 'storage_cost', 'total_monthly']}
                        },
                        'savings': {
                            'monthly_savings': round(monthly_savings, 4),
                            'annual_savings': round(annual_savings, 2),
                            'cost_reduction_percentage': round(cost_reduction_pct, 1)
                        },
                        'implementation': {
                            'method': vended_info['vending_method'],
                            'documentation': vended_info['documentation_url'],
                            'complexity': self._get_implementation_complexity(vended_info['log_type'])
                        }
                    })
                else:
                    # No retention: return absolute cost (annual cost to store current data)
                    # Assume data is accumulated over 12 months
                    monthly_ingestion_gb = stored_gb / 12
                    
                    # Calculate annual costs (12 months of ingestion + storage)
                    current_costs_monthly = self.pricing_calculator.calculate_cloudwatch_standard_cost(
                        monthly_ingestion_gb, stored_gb
                    )
                    current_annual_cost = current_costs_monthly['total_monthly'] * 12
                    
                    # Determine vended cost based on method
                    log_type = vended_info['log_type']
                    if log_type in ['waf_logs', 'route53_logs']:
                        vended_costs_monthly = self.pricing_calculator.calculate_vended_firehose_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    elif log_type == 'lambda_logs':
                        vended_costs_monthly = self.pricing_calculator.calculate_vended_lambda_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    else:
                        vended_costs_monthly = self.pricing_calculator.calculate_vended_s3_cost(
                            monthly_ingestion_gb, stored_gb
                        )
                    
                    vended_annual_cost = vended_costs_monthly['total_monthly'] * 12
                    
                    # Calculate absolute savings (annual)
                    annual_savings = current_annual_cost - vended_annual_cost
                    monthly_savings = annual_savings / 12
                    cost_reduction_pct = (annual_savings / current_annual_cost * 100) if current_annual_cost > 0 else 0
                    
                    opportunities.append({
                        'log_group_name': log_group_name,
                        'service': vended_info['service'],
                        'log_type': vended_info['log_type'],
                        'vending_method': vended_info['vending_method'],
                        'stored_gb': round(stored_gb, 4),
                        'retention_days': 'Never Expire',
                        'monthly_ingestion_gb': round(monthly_ingestion_gb, 4),
                        'estimated_ingestion_note': 'Estimated as stored_gb / 12 months',
                        'current_costs': {
                            'delivery_cost': round(current_costs_monthly['delivery_cost'], 4),
                            'storage_cost': round(current_costs_monthly['storage_cost'], 4),
                            'total_monthly': round(current_costs_monthly['total_monthly'], 4),
                            'total_annual': round(current_annual_cost, 2)
                        },
                        'vended_costs': {
                            'delivery_cost': round(vended_costs_monthly['delivery_cost'], 4),
                            'storage_cost': round(vended_costs_monthly['storage_cost'], 4),
                            'total_monthly': round(vended_costs_monthly['total_monthly'], 4),
                            'total_annual': round(vended_annual_cost, 2),
                            **{k: round(v, 4) for k, v in vended_costs_monthly.items() 
                               if k not in ['delivery_cost', 'storage_cost', 'total_monthly']}
                        },
                        'savings': {
                            'monthly_savings': round(monthly_savings, 4),
                            'annual_savings': round(annual_savings, 2),
                            'cost_reduction_percentage': round(cost_reduction_pct, 1)
                        },
                        'implementation': {
                            'method': vended_info['vending_method'],
                            'documentation': vended_info['documentation_url'],
                            'complexity': self._get_implementation_complexity(vended_info['log_type'])
                        }
                    })
        
        return opportunities
    
    def _get_implementation_complexity(self, log_type: str) -> str:
        """
        Get implementation complexity for vended log setup.
        
        Args:
            log_type: Type of vended log
            
        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        complexity_map = {
            'vpc_flow_logs': 'low',  # Simple configuration change
            'elb_access_logs': 'low',  # Enable in ELB settings
            'cloudfront_logs': 'low',  # Enable in CloudFront settings
            's3_access_logs': 'low',  # Enable in S3 bucket settings
            'cloudtrail_logs': 'low',  # Modify trail configuration
            'waf_logs': 'low',  # Change logging destination
            'route53_logs': 'medium',  # Requires query logging configuration
            'rds_logs': 'medium',  # Requires export task setup
            'lambda_logs': 'high'  # Requires subscription filter and Lambda function
        }
        
        return complexity_map.get(log_type, 'medium')
    
    async def get_vpc_flow_log_details(self, log_group_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about VPC Flow Logs configuration.
        
        Args:
            log_group_name: Name of the VPC Flow Logs log group
            
        Returns:
            Dict with VPC Flow Log details or None if not found
        """
        try:
            # Find VPC Flow Logs that use this log group
            response = self.ec2_client.describe_flow_logs(
                Filters=[
                    {
                        'Name': 'log-destination-type',
                        'Values': ['cloud-watch-logs']
                    }
                ]
            )
            
            for flow_log in response.get('FlowLogs', []):
                if flow_log.get('LogGroupName') == log_group_name:
                    return {
                        'flow_log_id': flow_log.get('FlowLogId'),
                        'resource_type': flow_log.get('ResourceType'),
                        'resource_id': flow_log.get('ResourceId'),
                        'traffic_type': flow_log.get('TrafficType'),
                        'log_format': flow_log.get('LogFormat'),
                        'max_aggregation_interval': flow_log.get('MaxAggregationInterval'),
                        'can_migrate_to_s3': True,
                        'migration_note': 'Create new Flow Log with S3 destination, then delete CloudWatch Flow Log'
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get VPC Flow Log details for {log_group_name}: {str(e)}")
            return None
