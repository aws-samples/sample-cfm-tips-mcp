"""
EC2 Right Sizing Playbook

This module implements the EC2 Right Sizing playbook from AWS Cost Optimization Playbooks.
It provides functions to identify and recommend right-sizing opportunities for EC2 instances.
"""

import logging
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from services.compute_optimizer import get_ec2_recommendations
from services.trusted_advisor import get_trusted_advisor_checks

logger = logging.getLogger(__name__)

def get_underutilized_instances(
    region: Optional[str] = None,
    lookback_period_days: int = 14,
    cpu_threshold: float = 40.0,
    memory_threshold: Optional[float] = None,
    network_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Identify underutilized EC2 instances using multiple data sources with fallback logic.
    Priority: 1) Compute Optimizer 2) Trusted Advisor 3) CloudWatch direct
    """
    
    # Try Compute Optimizer first (primary)
    try:
        logger.info("Attempting EC2 analysis with Compute Optimizer")
        result = _get_instances_from_compute_optimizer(region, lookback_period_days)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Compute Optimizer"
            return result
    except Exception as e:
        logger.warning(f"Compute Optimizer failed: {str(e)}")
    
    # Try Trusted Advisor (secondary)
    try:
        logger.info("Attempting EC2 analysis with Trusted Advisor")
        result = _get_instances_from_trusted_advisor(region)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Trusted Advisor"
            return result
    except Exception as e:
        logger.warning(f"Trusted Advisor failed: {str(e)}")
    
    # Try CloudWatch direct (tertiary)
    try:
        logger.info("Attempting EC2 analysis with CloudWatch")
        result = _get_instances_from_cloudwatch(region, lookback_period_days, cpu_threshold)
        result["data_source"] = "CloudWatch"
        return result
    except Exception as e:
        logger.error(f"All data sources failed. CloudWatch error: {str(e)}")
        return {
            "status": "error",
            "message": f"All data sources unavailable. Last error: {str(e)}",
            "attempted_sources": ["Compute Optimizer", "Trusted Advisor", "CloudWatch"]
        }

def _get_instances_from_compute_optimizer(region: Optional[str], lookback_period_days: int) -> Dict[str, Any]:
    """Get underutilized instances from Compute Optimizer"""
    recommendations_result = get_ec2_recommendations(region=region)
    
    if recommendations_result["status"] != "success":
        raise Exception("Compute Optimizer not available")
        
    recommendations = recommendations_result["data"].get("instanceRecommendations", [])
    underutilized_instances = []
    
    for rec in recommendations:
        if rec.get('finding') in ['Underprovisioned', 'Overprovisioned']:
            instance_details = {
                'instance_id': rec.get('instanceArn', '').split('/')[-1] if rec.get('instanceArn') else 'unknown',
                'instance_type': rec.get('currentInstanceType', 'unknown'),
                'finding': rec.get('finding', 'unknown'),
                'lookback_period_days': lookback_period_days
            }
            
            if rec.get('recommendationOptions'):
                option = rec['recommendationOptions'][0]
                instance_details['recommendation'] = {
                    'recommended_instance_type': option.get('instanceType', 'unknown'),
                    'estimated_monthly_savings': option.get('estimatedMonthlySavings', {}).get('value', 0)
                }
            
            underutilized_instances.append(instance_details)
    
    total_monthly_savings = sum(
        instance.get('recommendation', {}).get('estimated_monthly_savings', 0)
        for instance in underutilized_instances
    )
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances),
            "total_monthly_savings": total_monthly_savings
        },
        "message": f"Found {len(underutilized_instances)} underutilized EC2 instances via Compute Optimizer"
    }

def _get_instances_from_trusted_advisor(region: Optional[str]) -> Dict[str, Any]:
    """Get underutilized instances from Trusted Advisor"""
    ta_result = get_trusted_advisor_checks(["cost_optimizing"])
    
    if ta_result["status"] != "success":
        raise Exception("Trusted Advisor not available")
    
    underutilized_instances = []
    checks = ta_result["data"].get("checks", [])
    
    for check in checks:
        if "Low Utilization Amazon EC2 Instances" in check.get('name', ''):
            resources = check.get('result', {}).get('flaggedResources', [])
            for resource in resources:
                instance_details = {
                    'instance_id': resource.get('resourceId', 'unknown'),
                    'instance_type': resource.get('metadata', {}).get('Instance Type', 'unknown'),
                    'finding': 'Low Utilization',
                    'avg_cpu_utilization': float(resource.get('metadata', {}).get('Average CPU Utilization', '0').replace('%', '')),
                    'recommendation': {
                        'action': 'Consider downsizing or terminating',
                        'estimated_monthly_savings': 50  # Placeholder estimate
                    }
                }
                underutilized_instances.append(instance_details)
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances),
            "total_monthly_savings": len(underutilized_instances) * 50
        },
        "message": f"Found {len(underutilized_instances)} underutilized EC2 instances via Trusted Advisor"
    }

def _get_instances_from_cloudwatch(region: Optional[str], lookback_period_days: int, cpu_threshold: float) -> Dict[str, Any]:
    """Get underutilized instances from CloudWatch metrics directly"""
    if region:
        ec2_client = boto3.client('ec2', region_name=region)
        cloudwatch_client = boto3.client('cloudwatch', region_name=region)
    else:
        ec2_client = boto3.client('ec2')
        cloudwatch_client = boto3.client('cloudwatch')
        
    response = ec2_client.describe_instances(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_period_days)
    underutilized_instances = []
    
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            instance_type = instance['InstanceType']
            
            try:
                cpu_response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName='CPUUtilization',
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,
                    Statistics=['Average']
                )
                
                if cpu_response['Datapoints']:
                    avg_cpu = sum(dp['Average'] for dp in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])
                    
                    if avg_cpu < cpu_threshold:
                        underutilized_instances.append({
                            'instance_id': instance_id,
                            'instance_type': instance_type,
                            'avg_cpu_utilization': round(avg_cpu, 2),
                            'finding': 'Low CPU Utilization',
                            'recommendation': {
                                'action': 'Consider right-sizing',
                                'estimated_monthly_savings': 30  # Placeholder estimate
                            }
                        })
            except Exception:
                continue
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances),
            "total_monthly_savings": len(underutilized_instances) * 30
        },
        "message": f"Found {len(underutilized_instances)} underutilized EC2 instances via CloudWatch"
    }

def get_right_sizing_recommendation(
    instance_type: str,
    avg_cpu_utilization: float,
    region: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get right-sizing recommendation for an EC2 instance.
    
    Args:
        instance_type: Current instance type
        avg_cpu_utilization: Average CPU utilization percentage
        region: AWS region
        
    Returns:
        Dictionary containing right-sizing recommendation
    """
    try:
        # Use simplified pricing for now
        pricing_client = None
            
        # Get current instance pricing
        current_price = get_instance_price(instance_type, region, pricing_client)
        if not current_price:
            return None
            
        # Determine target instance family
        instance_family = instance_type.split('.')[0]
        
        # Simple right-sizing logic based on CPU utilization
        if avg_cpu_utilization < 20:
            # Recommend downsizing by 2 sizes within same family
            target_size = downsize_instance(instance_type, steps=2)
        elif avg_cpu_utilization < 40:
            # Recommend downsizing by 1 size within same family
            target_size = downsize_instance(instance_type, steps=1)
        else:
            # No recommendation needed
            return None
            
        if not target_size:
            return None
            
        # Get target instance pricing
        target_price = get_instance_price(target_size, region, pricing_client)
        if not target_price:
            return None
            
        # Calculate savings
        monthly_hours = 730  # Average hours in a month
        monthly_savings = (current_price - target_price) * monthly_hours
        
        return {
            "current_instance_type": instance_type,
            "recommended_instance_type": target_size,
            "current_hourly_cost": current_price,
            "recommended_hourly_cost": target_price,
            "estimated_monthly_savings": monthly_savings,
            "recommendation_reason": f"Average CPU utilization is {avg_cpu_utilization:.2f}%, which is below the threshold"
        }
        
    except Exception as e:
        logger.error(f"Error getting right-sizing recommendation: {str(e)}")
        return None

def get_instance_price(
    instance_type: str,
    region: Optional[str] = None,
    pricing_client = None
) -> Optional[float]:
    """Get EC2 instance price from AWS Price List API."""
    try:
        from services.pricing import get_ec2_pricing
        
        pricing_result = get_ec2_pricing(
            instance_type=instance_type,
            region=region or 'us-east-1'
        )
        
        if pricing_result.get('status') == 'success':
            return pricing_result.get('hourly_price', 0.1)
        else:
            # Fallback to estimate
            price_map = {
                't2.micro': 0.0116, 't2.small': 0.023, 't2.medium': 0.046,
                't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
                'm5.large': 0.096, 'm5.xlarge': 0.192
            }
            return price_map.get(instance_type, 0.1)
            
    except Exception as e:
        logger.warning(f"Error getting pricing for {instance_type}: {str(e)}")
        return 0.1  # Default fallback

def downsize_instance(
    instance_type: str,
    steps: int = 1
) -> Optional[str]:
    """
    Downsize an EC2 instance type by a number of steps.
    
    Args:
        instance_type: Current instance type
        steps: Number of steps to downsize
        
    Returns:
        Downsized instance type
    """
    # Instance size hierarchy (from largest to smallest)
    size_hierarchy = {
        'general': ['metal', '48xlarge', '32xlarge', '24xlarge', '18xlarge', '16xlarge', '12xlarge', '9xlarge', '8xlarge', '6xlarge', '4xlarge', '3xlarge', '2xlarge', 'xlarge', 'large', 'medium', 'small', 'micro', 'nano'],
        'compute': ['metal', '48xlarge', '32xlarge', '24xlarge', '18xlarge', '16xlarge', '12xlarge', '9xlarge', '8xlarge', '6xlarge', '4xlarge', '3xlarge', '2xlarge', 'xlarge', 'large', 'medium', 'small', 'micro', 'nano'],
        'memory': ['metal', '48xlarge', '32xlarge', '24xlarge', '18xlarge', '16xlarge', '12xlarge', '9xlarge', '8xlarge', '6xlarge', '4xlarge', '3xlarge', '2xlarge', 'xlarge', 'large', 'medium', 'small', 'micro', 'nano'],
        'storage': ['metal', '48xlarge', '32xlarge', '24xlarge', '18xlarge', '16xlarge', '12xlarge', '9xlarge', '8xlarge', '6xlarge', '4xlarge', '3xlarge', '2xlarge', 'xlarge', 'large', 'medium', 'small', 'micro', 'nano'],
        'accelerated': ['metal', '48xlarge', '32xlarge', '24xlarge', '18xlarge', '16xlarge', '12xlarge', '9xlarge', '8xlarge', '6xlarge', '4xlarge', '3xlarge', '2xlarge', 'xlarge', 'large', 'medium', 'small', 'micro', 'nano']
    }
    
    try:
        # Parse instance type
        parts = instance_type.split('.')
        if len(parts) != 2:
            return None
            
        family = parts[0]
        size = parts[1]
        
        # Determine instance category
        category = 'general'
        if family.startswith('c'):
            category = 'compute'
        elif family.startswith('r') or family.startswith('x'):
            category = 'memory'
        elif family.startswith('d') or family.startswith('i'):
            category = 'storage'
        elif family.startswith('p') or family.startswith('g'):
            category = 'accelerated'
            
        # Find current size index
        hierarchy = size_hierarchy[category]
        if size not in hierarchy:
            return None
            
        current_index = hierarchy.index(size)
        
        # Calculate target index
        target_index = current_index + steps
        if target_index >= len(hierarchy):
            return None
            
        # Get target size
        target_size = hierarchy[target_index]
        
        return f"{family}.{target_size}"
        
    except Exception as e:
        logger.error(f"Error downsizing instance: {str(e)}")
        return None

def get_stopped_instances(
    region: Optional[str] = None,
    min_stopped_days: int = 7
) -> Dict[str, Any]:
    """Identify stopped EC2 instances that could be terminated."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
        
        # Use paginator for EC2 describe_instances
        paginator = ec2_client.get_paginator('describe_instances')
        page_iterator = paginator.paginate(
            Filters=[{'Name': 'instance-state-name', 'Values': ['stopped']}]
        )
        
        stopped_instances = []
        
        # Process each page of results
        for page in page_iterator:
            for reservation in page['Reservations']:
                for instance in reservation['Instances']:
                    instance_details = {
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'state': instance['State']['Name'],
                        'launch_time': instance.get('LaunchTime', '').isoformat() if instance.get('LaunchTime') else '',
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
                        'recommendation': 'Consider terminating if no longer needed'
                    }
                    stopped_instances.append(instance_details)
        
        return {
            "status": "success",
            "data": {
                "stopped_instances": stopped_instances,
                "count": len(stopped_instances)
            },
            "message": f"Found {len(stopped_instances)} stopped EC2 instances"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_unattached_elastic_ips(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify unattached Elastic IP addresses."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
        
        # Note: describe_addresses doesn't support pagination via paginator,
        unattached_eips = []
        next_token = None
        
        while True:
            # Prepare pagination parameters
            params = {}
            if next_token:
                params['NextToken'] = next_token
                
            # Make the API call
            response = ec2_client.describe_addresses(**params)
            
            # Process results
            for address in response['Addresses']:
                if 'InstanceId' not in address and 'NetworkInterfaceId' not in address:
                    eip_details = {
                        'allocation_id': address.get('AllocationId', 'unknown'),
                        'public_ip': address.get('PublicIp', 'unknown'),
                        'domain': address.get('Domain', 'unknown'),
                        'tags': {tag['Key']: tag['Value'] for tag in address.get('Tags', [])},
                        'monthly_cost': 3.65,
                        'recommendation': 'Release if not needed'
                    }
                    unattached_eips.append(eip_details)
            
            # Check if there are more results
            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break
        
        total_monthly_cost = len(unattached_eips) * 3.65
        
        return {
            "status": "success",
            "data": {
                "unattached_eips": unattached_eips,
                "count": len(unattached_eips),
                "total_monthly_cost": total_monthly_cost
            },
            "message": f"Found {len(unattached_eips)} unattached Elastic IPs costing ${total_monthly_cost:.2f}/month"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_old_generation_instances(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify old generation EC2 instances."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'stopped']}]
        )
        
        old_generations = ['t1', 't2', 'm1', 'm2', 'm3', 'c1', 'c3', 'r3', 'i2', 'hs1']
        old_instances = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_type = instance['InstanceType']
                instance_family = instance_type.split('.')[0]
                
                if instance_family in old_generations:
                    modern_equivalent = _get_modern_equivalent(instance_family)
                    
                    instance_details = {
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance_type,
                        'instance_family': instance_family,
                        'state': instance['State']['Name'],
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
                        'recommendation': {
                            'action': 'Upgrade to modern generation',
                            'suggested_family': modern_equivalent,
                            'benefits': 'Better performance, lower cost'
                        }
                    }
                    old_instances.append(instance_details)
        
        return {
            "status": "success",
            "data": {
                "old_generation_instances": old_instances,
                "count": len(old_instances)
            },
            "message": f"Found {len(old_instances)} old generation EC2 instances"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _get_modern_equivalent(old_family: str) -> str:
    """Get modern equivalent for old instance family."""
    equivalents = {
        't1': 't3', 't2': 't3', 'm1': 'm5', 'm2': 'm5', 'm3': 'm5',
        'c1': 'c5', 'c3': 'c5', 'r3': 'r5', 'i2': 'i3', 'hs1': 'd3'
    }
    return equivalents.get(old_family, 'm5')

def get_instances_without_detailed_monitoring(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify instances without detailed monitoring."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        instances_without_monitoring = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                monitoring_state = instance.get('Monitoring', {}).get('State', 'disabled')
                
                if monitoring_state != 'enabled':
                    instance_details = {
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'monitoring_state': monitoring_state,
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
                        'recommendation': {
                            'action': 'Enable detailed monitoring',
                            'benefit': 'Better insights for right-sizing',
                            'additional_cost': '$2.10/month per instance'
                        }
                    }
                    instances_without_monitoring.append(instance_details)
        
        return {
            "status": "success",
            "data": {
                "instances_without_monitoring": instances_without_monitoring,
                "count": len(instances_without_monitoring)
            },
            "message": f"Found {len(instances_without_monitoring)} instances without detailed monitoring"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_right_sizing_report(
    underutilized_instances: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a comprehensive right-sizing report.
    
    Args:
        underutilized_instances: List of underutilized instances
        
    Returns:
        Dictionary containing the right-sizing report
    """
    try:
        # Calculate total savings
        total_monthly_savings = sum(
            instance.get('recommendation', {}).get('estimated_monthly_savings', 0)
            for instance in underutilized_instances
        )
        
        # Group by instance family
        family_savings = {}
        for instance in underutilized_instances:
            if 'recommendation' not in instance:
                continue
                
            instance_type = instance['instance_type']
            family = instance_type.split('.')[0]
            
            if family not in family_savings:
                family_savings[family] = {'count': 0, 'savings': 0}
                
            family_savings[family]['count'] += 1
            family_savings[family]['savings'] += instance['recommendation'].get('estimated_monthly_savings', 0)
            
        # Sort instances by savings potential
        sorted_instances = sorted(
            [i for i in underutilized_instances if 'recommendation' in i],
            key=lambda x: x['recommendation'].get('estimated_monthly_savings', 0),
            reverse=True
        )
        
        # Generate top recommendations
        top_recommendations = sorted_instances[:10] if len(sorted_instances) > 10 else sorted_instances
        
        return {
            "status": "success",
            "data": {
                "total_instances": len(underutilized_instances),
                "total_monthly_savings": total_monthly_savings,
                "family_savings": family_savings,
                "top_recommendations": top_recommendations
            },
            "message": f"Generated right-sizing report with potential monthly savings of ${total_monthly_savings:.2f}"
        }
        
    except Exception as e:
        logger.error(f"Error generating right-sizing report: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating right-sizing report: {str(e)}"
        }

# Additional EC2 Cost Framework Playbooks

def get_graviton_compatible_instances(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify instances that can be migrated to Graviton processors."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        graviton_compatible = []
        x86_families = ['m5', 'm4', 'c5', 'c4', 'r5', 'r4', 't3', 't2']
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_type = instance['InstanceType']
                family = instance_type.split('.')[0]
                
                if family in x86_families:
                    graviton_equivalent = _get_graviton_equivalent(family)
                    if graviton_equivalent:
                        graviton_compatible.append({
                            'instance_id': instance['InstanceId'],
                            'current_type': instance_type,
                            'graviton_equivalent': graviton_equivalent,
                            'estimated_savings': 0.2,
                            'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                        })
        
        return {
            "status": "success",
            "data": {
                "graviton_compatible_instances": graviton_compatible,
                "count": len(graviton_compatible)
            },
            "message": f"Found {len(graviton_compatible)} instances compatible with Graviton"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _get_graviton_equivalent(x86_family: str) -> Optional[str]:
    """Get Graviton equivalent for x86 instance family."""
    graviton_map = {
        'm5': 'm6g', 'm4': 'm6g', 'c5': 'c6g', 'c4': 'c6g',
        'r5': 'r6g', 'r4': 'r6g', 't3': 't4g', 't2': 't4g'
    }
    return graviton_map.get(x86_family)

def get_burstable_instances_analysis(
    region: Optional[str] = None,
    lookback_period_days: int = 14
) -> Dict[str, Any]:
    """Analyze burstable instances for credit usage and optimization."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
            cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            cloudwatch_client = boto3.client('cloudwatch')
            
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running']},
                {'Name': 'instance-type', 'Values': ['t2.*', 't3.*', 't4g.*']}
            ]
        )
        
        burstable_analysis = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_period_days)
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']
                
                try:
                    credit_response = cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/EC2',
                        MetricName='CPUCreditBalance',
                        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=['Average']
                    )
                    
                    if credit_response['Datapoints']:
                        avg_credits = sum(dp['Average'] for dp in credit_response['Datapoints']) / len(credit_response['Datapoints'])
                        
                        recommendation = 'optimal'
                        if avg_credits < 50:
                            recommendation = 'consider_unlimited_or_larger'
                        elif avg_credits > 500:
                            recommendation = 'consider_smaller_or_standard'
                            
                        burstable_analysis.append({
                            'instance_id': instance_id,
                            'instance_type': instance_type,
                            'avg_credit_balance': round(avg_credits, 2),
                            'recommendation': recommendation
                        })
                    else:
                        logger.info(f"No credit balance data found for instance {instance_id}")
                        
                except Exception as e:
                    logger.warning(f"Error retrieving credit balance for instance {instance_id}: {str(e)}")
                    # Add instance with error information for better tracking
                    burstable_analysis.append({
                        'instance_id': instance_id,
                        'instance_type': instance_type,
                        'error': f"Failed to retrieve credit data: {str(e)}",
                        'recommendation': 'manual_review_needed'
                    })
        
        return {
            "status": "success",
            "data": {
                "burstable_instances": burstable_analysis,
                "count": len(burstable_analysis)
            },
            "message": f"Analyzed {len(burstable_analysis)} burstable instances"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_spot_instance_opportunities(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify instances suitable for Spot pricing."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        spot_opportunities = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                
                is_fault_tolerant = (
                    'batch' in str(tags).lower() or
                    'dev' in str(tags).lower() or
                    'test' in str(tags).lower() or
                    instance.get('RootDeviceType') == 'instance-store'
                )
                
                if is_fault_tolerant:
                    spot_opportunities.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'estimated_savings': 0.7,
                        'tags': tags,
                        'reason': 'fault_tolerant_workload'
                    })
        
        return {
            "status": "success",
            "data": {
                "spot_opportunities": spot_opportunities,
                "count": len(spot_opportunities)
            },
            "message": f"Found {len(spot_opportunities)} instances suitable for Spot pricing"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_unused_capacity_reservations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify unused On-Demand Capacity Reservations."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_capacity_reservations(
            Filters=[{'Name': 'state', 'Values': ['active']}]
        )
        
        unused_reservations = []
        
        for reservation in response['CapacityReservations']:
            if reservation['AvailableInstanceCount'] == reservation['TotalInstanceCount']:
                unused_reservations.append({
                    'reservation_id': reservation['CapacityReservationId'],
                    'instance_type': reservation['InstanceType'],
                    'instance_count': reservation['TotalInstanceCount'],
                    'availability_zone': reservation['AvailabilityZone'],
                    'monthly_cost': reservation['TotalInstanceCount'] * 100
                })
        
        return {
            "status": "success",
            "data": {
                "unused_reservations": unused_reservations,
                "count": len(unused_reservations),
                "total_monthly_waste": sum(r['monthly_cost'] for r in unused_reservations)
            },
            "message": f"Found {len(unused_reservations)} unused capacity reservations"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_scheduling_opportunities(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Identify instances suitable for scheduling optimization."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        scheduling_opportunities = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                
                is_schedulable = (
                    'dev' in str(tags).lower() or
                    'test' in str(tags).lower() or
                    'staging' in str(tags).lower()
                )
                
                if is_schedulable:
                    scheduling_opportunities.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'estimated_savings': 0.6,
                        'tags': tags,
                        'recommendation': 'implement_start_stop_schedule'
                    })
        
        return {
            "status": "success",
            "data": {
                "scheduling_opportunities": scheduling_opportunities,
                "count": len(scheduling_opportunities)
            },
            "message": f"Found {len(scheduling_opportunities)} instances suitable for scheduling"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_commitment_plan_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze instances for Reserved Instance and Savings Plans opportunities."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        
        commitment_opportunities = []
        instance_usage = {}
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_type = instance['InstanceType']
                launch_time = instance['LaunchTime'].replace(tzinfo=None)
                uptime_days = (datetime.utcnow() - launch_time).days
                
                if instance_type not in instance_usage:
                    instance_usage[instance_type] = []
                    
                instance_usage[instance_type].append({
                    'instance_id': instance['InstanceId'],
                    'uptime_days': uptime_days
                })
        
        for instance_type, instances in instance_usage.items():
            stable_instances = [i for i in instances if i['uptime_days'] > 30]
            
            if len(stable_instances) >= 1:
                commitment_opportunities.append({
                    'instance_type': instance_type,
                    'instance_count': len(stable_instances),
                    'recommendation': 'reserved_instance_or_savings_plan',
                    'estimated_savings': 0.3,
                    'commitment_term': '1_year'
                })
        
        return {
            "status": "success",
            "data": {
                "commitment_opportunities": commitment_opportunities,
                "count": len(commitment_opportunities)
            },
            "message": f"Found {len(commitment_opportunities)} commitment opportunities"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_governance_violations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Detect EC2 governance violations and policy non-compliance."""
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        response = ec2_client.describe_instances()
        violations = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                
                required_tags = ['Environment', 'Owner', 'Project']
                missing_tags = [tag for tag in required_tags if tag not in tags]
                
                if missing_tags:
                    violations.append({
                        'instance_id': instance['InstanceId'],
                        'violation_type': 'missing_required_tags',
                        'missing_tags': missing_tags,
                        'severity': 'medium'
                    })
                
                if instance['InstanceType'].startswith(('x1', 'r5.24xlarge', 'm5.24xlarge')):
                    violations.append({
                        'instance_id': instance['InstanceId'],
                        'violation_type': 'oversized_instance',
                        'instance_type': instance['InstanceType'],
                        'severity': 'high'
                    })
        
        return {
            "status": "success",
            "data": {
                "violations": violations,
                "count": len(violations)
            },
            "message": f"Found {len(violations)} governance violations"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_comprehensive_ec2_report(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """Generate comprehensive EC2 optimization report covering all playbooks."""
    try:
        report_sections = {
            'underutilized_instances': get_underutilized_instances(region),
            'stopped_instances': get_stopped_instances(region),
            'unattached_eips': get_unattached_elastic_ips(region),
            'old_generation': get_old_generation_instances(region),
            'graviton_compatible': get_graviton_compatible_instances(region),
            'burstable_analysis': get_burstable_instances_analysis(region),
            'spot_opportunities': get_spot_instance_opportunities(region),
            'unused_reservations': get_unused_capacity_reservations(region),
            'scheduling_opportunities': get_scheduling_opportunities(region),
            'commitment_opportunities': get_commitment_plan_recommendations(region),
            'governance_violations': get_governance_violations(region)
        }
        
        total_savings = 0
        for section_name, section_data in report_sections.items():
            if section_data.get('status') == 'success':
                data = section_data.get('data', {})
                if 'total_monthly_savings' in data:
                    total_savings += data['total_monthly_savings']
        
        return {
            "status": "success",
            "data": {
                "report_sections": report_sections,
                "total_estimated_monthly_savings": total_savings,
                "region": region
            },
            "message": f"Generated comprehensive EC2 optimization report with ${total_savings:.2f} potential monthly savings"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating comprehensive report: {str(e)}"
        }
