"""
EBS Optimization Playbook

This module implements the EBS Optimization playbook from AWS Cost Optimization Playbooks.
It provides functions to identify and recommend optimization opportunities for EBS volumes.
Includes both core optimization functions and MCP runbook functions.
"""

import asyncio
import json
import logging
import time
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from mcp.types import TextContent

from services.compute_optimizer import get_ebs_recommendations
from services.trusted_advisor import get_trusted_advisor_checks
from utils.error_handler import ResponseFormatter, handle_aws_error
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task
from utils.documentation_links import add_documentation_links

logger = logging.getLogger(__name__)

def get_underutilized_volumes(
    region: Optional[str] = None,
    lookback_period_days: int = 30,
    iops_threshold: float = 100.0,
    throughput_threshold: float = 1.0,  # MiB/s
    min_volume_size: int = 100  # GB
) -> Dict[str, Any]:
    """
    Identify underutilized EBS volumes using multiple data sources with fallback logic.
    Priority: 1) Compute Optimizer 2) Trusted Advisor 3) CloudWatch direct
    """
    
    # Try Compute Optimizer first (primary)
    try:
        logger.info("Attempting EBS analysis with Compute Optimizer")
        result = _get_volumes_from_compute_optimizer(region, lookback_period_days)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Compute Optimizer"
            return result
    except Exception as e:
        logger.warning(f"Compute Optimizer failed: {str(e)}")
    
    # Try Trusted Advisor (secondary)
    try:
        logger.info("Attempting EBS analysis with Trusted Advisor")
        result = _get_volumes_from_trusted_advisor(region)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Trusted Advisor"
            return result
    except Exception as e:
        logger.warning(f"Trusted Advisor failed: {str(e)}")
    
    # Try CloudWatch direct (tertiary)
    try:
        logger.info("Attempting EBS analysis with CloudWatch")
        result = _get_volumes_from_cloudwatch(region, lookback_period_days, iops_threshold)
        result["data_source"] = "CloudWatch"
        return result
    except Exception as e:
        logger.error(f"All data sources failed. CloudWatch error: {str(e)}")
        return {
            "status": "error",
            "message": f"All data sources unavailable. Last error: {str(e)}",
            "attempted_sources": ["Compute Optimizer", "Trusted Advisor", "CloudWatch"]
        }

def _get_volumes_from_compute_optimizer(region: Optional[str], lookback_period_days: int) -> Dict[str, Any]:
    """Get underutilized volumes from Compute Optimizer"""
    recommendations_result = get_ebs_recommendations(region=region)
    
    if recommendations_result["status"] != "success":
        raise Exception("Compute Optimizer not available")
        
    recommendations = recommendations_result["data"].get("volumeRecommendations", [])
    underutilized_volumes = []
    
    for rec in recommendations:
        if rec.get('finding') in ['Underprovisioned', 'Overprovisioned']:
            volume_details = {
                'volume_id': rec.get('volumeArn', '').split('/')[-1] if rec.get('volumeArn') else 'unknown',
                'volume_type': rec.get('currentConfiguration', {}).get('volumeType', 'unknown'),
                'volume_size': rec.get('currentConfiguration', {}).get('volumeSize', 0),
                'finding': rec.get('finding', 'unknown'),
                'lookback_period_days': lookback_period_days
            }
            
            if rec.get('volumeRecommendationOptions'):
                option = rec['volumeRecommendationOptions'][0]
                volume_details['recommendation'] = {
                    'recommended_volume_type': option.get('configuration', {}).get('volumeType', 'unknown'),
                    'estimated_monthly_savings': option.get('estimatedMonthlySavings', {}).get('value', 0)
                }
            
            underutilized_volumes.append(volume_details)
    
    total_monthly_savings = sum(
        volume.get('recommendation', {}).get('estimated_monthly_savings', 0)
        for volume in underutilized_volumes
    )
    
    return {
        "status": "success",
        "data": {
            "underutilized_volumes": underutilized_volumes,
            "count": len(underutilized_volumes),
            "total_monthly_savings": total_monthly_savings
        },
        "message": f"Found {len(underutilized_volumes)} underutilized EBS volumes via Compute Optimizer"
    }

def _get_volumes_from_trusted_advisor(region: Optional[str]) -> Dict[str, Any]:
    """Get underutilized volumes from Trusted Advisor"""
    ta_result = get_trusted_advisor_checks(["cost_optimizing"])
    
    if ta_result["status"] != "success":
        raise Exception("Trusted Advisor not available")
    
    underutilized_volumes = []
    checks = ta_result["data"].get("checks", [])
    
    for check in checks:
        if "Underutilized Amazon EBS Volumes" in check.get('name', ''):
            resources = check.get('result', {}).get('flaggedResources', [])
            for resource in resources:
                volume_details = {
                    'volume_id': resource.get('resourceId', 'unknown'),
                    'volume_type': resource.get('metadata', {}).get('Volume Type', 'unknown'),
                    'volume_size': int(resource.get('metadata', {}).get('Volume Size', '0')),
                    'finding': 'Underutilized',
                    'recommendation': {
                        'action': 'Consider downsizing or changing type',
                        'estimated_monthly_savings': 20  # Placeholder estimate
                    }
                }
                underutilized_volumes.append(volume_details)
    
    return {
        "status": "success",
        "data": {
            "underutilized_volumes": underutilized_volumes,
            "count": len(underutilized_volumes),
            "total_monthly_savings": len(underutilized_volumes) * 20
        },
        "message": f"Found {len(underutilized_volumes)} underutilized EBS volumes via Trusted Advisor"
    }

def _get_volumes_from_cloudwatch(region: Optional[str], lookback_period_days: int, iops_threshold: float) -> Dict[str, Any]:
    """Get underutilized volumes from CloudWatch metrics directly"""
    if region:
        ec2_client = boto3.client('ec2', region_name=region)
        cloudwatch_client = boto3.client('cloudwatch', region_name=region)
    else:
        ec2_client = boto3.client('ec2')
        cloudwatch_client = boto3.client('cloudwatch')
    
    # Use paginator for describe_volumes
    paginator = ec2_client.get_paginator('describe_volumes')
    page_iterator = paginator.paginate()
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_period_days)
    underutilized_volumes = []
    
    # Process each page of volumes
    for page in page_iterator:
        for volume in page['Volumes']:
            volume_id = volume['VolumeId']
            volume_type = volume['VolumeType']
            volume_size = volume['Size']
            
            try:
                # Get IOPS metrics
                iops_response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/EBS',
                    MetricName='VolumeReadOps',
                    Dimensions=[{'Name': 'VolumeId', 'Value': volume_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,
                    Statistics=['Sum']
                )
                
                if iops_response['Datapoints']:
                    total_ops = sum(dp['Sum'] for dp in iops_response['Datapoints'])
                    avg_iops = total_ops / len(iops_response['Datapoints']) / 86400
                    
                    if avg_iops < iops_threshold:
                        underutilized_volumes.append({
                            'volume_id': volume_id,
                            'volume_type': volume_type,
                            'volume_size': volume_size,
                            'avg_iops': round(avg_iops, 2),
                            'finding': 'Low IOPS Utilization',
                            'recommendation': {
                                'action': 'Consider gp3 or smaller size',
                                'estimated_monthly_savings': volume_size * 0.02  # Rough estimate
                            }
                        })
            except Exception:
                continue
    
    return {
        "status": "success",
        "data": {
            "underutilized_volumes": underutilized_volumes,
            "count": len(underutilized_volumes),
            "total_monthly_savings": sum(v.get('recommendation', {}).get('estimated_monthly_savings', 0) for v in underutilized_volumes)
        },
        "message": f"Found {len(underutilized_volumes)} underutilized EBS volumes via CloudWatch"
    }

def get_volume_optimization_recommendation(
    volume_type: str,
    volume_size: int,
    avg_iops: float,
    avg_throughput: float,
    provisioned_iops: int = 0,
    provisioned_throughput: int = 0,
    region: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get optimization recommendation for an EBS volume.
    
    Args:
        volume_type: Current volume type
        volume_size: Volume size in GB
        avg_iops: Average IOPS
        avg_throughput: Average throughput in MiB/s
        provisioned_iops: Provisioned IOPS (for io1/io2/gp3)
        provisioned_throughput: Provisioned throughput (for gp3)
        region: AWS region
        
    Returns:
        Dictionary containing optimization recommendation
    """
    try:
        # Use simplified pricing for now
        pricing_client = None
            
        # Get current volume pricing
        current_price = get_volume_price(
            volume_type=volume_type,
            volume_size=volume_size,
            provisioned_iops=provisioned_iops,
            provisioned_throughput=provisioned_throughput,
            region=region,
            pricing_client=pricing_client
        )
        
        if not current_price:
            return None
            
        # Determine recommended volume type and configuration
        recommended_type = None
        recommended_iops = None
        recommended_throughput = None
        recommendation_reason = None
        
        if volume_type == 'io1' or volume_type == 'io2':
            # For provisioned IOPS volumes
            if avg_iops < 3000 and avg_throughput < 125:
                # Can use gp3 with baseline performance
                recommended_type = 'gp3'
                recommended_iops = 3000
                recommended_throughput = 125
                recommendation_reason = f"Average IOPS ({avg_iops:.2f}) and throughput ({avg_throughput:.2f} MiB/s) are below gp3 baseline"
            else:
                # Need gp3 with custom performance
                recommended_type = 'gp3'
                recommended_iops = max(3000, min(16000, int(avg_iops * 1.2)))  # 20% headroom
                recommended_throughput = max(125, min(1000, int(avg_throughput * 1.2)))  # 20% headroom
                recommendation_reason = f"Can use gp3 with custom performance instead of {volume_type}"
        elif volume_type == 'gp3':
            # For gp3 volumes with custom performance
            if avg_iops < 3000 and avg_throughput < 125:
                # Can use gp3 with baseline performance
                recommended_type = 'gp3'
                recommended_iops = 3000
                recommended_throughput = 125
                recommendation_reason = f"Can reduce to baseline performance (3000 IOPS, 125 MiB/s)"
            else:
                # Optimize custom performance
                recommended_type = 'gp3'
                recommended_iops = max(3000, min(16000, int(avg_iops * 1.2)))  # 20% headroom
                recommended_throughput = max(125, min(1000, int(avg_throughput * 1.2)))  # 20% headroom
                recommendation_reason = f"Can optimize custom performance based on actual usage"
        elif volume_type == 'gp2':
            # For gp2 volumes
            if avg_iops < 3000 and avg_throughput < 125:
                # Can use gp3 with baseline performance
                recommended_type = 'gp3'
                recommended_iops = 3000
                recommended_throughput = 125
                recommendation_reason = f"Can migrate to gp3 with baseline performance for better cost efficiency"
            else:
                # Need gp3 with custom performance
                recommended_type = 'gp3'
                recommended_iops = max(3000, min(16000, int(avg_iops * 1.2)))  # 20% headroom
                recommended_throughput = max(125, min(1000, int(avg_throughput * 1.2)))  # 20% headroom
                recommendation_reason = f"Can migrate to gp3 with custom performance for better cost efficiency"
        else:
            # No recommendation for other volume types
            return None
            
        # Get recommended volume pricing
        recommended_price = get_volume_price(
            volume_type=recommended_type,
            volume_size=volume_size,
            provisioned_iops=recommended_iops,
            provisioned_throughput=recommended_throughput,
            region=region,
            pricing_client=pricing_client
        )
        
        if not recommended_price:
            return None
            
        # Calculate savings
        monthly_hours = 730  # Average hours in a month
        monthly_savings = (current_price - recommended_price) * monthly_hours
        
        return {
            "current_volume_type": volume_type,
            "recommended_volume_type": recommended_type,
            "current_monthly_cost": current_price * monthly_hours,
            "recommended_monthly_cost": recommended_price * monthly_hours,
            "estimated_monthly_savings": monthly_savings,
            "recommended_iops": recommended_iops,
            "recommended_throughput": recommended_throughput,
            "recommendation_reason": recommendation_reason
        }
        
    except Exception as e:
        logger.error(f"Error getting volume optimization recommendation: {str(e)}")
        return None

def get_volume_price(
    volume_type: str,
    volume_size: int,
    provisioned_iops: int = 0,
    provisioned_throughput: int = 0,
    region: Optional[str] = None,
    pricing_client = None
) -> Optional[float]:
    """Get EBS volume price from AWS Price List API."""
    try:
        from services.pricing import get_ebs_pricing
        
        pricing_result = get_ebs_pricing(
            volume_type=volume_type,
            volume_size=volume_size,
            region=region or 'us-east-1'
        )
        
        if pricing_result.get('status') == 'success':
            return pricing_result.get('hourly_price', 0.01)
        else:
            # Fallback pricing
            price_per_gb = {'gp2': 0.10, 'gp3': 0.08, 'io1': 0.125, 'io2': 0.125}.get(volume_type, 0.10)
            return (price_per_gb * volume_size) / 730
            
    except Exception as e:
        logger.warning(f"Error getting EBS pricing: {str(e)}")
        price_per_gb = {'gp2': 0.10, 'gp3': 0.08, 'io1': 0.125, 'io2': 0.125}.get(volume_type, 0.10)
        return (price_per_gb * volume_size) / 730

def identify_unused_volumes(
    region: Optional[str] = None,
    min_age_days: int = 30
) -> Dict[str, Any]:
    """
    Identify unused EBS volumes (unattached volumes).
    """
    try:
        if region:
            ec2_client = boto3.client('ec2', region_name=region)
        else:
            ec2_client = boto3.client('ec2')
            
        # Use paginator for describe_volumes
        paginator = ec2_client.get_paginator('describe_volumes')
        page_iterator = paginator.paginate(
            Filters=[{'Name': 'status', 'Values': ['available']}]
        )
        
        cutoff_date = datetime.utcnow() - timedelta(days=min_age_days)
        unused_volumes = []
        
        # Process each page of volumes
        for page in page_iterator:
            for volume in page['Volumes']:
                create_time = volume['CreateTime'].replace(tzinfo=None)
                if create_time < cutoff_date:
                    unused_volumes.append({
                        'volume_id': volume['VolumeId'],
                        'volume_type': volume['VolumeType'],
                        'volume_size': volume['Size'],
                        'age_days': (datetime.utcnow() - create_time).days,
                        'estimated_monthly_cost': get_volume_price(volume['VolumeType'], volume['Size']) * 730 if get_volume_price(volume['VolumeType'], volume['Size']) else volume['Size'] * 0.10
                    })
        
        return {
            "status": "success",
            "data": {
                "unused_volumes": unused_volumes,
                "count": len(unused_volumes),
                "total_monthly_savings": sum(v['estimated_monthly_cost'] for v in unused_volumes)
            },
            "message": f"Found {len(unused_volumes)} unused EBS volumes"
        }
        
    except Exception as e:
        logger.error(f"Error identifying unused volumes: {str(e)}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

def generate_ebs_optimization_report(
    underutilized_volumes: List[Dict[str, Any]],
    unused_volumes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a comprehensive EBS optimization report.
    
    Args:
        underutilized_volumes: List of underutilized volumes
        unused_volumes: List of unused volumes
        
    Returns:
        Dictionary containing the EBS optimization report
    """
    try:
        # Calculate total savings
        underutilized_savings = sum(
            volume.get('recommendation', {}).get('estimated_monthly_savings', 0)
            for volume in underutilized_volumes
        )
        
        unused_savings = sum(volume.get('monthly_cost', 0) for volume in unused_volumes)
        
        total_monthly_savings = underutilized_savings + unused_savings
        
        # Group by volume type
        type_savings = {}
        for volume in underutilized_volumes:
            if 'recommendation' not in volume:
                continue
                
            volume_type = volume['volume_type']
            
            if volume_type not in type_savings:
                type_savings[volume_type] = {
                    'count': 0,
                    'savings': 0
                }
                
            type_savings[volume_type]['count'] += 1
            type_savings[volume_type]['savings'] += volume['recommendation'].get('estimated_monthly_savings', 0)
            
        # Sort volumes by savings potential
        sorted_underutilized = sorted(
            [v for v in underutilized_volumes if 'recommendation' in v],
            key=lambda x: x['recommendation'].get('estimated_monthly_savings', 0),
            reverse=True
        )
        
        sorted_unused = sorted(
            unused_volumes,
            key=lambda x: x.get('monthly_cost', 0),
            reverse=True
        )
        
        # Generate top recommendations
        top_underutilized = sorted_underutilized[:5] if len(sorted_underutilized) > 5 else sorted_underutilized
        top_unused = sorted_unused[:5] if len(sorted_unused) > 5 else sorted_unused
        
        return {
            "status": "success",
            "data": {
                "total_volumes": len(underutilized_volumes) + len(unused_volumes),
                "underutilized_count": len(underutilized_volumes),
                "unused_count": len(unused_volumes),
                "total_monthly_savings": total_monthly_savings,
                "underutilized_savings": underutilized_savings,
                "unused_savings": unused_savings,
                "type_savings": type_savings,
                "top_underutilized": top_underutilized,
                "top_unused": top_unused
            },
            "message": f"Generated EBS optimization report with potential monthly savings of ${total_monthly_savings:.2f}"
        }
        
    except Exception as e:
        logger.error(f"Error generating EBS optimization report: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating EBS optimization report: {str(e)}"
        }
# MCP Runbook Functions
# These functions provide MCP-compatible interfaces for the EBS optimization playbook

@handle_aws_error
async def run_ebs_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive EBS optimization analysis with parallel execution and session storage."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 30)
        iops_threshold = arguments.get("iops_threshold", 100.0)
        throughput_threshold = arguments.get("throughput_threshold", 1.0)
        
        # Initialize service orchestrator for parallel execution and session management
        orchestrator = ServiceOrchestrator()
        
        # Define parallel service calls for EBS analysis
        service_calls = [
            {
                'service': 'ebs',
                'operation': 'underutilized_volumes',
                'function': get_underutilized_volumes,
                'args': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'iops_threshold': iops_threshold,
                    'throughput_threshold': throughput_threshold
                }
            },
            {
                'service': 'ebs',
                'operation': 'unused_volumes',
                'function': identify_unused_volumes,
                'args': {
                    'region': region,
                    'min_age_days': 30
                }
            }
        ]
        
        # Execute parallel analysis
        results = await orchestrator.execute_parallel_analysis(
            service_calls=service_calls,
            analysis_type="ebs_optimization"
        )
        
        # Add documentation links
        results = add_documentation_links(results, "ebs")
        
        execution_time = time.time() - start_time
        
        # Format response with metadata
        results["ebs_optimization"] = {
            "analysis_type": "comprehensive_ebs_optimization",
            "region": region,
            "lookback_period_days": lookback_period_days,
            "session_id": results.get("report_metadata", {}).get("session_id"),
            "parallel_execution": True,
            "sql_storage": True
        }
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=results,
                message="EBS optimization analysis completed successfully",
                analysis_type="ebs_optimization",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error in EBS optimization analysis: {str(e)}")
        raise


@handle_aws_error
async def identify_unused_ebs_volumes(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unused EBS volumes that can be deleted."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        min_age_days = arguments.get("min_age_days", 30)
        
        result = identify_unused_volumes(
            region=region,
            min_age_days=min_age_days
        )
        
        # Add documentation links
        result = add_documentation_links(result, "ebs")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=result,
                message=f"Found {len(result.get('unused_volumes', []))} unused EBS volumes",
                analysis_type="ebs_unused",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error identifying unused EBS volumes: {str(e)}")
        raise


@handle_aws_error
async def generate_ebs_optimization_report_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed EBS optimization report."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        include_cost_analysis = arguments.get("include_cost_analysis", True)
        output_format = arguments.get("output_format", "json")
        
        # Get data from playbooks
        underutilized_result = get_underutilized_volumes(region=region)
        unused_result = identify_unused_volumes(region=region)
        
        # Generate comprehensive report
        report = generate_ebs_optimization_report(
            underutilized_volumes=underutilized_result.get("underutilized_volumes", []),
            unused_volumes=unused_result.get("unused_volumes", [])
        )
        
        # Add documentation links
        report = add_documentation_links(report, "ebs")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=report,
                message="EBS optimization report generated successfully",
                analysis_type="ebs_report",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error generating EBS report: {str(e)}")
        raise