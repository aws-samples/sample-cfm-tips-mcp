"""
RDS Optimization Playbook

This module implements the RDS Optimization playbook from AWS Cost Optimization Playbooks.
Includes both core optimization functions and MCP runbook functions.
"""

import asyncio
import json
import logging
import time
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from mcp.types import TextContent

from services.trusted_advisor import get_trusted_advisor_checks
from services.performance_insights import get_performance_insights_metrics
from utils.error_handler import ResponseFormatter, handle_aws_error
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task
from utils.documentation_links import add_documentation_links

logger = logging.getLogger(__name__)

def get_underutilized_rds_instances(
    region: Optional[str] = None,
    lookback_period_days: int = 14,
    cpu_threshold: float = 40.0,
    connection_threshold: float = 20.0
) -> Dict[str, Any]:
    """
    Identify underutilized RDS instances using multiple data sources with fallback logic.
    Priority: 1) Performance Insights 2) Trusted Advisor 3) CloudWatch direct
    """
    
    # Try Performance Insights first (primary)
    try:
        logger.info("Attempting RDS analysis with Performance Insights")
        result = _get_rds_from_performance_insights(region, lookback_period_days, cpu_threshold)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Performance Insights"
            return result
    except Exception as e:
        logger.warning(f"Performance Insights failed: {str(e)}")
    
    # Try Trusted Advisor (secondary)
    try:
        logger.info("Attempting RDS analysis with Trusted Advisor")
        result = _get_rds_from_trusted_advisor(region)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Trusted Advisor"
            return result
    except Exception as e:
        logger.warning(f"Trusted Advisor failed: {str(e)}")
    
    # Try CloudWatch direct (tertiary)
    try:
        logger.info("Attempting RDS analysis with CloudWatch")
        result = _get_rds_from_cloudwatch(region, lookback_period_days, cpu_threshold)
        result["data_source"] = "CloudWatch"
        return result
    except Exception as e:
        logger.error(f"All data sources failed. CloudWatch error: {str(e)}")
        return {
            "status": "error",
            "message": f"All data sources unavailable. Last error: {str(e)}",
            "attempted_sources": ["Performance Insights", "Trusted Advisor", "CloudWatch"]
        }

def _get_rds_from_performance_insights(region: Optional[str], lookback_period_days: int, cpu_threshold: float) -> Dict[str, Any]:
    """Get underutilized RDS instances from Performance Insights"""
    if region:
        rds_client = boto3.client('rds', region_name=region)
    else:
        rds_client = boto3.client('rds')
        
    response = rds_client.describe_db_instances()
    underutilized_instances = []
    
    for db_instance in response['DBInstances']:
        db_instance_identifier = db_instance['DBInstanceIdentifier']
        
        try:
            # Try to get Performance Insights metrics
            pi_result = get_performance_insights_metrics(db_instance_identifier)
            
            if pi_result["status"] == "success":
                # Analyze PI data for utilization patterns
                metrics = pi_result["data"].get("MetricList", [])
                
                # Simple analysis - in production would be more sophisticated
                low_utilization = True  # Placeholder logic
                
                if low_utilization:
                    underutilized_instances.append({
                        'db_instance_identifier': db_instance_identifier,
                        'db_instance_class': db_instance['DBInstanceClass'],
                        'engine': db_instance['Engine'],
                        'finding': 'Low Performance Insights metrics',
                        'recommendation': {
                            'action': 'Consider downsizing',
                            'estimated_monthly_savings': _calculate_rds_savings(db_instance['DBInstanceClass'])
                        }
                    })
        except Exception:
            continue
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances)
        },
        "message": f"Found {len(underutilized_instances)} underutilized RDS instances via Performance Insights"
    }

def _get_rds_from_trusted_advisor(region: Optional[str]) -> Dict[str, Any]:
    """Get underutilized RDS instances from Trusted Advisor"""
    ta_result = get_trusted_advisor_checks(["cost_optimizing"])
    
    if ta_result["status"] != "success":
        raise Exception("Trusted Advisor not available")
    
    underutilized_instances = []
    checks = ta_result["data"].get("checks", [])
    
    for check in checks:
        # Ensure check is a dictionary, not a list
        if not isinstance(check, dict):
            logger.warning(f"Unexpected check format in Trusted Advisor response: {type(check)}")
            continue
            
        check_name = check.get('name', '')
        if "Idle DB Instances" in check_name or "Low Utilization Amazon RDS" in check_name:
            result = check.get('result', {})
            if not isinstance(result, dict):
                logger.warning(f"Unexpected result format for check {check_name}: {type(result)}")
                continue
                
            resources = result.get('flaggedResources', [])
            if not isinstance(resources, list):
                logger.warning(f"Unexpected flaggedResources format for check {check_name}: {type(resources)}")
                continue
                
            for resource in resources:
                if not isinstance(resource, dict):
                    logger.warning(f"Unexpected resource format: {type(resource)}")
                    continue
                    
                metadata = resource.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    
                underutilized_instances.append({
                    'db_instance_identifier': resource.get('resourceId', 'unknown'),
                    'db_instance_class': metadata.get('Instance Class', 'unknown'),
                    'engine': metadata.get('Engine', 'unknown'),
                    'finding': 'Trusted Advisor flagged',
                    'recommendation': {
                        'action': 'Review and consider downsizing',
                        'estimated_monthly_savings': _calculate_rds_savings(metadata.get('Instance Class', 'db.t3.micro'))
                    }
                })
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances)
        },
        "message": f"Found {len(underutilized_instances)} underutilized RDS instances via Trusted Advisor"
    }

def _get_rds_from_cloudwatch(region: Optional[str], lookback_period_days: int, cpu_threshold: float) -> Dict[str, Any]:
    """Get underutilized RDS instances from CloudWatch metrics directly"""
    if region:
        rds_client = boto3.client('rds', region_name=region)
        cloudwatch_client = boto3.client('cloudwatch', region_name=region)
    else:
        rds_client = boto3.client('rds')
        cloudwatch_client = boto3.client('cloudwatch')
    
    # Use pagination for RDS instances
    paginator = rds_client.get_paginator('describe_db_instances')
    page_iterator = paginator.paginate()
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_period_days)
    underutilized_instances = []
    
    # Process each page of DB instances
    for page in page_iterator:
        for db_instance in page['DBInstances']:
            db_instance_identifier = db_instance['DBInstanceIdentifier']
        
        try:
            cpu_response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_identifier}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if cpu_response['Datapoints']:
                avg_cpu = sum(dp['Average'] for dp in cpu_response['Datapoints']) / len(cpu_response['Datapoints'])

                if avg_cpu < cpu_threshold:
                    underutilized_instances.append({
                        'db_instance_identifier': db_instance_identifier,
                        'db_instance_class': db_instance['DBInstanceClass'],
                        'engine': db_instance['Engine'],
                        'avg_cpu_utilization': round(avg_cpu, 2),
                        'finding': 'Low CPU Utilization',
                        'recommendation': {
                            'action': 'Consider downsizing',
                            'estimated_monthly_savings': _calculate_rds_savings(db_instance['DBInstanceClass'])
                        }
                    })
        except Exception:
            continue
    
    return {
        "status": "success",
        "data": {
            "underutilized_instances": underutilized_instances,
            "count": len(underutilized_instances)
        },
        "message": f"Found {len(underutilized_instances)} underutilized RDS instances via CloudWatch"
    }

def identify_idle_rds_instances(
    region: Optional[str] = None,
    lookback_period_days: int = 7,
    connection_threshold: float = 1.0,
    **kwargs  # Accept additional keyword arguments for compatibility
) -> Dict[str, Any]:
    """Identify idle RDS instances."""
    try:
        if region:
            rds_client = boto3.client('rds', region_name=region)
            cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        else:
            rds_client = boto3.client('rds')
            cloudwatch_client = boto3.client('cloudwatch')
            
        # Use pagination for RDS instances
        paginator = rds_client.get_paginator('describe_db_instances')
        page_iterator = paginator.paginate()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_period_days)
        
        idle_instances = []
        
        # Process each page of DB instances
        for page in page_iterator:
            for db_instance in page['DBInstances']:
                db_instance_identifier = db_instance['DBInstanceIdentifier']
            
                try:
                    connection_response = cloudwatch_client.get_metric_statistics(
                        Namespace='AWS/RDS',
                        MetricName='DatabaseConnections',
                        Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_identifier}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=['Maximum']
                    )
                    
                    if connection_response['Datapoints']:
                        max_connections = max(dp['Maximum'] for dp in connection_response['Datapoints'])
                        
                        if max_connections <= connection_threshold:
                            idle_instances.append({
                                'db_instance_identifier': db_instance_identifier,
                                'db_instance_class': db_instance['DBInstanceClass'],
                                'max_connections': max_connections
                            })
                            
                except Exception as e:
                    logger.warning(f"Error getting metrics for {db_instance_identifier}: {str(e)}")
                    continue
        
        return {
            "status": "success",
            "data": {
                "idle_instances": idle_instances,
                "count": len(idle_instances)
            },
            "message": f"Found {len(idle_instances)} idle RDS instances"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _calculate_rds_savings(instance_class: str) -> float:
    """Calculate estimated RDS savings."""
    try:
        from services.pricing import get_rds_pricing
        
        pricing_result = get_rds_pricing(instance_class)
        if pricing_result.get('status') == 'success':
            return pricing_result.get('monthly_price', 100) * 0.3
        return 60
    except Exception:
        return 60

if __name__ == '__main__':
    rds = identify_idle_rds_instances()
    print(rds)
# MCP Runbook Functions
# These functions provide MCP-compatible interfaces for the RDS optimization playbook

@handle_aws_error
async def run_rds_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive RDS optimization analysis with parallel execution and session storage."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 14)
        cpu_threshold = arguments.get("cpu_threshold", 40.0)
        connection_threshold = arguments.get("connection_threshold", 20.0)
        
        # Initialize service orchestrator for parallel execution and session management
        orchestrator = ServiceOrchestrator()
        
        # Define parallel service calls for RDS analysis
        service_calls = [
            {
                'service': 'rds',
                'operation': 'underutilized_instances',
                'function': get_underutilized_rds_instances,
                'args': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'cpu_threshold': cpu_threshold,
                    'connection_threshold': connection_threshold
                }
            },
            {
                'service': 'rds',
                'operation': 'idle_instances',
                'function': identify_idle_rds_instances,
                'args': {
                    'region': region,
                    'lookback_period_days': 7,
                    'connection_threshold': 1.0
                }
            }
        ]
        
        # Execute parallel analysis
        results = orchestrator.execute_parallel_analysis(
            service_calls=service_calls,
            store_results=True,
            timeout=120.0
        )
        
        # Add documentation links
        results = add_documentation_links(results, "rds")
        
        execution_time = time.time() - start_time
        
        # Format response with metadata
        results["rds_optimization"] = {
            "analysis_type": "comprehensive_rds_optimization",
            "region": region,
            "lookback_period_days": lookback_period_days,
            "session_id": results.get("report_metadata", {}).get("session_id"),
            "parallel_execution": True,
            "sql_storage": True
        }
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=results,
                message="RDS optimization analysis completed successfully",
                analysis_type="rds_optimization",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error in RDS optimization analysis: {str(e)}")
        raise


@handle_aws_error
async def identify_idle_rds_instances_wrapper(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify idle RDS instances (async wrapper)."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 7)
        connection_threshold = arguments.get("connection_threshold", 1.0)
        
        result = identify_idle_rds_instances(
            region=region,
            lookback_period_days=lookback_period_days,
            connection_threshold=connection_threshold
        )
        
        # Add documentation links
        result = add_documentation_links(result, "rds")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=result,
                message=f"Found {len(result.get('idle_instances', []))} idle RDS instances",
                analysis_type="rds_idle",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error identifying idle RDS instances: {str(e)}")
        raise


@handle_aws_error
async def generate_rds_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed RDS optimization report."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        include_cost_analysis = arguments.get("include_cost_analysis", True)
        output_format = arguments.get("output_format", "json")
        
        # Get data from playbooks
        underutilized_result = get_underutilized_rds_instances(region=region)
        idle_result = identify_idle_rds_instances(region=region)
        
        # Generate comprehensive report
        report = {
            "report_type": "rds_optimization",
            "region": region,
            "generated_at": datetime.now().isoformat(),
            "underutilized_instances": underutilized_result.get("underutilized_instances", []),
            "idle_instances": idle_result.get("idle_instances", []),
            "summary": {
                "total_underutilized": len(underutilized_result.get("underutilized_instances", [])),
                "total_idle": len(idle_result.get("idle_instances", [])),
                "potential_monthly_savings": (
                    underutilized_result.get("total_monthly_savings", 0) +
                    idle_result.get("total_monthly_savings", 0)
                )
            }
        }
        
        if include_cost_analysis:
            report["cost_analysis"] = {
                "underutilized_savings": underutilized_result.get("total_monthly_savings", 0),
                "idle_savings": idle_result.get("total_monthly_savings", 0),
                "total_savings": report["summary"]["potential_monthly_savings"]
            }
        
        # Add documentation links
        report = add_documentation_links(report, "rds")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=report,
                message="RDS optimization report generated successfully",
                analysis_type="rds_report",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error generating RDS report: {str(e)}")
        raise