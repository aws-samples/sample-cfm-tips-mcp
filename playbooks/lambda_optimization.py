"""
Lambda Optimization Playbook

This module implements the Lambda Optimization playbook from AWS Cost Optimization Playbooks.
"""

import logging
import boto3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from services.compute_optimizer import get_lambda_recommendations
from services.trusted_advisor import get_trusted_advisor_checks

logger = logging.getLogger(__name__)

def get_underutilized_lambda_functions(
    region: Optional[str] = None,
    lookback_period_days: int = 14,
    memory_utilization_threshold: float = 50.0,
    min_invocations: int = 100
) -> Dict[str, Any]:
    """
    Identify underutilized Lambda functions using multiple data sources with fallback logic.
    Priority: 1) Compute Optimizer 2) Trusted Advisor 3) CloudWatch direct
    """
    
    # Try Compute Optimizer first (primary)
    try:
        logger.info("Attempting Lambda analysis with Compute Optimizer")
        result = _get_lambda_from_compute_optimizer(region, lookback_period_days)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Compute Optimizer"
            return result
    except Exception as e:
        logger.warning(f"Compute Optimizer failed: {str(e)}")
    
    # Try Trusted Advisor (secondary)
    try:
        logger.info("Attempting Lambda analysis with Trusted Advisor")
        result = _get_lambda_from_trusted_advisor(region)
        if result["status"] == "success" and result["data"]["count"] > 0:
            result["data_source"] = "Trusted Advisor"
            return result
    except Exception as e:
        logger.warning(f"Trusted Advisor failed: {str(e)}")
    
    # Try CloudWatch direct (tertiary)
    try:
        logger.info("Attempting Lambda analysis with CloudWatch")
        result = _get_lambda_from_cloudwatch(region, lookback_period_days, min_invocations)
        result["data_source"] = "CloudWatch"
        return result
    except Exception as e:
        logger.error(f"All data sources failed. CloudWatch error: {str(e)}")
        return {
            "status": "error",
            "message": f"All data sources unavailable. Last error: {str(e)}",
            "attempted_sources": ["Compute Optimizer", "Trusted Advisor", "CloudWatch"]
        }

def _get_lambda_from_compute_optimizer(region: Optional[str], lookback_period_days: int) -> Dict[str, Any]:
    """Get underutilized Lambda functions from Compute Optimizer"""
    recommendations_result = get_lambda_recommendations(region=region)
    
    if recommendations_result["status"] != "success":
        raise Exception("Compute Optimizer not available")
        
    recommendations = recommendations_result["data"].get("lambdaFunctionRecommendations", [])
    analyzed_functions = []
    
    for rec in recommendations:
        if rec.get('finding') in ['Underprovisioned', 'Overprovisioned']:
            analyzed_functions.append({
                'function_name': rec.get('functionName', 'unknown'),
                'memory_size_mb': rec.get('currentMemorySize', 0),
                'finding': rec.get('finding', 'unknown'),
                'recommendation': {
                    'recommended_memory_size': rec.get('memorySizeRecommendationOptions', [{}])[0].get('memorySize', 0),
                    'estimated_monthly_savings': rec.get('memorySizeRecommendationOptions', [{}])[0].get('estimatedMonthlySavings', {}).get('value', 0)
                }
            })
    
    return {
        "status": "success",
        "data": {
            "analyzed_functions": analyzed_functions,
            "count": len(analyzed_functions)
        },
        "message": f"Found {len(analyzed_functions)} Lambda functions with optimization opportunities via Compute Optimizer"
    }

def _get_lambda_from_trusted_advisor(region: Optional[str]) -> Dict[str, Any]:
    """Get underutilized Lambda functions from Trusted Advisor"""
    ta_result = get_trusted_advisor_checks(["cost_optimizing"])
    
    if ta_result["status"] != "success":
        raise Exception("Trusted Advisor not available")
    
    analyzed_functions = []
    checks = ta_result["data"].get("checks", [])
    
    for check in checks:
        if "AWS Lambda Functions with High Error Rates" in check.get('name', '') or "Over-provisioned Lambda" in check.get('name', ''):
            resources = check.get('result', {}).get('flaggedResources', [])
            for resource in resources:
                analyzed_functions.append({
                    'function_name': resource.get('resourceId', 'unknown'),
                    'memory_size_mb': int(resource.get('metadata', {}).get('Memory Size', '0')),
                    'finding': 'Trusted Advisor flagged',
                    'recommendation': {
                        'action': 'Review memory allocation',
                        'estimated_monthly_savings': _calculate_lambda_savings(int(resource.get('metadata', {}).get('Memory Size', '128')))
                    }
                })
    
    return {
        "status": "success",
        "data": {
            "analyzed_functions": analyzed_functions,
            "count": len(analyzed_functions)
        },
        "message": f"Found {len(analyzed_functions)} Lambda functions with issues via Trusted Advisor"
    }

def _get_lambda_from_cloudwatch(region: Optional[str], lookback_period_days: int, min_invocations: int) -> Dict[str, Any]:
    """Get underutilized Lambda functions from CloudWatch metrics directly"""
    if region:
        lambda_client = boto3.client('lambda', region_name=region)
        cloudwatch_client = boto3.client('cloudwatch', region_name=region)
    else:
        lambda_client = boto3.client('lambda')
        cloudwatch_client = boto3.client('cloudwatch')
        
    response = lambda_client.list_functions()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_period_days)
    analyzed_functions = []
    
    for function in response['Functions']:
        function_name = function['FunctionName']
        
        try:
            # Get invocation metrics
            invocation_response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Sum']
            )
            
            # Get duration metrics for memory analysis
            duration_response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if invocation_response['Datapoints'] and duration_response['Datapoints']:
                total_invocations = sum(dp['Sum'] for dp in invocation_response['Datapoints'])
                avg_duration = sum(dp['Average'] for dp in duration_response['Datapoints']) / len(duration_response['Datapoints'])
                
                if total_invocations >= min_invocations:
                    # Simple heuristic: if duration is very low, might be over-provisioned
                    if avg_duration < 1000:  # Less than 1 second average
                        analyzed_functions.append({
                            'function_name': function_name,
                            'memory_size_mb': function['MemorySize'],
                            'total_invocations': int(total_invocations),
                            'avg_duration_ms': round(avg_duration, 2),
                            'finding': 'Potentially over-provisioned memory',
                            'recommendation': {
                                'action': 'Consider reducing memory allocation',
                                'estimated_monthly_savings': _calculate_lambda_savings(function['MemorySize'])
                            }
                        })
        except Exception:
            continue
    
    return {
        "status": "success",
        "data": {
            "analyzed_functions": analyzed_functions,
            "count": len(analyzed_functions)
        },
        "message": f"Analyzed {len(analyzed_functions)} Lambda functions via CloudWatch"
    }

def identify_unused_lambda_functions(
    region: Optional[str] = None,
    lookback_period_days: int = 30,
    max_invocations: int = 5
) -> Dict[str, Any]:
    """Identify unused Lambda functions."""
    try:
        if region:
            lambda_client = boto3.client('lambda', region_name=region)
            cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        else:
            lambda_client = boto3.client('lambda')
            cloudwatch_client = boto3.client('cloudwatch')
            
        response = lambda_client.list_functions()
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_period_days)
        
        unused_functions = []
        
        for function in response['Functions']:
            function_name = function['FunctionName']
            
            try:
                invocation_response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/Lambda',
                    MetricName='Invocations',
                    Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,
                    Statistics=['Sum']
                )
                
                total_invocations = 0
                if invocation_response['Datapoints']:
                    total_invocations = sum(dp['Sum'] for dp in invocation_response['Datapoints'])
                
                if total_invocations <= max_invocations:
                    unused_functions.append({
                        'function_name': function_name,
                        'memory_size_mb': function['MemorySize'],
                        'total_invocations': int(total_invocations),
                        'runtime': function.get('Runtime', '')
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting metrics for {function_name}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "data": {
                "unused_functions": unused_functions,
                "count": len(unused_functions)
            },
            "message": f"Found {len(unused_functions)} unused Lambda functions"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _calculate_lambda_savings(memory_size: int) -> float:
    """Calculate estimated Lambda savings."""
    try:
        from services.pricing import get_lambda_pricing
        
        pricing_result = get_lambda_pricing(memory_size)
        if pricing_result.get('status') == 'success':
            # Estimate savings based on memory optimization
            return (memory_size / 1024) * 0.0000166667 * 100000 * 0.3  # 30% savings estimate
        return 20
    except Exception:
        return 20