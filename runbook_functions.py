"""
Runbook Functions for AWS Cost Optimization

This module contains all the runbook/playbook functions for cost optimization analysis.

"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
from mcp.types import TextContent

# Import utility components
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task

# Import playbook modules
from playbooks.ec2.ec2_optimization import (
    get_underutilized_instances, get_right_sizing_recommendation, generate_right_sizing_report,
    get_stopped_instances, get_unattached_elastic_ips, get_old_generation_instances,
    get_instances_without_detailed_monitoring, get_graviton_compatible_instances,
    get_burstable_instances_analysis, get_spot_instance_opportunities,
    get_unused_capacity_reservations, get_scheduling_opportunities,
    get_commitment_plan_recommendations, get_governance_violations,
    generate_comprehensive_ec2_report
)
from playbooks.ebs.ebs_optimization import (
    get_underutilized_volumes, 
    identify_unused_volumes, 
    generate_ebs_optimization_report
)
from playbooks.rds.rds_optimization import (
    get_underutilized_rds_instances, 
    identify_idle_rds_instances
)
from playbooks.aws_lambda.lambda_optimization import (
    get_underutilized_lambda_functions, 
    identify_unused_lambda_functions
)
from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator
from playbooks.nat_gateway.nat_gateway_optimization import (
    count_all_nat_gateways,
    analyze_underutilized_nat_gateways,
    analyze_redundant_nat_gateways,
    analyze_unused_nat_gateways,



    generate_nat_gateway_optimization_report,
    get_idle_nat_gateways_from_trusted_advisor
)

logger = logging.getLogger(__name__)

# Helper functions for consistent error handling and response formatting
def _format_error_response(analysis_type: str, error_message: str, error_category: str, 
                          start_time: float, recommendations: List[Dict[str, Any]] = None) -> List[TextContent]:
    """Format consistent error responses for S3 optimization functions with enhanced orchestrator integration."""
    execution_time = time.time() - start_time
    
    logger.error(f"Formatting error response for {analysis_type}: {error_category} - {error_message}")
    
    error_response = {
        "status": "error",
        "analysis_type": analysis_type,
        "error_message": error_message,
        "error_category": error_category,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations or [],
        "orchestrator_integration": True,
        "enhanced_error_handling": True,
        "comprehensive_logging": True
    }
    
    return [TextContent(type="text", text=json.dumps(error_response, indent=2, default=str))]

def _format_success_response(result: Dict[str, Any], analysis_type: str, execution_time: float) -> Dict[str, Any]:
    """Format consistent success responses for S3 optimization functions with enhanced orchestrator integration."""
    formatted_result = result.copy()
    
    logger.info(f"Formatting success response for {analysis_type} with execution time: {execution_time:.2f}s")
    
    # Add runbook-level metadata with orchestrator integration details
    formatted_result["runbook_metadata"] = {
        "analysis_type": analysis_type,
        "runbook_execution_time": execution_time,
        "orchestrator_integration": True,
        "enhanced_error_handling": True,
        "comprehensive_logging": True,
        "session_integration": True,
        "parallel_execution_support": True,
        "performance_optimizations": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # Ensure consistent structure
    if "data" not in formatted_result:
        formatted_result["data"] = {}
    
    # Add orchestrator-specific metadata if available
    if "session_id" in result:
        formatted_result["runbook_metadata"]["session_id"] = result["session_id"]
    
    if "orchestrator_execution_time" in result:
        formatted_result["runbook_metadata"]["orchestrator_execution_time"] = result["orchestrator_execution_time"]
    
    logger.debug(f"Success response formatted for {analysis_type} with enhanced orchestrator metadata")
    
    return formatted_result

def _get_timeout_recommendations(analysis_type: str) -> List[Dict[str, Any]]:
    """Get timeout-specific recommendations for S3 analyses."""
    return [
        {
            "type": "timeout_optimization",
            "priority": "high",
            "title": "Increase Timeout Settings",
            "description": f"The {analysis_type} analysis timed out during execution",
            "action_items": [
                "Increase the timeout_seconds parameter in your request",
                "Reduce the lookback period to limit data volume",
                "Filter to specific bucket_names if analyzing many buckets",
                "Run analysis during off-peak hours for better performance"
            ]
        },
        {
            "type": "performance_optimization", 
            "priority": "medium",
            "title": "Optimize Analysis Scope",
            "description": "Reduce the scope of analysis to improve performance",
            "action_items": [
                "Use bucket_names parameter to analyze specific buckets only",
                "Reduce lookback_months/lookback_days parameters",
                "Disable detailed_breakdown if not needed",
                "Set store_results=False for faster execution"
            ]
        }
    ]

def _get_analysis_error_recommendations(analysis_type: str, error_message: str) -> List[Dict[str, Any]]:
    """Get analysis-specific error recommendations."""
    recommendations = []
    
    error_lower = error_message.lower()
    
    if "permission" in error_lower or "access" in error_lower:
        recommendations.append({
            "type": "permission_fix",
            "priority": "high", 
            "title": "Fix AWS Permissions",
            "description": f"The {analysis_type} analysis failed due to permission issues",
            "action_items": [
                "Check IAM permissions for S3, Cost Explorer, and Storage Lens services",
                "Verify AWS credentials are valid and not expired",
                "Ensure required service permissions are granted for the analysis",
                "Check if MFA is required for API access"
            ]
        })
    
    if "rate" in error_lower or "throttl" in error_lower:
        recommendations.append({
            "type": "rate_limit_handling",
            "priority": "medium",
            "title": "Handle API Rate Limits", 
            "description": f"The {analysis_type} analysis was throttled by AWS APIs",
            "action_items": [
                "Retry the analysis after a few minutes",
                "Reduce the scope of analysis to make fewer API calls",
                "Implement exponential backoff in retry logic",
                "Consider running analysis during off-peak hours"
            ]
        })
    
    if "region" in error_lower:
        recommendations.append({
            "type": "region_configuration",
            "priority": "medium",
            "title": "Check Region Configuration",
            "description": f"The {analysis_type} analysis had region-related issues",
            "action_items": [
                "Verify the specified region is valid and accessible",
                "Check if the region has the required AWS services enabled",
                "Ensure your AWS credentials have access to the specified region",
                "Try using a different region or omit region parameter for default"
            ]
        })
    
    # Default recommendation if no specific error pattern matched
    if not recommendations:
        recommendations.append({
            "type": "general_troubleshooting",
            "priority": "medium", 
            "title": "General Troubleshooting",
            "description": f"The {analysis_type} analysis encountered an error",
            "action_items": [
                "Check AWS service status for any ongoing issues",
                "Verify your AWS credentials and permissions",
                "Try reducing the scope of analysis",
                "Contact support if the issue persists"
            ]
        })
    
    return recommendations

# EC2 Right Sizing Runbook Functions
async def run_ec2_right_sizing_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive EC2 right-sizing analysis with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 14)
        
        # Define parallel service calls for EC2 analysis
        service_calls = [
            {
                'service': 'ec2',
                'operation': 'underutilized_instances',
                'function': get_underutilized_instances,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'cpu_threshold': arguments.get('cpu_threshold', 40.0),
                    'memory_threshold': arguments.get('memory_threshold'),
                    'network_threshold': arguments.get('network_threshold')
                },
                'priority': 3,
                'timeout': 45.0
            }
        ]
        
        # Execute analysis with session storage
        execution_summary = orchestrator.execute_parallel_analysis(service_calls)
        
        # Get the actual result from the first (and only) task
        if execution_summary['successful'] > 0:
            # Find the successful result
            for task_id, task_info in execution_summary['results'].items():
                if task_info['status'] == 'success':
                    # Get the stored data
                    stored_table = task_info.get('stored_table')
                    if stored_table:
                        try:
                            # First, check the table structure to understand available columns
                            # Use proper SQL escaping for table names
                            escaped_table = f'"{stored_table}"'
                            table_info = orchestrator.query_session_data(f"PRAGMA table_info({escaped_table})")
                            column_names = [col['name'] for col in table_info] if table_info else []
                            
                            # Query the stored data using appropriate columns
                            if 'value' in column_names:
                                stored_data = orchestrator.query_session_data(f"SELECT value FROM {escaped_table}")
                                if stored_data and stored_data[0].get('value'):
                                    original_result = json.loads(stored_data[0]['value'])
                                else:
                                    # Fallback: get all data from the table
                                    stored_data = orchestrator.query_session_data(f"SELECT * FROM {escaped_table}")
                                    original_result = {"data": stored_data} if stored_data else {"data": []}
                            else:
                                # Get all data from the table
                                stored_data = orchestrator.query_session_data(f"SELECT * FROM {escaped_table}")
                                original_result = {"data": stored_data} if stored_data else {"data": []}
                            
                            # Add session metadata
                            original_result['session_metadata'] = {
                                'session_id': orchestrator.session_id,
                                'stored_table': stored_table,
                                'parallel_execution': True,
                                'table_columns': column_names
                            }
                            return [TextContent(type="text", text=json.dumps(original_result, indent=2, default=str))]
                        except Exception as e:
                            logger.error(f"Error querying stored data from {stored_table}: {e}")
                            # Continue to fallback
        
        # Fallback to execution summary if no stored data
        return [TextContent(type="text", text=json.dumps(execution_summary, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in EC2 right-sizing analysis: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_ec2_right_sizing_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed EC2 right-sizing report."""
    try:
        # Get underutilized instances
        instances_result = get_underutilized_instances(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14),
            cpu_threshold=arguments.get("cpu_threshold", 40.0)
        )
        
        if instances_result["status"] != "success":
            return [TextContent(type="text", text=json.dumps(instances_result, indent=2))]
            
        # Generate report
        report_result = generate_right_sizing_report(
            instances_result["data"]["underutilized_instances"]
        )
        
        output_format = arguments.get("output_format", "json")
        if output_format == "markdown":
            # Convert to markdown format
            data = report_result["data"]
            report = f"""# EC2 Right Sizing Report

## Summary
- **Total Instances**: {data['total_instances']}
- **Monthly Savings**: ${data['total_monthly_savings']:.2f}

## Top Recommendations
"""
            for instance in data.get('top_recommendations', []):
                rec = instance.get('recommendation', {})
                report += f"""### {instance['instance_id']}
- **Current**: {rec.get('current_instance_type', 'N/A')}
- **Recommended**: {rec.get('recommended_instance_type', 'N/A')}
- **Monthly Savings**: ${rec.get('estimated_monthly_savings', 0):.2f}

"""
            return [TextContent(type="text", text=report)]
        else:
            return [TextContent(type="text", text=json.dumps(report_result, indent=2, default=str))]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# EBS Optimization Runbook Functions
async def run_ebs_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive EBS optimization analysis with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 30)
        
        # Define parallel service calls for EBS analysis
        service_calls = [
            {
                'service': 'ebs',
                'operation': 'underutilized_volumes',
                'function': get_underutilized_volumes,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'iops_threshold': arguments.get('iops_threshold', 100.0),
                    'throughput_threshold': arguments.get('throughput_threshold', 1.0)
                },
                'priority': 3,
                'timeout': 40.0
            },
            {
                'service': 'ebs',
                'operation': 'unused_volumes',
                'function': identify_unused_volumes,
                'kwargs': {'region': region},
                'priority': 2,
                'timeout': 30.0
            }
        ]
        
        # EBS-specific aggregation queries
        aggregation_queries = [
            {
                'name': 'ebs_total_savings',
                'query': '''
                    SELECT 
                        'underutilized' as category,
                        COUNT(*) as volume_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.potential_savings') AS REAL)), 0) as total_savings
                    FROM ebs_underutilized_volumes_*
                    WHERE json_extract(value, '$.potential_savings') IS NOT NULL
                    
                    UNION ALL
                    
                    SELECT 
                        'unused' as category,
                        COUNT(*) as volume_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.monthly_cost') AS REAL)), 0) as total_savings
                    FROM ebs_unused_volumes_*
                    WHERE json_extract(value, '$.monthly_cost') IS NOT NULL
                '''
            }
        ]
        
        # Execute comprehensive analysis
        report = orchestrator.create_comprehensive_report(service_calls, aggregation_queries)
        
        # Add EBS-specific insights
        report['ebs_optimization'] = {
            'analysis_type': 'comprehensive_ebs_optimization',
            'region': region,
            'lookback_period_days': lookback_period_days,
            'session_id': orchestrator.session_id,
            'parallel_execution': True,
            'sql_storage': True
        }
        
        return [TextContent(type="text", text=json.dumps(report, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in EBS optimization analysis: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_ebs_volumes(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unused EBS volumes."""
    try:
        result = identify_unused_volumes(
            region=arguments.get("region"),
            min_age_days=arguments.get("min_age_days", 30)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_ebs_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed EBS optimization report."""
    try:
        region = arguments.get("region")
        
        # Get underutilized and unused volumes
        underutilized_result = get_underutilized_volumes(region=region)
        unused_result = identify_unused_volumes(region=region)
        
        if underutilized_result["status"] != "success" or unused_result["status"] != "success":
            return [TextContent(type="text", text="Error getting volume data")]
            
        # Generate comprehensive report
        report_result = generate_ebs_optimization_report(
            underutilized_result["data"]["underutilized_volumes"],
            unused_result["data"]["unused_volumes"]
        )
        
        output_format = arguments.get("output_format", "json")
        if output_format == "markdown":
            data = report_result["data"]
            report = f"""# EBS Optimization Report

## Summary
- **Total Volumes**: {data['total_volumes']}
- **Monthly Savings**: ${data['total_monthly_savings']:.2f}
- **Unused Savings**: ${data['unused_savings']:.2f}

## Top Unused Volumes
"""
            for volume in data.get('top_unused', []):
                report += f"""### {volume['volume_id']}
- **Size**: {volume.get('volume_size', 'N/A')} GB
- **Monthly Cost**: ${volume.get('monthly_cost', 0):.2f}

"""
            return [TextContent(type="text", text=report)]
        else:
            return [TextContent(type="text", text=json.dumps(report_result, indent=2, default=str))]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# RDS Optimization Runbook Functions
async def run_rds_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive RDS optimization analysis with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 14)
        
        # Define parallel service calls for RDS analysis
        service_calls = [
            {
                'service': 'rds',
                'operation': 'underutilized_instances',
                'function': get_underutilized_rds_instances,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'cpu_threshold': arguments.get('cpu_threshold', 40.0),
                    'connection_threshold': arguments.get('connection_threshold', 20.0)
                },
                'priority': 3,
                'timeout': 45.0
            },
            {
                'service': 'rds',
                'operation': 'idle_instances',
                'function': identify_idle_rds_instances,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': arguments.get('idle_lookback_days', 7),
                    'connection_threshold': arguments.get('idle_connection_threshold', 1.0)
                },
                'priority': 2,
                'timeout': 30.0
            }
        ]
        
        # RDS-specific aggregation queries
        aggregation_queries = [
            {
                'name': 'rds_optimization_summary',
                'query': '''
                    SELECT 
                        'underutilized' as category,
                        COUNT(*) as instance_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.estimated_monthly_savings') AS REAL)), 0) as total_savings
                    FROM rds_underutilized_instances_*
                    WHERE json_extract(value, '$.estimated_monthly_savings') IS NOT NULL
                    
                    UNION ALL
                    
                    SELECT 
                        'idle' as category,
                        COUNT(*) as instance_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.monthly_cost') AS REAL)), 0) as total_savings
                    FROM rds_idle_instances_*
                    WHERE json_extract(value, '$.monthly_cost') IS NOT NULL
                '''
            }
        ]
        
        # Execute comprehensive analysis
        report = orchestrator.create_comprehensive_report(service_calls, aggregation_queries)
        
        # Add RDS-specific insights
        report['rds_optimization'] = {
            'analysis_type': 'comprehensive_rds_optimization',
            'region': region,
            'lookback_period_days': lookback_period_days,
            'session_id': orchestrator.session_id,
            'parallel_execution': True,
            'sql_storage': True
        }
        
        return [TextContent(type="text", text=json.dumps(report, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in RDS optimization analysis: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_idle_rds_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify idle RDS instances."""
    try:
        from playbooks.rds.rds_optimization import identify_idle_rds_instances as get_idle_rds
        result = get_idle_rds(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 7),
            connection_threshold=arguments.get("connection_threshold", 1.0)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_rds_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed RDS optimization report."""
    try:
        region = arguments.get("region")
        
        # Get data from playbooks
        from playbooks.rds.rds_optimization import identify_idle_rds_instances as get_idle_rds
        underutilized_result = get_underutilized_rds_instances(region=region)
        idle_result = get_idle_rds(region=region)
        
        combined_report = {
            "status": "success",
            "report_type": "RDS Comprehensive Optimization Report",
            "region": region or "default",
            "optimization_analysis": underutilized_result,
            "idle_instances_analysis": idle_result,
            "summary": {
                "underutilized_instances": underutilized_result.get("data", {}).get("count", 0),
                "idle_instances": idle_result.get("data", {}).get("count", 0)
            }
        }
        
        return [TextContent(type="text", text=json.dumps(combined_report, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# Lambda Optimization Runbook Functions
async def run_lambda_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive Lambda optimization analysis with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 14)
        
        # Import the unused lambda function
        from playbooks.aws_lambda.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
        
        # Define parallel service calls for Lambda analysis
        service_calls = [
            {
                'service': 'lambda',
                'operation': 'underutilized_functions',
                'function': get_underutilized_lambda_functions,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': lookback_period_days,
                    'memory_utilization_threshold': arguments.get('memory_utilization_threshold', 50.0),
                    'min_invocations': arguments.get('min_invocations', 100)
                },
                'priority': 2,
                'timeout': 35.0
            },
            {
                'service': 'lambda',
                'operation': 'unused_functions',
                'function': get_unused_lambda,
                'kwargs': {
                    'region': region,
                    'lookback_period_days': arguments.get('unused_lookback_days', 30),
                    'max_invocations': arguments.get('max_invocations', 5)
                },
                'priority': 2,
                'timeout': 25.0
            }
        ]
        
        # Lambda-specific aggregation queries
        aggregation_queries = [
            {
                'name': 'lambda_optimization_summary',
                'query': '''
                    SELECT 
                        'underutilized' as category,
                        COUNT(*) as function_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.potential_monthly_savings') AS REAL)), 0) as total_savings
                    FROM lambda_underutilized_functions_*
                    WHERE json_extract(value, '$.potential_monthly_savings') IS NOT NULL
                    
                    UNION ALL
                    
                    SELECT 
                        'unused' as category,
                        COUNT(*) as function_count,
                        COALESCE(SUM(CAST(json_extract(value, '$.monthly_cost') AS REAL)), 0) as total_savings
                    FROM lambda_unused_functions_*
                    WHERE json_extract(value, '$.monthly_cost') IS NOT NULL
                '''
            }
        ]
        
        # Execute comprehensive analysis
        report = orchestrator.create_comprehensive_report(service_calls, aggregation_queries)
        
        # Add Lambda-specific insights
        report['lambda_optimization'] = {
            'analysis_type': 'comprehensive_lambda_optimization',
            'region': region,
            'lookback_period_days': lookback_period_days,
            'session_id': orchestrator.session_id,
            'parallel_execution': True,
            'sql_storage': True
        }
        
        return [TextContent(type="text", text=json.dumps(report, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in Lambda optimization analysis: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_lambda_functions(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unused Lambda functions."""
    try:
        from playbooks.aws_lambda.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
        result = get_unused_lambda(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 30),
            max_invocations=arguments.get("max_invocations", 5)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_lambda_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed Lambda optimization report."""
    try:
        region = arguments.get("region")
        
        # Get data from playbooks
        from playbooks.aws_lambda.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
        optimization_result = get_underutilized_lambda_functions(region=region)
        unused_result = get_unused_lambda(region=region)
        
        combined_report = {
            "status": "success",
            "report_type": "Lambda Comprehensive Optimization Report",
            "region": region or "default",
            "optimization_analysis": optimization_result,
            "unused_functions_analysis": unused_result,
            "summary": {
                "functions_with_usage": optimization_result.get("data", {}).get("count", 0),
                "unused_functions": unused_result.get("data", {}).get("count", 0)
            }
        }
        
        return [TextContent(type="text", text=json.dumps(combined_report, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

async def run_comprehensive_cost_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive cost analysis across all services with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        services = arguments.get("services", ["ec2", "ebs", "rds", "lambda", "cloudtrail", "s3", "nat_gateway"])
        lookback_period_days = arguments.get("lookback_period_days", 14)
        
        # Build service calls based on requested services
        service_calls = []
        
        if "ec2" in services:
            service_calls.extend([
                {
                    'service': 'ec2',
                    'operation': 'underutilized_instances',
                    'function': get_underutilized_instances,
                    'kwargs': {'region': region, 'lookback_period_days': lookback_period_days},
                    'priority': 3,
                    'timeout': 45.0
                },
                {
                    'service': 'ec2',
                    'operation': 'stopped_instances',
                    'function': get_stopped_instances,
                    'kwargs': {'region': region},
                    'priority': 2,
                    'timeout': 30.0
                }
            ])
        
        if "ebs" in services:
            service_calls.extend([
                {
                    'service': 'ebs',
                    'operation': 'underutilized_volumes',
                    'function': get_underutilized_volumes,
                    'kwargs': {'region': region, 'lookback_period_days': lookback_period_days},
                    'priority': 2,
                    'timeout': 40.0
                },
                {
                    'service': 'ebs',
                    'operation': 'unused_volumes',
                    'function': identify_unused_volumes,
                    'kwargs': {'region': region},
                    'priority': 2,
                    'timeout': 30.0
                }
            ])
        
        if "rds" in services:
            service_calls.extend([
                {
                    'service': 'rds',
                    'operation': 'underutilized_instances',
                    'function': get_underutilized_rds_instances,
                    'kwargs': {'region': region, 'lookback_period_days': lookback_period_days},
                    'priority': 2,
                    'timeout': 45.0
                },
                {
                    'service': 'rds',
                    'operation': 'idle_instances',
                    'function': identify_idle_rds_instances,
                    'kwargs': {'region': region},
                    'priority': 1,
                    'timeout': 30.0
                }
            ])
        
        if "lambda" in services:
            service_calls.extend([
                {
                    'service': 'lambda',
                    'operation': 'underutilized_functions',
                    'function': get_underutilized_lambda_functions,
                    'kwargs': {'region': region, 'lookback_period_days': lookback_period_days},
                    'priority': 1,
                    'timeout': 35.0
                },
                {
                    'service': 'lambda',
                    'operation': 'unused_functions',
                    'function': identify_unused_lambda_functions,
                    'kwargs': {'region': region},
                    'priority': 1,
                    'timeout': 25.0
                }
            ])
        
        if "s3" in services:
            # Use S3OptimizationOrchestrator for comprehensive S3 analysis with enhanced error handling and logging
            async def s3_comprehensive_wrapper():
                s3_start_time = time.time()
                try:
                    logger.info(f"Starting S3 comprehensive analysis wrapper for region: {region} with enhanced orchestrator integration")
                    
                    # Create orchestrator with comprehensive error handling
                    try:
                        s3_orchestrator = S3OptimizationOrchestrator(region=region)
                        logger.debug(f"S3OptimizationOrchestrator created successfully for comprehensive analysis")
                    except Exception as orchestrator_error:
                        logger.error(f"Failed to create S3OptimizationOrchestrator: {str(orchestrator_error)}")
                        raise orchestrator_error
                    
                    # Prepare S3-specific parameters with enhanced validation
                    s3_params = {
                        "lookback_months": max(1, min(lookback_period_days // 30, 6)),  # Convert days to months with validation
                        "lookback_days": max(7, min(lookback_period_days, 90)),  # Limit for performance with minimum
                        "include_all_analyses": False,  # Focus on high-impact analyses
                        "store_results": True,
                        "include_cross_analysis": True,
                        "include_detailed_breakdown": True,
                        "prioritize_by_savings": True,
                        "timeout_seconds": max(30, min(arguments.get("timeout_seconds", 45), 120))  # Enhanced timeout validation
                    }
                    
                    logger.info(f"S3 comprehensive wrapper parameters validated: {s3_params}")
                    
                    # Execute comprehensive analysis with enhanced monitoring
                    result = await s3_orchestrator.execute_comprehensive_analysis(**s3_params)
                    
                    s3_execution_time = time.time() - s3_start_time
                    logger.info(f"S3 comprehensive analysis wrapper completed with status: {result.get('status')} in {s3_execution_time:.2f}s")
                    
                    # Add wrapper-specific metadata
                    if isinstance(result, dict):
                        result["wrapper_metadata"] = {
                            "wrapper_execution_time": s3_execution_time,
                            "orchestrator_integration": True,
                            "enhanced_error_handling": True,
                            "comprehensive_logging": True,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    return result
                    
                except asyncio.TimeoutError:
                    s3_execution_time = time.time() - s3_start_time
                    logger.error(f"S3 comprehensive analysis wrapper timed out after {s3_execution_time:.2f}s")
                    return {
                        "status": "error",
                        "service": "s3",
                        "operation": "comprehensive_analysis",
                        "error_message": f"S3 comprehensive analysis timed out after {s3_execution_time:.2f} seconds",
                        "error_category": "timeout_error",
                        "execution_time": s3_execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "orchestrator_integration": True,
                        "recommendations": _get_timeout_recommendations("comprehensive_s3_analysis")
                    }
                except Exception as s3_error:
                    s3_execution_time = time.time() - s3_start_time
                    logger.error(f"S3 comprehensive analysis wrapper failed: {str(s3_error)}", exc_info=True)
                    return {
                        "status": "error",
                        "service": "s3",
                        "operation": "comprehensive_analysis",
                        "error_message": f"S3 comprehensive analysis failed: {str(s3_error)}",
                        "error_category": "s3_analysis_error",
                        "execution_time": s3_execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "orchestrator_integration": True,
                        "recommendations": _get_analysis_error_recommendations("comprehensive_s3_analysis", str(s3_error))
                    }
            
            service_calls.append({
                'service': 's3',
                'operation': 'comprehensive_analysis',
                'function': lambda: asyncio.run(s3_comprehensive_wrapper()),
                'kwargs': {},
                'priority': 1,
                'timeout': 50.0
            })
        
        if "cloudtrail" in services:
            from playbooks.cloudtrail.cloudtrail_optimization import run_cloudtrail_optimization
            service_calls.append({
                'service': 'cloudtrail',
                'operation': 'optimization',
                'function': run_cloudtrail_optimization,
                'kwargs': {'region': region},
                'priority': 1,
                'timeout': 30.0
            })
        
        if "nat_gateway" in services:
            service_calls.extend([
                {
                    'service': 'nat_gateway',
                    'operation': 'underutilized_nat_gateways',
                    'function': analyze_underutilized_nat_gateways,
                    'kwargs': {
                        'region': region,
                        'data_transfer_threshold_gb': arguments.get('data_transfer_threshold_gb', 1.0),
                        'lookback_days': lookback_period_days,
                        'zero_cost_mode': arguments.get('zero_cost_mode', True)
                    },
                    'priority': 3,
                    'timeout': 45.0
                },


                {
                    'service': 'nat_gateway',
                    'operation': 'redundant_nat_gateways',
                    'function': analyze_redundant_nat_gateways,
                    'kwargs': {'region': region},
                    'priority': 2,
                    'timeout': 30.0
                },
                {
                    'service': 'nat_gateway',
                    'operation': 'unused_nat_gateways',
                    'function': analyze_unused_nat_gateways,
                    'kwargs': {'region': region},
                    'priority': 1,
                    'timeout': 25.0
                },

            ])
        
        # Cross-service aggregation queries
        aggregation_queries = [
            {
                'name': 'total_savings_by_service',
                'query': '''
                    SELECT 
                        'ec2' as service,
                        COUNT(*) as opportunities,
                        COALESCE(SUM(CAST(json_extract(value, '$.estimated_monthly_savings') AS REAL)), 0) as total_savings
                    FROM ec2_underutilized_instances_*
                    WHERE json_extract(value, '$.estimated_monthly_savings') IS NOT NULL
                    
                    UNION ALL
                    
                    SELECT 
                        'ebs' as service,
                        COUNT(*) as opportunities,
                        COALESCE(SUM(CAST(json_extract(value, '$.monthly_cost') AS REAL)), 0) as total_savings
                    FROM ebs_unused_volumes_*
                    WHERE json_extract(value, '$.monthly_cost') IS NOT NULL
                    
                    UNION ALL
                    
                    SELECT 
                        'rds' as service,
                        COUNT(*) as opportunities,
                        COALESCE(SUM(CAST(json_extract(value, '$.estimated_monthly_savings') AS REAL)), 0) as total_savings
                    FROM rds_underutilized_instances_*
                    WHERE json_extract(value, '$.estimated_monthly_savings') IS NOT NULL
                '''
            },
            {
                'name': 'optimization_summary',
                'query': '''
                    SELECT 
                        COUNT(*) as total_opportunities,
                        SUM(total_savings) as grand_total_savings
                    FROM (
                        SELECT COALESCE(SUM(CAST(json_extract(value, '$.estimated_monthly_savings') AS REAL)), 0) as total_savings
                        FROM ec2_underutilized_instances_*
                        WHERE json_extract(value, '$.estimated_monthly_savings') IS NOT NULL
                        
                        UNION ALL
                        
                        SELECT COALESCE(SUM(CAST(json_extract(value, '$.monthly_cost') AS REAL)), 0) as total_savings
                        FROM ebs_unused_volumes_*
                        WHERE json_extract(value, '$.monthly_cost') IS NOT NULL
                    )
                '''
            }
        ]
        
        # Execute comprehensive analysis with parallel execution and session storage
        report = orchestrator.create_comprehensive_report(service_calls, aggregation_queries)
        
        # Add comprehensive analysis metadata
        report['comprehensive_analysis'] = {
            'analysis_type': 'multi_service_comprehensive',
            'services_analyzed': services,
            'region': region,
            'lookback_period_days': lookback_period_days,
            'session_id': orchestrator.session_id,
            'parallel_execution': True,
            'sql_storage': True
        }
        
        return [TextContent(type="text", text=json.dumps(report, indent=2, default=str))]
        
        if "rds" in services:
            try:
                from playbooks.rds.rds_optimization import identify_idle_rds_instances as get_idle_rds
                comprehensive_report["analyses"]["rds"] = {
                    "optimization": get_underutilized_rds_instances(region=region),
                    "idle_instances": get_idle_rds(region=region)
                }
            except Exception as e:
                comprehensive_report["analyses"]["rds"] = {"error": str(e)}
        
        if "lambda" in services:
            try:
                from playbooks.aws_lambda.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
                comprehensive_report["analyses"]["lambda"] = {
                    "optimization": get_underutilized_lambda_functions(region=region),
                    "unused_functions": get_unused_lambda(region=region)
                }
            except Exception as e:
                comprehensive_report["analyses"]["lambda"] = {"error": str(e)}
        
        if "cloudtrail" in services:
            try:
                from playbooks.cloudtrail.cloudtrail_optimization import run_cloudtrail_optimization
                comprehensive_report["analyses"]["cloudtrail"] = run_cloudtrail_optimization(region=region)
            except Exception as e:
                comprehensive_report["analyses"]["cloudtrail"] = {"error": str(e)}
        
        if "s3" in services:
            try:
                logger.info(f"Starting S3 comprehensive analysis integration for legacy comprehensive analysis, region: {region}")
                
                # Use S3OptimizationOrchestrator for comprehensive S3 analysis with enhanced error handling
                async def s3_analysis():
                    try:
                        orchestrator = S3OptimizationOrchestrator(region=region)
                        logger.debug("S3OptimizationOrchestrator created successfully for legacy comprehensive analysis")
                        
                        # Prepare S3-specific parameters for legacy compatibility
                        s3_params = {
                            "lookback_months": min(6, max(1, lookback_period_days // 30)) if 'lookback_period_days' in locals() else 6,
                            "store_results": False,  # Don't store for this legacy function
                            "include_cross_analysis": False,  # Keep it simple for legacy compatibility
                            "timeout_seconds": 45.0,  # Reasonable timeout for legacy function
                            "include_cost_analysis": True
                        }
                        
                        logger.debug(f"S3 legacy comprehensive analysis parameters: {s3_params}")
                        result = await orchestrator.execute_comprehensive_analysis(**s3_params)
                        
                        logger.info(f"S3 legacy comprehensive analysis completed with status: {result.get('status')}")
                        return result
                        
                    except Exception as orchestrator_error:
                        logger.error(f"S3OptimizationOrchestrator execution failed in legacy function: {str(orchestrator_error)}")
                        return {
                            "status": "error",
                            "error_message": f"S3 orchestrator execution failed: {str(orchestrator_error)}",
                            "error_category": "orchestrator_execution_error",
                            "timestamp": datetime.now().isoformat()
                        }
                
                s3_result = asyncio.run(s3_analysis())
                
                if s3_result.get("status") == "success":
                    comprehensive_report["analyses"]["s3"] = {
                        "orchestrator_result": s3_result,
                        "total_potential_savings": s3_result.get("aggregated_results", {}).get("total_potential_savings", 0),
                        "successful_analyses": s3_result.get("analysis_metadata", {}).get("successful_analyses", 0),
                        "execution_time": s3_result.get("execution_time", 0),
                        "orchestrator_integration": True,
                        "enhanced_error_handling": True,
                        "from_cache": s3_result.get("from_cache", False)
                    }
                    logger.info(f"S3 legacy comprehensive analysis integrated successfully")
                else:
                    logger.warning(f"S3 legacy comprehensive analysis returned non-success status: {s3_result.get('status')}")
                    comprehensive_report["analyses"]["s3"] = {
                        "error": s3_result.get("message", "S3 analysis failed"),
                        "error_category": s3_result.get("error_category", "analysis_non_success"),
                        "orchestrator_integration": True,
                        "status": s3_result.get("status")
                    }
            except asyncio.TimeoutError:
                logger.error("S3 comprehensive analysis timed out in legacy comprehensive analysis")
                comprehensive_report["analyses"]["s3"] = {
                    "error": "S3 comprehensive analysis timed out",
                    "error_category": "timeout_error",
                    "orchestrator_integration": True,
                    "recommendations": _get_timeout_recommendations("s3_comprehensive")
                }
            except Exception as e:
                logger.error(f"Error in S3 comprehensive analysis for legacy function: {str(e)}", exc_info=True)
                comprehensive_report["analyses"]["s3"] = {
                    "error": str(e),
                    "error_category": "integration_error",
                    "orchestrator_integration": True,
                    "recommendations": _get_analysis_error_recommendations("s3_comprehensive", str(e))
                }
        
        return [TextContent(type="text", text=json.dumps(comprehensive_report, indent=2, default=str))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error running comprehensive analysis: {str(e)}")]

# Additional EC2 runbook functions
async def identify_stopped_ec2_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify stopped EC2 instances."""
    try:
        result = get_stopped_instances(
            region=arguments.get("region"),
            min_stopped_days=arguments.get("min_stopped_days", 7)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unattached_elastic_ips(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unattached Elastic IPs."""
    try:
        result = get_unattached_elastic_ips(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_old_generation_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify old generation instances."""
    try:
        result = get_old_generation_instances(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_instances_without_monitoring(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify instances without detailed monitoring."""
    try:
        result = get_instances_without_detailed_monitoring(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# CloudTrail optimization functions
async def get_management_trails(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get CloudTrail management trails."""
    try:
        from playbooks.cloudtrail.cloudtrail_optimization import get_management_trails as get_trails
        result = get_trails(region=arguments.get("region", "us-east-1"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_cloudtrail_trails_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudTrail trails analysis."""
    try:
        from playbooks.cloudtrail.cloudtrail_optimization import run_cloudtrail_optimization as analyze_trails
        result = analyze_trails(region=arguments.get("region", "us-east-1"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_cloudtrail_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate CloudTrail optimization report."""
    try:
        from playbooks.cloudtrail.cloudtrail_optimization import generate_cloudtrail_report as gen_report
        result = gen_report(
            region=arguments.get("region", "us-east-1"),
            output_format=arguments.get("output_format", "json")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str) if isinstance(result, dict) else result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_s3_comprehensive_optimization_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Run comprehensive S3 optimization using the S3OptimizationOrchestrator.
    
    This function executes all S3 optimization functionalities in parallel with
    intelligent orchestration, priority-based execution, and comprehensive reporting.
    """
    analysis_type = "comprehensive_optimization"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting comprehensive S3 optimization with orchestrator for region: {region} with arguments: {arguments}")
        
        # Validate bucket_names parameter if provided
        bucket_names = arguments.get("bucket_names")
        if bucket_names and not isinstance(bucket_names, list):
            logger.warning(f"bucket_names should be a list, got {type(bucket_names)}, converting to list")
            bucket_names = [bucket_names] if bucket_names else None
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare analysis parameters with validation
        analysis_params = {
            "bucket_names": bucket_names[:50] if bucket_names else None,  # Limit to 50 buckets for performance
            "lookback_days": max(7, min(arguments.get("lookback_days", 30), 365)),  # Validate range
            "include_detailed_breakdown": arguments.get("include_detailed_breakdown", True),
            "include_cross_analysis": arguments.get("include_cross_analysis", True),
            "include_trends": arguments.get("include_trends", False),
            "timeout_seconds": max(60, min(arguments.get("timeout_seconds", 120), 600)),  # Validate timeout
            "include_cost_analysis": arguments.get("include_cost_analysis", True),
            "store_results": arguments.get("store_results", True),
            "min_savings_threshold": max(0, arguments.get("min_savings_threshold", 10.0)),  # Validate threshold
            "max_recommendations_per_type": max(1, min(arguments.get("max_recommendations_per_type", 10), 50)),  # Validate range
            "focus_high_cost": arguments.get("focus_high_cost", True)
        }
        
        logger.info(f"Starting comprehensive S3 optimization with validated parameters: {analysis_params}")
        
        # Execute comprehensive optimization using orchestrator with comprehensive error handling
        try:
            result = await orchestrator.execute_comprehensive_analysis(**analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"Comprehensive S3 optimization completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Format result for better readability
            if result.get("status") == "success":
                # Extract key information from orchestrator result
                aggregated_results = result.get("aggregated_results", {})
                analysis_metadata = result.get("analysis_metadata", {})
                cross_analysis_data = result.get("cross_analysis_data", {})
                
                formatted_result = {
                    "status": "success",
                    "comprehensive_s3_optimization": {
                        "overview": {
                            "total_potential_savings": f"${aggregated_results.get('total_potential_savings', 0):.2f}",
                            "analyses_completed": f"{analysis_metadata.get('successful_analyses', 0)}/{analysis_metadata.get('total_analyses', 0)}",
                            "failed_analyses": analysis_metadata.get('failed_analyses', 0),
                            "execution_time": f"{result.get('execution_time', 0):.2f}s",
                            "orchestrator_execution_time": f"{result.get('orchestrator_execution_time', 0):.2f}s",
                            "runbook_execution_time": f"{execution_time:.2f}s"
                        },
                        "key_findings": aggregated_results.get("key_findings", []),
                        "top_recommendations": aggregated_results.get("top_recommendations", [])[:5],
                        "quick_wins": [rec for rec in aggregated_results.get("recommendations", []) if rec.get("priority") == "high"][:3],
                        "session_id": result.get("session_id"),
                        "stored_tables": analysis_metadata.get("stored_tables", []),
                        "detailed_results_available": True,
                        "from_cache": result.get("from_cache", False)
                    },
                    "performance_optimizations": result.get("performance_optimizations", {}),
                    "cross_analysis_insights": cross_analysis_data.get("insights", []),
                    "scope_configuration": {
                        "lookback_days": analysis_params.get("lookback_days"),
                        "bucket_count": len(bucket_names) if bucket_names else "all",
                        "analyses_executed": analysis_metadata.get("total_analyses", 0),
                        "parallel_execution": True,
                        "intelligent_orchestration": True,
                        "session_sql_integration": True,
                        "performance_optimizations_enabled": True,
                        "enhanced_error_handling": True,
                        "comprehensive_logging": True
                    },
                    "runbook_metadata": {
                        "analysis_type": analysis_type,
                        "runbook_execution_time": execution_time,
                        "orchestrator_integration": True,
                        "enhanced_error_handling": True,
                        "comprehensive_logging": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Add full results for detailed analysis
                formatted_result["full_analysis_results"] = result
                
                logger.info(f"Comprehensive S3 optimization completed successfully and formatted in {execution_time:.2f}s")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors
                logger.error(f"Comprehensive S3 optimization failed: {result.get('message', 'Unknown error')}")
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            logger.error(f"Comprehensive S3 optimization timed out after {analysis_params['timeout_seconds']}s")
            return _format_error_response(
                analysis_type,
                f"Comprehensive analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            logger.error(f"Comprehensive S3 optimization execution failed: {str(analysis_error)}")
            return _format_error_response(
                analysis_type,
                f"Comprehensive analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
            
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in comprehensive S3 optimization tool: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )

# S3 Optimization Runbook Functions
async def run_s3_general_spend_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Run S3 general spend analysis using S3OptimizationOrchestrator with enhanced integration.
    
    This function provides comprehensive S3 spending pattern analysis using Storage Lens and Cost Explorer
    with enhanced error handling, logging, and orchestrator integration.
    """
    analysis_type = "general_spend"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 {analysis_type} analysis for region: {region} with enhanced orchestrator integration")
        logger.debug(f"S3 {analysis_type} analysis arguments: {arguments}")
        
        # Validate required parameters with enhanced logging
        if not region:
            logger.warning(f"No region specified for S3 {analysis_type} analysis, using default region")
        
        # Create S3OptimizationOrchestrator with comprehensive error handling
        try:
            logger.debug(f"Initializing S3OptimizationOrchestrator for {analysis_type} analysis")
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.info(f"S3OptimizationOrchestrator created successfully for {analysis_type} with session: {orchestrator.session_id}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}", exc_info=True)
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time,
                recommendations=[{
                    "type": "orchestrator_troubleshooting",
                    "priority": "high",
                    "title": "Fix Orchestrator Initialization",
                    "description": "The S3OptimizationOrchestrator failed to initialize",
                    "action_items": [
                        "Check AWS credentials and permissions",
                        "Verify region parameter is valid",
                        "Ensure required AWS services are available in the region",
                        "Check for any service quotas or limits"
                    ]
                }]
            )
        
        # Prepare analysis parameters with enhanced validation and logging
        analysis_params = {
            "lookback_months": max(1, min(arguments.get("lookback_months", 6), 24)),  # Validate range 1-24 months
            "detailed_breakdown": arguments.get("detailed_breakdown", True),
            "include_trends": arguments.get("include_trends", False),
            "timeout_seconds": max(30, min(arguments.get("timeout_seconds", 45), 300)),  # Validate timeout 30-300s
            "store_results": arguments.get("store_results", True),
            "include_cost_analysis": arguments.get("include_cost_analysis", True)
        }
        
        logger.info(f"S3 {analysis_type} analysis parameters validated and prepared: {analysis_params}")
        
        # Execute general spend analysis with comprehensive error handling and monitoring
        try:
            logger.debug(f"Executing S3 {analysis_type} analysis via orchestrator")
            result = await orchestrator.execute_analysis(analysis_type, **analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"S3 {analysis_type} analysis completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Log session and storage information
            if result.get("session_id"):
                logger.debug(f"S3 {analysis_type} analysis stored in session: {result['session_id']}")
            
            # Format and return successful response
            if result.get("status") == "success":
                formatted_result = _format_success_response(result, analysis_type, execution_time)
                logger.info(f"S3 {analysis_type} analysis result formatted successfully with orchestrator metadata")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors with enhanced logging
                logger.warning(f"S3 {analysis_type} analysis returned non-success status: {result.get('status')}")
                logger.debug(f"S3 {analysis_type} analysis error details: {result.get('message', 'No error message')}")
                
                # Add orchestrator metadata to error result
                if isinstance(result, dict):
                    result["orchestrator_integration"] = True
                    result["enhanced_error_handling"] = True
                    result["comprehensive_logging"] = True
                
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"S3 {analysis_type} analysis timed out after {analysis_params['timeout_seconds']}s (total execution: {execution_time:.2f}s)")
            return _format_error_response(
                analysis_type,
                f"Analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            execution_time = time.time() - start_time
            logger.error(f"S3 {analysis_type} analysis execution failed after {execution_time:.2f}s: {str(analysis_error)}", exc_info=True)
            return _format_error_response(
                analysis_type,
                f"Analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in S3 {analysis_type} analysis after {execution_time:.2f}s: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time,
            recommendations=[{
                "type": "general_troubleshooting",
                "priority": "high",
                "title": "Investigate Unexpected Error",
                "description": f"An unexpected error occurred during {analysis_type} analysis",
                "action_items": [
                    "Check application logs for detailed error information",
                    "Verify AWS service availability and status",
                    "Ensure all required dependencies are available",
                    "Contact support if the issue persists"
                ]
            }]
        )

async def run_s3_storage_class_selection(arguments: Dict[str, Any]) -> List[TextContent]:
    """Provide guidance on choosing the most cost-effective storage class for new data using S3OptimizationOrchestrator."""
    analysis_type = "storage_class_selection"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 {analysis_type} analysis for region: {region} with arguments: {arguments}")
        
        # Validate required parameters
        if not region:
            logger.warning(f"No region specified for S3 {analysis_type} analysis, using default")
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare analysis parameters with validation
        valid_access_frequencies = ["high", "medium", "low", "archive", "unknown"]
        valid_retrieval_tolerances = ["immediate", "minutes", "hours", "days"]
        valid_durability_requirements = ["standard", "reduced"]
        
        access_frequency = arguments.get("access_frequency", "unknown")
        if access_frequency not in valid_access_frequencies:
            logger.warning(f"Invalid access_frequency '{access_frequency}', using 'unknown'")
            access_frequency = "unknown"
        
        retrieval_time_tolerance = arguments.get("retrieval_time_tolerance", "immediate")
        if retrieval_time_tolerance not in valid_retrieval_tolerances:
            logger.warning(f"Invalid retrieval_time_tolerance '{retrieval_time_tolerance}', using 'immediate'")
            retrieval_time_tolerance = "immediate"
        
        durability_requirement = arguments.get("durability_requirement", "standard")
        if durability_requirement not in valid_durability_requirements:
            logger.warning(f"Invalid durability_requirement '{durability_requirement}', using 'standard'")
            durability_requirement = "standard"
        
        analysis_params = {
            "access_frequency": access_frequency,
            "retrieval_time_tolerance": retrieval_time_tolerance,
            "durability_requirement": durability_requirement,
            "data_size_gb": max(1, arguments.get("data_size_gb", 100)),  # Validate minimum size
            "retention_period_days": max(1, arguments.get("retention_period_days", 365)),  # Validate minimum retention
            "timeout_seconds": max(15, min(arguments.get("timeout_seconds", 30), 300)),  # Validate timeout
            "store_results": arguments.get("store_results", True)
        }
        
        logger.debug(f"S3 {analysis_type} analysis parameters validated: {analysis_params}")
        
        # Execute storage class analysis with comprehensive error handling
        try:
            result = await orchestrator.execute_analysis("storage_class", **analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"S3 {analysis_type} analysis completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Format successful response
            if result.get("status") == "success":
                formatted_result = _format_success_response(result, analysis_type, execution_time)
                logger.debug(f"S3 {analysis_type} analysis result formatted successfully")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors
                logger.warning(f"S3 {analysis_type} analysis returned non-success status: {result.get('status')}")
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            logger.error(f"S3 {analysis_type} analysis timed out after {analysis_params['timeout_seconds']}s")
            return _format_error_response(
                analysis_type,
                f"Analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            logger.error(f"S3 {analysis_type} analysis execution failed: {str(analysis_error)}")
            return _format_error_response(
                analysis_type,
                f"Analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in S3 {analysis_type} analysis: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )

async def run_s3_storage_class_validation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Validate that existing data is stored in the most appropriate storage class using S3OptimizationOrchestrator."""
    analysis_type = "storage_class_validation"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 {analysis_type} for region: {region} with arguments: {arguments}")
        
        # Validate bucket_names parameter if provided
        bucket_names = arguments.get("bucket_names")
        if bucket_names and not isinstance(bucket_names, list):
            logger.warning(f"bucket_names should be a list, got {type(bucket_names)}, converting to list")
            bucket_names = [bucket_names] if bucket_names else None
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare analysis parameters with validation
        analysis_params = {
            "bucket_names": bucket_names[:50] if bucket_names else None,  # Limit to 50 buckets for performance
            "lookback_days": max(7, min(arguments.get("lookback_days", 90), 365)),  # Validate range
            "include_recommendations": arguments.get("include_recommendations", True),
            "min_object_size_mb": max(0.1, arguments.get("min_object_size_mb", 1)),  # Validate minimum size
            "timeout_seconds": max(30, min(arguments.get("timeout_seconds", 45), 300)),  # Validate timeout
            "store_results": arguments.get("store_results", True)
        }
        
        logger.debug(f"S3 {analysis_type} parameters validated: {analysis_params}")
        
        # Execute storage class validation analysis with comprehensive error handling
        try:
            result = await orchestrator.execute_analysis("storage_class", **analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"S3 {analysis_type} completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Format successful response
            if result.get("status") == "success":
                formatted_result = _format_success_response(result, analysis_type, execution_time)
                logger.debug(f"S3 {analysis_type} result formatted successfully")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors
                logger.warning(f"S3 {analysis_type} returned non-success status: {result.get('status')}")
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            logger.error(f"S3 {analysis_type} timed out after {analysis_params['timeout_seconds']}s")
            return _format_error_response(
                analysis_type,
                f"Analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            logger.error(f"S3 {analysis_type} execution failed: {str(analysis_error)}")
            return _format_error_response(
                analysis_type,
                f"Analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in S3 {analysis_type}: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )

async def run_s3_archive_optimization(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify and optimize long-term archive data storage for cost reduction using S3OptimizationOrchestrator."""
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 archive optimization for region: {region}")
        
        # Create S3OptimizationOrchestrator
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Prepare analysis parameters
        analysis_params = {
            "bucket_names": arguments.get("bucket_names"),
            "min_age_days": arguments.get("min_age_days", 180),
            "include_compliance_check": arguments.get("include_compliance_check", True),
            "archive_tier_preference": arguments.get("archive_tier_preference", "auto"),
            "timeout_seconds": arguments.get("timeout_seconds", 35),
            "store_results": arguments.get("store_results", True)
        }
        
        # Execute archive optimization analysis
        result = await orchestrator.execute_analysis("archive_optimization", **analysis_params)
        
        logger.info(f"S3 archive optimization completed with status: {result.get('status')}")
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in S3 archive optimization: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_s3_api_cost_minimization(arguments: Dict[str, Any]) -> List[TextContent]:
    """Minimize S3 API request charges through access pattern optimization using S3OptimizationOrchestrator."""
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 API cost minimization for region: {region}")
        
        # Create S3OptimizationOrchestrator
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Prepare analysis parameters
        analysis_params = {
            "bucket_names": arguments.get("bucket_names"),
            "lookback_days": arguments.get("lookback_days", 30),
            "include_cloudfront_analysis": arguments.get("include_cloudfront_analysis", True),
            "request_threshold": arguments.get("request_threshold", 10000),
            "timeout_seconds": arguments.get("timeout_seconds", 35),
            "store_results": arguments.get("store_results", True)
        }
        
        # Execute API cost minimization analysis
        result = await orchestrator.execute_analysis("api_cost", **analysis_params)
        
        logger.info(f"S3 API cost minimization completed with status: {result.get('status')}")
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in S3 API cost minimization: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_s3_multipart_cleanup(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify and clean up incomplete multipart uploads using S3OptimizationOrchestrator."""
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 multipart cleanup for region: {region}")
        
        # Create S3OptimizationOrchestrator
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Prepare analysis parameters
        analysis_params = {
            "bucket_names": arguments.get("bucket_names"),
            "min_age_days": arguments.get("min_age_days", 7),
            "max_results_per_bucket": arguments.get("max_results_per_bucket", 100),
            "include_cost_analysis": arguments.get("include_cost_analysis", True),
            "timeout_seconds": arguments.get("timeout_seconds", 45),
            "store_results": arguments.get("store_results", True)
        }
        
        # Execute multipart cleanup analysis
        result = await orchestrator.execute_analysis("multipart_cleanup", **analysis_params)
        
        logger.info(f"S3 multipart cleanup completed with status: {result.get('status')}")
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in S3 multipart cleanup: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_s3_governance_check(arguments: Dict[str, Any]) -> List[TextContent]:
    """Implement S3 cost controls and governance policy compliance checking using S3OptimizationOrchestrator."""
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting S3 governance check for region: {region}")
        
        # Create S3OptimizationOrchestrator
        orchestrator = S3OptimizationOrchestrator(region=region)
        
        # Prepare analysis parameters
        analysis_params = {
            "bucket_names": arguments.get("bucket_names"),
            "check_tagging": arguments.get("check_tagging", True),
            "check_lifecycle_policies": arguments.get("check_lifecycle_policies", True),
            "check_versioning": arguments.get("check_versioning", True),
            "organizational_standards": arguments.get("organizational_standards"),
            "timeout_seconds": arguments.get("timeout_seconds", 30),
            "store_results": arguments.get("store_results", True)
        }
        
        # Execute governance check analysis
        result = await orchestrator.execute_analysis("governance", **analysis_params)
        
        logger.info(f"S3 governance check completed with status: {result.get('status')}")
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in S3 governance check: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def run_s3_comprehensive_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive S3 analysis using S3OptimizationOrchestrator with parallel execution and session storage."""
    analysis_type = "comprehensive_analysis"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        bucket_names = arguments.get("bucket_names")
        
        logger.info(f"Starting comprehensive S3 analysis for region: {region} with arguments: {arguments}")
        
        # Validate bucket_names parameter if provided
        if bucket_names and not isinstance(bucket_names, list):
            logger.warning(f"bucket_names should be a list, got {type(bucket_names)}, converting to list")
            bucket_names = [bucket_names] if bucket_names else None
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare comprehensive analysis parameters with validation
        analysis_params = {
            "bucket_names": bucket_names[:50] if bucket_names else None,  # Limit to 50 buckets for performance
            "lookback_months": max(1, min(arguments.get('lookback_months', 6), 24)),  # Validate range
            "lookback_days": max(7, min(arguments.get('lookback_days', 90), 365)),  # Validate range
            "include_trends": arguments.get('include_trends', False),
            "detailed_breakdown": arguments.get('detailed_breakdown', True),
            "min_age_days": max(1, arguments.get('min_age_days', 7)),  # Validate minimum age
            "include_cost_analysis": arguments.get('include_cost_analysis', True),
            "store_results": arguments.get('store_results', True),
            "include_cross_analysis": arguments.get('include_cross_analysis', True),
            "timeout_seconds": max(60, min(arguments.get('timeout_seconds', 120), 600))  # Validate timeout
        }
        
        logger.debug(f"S3 {analysis_type} parameters validated: {analysis_params}")
        
        # Execute comprehensive analysis using orchestrator with comprehensive error handling
        try:
            result = await orchestrator.execute_comprehensive_analysis(**analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"Comprehensive S3 analysis completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Add S3-specific metadata to the result
            if result.get("status") == "success":
                result['s3_comprehensive_analysis'] = {
                    'analysis_type': 'comprehensive_s3_optimization',
                    'region': region,
                    'bucket_names': bucket_names,
                    'bucket_count': len(bucket_names) if bucket_names else "all",
                    'session_id': result.get('session_id'),
                    'parallel_execution': True,
                    'sql_storage': True,
                    'orchestrator_integration': True,
                    'enhanced_error_handling': True,
                    'comprehensive_logging': True,
                    'performance_optimizations': result.get('performance_optimizations', {}),
                    'stored_tables': result.get('analysis_metadata', {}).get('stored_tables', []),
                    'runbook_execution_time': execution_time
                }
                
                # Format successful response
                formatted_result = _format_success_response(result, analysis_type, execution_time)
                logger.debug(f"S3 {analysis_type} result formatted successfully")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors
                logger.warning(f"S3 {analysis_type} returned non-success status: {result.get('status')}")
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            logger.error(f"S3 {analysis_type} timed out after {analysis_params['timeout_seconds']}s")
            return _format_error_response(
                analysis_type,
                f"Comprehensive analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            logger.error(f"S3 {analysis_type} execution failed: {str(analysis_error)}")
            return _format_error_response(
                analysis_type,
                f"Comprehensive analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in S3 {analysis_type}: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )

async def run_s3_quick_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run a quick S3 analysis focusing on the most impactful optimizations using S3OptimizationOrchestrator."""
    analysis_type = "quick_analysis"
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        
        logger.info(f"Starting quick S3 analysis for region: {region} with arguments: {arguments}")
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare quick analysis parameters - focus on high-priority analyses with validation
        analysis_params = {
            "timeout_seconds": max(15, min(arguments.get("timeout_seconds", 30), 120)),  # Validate timeout
            "store_results": arguments.get("store_results", False),  # Skip storage for quick analysis
            "include_cross_analysis": False,  # Skip cross-analysis for speed
            "max_results_per_bucket": max(10, min(arguments.get("max_results_per_bucket", 50), 100)),  # Validate range
            "min_age_days": max(1, arguments.get("min_age_days", 7)),  # Validate minimum age
            "lookback_days": max(7, min(arguments.get("lookback_days", 30), 90))  # Shorter lookback for speed
        }
        
        logger.debug(f"Quick S3 analysis parameters validated: {analysis_params}")
        
        # Execute high-priority analyses only (multipart cleanup and governance)
        high_priority_analyses = ["multipart_cleanup", "governance"]
        
        results = {
            "status": "success",
            "data": {
                "analysis_type": "quick_scan",
                "timestamp": datetime.now().isoformat(),
                "recommendations": [],
                "analyses_completed": [],
                "analyses_failed": []
            },
            "message": "Quick S3 analysis completed",
            "runbook_metadata": {
                "analysis_type": analysis_type,
                "orchestrator_integration": True,
                "enhanced_error_handling": True,
                "comprehensive_logging": True
            }
        }
        
        # Run each high-priority analysis with comprehensive error handling
        for priority_analysis in high_priority_analyses:
            try:
                logger.debug(f"Starting quick {priority_analysis} analysis")
                result = await orchestrator.execute_analysis(priority_analysis, **analysis_params)
                
                if result.get("status") == "success":
                    results["data"]["analyses_completed"].append(priority_analysis)
                    
                    # Extract key recommendations
                    recommendations = result.get("data", {}).get("recommendations", [])
                    for rec in recommendations[:3]:  # Top 3 recommendations per analysis
                        results["data"]["recommendations"].append({
                            "analysis_type": priority_analysis,
                            "type": rec.get("type", "optimization"),
                            "priority": rec.get("priority", "medium"),
                            "description": rec.get("description", ""),
                            "potential_savings": rec.get("potential_savings", 0)
                        })
                    
                    logger.debug(f"Quick {priority_analysis} analysis completed successfully")
                else:
                    logger.warning(f"Quick {priority_analysis} analysis returned non-success status: {result.get('status')}")
                    results["data"]["analyses_failed"].append({
                        "analysis_type": priority_analysis,
                        "error": result.get("message", "Analysis failed"),
                        "status": result.get("status")
                    })
                        
            except asyncio.TimeoutError:
                logger.warning(f"Quick {priority_analysis} analysis timed out")
                results["data"]["analyses_failed"].append({
                    "analysis_type": priority_analysis,
                    "error": f"Analysis timed out after {analysis_params['timeout_seconds']} seconds",
                    "error_category": "timeout_error"
                })
            except Exception as e:
                logger.warning(f"Error in quick {priority_analysis} analysis: {str(e)}")
                results["data"]["analyses_failed"].append({
                    "analysis_type": priority_analysis,
                    "error": str(e),
                    "error_category": "analysis_error"
                })
        
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        results["runbook_metadata"]["runbook_execution_time"] = execution_time
        results["runbook_metadata"]["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Quick S3 analysis completed with {len(results['data']['recommendations'])} recommendations in {execution_time:.2f}s")
        return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in quick S3 analysis: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )

async def run_s3_bucket_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Analyze specific S3 buckets for optimization opportunities using S3OptimizationOrchestrator."""
    analysis_type = "bucket_analysis"
    start_time = time.time()
    
    try:
        bucket_names = arguments.get("bucket_names", [])
        if not bucket_names:
            logger.error(f"bucket_names parameter is required for {analysis_type}")
            return _format_error_response(
                analysis_type,
                "bucket_names parameter is required",
                "missing_required_parameter",
                start_time,
                recommendations=[{
                    "type": "parameter_fix",
                    "priority": "high",
                    "title": "Provide Bucket Names",
                    "description": "The bucket_names parameter is required for bucket-specific analysis",
                    "action_items": [
                        "Add bucket_names parameter as a list of bucket names",
                        "Example: bucket_names=['my-bucket-1', 'my-bucket-2']",
                        "Limit to 10 buckets for optimal performance"
                    ]
                }]
            )
        
        # Validate bucket_names parameter
        if not isinstance(bucket_names, list):
            logger.warning(f"bucket_names should be a list, got {type(bucket_names)}, converting to list")
            bucket_names = [bucket_names] if bucket_names else []
        
        region = arguments.get("region")
        
        logger.info(f"Starting bucket-specific S3 analysis for {len(bucket_names)} buckets in region: {region} with arguments: {arguments}")
        
        # Create S3OptimizationOrchestrator with enhanced error handling
        try:
            orchestrator = S3OptimizationOrchestrator(region=region)
            logger.debug(f"S3OptimizationOrchestrator created successfully for {analysis_type}")
        except Exception as orchestrator_error:
            logger.error(f"Failed to create S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}")
            return _format_error_response(
                analysis_type, 
                f"Orchestrator initialization failed: {str(orchestrator_error)}", 
                "orchestrator_init_error",
                start_time
            )
        
        # Prepare analysis parameters for specific buckets with validation
        analysis_params = {
            "bucket_names": bucket_names[:10],  # Limit to 10 buckets for performance
            "timeout_seconds": max(30, min(arguments.get("timeout_seconds", 40), 300)),  # Validate timeout
            "store_results": arguments.get("store_results", True),
            "include_cost_analysis": arguments.get("include_cost_analysis", True),
            "lookback_days": max(7, min(arguments.get("lookback_days", 30), 90)),  # Validate range
            "include_cross_analysis": arguments.get("include_cross_analysis", True)
        }
        
        logger.debug(f"Bucket-specific S3 analysis parameters validated: {analysis_params}")
        
        # Execute comprehensive analysis for the specific buckets with comprehensive error handling
        try:
            result = await orchestrator.execute_comprehensive_analysis(**analysis_params)
            execution_time = time.time() - start_time
            
            logger.info(f"Bucket-specific S3 analysis completed with status: {result.get('status')} in {execution_time:.2f}s")
            
            # Format results for bucket-specific analysis
            if result.get("status") == "success":
                aggregated_results = result.get("aggregated_results", {})
                
                formatted_result = {
                    "status": "success",
                    "data": {
                        "analysis_type": "bucket_specific",
                        "buckets_analyzed": bucket_names[:10],
                        "bucket_count": len(bucket_names[:10]),
                        "total_potential_savings": aggregated_results.get("total_potential_savings", 0),
                        "recommendations": aggregated_results.get("recommendations", []),
                        "key_findings": aggregated_results.get("key_findings", [])
                    },
                    "execution_metadata": {
                        "execution_time": result.get("execution_time", 0),
                        "runbook_execution_time": execution_time,
                        "session_id": result.get("session_id"),
                        "stored_tables": result.get("analysis_metadata", {}).get("stored_tables", []),
                        "from_cache": result.get("from_cache", False),
                        "orchestrator_integration": True,
                        "enhanced_error_handling": True
                    },
                    "message": f"Analyzed {len(bucket_names[:10])} specific buckets",
                    "runbook_metadata": {
                        "analysis_type": analysis_type,
                        "runbook_execution_time": execution_time,
                        "orchestrator_integration": True,
                        "enhanced_error_handling": True,
                        "comprehensive_logging": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                logger.info(f"Bucket-specific S3 analysis completed successfully and formatted")
                return [TextContent(type="text", text=json.dumps(formatted_result, indent=2, default=str))]
            else:
                # Handle analysis-level errors
                logger.error(f"Bucket-specific S3 analysis failed: {result.get('message', 'Unknown error')}")
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                
        except asyncio.TimeoutError:
            logger.error(f"Bucket-specific S3 analysis timed out after {analysis_params['timeout_seconds']}s")
            return _format_error_response(
                analysis_type,
                f"Bucket analysis timed out after {analysis_params['timeout_seconds']} seconds",
                "timeout_error",
                start_time,
                recommendations=_get_timeout_recommendations(analysis_type)
            )
        except Exception as analysis_error:
            logger.error(f"Bucket-specific S3 analysis execution failed: {str(analysis_error)}")
            return _format_error_response(
                analysis_type,
                f"Bucket analysis execution failed: {str(analysis_error)}",
                "analysis_execution_error", 
                start_time,
                recommendations=_get_analysis_error_recommendations(analysis_type, str(analysis_error))
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in bucket-specific S3 analysis: {str(e)}", exc_info=True)
        return _format_error_response(
            analysis_type,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            start_time
        )
# S3 Optimization Runbook Functions (Synchronous wrappers for orchestrator)
def s3_general_spend_analysis(region: str = None, **kwargs) -> Dict[str, Any]:
    """
    Run S3 general spend analysis using S3OptimizationOrchestrator (synchronous wrapper).
    
    This synchronous wrapper provides the same functionality as the async version
    with enhanced orchestrator integration, error handling, and comprehensive logging.
    """
    analysis_type = "general_spend"
    start_time = time.time()
    
    try:
        logger.info(f"Running S3 {analysis_type} analysis (sync wrapper) for region: {region} with enhanced orchestrator integration")
        logger.debug(f"S3 {analysis_type} sync wrapper kwargs: {kwargs}")
        
        # Validate parameters with enhanced logging
        if not region:
            logger.warning(f"No region specified for S3 {analysis_type} analysis, using default region")
        
        # Enhanced async function with comprehensive error handling
        async def run_analysis():
            try:
                logger.debug(f"Creating S3OptimizationOrchestrator for {analysis_type} sync wrapper")
                orchestrator = S3OptimizationOrchestrator(region=region)
                logger.info(f"S3OptimizationOrchestrator created successfully for {analysis_type} with session: {orchestrator.session_id}")
                
                # Execute analysis with enhanced monitoring
                result = await orchestrator.execute_analysis(analysis_type, **kwargs)
                logger.debug(f"S3 {analysis_type} orchestrator execution completed with status: {result.get('status')}")
                return result
                
            except Exception as orchestrator_error:
                logger.error(f"Failed to create or execute S3OptimizationOrchestrator for {analysis_type}: {str(orchestrator_error)}", exc_info=True)
                raise orchestrator_error
        
        # Execute async function with enhanced error handling
        try:
            result = asyncio.run(run_analysis())
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"S3 {analysis_type} analysis timed out in sync wrapper after {execution_time:.2f}s")
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "error_message": f"Analysis timed out after {execution_time:.2f} seconds",
                "error_category": "timeout_error",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "sync_wrapper": True,
                "orchestrator_integration": True,
                "recommendations": _get_timeout_recommendations(analysis_type)
            }
        except Exception as async_error:
            execution_time = time.time() - start_time
            logger.error(f"S3 {analysis_type} async execution failed in sync wrapper after {execution_time:.2f}s: {str(async_error)}", exc_info=True)
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "error_message": f"Async execution failed: {str(async_error)}",
                "error_category": "async_execution_error",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "sync_wrapper": True,
                "orchestrator_integration": True,
                "recommendations": _get_analysis_error_recommendations(analysis_type, str(async_error))
            }
        
        execution_time = time.time() - start_time
        logger.info(f"S3 {analysis_type} analysis completed with status: {result.get('status')} in {execution_time:.2f}s")
        
        # Add enhanced synchronous wrapper metadata
        if isinstance(result, dict):
            result["sync_wrapper_metadata"] = {
                "wrapper_execution_time": execution_time,
                "analysis_type": analysis_type,
                "enhanced_error_handling": True,
                "comprehensive_logging": True,
                "orchestrator_integration": True,
                "session_integration": True,
                "performance_optimizations": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log session information if available
            if result.get("session_id"):
                logger.debug(f"S3 {analysis_type} sync wrapper completed with session: {result['session_id']}")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error in S3 {analysis_type} sync wrapper after {execution_time:.2f}s: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "analysis_type": analysis_type,
            "error_message": f"Unexpected error in synchronous wrapper: {str(e)}",
            "error_category": "sync_wrapper_error",
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "sync_wrapper": True,
            "orchestrator_integration": True,
            "recommendations": _get_analysis_error_recommendations(analysis_type, str(e))
        }

def s3_storage_class_selection(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 storage class selection analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 storage class selection for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("storage_class", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 storage class selection completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 storage class selection: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 storage class selection: {str(e)}",
            "data": {}
        }

def s3_storage_class_validation(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 storage class validation analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 storage class validation for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("storage_class", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 storage class validation completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 storage class validation: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 storage class validation: {str(e)}",
            "data": {}
        }

def s3_archive_optimization(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 archive data optimization analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 archive optimization for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("archive_optimization", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 archive optimization completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 archive optimization: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 archive optimization: {str(e)}",
            "data": {}
        }

def s3_api_charge_minimization(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 API charge minimization analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 API charge minimization for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("api_cost", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 API charge minimization completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 API charge minimization: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 API charge minimization: {str(e)}",
            "data": {}
        }

def s3_multipart_cleanup(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 multipart upload cleanup analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 multipart cleanup for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("multipart_cleanup", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 multipart cleanup completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 multipart cleanup: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 multipart cleanup: {str(e)}",
            "data": {}
        }

def s3_governance_check(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run S3 governance compliance check using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running S3 governance check for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_analysis("governance", **kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"S3 governance check completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 governance check: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 governance check: {str(e)}",
            "data": {}
        }

def s3_comprehensive_analysis(region: str = None, **kwargs) -> Dict[str, Any]:
    """Run comprehensive S3 optimization analysis using S3OptimizationOrchestrator."""
    try:
        logger.info(f"Running comprehensive S3 analysis for region: {region}")
        
        async def run_analysis():
            orchestrator = S3OptimizationOrchestrator(region=region)
            return await orchestrator.execute_comprehensive_analysis(**kwargs)
        
        result = asyncio.run(run_analysis())
        logger.info(f"Comprehensive S3 analysis completed with status: {result.get('status')}")
        return result
    except Exception as e:
        logger.error(f"Error in S3 comprehensive analysis: {str(e)}")
        return {
            "status": "error",
            "message": f"Error in S3 comprehensive analysis: {str(e)}",
            "data": {}
        }

# NAT Gateway Optimization Runbook Functions
async def run_nat_gateway_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive NAT Gateway optimization analysis with parallel execution and session storage."""
    try:
        orchestrator = ServiceOrchestrator()
        region = arguments.get("region")
        lookback_period_days = arguments.get("lookback_period_days", 14)
        data_transfer_threshold_gb = arguments.get("data_transfer_threshold_gb", 1.0)
        
        # Define parallel service calls for NAT Gateway analysis
        service_calls = [
            {
                'service': 'nat_gateway',
                'operation': 'underutilized_nat_gateways',
                'function': analyze_underutilized_nat_gateways,
                'kwargs': {
                    'region': region,
                    'data_transfer_threshold_gb': data_transfer_threshold_gb,
                    'lookback_days': lookback_period_days
                },
                'priority': 3,
                'timeout': 60.0
            },
            {
                'service': 'nat_gateway',
                'operation': 'redundant_nat_gateways',
                'function': analyze_redundant_nat_gateways,
                'kwargs': {
                    'region': region
                },
                'priority': 2,
                'timeout': 45.0
            },
            {
                'service': 'nat_gateway',
                'operation': 'unused_nat_gateways',
                'function': analyze_unused_nat_gateways,
                'kwargs': {
                    'region': region
                },
                'priority': 1,
                'timeout': 30.0
            }
        ]
        
        # Execute parallel analysis with orchestrator (without aggregation first)
        result = orchestrator.create_comprehensive_report(
            service_calls=service_calls,
            aggregation_queries=[]
        )
        
        # Now build dynamic aggregation query based on actual stored tables
        stored_tables = result.get('report_metadata', {}).get('stored_tables', [])
        underutilized_table = next((t for t in stored_tables if 'underutilized' in t), None)
        unused_table = next((t for t in stored_tables if 'unused' in t), None)
        
        # Build aggregation query with actual table names
        if underutilized_table or unused_table:
            union_parts = []
            if underutilized_table:
                union_parts.append(f"SELECT nat_gateway_id, potential_monthly_savings FROM {underutilized_table} WHERE nat_gateway_id IS NOT NULL")
            if unused_table:
                union_parts.append(f"SELECT nat_gateway_id, potential_monthly_savings FROM {unused_table} WHERE nat_gateway_id IS NOT NULL")
            
            if union_parts:
                union_query = " UNION ".join(union_parts)
                aggregation_query = f'''
                    SELECT 
                        'summary' as category,
                        COUNT(DISTINCT nat_gateway_id) as nat_gateway_count,
                        COALESCE(SUM(potential_monthly_savings), 0.0) as total_savings
                    FROM ({union_query})
                '''
                
                # Execute aggregation
                aggregated_results = orchestrator.aggregate_results([
                    {'name': 'nat_gateway_optimization_summary', 'query': aggregation_query}
                ])
                
                # Add aggregated results to report
                result['aggregated_analysis'] = aggregated_results
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        logger.error(f"Error in NAT Gateway optimization analysis: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_underutilized_nat_gateways(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify underutilized NAT Gateways based on data transfer metrics with cost optimization."""
    try:
        region = arguments.get("region") or "us-east-1"  # Default to us-east-1 if None
        result = analyze_underutilized_nat_gateways(
            region=region,
            data_transfer_threshold_gb=arguments.get("data_transfer_threshold_gb", 1.0),
            lookback_days=arguments.get("lookback_days", 14),
            zero_cost_mode=arguments.get("zero_cost_mode", True),
            include_account_context=True
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_redundant_nat_gateways(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify potentially redundant NAT Gateways in the same availability zone."""
    try:
        region = arguments.get("region") or "us-east-1"  # Default to us-east-1 if None
        result = analyze_redundant_nat_gateways(
            region=region,
            include_account_context=True
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_nat_gateways(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify NAT Gateways that are not referenced by any route tables."""
    try:
        region = arguments.get("region") or "us-east-1"  # Default to us-east-1 if None
        result = analyze_unused_nat_gateways(
            region=region,
            include_account_context=True
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Report and Trusted Advisor functions removed - internal use only, not exposed as MCP tools

# Cross-AZ traffic analysis removed - speculative savings without actual traffic data

# VPC Endpoint analysis removed - requires actual traffic data to provide meaningful recommendations



# Secondary IP consolidation and Private NAT Gateway analysis removed - too complex for minimal value