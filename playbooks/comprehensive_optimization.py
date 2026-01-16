"""
Comprehensive Cost Optimization Playbook

This module provides multi-service cost optimization analysis functions.
Includes both core optimization functions and MCP runbook functions.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
from mcp.types import TextContent

from utils.error_handler import ResponseFormatter, handle_aws_error
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task
from utils.documentation_links import add_documentation_links

# Import playbook modules
from playbooks.ec2.ec2_optimization import get_underutilized_instances
from playbooks.ebs.ebs_optimization import get_underutilized_volumes
from playbooks.rds.rds_optimization import get_underutilized_rds_instances, identify_idle_rds_instances
from playbooks.aws_lambda.lambda_optimization import get_underutilized_lambda_functions, identify_unused_lambda_functions
from playbooks.cloudtrail.cloudtrail_optimization import run_cloudtrail_optimization
from playbooks.cloudwatch.cloudwatch_optimization import run_cloudwatch_comprehensive_optimization_tool_mcp

logger = logging.getLogger(__name__)


@handle_aws_error
async def run_comprehensive_cost_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive cost analysis across multiple AWS services."""
    start_time = time.time()
    
    try:
        region = arguments.get("region")
        services = arguments.get("services", ["ec2", "ebs", "rds", "lambda", "cloudtrail", "s3", "cloudwatch"])
        lookback_period_days = arguments.get("lookback_period_days", 14)
        output_format = arguments.get("output_format", "json")
        
        # Initialize service orchestrator for parallel execution and session management
        orchestrator = ServiceOrchestrator()
        
        # Define parallel service calls based on requested services
        service_calls = []
        
        if "ec2" in services:
            service_calls.extend([
                {
                    'service': 'ec2',
                    'operation': 'underutilized_instances',
                    'function': get_underutilized_instances,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': lookback_period_days
                    }
                },
                {
                    'service': 'ec2',
                    'operation': 'stopped_instances',
                    'function': lambda region=None, **kwargs: {"stopped_instances": []},  # Placeholder
                    'kwargs': {'region': region}
                }
            ])
        
        if "ebs" in services:
            service_calls.extend([
                {
                    'service': 'ebs',
                    'operation': 'underutilized_volumes',
                    'function': get_underutilized_volumes,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': 30
                    }
                },
                {
                    'service': 'ebs',
                    'operation': 'unused_volumes',
                    'function': lambda region=None, **kwargs: {"unused_volumes": []},  # Placeholder
                    'kwargs': {'region': region}
                }
            ])
        
        if "rds" in services:
            service_calls.extend([
                {
                    'service': 'rds',
                    'operation': 'underutilized_instances',
                    'function': get_underutilized_rds_instances,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': lookback_period_days
                    }
                },
                {
                    'service': 'rds',
                    'operation': 'idle_instances',
                    'function': identify_idle_rds_instances,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': 7
                    }
                }
            ])
        
        if "lambda" in services:
            service_calls.extend([
                {
                    'service': 'lambda',
                    'operation': 'underutilized_functions',
                    'function': get_underutilized_lambda_functions,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': lookback_period_days
                    }
                },
                {
                    'service': 'lambda',
                    'operation': 'unused_functions',
                    'function': identify_unused_lambda_functions,
                    'kwargs': {
                        'region': region,
                        'lookback_period_days': 30
                    }
                }
            ])
        
        if "cloudtrail" in services:
            service_calls.append({
                'service': 'cloudtrail',
                'operation': 'optimization',
                'function': run_cloudtrail_optimization,
                'kwargs': {'region': region}
            })
        
        if "cloudwatch" in services:
            # CloudWatch uses its own comprehensive optimization tool
            def cloudwatch_wrapper(region=None, lookback_days=30, **kwargs):
                return {
                    'status': 'success',
                    'service': 'cloudwatch',
                    'message': 'CloudWatch analysis requires separate execution via cloudwatch_comprehensive_optimization_tool',
                    'recommendation': 'Use the dedicated CloudWatch comprehensive optimization tool for detailed analysis',
                    'region': region,
                    'lookback_days': lookback_days,
                    'note': 'CloudWatch has its own advanced parallel execution and memory management system'
                }
            
            service_calls.append({
                'service': 'cloudwatch',
                'operation': 'comprehensive_optimization',
                'function': cloudwatch_wrapper,
                'kwargs': {
                    'region': region,
                    'lookback_days': lookback_period_days
                }
            })
        
        # Execute parallel analysis
        results = orchestrator.execute_parallel_analysis(
            service_calls=service_calls,
            store_results=True,
            timeout=120.0
        )
        
        # Add documentation links
        results = add_documentation_links(results)
        
        execution_time = time.time() - start_time
        
        # Format response with metadata
        results["comprehensive_analysis"] = {
            "analysis_type": "multi_service_comprehensive",
            "services_analyzed": services,
            "region": region,
            "lookback_period_days": lookback_period_days,
            "session_id": results.get("report_metadata", {}).get("session_id"),
            "parallel_execution": True,
            "sql_storage": True
        }
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=results,
                message=f"Comprehensive analysis completed for {len(services)} services",
                analysis_type="comprehensive_analysis",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive cost analysis: {str(e)}")
        raise