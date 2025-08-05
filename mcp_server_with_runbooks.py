#!/usr/bin/env python3
"""
CFM Tips - AWS Cost Optimization MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS cost analysis and optimization.
This server provides tools for analyzing AWS costs and optimizations by connecting to:
- AWS Cost Explorer
- Cost Optimization Hub (with correct permissions)
- Compute Optimizer
- CUR Reports in S3
- Trusted Advisor
- Performance Insights
- Cost Optimization Runbooks/Playbooks

Features:
- EC2 Right Sizing Analysis
- EBS Volume Optimization
- RDS Database Optimization  
- Lambda Function Optimization
- Comprehensive Multi-Service Analysis
- Real CloudWatch Metrics Integration
- Markdown and JSON Report Generation

Author: CFM Tips
License: MIT
Repository: https://github.com/aws-samples/sample-cfm-tips-mcp
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and setup centralized logging
from logging_config import setup_logging, log_function_entry, log_function_exit
logger = setup_logging()

# Initialize the MCP server
server = Server("cfm_tips")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for AWS cost analysis and runbooks."""
    log_function_entry(logger, "list_tools")
    try:
        tools = [
        # Cost Explorer Tools
        Tool(
            name="get_cost_explorer_data",
            description="Retrieve cost data from AWS Cost Explorer",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                    "granularity": {"type": "string", "enum": ["DAILY", "MONTHLY", "HOURLY"], "default": "MONTHLY"},
                    "metrics": {"type": "array", "items": {"type": "string"}, "default": ["BlendedCost", "UnblendedCost"]},
                    "group_by": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["start_date", "end_date"]
            }
        ),
        
        # Cost Optimization Hub Tools (Corrected)
        Tool(
            name="list_coh_enrollment",
            description="List Cost Optimization Hub enrollment statuses",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_organization_info": {"type": "boolean", "default": False}
                }
            }
        ),
        Tool(
            name="get_coh_recommendations",
            description="Get cost optimization recommendations from AWS Cost Optimization Hub",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "include_all_recommendations": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 100},
                    "order_by": {"type": "object"}
                }
            }
        ),
        Tool(
            name="get_coh_summaries",
            description="Get summaries of cost optimization recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "max_results": {"type": "integer", "default": 100}
                }
            }
        ),
        Tool(
            name="get_coh_recommendation",
            description="Get a specific cost optimization recommendation by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "recommendation_id": {"type": "string"}
                },
                "required": ["recommendation_id"]
            }
        ),
        
        # Compute Optimizer Tools
        Tool(
            name="get_compute_optimizer_recommendations",
            description="Get recommendations from AWS Compute Optimizer",
            inputSchema={
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": ["Ec2Instance", "AutoScalingGroup", "EbsVolume", "LambdaFunction", "EcsService"],
                        "default": "Ec2Instance"
                    }
                }
            }
        ),
        
        # EC2 Right Sizing Runbook Tools
        Tool(
            name="ec2_rightsizing",
            description="Run comprehensive EC2 right-sizing analysis to identify underutilized instances",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"},
                    "cpu_threshold": {"type": "number", "default": 40.0, "description": "CPU utilization threshold %"},
                    "memory_threshold": {"type": "number", "description": "Memory utilization threshold %"},
                    "network_threshold": {"type": "number", "description": "Network utilization threshold %"}
                }
            }
        ),
        Tool(
            name="ec2_report",
            description="Generate detailed EC2 right-sizing report with recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "include_cost_analysis": {"type": "boolean", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                }
            }
        ),
        Tool(
            name="ec2_stopped_instances",
            description="Identify stopped EC2 instances that could be terminated",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "min_stopped_days": {"type": "integer", "default": 7, "description": "Minimum days stopped to consider"}
                }
            }
        ),
        Tool(
            name="ec2_unattached_eips",
            description="Identify unattached Elastic IP addresses",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_old_generation",
            description="Identify old generation EC2 instances that should be upgraded",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_detailed_monitoring",
            description="Identify instances without detailed monitoring enabled",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_graviton_compatible",
            description="Identify instances compatible with Graviton processors",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_burstable_analysis",
            description="Analyze burstable instances for credit usage optimization",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 14}
                }
            }
        ),
        Tool(
            name="ec2_spot_opportunities",
            description="Identify instances suitable for Spot pricing",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_unused_reservations",
            description="Identify unused On-Demand Capacity Reservations",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_scheduling_opportunities",
            description="Identify instances suitable for scheduling optimization",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_commitment_plans",
            description="Analyze instances for Reserved Instance and Savings Plans opportunities",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_governance_violations",
            description="Detect EC2 governance violations and policy non-compliance",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        Tool(
            name="ec2_comprehensive_report",
            description="Generate comprehensive EC2 optimization report covering all playbooks",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"}
                }
            }
        ),
        
        # EBS Optimization Runbook Tools
        Tool(
            name="ebs_optimization",
            description="Run comprehensive EBS optimization analysis to identify unused and underutilized volumes",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 30, "description": "Days to analyze metrics"},
                    "iops_threshold": {"type": "number", "default": 100.0, "description": "IOPS utilization threshold"},
                    "throughput_threshold": {"type": "number", "default": 1.0, "description": "Throughput threshold MB/s"}
                }
            }
        ),
        Tool(
            name="ebs_unused",
            description="Identify unused EBS volumes that can be deleted",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "min_age_days": {"type": "integer", "default": 30, "description": "Minimum age of volumes to consider"}
                }
            }
        ),
        Tool(
            name="ebs_report",
            description="Generate detailed EBS optimization report with cost savings",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "include_cost_analysis": {"type": "boolean", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                }
            }
        ),
        
        # RDS Optimization Runbook Tools
        Tool(
            name="rds_optimization",
            description="Run comprehensive RDS optimization analysis to identify underutilized databases",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"},
                    "cpu_threshold": {"type": "number", "default": 40.0, "description": "CPU utilization threshold %"},
                    "connection_threshold": {"type": "number", "default": 20.0, "description": "Connection count threshold"}
                }
            }
        ),
        Tool(
            name="rds_idle",
            description="Identify idle RDS instances with minimal activity",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 7, "description": "Days to analyze"},
                    "connection_threshold": {"type": "number", "default": 1.0, "description": "Max connections to consider idle"}
                }
            }
        ),
        Tool(
            name="rds_report",
            description="Generate detailed RDS optimization report with recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "include_cost_analysis": {"type": "boolean", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                }
            }
        ),
        
        # Lambda Optimization Runbook Tools
        Tool(
            name="lambda_optimization",
            description="Run comprehensive Lambda optimization analysis to identify overprovisioned functions",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"},
                    "memory_utilization_threshold": {"type": "number", "default": 50.0, "description": "Memory utilization threshold %"},
                    "min_invocations": {"type": "integer", "default": 100, "description": "Minimum invocations to analyze"}
                }
            }
        ),
        Tool(
            name="lambda_unused",
            description="Identify unused Lambda functions with minimal invocations",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "lookback_period_days": {"type": "integer", "default": 30, "description": "Days to analyze"},
                    "max_invocations": {"type": "integer", "default": 5, "description": "Max invocations to consider unused"}
                }
            }
        ),
        Tool(
            name="lambda_report",
            description="Generate detailed Lambda optimization report with cost savings",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "include_cost_analysis": {"type": "boolean", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                }
            }
        ),
        
        # Comprehensive Runbook Tools
        Tool(
            name="comprehensive_analysis",
            description="Run comprehensive cost analysis across all services (EC2, EBS, RDS, Lambda)",
            inputSchema={
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "AWS region to analyze"},
                    "services": {"type": "array", "items": {"type": "string"}, "default": ["ec2", "ebs", "rds", "lambda"]},
                    "lookback_period_days": {"type": "integer", "default": 14},
                    "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                }
            }
        ),
        
        # Other Tools
        Tool(
            name="list_cur_reports",
            description="List Cost and Usage Reports (CUR) in an S3 bucket",
            inputSchema={
                "type": "object",
                "properties": {
                    "bucket_name": {"type": "string"},
                    "prefix": {"type": "string"}
                },
                "required": ["bucket_name"]
            }
        ),
        Tool(
            name="get_trusted_advisor_checks",
            description="Get AWS Trusted Advisor check results",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_categories": {"type": "array", "items": {"type": "string"}}
                }
            }
        ),
        Tool(
            name="get_performance_insights_metrics",
            description="Get Performance Insights metrics for an RDS instance",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_instance_identifier": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"}
                },
                "required": ["db_instance_identifier"]
            }
        )
    ]
        log_function_exit(logger, "list_tools", "success", None)
        logger.info(f"Successfully listed {len(tools)} MCP tools")
        return tools
    except Exception as e:
        logger.warning(f"Error listing tools: {str(e)}")
        log_function_exit(logger, "list_tools", "error", None)
        raise

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    start_time = datetime.now()
    log_function_entry(logger, f"call_tool[{name}]", arguments=arguments)
    
    try:
        # Core AWS Service Tools
        if name == "get_cost_explorer_data":
            return await get_cost_explorer_data(arguments)
        elif name == "list_coh_enrollment":
            return await list_cost_optimization_enrollment_statuses(arguments)
        elif name == "get_coh_recommendations":
            return await get_cost_optimization_recommendations(arguments)
        elif name == "get_coh_summaries":
            return await get_cost_optimization_recommendation_summaries(arguments)
        elif name == "get_coh_recommendation":
            return await get_cost_optimization_recommendation(arguments)
        elif name == "get_compute_optimizer_recommendations":
            return await get_compute_optimizer_recommendations(arguments)
        elif name == "list_cur_reports":
            return await list_cur_reports(arguments)
        elif name == "get_trusted_advisor_checks":
            return await get_trusted_advisor_checks(arguments)
        elif name == "get_performance_insights_metrics":
            return await get_performance_insights_metrics(arguments)
        
        # EC2 Optimization Runbook Tools
        elif name == "ec2_rightsizing":
            return await run_ec2_right_sizing_analysis(arguments)
        elif name == "ec2_report":
            return await generate_ec2_right_sizing_report(arguments)
        elif name == "ec2_stopped_instances":
            return await identify_stopped_ec2_instances(arguments)
        elif name == "ec2_unattached_eips":
            return await identify_unattached_elastic_ips(arguments)
        elif name == "ec2_old_generation":
            return await identify_old_generation_instances(arguments)
        elif name == "ec2_detailed_monitoring":
            return await identify_instances_without_monitoring(arguments)
        elif name == "ec2_graviton_compatible":
            return await identify_graviton_compatible_instances(arguments)
        elif name == "ec2_burstable_analysis":
            return await analyze_burstable_instances(arguments)
        elif name == "ec2_spot_opportunities":
            return await identify_spot_opportunities(arguments)
        elif name == "ec2_unused_reservations":
            return await identify_unused_reservations(arguments)
        elif name == "ec2_scheduling_opportunities":
            return await identify_scheduling_opportunities(arguments)
        elif name == "ec2_commitment_plans":
            return await analyze_commitment_plans(arguments)
        elif name == "ec2_governance_violations":
            return await identify_governance_violations(arguments)
        elif name == "ec2_comprehensive_report":
            return await generate_comprehensive_report(arguments)
        
        # EBS Optimization Runbook Tools
        elif name == "ebs_optimization":
            return await run_ebs_optimization_analysis(arguments)
        elif name == "ebs_unused":
            return await identify_unused_ebs_volumes(arguments)
        elif name == "ebs_report":
            return await generate_ebs_optimization_report(arguments)
        
        # RDS Optimization Runbook Tools
        elif name == "rds_optimization":
            return await run_rds_optimization_analysis(arguments)
        elif name == "rds_idle":
            return await identify_idle_rds_instances(arguments)
        elif name == "rds_report":
            return await generate_rds_optimization_report(arguments)
        
        # Lambda Optimization Runbook Tools
        elif name == "lambda_optimization":
            return await run_lambda_optimization_analysis(arguments)
        elif name == "lambda_unused":
            return await identify_unused_lambda_functions(arguments)
        elif name == "lambda_report":
            return await generate_lambda_optimization_report(arguments)
        
        # Comprehensive Analysis
        elif name == "comprehensive_analysis":
            return await run_comprehensive_cost_analysis(arguments)
        
        else:
            logger.warning(f"Unknown tool requested: {name}")
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error calling tool '{name}': {str(e)} (execution time: {execution_time:.2f}s)")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    log_function_exit(logger, f"call_tool[{name}]", "success", execution_time)

# Core AWS Service Functions (from v3)
async def get_cost_explorer_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Retrieve cost data from AWS Cost Explorer."""
    try:
        start_date = arguments["start_date"]
        end_date = arguments["end_date"]
        granularity = arguments.get("granularity", "MONTHLY")
        metrics = arguments.get("metrics", ["BlendedCost", "UnblendedCost"])
        group_by = arguments.get("group_by")
        
        ce_client = boto3.client('ce')
        params = {
            'TimePeriod': {'Start': start_date, 'End': end_date},
            'Granularity': granularity,
            'Metrics': metrics
        }
        if group_by:
            params['GroupBy'] = group_by
            
        response = ce_client.get_cost_and_usage(**params)
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved cost data from {start_date} to {end_date}"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def list_cost_optimization_enrollment_statuses(arguments: Dict[str, Any]) -> List[TextContent]:
    """List Cost Optimization Hub enrollment statuses."""
    try:
        include_organization_info = arguments.get("include_organization_info", False)
        client = boto3.client('cost-optimization-hub', region_name='us-east-1')
        
        params = {}
        if include_organization_info:
            params['includeOrganizationInfo'] = include_organization_info
            
        response = client.list_enrollment_statuses(**params)
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved {len(response.get('items', []))} enrollment statuses"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return [TextContent(type="text", text=json.dumps({
            "status": "error",
            "error_code": error_code,
            "message": f"AWS API Error: {error_code} - {error_message}",
            "required_permissions": ["cost-optimization-hub:ListEnrollmentStatuses"]
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_cost_optimization_recommendations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get cost optimization recommendations from AWS Cost Optimization Hub."""
    try:
        group_by = arguments.get("group_by")
        include_all_recommendations = arguments.get("include_all_recommendations", False)
        max_results = arguments.get("max_results", 100)
        order_by = arguments.get("order_by")
        
        client = boto3.client('cost-optimization-hub', region_name='us-east-1')
        params = {'maxResults': max_results}
        
        if group_by:
            params['groupBy'] = group_by
        if include_all_recommendations:
            params['includeAllRecommendations'] = include_all_recommendations
        if order_by:
            params['orderBy'] = order_by
            
        response = client.list_recommendations(**params)
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved {len(response.get('items', []))} cost optimization recommendations"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return [TextContent(type="text", text=json.dumps({
            "status": "error",
            "error_code": error_code,
            "message": f"AWS API Error: {error_code} - {error_message}",
            "required_permissions": ["cost-optimization-hub:ListRecommendations"]
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_cost_optimization_recommendation_summaries(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get summaries of cost optimization recommendations."""
    try:
        group_by = arguments.get("group_by")
        max_results = arguments.get("max_results", 100)
        
        client = boto3.client('cost-optimization-hub', region_name='us-east-1')
        params = {'maxResults': max_results}
        if group_by:
            params['groupBy'] = group_by
            
        response = client.list_recommendation_summaries(**params)
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved {len(response.get('items', []))} recommendation summaries"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return [TextContent(type="text", text=json.dumps({
            "status": "error",
            "error_code": error_code,
            "message": f"AWS API Error: {error_code} - {error_message}",
            "required_permissions": ["cost-optimization-hub:ListRecommendationSummaries"]
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_cost_optimization_recommendation(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get a specific cost optimization recommendation by ID."""
    try:
        recommendation_id = arguments["recommendation_id"]
        client = boto3.client('cost-optimization-hub', region_name='us-east-1')
        
        response = client.get_recommendation(recommendationId=recommendation_id)
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved recommendation {recommendation_id}"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return [TextContent(type="text", text=json.dumps({
            "status": "error",
            "error_code": error_code,
            "message": f"AWS API Error: {error_code} - {error_message}",
            "required_permissions": ["cost-optimization-hub:GetRecommendation"]
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_compute_optimizer_recommendations(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get recommendations from AWS Compute Optimizer."""
    try:
        resource_type = arguments.get("resource_type", "Ec2Instance")
        client = boto3.client('compute-optimizer')
        
        if resource_type == "Ec2Instance":
            response = client.get_ec2_instance_recommendations()
        elif resource_type == "AutoScalingGroup":
            response = client.get_auto_scaling_group_recommendations()
        elif resource_type == "EbsVolume":
            response = client.get_ebs_volume_recommendations()
        elif resource_type == "LambdaFunction":
            response = client.get_lambda_function_recommendations()
        elif resource_type == "EcsService":
            response = client.get_ecs_service_recommendations()
        else:
            return [TextContent(type="text", text=f"Error: Invalid resource type: {resource_type}")]
            
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved Compute Optimizer recommendations for {resource_type}"
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
# Import runbook functions
from runbook_functions import (
    run_ec2_right_sizing_analysis,
    generate_ec2_right_sizing_report,
    identify_stopped_ec2_instances,
    identify_unattached_elastic_ips,
    identify_old_generation_instances,
    identify_instances_without_monitoring,
    run_ebs_optimization_analysis,
    identify_unused_ebs_volumes,
    generate_ebs_optimization_report,
    run_rds_optimization_analysis,
    identify_idle_rds_instances,
    generate_rds_optimization_report,
    run_lambda_optimization_analysis,
    identify_unused_lambda_functions,
    generate_lambda_optimization_report,
    run_comprehensive_cost_analysis
)
from runbook_functions_extended import (
    identify_graviton_compatible_instances,
    analyze_burstable_instances,
    identify_spot_opportunities,
    identify_unused_reservations,
    identify_scheduling_opportunities,
    analyze_commitment_plans,
    identify_governance_violations,
    generate_comprehensive_report
)

# Additional AWS service functions
async def list_cur_reports(arguments: Dict[str, Any]) -> List[TextContent]:
    """List Cost and Usage Reports (CUR) in an S3 bucket."""
    try:
        bucket_name = arguments["bucket_name"]
        prefix = arguments.get("prefix")
        
        s3_client = boto3.client('s3')
        params = {'Bucket': bucket_name}
        if prefix:
            params['Prefix'] = prefix
            
        response = s3_client.list_objects_v2(**params)
        
        reports = []
        if 'Contents' in response:
            for obj in response['Contents']:
                reports.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
                
        result = {
            "status": "success",
            "data": {"reports": reports, "count": len(reports)},
            "message": f"Found {len(reports)} CUR reports in bucket {bucket_name}"
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_trusted_advisor_checks(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get AWS Trusted Advisor check results."""
    try:
        check_categories = arguments.get("check_categories")
        
        support_client = boto3.client('support', region_name='us-east-1')
        checks_response = support_client.describe_trusted_advisor_checks(language='en')
        
        checks = checks_response['checks']
        if check_categories:
            checks = [check for check in checks if check['category'] in check_categories]
            
        results = []
        for check in checks:
            check_id = check['id']
            try:
                result = support_client.describe_trusted_advisor_check_result(
                    checkId=check_id,
                    language='en'
                )
                results.append({
                    'check_id': check_id,
                    'name': check['name'],
                    'category': check['category'],
                    'result': result['result']
                })
            except Exception as check_error:
                logger.warning(f"Error getting result for check {check['name']}: {str(check_error)}")
                
        result = {
            "status": "success",
            "data": {"checks": results, "count": len(results)},
            "message": f"Retrieved {len(results)} Trusted Advisor check results"
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def get_performance_insights_metrics(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get Performance Insights metrics for an RDS instance."""
    try:
        db_instance_identifier = arguments["db_instance_identifier"]
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        
        pi_client = boto3.client('pi')
        
        if not start_time:
            end_datetime = datetime.utcnow()
            start_datetime = end_datetime - timedelta(hours=1)
            start_time = start_datetime.isoformat() + 'Z'
            end_time = end_datetime.isoformat() + 'Z'
        elif not end_time:
            end_time = datetime.utcnow().isoformat() + 'Z'
            
        metrics = [
            {'Metric': 'db.load.avg'},
            {'Metric': 'db.sampledload.avg'}
        ]
            
        response = pi_client.get_resource_metrics(
            ServiceType='RDS',
            Identifier=db_instance_identifier,
            StartTime=start_time,
            EndTime=end_time,
            MetricQueries=metrics,
            PeriodInSeconds=60
        )
            
        result = {
            "status": "success",
            "data": response,
            "message": f"Retrieved Performance Insights metrics for {db_instance_identifier}"
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main function to run the MCP server."""
    logger.info("Starting CFM Tips MCP Server")
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server initialized successfully")
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as e:
        logger.error(f"MCP server error: {str(e)}")
        raise
    finally:
        logger.info("CFM Tips MCP Server shutting down")

if __name__ == "__main__":
    try:
        logger.info("CFM Tips MCP Server starting up")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("CFM Tips MCP Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in MCP server: {str(e)}")
        sys.exit(1)
