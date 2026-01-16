#!/usr/bin/env python3
"""
CFM Tips - AWS Cost Optimization MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS cost analysis and optimization.
This server provides tools for analyzing AWS costs and optimizations by connecting to:
- AWS Cost Explorer
- Cost Optimization Hub
- Compute Optimizer
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

# Import centralized utilities
from utils.logging_config import setup_logging, log_function_entry, log_function_exit
from utils.documentation_links import add_documentation_links
from utils.error_handler import AWSErrorHandler, ResponseFormatter, handle_aws_error
from utils.aws_client_factory import (
    AWSClientFactory, get_cost_explorer_client, get_cost_optimization_hub_client,
    get_compute_optimizer_client, get_trusted_advisor_client, get_performance_insights_client
)

logger = setup_logging()

# Initialize the MCP server
server = Server("cfm_tips")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for AWS cost analysis and runbooks."""
    log_function_entry(logger, "list_tools")
    try:
        # Define all available MCP tools
        tools = [
            # Cost Explorer
            Tool(
                name="get_cost_explorer_data",
                description="Retrieve cost data from AWS Cost Explorer. Use this data to analyze spending patterns and provide AWS Well-Architected Framework Cost Optimization pillar recommendations.",
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
            
            # Cost Optimization Hub
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
                description="Get cost optimization recommendations from AWS Cost Optimization Hub. Analyze these recommendations and provide additional AWS Well-Architected Framework insights for Cost Optimization, Performance Efficiency, and Operational Excellence pillars.",
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
            
            # Compute Optimizer
            Tool(
                name="get_compute_optimizer_recommendations",
                description="Get recommendations from AWS Compute Optimizer. Use these findings to suggest AWS Well-Architected Framework improvements for Performance Efficiency, Cost Optimization, and Reliability pillars.",
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
            
            # EC2 Optimization
            Tool(
                name="ec2_rightsizing",
                description="Run comprehensive EC2 right-sizing analysis to identify underutilized instances. Based on findings, provide AWS Well-Architected Framework recommendations for Cost Optimization (right-sizing), Performance Efficiency (instance types), and Reliability (availability zones) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"},
                        "cpu_threshold": {"type": "number", "default": 40.0, "description": "CPU utilization threshold %"}
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
            
            # Extended EC2 Tools
            Tool(
                name="ec2_stopped_instances",
                description="Identify stopped EC2 instances that could be terminated. Provide Well-Architected recommendations for Cost Optimization and Operational Excellence.",
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
                description="Identify unattached Elastic IP addresses. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_old_generation",
                description="Identify old generation EC2 instances that should be upgraded. Provide Well-Architected recommendations for Performance Efficiency and Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_detailed_monitoring",
                description="Identify instances without detailed monitoring enabled. Provide Well-Architected recommendations for Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_graviton_compatible",
                description="Identify instances compatible with Graviton processors. Provide Well-Architected recommendations for Cost Optimization and Performance Efficiency.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_burstable_analysis",
                description="Analyze burstable instances for credit usage optimization. Provide Well-Architected recommendations for Performance Efficiency and Cost Optimization.",
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
                description="Identify instances suitable for Spot pricing. Provide Well-Architected recommendations for Cost Optimization and Reliability.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_unused_reservations",
                description="Identify unused On-Demand Capacity Reservations. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_scheduling_opportunities",
                description="Identify instances suitable for scheduling optimization. Provide Well-Architected recommendations for Cost Optimization and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_commitment_plans",
                description="Analyze instances for Reserved Instance and Savings Plans opportunities. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_governance_violations",
                description="Detect EC2 governance violations and policy non-compliance. Provide Well-Architected recommendations for Security and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="ec2_comprehensive_report",
                description="Generate comprehensive EC2 optimization report covering all playbooks. Provide holistic Well-Architected recommendations across all pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            
            # EBS Optimization
            Tool(
                name="ebs_optimization",
                description="Run comprehensive EBS optimization analysis to identify unused and underutilized volumes. Use results to suggest AWS Well-Architected Framework improvements for Cost Optimization (unused resources), Performance Efficiency (volume types), and Reliability (backup strategies) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 30, "description": "Days to analyze metrics"}
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
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="ebs_unused",
                description="Identify unused EBS volumes that can be deleted. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "min_age_days": {"type": "integer", "default": 30, "description": "Minimum age of volumes to consider"}
                    }
                }
            ),
            
            # RDS Optimization
            Tool(
                name="rds_optimization",
                description="Run comprehensive RDS optimization analysis to identify underutilized databases. Analyze findings to provide AWS Well-Architected Framework recommendations for Cost Optimization (right-sizing), Performance Efficiency (instance classes), Reliability (Multi-AZ), and Security (encryption) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"}
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
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="rds_idle",
                description="Identify idle RDS instances with minimal activity. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 7, "description": "Days to analyze"},
                        "connection_threshold": {"type": "number", "default": 1.0, "description": "Max connections to consider idle"}
                    }
                }
            ),
            
            # Lambda Optimization
            Tool(
                name="lambda_optimization",
                description="Run comprehensive Lambda optimization analysis to identify overprovisioned functions. Use findings to suggest AWS Well-Architected Framework improvements for Cost Optimization (memory sizing), Performance Efficiency (execution time), and Operational Excellence (monitoring) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 14, "description": "Days to analyze metrics"}
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
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="lambda_unused",
                description="Identify unused Lambda functions with minimal invocations. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_period_days": {"type": "integer", "default": 30, "description": "Days to analyze"},
                        "max_invocations": {"type": "integer", "default": 5, "description": "Max invocations to consider unused"}
                    }
                }
            ),
            
            # S3 Optimization
            Tool(
                name="s3_comprehensive_analysis",
                description="Run comprehensive S3 cost optimization analysis. Analyze results to provide AWS Well-Architected Framework recommendations for Cost Optimization (storage classes, lifecycle policies), Security (encryption, access controls), and Operational Excellence (monitoring, automation) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (optional)"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="s3_general_spend_analysis",
                description="Analyze overall S3 spending patterns and usage to identify optimization opportunities. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_months": {"type": "integer", "default": 12, "description": "Number of months to analyze"},
                        "include_trends": {"type": "boolean", "default": True, "description": "Whether to include trend analysis"},
                        "detailed_breakdown": {"type": "boolean", "default": True, "description": "Whether to include detailed cost breakdown"}
                    }
                }
            ),
            Tool(
                name="s3_storage_class_selection",
                description="Provide guidance on choosing the most cost-effective storage class for new data. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "access_frequency": {"type": "string", "enum": ["high", "medium", "low", "archive", "unknown"], "default": "unknown"},
                        "retrieval_time_tolerance": {"type": "string", "enum": ["immediate", "minutes", "hours", "days"], "default": "immediate"},
                        "durability_requirement": {"type": "string", "enum": ["standard", "reduced"], "default": "standard"},
                        "data_size_gb": {"type": "number", "default": 100},
                        "retention_period_days": {"type": "integer", "default": 365}
                    }
                }
            ),
            Tool(
                name="s3_storage_class_validation",
                description="Validate that existing data is stored in the most appropriate storage class. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (default: all buckets)"},
                        "lookback_days": {"type": "integer", "default": 90, "description": "Days to analyze access patterns"},
                        "include_recommendations": {"type": "boolean", "default": True, "description": "Whether to include transition recommendations"},
                        "min_object_size_mb": {"type": "number", "default": 1, "description": "Minimum object size to analyze in MB"}
                    }
                }
            ),
            Tool(
                name="s3_archive_optimization",
                description="Identify and optimize long-term archive data storage for cost reduction. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (default: all buckets)"},
                        "min_age_days": {"type": "integer", "default": 180, "description": "Minimum age in days for archive consideration"},
                        "include_compliance_check": {"type": "boolean", "default": True, "description": "Whether to check compliance requirements"},
                        "archive_tier_preference": {"type": "string", "enum": ["instant", "flexible", "deep_archive", "auto"], "default": "auto"}
                    }
                }
            ),
            Tool(
                name="s3_api_cost_minimization",
                description="Minimize S3 API request charges through access pattern optimization. Provide Well-Architected recommendations for Cost Optimization and Performance Efficiency.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (default: all buckets)"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze request patterns"},
                        "include_cloudfront_analysis": {"type": "boolean", "default": True, "description": "Whether to analyze CloudFront caching opportunities"},
                        "request_threshold": {"type": "integer", "default": 10000, "description": "Minimum requests per month to analyze"}
                    }
                }
            ),
            Tool(
                name="s3_multipart_cleanup",
                description="Identify and clean up incomplete multipart uploads to eliminate storage waste. Provide Well-Architected recommendations for Cost Optimization and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (default: all buckets)"},
                        "min_age_days": {"type": "integer", "default": 7, "description": "Minimum age in days for cleanup consideration"},
                        "max_results_per_bucket": {"type": "integer", "default": 1000, "description": "Maximum uploads to analyze per bucket"},
                        "include_cost_analysis": {"type": "boolean", "default": True, "description": "Whether to calculate waste costs"}
                    }
                }
            ),
            Tool(
                name="s3_governance_check",
                description="Implement S3 cost controls and governance policy compliance checking. Provide Well-Architected recommendations for Security and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (default: all buckets)"},
                        "check_tagging": {"type": "boolean", "default": True, "description": "Whether to check cost allocation tags"},
                        "check_lifecycle_policies": {"type": "boolean", "default": True, "description": "Whether to check lifecycle policies"},
                        "check_versioning": {"type": "boolean", "default": True, "description": "Whether to check versioning settings"},
                        "organizational_standards": {"type": "object", "description": "Custom organizational governance standards"}
                    }
                }
            ),
            Tool(
                name="s3_comprehensive_optimization_tool",
                description="Run comprehensive S3 optimization with unified tool - executes all 8 functionalities in parallel with intelligent orchestration. Provide holistic Well-Architected recommendations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "Specific buckets to analyze (optional)"},
                        "lookback_days": {"type": "integer", "description": "Days to analyze (default: 30)", "default": 30},
                        "include_detailed_breakdown": {"type": "boolean", "description": "Include detailed cost breakdown", "default": True},
                        "include_cross_analysis": {"type": "boolean", "description": "Include cross-analysis insights", "default": True},
                        "include_trends": {"type": "boolean", "description": "Include trend analysis", "default": False},
                        "timeout_seconds": {"type": "integer", "description": "Total timeout for comprehensive analysis", "default": 120},
                        "individual_timeout": {"type": "integer", "description": "Timeout per individual analysis", "default": 45},
                        "include_executive_summary": {"type": "boolean", "description": "Include executive summary", "default": True},
                        "min_savings_threshold": {"type": "number", "description": "Minimum savings to include in recommendations", "default": 10.0},
                        "max_recommendations_per_type": {"type": "integer", "description": "Maximum recommendations per type", "default": 10},
                        "enabled_analyses": {"type": "array", "items": {"type": "string"}, "description": "Specific analyses to run (optional)"},
                        "disabled_analyses": {"type": "array", "items": {"type": "string"}, "description": "Analyses to skip (optional)"},
                        "min_monthly_cost": {"type": "number", "description": "Only analyze buckets above this monthly cost"},
                        "focus_high_cost": {"type": "boolean", "description": "Prioritize high-cost buckets", "default": True},
                        "max_parallel_analyses": {"type": "integer", "description": "Maximum parallel analyses", "default": 6},
                        "store_results": {"type": "boolean", "description": "Store results in session database", "default": True}
                    }
                }
            ),
            Tool(
                name="s3_quick_analysis",
                description="Run a quick S3 analysis to identify top spending buckets, incomplete multipart uploads, and governance issues. Fast 30-second analysis ideal for getting spending overview, listing highest cost buckets, and finding quick wins. Use this when asked about S3 costs, top spending buckets, or bucket spending breakdown. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="s3_bucket_analysis",
                description="Analyze specific S3 buckets for optimization opportunities. Provide Well-Architected recommendations for Cost Optimization and Security.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bucket_names": {"type": "array", "items": {"type": "string"}, "description": "List of bucket names to analyze (required)"},
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    },
                    "required": ["bucket_names"]
                }
            ),
            
            # CloudTrail Optimization
            Tool(
                name="generate_cloudtrail_report",
                description="Generate CloudTrail optimization report. Use findings to suggest AWS Well-Architected Framework improvements for Security (logging, monitoring), Operational Excellence (observability), and Cost Optimization (log retention) pillars.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="get_management_trails",
                description="Get CloudTrail management trails. Provide Well-Architected recommendations for Security and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="run_cloudtrail_trails_analysis",
                description="Run CloudTrail trails analysis for optimization. Provide Well-Architected recommendations for Security, Cost Optimization, and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            
            # Additional AWS Service Tools
            Tool(
                name="get_trusted_advisor_checks",
                description="Get AWS Trusted Advisor check results. Provide Well-Architected recommendations across all pillars based on findings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "check_categories": {"type": "array", "items": {"type": "string"}}
                    }
                }
            ),
            Tool(
                name="get_performance_insights_metrics",
                description="Get Performance Insights metrics for an RDS instance. Provide Well-Architected recommendations for Performance Efficiency and Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "db_instance_identifier": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"}
                    },
                    "required": ["db_instance_identifier"]
                }
            ),
            
            # CloudWatch Optimization
            Tool(
                name="cloudwatch_general_spend_analysis",
                description="Run CloudWatch general spend analysis to understand cost breakdown across logs, metrics, alarms, and dashboards. Provide Well-Architected recommendations for Cost Optimization and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze CloudWatch usage"},
                        "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "default": 10, "description": "Number of items per page"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="cloudwatch_metrics_optimization",
                description="Run CloudWatch metrics optimization analysis to identify custom metrics cost optimization opportunities. Provide Well-Architected recommendations for Cost Optimization and Performance Efficiency.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze metrics usage"},
                        "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "default": 10, "description": "Number of items per page"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="cloudwatch_logs_optimization",
                description="Run CloudWatch logs optimization analysis to identify log retention and ingestion cost optimization opportunities. Provide Well-Architected recommendations for Cost Optimization and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze log usage"},
                        "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "default": 10, "description": "Number of items per page"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="cloudwatch_alarms_and_dashboards_optimization",
                description="Run CloudWatch alarms and dashboards optimization analysis to identify monitoring efficiency improvements. Provide Well-Architected recommendations for Operational Excellence and Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze alarms and dashboards"},
                        "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "default": 10, "description": "Number of items per page"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="cloudwatch_comprehensive_optimization_tool",
                description="Run comprehensive CloudWatch optimization using the unified optimization tool with intelligent orchestration. Returns aggregate summary by default to avoid token overflow. Use detail_level='full' for complete resource details or query session data for specific resources. Provide holistic Well-Architected recommendations for Cost Optimization, Performance Efficiency, and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze CloudWatch usage"},
                        "detail_level": {"type": "string", "enum": ["summary", "full"], "default": "summary", "description": "Level of detail in response. 'summary' returns aggregate metrics (recommended), 'full' returns all resources (may exceed token limits)"},
                        "page": {"type": "integer", "default": 1, "description": "Page number for pagination"},
                        "page_size": {"type": "integer", "default": 10, "description": "Number of items per page"},
                        "include_executive_summary": {"type": "boolean", "default": True, "description": "Include executive summary"},
                        "timeout_seconds": {"type": "integer", "default": 120, "description": "Total timeout for comprehensive analysis"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="query_cloudwatch_analysis_results",
                description="Query stored CloudWatch analysis results using SQL queries. Provide Well-Architected recommendations based on historical analysis data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to query"},
                        "query": {"type": "string", "description": "SQL query to execute on stored results"},
                        "limit": {"type": "integer", "default": 100, "description": "Maximum number of results to return"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="validate_cloudwatch_cost_preferences",
                description="Validate CloudWatch cost preferences and get functionality coverage estimates. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "cost_preferences": {"type": "object", "description": "Cost preference settings to validate"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            Tool(
                name="get_cloudwatch_cost_estimate",
                description="Get detailed cost estimate for CloudWatch optimization analysis based on enabled features. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "analysis_type": {"type": "string", "enum": ["general", "metrics", "logs", "alarms", "comprehensive"], "default": "comprehensive"},
                        "lookback_days": {"type": "integer", "default": 30, "description": "Days to analyze for cost estimation"},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            
            # Database Savings Plans Tools
            Tool(
                name="database_savings_plans_analysis",
                description="Run comprehensive Database Savings Plans analysis with automated recommendations for AWS database services. Analyzes current on-demand usage for Amazon Aurora, RDS, DynamoDB, ElastiCache (Valkey), DocumentDB, Neptune, Keyspaces, Timestream, and DMS. Focuses on latest-generation instance families (M7, R7, R8) eligible for Database Savings Plans with 1-year terms and no upfront payment. Provides Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze (optional, defaults to all regions)"},
                        "lookback_period_days": {"type": "integer", "default": 30, "description": "Analysis period in days (30, 60, or 90)", "enum": [30, 60, 90]},
                        "services": {"type": "array", "items": {"type": "string"}, "description": "Database services to analyze (optional, defaults to all supported services)"},
                        "include_ri_comparison": {"type": "boolean", "default": True, "description": "Compare Database Savings Plans with Reserved Instances for older generation instances"}
                    }
                }
            ),
            Tool(
                name="database_savings_plans_purchase_analyzer",
                description="Purchase analyzer mode for Database Savings Plans - model custom commitment scenarios with user-specified hourly amounts. Simulate projected cost, coverage, and utilization for custom commitments within Database Savings Plans constraints (1-year terms, no upfront payment, latest-generation instances only). Provides Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "hourly_commitment": {"type": "number", "description": "Custom hourly commitment amount in USD (required)", "minimum": 0.01, "maximum": 10000.0},
                        "commitment_term": {"type": "string", "enum": ["1_YEAR"], "default": "1_YEAR", "description": "Commitment term (only 1_YEAR supported for Database Savings Plans)"},
                        "payment_option": {"type": "string", "enum": ["NO_UPFRONT"], "default": "NO_UPFRONT", "description": "Payment option (only NO_UPFRONT supported for 1-year Database Savings Plans)"},
                        "region": {"type": "string", "description": "AWS region to analyze (optional, defaults to all regions)"},
                        "adjusted_usage_projection": {"type": "number", "description": "Optional adjusted hourly usage projection for future scenarios", "minimum": 0.01}
                    },
                    "required": ["hourly_commitment"]
                }
            ),
            Tool(
                name="database_savings_plans_existing_analysis",
                description="Analyze existing Database Savings Plans utilization and coverage to optimize current commitments and identify gaps. Reviews current Database Savings Plans performance, calculates utilization and coverage percentages, identifies over-commitment and unused capacity. Provides Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze (optional, defaults to all regions)"},
                        "lookback_period_days": {"type": "integer", "default": 30, "description": "Analysis period in days (30, 60, or 90)", "enum": [30, 60, 90]}
                    }
                }
            ),
            
            # Comprehensive Analysis
            Tool(
                name="comprehensive_analysis",
                description="Run comprehensive cost analysis across all services (EC2, EBS, RDS, Lambda, CloudTrail, S3, CloudWatch). Use the multi-service findings to provide holistic AWS Well-Architected Framework recommendations across all five pillars: Cost Optimization, Performance Efficiency, Reliability, Security, and Operational Excellence.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "services": {"type": "array", "items": {"type": "string"}, "default": ["ec2", "ebs", "rds", "lambda", "cloudtrail", "s3"]},
                        "lookback_period_days": {"type": "integer", "default": 14},
                        "output_format": {"type": "string", "enum": ["json", "markdown"], "default": "json"}
                    }
                }
            ),
            
            # NAT Gateway Optimization Tools
            Tool(
                name="nat_gateway_optimization",
                description="Run comprehensive NAT Gateway optimization analysis to identify underutilized, redundant, and unused NAT Gateways. Provide Well-Architected recommendations for Cost Optimization and Network Architecture.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "data_transfer_threshold_gb": {"type": "number", "description": "Minimum GB of data transfer to consider utilized", "default": 1.0},
                        "lookback_days": {"type": "integer", "description": "Number of days to analyze metrics", "default": 14}
                    }
                }
            ),
            Tool(
                name="nat_gateway_underutilized",
                description="Identify underutilized NAT Gateways based on data transfer metrics with cost optimization. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"},
                        "data_transfer_threshold_gb": {"type": "number", "description": "Minimum GB threshold for utilization", "default": 1.0},
                        "lookback_days": {"type": "integer", "description": "Number of days to analyze", "default": 14},
                        "zero_cost_mode": {"type": "boolean", "description": "Only use CloudWatch if Trusted Advisor unavailable (enables $0 cost analysis)", "default": True}
                    }
                }
            ),
            Tool(
                name="nat_gateway_redundant",
                description="Identify potentially redundant NAT Gateways in the same availability zone. Provide Well-Architected recommendations for Cost Optimization and Reliability.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
                }
            ),
            Tool(
                name="nat_gateway_unused",
                description="Identify NAT Gateways that are not referenced by any route tables. Provide Well-Architected recommendations for Cost Optimization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "AWS region to analyze"}
                    }
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
        elif name == "get_compute_optimizer_recommendations":
            return await get_compute_optimizer_recommendations(arguments)
        
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
        elif name == "ebs_report":
            return await generate_ebs_optimization_report(arguments)
        elif name == "ebs_unused":
            return await identify_unused_ebs_volumes(arguments)
        
        # RDS Optimization Runbook Tools
        elif name == "rds_optimization":
            return await run_rds_optimization_analysis(arguments)
        elif name == "rds_report":
            return await generate_rds_optimization_report(arguments)
        elif name == "rds_idle":
            return await identify_idle_rds_instances_wrapper(arguments)
        
        # Lambda Optimization Runbook Tools
        elif name == "lambda_optimization":
            return await run_lambda_optimization_analysis(arguments)
        elif name == "lambda_report":
            return await generate_lambda_optimization_report(arguments)
        elif name == "lambda_unused":
            return await identify_unused_lambda_functions(arguments)
        
        # S3 Optimization Tools
        elif name == "s3_comprehensive_analysis":
            return await run_s3_comprehensive_analysis(arguments)
        elif name == "s3_general_spend_analysis":
            return await run_s3_general_spend_analysis(arguments)
        elif name == "s3_storage_class_selection":
            return await run_s3_storage_class_selection(arguments)
        elif name == "s3_storage_class_validation":
            return await run_s3_storage_class_validation(arguments)
        elif name == "s3_archive_optimization":
            return await run_s3_archive_optimization(arguments)
        elif name == "s3_api_cost_minimization":
            return await run_s3_api_cost_minimization(arguments)
        elif name == "s3_multipart_cleanup":
            return await run_s3_multipart_cleanup(arguments)
        elif name == "s3_governance_check":
            return await run_s3_governance_check(arguments)
        elif name == "s3_comprehensive_optimization_tool":
            return await run_s3_comprehensive_optimization_tool(arguments)
        elif name == "s3_quick_analysis":
            return await run_s3_quick_analysis(arguments)
        elif name == "s3_bucket_analysis":
            return await run_s3_bucket_analysis(arguments)
        
        # CloudTrail Tools
        elif name == "generate_cloudtrail_report":
            return await generate_cloudtrail_report(arguments)
        elif name == "get_management_trails":
            return await get_management_trails(arguments)
        elif name == "run_cloudtrail_trails_analysis":
            return await run_cloudtrail_trails_analysis(arguments)
        
        # CloudWatch Optimization Tools
        elif name == "cloudwatch_general_spend_analysis":
            return await run_cloudwatch_general_spend_analysis(arguments)
        elif name == "cloudwatch_metrics_optimization":
            return await run_cloudwatch_metrics_optimization(arguments)
        elif name == "cloudwatch_logs_optimization":
            return await run_cloudwatch_logs_optimization(arguments)
        elif name == "cloudwatch_alarms_and_dashboards_optimization":
            return await run_cloudwatch_alarms_and_dashboards_optimization(arguments)
        elif name == "cloudwatch_comprehensive_optimization_tool":
            return await run_cloudwatch_comprehensive_optimization_tool(arguments)
        elif name == "query_cloudwatch_analysis_results":
            return await query_cloudwatch_analysis_results(arguments)
        elif name == "validate_cloudwatch_cost_preferences":
            return await validate_cloudwatch_cost_preferences(arguments)
        elif name == "get_cloudwatch_cost_estimate":
            return await get_cloudwatch_cost_estimate(arguments)
        
        # Database Savings Plans Tools
        elif name == "database_savings_plans_analysis":
            return await run_database_savings_plans_analysis(arguments)
        elif name == "database_savings_plans_purchase_analyzer":
            return await run_purchase_analyzer(arguments)
        elif name == "database_savings_plans_existing_analysis":
            return await analyze_existing_savings_plans(arguments)
        
        # Additional AWS Service Tools
        elif name == "get_trusted_advisor_checks":
            return await get_trusted_advisor_checks(arguments)
        elif name == "get_performance_insights_metrics":
            return await get_performance_insights_metrics(arguments)
        
        # Comprehensive Analysis
        elif name == "comprehensive_analysis":
            return await run_comprehensive_cost_analysis(arguments)
        
        # NAT Gateway Optimization Tools
        elif name == "nat_gateway_optimization":
            return await run_nat_gateway_optimization_analysis(arguments)
        elif name == "nat_gateway_underutilized":
            return await identify_underutilized_nat_gateways(arguments)
        elif name == "nat_gateway_redundant":
            return await identify_redundant_nat_gateways(arguments)
        elif name == "nat_gateway_unused":
            return await identify_unused_nat_gateways(arguments)
        
        else:
            logger.warning(f"Unknown tool requested: {name}")
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error calling tool '{name}': {str(e)} (execution time: {execution_time:.2f}s)")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    log_function_exit(logger, f"call_tool[{name}]", "success", execution_time)

# Core AWS Service Functions
@handle_aws_error
async def get_cost_explorer_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Retrieve cost data from AWS Cost Explorer."""
    start_date = arguments["start_date"]
    end_date = arguments["end_date"]
    granularity = arguments.get("granularity", "MONTHLY")
    metrics = arguments.get("metrics", ["BlendedCost", "UnblendedCost"])
    group_by = arguments.get("group_by")
    
    ce_client = get_cost_explorer_client()
    params = {
        'TimePeriod': {'Start': start_date, 'End': end_date},
        'Granularity': granularity,
        'Metrics': metrics
    }
    if group_by:
        params['GroupBy'] = group_by
        
    response = ce_client.get_cost_and_usage(**params)
    result = ResponseFormatter.success_response(
        data=response,
        message=f"Retrieved cost data from {start_date} to {end_date}",
        analysis_type="cost_explorer"
    )
    
    # Add documentation links
    result = add_documentation_links(result)
    return ResponseFormatter.to_text_content(result)

@handle_aws_error
async def list_cost_optimization_enrollment_statuses(arguments: Dict[str, Any]) -> List[TextContent]:
    """List Cost Optimization Hub enrollment statuses."""
    include_organization_info = arguments.get("include_organization_info", False)
    client = get_cost_optimization_hub_client()
    
    params = {}
    if include_organization_info:
        params['includeOrganizationInfo'] = include_organization_info
        
    # Initialize variables for pagination
    all_items = []
    next_token = None

    # Use pagination to retrieve all results
    while True:
        # Add NextToken if we have one from a previous call
        if next_token:
            params['nextToken'] = next_token

        # Make the API call
        response = client.list_enrollment_statuses(**params)

        # Add items from this page to our collection
        if 'items' in response:
            all_items.extend(response['items'])

        # Check if there are more pages
        if 'nextToken' in response:
            next_token = response['nextToken']
        else:
            break

    # Create our final result with all items
    result = ResponseFormatter.success_response(
        data={
            "items": all_items,
            "count": len(all_items)
        },
        message=f"Retrieved {len(all_items)} enrollment statuses",
        analysis_type="cost_optimization_hub_enrollment"
    )
    
    # Add documentation links
    result = add_documentation_links(result)
    return ResponseFormatter.to_text_content(result)

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

        # Initialize variables for pagination
        all_items = []
        next_token = None

        # Use pagination to retrieve all results
        while True:
            # Add NextToken if we have one from a previous call
            if next_token:
                params['nextToken'] = next_token

            # Make the API call
            response = client.list_recommendations(**params)

            # Add items from this page to our collection
            if 'items' in response:
                all_items.extend(response['items'])

            # Check if there are more pages
            if 'nextToken' in response:
                next_token = response['nextToken']
            else:
                break

            # If the user requested a specific maximum number of results and we've reached it, stop
            if max_results is not None and len(all_items) >= max_results:
                all_items = all_items[:max_results]  # Truncate to exact requested number
                break

        # Create our final result with all items
        result = {
            "status": "success",
            "data": {
                "items": all_items,
                "count": len(all_items)
            },
            "message": f"Retrieved {len(all_items)} cost optimization recommendations"
        }
        # Add documentation links
        result = add_documentation_links(result)
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
        # Add documentation links
        result = add_documentation_links(result)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except ClientError as e:
        error_msg = f"AWS API Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Import runbook functions from playbooks
from playbooks.ec2.ec2_optimization import (
    run_ec2_right_sizing_analysis,
    generate_ec2_right_sizing_report,
    identify_stopped_ec2_instances,
    identify_unattached_elastic_ips,
    identify_old_generation_instances,
    identify_instances_without_monitoring,
    identify_graviton_compatible_instances_mcp as identify_graviton_compatible_instances,
    analyze_burstable_instances_mcp as analyze_burstable_instances,
    identify_spot_opportunities_mcp as identify_spot_opportunities,
    identify_unused_reservations_mcp as identify_unused_reservations,
    identify_scheduling_opportunities_mcp as identify_scheduling_opportunities,
    analyze_commitment_plans_mcp as analyze_commitment_plans,
    identify_governance_violations_mcp as identify_governance_violations,
    generate_comprehensive_report_mcp as generate_comprehensive_report
)

from playbooks.ebs.ebs_optimization import (
    run_ebs_optimization_analysis,
    generate_ebs_optimization_report_mcp as generate_ebs_optimization_report,
    identify_unused_ebs_volumes
)

from playbooks.rds.rds_optimization import (
    run_rds_optimization_analysis,
    generate_rds_optimization_report,
    identify_idle_rds_instances_wrapper
)

from playbooks.rds.database_savings_plans import (
    run_database_savings_plans_analysis,
    run_purchase_analyzer,
    analyze_existing_savings_plans
)

from playbooks.aws_lambda.lambda_optimization import (
    run_lambda_optimization_analysis,
    generate_lambda_optimization_report,
    identify_unused_lambda_functions_mcp as identify_unused_lambda_functions
)

from playbooks.s3.s3_optimization_orchestrator import (
    run_s3_comprehensive_analysis,
    run_s3_general_spend_analysis,
    run_s3_storage_class_selection,
    run_s3_storage_class_validation,
    run_s3_archive_optimization,
    run_s3_api_cost_minimization,
    run_s3_multipart_cleanup,
    run_s3_governance_check,
    run_s3_comprehensive_optimization_tool,
    run_s3_quick_analysis,
    run_s3_bucket_analysis
)

from playbooks.cloudtrail.cloudtrail_optimization import (
    generate_cloudtrail_report_mcp as generate_cloudtrail_report,
    get_management_trails_mcp as get_management_trails,
    run_cloudtrail_trails_analysis_mcp as run_cloudtrail_trails_analysis
)

from playbooks.comprehensive_optimization import (
    run_comprehensive_cost_analysis
)

from playbooks.cloudwatch.cloudwatch_optimization import (
    run_cloudwatch_general_spend_analysis_mcp as run_cloudwatch_general_spend_analysis,
    run_cloudwatch_metrics_optimization_mcp as run_cloudwatch_metrics_optimization,
    run_cloudwatch_logs_optimization_mcp as run_cloudwatch_logs_optimization,
    run_cloudwatch_alarms_and_dashboards_optimization_mcp as run_cloudwatch_alarms_and_dashboards_optimization,
    run_cloudwatch_comprehensive_optimization_tool_mcp as run_cloudwatch_comprehensive_optimization_tool,
    query_cloudwatch_analysis_results_mcp as query_cloudwatch_analysis_results,
    validate_cloudwatch_cost_preferences_mcp as validate_cloudwatch_cost_preferences,
    get_cloudwatch_cost_estimate_mcp as get_cloudwatch_cost_estimate
)

from runbook_functions import (
    run_nat_gateway_optimization_analysis,
    identify_underutilized_nat_gateways,
    identify_redundant_nat_gateways,
    identify_unused_nat_gateways
)

# Additional AWS service functions
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
        
        # Add documentation links
        result = add_documentation_links(result)
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
        
        # Add documentation links
        result = add_documentation_links(result, "rds")
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