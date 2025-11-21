"""
CloudWatch Optimization MCP Wrapper Functions

This module provides MCP-compatible wrapper functions for CloudWatch optimization analysis.
These functions follow the same pattern as other service optimization modules in the CFM Tips project.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any
from mcp.types import TextContent

from utils.logging_config import log_function_entry, log_function_exit
from utils.error_handler import handle_aws_error, ResponseFormatter
from utils.documentation_links import add_documentation_links

logger = logging.getLogger(__name__)

@handle_aws_error
async def run_cloudwatch_general_spend_analysis_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudWatch general spend analysis to understand cost breakdown across logs, metrics, alarms, and dashboards."""
    log_function_entry(logger, "run_cloudwatch_general_spend_analysis_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.cloudwatch_optimization_analyzer import CloudWatchOptimizationAnalyzer
        from playbooks.cloudwatch.cost_controller import CostPreferences
        
        region = arguments.get("region")
        page = arguments.get("page", 1)
        timeout_seconds = arguments.get("timeout_seconds", 120)
        
        # Build cost preferences from arguments
        cost_preferences = CostPreferences(
            allow_cost_explorer=arguments.get("allow_cost_explorer", False),
            allow_aws_config=arguments.get("allow_aws_config", False),
            allow_cloudtrail=arguments.get("allow_cloudtrail", False),
            allow_minimal_cost_metrics=arguments.get("allow_minimal_cost_metrics", False)
        )
        
        # Initialize analyzer
        analyzer = CloudWatchOptimizationAnalyzer(region=region, cost_preferences=cost_preferences)
        
        # Remove internal parameters from arguments
        analysis_args = {k: v for k, v in arguments.items() if k not in ['page', 'timeout_seconds']}
        
        # Execute analysis with timeout
        result = await asyncio.wait_for(
            analyzer.analyze_general_spend(page=page, **analysis_args),
            timeout=timeout_seconds
        )
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=result,
            message="CloudWatch general spend analysis completed successfully",
            analysis_type="cloudwatch_general_spend"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "run_cloudwatch_general_spend_analysis_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except asyncio.TimeoutError:
        error_message = f"CloudWatch general spend analysis timed out after {timeout_seconds} seconds"
        logger.error(error_message)
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            error_message, "timeout_error", "cloudwatch_general_spend"
        ))
    except Exception as e:
        logger.error(f"CloudWatch general spend analysis failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "analysis_error", "cloudwatch_general_spend"
        ))

@handle_aws_error
async def run_cloudwatch_metrics_optimization_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudWatch metrics optimization analysis to identify custom metrics cost optimization opportunities."""
    log_function_entry(logger, "run_cloudwatch_metrics_optimization_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.cloudwatch_optimization_analyzer import CloudWatchOptimizationAnalyzer
        from playbooks.cloudwatch.cost_controller import CostPreferences
        
        region = arguments.get("region")
        page = arguments.get("page", 1)
        timeout_seconds = arguments.get("timeout_seconds", 120)
        
        # Build cost preferences from arguments
        cost_preferences = CostPreferences(
            allow_cost_explorer=arguments.get("allow_cost_explorer", False),
            allow_aws_config=arguments.get("allow_aws_config", False),
            allow_cloudtrail=arguments.get("allow_cloudtrail", False),
            allow_minimal_cost_metrics=arguments.get("allow_minimal_cost_metrics", False)
        )
        
        # Initialize analyzer
        analyzer = CloudWatchOptimizationAnalyzer(region=region, cost_preferences=cost_preferences)
        
        # Remove internal parameters from arguments
        analysis_args = {k: v for k, v in arguments.items() if k not in ['page', 'timeout_seconds']}
        
        # Execute analysis with timeout
        result = await asyncio.wait_for(
            analyzer.analyze_metrics_optimization(page=page, **analysis_args),
            timeout=timeout_seconds
        )
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=result,
            message="CloudWatch metrics optimization analysis completed successfully",
            analysis_type="cloudwatch_metrics_optimization"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "run_cloudwatch_metrics_optimization_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except asyncio.TimeoutError:
        error_message = f"CloudWatch metrics optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(error_message)
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            error_message, "timeout_error", "cloudwatch_metrics_optimization"
        ))
    except Exception as e:
        logger.error(f"CloudWatch metrics optimization analysis failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "analysis_error", "cloudwatch_metrics_optimization"
        ))

@handle_aws_error
async def run_cloudwatch_logs_optimization_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudWatch logs optimization analysis to identify log retention and ingestion cost optimization opportunities."""
    log_function_entry(logger, "run_cloudwatch_logs_optimization_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.cloudwatch_optimization_analyzer import CloudWatchOptimizationAnalyzer
        from playbooks.cloudwatch.cost_controller import CostPreferences
        
        region = arguments.get("region")
        page = arguments.get("page", 1)
        timeout_seconds = arguments.get("timeout_seconds", 120)
        
        # Build cost preferences from arguments
        cost_preferences = CostPreferences(
            allow_cost_explorer=arguments.get("allow_cost_explorer", False),
            allow_aws_config=arguments.get("allow_aws_config", False),
            allow_cloudtrail=arguments.get("allow_cloudtrail", False),
            allow_minimal_cost_metrics=arguments.get("allow_minimal_cost_metrics", False)
        )
        
        # Initialize analyzer
        analyzer = CloudWatchOptimizationAnalyzer(region=region, cost_preferences=cost_preferences)
        
        # Remove internal parameters from arguments
        analysis_args = {k: v for k, v in arguments.items() if k not in ['page', 'timeout_seconds']}
        
        # Execute analysis with timeout
        result = await asyncio.wait_for(
            analyzer.analyze_logs_optimization(page=page, **analysis_args),
            timeout=timeout_seconds
        )
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=result,
            message="CloudWatch logs optimization analysis completed successfully",
            analysis_type="cloudwatch_logs_optimization"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "run_cloudwatch_logs_optimization_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except asyncio.TimeoutError:
        error_message = f"CloudWatch logs optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(error_message)
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            error_message, "timeout_error", "cloudwatch_logs_optimization"
        ))
    except Exception as e:
        logger.error(f"CloudWatch logs optimization analysis failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "analysis_error", "cloudwatch_logs_optimization"
        ))

@handle_aws_error
async def run_cloudwatch_alarms_and_dashboards_optimization_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudWatch alarms and dashboards optimization analysis to identify monitoring efficiency improvements."""
    log_function_entry(logger, "run_cloudwatch_alarms_and_dashboards_optimization_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.cloudwatch_optimization_analyzer import CloudWatchOptimizationAnalyzer
        from playbooks.cloudwatch.cost_controller import CostPreferences
        
        region = arguments.get("region")
        page = arguments.get("page", 1)
        timeout_seconds = arguments.get("timeout_seconds", 120)
        
        # Build cost preferences from arguments
        cost_preferences = CostPreferences(
            allow_cost_explorer=arguments.get("allow_cost_explorer", False),
            allow_aws_config=arguments.get("allow_aws_config", False),
            allow_cloudtrail=arguments.get("allow_cloudtrail", False),
            allow_minimal_cost_metrics=arguments.get("allow_minimal_cost_metrics", False)
        )
        
        # Initialize analyzer
        analyzer = CloudWatchOptimizationAnalyzer(region=region, cost_preferences=cost_preferences)
        
        # Remove internal parameters from arguments
        analysis_args = {k: v for k, v in arguments.items() if k not in ['page', 'timeout_seconds']}
        
        # Execute analysis with timeout
        result = await asyncio.wait_for(
            analyzer.analyze_alarms_optimization(page=page, **analysis_args),
            timeout=timeout_seconds
        )
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=result,
            message="CloudWatch alarms and dashboards optimization analysis completed successfully",
            analysis_type="cloudwatch_alarms_dashboards_optimization"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "run_cloudwatch_alarms_and_dashboards_optimization_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except asyncio.TimeoutError:
        error_message = f"CloudWatch alarms and dashboards optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(error_message)
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            error_message, "timeout_error", "cloudwatch_alarms_dashboards_optimization"
        ))
    except Exception as e:
        logger.error(f"CloudWatch alarms and dashboards optimization analysis failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "analysis_error", "cloudwatch_alarms_dashboards_optimization"
        ))

@handle_aws_error
async def run_cloudwatch_comprehensive_optimization_tool_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive CloudWatch optimization using the unified optimization tool with intelligent orchestration."""
    log_function_entry(logger, "run_cloudwatch_comprehensive_optimization_tool_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.cloudwatch_optimization_tool import CloudWatchOptimizationTool
        
        region = arguments.get("region")
        timeout_seconds = arguments.get("timeout_seconds", 120)
        
        # Initialize comprehensive optimization tool
        tool = CloudWatchOptimizationTool(region=region)
        
        # Execute comprehensive analysis with timeout
        result = await asyncio.wait_for(
            tool.execute_comprehensive_optimization_analysis(**arguments),
            timeout=timeout_seconds
        )
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=result,
            message="CloudWatch comprehensive optimization analysis completed successfully",
            analysis_type="cloudwatch_comprehensive_optimization"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "run_cloudwatch_comprehensive_optimization_tool_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except asyncio.TimeoutError:
        error_message = f"CloudWatch comprehensive optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(error_message)
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            error_message, "timeout_error", "cloudwatch_comprehensive_optimization"
        ))
    except Exception as e:
        logger.error(f"CloudWatch comprehensive optimization analysis failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "analysis_error", "cloudwatch_comprehensive_optimization"
        ))

@handle_aws_error
async def query_cloudwatch_analysis_results_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Query stored CloudWatch analysis results using SQL queries."""
    log_function_entry(logger, "query_cloudwatch_analysis_results_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        # Import the sync function from runbook_functions and call it
        from runbook_functions import query_cloudwatch_analysis_results
        
        # Call the existing function and convert to MCP format
        result = await query_cloudwatch_analysis_results(arguments)
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "query_cloudwatch_analysis_results_mcp", "success", execution_time)
        
        return result
        
    except Exception as e:
        logger.error(f"CloudWatch analysis results query failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "query_error", "cloudwatch_analysis_results"
        ))

@handle_aws_error
async def validate_cloudwatch_cost_preferences_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Validate CloudWatch cost preferences and get functionality coverage estimates."""
    log_function_entry(logger, "validate_cloudwatch_cost_preferences_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        # Import the sync function from runbook_functions and call it
        from runbook_functions import validate_cloudwatch_cost_preferences
        
        # Call the existing function and convert to MCP format
        result = await validate_cloudwatch_cost_preferences(arguments)
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "validate_cloudwatch_cost_preferences_mcp", "success", execution_time)
        
        return result
        
    except Exception as e:
        logger.error(f"CloudWatch cost preferences validation failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "validation_error", "cloudwatch_cost_preferences"
        ))

@handle_aws_error
async def get_cloudwatch_cost_estimate_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get detailed cost estimate for CloudWatch optimization analysis based on enabled features."""
    log_function_entry(logger, "get_cloudwatch_cost_estimate_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        # Import the sync function from runbook_functions and call it
        from runbook_functions import get_cloudwatch_cost_estimate
        
        # Call the existing function and convert to MCP format
        result = await get_cloudwatch_cost_estimate(arguments)
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "get_cloudwatch_cost_estimate_mcp", "success", execution_time)
        
        return result
        
    except Exception as e:
        logger.error(f"CloudWatch cost estimate failed: {str(e)}")
        return ResponseFormatter.to_text_content(ResponseFormatter.error_response(
            str(e), "estimation_error", "cloudwatch_cost_estimate"
        ))