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
        
    except asyncio.TimeoutError as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"CloudWatch general spend analysis timed out after {timeout_seconds} seconds"
        logger.error(f"{error_message}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": "TimeoutError",
            "message": error_message,
            "context": "cloudwatch_general_spend",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch general spend analysis failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_general_spend",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

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
        
    except asyncio.TimeoutError as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"CloudWatch metrics optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(f"{error_message}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": "TimeoutError",
            "message": error_message,
            "context": "cloudwatch_metrics_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch metrics optimization analysis failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_metrics_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

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
        
    except asyncio.TimeoutError as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"CloudWatch logs optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(f"{error_message}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": "TimeoutError",
            "message": error_message,
            "context": "cloudwatch_logs_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch logs optimization analysis failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_logs_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

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
        
    except asyncio.TimeoutError as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"CloudWatch alarms and dashboards optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(f"{error_message}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": "TimeoutError",
            "message": error_message,
            "context": "cloudwatch_alarms_dashboards_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch alarms and dashboards optimization analysis failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_alarms_dashboards_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

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
        
    except asyncio.TimeoutError as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"CloudWatch comprehensive optimization analysis timed out after {timeout_seconds} seconds"
        logger.error(f"{error_message}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": "TimeoutError",
            "message": error_message,
            "context": "cloudwatch_comprehensive_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch comprehensive optimization analysis failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_comprehensive_optimization",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

@handle_aws_error
async def query_cloudwatch_analysis_results_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Query stored CloudWatch analysis results using SQL queries."""
    log_function_entry(logger, "query_cloudwatch_analysis_results_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
        from utils.session_manager import get_session_manager
        
        region = arguments.get("region")
        query = arguments.get("query", "SELECT * FROM cloudwatch_analysis_results LIMIT 10")
        limit = arguments.get("limit", 100)
        
        # Initialize orchestrator
        orchestrator = CloudWatchOptimizationOrchestrator(region=region)
        
        # Get session manager and execute query
        session_manager = get_session_manager()
        
        try:
            # Execute the SQL query on stored results
            results = session_manager.execute_query(orchestrator.session_id, query)
            
            # Limit results if needed
            if len(results) > limit:
                results = results[:limit]
            
            # Format response
            formatted_result = ResponseFormatter.success_response(
                data={
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "session_id": orchestrator.session_id
                },
                message=f"Retrieved {len(results)} CloudWatch analysis results",
                analysis_type="cloudwatch_query_results"
            )
            
            # Add documentation links
            formatted_result = add_documentation_links(formatted_result, "cloudwatch")
            
            execution_time = time.time() - start_time
            log_function_exit(logger, "query_cloudwatch_analysis_results_mcp", "success", execution_time)
            
            return ResponseFormatter.to_text_content(formatted_result)
            
        except Exception as query_error:
            # If query fails, return helpful error message
            formatted_result = ResponseFormatter.success_response(
                data={
                    "query": query,
                    "results": [],
                    "count": 0,
                    "session_id": orchestrator.session_id,
                    "note": f"Query execution failed: {str(query_error)}. This may be because no analysis has been run yet in this session."
                },
                message="No results found or query failed",
                analysis_type="cloudwatch_query_results"
            )
            
            formatted_result = add_documentation_links(formatted_result, "cloudwatch")
            return ResponseFormatter.to_text_content(formatted_result)
        
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch analysis results query failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_analysis_results",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

@handle_aws_error
async def validate_cloudwatch_cost_preferences_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Validate CloudWatch cost preferences and get functionality coverage estimates."""
    log_function_entry(logger, "validate_cloudwatch_cost_preferences_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
        
        region = arguments.get("region")
        cost_preferences = arguments.get("cost_preferences", {})
        
        # Initialize orchestrator
        orchestrator = CloudWatchOptimizationOrchestrator(region=region)
        
        # Validate cost preferences using orchestrator method
        validation_result = orchestrator.validate_cost_preferences(**cost_preferences)
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=validation_result,
            message="CloudWatch cost preferences validated successfully",
            analysis_type="cloudwatch_cost_preferences_validation"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "validate_cloudwatch_cost_preferences_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch cost preferences validation failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_cost_preferences",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)

@handle_aws_error
async def get_cloudwatch_cost_estimate_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get detailed cost estimate for CloudWatch optimization analysis based on enabled features."""
    log_function_entry(logger, "get_cloudwatch_cost_estimate_mcp", arguments=arguments)
    start_time = time.time()
    
    try:
        from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
        
        region = arguments.get("region")
        analysis_type = arguments.get("analysis_type", "comprehensive")
        lookback_days = arguments.get("lookback_days", 30)
        
        # Initialize orchestrator
        orchestrator = CloudWatchOptimizationOrchestrator(region=region)
        
        # Prepare analysis scope
        analysis_scope = {
            'lookback_days': lookback_days,
            'analysis_types': [analysis_type] if analysis_type != "comprehensive" else ['general_spend', 'logs_optimization', 'metrics_optimization', 'alarms_and_dashboards']
        }
        
        # Get cost estimate using orchestrator method
        cost_estimate = orchestrator.get_cost_estimate(analysis_scope=analysis_scope, **arguments)
        
        # Format response
        formatted_result = ResponseFormatter.success_response(
            data=cost_estimate,
            message=f"CloudWatch cost estimate generated for {analysis_type} analysis",
            analysis_type="cloudwatch_cost_estimate"
        )
        
        # Add documentation links
        formatted_result = add_documentation_links(formatted_result, "cloudwatch")
        
        execution_time = time.time() - start_time
        log_function_exit(logger, "get_cloudwatch_cost_estimate_mcp", "success", execution_time)
        
        return ResponseFormatter.to_text_content(formatted_result)
        
    except Exception as e:
        import traceback
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"CloudWatch cost estimate failed: {str(e)}\n{error_traceback}")
        error_dict = {
            "status": "error",
            "error_code": type(e).__name__,
            "message": str(e),
            "context": "cloudwatch_cost_estimate",
            "traceback": error_traceback
        }
        return ResponseFormatter.to_text_content(error_dict)