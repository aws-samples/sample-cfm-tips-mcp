#!/usr/bin/env python3
"""
CFM Tips — AWS Lambda handler for remote MCP over Streamable HTTP.

Uses awslabs.mcp_lambda_handler to expose the legacy tool surface
as a Lambda-backed MCP server invokable via HTTP.

Wraps the existing mcp_server_with_runbooks call_tool dispatcher
so every tool available locally is also available remotely.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import traceback

from awslabs.mcp_lambda_handler import MCPLambdaHandler

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# MCP server (module-level so it survives across warm invocations)
# ---------------------------------------------------------------------------
mcp_server = MCPLambdaHandler(
    name="cfm-tips",
    version="1.0.0",
)


def _run_async(coro):
    """Run an async coroutine from sync context (Lambda handler is sync)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Import the legacy call_tool dispatcher — it handles all tool routing
# ---------------------------------------------------------------------------
from mcp_server_with_runbooks import call_tool, list_tools as legacy_list_tools


def _call_legacy_tool(tool_name: str, arguments: dict) -> str:
    """Call a legacy tool and return its text result as a string."""
    results = _run_async(call_tool(tool_name, arguments))
    # call_tool returns List[TextContent], extract text
    texts = []
    for r in results:
        if hasattr(r, 'text'):
            texts.append(r.text)
        else:
            texts.append(str(r))
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Progressive discovery tools — thin wrappers that enumerate and dispatch
# to the legacy tool surface.
# ---------------------------------------------------------------------------

# Build a tool catalog from the legacy list_tools at import time
_tool_catalog = None


def _get_catalog():
    global _tool_catalog
    if _tool_catalog is None:
        tools = _run_async(legacy_list_tools())
        _tool_catalog = {t.name: t for t in tools}
    return _tool_catalog


# Group tools by service prefix for the targets endpoint
_SERVICE_PREFIXES = {
    "amazon_ec2": ["ec2_"],
    "amazon_ebs": ["ebs_"],
    "amazon_rds": ["rds_"],
    "aws_lambda": ["lambda_"],
    "amazon_s3": ["s3_"],
    "aws_cloudtrail": ["cloudtrail", "get_management_trails", "run_cloudtrail"],
    "amazon_cloudwatch": ["cloudwatch_"],
    "aws_nat_gateway": ["nat_gateway_"],
    "database_savings_plans": ["database_savings_plans_"],
    "aws_cost_services": [
        "get_cost_explorer_data",
        "list_coh_enrollment",
        "get_coh_recommendations",
        "get_compute_optimizer_recommendations",
        "get_trusted_advisor_checks",
        "get_performance_insights_metrics",
        "comprehensive_analysis",
    ],
}

_SERVICE_DISPLAY = {
    "amazon_ec2": "Amazon EC2",
    "amazon_ebs": "Amazon EBS",
    "amazon_rds": "Amazon RDS",
    "aws_lambda": "AWS Lambda",
    "amazon_s3": "Amazon S3",
    "aws_cloudtrail": "AWS CloudTrail",
    "amazon_cloudwatch": "Amazon CloudWatch",
    "aws_nat_gateway": "NAT Gateway",
    "database_savings_plans": "Database Savings Plans",
    "aws_cost_services": "AWS Cost Services",
}


def _classify_tool(tool_name: str) -> str:
    """Return the service id a tool belongs to."""
    for svc_id, prefixes in _SERVICE_PREFIXES.items():
        for prefix in prefixes:
            if tool_name == prefix or tool_name.startswith(prefix):
                return svc_id
    return "other"


@mcp_server.tool()
def get_optimization_targets() -> str:
    """Get available AWS services for cost optimization.

    Entry point for progressive discovery. Returns service ids,
    display names, and available operation counts.
    """
    catalog = _get_catalog()
    # Group tools by service
    service_tools = {}
    for name in catalog:
        svc = _classify_tool(name)
        service_tools.setdefault(svc, []).append(name)

    targets = []
    for svc_id, tools in sorted(service_tools.items()):
        targets.append({
            "target_id": svc_id,
            "display_name": _SERVICE_DISPLAY.get(svc_id, svc_id),
            "operation_count": len(tools),
        })

    return json.dumps({
        "status": "success",
        "targets": targets,
        "total_operations": len(catalog),
    }, indent=2)


@mcp_server.tool()
def get_optimization_runbook_for_target(target: str) -> str:
    """Get available optimization operations for a target service.

    Returns operation names, descriptions, and parameters.

    Args:
        target: Target id from get_optimization_targets (e.g. 'amazon_ec2')
    """
    catalog = _get_catalog()
    operations = []
    for name, tool in catalog.items():
        if _classify_tool(name) == target:
            operations.append({
                "operation": name,
                "description": tool.description,
                "parameters": tool.inputSchema.get("properties", {}),
            })

    if not operations:
        return json.dumps({
            "status": "error",
            "message": f"No operations found for target '{target}'",
        }, indent=2)

    return json.dumps({
        "status": "success",
        "target": target,
        "display_name": _SERVICE_DISPLAY.get(target, target),
        "operations": operations,
    }, indent=2)


@mcp_server.tool()
def execute_runbook(operation: str, parameters: dict = None) -> str:
    """Execute a specific optimization operation.

    Provide the operation name and parameters.

    Args:
        operation: Operation name from get_optimization_runbook_for_target
        parameters: Operation parameters as key-value pairs
    """
    params = parameters or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            params = {}
    if not isinstance(params, dict):
        params = {}

    catalog = _get_catalog()
    if operation not in catalog:
        return json.dumps({
            "status": "error",
            "message": f"Unknown operation: {operation}",
        }, indent=2)

    try:
        result = _call_legacy_tool(operation, params)
        return result
    except Exception as e:
        logger.error(f"execute_runbook error: {e}\n{traceback.format_exc()}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }, indent=2)


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """AWS Lambda handler — delegates to MCPLambdaHandler."""
    return mcp_server.handle_request(event, context)
