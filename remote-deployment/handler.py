#!/usr/bin/env python3
"""
CFM Tips — AWS Lambda handler for remote MCP over Streamable HTTP.

Uses awslabs.mcp_lambda_handler to expose the progressive discovery API
as a Lambda-backed MCP server invokable via HTTP.

Supports cross-account analysis via STS AssumeRole — pass a role_arn
parameter to any tool to analyze a different AWS account.
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
from playbook_resolver import GlobalResolver, ToolNotAvailableError

logger = setup_logging()

# ---------------------------------------------------------------------------
# MCP server (module-level so it survives across warm invocations)
# ---------------------------------------------------------------------------
mcp_server = MCPLambdaHandler(
    name="cfm-tips",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global resolver — singleton, lazy-initializes on first tool call via
# _ensure_started().  Lambda can't await at module level, so we accept
# the lazy path here.  For non-Lambda contexts prefer GlobalResolver.create().
# ---------------------------------------------------------------------------
global_resolver = GlobalResolver()


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
# Tool definitions
# ---------------------------------------------------------------------------

@mcp_server.tool()
def get_optimization_targets(role_arn: str = "") -> str:
    """Get available AWS services for cost optimization.

    Entry point for progressive discovery. Returns service ids,
    display names, available operation counts, and active data sources.

    Args:
        role_arn: IAM role ARN to assume for cross-account analysis (optional)
    """
    resolver = _run_async(GlobalResolver.for_account(role_arn)) if role_arn else global_resolver
    result = _run_async(resolver.get_optimization_targets())
    return json.dumps(result, indent=2, default=str)


@mcp_server.tool()
def get_optimization_runbook_for_target(target: str, role_arn: str = "") -> str:
    """Get available optimization operations for a target service.

    Returns operation names, descriptions, parameters, and the
    data source backing each operation.

    Args:
        target: Target id from get_optimization_targets (e.g. 'amazon_ec2')
        role_arn: IAM role ARN to assume for cross-account analysis (optional)
    """
    resolver = _run_async(GlobalResolver.for_account(role_arn)) if role_arn else global_resolver
    result = _run_async(resolver.get_operations_for_target(target))
    return json.dumps(result, indent=2, default=str)


@mcp_server.tool()
def execute_runbook(operation: str, parameters: dict = None, role_arn: str = "") -> str:
    """Execute a specific optimization operation.

    Provide the operation name and parameters. Results include pagination,
    sorting by savings, and documentation links.

    Args:
        operation: Operation name from get_optimization_runbook_for_target
        parameters: Operation parameters as key-value pairs
        role_arn: IAM role ARN to assume for cross-account analysis (optional)
    """
    params = parameters or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            params = {}
    if not isinstance(params, dict):
        params = {}

    effective_role = role_arn or params.pop("role_arn", "") or ""
    params.pop("role_arn", None)

    resolver = _run_async(GlobalResolver.for_account(effective_role)) if effective_role else global_resolver

    try:
        target_id, tool_name = _run_async(
            resolver.resolve_operation_name(operation)
        )
    except ToolNotAvailableError as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)

    logger.info(f"Executing {operation} -> {target_id}.{tool_name} with {params}")

    try:
        result = _run_async(
            resolver.execute_operation(target_id, tool_name, **params)
        )
    except Exception as e:
        logger.error(f"execute_runbook error: {e}\n{traceback.format_exc()}")
        result = {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """AWS Lambda handler — delegates to MCPLambdaHandler."""
    return mcp_server.handle_request(event, context)
