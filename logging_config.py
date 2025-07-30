"""
Centralized logging configuration for CFM Tips MCP Server
"""

import logging
import sys
from datetime import datetime

def setup_logging():
    """Configure comprehensive logging for the application."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler('cfm_tips_mcp.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Error file handler
    error_handler = logging.FileHandler('cfm_tips_mcp_errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    return logging.getLogger(__name__)

def log_function_entry(logger, func_name, **kwargs):
    """Log function entry with parameters."""
    logger.info(f"Entering {func_name} with params: {kwargs}")

def log_function_exit(logger, func_name, result_status=None, execution_time=None):
    """Log function exit with results."""
    msg = f"Exiting {func_name}"
    if result_status:
        msg += f" - Status: {result_status}"
    if execution_time:
        msg += f" - Time: {execution_time:.2f}s"
    logger.info(msg)

def log_aws_api_call(logger, service, operation, **params):
    """Log AWS API calls."""
    logger.info(f"AWS API Call: {service}.{operation} with params: {params}")

def log_aws_api_error(logger, service, operation, error):
    """Log AWS API errors."""
    logger.error(f"AWS API Error: {service}.{operation} - {str(error)}")