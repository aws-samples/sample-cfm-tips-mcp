"""
Centralized logging configuration for CFM Tips MCP Server

"""

import logging
import sys
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class StandardFormatter(logging.Formatter):
    """Enhanced standard formatter with more context."""
    
    def __init__(self):
        super().__init__(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )


def setup_logging(structured: bool = False, log_level: str = "INFO"):
    """
    Configure comprehensive logging for the application.
    
    Args:
        structured: Whether to use structured JSON logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # Create appropriate formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = StandardFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handlers
    try:
        # Try to create logs directory if it doesn't exist
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Try main log file in logs directory first
        log_file = os.path.join(log_dir, 'cfm_tips_mcp.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Try error log file
        error_file = os.path.join(log_dir, 'cfm_tips_mcp_errors.log')
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
    except (OSError, PermissionError) as e:
        # If we can't write to logs directory, try current directory
        try:
            file_handler = logging.FileHandler('cfm_tips_mcp.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            error_handler = logging.FileHandler('cfm_tips_mcp_errors.log')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
        except (OSError, PermissionError):
            # If we can't write anywhere, try temp directory
            try:
                temp_dir = tempfile.gettempdir()
                temp_log = os.path.join(temp_dir, 'cfm_tips_mcp.log')
                file_handler = logging.FileHandler(temp_log)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                
                temp_error = os.path.join(temp_dir, 'cfm_tips_mcp_errors.log')
                error_handler = logging.FileHandler(temp_error)
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(formatter)
                root_logger.addHandler(error_handler)
                
                # Log where we're writing files
                print(f"Warning: Using temp directory for logs: {temp_dir}")
                
            except (OSError, PermissionError):
                # If all else fails, raise error since we need file logging
                raise RuntimeError("Could not create log files in any location")
    
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


def create_structured_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Create a logger with structured logging capabilities.
    
    Args:
        name: Logger name
        extra_fields: Additional fields to include in all log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create adapter to add extra fields
        logger = logging.LoggerAdapter(logger, extra_fields)
    
    return logger


def log_s3_operation(logger, operation: str, bucket_name: Optional[str] = None, 
                    object_key: Optional[str] = None, **kwargs):
    """
    Log S3 operations with structured data.
    
    Args:
        logger: Logger instance
        operation: S3 operation name
        bucket_name: S3 bucket name
        object_key: S3 object key
        **kwargs: Additional operation parameters
    """
    log_data = {
        'operation_type': 's3_operation',
        'operation': operation,
        'bucket_name': bucket_name,
        'object_key': object_key
    }
    log_data.update(kwargs)
    
    logger.info(f"S3 Operation: {operation}", extra=log_data)


def log_analysis_start(logger, analysis_type: str, session_id: Optional[str] = None, **kwargs):
    """
    Log analysis start with structured data.
    
    Args:
        logger: Logger instance
        analysis_type: Type of analysis
        session_id: Session identifier
        **kwargs: Additional analysis parameters
    """
    log_data = {
        'event_type': 'analysis_start',
        'analysis_type': analysis_type,
        'session_id': session_id
    }
    log_data.update(kwargs)
    
    logger.info(f"Starting analysis: {analysis_type}", extra=log_data)


def log_analysis_complete(logger, analysis_type: str, status: str, execution_time: float,
                         session_id: Optional[str] = None, **kwargs):
    """
    Log analysis completion with structured data.
    
    Args:
        logger: Logger instance
        analysis_type: Type of analysis
        status: Analysis status
        execution_time: Execution time in seconds
        session_id: Session identifier
        **kwargs: Additional analysis results
    """
    log_data = {
        'event_type': 'analysis_complete',
        'analysis_type': analysis_type,
        'status': status,
        'execution_time': execution_time,
        'session_id': session_id
    }
    log_data.update(kwargs)
    
    logger.info(f"Completed analysis: {analysis_type} - Status: {status}", extra=log_data)


def log_cost_optimization_finding(logger, finding_type: str, resource_id: str, 
                                 potential_savings: Optional[float] = None, **kwargs):
    """
    Log cost optimization findings with structured data.
    
    Args:
        logger: Logger instance
        finding_type: Type of optimization finding
        resource_id: Resource identifier
        potential_savings: Estimated cost savings
        **kwargs: Additional finding details
    """
    log_data = {
        'event_type': 'cost_optimization_finding',
        'finding_type': finding_type,
        'resource_id': resource_id,
        'potential_savings': potential_savings
    }
    log_data.update(kwargs)
    
    logger.info(f"Cost optimization finding: {finding_type} for {resource_id}", extra=log_data)


def log_session_operation(logger, operation: str, session_id: str, **kwargs):
    """
    Log session operations with structured data.
    
    Args:
        logger: Logger instance
        operation: Session operation
        session_id: Session identifier
        **kwargs: Additional operation details
    """
    log_data = {
        'event_type': 'session_operation',
        'operation': operation,
        'session_id': session_id
    }
    log_data.update(kwargs)
    
    logger.info(f"Session operation: {operation} for session {session_id}", extra=log_data)


def log_cloudwatch_operation(logger, operation: str, component: Optional[str] = None, 
                            cost_incurred: bool = False, **kwargs):
    """
    Log CloudWatch operations with structured data and cost tracking.
    
    Args:
        logger: Logger instance
        operation: CloudWatch operation name
        component: CloudWatch component (logs, metrics, alarms, dashboards)
        cost_incurred: Whether the operation incurred costs
        **kwargs: Additional operation parameters
    """
    log_data = {
        'operation_type': 'cloudwatch_operation',
        'operation': operation,
        'component': component,
        'cost_incurred': cost_incurred
    }
    log_data.update(kwargs)
    
    if cost_incurred:
        logger.warning(f"CloudWatch Operation (COST INCURRED): {operation}", extra=log_data)
    else:
        logger.info(f"CloudWatch Operation: {operation}", extra=log_data)


# CloudWatch-specific logging methods consolidated into log_cloudwatch_operation
# These specialized methods have been removed in favor of the generic log_cloudwatch_operation method


# Removed setup_cloudwatch_logging - use setup_logging instead with log_cloudwatch_operation for CloudWatch-specific events