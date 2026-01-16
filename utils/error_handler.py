"""
Centralized Error Handler for AWS Cost Optimization MCP Server

Provides consistent error handling and formatting across all modules.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError, NoCredentialsError
from mcp.types import TextContent
import json

logger = logging.getLogger(__name__)


class AWSErrorHandler:
    """Centralized AWS error handling and formatting."""
    
    # Common AWS error codes and their required permissions
    PERMISSION_MAP = {
        'AccessDenied': 'Check IAM permissions for the requested service',
        'UnauthorizedOperation': 'Verify IAM policy allows the requested operation',
        'InvalidUserID.NotFound': 'Check AWS credentials configuration',
        'TokenRefreshRequired': 'AWS credentials may have expired',
        'OptInRequired': 'Service may need to be enabled in AWS Console',
        'ServiceUnavailable': 'AWS service temporarily unavailable',
        'ThrottlingException': 'Request rate exceeded, implement retry logic',
        'ValidationException': 'Check request parameters and format'
    }
    
    @staticmethod
    def format_client_error(e: ClientError, context: str, 
                          required_permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Format AWS ClientError into standardized error response.
        
        Args:
            e: The ClientError exception
            context: Context where the error occurred
            required_permissions: List of required IAM permissions
            
        Returns:
            Standardized error response dictionary
        """
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        
        logger.error(f"AWS API Error in {context}: {error_code} - {error_message}\n{error_traceback}")
        
        response = {
            "status": "error",
            "error_code": error_code,
            "message": f"AWS API Error: {error_code} - {error_message}",
            "context": context,
            "traceback": error_traceback,
            "timestamp": logger.handlers[0].formatter.formatTime(logger.makeRecord(
                logger.name, logging.ERROR, __file__, 0, "", (), None
            )) if logger.handlers else None
        }
        
        # Add permission guidance
        if required_permissions:
            response["required_permissions"] = required_permissions
        elif error_code in AWSErrorHandler.PERMISSION_MAP:
            response["permission_guidance"] = AWSErrorHandler.PERMISSION_MAP[error_code]
        
        # Add retry guidance for throttling
        if error_code in ['ThrottlingException', 'RequestLimitExceeded']:
            response["retry_guidance"] = {
                "retryable": True,
                "suggested_delay": "exponential backoff starting at 1 second"
            }
        
        return response
    
    @staticmethod
    def format_no_credentials_error(context: str) -> Dict[str, Any]:
        """Format NoCredentialsError into standardized response."""
        logger.error(f"AWS credentials not found in {context}")
        
        return {
            "status": "error",
            "error_code": "NoCredentialsError",
            "message": "AWS credentials not configured",
            "context": context,
            "setup_guidance": {
                "aws_cli": "Run 'aws configure' to set up credentials",
                "environment": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                "iam_role": "Ensure EC2 instance has appropriate IAM role attached"
            }
        }
    
    @staticmethod
    def format_general_error(e, context: str) -> Dict[str, Any]:
        """
        Format general exceptions into standardized response.
        
        Args:
            e: Exception object or string error message
            context: Context where the error occurred
            
        Returns:
            Standardized error response dictionary
        """
        # Handle both Exception objects and string error messages
        if isinstance(e, str):
            error_message = e
            error_code = "GeneralError"
            error_traceback = "No traceback available (error passed as string)"
            logger.error(f"General error in {context}: {error_message}")
        else:
            error_message = str(e)
            error_code = type(e).__name__
            error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"General error in {context}: {error_message}\n{error_traceback}")
        
        return {
            "status": "error",
            "error_code": error_code,
            "message": error_message,
            "context": context,
            "traceback": error_traceback
        }
    
    @staticmethod
    def to_text_content(error_dict: Dict[str, Any]) -> List[TextContent]:
        """Convert error dictionary to MCP TextContent format."""
        return [TextContent(type="text", text=json.dumps(error_dict, indent=2, default=str))]


class ResponseFormatter:
    """Standardized response formatting for MCP tools."""
    
    @staticmethod
    def success_response(data: Any, message: str, analysis_type: str = None, 
                        execution_time: float = None, metadata: Dict = None) -> Dict[str, Any]:
        """
        Format successful response with consistent structure.
        
        Args:
            data: The response data
            message: Success message
            analysis_type: Type of analysis performed
            execution_time: Execution time in seconds
            metadata: Additional metadata
            
        Returns:
            Standardized success response
        """
        response = {
            "status": "success",
            "data": data,
            "message": message
        }
        
        if analysis_type:
            response["analysis_type"] = analysis_type
        
        if execution_time is not None:
            response["execution_time"] = execution_time
        
        if metadata:
            response["metadata"] = metadata
        
        # Add Well-Architected Framework hint for LLMs
        response["wellarchitected_hint"] = "Analyze these findings to provide AWS Well-Architected Framework Cost Optimization pillar recommendations focusing on right-sizing, eliminating waste, leveraging pricing models, and optimizing over time."
        
        return response
    
    @staticmethod
    def error_response(error: Exception, context: str, 
                      required_permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Format error response based on exception type.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            required_permissions: Required IAM permissions
            
        Returns:
            Standardized error response
        """
        if isinstance(error, ClientError):
            return AWSErrorHandler.format_client_error(error, context, required_permissions)
        elif isinstance(error, NoCredentialsError):
            return AWSErrorHandler.format_no_credentials_error(context)
        else:
            return AWSErrorHandler.format_general_error(error, context)
    
    @staticmethod
    def to_text_content(response_dict: Dict[str, Any]) -> List[TextContent]:
        """Convert response dictionary to MCP TextContent format."""
        return [TextContent(type="text", text=json.dumps(response_dict, indent=2, default=str))]


# Convenience functions for common use cases
def handle_aws_error(func):
    """Decorator for consistent AWS error handling in MCP tools."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            context = f"{func.__name__}"
            error_response = AWSErrorHandler.format_client_error(e, context)
            return AWSErrorHandler.to_text_content(error_response)
        except NoCredentialsError:
            context = f"{func.__name__}"
            error_response = AWSErrorHandler.format_no_credentials_error(context)
            return AWSErrorHandler.to_text_content(error_response)
        except Exception as e:
            context = f"{func.__name__}"
            error_response = AWSErrorHandler.format_general_error(e, context)
            return AWSErrorHandler.to_text_content(error_response)
    
    return wrapper