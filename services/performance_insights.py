"""
AWS Performance Insights service module.

This module provides functions for interacting with the AWS Performance Insights API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

from utils.error_handler import AWSErrorHandler, ResponseFormatter
from utils.aws_client_factory import get_performance_insights_client

logger = logging.getLogger(__name__)

def get_performance_insights_metrics(
    db_instance_identifier: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Performance Insights metrics for an RDS instance.
    
    Args:
        db_instance_identifier: RDS instance identifier
        start_time: Start time for metrics (ISO format)
        end_time: End time for metrics (ISO format)
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Performance Insights metrics
    """
    try:
        # Create Performance Insights client
        pi_client = get_performance_insights_client(region)
            
        # Set default time range if not provided
        if not start_time:
            end_datetime = datetime.utcnow()
            start_datetime = end_datetime - timedelta(hours=1)
            start_time = start_datetime.isoformat() + 'Z'
            end_time = end_datetime.isoformat() + 'Z'
        elif not end_time:
            end_time = datetime.utcnow().isoformat() + 'Z'
            
        # Define metrics to retrieve
        metrics = [
            {'Metric': 'db.load.avg'},
            {'Metric': 'db.sampledload.avg'}
        ]
            
        # Make the API call
        response = pi_client.get_resource_metrics(
            ServiceType='RDS',
            Identifier=db_instance_identifier,
            StartTime=start_time,
            EndTime=end_time,
            MetricQueries=metrics,
            PeriodInSeconds=60
        )
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved Performance Insights metrics for {db_instance_identifier}"
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        
        # Handle specific authorization errors gracefully
        if error_code in ['NotAuthorizedException', 'AccessDenied', 'UnauthorizedOperation']:
            logger.warning(f"Performance Insights not authorized for {db_instance_identifier}: {str(e)}")
            return {
                "status": "success",
                "data": {
                    "MetricList": [],
                    "AlignedStartTime": start_time,
                    "AlignedEndTime": end_time,
                    "Identifier": db_instance_identifier
                },
                "message": f"Performance Insights not enabled or authorized for {db_instance_identifier}",
                "warning": "Performance Insights requires explicit enablement and permissions"
            }
        else:
            logger.error(f"Error in Performance Insights API: {str(e)}")
            return {
                "status": "error",
                "message": f"Performance Insights API error: {str(e)}",
                "error_code": error_code
            }
        
    except Exception as e:
        logger.error(f"Unexpected error in Performance Insights service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }