"""
TODO

AWS Cost and Usage Reports (CUR) service module.

This module provides functions for interacting with CUR reports in S3 buckets.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def list_cur_reports(
    bucket_name: str,
    prefix: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    List Cost and Usage Reports (CUR) in an S3 bucket.
    
    Args:
        bucket_name: S3 bucket name containing CUR reports
        prefix: S3 prefix for CUR reports (optional)
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the list of CUR reports
    """
    try:
        # Create S3 client
        if region:
            s3_client = boto3.client('s3', region_name=region)
        else:
            s3_client = boto3.client('s3')
        
        # Prepare parameters
        params = {
            'Bucket': bucket_name
        }
        if prefix:
            params['Prefix'] = prefix
            
        # List objects in the bucket
        response = s3_client.list_objects_v2(**params)
        
        # Extract report information
        reports = []
        if 'Contents' in response:
            for obj in response['Contents']:
                reports.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
                
        return {
            "status": "success",
            "data": {
                "reports": reports,
                "count": len(reports)
            },
            "message": f"Found {len(reports)} CUR reports in bucket {bucket_name}"
        }
        
    except ClientError as e:
        logger.error(f"Error listing CUR reports: {str(e)}")
        return {
            "status": "error",
            "message": f"Error listing CUR reports: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error listing CUR reports: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_cur_report_definition(
    report_name: str,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the definition of a Cost and Usage Report.
    
    Args:
        report_name: Name of the CUR report
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the CUR report definition
    """
    try:
        # Create CUR client
        if region:
            cur_client = boto3.client('cur', region_name=region)
        else:
            cur_client = boto3.client('cur')
            
        # Make the API call
        response = cur_client.describe_report_definitions(
            ReportNames=[report_name]
        )
        
        # Extract report information
        reports = response.get('ReportDefinitions', [])
        
        if not reports:
            return {
                "status": "error",
                "message": f"CUR report '{report_name}' not found"
            }
                
        return {
            "status": "success",
            "data": {
                "report": reports[0]
            },
            "message": f"Retrieved definition for CUR report '{report_name}'"
        }
        
    except ClientError as e:
        logger.error(f"Error getting CUR report definition: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting CUR report definition: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error getting CUR report definition: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def list_cur_report_definitions(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all Cost and Usage Report definitions.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the list of CUR report definitions
    """
    try:
        # Create CUR client
        if region:
            cur_client = boto3.client('cur', region_name=region)
        else:
            cur_client = boto3.client('cur')
            
        # Make the API call
        response = cur_client.describe_report_definitions()
        
        # Extract report information
        reports = response.get('ReportDefinitions', [])
                
        return {
            "status": "success",
            "data": {
                "reports": reports,
                "count": len(reports)
            },
            "message": f"Retrieved {len(reports)} CUR report definitions"
        }
        
    except ClientError as e:
        logger.error(f"Error listing CUR report definitions: {str(e)}")
        return {
            "status": "error",
            "message": f"Error listing CUR report definitions: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error listing CUR report definitions: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
