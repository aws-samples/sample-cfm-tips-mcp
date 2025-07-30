"""
AWS Cost Explorer service module.

This module provides functions for interacting with the AWS Cost Explorer API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_cost_and_usage(
    start_date: str,
    end_date: str,
    granularity: str = "MONTHLY",
    metrics: List[str] = None,
    group_by: Optional[List[Dict[str, str]]] = None,
    filter_expr: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve cost and usage data from AWS Cost Explorer.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        granularity: Time granularity (DAILY, MONTHLY, HOURLY)
        metrics: List of cost metrics to retrieve
        group_by: Optional grouping dimensions
        filter_expr: Optional filters
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Cost Explorer API response
    """
    try:
        # Set default metrics if not provided
        if metrics is None:
            metrics = ["BlendedCost", "UnblendedCost"]
            
        # Create Cost Explorer client
        if region:
            ce_client = boto3.client('ce', region_name=region)
        else:
            ce_client = boto3.client('ce')
            
        # Prepare the request parameters
        params = {
            'TimePeriod': {
                'Start': start_date,
                'End': end_date
            },
            'Granularity': granularity,
            'Metrics': metrics
        }
        
        # Add optional parameters if provided
        if group_by:
            params['GroupBy'] = group_by
            
        if filter_expr:
            params['Filter'] = filter_expr
            
        # Make the API call
        response = ce_client.get_cost_and_usage(**params)
        
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved cost data from {start_date} to {end_date}"
        }
        
    except ClientError as e:
        logger.error(f"Error in Cost Explorer API: {str(e)}")
        return {
            "status": "error",
            "message": f"Cost Explorer API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Cost Explorer service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_cost_forecast(
    start_date: str,
    end_date: str,
    granularity: str = "MONTHLY",
    metric: str = "BLENDED_COST",
    filter_expr: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a cost forecast from AWS Cost Explorer.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        granularity: Time granularity (DAILY, MONTHLY)
        metric: Cost metric to forecast
        filter_expr: Optional filters
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Cost Explorer forecast response
    """
    try:
        # Create Cost Explorer client
        if region:
            ce_client = boto3.client('ce', region_name=region)
        else:
            ce_client = boto3.client('ce')
            
        # Prepare the request parameters
        params = {
            'TimePeriod': {
                'Start': start_date,
                'End': end_date
            },
            'Granularity': granularity,
            'Metric': metric
        }
        
        # Add optional filter if provided
        if filter_expr:
            params['Filter'] = filter_expr
            
        # Make the API call
        response = ce_client.get_cost_forecast(**params)
        
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved cost forecast from {start_date} to {end_date}"
        }
        
    except ClientError as e:
        logger.error(f"Error in Cost Explorer forecast API: {str(e)}")
        return {
            "status": "error",
            "message": f"Cost Explorer forecast API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Cost Explorer forecast service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_cost_categories(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    List cost categories from AWS Cost Explorer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the cost categories
    """
    try:
        # Create Cost Explorer client
        if region:
            ce_client = boto3.client('ce', region_name=region)
        else:
            ce_client = boto3.client('ce')
            
        # Make the API call
        response = ce_client.list_cost_category_definitions()
        
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {len(response.get('CostCategoryReferences', []))} cost categories"
        }
        
    except ClientError as e:
        logger.error(f"Error listing cost categories: {str(e)}")
        return {
            "status": "error",
            "message": f"Error listing cost categories: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error listing cost categories: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
