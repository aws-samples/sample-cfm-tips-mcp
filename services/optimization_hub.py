"""
AWS Cost Optimization Hub service module.

This module provides functions for interacting with the AWS Cost Optimization Hub API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_recommendations(
    resource_type: Optional[str] = None,
    region: Optional[str] = None,
    account_id: Optional[str] = None,
    client_region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get cost optimization recommendations from AWS Cost Optimization Hub.
    
    Args:
        resource_type: Resource type to analyze (e.g., EC2, RDS)
        region: AWS region to filter recommendations
        account_id: AWS account ID to filter recommendations
        client_region: Region for the boto3 client (optional)
        
    Returns:
        Dictionary containing the optimization recommendations
    """
    try:
        # Create Cost Optimization Hub client
        if client_region:
            client = boto3.client('cost-optimization-hub', region_name=client_region)
        else:
            client = boto3.client('cost-optimization-hub')
        
        # Prepare filters based on parameters
        filters = {}
        if resource_type:
            filters['resourceType'] = {'values': [resource_type]}
        if region:
            filters['region'] = {'values': [region]}
        if account_id:
            filters['accountId'] = {'values': [account_id]}
            
        # Make the API call
        if filters:
            response = client.get_recommendations(filters=filters)
        else:
            response = client.get_recommendations()
            
        # Extract recommendation count
        recommendation_count = len(response.get('recommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} cost optimization recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Cost Optimization Hub API: {str(e)}")
        return {
            "status": "error",
            "message": f"Cost Optimization Hub API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Cost Optimization Hub service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_recommendation_summary(
    client_region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get a summary of cost optimization recommendations.
    
    Args:
        client_region: Region for the boto3 client (optional)
        
    Returns:
        Dictionary containing the recommendation summary
    """
    try:
        # Create Cost Optimization Hub client
        if client_region:
            client = boto3.client('cost-optimization-hub', region_name=client_region)
        else:
            client = boto3.client('cost-optimization-hub')
            
        # Make the API call
        response = client.get_recommendation_summary()
            
        return {
            "status": "success",
            "data": response,
            "message": "Retrieved cost optimization recommendation summary"
        }
        
    except ClientError as e:
        logger.error(f"Error getting recommendation summary: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting recommendation summary: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error getting recommendation summary: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_savings_plans_recommendations(
    lookback_period: str = "SIXTY_DAYS",
    payment_option: str = "NO_UPFRONT",
    term: str = "ONE_YEAR",
    client_region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Savings Plans recommendations from AWS Cost Optimization Hub.
    
    Args:
        lookback_period: Historical data period to analyze
        payment_option: Payment option for Savings Plans
        term: Term length for Savings Plans
        client_region: Region for the boto3 client (optional)
        
    Returns:
        Dictionary containing the Savings Plans recommendations
    """
    try:
        # Create Cost Optimization Hub client
        if client_region:
            client = boto3.client('cost-optimization-hub', region_name=client_region)
        else:
            client = boto3.client('cost-optimization-hub')
            
        # Make the API call
        response = client.get_savings_plans_recommendations(
            lookbackPeriod=lookback_period,
            paymentOption=payment_option,
            term=term
        )
            
        return {
            "status": "success",
            "data": response,
            "message": "Retrieved Savings Plans recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error getting Savings Plans recommendations: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting Savings Plans recommendations: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error getting Savings Plans recommendations: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
