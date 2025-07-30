"""
AWS Compute Optimizer service module.

This module provides functions for interacting with the AWS Compute Optimizer API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def get_ec2_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get EC2 instance recommendations from AWS Compute Optimizer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the EC2 recommendations
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_ec2_instance_recommendations()
        
        # Extract recommendation count
        recommendation_count = len(response.get('instanceRecommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} EC2 instance recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Compute Optimizer EC2 API: {str(e)}")
        return {
            "status": "error",
            "message": f"Compute Optimizer EC2 API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Compute Optimizer EC2 service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_asg_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Auto Scaling Group recommendations from AWS Compute Optimizer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the ASG recommendations
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_auto_scaling_group_recommendations()
        
        # Extract recommendation count
        recommendation_count = len(response.get('autoScalingGroupRecommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} Auto Scaling Group recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Compute Optimizer ASG API: {str(e)}")
        return {
            "status": "error",
            "message": f"Compute Optimizer ASG API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Compute Optimizer ASG service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_ebs_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get EBS volume recommendations from AWS Compute Optimizer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the EBS recommendations
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_ebs_volume_recommendations()
        
        # Extract recommendation count
        recommendation_count = len(response.get('volumeRecommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} EBS volume recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Compute Optimizer EBS API: {str(e)}")
        return {
            "status": "error",
            "message": f"Compute Optimizer EBS API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Compute Optimizer EBS service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_lambda_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Lambda function recommendations from AWS Compute Optimizer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Lambda recommendations
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_lambda_function_recommendations()
        
        # Extract recommendation count
        recommendation_count = len(response.get('lambdaFunctionRecommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} Lambda function recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Compute Optimizer Lambda API: {str(e)}")
        return {
            "status": "error",
            "message": f"Compute Optimizer Lambda API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Compute Optimizer Lambda service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_ecs_recommendations(
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get ECS service recommendations from AWS Compute Optimizer.
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the ECS recommendations
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_ecs_service_recommendations()
        
        # Extract recommendation count
        recommendation_count = len(response.get('ecsServiceRecommendations', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {recommendation_count} ECS service recommendations"
        }
        
    except ClientError as e:
        logger.error(f"Error in Compute Optimizer ECS API: {str(e)}")
        return {
            "status": "error",
            "message": f"Compute Optimizer ECS API error: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in Compute Optimizer ECS service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_recommendation_summaries(
    resource_type: str = "Ec2Instance",
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommendation summaries from AWS Compute Optimizer.
    
    Args:
        resource_type: Type of resource (Ec2Instance, AutoScalingGroup, EbsVolume, LambdaFunction, EcsService)
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the recommendation summaries
    """
    try:
        # Create Compute Optimizer client
        if region:
            client = boto3.client('compute-optimizer', region_name=region)
        else:
            client = boto3.client('compute-optimizer')
            
        # Make the API call
        response = client.get_recommendation_summaries(
            resourceType=resource_type
        )
        
        # Extract summary count
        summary_count = len(response.get('recommendationSummaries', []))
            
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved {summary_count} recommendation summaries for {resource_type}"
        }
        
    except ClientError as e:
        logger.error(f"Error getting recommendation summaries: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting recommendation summaries: {str(e)}",
            "error_code": e.response['Error']['Code'] if 'Error' in e.response else "Unknown"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error getting recommendation summaries: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
