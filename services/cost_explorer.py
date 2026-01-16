"""
AWS Cost Explorer service module.

This module provides functions for interacting with the AWS Cost Explorer API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from utils.aws_client_factory import AWSClientFactory

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
            
        # Create Cost Explorer client using factory
        try:
            ce_client = AWSClientFactory.get_client('ce', region=region)
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            return {
                "status": "error",
                "message": "AWS credentials not configured. Please run 'aws configure' or set AWS environment variables.",
                "error_code": "NoCredentialsError",
                "required_action": "Configure AWS credentials"
            }
        except Exception as e:
            logger.error(f"Failed to create Cost Explorer client: {str(e)}")
            return {
                "status": "error", 
                "message": f"Failed to create Cost Explorer client: {str(e)}",
                "error_code": "ClientCreationError"
            }
            
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
        # Create Cost Explorer client using factory
        try:
            ce_client = AWSClientFactory.get_client('ce', region=region)
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            return {
                "status": "error",
                "message": "AWS credentials not configured. Please run 'aws configure' or set AWS environment variables.",
                "error_code": "NoCredentialsError",
                "required_action": "Configure AWS credentials"
            }
        except Exception as e:
            logger.error(f"Failed to create Cost Explorer client: {str(e)}")
            return {
                "status": "error", 
                "message": f"Failed to create Cost Explorer client: {str(e)}",
                "error_code": "ClientCreationError"
            }
            
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
        # Create Cost Explorer client using factory
        try:
            ce_client = AWSClientFactory.get_client('ce', region=region)
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            return {
                "status": "error",
                "message": "AWS credentials not configured. Please run 'aws configure' or set AWS environment variables.",
                "error_code": "NoCredentialsError",
                "required_action": "Configure AWS credentials"
            }
        except Exception as e:
            logger.error(f"Failed to create Cost Explorer client: {str(e)}")
            return {
                "status": "error", 
                "message": f"Failed to create Cost Explorer client: {str(e)}",
                "error_code": "ClientCreationError"
            }
            
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


def get_database_usage_by_service(
    start_date: str,
    end_date: str,
    services: Optional[List[str]] = None,
    region: Optional[str] = None,
    granularity: str = "DAILY",
    account_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve database usage data grouped by service, region, and instance family.
    
    This function queries Cost Explorer for on-demand database usage across
    supported AWS database services including Amazon Aurora, Amazon RDS,
    Amazon DynamoDB, Amazon ElastiCache, Amazon DocumentDB, Amazon Neptune,
    Amazon Keyspaces, Amazon Timestream, and AWS DMS.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        services: Optional list of database services to filter (e.g., ['rds', 'aurora', 'dynamodb'])
                 If None, retrieves all supported database services
        region: AWS region to filter by (optional)
        granularity: Time granularity (DAILY, MONTHLY, HOURLY)
        account_id: AWS account ID to filter by (optional, for multi-account analysis)
        
    Returns:
        Dictionary containing:
        {
            "status": "success" | "error",
            "data": {
                "results_by_time": [...],  # Cost Explorer time series data
                "group_definitions": [...],  # Grouping dimensions used
                "total_cost": float,  # Total cost across all services
                "service_breakdown": {  # Costs grouped by service
                    "service_name": float,
                    ...
                },
                "region_breakdown": {  # Costs grouped by region
                    "region_name": float,
                    ...
                },
                "instance_family_breakdown": {  # Costs grouped by instance family
                    "family_name": float,
                    ...
                }
            },
            "message": str
        }
        
    Example:
        >>> result = get_database_usage_by_service(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     services=["rds", "aurora"],
        ...     region="us-east-1"
        ... )
    """
    try:
        # Map of service names to AWS service codes
        service_map = {
            "rds": "Amazon Relational Database Service",
            "aurora": "Amazon Relational Database Service",  # Aurora is part of RDS
            "dynamodb": "Amazon DynamoDB",
            "elasticache": "Amazon ElastiCache",
            "documentdb": "Amazon DocumentDB",
            "neptune": "Amazon Neptune",
            "keyspaces": "Amazon Keyspaces",
            "timestream": "Amazon Timestream",
            "dms": "AWS Database Migration Service"
        }
        
        # Create Cost Explorer client using factory
        try:
            ce_client = AWSClientFactory.get_client('ce', region=region)
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            return {
                "status": "error",
                "message": "AWS credentials not configured. Please run 'aws configure' or set AWS environment variables.",
                "error_code": "NoCredentialsError",
                "required_action": "Configure AWS credentials"
            }
        except Exception as e:
            logger.error(f"Failed to create Cost Explorer client: {str(e)}")
            return {
                "status": "error", 
                "message": f"Failed to create Cost Explorer client: {str(e)}",
                "error_code": "ClientCreationError"
            }
        
        # Build service filter
        service_filter = None
        if services:
            # Convert service names to AWS service codes
            service_codes = []
            for service in services:
                service_lower = service.lower()
                if service_lower in service_map:
                    service_code = service_map[service_lower]
                    if service_code not in service_codes:
                        service_codes.append(service_code)
                else:
                    logger.warning(f"Unknown database service: {service}")
            
            if service_codes:
                if len(service_codes) == 1:
                    service_filter = {
                        "Dimensions": {
                            "Key": "SERVICE",
                            "Values": service_codes
                        }
                    }
                else:
                    service_filter = {
                        "Dimensions": {
                            "Key": "SERVICE",
                            "Values": service_codes
                        }
                    }
        else:
            # Default to all database services
            service_filter = {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": list(set(service_map.values()))
                }
            }
        
        # Build additional filters
        filters = []
        
        # Add service filter
        if service_filter:
            filters.append(service_filter)
        
        # Add region filter if specified
        if region:
            filters.append({
                "Dimensions": {
                    "Key": "REGION",
                    "Values": [region]
                }
            })
        
        # Add account filter if specified
        if account_id:
            filters.append({
                "Dimensions": {
                    "Key": "LINKED_ACCOUNT",
                    "Values": [account_id]
                }
            })
        
        # Combine filters
        if len(filters) > 1:
            filter_expr = {"And": filters}
        elif len(filters) == 1:
            filter_expr = filters[0]
        else:
            filter_expr = None
        
        # Define grouping dimensions (AWS allows max 2 GroupBy values)
        group_by = [
            {"Type": "DIMENSION", "Key": "SERVICE"},
            {"Type": "DIMENSION", "Key": "REGION"}
        ]
        
        # Prepare the request parameters
        params = {
            'TimePeriod': {
                'Start': start_date,
                'End': end_date
            },
            'Granularity': granularity,
            'Metrics': ['UnblendedCost'],
            'GroupBy': group_by
        }
        
        # Add filter if we have one
        if filter_expr:
            params['Filter'] = filter_expr
        
        logger.info(f"Querying database usage from {start_date} to {end_date}" + 
                    (f" for account {account_id}" if account_id else ""))
        
        # Make the API call
        response = ce_client.get_cost_and_usage(**params)
        
        # Process the response to create breakdowns
        service_breakdown = {}
        region_breakdown = {}
        instance_family_breakdown = {}
        total_cost = 0.0
        
        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                keys = group.get('Keys', [])
                cost = float(group.get('Metrics', {}).get('UnblendedCost', {}).get('Amount', 0))
                
                # Keys are in order: SERVICE, REGION (only 2 GroupBy allowed)
                if len(keys) >= 1:
                    service = keys[0]
                    service_breakdown[service] = service_breakdown.get(service, 0.0) + cost
                
                if len(keys) >= 2:
                    region_key = keys[1]
                    region_breakdown[region_key] = region_breakdown.get(region_key, 0.0) + cost
                
                total_cost += cost
        
        # Get instance family breakdown with a separate query
        # This is needed because AWS only allows 2 GroupBy dimensions
        instance_family_breakdown = _get_instance_family_breakdown(
            ce_client, start_date, end_date, filter_expr, granularity
        )
        
        return {
            "status": "success",
            "data": {
                "results_by_time": response.get('ResultsByTime', []),
                "group_definitions": response.get('GroupDefinitions', []),
                "total_cost": total_cost,
                "service_breakdown": service_breakdown,
                "region_breakdown": region_breakdown,
                "instance_family_breakdown": instance_family_breakdown,
                "dimension_value_attributes": response.get('DimensionValueAttributes', [])
            },
            "message": f"Retrieved database usage data from {start_date} to {end_date}"
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Cost Explorer API error: {error_code} - {error_message}")
        
        return {
            "status": "error",
            "message": f"Cost Explorer API error: {error_message}",
            "error_code": error_code,
            "required_permissions": [
                "ce:GetCostAndUsage"
            ] if error_code == "AccessDeniedException" else None
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in get_database_usage_by_service: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def _get_instance_family_breakdown(ce_client, start_date: str, end_date: str, base_filter: dict, granularity: str = "DAILY") -> dict:
    """
    Get instance family breakdown using a separate Cost Explorer query.
    
    This is needed because AWS Cost Explorer only allows 2 GroupBy dimensions,
    so we can't get SERVICE, REGION, and INSTANCE_TYPE_FAMILY in one call.
    """
    try:
        # Query with INSTANCE_TYPE_FAMILY grouping
        params = {
            'TimePeriod': {
                'Start': start_date,
                'End': end_date
            },
            'Granularity': granularity,
            'Metrics': ['UnblendedCost'],
            'GroupBy': [
                {"Type": "DIMENSION", "Key": "INSTANCE_TYPE_FAMILY"}
            ]
        }
        
        # Add the same filter as the main query
        if base_filter:
            params['Filter'] = base_filter
        
        logger.debug("Querying instance family breakdown")
        response = ce_client.get_cost_and_usage(**params)
        
        # Process the response
        instance_family_breakdown = {}
        
        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                keys = group.get('Keys', [])
                cost = float(group.get('Metrics', {}).get('UnblendedCost', {}).get('Amount', 0))
                
                if len(keys) >= 1 and keys[0]:  # Only add if instance family is not empty
                    instance_family = keys[0]
                    instance_family_breakdown[instance_family] = instance_family_breakdown.get(instance_family, 0.0) + cost
        
        logger.debug(f"Retrieved {len(instance_family_breakdown)} instance families")
        return instance_family_breakdown
        
    except Exception as e:
        logger.warning(f"Failed to get instance family breakdown: {str(e)}")
        return {}
