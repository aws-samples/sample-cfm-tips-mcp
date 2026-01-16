"""
AWS Savings Plans service module.

This module provides functions for interacting with the AWS Savings Plans API
and Cost Explorer for Database Savings Plans analysis.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from utils.cache_decorator import dao_cache
from utils.aws_client_factory import AWSClientFactory

logger = logging.getLogger(__name__)


@dao_cache(ttl_seconds=3600)  # Cache for 1 hour - pricing data changes infrequently
def get_savings_plans_offerings(
    service_codes: List[str],
    payment_options: List[str],
    plan_types: List[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query available Savings Plans offerings.
    
    Args:
        service_codes: List of AWS service codes (e.g., ['AmazonRDS', 'AmazonDynamoDB'])
        payment_options: List of payment options (e.g., ['ALL_UPFRONT', 'PARTIAL_UPFRONT', 'NO_UPFRONT'])
        plan_types: List of plan types (default: ['DATABASE'])
        region: AWS region (optional, uses default if not specified)
        
    Returns:
        Dictionary containing the Savings Plans offerings response
        
    Example:
        >>> offerings = get_savings_plans_offerings(
        ...     service_codes=['AmazonRDS'],
        ...     payment_options=['PARTIAL_UPFRONT'],
        ...     plan_types=['DATABASE']
        ... )
    """
    try:
        # Set default plan types if not provided
        if plan_types is None:
            plan_types = ["DATABASE"]
            
        # Create Savings Plans client
        if region:
            sp_client = boto3.client('savingsplans', region_name=region)
        else:
            sp_client = boto3.client('savingsplans')
            
        logger.info(f"Querying Savings Plans offerings for services: {service_codes}")
        
        # Prepare filters
        filters = []
        
        # Add service code filters
        if service_codes:
            filters.append({
                'name': 'serviceCode',
                'values': service_codes
            })
        
        # Add payment option filters
        if payment_options:
            filters.append({
                'name': 'paymentOption',
                'values': payment_options
            })
        
        # Add plan type filters
        if plan_types:
            filters.append({
                'name': 'planType',
                'values': plan_types
            })
        
        # Query offerings
        offerings = []
        next_token = None
        
        while True:
            params = {
                'filters': filters,
                'maxResults': 100
            }
            
            if next_token:
                params['nextToken'] = next_token
                
            response = sp_client.describe_savings_plans_offerings(**params)
            offerings.extend(response.get('searchResults', []))
            
            next_token = response.get('nextToken')
            if not next_token:
                break
        
        logger.info(f"Retrieved {len(offerings)} Savings Plans offerings")
        
        return {
            "status": "success",
            "data": {
                "offerings": offerings,
                "count": len(offerings)
            },
            "message": f"Retrieved {len(offerings)} Savings Plans offerings"
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Error querying Savings Plans offerings: {error_code} - {error_message}")
        
        return {
            "status": "error",
            "error_code": error_code,
            "message": f"Savings Plans API error: {error_message}",
            "required_permissions": [
                "savingsplans:DescribeSavingsPlansOfferings"
            ],
            "guidance": "Ensure IAM role has Savings Plans read permissions"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error querying Savings Plans offerings: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@dao_cache(ttl_seconds=600)  # Cache for 10 minutes
def get_savings_plans_utilization(
    time_period: Dict[str, str],
    granularity: str = "MONTHLY",
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Savings Plans utilization metrics from Cost Explorer.
    
    Args:
        time_period: Dictionary with 'Start' and 'End' dates in YYYY-MM-DD format
        granularity: Time granularity (DAILY, MONTHLY, HOURLY)
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Savings Plans utilization response
        
    Example:
        >>> utilization = get_savings_plans_utilization(
        ...     time_period={'Start': '2024-01-01', 'End': '2024-01-31'},
        ...     granularity='MONTHLY'
        ... )
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
            
        logger.info(f"Querying Savings Plans utilization from {time_period['Start']} to {time_period['End']}")
        
        # Query utilization
        response = ce_client.get_savings_plans_utilization(
            TimePeriod=time_period,
            Granularity=granularity
        )
        
        logger.info("Successfully retrieved Savings Plans utilization data")
        
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved Savings Plans utilization from {time_period['Start']} to {time_period['End']}"
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Error querying Savings Plans utilization: {error_code} - {error_message}")
        
        return {
            "status": "error",
            "error_code": error_code,
            "message": f"Cost Explorer API error: {error_message}",
            "required_permissions": [
                "ce:GetSavingsPlansUtilization"
            ],
            "guidance": "Ensure IAM role has Cost Explorer read permissions"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error querying Savings Plans utilization: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@dao_cache(ttl_seconds=600)  # Cache for 10 minutes
def get_savings_plans_coverage(
    time_period: Dict[str, str],
    granularity: str = "MONTHLY",
    group_by: Optional[List[Dict[str, str]]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Savings Plans coverage metrics from Cost Explorer.
    
    Args:
        time_period: Dictionary with 'Start' and 'End' dates in YYYY-MM-DD format
        granularity: Time granularity (DAILY, MONTHLY, HOURLY)
        group_by: Optional grouping dimensions (e.g., [{'Type': 'DIMENSION', 'Key': 'SERVICE'}])
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Savings Plans coverage response
        
    Example:
        >>> coverage = get_savings_plans_coverage(
        ...     time_period={'Start': '2024-01-01', 'End': '2024-01-31'},
        ...     granularity='MONTHLY',
        ...     group_by=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        ... )
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
            
        logger.info(f"Querying Savings Plans coverage from {time_period['Start']} to {time_period['End']}")
        
        # Prepare parameters
        params = {
            'TimePeriod': time_period,
            'Granularity': granularity
        }
        
        if group_by:
            params['GroupBy'] = group_by
        
        # Query coverage
        response = ce_client.get_savings_plans_coverage(**params)
        
        logger.info("Successfully retrieved Savings Plans coverage data")
        
        return {
            "status": "success",
            "data": response,
            "message": f"Retrieved Savings Plans coverage from {time_period['Start']} to {time_period['End']}"
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Error querying Savings Plans coverage: {error_code} - {error_message}")
        
        return {
            "status": "error",
            "error_code": error_code,
            "message": f"Cost Explorer API error: {error_message}",
            "required_permissions": [
                "ce:GetSavingsPlansCoverage"
            ],
            "guidance": "Ensure IAM role has Cost Explorer read permissions"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error querying Savings Plans coverage: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


@dao_cache(ttl_seconds=300)  # Cache for 5 minutes - savings plans data changes less frequently
def get_existing_savings_plans(
    savings_plan_arns: Optional[List[str]] = None,
    savings_plan_ids: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get existing Savings Plans from the Savings Plans API.
    
    Args:
        savings_plan_arns: List of Savings Plan ARNs to filter by (optional)
        savings_plan_ids: List of Savings Plan IDs to filter by (optional)
        states: List of states to filter by (e.g., ['active', 'payment-pending'])
        region: AWS region (optional)
        
    Returns:
        Dictionary containing existing Savings Plans
        
    Example:
        >>> existing_plans = get_existing_savings_plans(
        ...     states=['active']
        ... )
    """
    try:
        # Create Savings Plans client
        if region:
            sp_client = boto3.client('savingsplans', region_name=region)
        else:
            sp_client = boto3.client('savingsplans')
            
        logger.info("Querying existing Savings Plans")
        
        # Prepare parameters
        params = {}
        
        if savings_plan_arns:
            params['savingsPlansArns'] = savings_plan_arns
            
        if savings_plan_ids:
            params['savingsPlansIds'] = savings_plan_ids
            
        if states:
            params['states'] = states
        
        # Query existing savings plans
        savings_plans = []
        next_token = None
        
        while True:
            if next_token:
                params['nextToken'] = next_token
                
            response = sp_client.describe_savings_plans(**params)
            savings_plans.extend(response.get('savingsPlans', []))
            
            next_token = response.get('nextToken')
            if not next_token:
                break
        
        logger.info(f"Retrieved {len(savings_plans)} existing Savings Plans")
        
        return {
            "status": "success",
            "data": {
                "savings_plans": savings_plans,
                "count": len(savings_plans)
            },
            "message": f"Retrieved {len(savings_plans)} existing Savings Plans"
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Error querying existing Savings Plans: {error_code} - {error_message}")
        
        return {
            "status": "error",
            "error_code": error_code,
            "message": f"Savings Plans API error: {error_message}",
            "required_permissions": [
                "savingsplans:DescribeSavingsPlans"
            ],
            "guidance": "Ensure IAM role has Savings Plans read permissions"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error querying existing Savings Plans: {str(e)}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }


def calculate_savings_plans_rates(
    on_demand_rate: float,
    commitment_term: str,
    payment_option: str,
    service_code: str = "AmazonRDS"
) -> Dict[str, Any]:
    """
    Calculate Savings Plans rates based on on-demand pricing.
    
    This function applies typical discount rates for Database Savings Plans.
    Actual rates should be retrieved from the Pricing API for production use.
    
    Args:
        on_demand_rate: On-demand hourly rate
        commitment_term: Commitment term ('1_YEAR' or '3_YEAR')
        payment_option: Payment option ('ALL_UPFRONT', 'PARTIAL_UPFRONT', 'NO_UPFRONT')
        service_code: AWS service code (default: 'AmazonRDS')
        
    Returns:
        Dictionary containing calculated rates and savings
        
    Example:
        >>> rates = calculate_savings_plans_rates(
        ...     on_demand_rate=1.50,
        ...     commitment_term='1_YEAR',
        ...     payment_option='PARTIAL_UPFRONT'
        ... )
    """
    try:
        logger.debug(f"Calculating Savings Plans rates for {service_code} - {commitment_term} - {payment_option}")
        
        # Database Savings Plans discount rates (Gen7+ instances only)
        # Note: Only No Upfront payment option is supported for 1-year terms
        # These are approximate and should be replaced with actual Pricing API data
        discount_rates = {
            "1_YEAR": {
                "NO_UPFRONT": 0.25        # 25% discount (only supported option)
            },
            "3_YEAR": {
                "ALL_UPFRONT": 0.42,      # 42% discount
                "PARTIAL_UPFRONT": 0.40,  # 40% discount
                "NO_UPFRONT": 0.37        # 37% discount
            }
        }
        
        # Validate inputs
        if commitment_term not in discount_rates:
            return {
                "status": "error",
                "error_code": "ValidationError",
                "message": f"Invalid commitment term: {commitment_term}",
                "valid_values": ["1_YEAR", "3_YEAR"]
            }
        
        if payment_option not in discount_rates[commitment_term]:
            valid_options = list(discount_rates[commitment_term].keys())
            return {
                "status": "error",
                "error_code": "ValidationError",
                "message": f"Invalid payment option '{payment_option}' for {commitment_term} term. Database Savings Plans only supports No Upfront for 1-year terms.",
                "valid_values": valid_options,
                "limitation": "Database Savings Plans 1-year terms only support No Upfront payment option"
            }
        
        if on_demand_rate <= 0:
            return {
                "status": "error",
                "error_code": "ValidationError",
                "message": "On-demand rate must be positive",
                "provided_value": on_demand_rate,
                "valid_range": "0.01 to 10000.00"
            }
        
        # Calculate discount
        discount_rate = discount_rates[commitment_term][payment_option]
        savings_plan_rate = on_demand_rate * (1 - discount_rate)
        hourly_savings = on_demand_rate - savings_plan_rate
        
        # Calculate annual costs
        hours_per_year = 8760
        annual_on_demand_cost = on_demand_rate * hours_per_year
        annual_savings_plan_cost = savings_plan_rate * hours_per_year
        annual_savings = hourly_savings * hours_per_year
        
        # Calculate upfront and recurring costs based on payment option
        if payment_option == "ALL_UPFRONT":
            upfront_cost = annual_savings_plan_cost
            recurring_hourly_cost = 0.0
        elif payment_option == "PARTIAL_UPFRONT":
            upfront_cost = annual_savings_plan_cost * 0.5
            recurring_hourly_cost = savings_plan_rate * 0.5
        else:  # NO_UPFRONT
            upfront_cost = 0.0
            recurring_hourly_cost = savings_plan_rate
        
        logger.debug(f"Calculated {discount_rate*100}% discount: ${savings_plan_rate:.4f}/hour")
        
        return {
            "status": "success",
            "data": {
                "on_demand_rate": round(on_demand_rate, 4),
                "savings_plan_rate": round(savings_plan_rate, 4),
                "hourly_savings": round(hourly_savings, 4),
                "discount_percentage": round(discount_rate * 100, 2),
                "commitment_term": commitment_term,
                "payment_option": payment_option,
                "annual_on_demand_cost": round(annual_on_demand_cost, 2),
                "annual_savings_plan_cost": round(annual_savings_plan_cost, 2),
                "annual_savings": round(annual_savings, 2),
                "upfront_cost": round(upfront_cost, 2),
                "recurring_hourly_cost": round(recurring_hourly_cost, 4),
                "service_code": service_code
            },
            "message": f"Calculated {discount_rate*100}% savings for {commitment_term} {payment_option}"
        }
        
    except Exception as e:
        logger.error(f"Error calculating Savings Plans rates: {str(e)}")
        return {
            "status": "error",
            "message": f"Calculation error: {str(e)}"
        }
