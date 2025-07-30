"""
AWS Pricing service module using AWS Price List API and MCP server.

This module provides functions for getting AWS pricing information.
"""

import logging
import boto3
import json
from typing import Dict, Optional, Any, List
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def _get_all_aws_regions() -> Dict[str, str]:
    """Get comprehensive mapping of AWS regions to location names."""
    return {
        'us-east-1': 'US East (N. Virginia)', 'us-east-2': 'US East (Ohio)',
        'us-west-1': 'US West (N. California)', 'us-west-2': 'US West (Oregon)',
        'eu-central-1': 'Europe (Frankfurt)', 'eu-west-1': 'Europe (Ireland)',
        'eu-west-2': 'Europe (London)', 'eu-west-3': 'Europe (Paris)',
        'eu-north-1': 'Europe (Stockholm)', 'eu-south-1': 'Europe (Milan)',
        'ap-northeast-1': 'Asia Pacific (Tokyo)', 'ap-northeast-2': 'Asia Pacific (Seoul)',
        'ap-northeast-3': 'Asia Pacific (Osaka)', 'ap-southeast-1': 'Asia Pacific (Singapore)',
        'ap-southeast-2': 'Asia Pacific (Sydney)', 'ap-southeast-3': 'Asia Pacific (Jakarta)',
        'ap-south-1': 'Asia Pacific (Mumbai)', 'ap-east-1': 'Asia Pacific (Hong Kong)',
        'ca-central-1': 'Canada (Central)', 'sa-east-1': 'South America (Sao Paulo)',
        'me-south-1': 'Middle East (Bahrain)', 'af-south-1': 'Africa (Cape Town)',
        'us-gov-east-1': 'AWS GovCloud (US-East)', 'us-gov-west-1': 'AWS GovCloud (US-West)',
        # Local Zones
        'us-east-1-bos-1a': 'US East (Boston)', 'us-east-1-chi-1a': 'US East (Chicago)',
        'us-east-1-dfw-1a': 'US East (Dallas)', 'us-east-1-iah-1a': 'US East (Houston)',
        'us-east-1-mci-1a': 'US East (Kansas City)', 'us-east-1-mia-1a': 'US East (Miami)',
        'us-east-1-msp-1a': 'US East (Minneapolis)', 'us-east-1-nyc-1a': 'US East (New York)',
        'us-east-1-phl-1a': 'US East (Philadelphia)', 'us-west-2-den-1a': 'US West (Denver)',
        'us-west-2-las-1a': 'US West (Las Vegas)', 'us-west-2-lax-1a': 'US West (Los Angeles)',
        'us-west-2-phx-1a': 'US West (Phoenix)', 'us-west-2-pdx-1a': 'US West (Portland)',
        'us-west-2-sea-1a': 'US West (Seattle)', 'eu-west-1-lhr-1a': 'Europe (London)',
        'ap-northeast-1-nrt-1a': 'Asia Pacific (Tokyo)', 'ap-southeast-1-sin-1a': 'Asia Pacific (Singapore)',
        # Wavelength Zones
        'us-east-1-wl1-bos-wlz-1': 'US East (Boston Wavelength)', 'us-east-1-wl1-chi-wlz-1': 'US East (Chicago Wavelength)',
        'us-east-1-wl1-dfw-wlz-1': 'US East (Dallas Wavelength)', 'us-east-1-wl1-mia-wlz-1': 'US East (Miami Wavelength)',
        'us-east-1-wl1-nyc-wlz-1': 'US East (New York Wavelength)', 'us-west-2-wl1-den-wlz-1': 'US West (Denver Wavelength)',
        'us-west-2-wl1-las-wlz-1': 'US West (Las Vegas Wavelength)', 'us-west-2-wl1-lax-wlz-1': 'US West (Los Angeles Wavelength)',
        'us-west-2-wl1-phx-wlz-1': 'US West (Phoenix Wavelength)', 'us-west-2-wl1-sea-wlz-1': 'US West (Seattle Wavelength)',
        'eu-west-1-wl1-lhr-wlz-1': 'Europe (London Wavelength)', 'ap-northeast-1-wl1-nrt-wlz-1': 'Asia Pacific (Tokyo Wavelength)',
        'ap-southeast-1-wl1-sin-wlz-1': 'Asia Pacific (Singapore Wavelength)', 'ap-southeast-2-wl1-syd-wlz-1': 'Asia Pacific (Sydney Wavelength)'
    }

def get_ec2_pricing(
    instance_type: str,
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """Get EC2 instance pricing from AWS Price List API."""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        region_map = _get_all_aws_regions()
        location = region_map.get(region, 'US East (N. Virginia)')
        
        response = pricing_client.get_products(
            ServiceCode='AmazonEC2',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
            ]
        )
        
        if response['PriceList']:
            price_data = json.loads(response['PriceList'][0])
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dimensions = terms[term_key]['priceDimensions']
            dimension_key = list(price_dimensions.keys())[0]
            hourly_price = float(price_dimensions[dimension_key]['pricePerUnit']['USD'])
            
            return {
                'status': 'success',
                'instance_type': instance_type,
                'region': region,
                'hourly_price': hourly_price,
                'monthly_price': hourly_price * 730,
                'source': 'aws_price_list_api'
            }
        
        return {'status': 'error', 'message': 'No pricing found', 'hourly_price': 0.1}
        
    except Exception as e:
        logger.error(f"Error getting EC2 pricing: {str(e)}")
        return {'status': 'error', 'message': str(e), 'hourly_price': 0.1}

def get_ebs_pricing(
    volume_type: str,
    volume_size: int,
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """Get EBS volume pricing from AWS Price List API."""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        region_map = _get_all_aws_regions()
        location = region_map.get(region, 'US East (N. Virginia)')
        
        response = pricing_client.get_products(
            ServiceCode='AmazonEC2',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type.upper()},
                {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location}
            ]
        )
        
        if response['PriceList']:
            price_data = json.loads(response['PriceList'][0])
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dimensions = terms[term_key]['priceDimensions']
            dimension_key = list(price_dimensions.keys())[0]
            gb_price = float(price_dimensions[dimension_key]['pricePerUnit']['USD'])
            
            monthly_price = gb_price * volume_size
            hourly_price = monthly_price / 730
            
            return {
                'status': 'success',
                'volume_type': volume_type,
                'volume_size': volume_size,
                'region': region,
                'hourly_price': hourly_price,
                'monthly_price': monthly_price,
                'price_per_gb_month': gb_price,
                'source': 'aws_price_list_api'
            }
        
        return {'status': 'error', 'message': 'No pricing found', 'hourly_price': 0.01}
        
    except Exception as e:
        logger.error(f"Error getting EBS pricing: {str(e)}")
        return {'status': 'error', 'message': str(e), 'hourly_price': 0.01}

def get_rds_pricing(
    instance_class: str,
    engine: str = 'mysql',
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """Get RDS instance pricing from AWS Price List API."""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        region_map = _get_all_aws_regions()
        location = region_map.get(region, 'US East (N. Virginia)')
        
        response = pricing_client.get_products(
            ServiceCode='AmazonRDS',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_class},
                {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine.title()},
                {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
            ]
        )
        
        if response['PriceList']:
            price_data = json.loads(response['PriceList'][0])
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dimensions = terms[term_key]['priceDimensions']
            dimension_key = list(price_dimensions.keys())[0]
            hourly_price = float(price_dimensions[dimension_key]['pricePerUnit']['USD'])
            
            return {
                'status': 'success',
                'instance_class': instance_class,
                'engine': engine,
                'region': region,
                'hourly_price': hourly_price,
                'monthly_price': hourly_price * 730,
                'source': 'aws_price_list_api'
            }
        
        return {'status': 'error', 'message': 'No pricing found', 'hourly_price': 0.1}
        
    except Exception as e:
        logger.error(f"Error getting RDS pricing: {str(e)}")
        return {'status': 'error', 'message': str(e), 'hourly_price': 0.1}

def get_lambda_pricing(
    memory_size: int,
    region: str = 'us-east-1'
) -> Dict[str, Any]:
    """Get Lambda function pricing from AWS Price List API."""
    try:
        pricing_client = boto3.client('pricing', region_name='us-east-1')
        region_map = _get_all_aws_regions()
        location = region_map.get(region, 'US East (N. Virginia)')
        
        response = pricing_client.get_products(
            ServiceCode='AWSLambda',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location}
            ]
        )
        
        if response['PriceList']:
            gb_seconds_price = 0.0000166667
            requests_price = 0.0000002
            memory_gb = memory_size / 1024
            
            return {
                'status': 'success',
                'memory_size_mb': memory_size,
                'memory_size_gb': memory_gb,
                'region': region,
                'price_per_gb_second': gb_seconds_price,
                'price_per_request': requests_price,
                'source': 'aws_price_list_api'
            }
        
        return {'status': 'error', 'message': 'No pricing found', 'price_per_gb_second': 0.0000166667}
        
    except Exception as e:
        logger.error(f"Error getting Lambda pricing: {str(e)}")
        return {'status': 'error', 'message': str(e), 'price_per_gb_second': 0.0000166667}

def get_all_regions() -> List[str]:
    """Get list of all supported AWS regions."""
    return list(_get_all_aws_regions().keys())

def get_local_zones() -> List[str]:
    """Get list of AWS Local Zones."""
    return [region for region in _get_all_aws_regions().keys() if '-1a' in region]

def get_wavelength_zones() -> List[str]:
    """Get list of AWS Wavelength Zones."""
    return [region for region in _get_all_aws_regions().keys() if '-wlz-' in region]

def get_standard_regions() -> List[str]:
    """Get list of standard AWS regions (excluding Local and Wavelength Zones)."""
    return [region for region in _get_all_aws_regions().keys() if '-1a' not in region and '-wlz-' not in region]

def is_local_zone(region: str) -> bool:
    """Check if region is a Local Zone."""
    return '-1a' in region

def is_wavelength_zone(region: str) -> bool:
    """Check if region is a Wavelength Zone."""
    return '-wlz-' in region

def get_zone_type(region: str) -> str:
    """Get zone type: standard, local, or wavelength."""
    if is_wavelength_zone(region):
        return 'wavelength'
    elif is_local_zone(region):
        return 'local'
    else:
        return 'standard'

def get_pricing_for_all_regions(service_function, *args, **kwargs) -> Dict[str, Any]:
    """Get pricing across all AWS regions, Local Zones, and Wavelength Zones."""
    results = {}
    errors = {}
    
    for region in get_all_regions():
        try:
            result = service_function(*args, region=region, **kwargs)
            if result.get('status') == 'success':
                results[region] = result
            else:
                # Store error results separately
                errors[region] = {
                    'error_message': result.get('message', 'Unknown error'),
                    'region': region
                }
                logger.warning(f"Failed to get pricing for region {region}: {result.get('message', 'Unknown error')}")
        except Exception as e:
            # Log and track exceptions
            error_message = str(e)
            errors[region] = {
                'error_message': error_message,
                'region': region,
                'exception_type': type(e).__name__
            }
            logger.warning(f"Exception while getting pricing for region {region}: {error_message}")
    
    # Calculate success rate
    total_regions = len(get_all_regions())
    success_count = len(results)
    success_rate = (success_count / total_regions) * 100 if total_regions > 0 else 0
    
    return {
        'status': 'success',
        'total_regions': total_regions,
        'regions_with_pricing': success_count,
        'success_rate': f"{success_rate:.1f}%",
        'standard_regions_analyzed': len([r for r in results.keys() if get_zone_type(r) == 'standard']),
        'local_zones_analyzed': len([r for r in results.keys() if get_zone_type(r) == 'local']),
        'wavelength_zones_analyzed': len([r for r in results.keys() if get_zone_type(r) == 'wavelength']),
        'pricing_by_region': results,
        'errors_by_region': errors if errors else None
    }