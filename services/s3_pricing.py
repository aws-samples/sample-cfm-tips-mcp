"""
AWS S3 Pricing service module using AWS Price List API.

This module provides functions for getting S3 pricing information including
storage classes, request costs, data transfer, and lifecycle savings calculations.
"""

import logging
import boto3
import json
from typing import Dict, Optional, Any, List
from botocore.exceptions import ClientError
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class S3Pricing:
    """S3-specific pricing calculations and cost modeling."""
    
    def __init__(self, region: str = 'us-east-1'):
        """Initialize S3 pricing service.
        
        Args:
            region: AWS region for pricing calculations
        """
        self.region = region
        self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        self._region_map = self._get_all_aws_regions()
        self.location = self._region_map.get(region, 'US East (N. Virginia)')
        
        # S3 storage class mappings for pricing API
        self._storage_class_map = {
            'STANDARD': 'General Purpose',
            'STANDARD_IA': 'Infrequent Access',
            'ONEZONE_IA': 'One Zone - Infrequent Access',
            'REDUCED_REDUNDANCY': 'Reduced Redundancy',
            'GLACIER': 'Amazon Glacier',
            'GLACIER_IR': 'Glacier Instant Retrieval',
            'DEEP_ARCHIVE': 'Glacier Deep Archive'
        }
        
        # Request type mappings
        self._request_type_map = {
            'PUT': 'PUT, COPY, POST, LIST requests',
            'GET': 'GET, SELECT, and all other requests',
            'LIST': 'PUT, COPY, POST, LIST requests',
            'DELETE': 'DELETE requests',
            'LIFECYCLE': 'Lifecycle Transition requests'
        }

    def _get_all_aws_regions(self) -> Dict[str, str]:
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
            'us-gov-east-1': 'AWS GovCloud (US-East)', 'us-gov-west-1': 'AWS GovCloud (US-West)'
        }

    def get_storage_class_pricing(self, storage_class: str) -> Dict[str, Any]:
        """Get pricing for a specific S3 storage class.
        
        Args:
            storage_class: S3 storage class (STANDARD, STANDARD_IA, etc.)
            
        Returns:
            Dict containing pricing information for the storage class
        """
        try:
            # Map storage class to pricing API format
            pricing_storage_class = self._storage_class_map.get(storage_class, storage_class)
            
            response = self.pricing_client.get_products(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self.location},
                    {'Type': 'TERM_MATCH', 'Field': 'storageClass', 'Value': pricing_storage_class},
                    {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': 'Standard'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                terms = price_data['terms']['OnDemand']
                term_key = list(terms.keys())[0]
                price_dimensions = terms[term_key]['priceDimensions']
                dimension_key = list(price_dimensions.keys())[0]
                
                price_per_gb = float(price_dimensions[dimension_key]['pricePerUnit']['USD'])
                
                return {
                    'status': 'success',
                    'storage_class': storage_class,
                    'region': self.region,
                    'location': self.location,
                    'price_per_gb_month': price_per_gb,
                    'price_per_tb_month': price_per_gb * 1024,
                    'unit': price_dimensions[dimension_key]['unit'],
                    'description': price_dimensions[dimension_key]['description'],
                    'source': 'aws_price_list_api'
                }
            
            # Fallback pricing if API doesn't return data
            fallback_prices = {
                'STANDARD': 0.023,
                'STANDARD_IA': 0.0125,
                'ONEZONE_IA': 0.01,
                'GLACIER': 0.004,
                'GLACIER_IR': 0.004,
                'DEEP_ARCHIVE': 0.00099
            }
            
            fallback_price = fallback_prices.get(storage_class, 0.023)
            
            return {
                'status': 'fallback',
                'storage_class': storage_class,
                'region': self.region,
                'price_per_gb_month': fallback_price,
                'price_per_tb_month': fallback_price * 1024,
                'message': 'Using fallback pricing - API returned no results',
                'source': 'fallback_pricing'
            }
            
        except Exception as e:
            logger.error(f"Error getting S3 storage class pricing for {storage_class}: {str(e)}")
            return {
                'status': 'error',
                'storage_class': storage_class,
                'region': self.region,
                'message': str(e),
                'price_per_gb_month': 0.023,  # Default to Standard pricing
                'source': 'error_fallback'
            }

    def calculate_lifecycle_savings(self, current_class: str, target_class: str, size_gb: float) -> Dict[str, Any]:
        """Calculate savings from transitioning between storage classes.
        
        Args:
            current_class: Current S3 storage class
            target_class: Target S3 storage class
            size_gb: Size in GB to transition
            
        Returns:
            Dict containing savings calculations
        """
        try:
            current_pricing = self.get_storage_class_pricing(current_class)
            target_pricing = self.get_storage_class_pricing(target_class)
            
            if current_pricing['status'] == 'error' or target_pricing['status'] == 'error':
                return {
                    'status': 'error',
                    'message': 'Failed to get pricing for one or both storage classes',
                    'monthly_savings': 0,
                    'annual_savings': 0
                }
            
            current_monthly_cost = current_pricing['price_per_gb_month'] * size_gb
            target_monthly_cost = target_pricing['price_per_gb_month'] * size_gb
            
            monthly_savings = current_monthly_cost - target_monthly_cost
            annual_savings = monthly_savings * 12
            
            # Calculate percentage savings
            savings_percentage = (monthly_savings / current_monthly_cost * 100) if current_monthly_cost > 0 else 0
            
            return {
                'status': 'success',
                'current_class': current_class,
                'target_class': target_class,
                'size_gb': size_gb,
                'current_monthly_cost': round(current_monthly_cost, 4),
                'target_monthly_cost': round(target_monthly_cost, 4),
                'monthly_savings': round(monthly_savings, 4),
                'annual_savings': round(annual_savings, 2),
                'savings_percentage': round(savings_percentage, 2),
                'region': self.region,
                'current_price_per_gb': current_pricing['price_per_gb_month'],
                'target_price_per_gb': target_pricing['price_per_gb_month']
            }
            
        except Exception as e:
            logger.error(f"Error calculating lifecycle savings: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'monthly_savings': 0,
                'annual_savings': 0
            }

    def estimate_request_costs(self, request_counts: Dict[str, int]) -> Dict[str, Any]:
        """Estimate costs for S3 API requests.
        
        Args:
            request_counts: Dict with request types and counts
                          e.g., {'GET': 10000, 'PUT': 1000, 'LIST': 100, 'DELETE': 50}
            
        Returns:
            Dict containing request cost estimates
        """
        try:
            # Get request pricing from API
            response = self.pricing_client.get_products(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self.location},
                    {'Type': 'TERM_MATCH', 'Field': 'group', 'Value': 'S3-API-Tier1'}
                ]
            )
            
            # Fallback pricing per 1000 requests (typical S3 pricing)
            request_pricing = {
                'PUT': 0.005,    # PUT, COPY, POST, LIST per 1000 requests
                'GET': 0.0004,   # GET, SELECT per 1000 requests  
                'LIST': 0.005,   # Same as PUT
                'DELETE': 0.0,   # DELETE requests are free
                'LIFECYCLE': 0.01 # Lifecycle transition requests per 1000
            }
            
            # If API returns pricing, try to parse it
            if response['PriceList']:
                try:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data['terms']['OnDemand']
                    term_key = list(terms.keys())[0]
                    price_dimensions = terms[term_key]['priceDimensions']
                    dimension_key = list(price_dimensions.keys())[0]
                    
                    # Update pricing from API if available
                    api_price = float(price_dimensions[dimension_key]['pricePerUnit']['USD'])
                    if 'PUT' in price_dimensions[dimension_key]['description']:
                        request_pricing['PUT'] = api_price
                        request_pricing['LIST'] = api_price
                except Exception as parse_error:
                    logger.warning(f"Could not parse API pricing, using fallback: {parse_error}")
            
            total_cost = 0
            cost_breakdown = {}
            
            for request_type, count in request_counts.items():
                if request_type in request_pricing:
                    # Calculate cost per 1000 requests
                    cost_per_thousand = request_pricing[request_type]
                    request_cost = (count / 1000) * cost_per_thousand
                    cost_breakdown[request_type] = {
                        'count': count,
                        'cost_per_thousand': cost_per_thousand,
                        'total_cost': round(request_cost, 6)
                    }
                    total_cost += request_cost
                else:
                    logger.warning(f"Unknown request type: {request_type}")
                    cost_breakdown[request_type] = {
                        'count': count,
                        'cost_per_thousand': 0,
                        'total_cost': 0,
                        'note': 'Unknown request type'
                    }
            
            return {
                'status': 'success',
                'region': self.region,
                'total_monthly_cost': round(total_cost, 4),
                'total_annual_cost': round(total_cost * 12, 2),
                'cost_breakdown': cost_breakdown,
                'pricing_source': 'aws_price_list_api' if response['PriceList'] else 'fallback_pricing'
            }
            
        except Exception as e:
            logger.error(f"Error estimating request costs: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'total_monthly_cost': 0,
                'total_annual_cost': 0
            }

    def get_data_transfer_pricing(self) -> Dict[str, Any]:
        """Get S3 data transfer pricing for cross-region transfers.
        
        Returns:
            Dict containing data transfer pricing information
        """
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonS3',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self.location},
                    {'Type': 'TERM_MATCH', 'Field': 'transferType', 'Value': 'AWS Outbound'}
                ]
            )
            
            # Fallback pricing structure (typical AWS data transfer pricing)
            transfer_pricing = {
                'data_transfer_out_internet': {
                    'first_1gb': 0.0,      # First 1 GB free
                    'next_9999gb': 0.09,   # Next 9.999 TB
                    'next_40tb': 0.085,    # Next 40 TB
                    'next_100tb': 0.07,    # Next 100 TB
                    'over_150tb': 0.05     # Over 150 TB
                },
                'data_transfer_out_cloudfront': 0.0,  # Free to CloudFront
                'data_transfer_cross_region': 0.02,   # Cross-region transfer
                'data_transfer_same_region': 0.0      # Same region transfer (free)
            }
            
            # Try to get actual pricing from API
            if response['PriceList']:
                try:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data['terms']['OnDemand']
                    term_key = list(terms.keys())[0]
                    price_dimensions = terms[term_key]['priceDimensions']
                    
                    # Parse tiered pricing if available
                    for dimension_key, dimension in price_dimensions.items():
                        price_per_gb = float(dimension['pricePerUnit']['USD'])
                        description = dimension.get('description', '').lower()
                        
                        if 'first' in description and '1 gb' in description:
                            transfer_pricing['data_transfer_out_internet']['first_1gb'] = price_per_gb
                        elif 'next' in description and '10 tb' in description:
                            transfer_pricing['data_transfer_out_internet']['next_9999gb'] = price_per_gb
                            
                except Exception as parse_error:
                    logger.warning(f"Could not parse transfer pricing, using fallback: {parse_error}")
            
            return {
                'status': 'success',
                'region': self.region,
                'location': self.location,
                'transfer_pricing': transfer_pricing,
                'notes': {
                    'data_transfer_in': 'Data transfer IN to S3 is free from internet',
                    'cloudfront_integration': 'Data transfer to CloudFront is free',
                    'same_az_transfer': 'Data transfer within same AZ is free',
                    'cross_az_transfer': 'Cross-AZ transfer may incur charges'
                },
                'source': 'aws_price_list_api' if response['PriceList'] else 'fallback_pricing'
            }
            
        except Exception as e:
            logger.error(f"Error getting data transfer pricing: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'transfer_pricing': {}
            }

    def calculate_multipart_waste_cost(self, incomplete_uploads: List[Dict]) -> Dict[str, Any]:
        """Calculate cost of incomplete multipart uploads.
        
        Args:
            incomplete_uploads: List of incomplete upload objects with size info
            
        Returns:
            Dict containing waste cost calculations
        """
        try:
            total_waste_gb = 0
            upload_count = len(incomplete_uploads)
            
            for upload in incomplete_uploads:
                # Get size from upload object (assuming it has 'size' or 'parts_size')
                size_bytes = upload.get('size', upload.get('parts_size', 0))
                size_gb = size_bytes / (1024 ** 3) if size_bytes > 0 else 0
                total_waste_gb += size_gb
            
            # Get Standard storage pricing for waste calculation
            standard_pricing = self.get_storage_class_pricing('STANDARD')
            
            if standard_pricing['status'] == 'error':
                return {
                    'status': 'error',
                    'message': 'Could not get Standard storage pricing',
                    'monthly_waste_cost': 0
                }
            
            price_per_gb = standard_pricing['price_per_gb_month']
            monthly_waste_cost = total_waste_gb * price_per_gb
            annual_waste_cost = monthly_waste_cost * 12
            
            return {
                'status': 'success',
                'region': self.region,
                'incomplete_upload_count': upload_count,
                'total_waste_gb': round(total_waste_gb, 4),
                'price_per_gb_month': price_per_gb,
                'monthly_waste_cost': round(monthly_waste_cost, 4),
                'annual_waste_cost': round(annual_waste_cost, 2),
                'potential_savings': {
                    'immediate': round(monthly_waste_cost, 4),
                    'annual': round(annual_waste_cost, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating multipart waste cost: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'monthly_waste_cost': 0,
                'annual_waste_cost': 0
            }

    def get_all_storage_class_pricing(self) -> Dict[str, Any]:
        """Get pricing for all S3 storage classes.
        
        Returns:
            Dict containing pricing for all storage classes
        """
        try:
            all_pricing = {}
            
            for storage_class in self._storage_class_map.keys():
                pricing = self.get_storage_class_pricing(storage_class)
                all_pricing[storage_class] = pricing
            
            return {
                'status': 'success',
                'region': self.region,
                'location': self.location,
                'storage_class_pricing': all_pricing,
                'comparison': self._create_pricing_comparison(all_pricing)
            }
            
        except Exception as e:
            logger.error(f"Error getting all storage class pricing: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'storage_class_pricing': {}
            }

    def _create_pricing_comparison(self, all_pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comparison table of storage class pricing.
        
        Args:
            all_pricing: Dict of all storage class pricing
            
        Returns:
            Dict with pricing comparison data
        """
        try:
            comparison = []
            
            for storage_class, pricing_data in all_pricing.items():
                if pricing_data.get('status') in ['success', 'fallback']:
                    comparison.append({
                        'storage_class': storage_class,
                        'price_per_gb_month': pricing_data['price_per_gb_month'],
                        'price_per_tb_month': pricing_data['price_per_tb_month'],
                        'status': pricing_data['status']
                    })
            
            # Sort by price (cheapest first)
            comparison.sort(key=lambda x: x['price_per_gb_month'])
            
            return {
                'sorted_by_price': comparison,
                'cheapest': comparison[0] if comparison else None,
                'most_expensive': comparison[-1] if comparison else None
            }
            
        except Exception as e:
            logger.error(f"Error creating pricing comparison: {str(e)}")
            return {'error': str(e)}