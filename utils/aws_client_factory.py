"""
AWS Client Factory for Centralized Client Management

Provides consistent AWS client creation with proper error handling and configuration.
"""

import boto3
import logging
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

logger = logging.getLogger(__name__)


class AWSClientFactory:
    """Centralized AWS client creation and management."""
    
    # Default configurations for different services
    DEFAULT_CONFIGS = {
        'cost-optimization-hub': {
            'region_name': 'us-east-1',  # COH is only available in us-east-1
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=60
            )
        },
        'ce': {  # Cost Explorer
            'region_name': 'us-east-1',  # CE is only available in us-east-1
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=60
            )
        },
        'support': {  # Trusted Advisor
            'region_name': 'us-east-1',  # Support API only in us-east-1
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=60
            )
        },
        'compute-optimizer': {
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=60
            )
        },
        'pi': {  # Performance Insights
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=30
            )
        },
        'default': {
            'config': Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=30
            )
        }
    }
    
    _clients: Dict[str, Any] = {}  # Client cache
    
    @classmethod
    def get_client(cls, service_name: str, region: Optional[str] = None, 
                   force_new: bool = False) -> boto3.client:
        """
        Get AWS client with proper configuration and caching.
        
        Args:
            service_name: AWS service name (e.g., 'ec2', 's3', 'cost-optimization-hub')
            region: AWS region (optional, uses service defaults or session default)
            force_new: Force creation of new client instead of using cache
            
        Returns:
            Configured boto3 client
            
        Raises:
            NoCredentialsError: If AWS credentials are not configured
            ClientError: If client creation fails
        """
        # Create cache key
        cache_key = f"{service_name}:{region or 'default'}"
        
        # Return cached client if available and not forcing new
        if not force_new and cache_key in cls._clients:
            logger.debug(f"Using cached client for {service_name}")
            return cls._clients[cache_key]
        
        try:
            # Get service configuration
            service_config = cls.DEFAULT_CONFIGS.get(service_name, cls.DEFAULT_CONFIGS['default'])
            
            # Prepare client arguments
            client_args = {
                'service_name': service_name,
                'config': service_config.get('config')
            }
            
            # Set region - priority: parameter > service default > session default
            if region:
                client_args['region_name'] = region
            elif 'region_name' in service_config:
                client_args['region_name'] = service_config['region_name']
            
            # Create client
            client = boto3.client(**client_args)
            
            # Cache the client
            cls._clients[cache_key] = client
            
            logger.debug(f"Created new {service_name} client for region {client_args.get('region_name', 'default')}")
            return client
            
        except NoCredentialsError:
            logger.error(f"AWS credentials not configured for {service_name} client")
            raise
        except Exception as e:
            logger.error(f"Failed to create {service_name} client: {str(e)}")
            raise
    
    @classmethod
    def get_session(cls, region: Optional[str] = None) -> boto3.Session:
        """
        Get AWS session with proper configuration.
        
        Args:
            region: AWS region (optional)
            
        Returns:
            Configured boto3 session
        """
        try:
            session_args = {}
            if region:
                session_args['region_name'] = region
            
            session = boto3.Session(**session_args)
            
            # Verify credentials by making a simple call
            sts_client = session.client('sts')
            sts_client.get_caller_identity()
            
            logger.debug(f"Created AWS session for region {region or 'default'}")
            return session
            
        except NoCredentialsError:
            logger.error("AWS credentials not configured")
            raise
        except Exception as e:
            logger.error(f"Failed to create AWS session: {str(e)}")
            raise
    
    @classmethod
    def clear_cache(cls):
        """Clear the client cache."""
        cls._clients.clear()
        logger.debug("Cleared AWS client cache")
    
    @classmethod
    def get_available_regions(cls, service_name: str) -> list:
        """
        Get available regions for a service.
        
        Args:
            service_name: AWS service name
            
        Returns:
            List of available regions
        """
        try:
            session = boto3.Session()
            return session.get_available_regions(service_name)
        except Exception as e:
            logger.warning(f"Could not get regions for {service_name}: {str(e)}")
            return []
    
    @classmethod
    def validate_region(cls, service_name: str, region: str) -> bool:
        """
        Validate if a region is available for a service.
        
        Args:
            service_name: AWS service name
            region: AWS region to validate
            
        Returns:
            True if region is valid for service
        """
        available_regions = cls.get_available_regions(service_name)
        return region in available_regions if available_regions else True
    
    @classmethod
    def get_caller_identity(cls) -> Dict[str, Any]:
        """
        Get AWS caller identity information.
        
        Returns:
            Dictionary with account ID, user ARN, etc.
        """
        try:
            sts_client = cls.get_client('sts')
            return sts_client.get_caller_identity()
        except Exception as e:
            logger.error(f"Failed to get caller identity: {str(e)}")
            raise


# Convenience functions
def get_cost_explorer_client() -> boto3.client:
    """Get Cost Explorer client (always us-east-1)."""
    return AWSClientFactory.get_client('ce')


def get_cost_optimization_hub_client() -> boto3.client:
    """Get Cost Optimization Hub client (always us-east-1)."""
    return AWSClientFactory.get_client('cost-optimization-hub')


def get_compute_optimizer_client(region: Optional[str] = None) -> boto3.client:
    """Get Compute Optimizer client."""
    return AWSClientFactory.get_client('compute-optimizer', region)


def get_trusted_advisor_client() -> boto3.client:
    """Get Trusted Advisor client (always us-east-1)."""
    return AWSClientFactory.get_client('support')


def get_performance_insights_client(region: Optional[str] = None) -> boto3.client:
    """Get Performance Insights client."""
    return AWSClientFactory.get_client('pi', region)


def get_regional_client(service_name: str, region: str) -> boto3.client:
    """Get regional client for EC2, EBS, RDS, Lambda, S3, etc."""
    return AWSClientFactory.get_client(service_name, region)