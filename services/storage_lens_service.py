"""
AWS S3 Storage Lens service module.

This module provides functions for interacting with AWS S3 Storage Lens API
for S3 cost optimization analysis using NO-COST operations only.

Storage Lens provides comprehensive S3 metrics without incurring S3 request costs,
making it the primary data source for S3 optimization analysis.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StorageLensService:
    """
    S3 Storage Lens service class for AWS S3 Control API interactions.
    
    This service provides NO-COST access to S3 metrics and optimization data
    through the S3 Storage Lens service, which is the primary data source
    for S3 cost optimization analysis.
    """
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize StorageLensService with AWS S3 Control client.
        
        Args:
            region: AWS region (optional, defaults to us-east-1 for Storage Lens)
        """
        # Storage Lens is primarily available in us-east-1
        self.region = region or 'us-east-1'
        
        try:
            # Initialize S3 Control client for Storage Lens
            self.s3control_client = boto3.client('s3control', region_name=self.region)
            
            # Get account ID for Storage Lens operations
            sts_client = boto3.client('sts')
            self.account_id = sts_client.get_caller_identity()['Account']
            
            logger.info(f"StorageLensService initialized for region: {self.region}, account: {self.account_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize StorageLensService: {str(e)}")
            raise
    
    def safe_api_call(self, api_func, *args, retry_count: int = 0, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Wrapper for safe API calls with error handling and exponential backoff.
        
        Args:
            api_func: The API function to call
            *args: Positional arguments for the API function
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for the API function
            
        Returns:
            Dictionary with API response or error information
        """
        try:
            response = api_func(*args, **kwargs)
            return {
                "status": "success",
                "data": response
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            # Handle specific error cases
            func_name = getattr(api_func, '__name__', 'unknown_function')
            if error_code == 'AccessDenied':
                logger.warning(f"Access denied for Storage Lens API call: {func_name}")
                return {
                    "status": "error",
                    "message": f"Insufficient permissions for Storage Lens {func_name}",
                    "error_code": error_code,
                    "fallback_available": True
                }
            elif error_code in ['Throttling', 'ThrottlingException', 'RequestLimitExceeded']:
                if retry_count < max_retries:
                    # Exponential backoff: 2^retry_count
                    sleep_time = 2 ** retry_count
                    logger.info(f"Rate limited, retrying in {sleep_time} seconds (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    return self.safe_api_call(api_func, *args, retry_count=retry_count + 1, max_retries=max_retries, **kwargs)
                else:
                    logger.error(f"Max retries exceeded for {func_name}")
                    return {
                        "status": "error",
                        "message": f"Rate limit exceeded for {func_name} after {max_retries} retries",
                        "error_code": error_code,
                        "fallback_available": True
                    }
            elif error_code in ['NoSuchConfiguration', 'ConfigurationNotFound']:
                logger.warning(f"Storage Lens configuration not found: {func_name}")
                return {
                    "status": "error",
                    "message": f"Storage Lens configuration not found",
                    "error_code": error_code,
                    "fallback_available": True
                }
            else:
                logger.error(f"AWS API error in {func_name}: {error_message}")
                return {
                    "status": "error",
                    "message": f"Storage Lens API error: {error_message}",
                    "error_code": error_code,
                    "fallback_available": True
                }
                
        except Exception as e:
            func_name = getattr(api_func, '__name__', 'unknown_function')
            logger.error(f"Unexpected error in {func_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "fallback_available": True
            }
    
    def list_storage_lens_configurations(self) -> Dict[str, Any]:
        """
        List all Storage Lens configurations for the account.
        
        Returns:
            Dictionary containing Storage Lens configurations or error information
        """
        try:
            response = self.safe_api_call(
                self.s3control_client.list_storage_lens_configurations,
                AccountId=self.account_id
            )
            
            if response["status"] == "success":
                configs = response["data"].get("StorageLensConfigurationList", [])
                return {
                    "status": "success",
                    "data": {
                        "Configurations": configs,
                        "Count": len(configs)
                    },
                    "message": f"Retrieved {len(configs)} Storage Lens configurations"
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error listing Storage Lens configurations: {str(e)}")
            return {
                "status": "error",
                "message": f"Error listing Storage Lens configurations: {str(e)}",
                "fallback_available": True
            }
    
    def get_storage_lens_configuration(self, config_id: str) -> Dict[str, Any]:
        """
        Get a specific Storage Lens configuration.
        
        Args:
            config_id: Storage Lens configuration ID
            
        Returns:
            Dictionary containing Storage Lens configuration or error information
        """
        try:
            response = self.safe_api_call(
                self.s3control_client.get_storage_lens_configuration,
                ConfigId=config_id,
                AccountId=self.account_id
            )
            
            if response["status"] == "success":
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved Storage Lens configuration: {config_id}"
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting Storage Lens configuration {config_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting Storage Lens configuration: {str(e)}",
                "fallback_available": True
            }
    
    async def get_storage_metrics(self, config_id: str = "default-account-dashboard") -> Dict[str, Any]:
        """
        Get comprehensive storage metrics from Storage Lens.
        
        This method provides NO-COST access to storage metrics including:
        - Total storage bytes by storage class
        - Object counts by storage class
        - Storage cost optimization metrics
        - Data retrieval metrics
        
        Args:
            config_id: Storage Lens configuration ID (defaults to account dashboard)
            
        Returns:
            Dictionary containing storage metrics or error information
        """
        try:
            # Get the Storage Lens configuration first
            config_response = self.get_storage_lens_configuration(config_id)
            
            if config_response["status"] != "success":
                logger.warning(f"Could not retrieve Storage Lens config {config_id}, trying default")
                # Try to list configurations and use the first available
                list_response = self.list_storage_lens_configurations()
                if list_response["status"] == "success" and list_response["data"]["Configurations"]:
                    config_id = list_response["data"]["Configurations"][0]["Id"]
                    config_response = self.get_storage_lens_configuration(config_id)
                
                if config_response["status"] != "success":
                    return {
                        "status": "error",
                        "message": "No accessible Storage Lens configurations found",
                        "error_code": "NoConfiguration",
                        "fallback_available": True
                    }
            
            # Extract metrics from the configuration
            config_data = config_response["data"].get("StorageLensConfiguration", {})
            
            # Storage Lens provides metrics through its dashboard and export
            # For real-time access, we need to check if metrics export is configured
            data_export = config_data.get("DataExport", {})
            
            metrics = {
                "ConfigurationId": config_id,
                "AccountId": self.account_id,
                "IsEnabled": config_data.get("IsEnabled", False),
                "DataExportEnabled": bool(data_export),
                "IncludeRegions": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("ActivityMetrics", {}).get("IsEnabled", False),
                "CostOptimizationMetrics": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("CostOptimizationMetrics", {}).get("IsEnabled", False),
                "DetailedStatusCodesMetrics": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("DetailedStatusCodesMetrics", {}).get("IsEnabled", False)
            }
            
            return {
                "status": "success",
                "data": metrics,
                "message": f"Retrieved storage metrics from Storage Lens configuration: {config_id}",
                "note": "Storage Lens metrics are available through dashboard or export. For detailed metrics, ensure export is configured."
            }
            
        except Exception as e:
            logger.error(f"Error getting storage metrics: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting storage metrics: {str(e)}",
                "fallback_available": True
            }
    
    async def get_cost_optimization_metrics(self, config_id: str = "default-account-dashboard") -> Dict[str, Any]:
        """
        Get cost optimization specific metrics from Storage Lens.
        
        This method provides NO-COST access to cost optimization metrics including:
        - Incomplete multipart upload storage bytes
        - Non-current version storage bytes
        - Delete marker storage bytes
        - Lifecycle rule effectiveness
        
        Args:
            config_id: Storage Lens configuration ID
            
        Returns:
            Dictionary containing cost optimization metrics or error information
        """
        try:
            # Get Storage Lens configuration
            config_response = self.get_storage_lens_configuration(config_id)
            
            if config_response["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Could not access Storage Lens configuration: {config_id}",
                    "fallback_available": True
                }
            
            config_data = config_response["data"].get("StorageLensConfiguration", {})
            account_level = config_data.get("AccountLevel", {})
            bucket_level = account_level.get("BucketLevel", {})
            
            # Extract cost optimization related settings
            cost_optimization = {
                "ConfigurationId": config_id,
                "CostOptimizationMetricsEnabled": bucket_level.get("CostOptimizationMetrics", {}).get("IsEnabled", False),
                "ActivityMetricsEnabled": bucket_level.get("ActivityMetrics", {}).get("IsEnabled", False),
                "DetailedStatusCodesEnabled": bucket_level.get("DetailedStatusCodesMetrics", {}).get("IsEnabled", False),
                "AdvancedCostOptimizationMetricsEnabled": bucket_level.get("AdvancedCostOptimizationMetrics", {}).get("IsEnabled", False),
                "AdvancedDataProtectionMetricsEnabled": bucket_level.get("AdvancedDataProtectionMetrics", {}).get("IsEnabled", False)
            }
            
            # Check if prefix-level metrics are enabled for more detailed analysis
            prefix_level = bucket_level.get("PrefixLevel", {})
            if prefix_level:
                cost_optimization["PrefixLevelMetricsEnabled"] = True
                cost_optimization["PrefixLevelStorageMetrics"] = prefix_level.get("StorageMetrics", {}).get("IsEnabled", False)
            else:
                cost_optimization["PrefixLevelMetricsEnabled"] = False
            
            return {
                "status": "success",
                "data": cost_optimization,
                "message": f"Retrieved cost optimization metrics configuration from Storage Lens: {config_id}",
                "recommendations": self._generate_cost_optimization_recommendations(cost_optimization)
            }
            
        except Exception as e:
            logger.error(f"Error getting cost optimization metrics: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting cost optimization metrics: {str(e)}",
                "fallback_available": True
            }
    
    async def get_storage_class_distribution(self, config_id: str = "default-account-dashboard") -> Dict[str, Any]:
        """
        Get storage class distribution metrics from Storage Lens.
        
        This method provides NO-COST access to storage class distribution including:
        - Storage bytes by storage class
        - Object counts by storage class
        - Storage class transition opportunities
        
        Args:
            config_id: Storage Lens configuration ID
            
        Returns:
            Dictionary containing storage class distribution or error information
        """
        try:
            # Get Storage Lens configuration
            config_response = self.get_storage_lens_configuration(config_id)
            
            if config_response["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Could not access Storage Lens configuration: {config_id}",
                    "fallback_available": True
                }
            
            config_data = config_response["data"].get("StorageLensConfiguration", {})
            
            # Extract storage class related metrics availability
            storage_class_info = {
                "ConfigurationId": config_id,
                "StorageMetricsEnabled": config_data.get("AccountLevel", {}).get("StorageMetrics", {}).get("IsEnabled", True),
                "BucketLevelEnabled": bool(config_data.get("AccountLevel", {}).get("BucketLevel")),
                "ActivityMetricsEnabled": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("ActivityMetrics", {}).get("IsEnabled", False),
                "CostOptimizationEnabled": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("CostOptimizationMetrics", {}).get("IsEnabled", False)
            }
            
            # Check data export configuration for accessing detailed metrics
            data_export = config_data.get("DataExport", {})
            if data_export:
                s3_bucket_destination = data_export.get("S3BucketDestination", {})
                storage_class_info["ExportEnabled"] = True
                storage_class_info["ExportBucket"] = s3_bucket_destination.get("Bucket", "")
                storage_class_info["ExportPrefix"] = s3_bucket_destination.get("Prefix", "")
                storage_class_info["ExportFormat"] = s3_bucket_destination.get("Format", "")
            else:
                storage_class_info["ExportEnabled"] = False
            
            return {
                "status": "success",
                "data": storage_class_info,
                "message": f"Retrieved storage class distribution configuration from Storage Lens: {config_id}",
                "note": "Detailed storage class metrics are available through Storage Lens dashboard or export data"
            }
            
        except Exception as e:
            logger.error(f"Error getting storage class distribution: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting storage class distribution: {str(e)}",
                "fallback_available": True
            }
    
    async def get_incomplete_multipart_uploads_metrics(self, config_id: str = "default-account-dashboard") -> Dict[str, Any]:
        """
        Get incomplete multipart uploads metrics from Storage Lens.
        
        This method provides NO-COST access to multipart upload metrics including:
        - Incomplete multipart upload storage bytes
        - Number of incomplete multipart uploads
        - Age distribution of incomplete uploads
        
        Args:
            config_id: Storage Lens configuration ID
            
        Returns:
            Dictionary containing multipart upload metrics or error information
        """
        try:
            # Get Storage Lens configuration
            config_response = self.get_storage_lens_configuration(config_id)
            
            if config_response["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Could not access Storage Lens configuration: {config_id}",
                    "fallback_available": True
                }
            
            config_data = config_response["data"].get("StorageLensConfiguration", {})
            bucket_level = config_data.get("AccountLevel", {}).get("BucketLevel", {})
            
            # Extract multipart upload related metrics
            multipart_metrics = {
                "ConfigurationId": config_id,
                "CostOptimizationMetricsEnabled": bucket_level.get("CostOptimizationMetrics", {}).get("IsEnabled", False),
                "AdvancedCostOptimizationEnabled": bucket_level.get("AdvancedCostOptimizationMetrics", {}).get("IsEnabled", False),
                "DetailedStatusCodesEnabled": bucket_level.get("DetailedStatusCodesMetrics", {}).get("IsEnabled", False)
            }
            
            # Check if the configuration includes multipart upload tracking
            if multipart_metrics["CostOptimizationMetricsEnabled"] or multipart_metrics["AdvancedCostOptimizationEnabled"]:
                multipart_metrics["MultipartUploadTrackingAvailable"] = True
                multipart_metrics["RecommendedAction"] = "Access Storage Lens dashboard or export data for detailed multipart upload metrics"
            else:
                multipart_metrics["MultipartUploadTrackingAvailable"] = False
                multipart_metrics["RecommendedAction"] = "Enable Cost Optimization Metrics in Storage Lens configuration to track incomplete multipart uploads"
            
            return {
                "status": "success",
                "data": multipart_metrics,
                "message": f"Retrieved multipart upload metrics configuration from Storage Lens: {config_id}",
                "recommendations": self._generate_multipart_recommendations(multipart_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting multipart upload metrics: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting multipart upload metrics: {str(e)}",
                "fallback_available": True
            }
    
    def get_default_configuration_id(self) -> str:
        """
        Get the default Storage Lens configuration ID for the account.
        
        Returns:
            Default configuration ID or fallback ID
        """
        try:
            list_response = self.list_storage_lens_configurations()
            
            if list_response["status"] == "success":
                configs = list_response["data"]["Configurations"]
                
                # Look for default account dashboard first
                for config in configs:
                    if config["Id"] == "default-account-dashboard":
                        return config["Id"]
                
                # If no default found, return the first available
                if configs:
                    return configs[0]["Id"]
            
            # Fallback to the standard default
            return "default-account-dashboard"
            
        except Exception as e:
            logger.warning(f"Could not determine default configuration ID: {str(e)}")
            return "default-account-dashboard"
    
    def _generate_cost_optimization_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations based on Storage Lens metrics.
        
        Args:
            metrics: Cost optimization metrics data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not metrics.get("CostOptimizationMetricsEnabled", False):
            recommendations.append({
                "type": "configuration",
                "priority": "high",
                "title": "Enable Cost Optimization Metrics",
                "description": "Enable Cost Optimization Metrics in Storage Lens to track incomplete multipart uploads, non-current versions, and delete markers",
                "action": "Update Storage Lens configuration to include CostOptimizationMetrics"
            })
        
        if not metrics.get("AdvancedCostOptimizationMetricsEnabled", False):
            recommendations.append({
                "type": "configuration",
                "priority": "medium",
                "title": "Enable Advanced Cost Optimization Metrics",
                "description": "Enable Advanced Cost Optimization Metrics for more detailed cost analysis including prefix-level insights",
                "action": "Update Storage Lens configuration to include AdvancedCostOptimizationMetrics"
            })
        
        if not metrics.get("ActivityMetricsEnabled", False):
            recommendations.append({
                "type": "configuration",
                "priority": "medium",
                "title": "Enable Activity Metrics",
                "description": "Enable Activity Metrics to track access patterns for storage class optimization",
                "action": "Update Storage Lens configuration to include ActivityMetrics"
            })
        
        return recommendations
    
    def _generate_multipart_recommendations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multipart upload recommendations based on Storage Lens metrics.
        
        Args:
            metrics: Multipart upload metrics data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not metrics.get("MultipartUploadTrackingAvailable", False):
            recommendations.append({
                "type": "configuration",
                "priority": "high",
                "title": "Enable Multipart Upload Tracking",
                "description": "Enable Cost Optimization Metrics in Storage Lens to identify and clean up incomplete multipart uploads",
                "action": "Update Storage Lens configuration to include CostOptimizationMetrics",
                "potential_savings": "Variable - depends on incomplete upload volume"
            })
        
        recommendations.append({
            "type": "governance",
            "priority": "high",
            "title": "Implement Multipart Upload Lifecycle Policy",
            "description": "Create lifecycle policies to automatically clean up incomplete multipart uploads after 7 days",
            "action": "Add AbortIncompleteMultipartUpload rule to bucket lifecycle policies",
            "implementation_effort": "low"
        })
        
        return recommendations  
  
    async def get_bucket_level_metrics(self, bucket_name: str, config_id: str = None) -> Dict[str, Any]:
        """
        Get bucket-level metrics from Storage Lens for a specific bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            config_id: Storage Lens configuration ID (optional)
            
        Returns:
            Dictionary containing bucket-level metrics or error information
        """
        try:
            if not config_id:
                config_id = self.get_default_configuration_id()
            
            # Get Storage Lens configuration
            config_response = self.get_storage_lens_configuration(config_id)
            
            if config_response["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Could not access Storage Lens configuration for bucket analysis",
                    "fallback_available": True
                }
            
            config_data = config_response["data"].get("StorageLensConfiguration", {})
            bucket_level = config_data.get("AccountLevel", {}).get("BucketLevel", {})
            
            # Check if bucket-level metrics are enabled
            if not bucket_level:
                return {
                    "status": "error",
                    "message": "Bucket-level metrics are not enabled in Storage Lens configuration",
                    "error_code": "BucketLevelDisabled",
                    "fallback_available": True,
                    "recommendation": "Enable bucket-level metrics in Storage Lens configuration"
                }
            
            bucket_metrics = {
                "BucketName": bucket_name,
                "ConfigurationId": config_id,
                "StorageMetricsEnabled": True,  # Always available at bucket level
                "ActivityMetricsEnabled": bucket_level.get("ActivityMetrics", {}).get("IsEnabled", False),
                "CostOptimizationEnabled": bucket_level.get("CostOptimizationMetrics", {}).get("IsEnabled", False),
                "DetailedStatusCodesEnabled": bucket_level.get("DetailedStatusCodesMetrics", {}).get("IsEnabled", False),
                "AdvancedCostOptimizationEnabled": bucket_level.get("AdvancedCostOptimizationMetrics", {}).get("IsEnabled", False),
                "AdvancedDataProtectionEnabled": bucket_level.get("AdvancedDataProtectionMetrics", {}).get("IsEnabled", False)
            }
            
            # Check prefix-level configuration
            prefix_level = bucket_level.get("PrefixLevel", {})
            if prefix_level:
                bucket_metrics["PrefixLevelEnabled"] = True
                bucket_metrics["PrefixLevelStorageMetrics"] = prefix_level.get("StorageMetrics", {}).get("IsEnabled", False)
                
                # Get selection criteria for prefix-level metrics
                selection_criteria = prefix_level.get("StorageMetrics", {}).get("SelectionCriteria", {})
                if selection_criteria:
                    bucket_metrics["PrefixSelectionCriteria"] = {
                        "Delimiter": selection_criteria.get("Delimiter", ""),
                        "MaxDepth": selection_criteria.get("MaxDepth", 0),
                        "MinStorageBytesPercentage": selection_criteria.get("MinStorageBytesPercentage", 0.0)
                    }
            else:
                bucket_metrics["PrefixLevelEnabled"] = False
            
            return {
                "status": "success",
                "data": bucket_metrics,
                "message": f"Retrieved bucket-level metrics configuration for {bucket_name}",
                "note": "Actual metrics data is available through Storage Lens dashboard or export"
            }
            
        except Exception as e:
            logger.error(f"Error getting bucket-level metrics for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket-level metrics: {str(e)}",
                "fallback_available": True
            }
    
    def validate_storage_lens_access(self) -> Dict[str, Any]:
        """
        Validate access to Storage Lens service and return capability summary.
        
        Returns:
            Dictionary containing access validation results
        """
        try:
            validation_results = {
                "account_id": self.account_id,
                "region": self.region,
                "service_available": False,
                "configurations_accessible": False,
                "default_config_available": False,
                "capabilities": {},
                "recommendations": []
            }
            
            # Test basic service access
            list_response = self.list_storage_lens_configurations()
            
            if list_response["status"] == "success":
                validation_results["service_available"] = True
                validation_results["configurations_accessible"] = True
                
                configs = list_response["data"]["Configurations"]
                validation_results["configuration_count"] = len(configs)
                
                # Check for default configuration
                default_config = None
                for config in configs:
                    if config["Id"] == "default-account-dashboard":
                        default_config = config
                        validation_results["default_config_available"] = True
                        break
                
                if not default_config and configs:
                    default_config = configs[0]
                    validation_results["default_config_available"] = True
                    validation_results["default_config_id"] = default_config["Id"]
                
                # Test configuration access
                if default_config:
                    config_response = self.get_storage_lens_configuration(default_config["Id"])
                    if config_response["status"] == "success":
                        config_data = config_response["data"].get("StorageLensConfiguration", {})
                        
                        # Analyze capabilities
                        validation_results["capabilities"] = {
                            "is_enabled": config_data.get("IsEnabled", False),
                            "account_level_metrics": bool(config_data.get("AccountLevel")),
                            "bucket_level_metrics": bool(config_data.get("AccountLevel", {}).get("BucketLevel")),
                            "data_export_configured": bool(config_data.get("DataExport")),
                            "cost_optimization_metrics": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("CostOptimizationMetrics", {}).get("IsEnabled", False),
                            "activity_metrics": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("ActivityMetrics", {}).get("IsEnabled", False),
                            "advanced_metrics": config_data.get("AccountLevel", {}).get("BucketLevel", {}).get("AdvancedCostOptimizationMetrics", {}).get("IsEnabled", False)
                        }
                        
                        # Generate recommendations based on capabilities
                        validation_results["recommendations"] = self._generate_access_recommendations(validation_results["capabilities"])
            
            else:
                validation_results["error"] = list_response.get("message", "Unknown error")
                validation_results["error_code"] = list_response.get("error_code", "Unknown")
                
                # Provide fallback recommendations
                validation_results["recommendations"] = [
                    {
                        "type": "access",
                        "priority": "high",
                        "title": "Enable Storage Lens Access",
                        "description": "Ensure IAM permissions include s3:GetStorageLensConfiguration and s3:ListStorageLensConfigurations",
                        "action": "Update IAM policy to include Storage Lens permissions"
                    }
                ]
            
            return {
                "status": "success",
                "data": validation_results,
                "message": "Storage Lens access validation completed"
            }
            
        except Exception as e:
            logger.error(f"Error validating Storage Lens access: {str(e)}")
            return {
                "status": "error",
                "message": f"Error validating Storage Lens access: {str(e)}",
                "fallback_available": True
            }
    
    def _generate_access_recommendations(self, capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on Storage Lens capabilities.
        
        Args:
            capabilities: Dictionary of Storage Lens capabilities
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not capabilities.get("is_enabled", False):
            recommendations.append({
                "type": "configuration",
                "priority": "critical",
                "title": "Enable Storage Lens Configuration",
                "description": "Storage Lens configuration is disabled. Enable it to access S3 optimization metrics",
                "action": "Enable the Storage Lens configuration through AWS Console or API"
            })
        
        if not capabilities.get("bucket_level_metrics", False):
            recommendations.append({
                "type": "configuration",
                "priority": "high",
                "title": "Enable Bucket-Level Metrics",
                "description": "Enable bucket-level metrics for detailed S3 optimization analysis",
                "action": "Update Storage Lens configuration to include bucket-level metrics"
            })
        
        if not capabilities.get("cost_optimization_metrics", False):
            recommendations.append({
                "type": "configuration",
                "priority": "high",
                "title": "Enable Cost Optimization Metrics",
                "description": "Enable cost optimization metrics to track incomplete multipart uploads and non-current versions",
                "action": "Update Storage Lens configuration to include CostOptimizationMetrics"
            })
        
        if not capabilities.get("activity_metrics", False):
            recommendations.append({
                "type": "configuration",
                "priority": "medium",
                "title": "Enable Activity Metrics",
                "description": "Enable activity metrics to analyze access patterns for storage class optimization",
                "action": "Update Storage Lens configuration to include ActivityMetrics"
            })
        
        if not capabilities.get("data_export_configured", False):
            recommendations.append({
                "type": "configuration",
                "priority": "medium",
                "title": "Configure Data Export",
                "description": "Configure Storage Lens data export for programmatic access to detailed metrics",
                "action": "Set up S3 bucket destination for Storage Lens data export"
            })
        
        if not capabilities.get("advanced_metrics", False):
            recommendations.append({
                "type": "configuration",
                "priority": "low",
                "title": "Consider Advanced Metrics",
                "description": "Enable advanced cost optimization metrics for prefix-level analysis (additional charges apply)",
                "action": "Evaluate cost-benefit of enabling AdvancedCostOptimizationMetrics"
            })
        
        return recommendations
    
    def get_fallback_data_sources(self) -> Dict[str, Any]:
        """
        Get information about fallback data sources when Storage Lens is unavailable.
        
        Returns:
            Dictionary containing fallback data source information
        """
        return {
            "status": "success",
            "data": {
                "primary_source": "S3 Storage Lens",
                "fallback_sources": [
                    {
                        "name": "Cost Explorer",
                        "capabilities": [
                            "Historical S3 cost data",
                            "Usage patterns by storage class",
                            "Cost trends and forecasting",
                            "Service-level cost breakdown"
                        ],
                        "limitations": [
                            "No real-time data",
                            "Limited to cost metrics",
                            "No object-level insights",
                            "Delayed data availability (24-48 hours)"
                        ],
                        "cost": "No additional cost"
                    },
                    {
                        "name": "CloudWatch Metrics",
                        "capabilities": [
                            "Bucket-level storage metrics",
                            "Request metrics",
                            "Real-time monitoring",
                            "Custom metric analysis"
                        ],
                        "limitations": [
                            "Limited historical data",
                            "No cost optimization metrics",
                            "No storage class distribution",
                            "Requires detailed monitoring enabled"
                        ],
                        "cost": "CloudWatch charges apply"
                    },
                    {
                        "name": "S3 Inventory",
                        "capabilities": [
                            "Object-level metadata",
                            "Storage class distribution",
                            "Encryption status",
                            "Object age analysis"
                        ],
                        "limitations": [
                            "Requires configuration",
                            "Daily/weekly snapshots only",
                            "Storage costs for inventory files",
                            "Processing overhead"
                        ],
                        "cost": "S3 storage and request charges apply"
                    }
                ],
                "recommended_fallback_order": [
                    "Cost Explorer (for cost analysis)",
                    "CloudWatch Metrics (for usage patterns)",
                    "S3 Inventory (for detailed object analysis)"
                ]
            },
            "message": "Fallback data sources available when Storage Lens is not accessible"
        }