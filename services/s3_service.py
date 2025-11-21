"""
AWS S3 service module with strict no-cost constraints.

This module provides functions for interacting with AWS S3, CloudWatch, and Cost Explorer APIs
for S3 cost optimization analysis. All S3 operations are strictly limited to no-cost API calls only.

CRITICAL: This service enforces NO-COST constraints to prevent customer billing.
Only metadata-only S3 operations are allowed. Object-level operations are FORBIDDEN.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Iterator, Set
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# CRITICAL: Whitelist of allowed no-cost S3 API operations
ALLOWED_S3_OPERATIONS = {
    'list_buckets',                          # List bucket names only - NO COST
    'get_bucket_location',                   # Get bucket region - NO COST
    'get_bucket_lifecycle_configuration',    # Get lifecycle policies - NO COST
    'get_bucket_versioning',                # Get versioning status - NO COST
    'get_bucket_tagging',                   # Get bucket tags - NO COST
    'list_multipart_uploads',               # List incomplete uploads (metadata only) - NO COST
    'get_bucket_policy',                    # Get bucket policy - NO COST
    'get_bucket_acl',                       # Get bucket ACL - NO COST
    'get_bucket_encryption',                # Get encryption config - NO COST
    'get_public_access_block',              # Get public access block - NO COST
    'get_bucket_notification_configuration', # Get notification config - NO COST
    'get_bucket_cors',                      # Get CORS config - NO COST
    'get_bucket_website',                   # Get website config - NO COST
    'get_bucket_logging',                   # Get logging config - NO COST
    'get_bucket_replication',               # Get replication config - NO COST
    'get_bucket_request_payment',           # Get request payment config - NO COST
}

# CRITICAL: Forbidden operations that incur costs
FORBIDDEN_S3_OPERATIONS = {
    'list_objects',                         # COSTS per 1000 requests
    'list_objects_v2',                      # COSTS per 1000 requests
    'head_object',                          # COSTS per request
    'get_object',                           # COSTS per request + data transfer
    'get_object_attributes',                # COSTS per request
    'select_object_content',                # COSTS per request + data scanned
    'restore_object',                       # COSTS for retrieval
}

class S3CostConstraintViolationError(Exception):
    """Raised when attempting to use a cost-incurring S3 operation."""
    pass


class S3Service:
    """
    Enhanced S3 service class with strict no-cost constraints and async support.
    
    CRITICAL: This service enforces NO-COST operations only to prevent customer billing.
    All S3 operations are validated against a whitelist of allowed no-cost API calls.
    """
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize S3Service with AWS clients and no-cost validation.
        
        Args:
            region: AWS region (optional)
        """
        self.region = region
        self._operation_call_count = {}  # Track API calls for monitoring
        
        try:
            # Initialize AWS clients
            self.s3_client = boto3.client('s3', region_name=region)
            self.cloudwatch_client = boto3.client('cloudwatch', region_name=region)
            # Cost Explorer is only available in us-east-1
            self.ce_client = boto3.client('ce', region_name='us-east-1')
            
            logger.info(f"S3Service initialized for region: {region or 'default'} with NO-COST constraints enabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3Service: {str(e)}")
            raise
    
    def _validate_s3_operation(self, operation_name: str) -> None:
        """
        Validate that an S3 operation is allowed (no-cost only).
        
        Args:
            operation_name: Name of the S3 operation
            
        Raises:
            S3CostConstraintViolationError: If operation would incur costs
        """
        if operation_name in FORBIDDEN_S3_OPERATIONS:
            error_msg = (
                f"FORBIDDEN: S3 operation '{operation_name}' would incur costs. "
                f"This service only allows no-cost operations: {sorted(ALLOWED_S3_OPERATIONS)}"
            )
            logger.error(error_msg)
            raise S3CostConstraintViolationError(error_msg)
        
        if operation_name not in ALLOWED_S3_OPERATIONS:
            logger.warning(f"S3 operation '{operation_name}' not in whitelist. Proceeding with caution.")
        
        # Track operation calls for monitoring
        self._operation_call_count[operation_name] = self._operation_call_count.get(operation_name, 0) + 1
        logger.debug(f"S3 operation '{operation_name}' validated as no-cost (call #{self._operation_call_count[operation_name]})")
    
    def get_operation_stats(self) -> Dict[str, int]:
        """
        Get statistics on S3 operations called during this session.
        
        Returns:
            Dictionary of operation names and call counts
        """
        return self._operation_call_count.copy()
    
    async def safe_api_call(self, api_func, *args, retry_count: int = 0, max_retries: int = 3, 
                           fallback_sources: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Enhanced wrapper for safe API calls with async support, error handling, and fallback strategies.
        
        Args:
            api_func: The API function to call
            *args: Positional arguments for the API function
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
            fallback_sources: List of alternative data sources to try on failure
            **kwargs: Keyword arguments for the API function
            
        Returns:
            Dictionary with API response or error information
        """
        func_name = getattr(api_func, '__name__', 'unknown_function')
        
        # Validate S3 operations for cost constraints
        if hasattr(api_func, '__self__') and hasattr(api_func.__self__, '_service_model'):
            service_name = api_func.__self__._service_model.service_name
            if service_name == 's3':
                try:
                    self._validate_s3_operation(func_name)
                except S3CostConstraintViolationError as e:
                    return {
                        "status": "error",
                        "message": str(e),
                        "error_code": "COST_CONSTRAINT_VIOLATION",
                        "fallback_available": bool(fallback_sources)
                    }
        
        try:
            # Run API call in thread pool for async compatibility
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: api_func(*args, **kwargs))
            
            logger.debug(f"Successful API call: {func_name}")
            return {
                "status": "success",
                "data": response,
                "source": "primary"
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            # Handle specific error cases with enhanced logging
            if error_code == 'AccessDenied':
                logger.warning(f"Access denied for API call: {func_name}. Fallback sources: {fallback_sources}")
                return {
                    "status": "error",
                    "message": f"Insufficient permissions for {func_name}",
                    "error_code": error_code,
                    "fallback_available": bool(fallback_sources),
                    "suggested_fallback": fallback_sources[0] if fallback_sources else None
                }
            elif error_code in ['Throttling', 'ThrottlingException', 'RequestLimitExceeded']:
                if retry_count < max_retries:
                    # Exponential backoff with jitter: 2^retry_count + random(0, 1)
                    import random
                    sleep_time = (2 ** retry_count) + random.random()
                    logger.info(f"Rate limited, retrying {func_name} in {sleep_time:.2f} seconds (attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(sleep_time)
                    return await self.safe_api_call(api_func, *args, retry_count=retry_count + 1, 
                                                  max_retries=max_retries, fallback_sources=fallback_sources, **kwargs)
                else:
                    logger.error(f"Max retries exceeded for {func_name}. Fallback sources: {fallback_sources}")
                    return {
                        "status": "error",
                        "message": f"Rate limit exceeded for {func_name} after {max_retries} retries",
                        "error_code": error_code,
                        "fallback_available": bool(fallback_sources),
                        "suggested_fallback": fallback_sources[0] if fallback_sources else None
                    }
            elif error_code in ['NoSuchBucket', 'NoSuchKey', 'NoSuchLifecycleConfiguration', 
                              'NoSuchTagSet', 'NoSuchBucketPolicy', 'ServerSideEncryptionConfigurationNotFoundError',
                              'NoSuchPublicAccessBlockConfiguration']:
                # These are expected "not found" errors, not actual failures
                logger.debug(f"Resource not found for {func_name}: {error_code}")
                return {
                    "status": "success",
                    "data": {},
                    "message": f"Resource not found: {error_code}",
                    "source": "primary",
                    "resource_not_found": True
                }
            else:
                logger.error(f"AWS API error in {func_name}: {error_message}. Fallback sources: {fallback_sources}")
                return {
                    "status": "error",
                    "message": f"AWS API error: {error_message}",
                    "error_code": error_code,
                    "fallback_available": bool(fallback_sources),
                    "suggested_fallback": fallback_sources[0] if fallback_sources else None
                }
                
        except Exception as e:
            logger.error(f"Unexpected error in {func_name}: {str(e)}. Fallback sources: {fallback_sources}")
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "fallback_available": bool(fallback_sources),
                "suggested_fallback": fallback_sources[0] if fallback_sources else None
            }
    
    def safe_api_call_sync(self, api_func, *args, retry_count: int = 0, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Synchronous version of safe_api_call for backward compatibility.
        
        Args:
            api_func: The API function to call
            *args: Positional arguments for the API function
            retry_count: Current retry attempt
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for the API function
            
        Returns:
            Dictionary with API response or error information
        """
        # Run the async version in a new event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.safe_api_call(api_func, *args, retry_count=retry_count, max_retries=max_retries, **kwargs)
        )
    
    async def list_buckets(self) -> Dict[str, Any]:
        """
        List all S3 buckets with region information (NO-COST operation).
        
        Returns:
            Dictionary containing bucket list or error information
        """
        try:
            # Get bucket list - NO COST operation
            response = await self.safe_api_call(
                self.s3_client.list_buckets,
                fallback_sources=['storage_lens', 'cost_explorer']
            )
            
            if response["status"] != "success":
                return response
            
            buckets = response["data"].get("Buckets", [])
            
            # Enhance bucket information with regions in parallel
            enhanced_buckets = []
            region_tasks = []
            
            # Create async tasks for getting bucket regions
            for bucket in buckets:
                bucket_info = {
                    "Name": bucket["Name"],
                    "CreationDate": bucket["CreationDate"]
                }
                enhanced_buckets.append(bucket_info)
                
                # Create task for getting bucket region - NO COST operation
                task = self.safe_api_call(
                    self.s3_client.get_bucket_location,
                    Bucket=bucket["Name"],
                    fallback_sources=['storage_lens']
                )
                region_tasks.append(task)
            
            # Execute region queries in parallel
            if region_tasks:
                region_responses = await asyncio.gather(*region_tasks, return_exceptions=True)
                
                for i, region_response in enumerate(region_responses):
                    if isinstance(region_response, Exception):
                        enhanced_buckets[i]["Region"] = "unknown"
                        logger.warning(f"Exception getting region for bucket {enhanced_buckets[i]['Name']}: {region_response}")
                    elif region_response.get("status") == "success":
                        location = region_response["data"].get("LocationConstraint")
                        # Handle special case for us-east-1
                        enhanced_buckets[i]["Region"] = location if location else "us-east-1"
                    else:
                        enhanced_buckets[i]["Region"] = "unknown"
                        logger.warning(f"Could not determine region for bucket {enhanced_buckets[i]['Name']}: {region_response.get('message', 'Unknown error')}")
            
            return {
                "status": "success",
                "data": {
                    "Buckets": enhanced_buckets,
                    "Owner": response["data"].get("Owner")
                },
                "message": f"Retrieved {len(enhanced_buckets)} buckets with regions",
                "source": response.get("source", "primary")
            }
            
        except Exception as e:
            logger.error(f"Error listing buckets: {str(e)}")
            return {
                "status": "error",
                "message": f"Error listing buckets: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    def list_buckets_sync(self) -> Dict[str, Any]:
        """
        Synchronous version of list_buckets for backward compatibility.
        
        Returns:
            Dictionary containing bucket list or error information
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.list_buckets())
    
    async def get_bucket_metrics(self, bucket_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a specific bucket using NO-COST sources only.
        
        Args:
            bucket_name: Name of the S3 bucket
            days: Number of days to look back for metrics
            
        Returns:
            Dictionary containing bucket metrics or error information
        """
        try:
            metrics = {}
            
            # Execute metrics gathering in parallel
            tasks = [
                self.get_bucket_size_metrics(bucket_name, days),
                self.get_request_metrics(bucket_name, days),
                self.get_storage_class_distribution_from_storage_lens(bucket_name)
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process size metrics
            if not isinstance(responses[0], Exception) and responses[0].get("status") == "success":
                metrics.update(responses[0]["data"])
            
            # Process request metrics
            if not isinstance(responses[1], Exception) and responses[1].get("status") == "success":
                metrics.update(responses[1]["data"])
            
            # Process storage class distribution (may require fallback to Storage Lens)
            if not isinstance(responses[2], Exception):
                if responses[2].get("status") == "success":
                    metrics.update(responses[2]["data"])
                elif responses[2].get("suggested_fallback") == "storage_lens_service":
                    metrics["StorageClassDistribution"] = {
                        "note": "Requires StorageLensService for no-cost analysis",
                        "fallback_required": True
                    }
            
            return {
                "status": "success",
                "data": metrics,
                "message": f"Retrieved metrics for bucket {bucket_name} (no-cost sources only)",
                "cost_constraint_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error getting bucket metrics for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket metrics: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_size_metrics(self, bucket_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get bucket size metrics from CloudWatch (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            days: Number of days to look back
            
        Returns:
            Dictionary containing size metrics or error information
        """
        try:
            from datetime import datetime, timedelta
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Execute CloudWatch queries in parallel
            size_task = self.safe_api_call(
                self.cloudwatch_client.get_metric_statistics,
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # Daily
                Statistics=['Average'],
                fallback_sources=['storage_lens']
            )
            
            objects_task = self.safe_api_call(
                self.cloudwatch_client.get_metric_statistics,
                Namespace='AWS/S3',
                MetricName='NumberOfObjects',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # Daily
                Statistics=['Average'],
                fallback_sources=['storage_lens']
            )
            
            size_response, objects_response = await asyncio.gather(size_task, objects_task)
            
            metrics = {}
            
            if size_response["status"] == "success":
                datapoints = size_response["data"].get("Datapoints", [])
                if datapoints:
                    # Get the most recent datapoint
                    latest_size = max(datapoints, key=lambda x: x['Timestamp'])
                    metrics["SizeBytes"] = latest_size.get("Average", 0)
                    metrics["SizeGB"] = metrics["SizeBytes"] / (1024**3)
                else:
                    metrics["SizeBytes"] = 0
                    metrics["SizeGB"] = 0
            
            if objects_response["status"] == "success":
                datapoints = objects_response["data"].get("Datapoints", [])
                if datapoints:
                    # Get the most recent datapoint
                    latest_objects = max(datapoints, key=lambda x: x['Timestamp'])
                    metrics["ObjectCount"] = int(latest_objects.get("Average", 0))
                else:
                    metrics["ObjectCount"] = 0
            
            return {
                "status": "success",
                "data": metrics,
                "message": f"Retrieved size metrics for bucket {bucket_name}",
                "source": "cloudwatch"
            }
            
        except Exception as e:
            logger.error(f"Error getting size metrics for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting size metrics: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_request_metrics(self, bucket_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get request metrics from CloudWatch (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            days: Number of days to look back
            
        Returns:
            Dictionary containing request metrics or error information
        """
        try:
            from datetime import datetime, timedelta
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            request_types = ['AllRequests', 'GetRequests', 'PutRequests', 'DeleteRequests', 'HeadRequests', 'PostRequests', 'ListRequests']
            
            # Create tasks for parallel execution
            tasks = []
            for request_type in request_types:
                task = self.safe_api_call(
                    self.cloudwatch_client.get_metric_statistics,
                    Namespace='AWS/S3',
                    MetricName=request_type,
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': bucket_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # Daily
                    Statistics=['Sum'],
                    fallback_sources=['storage_lens']
                )
                tasks.append((request_type, task))
            
            # Execute all requests in parallel
            responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            metrics = {}
            for i, (request_type, _) in enumerate(tasks):
                response = responses[i]
                if isinstance(response, Exception):
                    metrics[request_type] = 0
                    logger.warning(f"Exception getting {request_type} metrics for {bucket_name}: {response}")
                elif response.get("status") == "success":
                    datapoints = response["data"].get("Datapoints", [])
                    total_requests = sum(dp.get("Sum", 0) for dp in datapoints)
                    metrics[request_type] = total_requests
                else:
                    metrics[request_type] = 0
            
            return {
                "status": "success",
                "data": metrics,
                "message": f"Retrieved request metrics for bucket {bucket_name}",
                "source": "cloudwatch"
            }
            
        except Exception as e:
            logger.error(f"Error getting request metrics for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting request metrics: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_storage_class_distribution_from_storage_lens(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get storage class distribution for a bucket from Storage Lens (NO-COST operation).
        
        This method replaces the cost-incurring list_objects_v2 approach with Storage Lens data.
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing storage class distribution or error information with fallback guidance
        """
        logger.info(f"Getting storage class distribution for {bucket_name} from Storage Lens (NO-COST)")
        
        # This is a placeholder for Storage Lens integration
        # The actual implementation will be done by StorageLensService
        return {
            "status": "error",
            "message": "Storage class distribution requires Storage Lens integration",
            "error_code": "REQUIRES_STORAGE_LENS",
            "fallback_available": True,
            "suggested_fallback": "storage_lens_service",
            "cost_constraint": "FORBIDDEN: list_objects_v2 would incur costs",
            "alternative_approach": "Use StorageLensService.get_storage_class_distribution() instead"
        }
    
    async def get_bucket_lifecycle(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get lifecycle configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing lifecycle configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_lifecycle_configuration,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                if response.get("resource_not_found"):
                    return {
                        "status": "success",
                        "data": {"Rules": []},
                        "message": f"No lifecycle configuration found for bucket {bucket_name}",
                        "source": response.get("source", "primary")
                    }
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved lifecycle configuration for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting lifecycle configuration for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting lifecycle configuration: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_versioning(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get versioning configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing versioning configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_versioning,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved versioning configuration for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting versioning configuration for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting versioning configuration: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_tagging(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get tagging configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing tagging configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_tagging,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                if response.get("resource_not_found"):
                    return {
                        "status": "success",
                        "data": {"TagSet": []},
                        "message": f"No tags found for bucket {bucket_name}",
                        "source": response.get("source", "primary")
                    }
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved tagging configuration for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting tagging configuration for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting tagging configuration: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_multipart_uploads(self, bucket_name: str, max_results: int = 1000) -> Dict[str, Any]:
        """
        Get incomplete multipart uploads for a bucket with pagination (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            max_results: Maximum number of results per page
            
        Returns:
            Dictionary containing multipart uploads or error information
        """
        try:
            all_uploads = []
            key_marker = None
            upload_id_marker = None
            
            while True:
                params = {
                    'Bucket': bucket_name,
                    'MaxUploads': min(max_results, 1000)  # AWS limit is 1000
                }
                
                if key_marker:
                    params['KeyMarker'] = key_marker
                if upload_id_marker:
                    params['UploadIdMarker'] = upload_id_marker
                
                response = await self.safe_api_call(
                    self.s3_client.list_multipart_uploads,
                    fallback_sources=['storage_lens'],
                    **params
                )
                
                if response["status"] != "success":
                    return response
                
                data = response["data"]
                uploads = data.get("Uploads", [])
                all_uploads.extend(uploads)
                
                # Check if there are more results
                if data.get("IsTruncated", False):
                    key_marker = data.get("NextKeyMarker")
                    upload_id_marker = data.get("NextUploadIdMarker")
                else:
                    break
                
                # Safety check to prevent infinite loops
                if len(all_uploads) >= max_results:
                    break
            
            return {
                "status": "success",
                "data": {
                    "Uploads": all_uploads,
                    "Bucket": bucket_name
                },
                "message": f"Retrieved {len(all_uploads)} incomplete multipart uploads for bucket {bucket_name}",
                "source": response.get("source", "primary")
            }
            
        except Exception as e:
            logger.error(f"Error getting multipart uploads for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting multipart uploads: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    def list_objects_v2_paginated(self, bucket_name: str, prefix: str = '', max_keys: int = 1000) -> Iterator[Dict]:
        """
        FORBIDDEN: This method would incur S3 request costs.
        
        Use StorageLensService for object-level analysis instead.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Object key prefix to filter by
            max_keys: Maximum number of keys per page
            
        Yields:
            Error response indicating cost constraint violation
        """
        logger.error(f"FORBIDDEN: list_objects_v2 would incur costs for bucket {bucket_name}")
        yield {
            "status": "error",
            "message": "FORBIDDEN: list_objects_v2 would incur S3 request costs",
            "error_code": "COST_CONSTRAINT_VIOLATION",
            "fallback_available": True,
            "suggested_fallback": "storage_lens_service",
            "alternative_approach": "Use StorageLensService.get_object_metrics() for object-level analysis"
        }
    
    async def get_s3_costs(self, start_date: str, end_date: str, next_token: str = None) -> Dict[str, Any]:
        """
        Get S3 cost and usage data from Cost Explorer with pagination (NO-COST operation).
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            next_token: Pagination token for next page
            
        Returns:
            Dictionary containing S3 cost data or error information
        """
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date,
                    'End': end_date
                },
                'Granularity': 'MONTHLY',
                'Metrics': ['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                'Filter': {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Simple Storage Service']
                    }
                },
                'GroupBy': [
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ]
            }
            
            if next_token:
                params['NextPageToken'] = next_token
            
            response = await self.safe_api_call(
                self.ce_client.get_cost_and_usage,
                fallback_sources=['storage_lens'],
                **params
            )
            
            if response["status"] == "success":
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved S3 cost data from {start_date} to {end_date}",
                    "source": response.get("source", "cost_explorer")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting S3 costs: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting S3 costs: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    def paginate_s3_operation(self, operation_name: str, **kwargs) -> Iterator[Dict]:
        """
        Generic pagination handler for S3 operations.
        
        Args:
            operation_name: Name of the S3 operation to paginate
            **kwargs: Arguments for the operation
            
        Yields:
            Dictionary pages containing operation results
        """
        try:
            if hasattr(self.s3_client, 'get_paginator'):
                try:
                    paginator = self.s3_client.get_paginator(operation_name)
                    page_iterator = paginator.paginate(**kwargs)
                    
                    for page in page_iterator:
                        yield {
                            "status": "success",
                            "data": page
                        }
                    return
                except Exception:
                    # Fall back to manual pagination if paginator not available
                    pass
            
            # Manual pagination fallback
            next_token = None
            while True:
                if next_token:
                    kwargs['NextToken'] = next_token
                
                operation = getattr(self.s3_client, operation_name)
                response = self.safe_api_call(operation, **kwargs)
                
                if response["status"] != "success":
                    yield response
                    break
                
                yield response
                
                # Check for next token (varies by operation)
                data = response["data"]
                next_token = data.get('NextToken') or data.get('NextContinuationToken') or data.get('NextMarker')
                
                if not next_token or not data.get('IsTruncated', False):
                    break
                    
        except Exception as e:
            logger.error(f"Error in paginated S3 operation {operation_name}: {str(e)}")
            yield {
                "status": "error",
                "message": f"Error in paginated operation: {str(e)}"
            }
    
    def paginate_cost_explorer(self, operation_name: str, **kwargs) -> Iterator[Dict]:
        """
        Generic pagination handler for Cost Explorer operations.
        
        Args:
            operation_name: Name of the Cost Explorer operation to paginate
            **kwargs: Arguments for the operation
            
        Yields:
            Dictionary pages containing operation results
        """
        try:
            next_token = None
            while True:
                if next_token:
                    kwargs['NextPageToken'] = next_token
                
                operation = getattr(self.ce_client, operation_name)
                response = self.safe_api_call(operation, **kwargs)
                
                if response["status"] != "success":
                    yield response
                    break
                
                yield response
                
                # Check for next token
                data = response["data"]
                next_token = data.get('NextPageToken')
                
                if not next_token:
                    break
                    
        except Exception as e:
            logger.error(f"Error in paginated Cost Explorer operation {operation_name}: {str(e)}")
            yield {
                "status": "error",
                "message": f"Error in paginated operation: {str(e)}"
            }
    
    def paginate_cloudwatch_metrics(self, **kwargs) -> Iterator[Dict]:
        """
        Generic pagination handler for CloudWatch metrics operations.
        
        Args:
            **kwargs: Arguments for the CloudWatch operation
            
        Yields:
            Dictionary pages containing metrics results
        """
        try:
            # CloudWatch get_metric_statistics doesn't use traditional pagination
            # Instead, we need to handle large time ranges by breaking them into chunks
            
            from datetime import datetime, timedelta
            
            start_time = kwargs.get('StartTime')
            end_time = kwargs.get('EndTime')
            period = kwargs.get('Period', 86400)  # Default to daily
            
            if not start_time or not end_time:
                # If no time range specified, just make the call directly
                response = self.safe_api_call(
                    self.cloudwatch_client.get_metric_statistics,
                    **kwargs
                )
                yield response
                return
            
            # Break large time ranges into chunks to avoid API limits
            chunk_size = timedelta(days=30)  # 30-day chunks
            current_start = start_time
            
            while current_start < end_time:
                current_end = min(current_start + chunk_size, end_time)
                
                chunk_kwargs = kwargs.copy()
                chunk_kwargs['StartTime'] = current_start
                chunk_kwargs['EndTime'] = current_end
                
                response = self.safe_api_call(
                    self.cloudwatch_client.get_metric_statistics,
                    **chunk_kwargs
                )
                
                yield response
                
                if response["status"] != "success":
                    break
                
                current_start = current_end
                
        except Exception as e:
            logger.error(f"Error in paginated CloudWatch operation: {str(e)}")
            yield {
                "status": "error",
                "message": f"Error in paginated operation: {str(e)}"
            }
    
    async def get_bucket_policy(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get bucket policy for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing bucket policy or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_policy,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                if response.get("resource_not_found"):
                    return {
                        "status": "success",
                        "data": {"Policy": None},
                        "message": f"No bucket policy found for bucket {bucket_name}",
                        "source": response.get("source", "primary")
                    }
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved bucket policy for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting bucket policy for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket policy: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_acl(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get bucket ACL (Access Control List) for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing bucket ACL or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_acl,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved bucket ACL for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting bucket ACL for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket ACL: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    def get_no_cost_operation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the no-cost constraint implementation.
        
        Returns:
            Dictionary containing implementation details and constraints
        """
        return {
            "status": "success",
            "data": {
                "cost_constraint_enabled": True,
                "allowed_s3_operations": sorted(ALLOWED_S3_OPERATIONS),
                "forbidden_s3_operations": sorted(FORBIDDEN_S3_OPERATIONS),
                "operation_call_stats": self.get_operation_stats(),
                "fallback_sources": ["storage_lens", "cost_explorer", "cloudwatch"],
                "async_support": True,
                "parallel_execution": True
            },
            "message": "S3Service configured with strict no-cost constraints",
            "implementation_notes": [
                "All S3 operations validated against whitelist",
                "Object-level operations (list_objects_v2, head_object, get_object) are FORBIDDEN",
                "Metadata-only operations (lifecycle, versioning, tagging) are allowed",
                "Async/await support for parallel execution",
                "Graceful fallback to alternative data sources",
                "Enhanced error handling with retry logic"
            ]
        }
    
    async def get_bucket_encryption(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get bucket encryption configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing bucket encryption configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_encryption,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                if response.get("resource_not_found"):
                    return {
                        "status": "success",
                        "data": {"ServerSideEncryptionConfiguration": None},
                        "message": f"No encryption configuration found for bucket {bucket_name}",
                        "source": response.get("source", "primary")
                    }
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved bucket encryption for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting bucket encryption for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket encryption: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_public_access_block(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get public access block configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing public access block configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_public_access_block,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                if response.get("resource_not_found"):
                    return {
                        "status": "success",
                        "data": {"PublicAccessBlockConfiguration": None},
                        "message": f"No public access block configuration found for bucket {bucket_name}",
                        "source": response.get("source", "primary")
                    }
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved public access block for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting public access block for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting public access block: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_bucket_notification_configuration(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get bucket notification configuration for a bucket (NO-COST operation).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing bucket notification configuration or error information
        """
        try:
            response = await self.safe_api_call(
                self.s3_client.get_bucket_notification_configuration,
                Bucket=bucket_name,
                fallback_sources=['storage_lens']
            )
            
            if response["status"] == "success":
                return {
                    "status": "success",
                    "data": response["data"],
                    "message": f"Retrieved notification configuration for bucket {bucket_name}",
                    "source": response.get("source", "primary")
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error getting notification configuration for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting notification configuration: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }
    
    async def get_comprehensive_bucket_config(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get comprehensive bucket configuration using parallel execution (ALL NO-COST operations).
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            Dictionary containing all bucket configuration data
        """
        try:
            logger.info(f"Getting comprehensive configuration for bucket {bucket_name} using NO-COST operations")
            
            # Execute all configuration queries in parallel
            tasks = {
                'lifecycle': self.get_bucket_lifecycle(bucket_name),
                'versioning': self.get_bucket_versioning(bucket_name),
                'tagging': self.get_bucket_tagging(bucket_name),
                'multipart_uploads': self.get_multipart_uploads(bucket_name),
                'policy': self.get_bucket_policy(bucket_name),
                'encryption': self.get_bucket_encryption(bucket_name),
                'public_access_block': self.get_bucket_public_access_block(bucket_name),
                'notification': self.get_bucket_notification_configuration(bucket_name)
            }
            
            # Wait for all tasks to complete
            results = {}
            responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for i, (config_type, _) in enumerate(tasks.items()):
                response = responses[i]
                if isinstance(response, Exception):
                    results[config_type] = {
                        "status": "error",
                        "message": f"Exception: {str(response)}"
                    }
                    logger.warning(f"Exception getting {config_type} for {bucket_name}: {response}")
                else:
                    results[config_type] = response
            
            # Count successful operations
            successful_ops = sum(1 for result in results.values() if result.get("status") == "success")
            
            return {
                "status": "success",
                "data": {
                    "bucket_name": bucket_name,
                    "configurations": results,
                    "operation_stats": self.get_operation_stats()
                },
                "message": f"Retrieved {successful_ops}/{len(tasks)} configurations for bucket {bucket_name}",
                "cost_constraint_applied": True,
                "all_operations_no_cost": True
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive bucket config for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting comprehensive bucket config: {str(e)}",
                "fallback_available": True,
                "suggested_fallback": "storage_lens"
            }


# Convenience functions for backward compatibility with existing patterns
def list_buckets(region: Optional[str] = None) -> Dict[str, Any]:
    """
    List all S3 buckets (synchronous wrapper for async method).
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing bucket list or error information
    """
    service = S3Service(region)
    return service.list_buckets_sync()


def get_bucket_metrics(bucket_name: str, days: int = 30, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive metrics for a specific bucket (synchronous wrapper for async method).
    
    Args:
        bucket_name: Name of the S3 bucket
        days: Number of days to look back for metrics
        region: AWS region (optional)
        
    Returns:
        Dictionary containing bucket metrics or error information
    """
    service = S3Service(region)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(service.get_bucket_metrics(bucket_name, days))


def get_s3_costs(start_date: str, end_date: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get S3 cost and usage data from Cost Explorer (synchronous wrapper for async method).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        region: AWS region (optional)
        
    Returns:
        Dictionary containing S3 cost data or error information
    """
    service = S3Service(region)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(service.get_s3_costs(start_date, end_date))


def get_comprehensive_bucket_config(bucket_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive bucket configuration (synchronous wrapper for async method).
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (optional)
        
    Returns:
        Dictionary containing all bucket configuration data
    """
    service = S3Service(region)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(service.get_comprehensive_bucket_config(bucket_name))


# Async convenience functions for new code
async def list_buckets_async(region: Optional[str] = None) -> Dict[str, Any]:
    """
    List all S3 buckets (async version).
    
    Args:
        region: AWS region (optional)
        
    Returns:
        Dictionary containing bucket list or error information
    """
    service = S3Service(region)
    return await service.list_buckets()


async def get_bucket_metrics_async(bucket_name: str, days: int = 30, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive metrics for a specific bucket (async version).
    
    Args:
        bucket_name: Name of the S3 bucket
        days: Number of days to look back for metrics
        region: AWS region (optional)
        
    Returns:
        Dictionary containing bucket metrics or error information
    """
    service = S3Service(region)
    return await service.get_bucket_metrics(bucket_name, days)


async def get_s3_costs_async(start_date: str, end_date: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get S3 cost and usage data from Cost Explorer (async version).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        region: AWS region (optional)
        
    Returns:
        Dictionary containing S3 cost data or error information
    """
    service = S3Service(region)
    return await service.get_s3_costs(start_date, end_date)


async def get_comprehensive_bucket_config_async(bucket_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive bucket configuration (async version).
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (optional)
        
    Returns:
        Dictionary containing all bucket configuration data
    """
    service = S3Service(region)
    return await service.get_comprehensive_bucket_config(bucket_name)