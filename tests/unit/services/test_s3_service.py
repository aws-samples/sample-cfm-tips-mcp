"""
Unit tests for S3Service class with strict no-cost constraint validation.

Tests all S3Service functionality while ensuring no cost-incurring operations
are performed. Includes comprehensive validation of the cost constraint system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

from services.s3_service import S3Service, S3CostConstraintViolationError, ALLOWED_S3_OPERATIONS, FORBIDDEN_S3_OPERATIONS


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestS3ServiceCostConstraints:
    """Test cost constraint validation in S3Service."""
    
    def test_allowed_operations_whitelist(self):
        """Test that allowed operations are properly whitelisted."""
        expected_allowed = {
            'list_buckets', 'get_bucket_location', 'get_bucket_lifecycle_configuration',
            'get_bucket_versioning', 'get_bucket_tagging', 'list_multipart_uploads',
            'get_bucket_policy', 'get_bucket_acl', 'get_bucket_encryption',
            'get_public_access_block', 'get_bucket_notification_configuration',
            'get_bucket_cors', 'get_bucket_website', 'get_bucket_logging',
            'get_bucket_replication', 'get_bucket_request_payment'
        }
        
        assert ALLOWED_S3_OPERATIONS == expected_allowed
    
    def test_forbidden_operations_list(self):
        """Test that forbidden operations are properly identified."""
        expected_forbidden = {
            'list_objects', 'list_objects_v2', 'head_object', 'get_object',
            'get_object_attributes', 'select_object_content', 'restore_object'
        }
        
        assert FORBIDDEN_S3_OPERATIONS == expected_forbidden
    
    def test_validate_allowed_operation(self, mock_aws_credentials):
        """Test validation of allowed operations."""
        service = S3Service(region="us-east-1")
        
        # Should not raise exception
        service._validate_s3_operation('list_buckets')
        service._validate_s3_operation('get_bucket_location')
        service._validate_s3_operation('get_bucket_lifecycle_configuration')
        
        # Check operation tracking
        assert service._operation_call_count['list_buckets'] == 1
        assert service._operation_call_count['get_bucket_location'] == 1
    
    def test_validate_forbidden_operation(self, mock_aws_credentials):
        """Test validation rejects forbidden operations."""
        service = S3Service(region="us-east-1")
        
        forbidden_ops = ['list_objects', 'list_objects_v2', 'head_object', 'get_object']
        
        for op in forbidden_ops:
            with pytest.raises(S3CostConstraintViolationError) as exc_info:
                service._validate_s3_operation(op)
            
            assert f"FORBIDDEN: S3 operation '{op}' would incur costs" in str(exc_info.value)
    
    def test_validate_unknown_operation_warning(self, mock_aws_credentials, caplog):
        """Test that unknown operations generate warnings but don't fail."""
        service = S3Service(region="us-east-1")
        
        # Should not raise exception but should log warning
        service._validate_s3_operation('unknown_operation')
        
        assert "not in whitelist" in caplog.text
        assert service._operation_call_count['unknown_operation'] == 1
    
    def test_operation_stats_tracking(self, mock_aws_credentials):
        """Test operation statistics tracking."""
        service = S3Service(region="us-east-1")
        
        # Call operations multiple times
        service._validate_s3_operation('list_buckets')
        service._validate_s3_operation('list_buckets')
        service._validate_s3_operation('get_bucket_location')
        
        stats = service.get_operation_stats()
        
        assert stats['list_buckets'] == 2
        assert stats['get_bucket_location'] == 1
        assert len(stats) == 2


@pytest.mark.unit
class TestS3ServiceInitialization:
    """Test S3Service initialization and client setup."""
    
    def test_initialization_with_region(self, mock_aws_credentials):
        """Test service initialization with specific region."""
        service = S3Service(region="us-west-2")
        
        assert service.region == "us-west-2"
        assert service._operation_call_count == {}
        assert hasattr(service, 's3_client')
        assert hasattr(service, 'cloudwatch_client')
        assert hasattr(service, 'ce_client')
    
    def test_initialization_without_region(self, mock_aws_credentials):
        """Test service initialization without region."""
        service = S3Service()
        
        assert service.region is None
        assert hasattr(service, 's3_client')
    
    def test_initialization_failure(self):
        """Test handling of initialization failures."""
        with patch('boto3.client', side_effect=Exception("AWS client error")):
            with pytest.raises(Exception) as exc_info:
                S3Service(region="us-east-1")
            
            assert "AWS client error" in str(exc_info.value)


@pytest.mark.unit
class TestS3ServiceSafeApiCall:
    """Test safe API call wrapper functionality."""
    
    @pytest.mark.asyncio
    async def test_safe_api_call_success(self, mock_aws_credentials):
        """Test successful API call."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock(return_value={"Buckets": []})
        mock_func.__name__ = "list_buckets"
        
        result = await service.safe_api_call(mock_func)
        
        assert result["status"] == "success"
        assert result["data"] == {"Buckets": []}
        assert result["source"] == "primary"
    
    @pytest.mark.asyncio
    async def test_safe_api_call_cost_constraint_violation(self, mock_aws_credentials):
        """Test API call with cost constraint violation."""
        service = S3Service(region="us-east-1")
        
        # Mock S3 client method that would incur costs
        mock_s3_client = Mock()
        mock_s3_client._service_model.service_name = 's3'
        
        mock_func = Mock()
        mock_func.__name__ = "list_objects_v2"
        mock_func.__self__ = mock_s3_client
        
        result = await service.safe_api_call(mock_func, fallback_sources=['storage_lens'])
        
        assert result["status"] == "error"
        assert result["error_code"] == "COST_CONSTRAINT_VIOLATION"
        assert result["fallback_available"] is True
        assert "FORBIDDEN" in result["message"]
    
    @pytest.mark.asyncio
    async def test_safe_api_call_access_denied(self, mock_aws_credentials):
        """Test API call with access denied error."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock()
        mock_func.__name__ = "list_buckets"
        mock_func.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='ListBuckets'
        )
        
        result = await service.safe_api_call(mock_func, fallback_sources=['storage_lens'])
        
        assert result["status"] == "error"
        assert result["error_code"] == "AccessDenied"
        assert result["fallback_available"] is True
        assert result["suggested_fallback"] == "storage_lens"
    
    @pytest.mark.asyncio
    async def test_safe_api_call_throttling_with_retry(self, mock_aws_credentials):
        """Test API call with throttling and retry logic."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock()
        mock_func.__name__ = "list_buckets"
        
        # First call fails with throttling, second succeeds
        mock_func.side_effect = [
            ClientError(
                error_response={'Error': {'Code': 'Throttling', 'Message': 'Rate exceeded'}},
                operation_name='ListBuckets'
            ),
            {"Buckets": []}
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.safe_api_call(mock_func, max_retries=3)
        
        assert result["status"] == "success"
        assert result["data"] == {"Buckets": []}
        assert mock_func.call_count == 2
    
    @pytest.mark.asyncio
    async def test_safe_api_call_max_retries_exceeded(self, mock_aws_credentials):
        """Test API call with max retries exceeded."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock()
        mock_func.__name__ = "list_buckets"
        mock_func.side_effect = ClientError(
            error_response={'Error': {'Code': 'Throttling', 'Message': 'Rate exceeded'}},
            operation_name='ListBuckets'
        )
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await service.safe_api_call(mock_func, max_retries=2)
        
        assert result["status"] == "error"
        assert result["error_code"] == "Throttling"
        assert "Max retries exceeded" in result["message"]
        assert mock_func.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_safe_api_call_resource_not_found(self, mock_aws_credentials):
        """Test API call with resource not found (expected error)."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock()
        mock_func.__name__ = "get_bucket_lifecycle_configuration"
        mock_func.side_effect = ClientError(
            error_response={'Error': {'Code': 'NoSuchLifecycleConfiguration', 'Message': 'Not found'}},
            operation_name='GetBucketLifecycleConfiguration'
        )
        
        result = await service.safe_api_call(mock_func)
        
        assert result["status"] == "success"  # Not found is treated as success
        assert result["data"] == {}
        assert result["resource_not_found"] is True
    
    def test_safe_api_call_sync(self, mock_aws_credentials):
        """Test synchronous version of safe API call."""
        service = S3Service(region="us-east-1")
        
        mock_func = Mock(return_value={"Buckets": []})
        mock_func.__name__ = "list_buckets"
        
        result = service.safe_api_call_sync(mock_func)
        
        assert result["status"] == "success"
        assert result["data"] == {"Buckets": []}


@pytest.mark.unit
class TestS3ServiceBucketOperations:
    """Test S3Service bucket-level operations (all no-cost)."""
    
    @pytest.mark.asyncio
    async def test_list_buckets_success(self, mock_aws_credentials):
        """Test successful bucket listing with region information."""
        service = S3Service(region="us-east-1")
        
        # Mock list_buckets response
        mock_buckets_response = {
            "Buckets": [
                {"Name": "test-bucket-1", "CreationDate": datetime.now()},
                {"Name": "test-bucket-2", "CreationDate": datetime.now()}
            ],
            "Owner": {"ID": "owner-id"}
        }
        
        # Mock get_bucket_location responses
        mock_location_responses = [
            {"LocationConstraint": "us-west-2"},
            {"LocationConstraint": None}  # us-east-1 returns None
        ]
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            # Setup mock responses
            mock_safe_call.side_effect = [
                {"status": "success", "data": mock_buckets_response},  # list_buckets
                {"status": "success", "data": mock_location_responses[0]},  # get_bucket_location 1
                {"status": "success", "data": mock_location_responses[1]}   # get_bucket_location 2
            ]
            
            result = await service.list_buckets()
        
        assert result["status"] == "success"
        assert len(result["data"]["Buckets"]) == 2
        assert result["data"]["Buckets"][0]["Region"] == "us-west-2"
        assert result["data"]["Buckets"][1]["Region"] == "us-east-1"  # None becomes us-east-1
        assert "Retrieved 2 buckets with regions" in result["message"]
    
    @pytest.mark.asyncio
    async def test_list_buckets_region_failure(self, mock_aws_credentials):
        """Test bucket listing with region lookup failures."""
        service = S3Service(region="us-east-1")
        
        mock_buckets_response = {
            "Buckets": [{"Name": "test-bucket", "CreationDate": datetime.now()}],
            "Owner": {"ID": "owner-id"}
        }
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.side_effect = [
                {"status": "success", "data": mock_buckets_response},  # list_buckets
                {"status": "error", "message": "Access denied"}       # get_bucket_location
            ]
            
            result = await service.list_buckets()
        
        assert result["status"] == "success"
        assert result["data"]["Buckets"][0]["Region"] == "unknown"
    
    def test_list_buckets_sync(self, mock_aws_credentials):
        """Test synchronous bucket listing."""
        service = S3Service(region="us-east-1")
        
        with patch.object(service, 'list_buckets') as mock_async_list:
            mock_async_list.return_value = {"status": "success", "data": {"Buckets": []}}
            
            result = service.list_buckets_sync()
        
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_bucket_lifecycle(self, mock_aws_credentials):
        """Test getting bucket lifecycle configuration."""
        service = S3Service(region="us-east-1")
        
        mock_lifecycle = {
            "Rules": [
                {
                    "ID": "rule1",
                    "Status": "Enabled",
                    "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}]
                }
            ]
        }
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.return_value = {"status": "success", "data": mock_lifecycle}
            
            result = await service.get_bucket_lifecycle("test-bucket")
        
        assert result["status"] == "success"
        assert result["data"] == mock_lifecycle
        assert "Retrieved lifecycle configuration" in result["message"]
    
    @pytest.mark.asyncio
    async def test_get_bucket_lifecycle_not_found(self, mock_aws_credentials):
        """Test getting lifecycle configuration when none exists."""
        service = S3Service(region="us-east-1")
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.return_value = {
                "status": "success", 
                "data": {}, 
                "resource_not_found": True
            }
            
            result = await service.get_bucket_lifecycle("test-bucket")
        
        assert result["status"] == "success"
        assert result["data"] == {"Rules": []}
        assert "No lifecycle configuration found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_get_bucket_versioning(self, mock_aws_credentials):
        """Test getting bucket versioning configuration."""
        service = S3Service(region="us-east-1")
        
        mock_versioning = {"Status": "Enabled", "MfaDelete": "Disabled"}
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.return_value = {"status": "success", "data": mock_versioning}
            
            result = await service.get_bucket_versioning("test-bucket")
        
        assert result["status"] == "success"
        assert result["data"] == mock_versioning
        assert "Retrieved versioning configuration" in result["message"]
    
    @pytest.mark.asyncio
    async def test_get_bucket_tagging(self, mock_aws_credentials):
        """Test getting bucket tagging configuration."""
        service = S3Service(region="us-east-1")
        
        mock_tags = {
            "TagSet": [
                {"Key": "Environment", "Value": "Production"},
                {"Key": "Owner", "Value": "DataTeam"}
            ]
        }
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.return_value = {"status": "success", "data": mock_tags}
            
            result = await service.get_bucket_tagging("test-bucket")
        
        assert result["status"] == "success"
        assert result["data"] == mock_tags


@pytest.mark.unit
class TestS3ServiceMetrics:
    """Test S3Service metrics operations (no-cost CloudWatch only)."""
    
    @pytest.mark.asyncio
    async def test_get_bucket_metrics_success(self, mock_aws_credentials):
        """Test successful bucket metrics retrieval."""
        service = S3Service(region="us-east-1")
        
        # Mock the individual metric methods
        with patch.object(service, 'get_bucket_size_metrics') as mock_size, \
             patch.object(service, 'get_request_metrics') as mock_requests, \
             patch.object(service, 'get_storage_class_distribution_from_storage_lens') as mock_storage_class:
            
            mock_size.return_value = {
                "status": "success",
                "data": {"SizeBytes": 1000000, "ObjectCount": 100}
            }
            mock_requests.return_value = {
                "status": "success", 
                "data": {"AllRequests": 5000, "GetRequests": 4000}
            }
            mock_storage_class.return_value = {
                "status": "error",
                "suggested_fallback": "storage_lens_service"
            }
            
            result = await service.get_bucket_metrics("test-bucket", days=30)
        
        assert result["status"] == "success"
        assert result["data"]["SizeBytes"] == 1000000
        assert result["data"]["AllRequests"] == 5000
        assert result["data"]["StorageClassDistribution"]["fallback_required"] is True
        assert result["cost_constraint_applied"] is True
    
    @pytest.mark.asyncio
    async def test_get_bucket_size_metrics(self, mock_aws_credentials):
        """Test bucket size metrics from CloudWatch."""
        service = S3Service(region="us-east-1")
        
        mock_size_response = {
            "Datapoints": [
                {"Timestamp": datetime.now(), "Average": 1000000000}  # 1GB
            ]
        }
        mock_objects_response = {
            "Datapoints": [
                {"Timestamp": datetime.now(), "Average": 1000}
            ]
        }
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.side_effect = [
                {"status": "success", "data": mock_size_response},
                {"status": "success", "data": mock_objects_response}
            ]
            
            result = await service.get_bucket_size_metrics("test-bucket", days=30)
        
        assert result["status"] == "success"
        assert result["data"]["SizeBytes"] == 1000000000
        assert result["data"]["SizeGB"] == 1.0
        assert result["data"]["ObjectCount"] == 1000
        assert result["source"] == "cloudwatch"
    
    @pytest.mark.asyncio
    async def test_get_request_metrics(self, mock_aws_credentials):
        """Test request metrics from CloudWatch."""
        service = S3Service(region="us-east-1")
        
        # Mock responses for different request types
        mock_responses = [
            {"status": "success", "data": {"Datapoints": [{"Sum": 5000}]}},  # AllRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 4000}]}},  # GetRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 1000}]}},  # PutRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 100}]}},   # DeleteRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 500}]}},   # HeadRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 200}]}},   # PostRequests
            {"status": "success", "data": {"Datapoints": [{"Sum": 50}]}}     # ListRequests
        ]
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.side_effect = mock_responses
            
            result = await service.get_request_metrics("test-bucket", days=30)
        
        assert result["status"] == "success"
        assert result["data"]["AllRequests"] == 5000
        assert result["data"]["GetRequests"] == 4000
        assert result["data"]["PutRequests"] == 1000
        assert result["source"] == "cloudwatch"
    
    @pytest.mark.asyncio
    async def test_get_storage_class_distribution_cost_constraint(self, mock_aws_credentials):
        """Test that storage class distribution enforces cost constraints."""
        service = S3Service(region="us-east-1")
        
        result = await service.get_storage_class_distribution_from_storage_lens("test-bucket")
        
        assert result["status"] == "error"
        assert result["error_code"] == "REQUIRES_STORAGE_LENS"
        assert result["cost_constraint"] == "FORBIDDEN: list_objects_v2 would incur costs"
        assert result["suggested_fallback"] == "storage_lens_service"
        assert result["fallback_available"] is True


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestS3ServiceCostConstraintIntegration:
    """Integration tests for cost constraint validation."""
    
    @pytest.mark.asyncio
    async def test_no_forbidden_operations_in_bucket_listing(self, mock_aws_credentials, 
                                                           cost_constraint_validator):
        """Test that bucket listing uses only allowed operations."""
        service = S3Service(region="us-east-1")
        
        # Patch the validation method to track operations
        original_validate = service._validate_s3_operation
        def tracking_validate(op_name):
            cost_constraint_validator.validate_operation(op_name)
            return original_validate(op_name)
        
        service._validate_s3_operation = tracking_validate
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.side_effect = [
                {"status": "success", "data": {"Buckets": [{"Name": "test", "CreationDate": datetime.now()}]}},
                {"status": "success", "data": {"LocationConstraint": "us-west-2"}}
            ]
            
            await service.list_buckets()
        
        summary = cost_constraint_validator.get_operation_summary()
        assert len(summary["forbidden_called"]) == 0
        assert "list_buckets" in summary["allowed_called"]
        assert "get_bucket_location" in summary["allowed_called"]
    
    @pytest.mark.asyncio
    async def test_no_forbidden_operations_in_metrics(self, mock_aws_credentials,
                                                    cost_constraint_validator):
        """Test that metrics collection uses only allowed operations."""
        service = S3Service(region="us-east-1")
        
        # Track all operations
        original_validate = service._validate_s3_operation
        def tracking_validate(op_name):
            cost_constraint_validator.validate_operation(op_name)
            return original_validate(op_name)
        
        service._validate_s3_operation = tracking_validate
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.return_value = {"status": "success", "data": {"Datapoints": []}}
            
            await service.get_bucket_size_metrics("test-bucket")
            await service.get_request_metrics("test-bucket")
        
        summary = cost_constraint_validator.get_operation_summary()
        assert len(summary["forbidden_called"]) == 0
        # CloudWatch operations are not S3 operations, so no S3 operations should be tracked
        assert len(summary["operations"]) == 0
    
    def test_cost_constraint_error_details(self, mock_aws_credentials):
        """Test detailed error information for cost constraint violations."""
        service = S3Service(region="us-east-1")
        
        with pytest.raises(S3CostConstraintViolationError) as exc_info:
            service._validate_s3_operation('list_objects_v2')
        
        error_msg = str(exc_info.value)
        assert "FORBIDDEN" in error_msg
        assert "list_objects_v2" in error_msg
        assert "would incur costs" in error_msg
        assert "no-cost operations" in error_msg
        # Should list allowed operations
        assert "list_buckets" in error_msg


@pytest.mark.unit
class TestS3ServicePerformance:
    """Performance tests for S3Service operations."""
    
    @pytest.mark.asyncio
    async def test_parallel_bucket_region_lookup(self, mock_aws_credentials, performance_tracker):
        """Test that bucket region lookups are performed in parallel."""
        service = S3Service(region="us-east-1")
        
        # Mock 5 buckets
        mock_buckets = [
            {"Name": f"bucket-{i}", "CreationDate": datetime.now()}
            for i in range(5)
        ]
        
        async def slow_region_lookup(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API delay
            return {"status": "success", "data": {"LocationConstraint": "us-west-2"}}
        
        with patch.object(service, 'safe_api_call') as mock_safe_call:
            mock_safe_call.side_effect = [
                {"status": "success", "data": {"Buckets": mock_buckets}},  # list_buckets
                *[slow_region_lookup() for _ in range(5)]  # 5 region lookups
            ]
            
            performance_tracker.start_timer("bucket_listing")
            result = await service.list_buckets()
            execution_time = performance_tracker.end_timer("bucket_listing")
        
        assert result["status"] == "success"
        assert len(result["data"]["Buckets"]) == 5
        # With parallel execution, 5 operations taking 0.1s each should complete in ~0.1s, not 0.5s
        assert execution_time < 0.3  # Allow some overhead
    
    @pytest.mark.asyncio
    async def test_parallel_metrics_collection(self, mock_aws_credentials, performance_tracker):
        """Test that metrics are collected in parallel."""
        service = S3Service(region="us-east-1")
        
        async def slow_metric_call(*args, **kwargs):
            await asyncio.sleep(0.05)  # Small delay per metric
            return {"status": "success", "data": {"Datapoints": []}}
        
        with patch.object(service, 'get_bucket_size_metrics', side_effect=slow_metric_call), \
             patch.object(service, 'get_request_metrics', side_effect=slow_metric_call), \
             patch.object(service, 'get_storage_class_distribution_from_storage_lens', side_effect=slow_metric_call):
            
            performance_tracker.start_timer("metrics_collection")
            result = await service.get_bucket_metrics("test-bucket")
            execution_time = performance_tracker.end_timer("metrics_collection")
        
        assert result["status"] == "success"
        # 3 parallel operations at 0.05s each should complete in ~0.05s, not 0.15s
        assert execution_time < 0.1