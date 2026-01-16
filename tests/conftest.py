"""
Pytest configuration and fixtures for S3 optimization testing.

This module provides common fixtures and configuration for all tests,
including mocked AWS services, test data factories, and performance monitoring.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import boto3
from moto import mock_aws
import json

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    with patch.dict('os.environ', {
        'AWS_ACCESS_KEY_ID': 'testing',
        'AWS_SECRET_ACCESS_KEY': 'testing',
        'AWS_SECURITY_TOKEN': 'testing',
        'AWS_SESSION_TOKEN': 'testing',
        'AWS_DEFAULT_REGION': 'us-east-1'
    }):
        yield


@pytest.fixture
def mock_s3_client(mock_aws_credentials):
    """Mock S3 client with moto."""
    with mock_aws():
        yield boto3.client('s3', region_name='us-east-1')


@pytest.fixture
def mock_cloudwatch_client(mock_aws_credentials):
    """Mock CloudWatch client with moto."""
    with mock_aws():
        yield boto3.client('cloudwatch', region_name='us-east-1')


@pytest.fixture
def mock_ce_client(mock_aws_credentials):
    """Mock Cost Explorer client with moto."""
    with mock_aws():
        yield boto3.client('ce', region_name='us-east-1')


@pytest.fixture
def mock_s3control_client(mock_aws_credentials):
    """Mock S3 Control client with moto."""
    with mock_aws():
        yield boto3.client('s3control', region_name='us-east-1')


@pytest.fixture
def sample_buckets():
    """Sample bucket data for testing."""
    return [
        {
            "Name": "test-bucket-1",
            "CreationDate": datetime.now() - timedelta(days=30),
            "Region": "us-east-1"
        },
        {
            "Name": "test-bucket-2", 
            "CreationDate": datetime.now() - timedelta(days=60),
            "Region": "us-west-2"
        },
        {
            "Name": "test-bucket-3",
            "CreationDate": datetime.now() - timedelta(days=90),
            "Region": "eu-west-1"
        }
    ]


@pytest.fixture
def sample_cost_explorer_data():
    """Sample Cost Explorer response data."""
    return {
        "ResultsByTime": [
            {
                "TimePeriod": {
                    "Start": "2024-01-01",
                    "End": "2024-01-02"
                },
                "Groups": [
                    {
                        "Keys": ["S3-Storage-Standard"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": "10.50", "Unit": "USD"},
                            "UsageQuantity": {"Amount": "1000", "Unit": "GB-Mo"}
                        }
                    },
                    {
                        "Keys": ["S3-Storage-StandardIA"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": "5.25", "Unit": "USD"},
                            "UsageQuantity": {"Amount": "500", "Unit": "GB-Mo"}
                        }
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_storage_lens_config():
    """Sample Storage Lens configuration."""
    return {
        "StorageLensConfiguration": {
            "Id": "test-config",
            "AccountLevel": {
                "StorageMetrics": {"IsEnabled": True},
                "BucketLevel": {
                    "ActivityMetrics": {"IsEnabled": True},
                    "CostOptimizationMetrics": {"IsEnabled": True},
                    "DetailedStatusCodesMetrics": {"IsEnabled": False},
                    "AdvancedCostOptimizationMetrics": {"IsEnabled": False},
                    "AdvancedDataProtectionMetrics": {"IsEnabled": False}
                }
            },
            "IsEnabled": True,
            "DataExport": {
                "S3BucketDestination": {
                    "Bucket": "storage-lens-export-bucket",
                    "Prefix": "exports/",
                    "Format": "CSV"
                }
            }
        }
    }


@pytest.fixture
def mock_s3_service():
    """Mock S3Service instance."""
    service = Mock()
    service.region = "us-east-1"
    service._operation_call_count = {}
    
    # Mock async methods
    async def mock_list_buckets():
        return {
            "status": "success",
            "data": {
                "Buckets": [
                    {"Name": "test-bucket-1", "CreationDate": datetime.now(), "Region": "us-east-1"},
                    {"Name": "test-bucket-2", "CreationDate": datetime.now(), "Region": "us-west-2"}
                ]
            }
        }
    
    async def mock_get_bucket_metrics(bucket_name, days=30):
        return {
            "status": "success",
            "data": {
                "SizeBytes": 1000000000,
                "SizeGB": 1.0,
                "ObjectCount": 1000,
                "AllRequests": 5000,
                "GetRequests": 4000,
                "PutRequests": 1000
            }
        }
    
    service.list_buckets = mock_list_buckets
    service.get_bucket_metrics = mock_get_bucket_metrics
    service.get_operation_stats = Mock(return_value={"list_buckets": 1, "get_bucket_location": 2})
    
    return service


@pytest.fixture
def mock_storage_lens_service():
    """Mock StorageLensService instance."""
    service = Mock()
    service.account_id = "123456789012"
    service.region = "us-east-1"
    
    # Mock async methods
    async def mock_get_storage_metrics(config_id="default-account-dashboard"):
        return {
            "status": "success",
            "data": {
                "ConfigurationId": config_id,
                "AccountId": "123456789012",
                "IsEnabled": True,
                "CostOptimizationMetrics": True,
                "DetailedStatusCodesMetrics": False
            }
        }
    
    async def mock_get_cost_optimization_metrics(config_id="default-account-dashboard"):
        return {
            "status": "success",
            "data": {
                "ConfigurationId": config_id,
                "CostOptimizationMetricsEnabled": True,
                "MultipartUploadTrackingAvailable": True
            }
        }
    
    service.get_storage_metrics = mock_get_storage_metrics
    service.get_cost_optimization_metrics = mock_get_cost_optimization_metrics
    
    return service


@pytest.fixture
def mock_pricing_service():
    """Mock S3Pricing service instance."""
    service = Mock()
    service.region = "us-east-1"
    
    def mock_get_storage_pricing():
        return {
            "status": "success",
            "storage_pricing": {
                "STANDARD": 0.023,
                "STANDARD_IA": 0.0125,
                "ONEZONE_IA": 0.01,
                "GLACIER": 0.004,
                "DEEP_ARCHIVE": 0.00099
            }
        }
    
    def mock_estimate_request_costs(requests):
        total_cost = sum(count * 0.0004 for count in requests.values())
        return {
            "status": "success",
            "total_cost": total_cost,
            "breakdown": {req_type: count * 0.0004 for req_type, count in requests.items()}
        }
    
    service.get_storage_pricing = mock_get_storage_pricing
    service.estimate_request_costs = mock_estimate_request_costs
    
    return service


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor."""
    monitor = Mock()
    monitor.start_analysis_monitoring = Mock(return_value="test_session_123")
    monitor.end_analysis_monitoring = Mock()
    monitor.record_metric = Mock()
    monitor.record_cache_hit = Mock()
    monitor.record_cache_miss = Mock()
    return monitor


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager."""
    manager = Mock()
    manager.start_memory_tracking = Mock(return_value="test_tracker_123")
    manager.stop_memory_tracking = Mock(return_value={"peak_memory_mb": 50.0, "avg_memory_mb": 30.0})
    manager.register_large_object = Mock()
    manager.add_cache_reference = Mock()
    manager.set_performance_monitor = Mock()
    return manager


@pytest.fixture
def mock_timeout_handler():
    """Mock timeout handler."""
    handler = Mock()
    handler.get_timeout_for_analysis = Mock(return_value=60.0)
    handler.record_execution_time = Mock()
    handler.get_complexity_level = Mock(return_value="medium")
    handler.set_performance_monitor = Mock()
    return handler


@pytest.fixture
def mock_cache():
    """Mock cache instance."""
    cache = Mock()
    cache.get = Mock(return_value=None)  # Cache miss by default
    cache.put = Mock()
    cache.invalidate = Mock()
    cache.set_performance_monitor = Mock()
    return cache


@pytest.fixture
def mock_service_orchestrator():
    """Mock ServiceOrchestrator instance."""
    orchestrator = Mock()
    orchestrator.session_id = "test_session_123"
    orchestrator.execute_parallel_analysis = Mock(return_value={
        "status": "success",
        "successful": 5,
        "total_tasks": 6,
        "results": {
            "general_spend": {"status": "success", "data": {}},
            "storage_class": {"status": "success", "data": {}},
            "archive_optimization": {"status": "success", "data": {}},
            "api_cost": {"status": "success", "data": {}},
            "multipart_cleanup": {"status": "success", "data": {}}
        },
        "stored_tables": ["general_spend_results", "storage_class_results"]
    })
    orchestrator.query_session_data = Mock(return_value=[])
    orchestrator.get_stored_tables = Mock(return_value=["test_table_1", "test_table_2"])
    return orchestrator


@pytest.fixture
def cost_constraint_validator():
    """Fixture for validating no-cost constraints."""
    class CostConstraintValidator:
        def __init__(self):
            self.forbidden_operations = {
                'list_objects', 'list_objects_v2', 'head_object', 
                'get_object', 'get_object_attributes', 'select_object_content'
            }
            self.allowed_operations = {
                'list_buckets', 'get_bucket_location', 'get_bucket_lifecycle_configuration',
                'get_bucket_versioning', 'get_bucket_tagging', 'list_multipart_uploads'
            }
            self.operation_calls = []
        
        def validate_operation(self, operation_name: str):
            """Validate that an operation doesn't incur costs."""
            self.operation_calls.append(operation_name)
            if operation_name in self.forbidden_operations:
                raise ValueError(f"FORBIDDEN: Operation {operation_name} would incur costs")
            return operation_name in self.allowed_operations
        
        def get_operation_summary(self):
            """Get summary of operations called."""
            return {
                "total_operations": len(self.operation_calls),
                "unique_operations": len(set(self.operation_calls)),
                "operations": list(set(self.operation_calls)),
                "forbidden_called": [op for op in self.operation_calls if op in self.forbidden_operations],
                "allowed_called": [op for op in self.operation_calls if op in self.allowed_operations]
            }
    
    return CostConstraintValidator()


@pytest.fixture
def performance_test_data():
    """Test data for performance testing."""
    return {
        "small_dataset": {
            "bucket_count": 5,
            "object_count_per_bucket": 100,
            "expected_max_time": 10.0
        },
        "medium_dataset": {
            "bucket_count": 50,
            "object_count_per_bucket": 1000,
            "expected_max_time": 30.0
        },
        "large_dataset": {
            "bucket_count": 500,
            "object_count_per_bucket": 10000,
            "expected_max_time": 120.0
        }
    }


@pytest.fixture
def timeout_test_scenarios():
    """Test scenarios for timeout handling."""
    return {
        "quick_analysis": {
            "analysis_type": "general_spend",
            "timeout_seconds": 5.0,
            "expected_behavior": "complete"
        },
        "medium_analysis": {
            "analysis_type": "comprehensive",
            "timeout_seconds": 30.0,
            "expected_behavior": "complete"
        },
        "timeout_scenario": {
            "analysis_type": "comprehensive",
            "timeout_seconds": 1.0,
            "expected_behavior": "timeout"
        }
    }


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_bucket_data(count: int = 3) -> List[Dict[str, Any]]:
        """Create sample bucket data."""
        buckets = []
        for i in range(count):
            buckets.append({
                "Name": f"test-bucket-{i+1}",
                "CreationDate": datetime.now() - timedelta(days=30*(i+1)),
                "Region": ["us-east-1", "us-west-2", "eu-west-1"][i % 3]
            })
        return buckets
    
    @staticmethod
    def create_cost_data(days: int = 30) -> Dict[str, Any]:
        """Create sample cost data."""
        results = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            results.append({
                "TimePeriod": {"Start": date, "End": date},
                "Groups": [
                    {
                        "Keys": ["S3-Storage-Standard"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": str(10.0 + i * 0.5), "Unit": "USD"},
                            "UsageQuantity": {"Amount": str(1000 + i * 10), "Unit": "GB-Mo"}
                        }
                    }
                ]
            })
        return {"ResultsByTime": results}
    
    @staticmethod
    def create_analysis_result(analysis_type: str, status: str = "success") -> Dict[str, Any]:
        """Create sample analysis result."""
        return {
            "status": status,
            "analysis_type": analysis_type,
            "data": {
                "total_cost": 100.0,
                "optimization_opportunities": 3,
                "potential_savings": 25.0
            },
            "recommendations": [
                {
                    "type": "cost_optimization",
                    "priority": "high",
                    "title": f"Optimize {analysis_type}",
                    "potential_savings": 25.0
                }
            ],
            "execution_time": 5.0,
            "timestamp": datetime.now().isoformat()
        }


@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory()


# Performance testing utilities
class PerformanceTracker:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = datetime.now()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        if name in self.start_times:
            duration = (datetime.now() - self.start_times[name]).total_seconds()
            self.metrics[name] = duration
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def assert_performance(self, name: str, max_time: float):
        """Assert that an operation completed within time limit."""
        if name in self.metrics:
            assert self.metrics[name] <= max_time, f"Operation {name} took {self.metrics[name]:.2f}s, expected <= {max_time}s"
        else:
            raise ValueError(f"No timing data for operation: {name}")


@pytest.fixture
def performance_tracker():
    """Performance tracker fixture."""
    return PerformanceTracker()


# Async test utilities
def async_test(coro):
    """Decorator to run async tests."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.no_cost_validation = pytest.mark.no_cost_validation
pytest.mark.live = pytest.mark.live
pytest.mark.cloudwatch = pytest.mark.cloudwatch