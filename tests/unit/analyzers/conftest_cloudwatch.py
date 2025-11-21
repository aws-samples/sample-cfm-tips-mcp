"""
CloudWatch-specific pytest configuration and fixtures.

This module provides CloudWatch-specific fixtures for testing analyzers and services.
"""

import pytest
import boto3
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from moto import mock_aws

from services.cloudwatch_service import CloudWatchOperationResult


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    import os
    with patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'testing',
        'AWS_SECRET_ACCESS_KEY': 'testing',
        'AWS_SECURITY_TOKEN': 'testing',
        'AWS_SESSION_TOKEN': 'testing',
        'AWS_DEFAULT_REGION': 'us-east-1'
    }):
        yield


@pytest.fixture
def mock_cloudwatch_client(mock_aws_credentials):
    """Mock CloudWatch client with moto."""
    with mock_aws():
        yield boto3.client('cloudwatch', region_name='us-east-1')


@pytest.fixture
def mock_logs_client(mock_aws_credentials):
    """Mock CloudWatch Logs client with moto."""
    with mock_aws():
        yield boto3.client('logs', region_name='us-east-1')


@pytest.fixture
def mock_ce_client(mock_aws_credentials):
    """Mock Cost Explorer client with moto."""
    with mock_aws():
        yield boto3.client('ce', region_name='us-east-1')


@pytest.fixture
def sample_cloudwatch_cost_data():
    """Sample CloudWatch Cost Explorer response data."""
    return {
        "ResultsByTime": [
            {
                "TimePeriod": {
                    "Start": "2024-01-01",
                    "End": "2024-01-02"
                },
                "Groups": [
                    {
                        "Keys": ["DataIngestion-Bytes"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": "5.25", "Unit": "USD"},
                            "UsageQuantity": {"Amount": "10.5", "Unit": "GB"}
                        }
                    },
                    {
                        "Keys": ["DataStorage-ByteHrs"],
                        "Metrics": {
                            "UnblendedCost": {"Amount": "2.10", "Unit": "USD"},
                            "UsageQuantity": {"Amount": "70.0", "Unit": "GB-Hours"}
                        }
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_cloudwatch_alarms():
    """Sample CloudWatch alarms data."""
    return [
        {
            "AlarmName": "test-alarm-1",
            "AlarmDescription": "Test alarm with actions",
            "StateValue": "OK",
            "AlarmActions": ["arn:aws:sns:us-east-1:123456789012:test-topic"],
            "Period": 300,
            "MetricName": "CPUUtilization"
        },
        {
            "AlarmName": "test-alarm-2",
            "AlarmDescription": "Test alarm without actions",
            "StateValue": "INSUFFICIENT_DATA",
            "AlarmActions": [],
            "Period": 60,  # High resolution
            "MetricName": "NetworkIn"
        }
    ]


@pytest.fixture
def sample_cloudwatch_log_groups():
    """Sample CloudWatch log groups data."""
    return [
        {
            "logGroupName": "/aws/lambda/test-function",
            "creationTime": int((datetime.now() - timedelta(days=30)).timestamp() * 1000),
            "retentionInDays": 14,
            "storedBytes": 1024000
        },
        {
            "logGroupName": "/aws/apigateway/test-api",
            "creationTime": int((datetime.now() - timedelta(days=400)).timestamp() * 1000),
            "storedBytes": 2048000
            # No retention policy
        }
    ]


@pytest.fixture
def mock_cloudwatch_pricing_service():
    """Mock CloudWatch pricing service instance."""
    service = Mock()
    service.region = "us-east-1"
    
    def mock_get_logs_pricing():
        return {
            "status": "success",
            "logs_pricing": {
                "ingestion_per_gb": 0.50,
                "storage_per_gb_month": 0.03,
                "insights_per_gb_scanned": 0.005
            }
        }
    
    def mock_calculate_logs_cost(log_groups_data):
        total_cost = 0.0
        for log_group in log_groups_data:
            stored_gb = log_group.get('storedBytes', 0) / (1024**3)
            total_cost += stored_gb * 0.03
        
        return {
            "status": "success",
            "total_monthly_cost": total_cost,
            "cost_breakdown": {
                "storage_cost": total_cost,
                "ingestion_cost": 0.0,
                "insights_cost": 0.0
            }
        }
    
    service.get_logs_pricing = mock_get_logs_pricing
    service.calculate_logs_cost = mock_calculate_logs_cost
    
    return service


@pytest.fixture
def mock_cloudwatch_service():
    """Mock CloudWatch service instance."""
    service = Mock()
    service.region = "us-east-1"
    service.operation_count = 0
    service.cost_incurring_operations = []
    service.total_execution_time = 0.0
    
    # Mock async methods
    async def mock_list_metrics(namespace=None, metric_name=None, dimensions=None):
        return CloudWatchOperationResult(
            success=True,
            data={
                'metrics': [
                    {'Namespace': 'AWS/EC2', 'MetricName': 'CPUUtilization'},
                    {'Namespace': 'Custom/App', 'MetricName': 'RequestCount'}
                ],
                'total_count': 2
            },
            operation_name='list_metrics',
            operation_type='free'
        )
    
    async def mock_describe_alarms(alarm_names=None):
        return CloudWatchOperationResult(
            success=True,
            data={
                'alarms': [
                    {
                        'AlarmName': 'test-alarm',
                        'StateValue': 'OK',
                        'AlarmActions': ['arn:aws:sns:us-east-1:123456789012:test-topic'],
                        'Period': 300
                    }
                ],
                'total_count': 1,
                'analysis': {
                    'total_alarms': 1,
                    'alarms_by_state': {'OK': 1},
                    'alarms_without_actions': []
                }
            },
            operation_name='describe_alarms',
            operation_type='free'
        )
    
    service.list_metrics = mock_list_metrics
    service.describe_alarms = mock_describe_alarms
    
    return service