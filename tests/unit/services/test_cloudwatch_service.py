"""
Unit tests for CloudWatchService.

Tests cost-aware operations, error handling, fallback mechanisms,
and comprehensive CloudWatch API integration with strict cost constraint validation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta, timezone
import json

from services.cloudwatch_service import (
    CloudWatchService, CloudWatchServiceConfig, CloudWatchOperationResult,
    create_cloudwatch_service
)
from playbooks.cloudwatch.cost_controller import CostPreferences
from botocore.exceptions import ClientError


@pytest.mark.unit
class TestCloudWatchServiceConfig:
    """Test CloudWatchServiceConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CloudWatchServiceConfig()
        
        assert config.region is None
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout_seconds == 30.0
        assert config.enable_cost_tracking is True
        assert config.enable_fallback is True
        assert config.cost_preferences is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        cost_prefs = CostPreferences(allow_cost_explorer=True)
        config = CloudWatchServiceConfig(
            region='us-west-2',
            max_retries=5,
            retry_delay=2.0,
            timeout_seconds=60.0,
            enable_cost_tracking=False,
            enable_fallback=False,
            cost_preferences=cost_prefs
        )
        
        assert config.region == 'us-west-2'
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.timeout_seconds == 60.0
        assert config.enable_cost_tracking is False
        assert config.enable_fallback is False
        assert config.cost_preferences == cost_prefs


@pytest.mark.unit
class TestCloudWatchOperationResult:
    """Test CloudWatchOperationResult dataclass."""
    
    def test_default_result(self):
        """Test default operation result values."""
        result = CloudWatchOperationResult(success=True)
        
        assert result.success is True
        assert result.data is None
        assert result.error_message is None
        assert result.operation_name == ""
        assert result.cost_incurred is False
        assert result.operation_type == "free"
        assert result.execution_time == 0.0
        assert result.fallback_used is False
        assert result.primary_data_source == "cloudwatch_api"
        assert result.api_calls_made == []
    
    def test_custom_result(self):
        """Test custom operation result values."""
        result = CloudWatchOperationResult(
            success=False,
            data={'test': 'data'},
            error_message='Test error',
            operation_name='test_operation',
            cost_incurred=True,
            operation_type='paid',
            execution_time=1.5,
            fallback_used=True,
            primary_data_source='cost_explorer',
            api_calls_made=['test_api_call']
        )
        
        assert result.success is False
        assert result.data == {'test': 'data'}
        assert result.error_message == 'Test error'
        assert result.operation_name == 'test_operation'
        assert result.cost_incurred is True
        assert result.operation_type == 'paid'
        assert result.execution_time == 1.5
        assert result.fallback_used is True
        assert result.primary_data_source == 'cost_explorer'
        assert result.api_calls_made == ['test_api_call']


@pytest.mark.unit
class TestCloudWatchService:
    """Test CloudWatchService factory and lifecycle management."""
    
    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for testing."""
        with patch('services.cloudwatch_service.boto3') as mock_boto3:
            mock_cloudwatch = MagicMock()
            mock_logs = MagicMock()
            mock_ce = MagicMock()
            mock_pricing = MagicMock()
            
            mock_boto3.client.side_effect = lambda service, **kwargs: {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }[service]
            
            yield {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }
    
    @pytest.fixture
    def cloudwatch_service(self, mock_boto3_clients):
        """Create CloudWatchService instance for testing."""
        config = CloudWatchServiceConfig(region='us-east-1')
        return CloudWatchService(config)
    
    def test_initialization(self, cloudwatch_service):
        """Test CloudWatchService initialization."""
        assert cloudwatch_service.region == 'us-east-1'
        assert isinstance(cloudwatch_service.cost_preferences, CostPreferences)
        assert cloudwatch_service._dao is not None
        assert cloudwatch_service._pricing_dao is not None
    
    def test_update_cost_preferences_dict(self, cloudwatch_service):
        """Test updating cost preferences with CostPreferences object."""
        prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_minimal_cost_metrics=True
        )
        
        cloudwatch_service.update_cost_preferences(prefs)
        
        assert cloudwatch_service.cost_preferences.allow_cost_explorer is True
        assert cloudwatch_service.cost_preferences.allow_minimal_cost_metrics is True
        assert cloudwatch_service.cost_preferences.allow_aws_config is False
    
    def test_get_general_spend_service(self, cloudwatch_service):
        """Test getting general spend service (lazy loading)."""
        service1 = cloudwatch_service.getGeneralSpendService()
        service2 = cloudwatch_service.getGeneralSpendService()
        
        assert service1 is not None
        assert service1 is service2  # Same instance (lazy loaded)
        
    def test_get_metrics_service(self, cloudwatch_service):
        """Test getting metrics service (lazy loading)."""
        service1 = cloudwatch_service.getMetricsService()
        service2 = cloudwatch_service.getMetricsService()
        
        assert service1 is not None
        assert service1 is service2  # Same instance (lazy loaded)
        
    def test_get_logs_service(self, cloudwatch_service):
        """Test getting logs service (lazy loading)."""
        service1 = cloudwatch_service.getLogsService()
        service2 = cloudwatch_service.getLogsService()
        
        assert service1 is not None
        assert service1 is service2  # Same instance (lazy loaded)
        
    def test_get_alarms_service(self, cloudwatch_service):
        """Test getting alarms service (lazy loading)."""
        service1 = cloudwatch_service.getAlarmsService()
        service2 = cloudwatch_service.getAlarmsService()
        
        assert service1 is not None
        assert service1 is service2  # Same instance (lazy loaded)
        
    def test_get_dashboards_service(self, cloudwatch_service):
        """Test getting dashboards service (lazy loading)."""
        service1 = cloudwatch_service.getDashboardsService()
        service2 = cloudwatch_service.getDashboardsService()
        
        assert service1 is not None
        assert service1 is service2  # Same instance (lazy loaded)
    
    def test_update_cost_preferences_propagation(self, cloudwatch_service):
        """Test that cost preferences are propagated to all tip services."""
        # Initialize all services
        general = cloudwatch_service.getGeneralSpendService()
        metrics = cloudwatch_service.getMetricsService()
        logs = cloudwatch_service.getLogsService()
        alarms = cloudwatch_service.getAlarmsService()
        dashboards = cloudwatch_service.getDashboardsService()
        
        # Update preferences
        new_prefs = CostPreferences(allow_cost_explorer=True)
        cloudwatch_service.update_cost_preferences(new_prefs)
        
        # Verify all services have updated preferences
        assert general.cost_preferences.allow_cost_explorer is True
        assert metrics.cost_preferences.allow_cost_explorer is True
        assert logs.cost_preferences.allow_cost_explorer is True
        assert alarms.cost_preferences.allow_cost_explorer is True
        assert dashboards.cost_preferences.allow_cost_explorer is True
    
    def test_get_service_statistics(self, cloudwatch_service):
        """Test service statistics retrieval."""
        stats = cloudwatch_service.get_service_statistics()
        
        assert 'service_info' in stats
        assert 'cache_statistics' in stats
        assert 'cost_control_status' in stats
        assert stats['service_info']['region'] == 'us-east-1'
        assert 'initialized_services' in stats['service_info']
    
    def test_clear_cache(self, cloudwatch_service):
        """Test cache clearing."""
        # This should not raise an error
        cloudwatch_service.clear_cache()
        
        # Verify cache is cleared by checking stats
        stats = cloudwatch_service.get_service_statistics()
        cache_stats = stats['cache_statistics']
        assert cache_stats['total_entries'] == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestCloudWatchServiceConvenienceFunctions:
    """Test convenience functions for CloudWatchService."""
    
    @patch('services.cloudwatch_service.boto3')
    async def test_create_cloudwatch_service_default(self, mock_boto3):
        """Test creating CloudWatchService with default parameters."""
        # Mock all required clients
        mock_cloudwatch = MagicMock()
        mock_logs = MagicMock()
        mock_pricing = MagicMock()
        
        mock_boto3.client.side_effect = lambda service, **kwargs: {
            'cloudwatch': mock_cloudwatch,
            'logs': mock_logs,
            'pricing': mock_pricing
        }[service]
        
        service = await create_cloudwatch_service()
        
        assert isinstance(service, CloudWatchService)
        assert service.region is None
        assert isinstance(service.cost_preferences, CostPreferences)
    
    @patch('services.cloudwatch_service.boto3')
    async def test_create_cloudwatch_service_with_params(self, mock_boto3):
        """Test creating CloudWatchService with custom parameters."""
        # Mock all required clients
        mock_cloudwatch = MagicMock()
        mock_logs = MagicMock()
        mock_pricing = MagicMock()
        
        mock_boto3.client.side_effect = lambda service, **kwargs: {
            'cloudwatch': mock_cloudwatch,
            'logs': mock_logs,
            'pricing': mock_pricing
        }[service]
        
        cost_prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_minimal_cost_metrics=True
        )
        
        service = await create_cloudwatch_service(
            region='us-west-2',
            cost_preferences=cost_prefs
        )
        
        assert service.region == 'us-west-2'
        assert service.cost_preferences.allow_cost_explorer is True
        assert service.cost_preferences.allow_minimal_cost_metrics is True
    
    # TODO: Implement check_cloudwatch_service_connectivity function in cloudwatch_service.py
    # @patch('services.cloudwatch_service.boto3')
    # async def test_test_cloudwatch_service_connectivity_success(self, mock_boto3):
    #     """Test connectivity testing with successful operations."""
    #     # Mock successful responses for all operations
    #     mock_cloudwatch = MagicMock()
    #     mock_logs = MagicMock()
    #     mock_ce = MagicMock()
    #     
    #     mock_boto3.client.side_effect = lambda service, **kwargs: {
    #         'cloudwatch': mock_cloudwatch,
    #         'logs': mock_logs,
    #         'ce': mock_ce
    #     }[service]
    #     
    #     # Mock paginator responses
    #     mock_paginator = MagicMock()
    #     mock_paginator.paginate.return_value = [{'Metrics': [], 'MetricAlarms': [], 'CompositeAlarms': [], 'DashboardEntries': [], 'logGroups': []}]
    #     mock_cloudwatch.get_paginator.return_value = mock_paginator
    #     mock_logs.get_paginator.return_value = mock_paginator
    #     
    #     service = CloudWatchService()
    #     result = await check_cloudwatch_service_connectivity(service)
    #     
    #     assert result['connectivity_test'] is True
    #     assert result['overall_success'] is True
    #     assert 'tests' in result
    #     assert len(result['tests']) == 4  # Four free operations tested
    #     
    #     for test_name, test_result in result['tests'].items():
    #         assert test_result['success'] is True
    #         assert test_result['execution_time'] >= 0
    #         assert test_result['error'] is None
    # 
    # @patch('services.cloudwatch_service.boto3')
    # async def test_test_cloudwatch_service_connectivity_failure(self, mock_boto3):
    #     """Test connectivity testing with failed operations."""
    #     # Mock failed responses
    #     mock_cloudwatch = MagicMock()
    #     mock_logs = MagicMock()
    #     mock_ce = MagicMock()
    #     
    #     mock_boto3.client.side_effect = lambda service, **kwargs: {
    #         'cloudwatch': mock_cloudwatch,
    #         'logs': mock_logs,
    #         'ce': mock_ce
    #     }[service]
    #     
    #     # Mock paginator to raise exception
    #     mock_paginator = MagicMock()
    #     mock_paginator.paginate.side_effect = ClientError(
    #         error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
    #         operation_name='ListMetrics'
    #     )
    #     mock_cloudwatch.get_paginator.return_value = mock_paginator
    #     mock_logs.get_paginator.return_value = mock_paginator
    #     
    #     service = CloudWatchService()
    #     result = await check_cloudwatch_service_connectivity(service)
    #     
    #     assert result['connectivity_test'] is True
    #     assert result['overall_success'] is False
    #     
    #     for test_name, test_result in result['tests'].items():
    #         assert test_result['success'] is False
    #         assert 'Access denied' in test_result['error']


@pytest.mark.unit
class TestCloudWatchServiceCostConstraints:
    """Test cost constraint validation and enforcement."""
    
    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for testing."""
        with patch('services.cloudwatch_service.boto3') as mock_boto3:
            mock_cloudwatch = MagicMock()
            mock_logs = MagicMock()
            mock_ce = MagicMock()
            mock_pricing = MagicMock()
            
            mock_boto3.client.side_effect = lambda service, **kwargs: {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }[service]
            
            yield {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }
    
    @pytest.fixture
    def cloudwatch_service(self, mock_boto3_clients):
        """Create CloudWatchService instance for testing."""
        return CloudWatchService()
    
    def test_cost_preferences_initialization(self, cloudwatch_service):
        """Test that cost preferences are properly initialized."""
        assert isinstance(cloudwatch_service.cost_preferences, CostPreferences)
        assert cloudwatch_service.cost_preferences.allow_cost_explorer is False
        assert cloudwatch_service.cost_preferences.allow_minimal_cost_metrics is False
        assert cloudwatch_service.cost_preferences.allow_aws_config is False
        assert cloudwatch_service.cost_preferences.allow_cloudtrail is False
    
    def test_cost_preferences_update(self, cloudwatch_service):
        """Test updating cost preferences."""
        new_prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_minimal_cost_metrics=True
        )
        
        cloudwatch_service.update_cost_preferences(new_prefs)
        
        assert cloudwatch_service.cost_preferences.allow_cost_explorer is True
        assert cloudwatch_service.cost_preferences.allow_minimal_cost_metrics is True
        assert cloudwatch_service.cost_preferences.allow_aws_config is False
    
    def test_cost_control_status_in_statistics(self, cloudwatch_service):
        """Test that cost control status is included in statistics."""
        stats = cloudwatch_service.get_service_statistics()
        
        assert 'cost_control_status' in stats
        assert stats['cost_control_status']['cost_controller_active'] is True
        assert stats['cost_control_status']['preferences_validated'] is True


if __name__ == "__main__":
    pytest.main([__file__])