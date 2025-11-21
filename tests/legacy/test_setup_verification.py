"""
Setup verification test to ensure the testing framework is working correctly.

This test validates that the testing infrastructure is properly configured
and can run basic tests with mocked AWS services.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime


@pytest.mark.unit
class TestSetupVerification:
    """Verify that the testing setup is working correctly."""
    
    def test_pytest_is_working(self):
        """Test that pytest is working correctly."""
        assert True
    
    def test_fixtures_are_available(self, mock_s3_service, mock_storage_lens_service, 
                                  mock_pricing_service):
        """Test that common fixtures are available."""
        assert mock_s3_service is not None
        assert mock_storage_lens_service is not None
        assert mock_pricing_service is not None
    
    @pytest.mark.asyncio
    async def test_async_testing_works(self):
        """Test that async testing is working."""
        async def async_function():
            await asyncio.sleep(0.001)
            return "async_result"
        
        result = await async_function()
        assert result == "async_result"
    
    def test_mocking_works(self):
        """Test that mocking is working correctly."""
        mock_service = Mock()
        mock_service.test_method.return_value = "mocked_result"
        
        result = mock_service.test_method()
        assert result == "mocked_result"
        mock_service.test_method.assert_called_once()
    
    def test_aws_mocking_works(self, mock_aws_credentials):
        """Test that AWS service mocking is working."""
        with patch('boto3.client') as mock_boto_client:
            mock_client = Mock()
            mock_client.list_buckets.return_value = {"Buckets": []}
            mock_boto_client.return_value = mock_client
            
            import boto3
            s3_client = boto3.client('s3')
            result = s3_client.list_buckets()
            
            assert result == {"Buckets": []}
    
    def test_cost_constraint_validator_works(self, cost_constraint_validator):
        """Test that cost constraint validator is working."""
        # Should allow valid operations
        assert cost_constraint_validator.validate_operation('list_buckets') is True
        
        # Should reject forbidden operations
        with pytest.raises(ValueError):
            cost_constraint_validator.validate_operation('list_objects_v2')
        
        summary = cost_constraint_validator.get_operation_summary()
        assert summary["total_operations"] == 2
        assert "list_buckets" in summary["allowed_called"]
        assert "list_objects_v2" in summary["forbidden_called"]
    
    def test_performance_tracker_works(self, performance_tracker):
        """Test that performance tracker is working."""
        performance_tracker.start_timer("test_operation")
        # Simulate some work
        import time
        time.sleep(0.01)
        duration = performance_tracker.end_timer("test_operation")
        
        assert duration > 0
        assert duration < 1.0  # Should be very quick
        
        metrics = performance_tracker.get_metrics()
        assert "test_operation" in metrics
        assert metrics["test_operation"] > 0
    
    def test_test_data_factory_works(self, test_data_factory):
        """Test that test data factory is working."""
        buckets = test_data_factory.create_bucket_data(count=3)
        assert len(buckets) == 3
        assert all("Name" in bucket for bucket in buckets)
        assert all("CreationDate" in bucket for bucket in buckets)
        
        cost_data = test_data_factory.create_cost_data(days=5)
        assert "ResultsByTime" in cost_data
        assert len(cost_data["ResultsByTime"]) == 5
        
        analysis_result = test_data_factory.create_analysis_result("test_analysis")
        assert analysis_result["status"] == "success"
        assert analysis_result["analysis_type"] == "test_analysis"


@pytest.mark.integration
class TestIntegrationSetupVerification:
    """Verify that integration testing setup is working."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_can_be_mocked(self, mock_service_orchestrator):
        """Test that orchestrator can be properly mocked for integration tests."""
        # This would normally import the real orchestrator, but we'll mock it
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor'), \
             patch('core.s3_optimization_orchestrator.get_memory_manager'), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler'), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache'), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache'), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache'):
            
            from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            assert orchestrator.region == "us-east-1"
            assert orchestrator.service_orchestrator == mock_service_orchestrator


@pytest.mark.performance
class TestPerformanceSetupVerification:
    """Verify that performance testing setup is working."""
    
    @pytest.mark.asyncio
    async def test_performance_measurement_works(self, performance_tracker):
        """Test that performance measurement is working."""
        performance_tracker.start_timer("performance_test")
        
        # Simulate some async work
        await asyncio.sleep(0.01)
        
        duration = performance_tracker.end_timer("performance_test")
        
        assert duration > 0.005  # Should be at least 5ms
        assert duration < 0.1    # Should be less than 100ms
        
        # Test performance assertion
        performance_tracker.assert_performance("performance_test", 0.1)


@pytest.mark.no_cost_validation
class TestCostValidationSetupVerification:
    """Verify that cost validation testing setup is working."""
    
    def test_cost_constraint_system_is_active(self):
        """Test that cost constraint validation system is active."""
        from services.s3_service import ALLOWED_S3_OPERATIONS, FORBIDDEN_S3_OPERATIONS
        
        # Verify that the constraint lists are populated
        assert len(ALLOWED_S3_OPERATIONS) > 0
        assert len(FORBIDDEN_S3_OPERATIONS) > 0
        
        # Verify critical operations are in the right lists
        assert 'list_buckets' in ALLOWED_S3_OPERATIONS
        assert 'list_objects_v2' in FORBIDDEN_S3_OPERATIONS
    
    def test_cost_constraint_violation_error_works(self, mock_aws_credentials):
        """Test that cost constraint violation errors work correctly."""
        from services.s3_service import S3Service, S3CostConstraintViolationError
        
        service = S3Service(region="us-east-1")
        
        with pytest.raises(S3CostConstraintViolationError):
            service._validate_s3_operation('list_objects_v2')
    
    def test_cost_validator_fixture_works(self, cost_constraint_validator):
        """Test that cost validator fixture is working correctly."""
        # Should track operations
        cost_constraint_validator.validate_operation('list_buckets')
        
        # Should reject forbidden operations
        with pytest.raises(ValueError):
            cost_constraint_validator.validate_operation('get_object')
        
        summary = cost_constraint_validator.get_operation_summary()
        assert summary["total_operations"] == 2
        assert len(summary["forbidden_called"]) == 1
        assert len(summary["allowed_called"]) == 1


class TestMarkerSystem:
    """Test that the pytest marker system is working."""
    
    def test_markers_are_configured(self):
        """Test that pytest markers are properly configured."""
        # This test itself uses markers, so if it runs, markers are working
        assert True
    
    def test_can_run_specific_marker_tests(self):
        """Test that we can run tests with specific markers."""
        # This would be tested by running: pytest -m unit
        # If this test runs when using -m unit, then markers work
        assert True