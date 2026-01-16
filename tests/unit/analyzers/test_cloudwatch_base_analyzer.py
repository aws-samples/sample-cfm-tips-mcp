"""
Unit tests for CloudWatch BaseAnalyzer abstract class.

Tests the CloudWatch-specific base analyzer interface, common functionality, and abstract method enforcement.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from abc import ABC
from datetime import datetime

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer


class ConcreteCloudWatchAnalyzer(BaseAnalyzer):
    """Concrete implementation of BaseAnalyzer for CloudWatch testing."""
    
    async def analyze(self, **kwargs):
        """Test implementation of analyze method."""
        return {
            'status': 'success',
            'analysis_type': 'test',
            'data': {'test': 'data'},
            'cost_incurred': False,
            'cost_incurring_operations': []
        }
    
    def get_recommendations(self, analysis_results):
        """Test implementation of get_recommendations method."""
        return [
            {
                'type': 'test_recommendation',
                'priority': 'medium',
                'title': 'Test Recommendation',
                'description': 'Test description'
            }
        ]


@pytest.mark.unit
class TestCloudWatchBaseAnalyzer:
    """Test cases for CloudWatch BaseAnalyzer abstract class."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'cost_explorer_service': Mock(),
            'config_service': Mock(),
            'metrics_service': Mock(),
            'cloudwatch_service': Mock(),
            'pricing_service': Mock(),
            'performance_monitor': Mock(),
            'memory_manager': Mock()
        }
    
    @pytest.fixture
    def concrete_analyzer(self, mock_services):
        """Create concrete analyzer instance for testing."""
        return ConcreteCloudWatchAnalyzer(**mock_services)
    
    def test_initialization(self, concrete_analyzer, mock_services):
        """Test BaseAnalyzer initialization."""
        assert concrete_analyzer.cost_explorer_service == mock_services['cost_explorer_service']
        assert concrete_analyzer.config_service == mock_services['config_service']
        assert concrete_analyzer.metrics_service == mock_services['metrics_service']
        assert concrete_analyzer.cloudwatch_service == mock_services['cloudwatch_service']
        assert concrete_analyzer.pricing_service == mock_services['pricing_service']
        assert concrete_analyzer.performance_monitor == mock_services['performance_monitor']
        assert concrete_analyzer.memory_manager == mock_services['memory_manager']
        
        # Check default values
        assert concrete_analyzer.analysis_type == 'concretecloudwatch'
        assert concrete_analyzer.version == '1.0.0'
        assert concrete_analyzer.execution_count == 0
        assert concrete_analyzer.last_execution is None
        assert concrete_analyzer.logger is not None
    
    def test_abstract_class_cannot_be_instantiated(self, mock_services):
        """Test that BaseAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnalyzer(**mock_services)
    
    def test_prepare_analysis_context(self, concrete_analyzer):
        """Test analysis context preparation."""
        context = concrete_analyzer.prepare_analysis_context(
            region='us-east-1',
            lookback_days=30,
            session_id='test_session'
        )
        
        assert context['analysis_type'] == 'concretecloudwatch'
        assert context['analyzer_version'] == '1.0.0'
        assert context['region'] == 'us-east-1'
        assert context['lookback_days'] == 30
        assert context['session_id'] == 'test_session'
        assert 'cost_constraints' in context
        assert context['cost_constraints']['prioritize_cost_explorer'] is True
    
    def test_get_analyzer_info(self, concrete_analyzer):
        """Test analyzer information retrieval."""
        info = concrete_analyzer.get_analyzer_info()
        
        assert info['analysis_type'] == 'concretecloudwatch'
        assert info['class_name'] == 'ConcreteCloudWatchAnalyzer'
        assert info['version'] == '1.0.0'
        assert info['execution_count'] == 0
        assert info['last_execution'] is None
        assert 'services' in info
        assert 'cost_optimization' in info
    
    def test_validate_parameters_valid(self, concrete_analyzer):
        """Test parameter validation with valid parameters."""
        validation = concrete_analyzer.validate_parameters(
            region='us-east-1',
            lookback_days=30,
            timeout_seconds=60
        )
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
        assert len(validation['warnings']) == 0
    
    def test_validate_parameters_invalid_region(self, concrete_analyzer):
        """Test parameter validation with invalid region."""
        validation = concrete_analyzer.validate_parameters(region=123)
        
        assert validation['valid'] is False
        assert any('Region must be a string' in error for error in validation['errors'])
    
    def test_validate_parameters_invalid_lookback_days(self, concrete_analyzer):
        """Test parameter validation with invalid lookback_days."""
        validation = concrete_analyzer.validate_parameters(lookback_days=-5)
        
        assert validation['valid'] is False
        assert any('lookback_days must be a positive integer' in error for error in validation['errors'])
    
    def test_validate_parameters_large_lookback_warning(self, concrete_analyzer):
        """Test parameter validation with large lookback_days generates warning."""
        validation = concrete_analyzer.validate_parameters(lookback_days=400)
        
        assert validation['valid'] is True
        assert len(validation['warnings']) > 0
        assert any('lookback_days > 365' in warning for warning in validation['warnings'])
    
    def test_validate_parameters_invalid_timeout(self, concrete_analyzer):
        """Test parameter validation with invalid timeout."""
        validation = concrete_analyzer.validate_parameters(timeout_seconds='invalid')
        
        assert validation['valid'] is False
        assert any('timeout_seconds must be a positive number' in error for error in validation['errors'])
    
    def test_validate_parameters_list_validation(self, concrete_analyzer):
        """Test parameter validation with list parameters."""
        # Valid list
        validation = concrete_analyzer.validate_parameters(
            log_group_names=['/aws/lambda/test']
        )
        assert validation['valid'] is True
        
        # Invalid list type
        validation = concrete_analyzer.validate_parameters(
            log_group_names='not-a-list'
        )
        assert validation['valid'] is False
        assert any('log_group_names must be a list' in error for error in validation['errors'])
        
        # Invalid list items
        validation = concrete_analyzer.validate_parameters(
            log_group_names=[123, 456]
        )
        assert validation['valid'] is False
        assert any('All log group names must be strings' in error for error in validation['errors'])
    
    def test_create_recommendation(self, concrete_analyzer):
        """Test creation of standardized recommendation."""
        recommendation = concrete_analyzer.create_recommendation(
            rec_type='cost_optimization',
            priority='high',
            title='Test Recommendation',
            description='Test description',
            potential_savings=25.50,
            affected_resources=['resource1', 'resource2'],
            action_items=['action1', 'action2'],
            cloudwatch_component='logs'
        )
        
        assert recommendation['type'] == 'cost_optimization'
        assert recommendation['priority'] == 'high'
        assert recommendation['title'] == 'Test Recommendation'
        assert recommendation['description'] == 'Test description'
        assert recommendation['potential_savings'] == 25.50
        assert recommendation['potential_savings_formatted'] == '$25.50'
        assert recommendation['affected_resources'] == ['resource1', 'resource2']
        assert recommendation['resource_count'] == 2
        assert recommendation['action_items'] == ['action1', 'action2']
        assert recommendation['cloudwatch_component'] == 'logs'
        assert recommendation['analyzer'] == 'concretecloudwatch'
        assert 'created_at' in recommendation
    
    def test_handle_analysis_error(self, concrete_analyzer):
        """Test error handling with different error categories."""
        context = {'test': 'context'}
        
        # Test permission error
        permission_error = Exception("Access denied to CloudWatch")
        result = concrete_analyzer.handle_analysis_error(permission_error, context)
        
        assert result['status'] == 'error'
        assert result['error_category'] == 'permissions'
        assert result['cost_incurred'] is False
        assert len(result['recommendations']) > 0
        assert result['recommendations'][0]['type'] == 'permission_fix'
        
        # Test rate limiting error
        throttle_error = Exception("Rate exceeded - throttling request")
        result = concrete_analyzer.handle_analysis_error(throttle_error, context)
        
        assert result['error_category'] == 'rate_limiting'
        assert result['recommendations'][0]['type'] == 'rate_limit_optimization'
    
    def test_log_analysis_start_and_complete(self, concrete_analyzer, mock_services):
        """Test analysis logging methods."""
        context = {'test': 'context'}
        result = {
            'status': 'success',
            'execution_time': 5.0,
            'recommendations': [{'type': 'test'}],
            'cost_incurred': True,
            'cost_incurring_operations': ['cost_explorer'],
            'primary_data_source': 'cost_explorer'
        }
        
        # Test start logging
        concrete_analyzer.log_analysis_start(context)
        assert concrete_analyzer.execution_count == 1
        assert concrete_analyzer.last_execution is not None
        
        # Test completion logging
        concrete_analyzer.log_analysis_complete(context, result)
        
        # Verify performance monitor was called if available
        if mock_services['performance_monitor']:
            assert mock_services['performance_monitor'].record_metric.called
    
    @pytest.mark.asyncio
    async def test_concrete_analyze_method(self, concrete_analyzer):
        """Test that concrete implementation works."""
        result = await concrete_analyzer.analyze(region='us-east-1')
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'test'
        assert result['data']['test'] == 'data'
    
    def test_concrete_get_recommendations_method(self, concrete_analyzer):
        """Test that concrete get_recommendations works."""
        analysis_results = {'data': {'test': 'data'}}
        recommendations = concrete_analyzer.get_recommendations(analysis_results)
        
        assert len(recommendations) == 1
        assert recommendations[0]['type'] == 'test_recommendation'
        assert recommendations[0]['title'] == 'Test Recommendation'
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling_success(self, concrete_analyzer, mock_services):
        """Test successful execution with error handling wrapper."""
        result = await concrete_analyzer.execute_with_error_handling(
            region='us-east-1',
            lookback_days=30
        )
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'test'  # From the concrete analyze method
        assert 'timestamp' in result
        assert 'cost_incurred' in result
        assert 'recommendations' in result
        assert concrete_analyzer.execution_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling_validation_failure(self, concrete_analyzer):
        """Test execution with parameter validation failure."""
        result = await concrete_analyzer.execute_with_error_handling(
            region=123,  # Invalid region
            lookback_days=-5  # Invalid lookback_days
        )
        
        assert result['status'] == 'error'
        assert result['error_message'] == 'Parameter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) >= 2
    
    def test_memory_management_integration(self, concrete_analyzer, mock_services):
        """Test memory management integration."""
        mock_services['memory_manager'].start_memory_tracking.return_value = 'tracker_123'
        
        # Memory manager should be available
        assert concrete_analyzer.memory_manager is not None
        
        # Test memory tracking can be started
        tracker_id = concrete_analyzer.memory_manager.start_memory_tracking()
        assert tracker_id == 'tracker_123'


class IncompleteCloudWatchAnalyzer(BaseAnalyzer):
    """Incomplete analyzer implementation for testing abstract method enforcement."""
    
    # Missing analyze method implementation
    def get_recommendations(self, analysis_results):
        return []


class IncompleteCloudWatchAnalyzer2(BaseAnalyzer):
    """Another incomplete analyzer implementation for testing."""
    
    async def analyze(self, **kwargs):
        return {}
    
    # Missing get_recommendations method implementation


@pytest.mark.unit
class TestCloudWatchBaseAnalyzerAbstractMethods:
    """Test abstract method enforcement."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'cost_explorer_service': Mock(),
            'config_service': Mock(),
            'metrics_service': Mock(),
            'cloudwatch_service': Mock(),
            'pricing_service': Mock(),
            'performance_monitor': Mock(),
            'memory_manager': Mock()
        }
    
    def test_incomplete_analyzer_missing_analyze(self, mock_services):
        """Test that analyzer without analyze method cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCloudWatchAnalyzer(**mock_services)
    
    def test_incomplete_analyzer_missing_get_recommendations(self, mock_services):
        """Test that analyzer without get_recommendations method cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCloudWatchAnalyzer2(**mock_services)


if __name__ == "__main__":
    pytest.main([__file__])