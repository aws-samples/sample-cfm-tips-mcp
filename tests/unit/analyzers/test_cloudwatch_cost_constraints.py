"""
Unit tests for CloudWatch cost constraint validation and enforcement.

Tests that no unexpected charges can occur, validates cost control flags,
and ensures proper cost transparency and tracking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from playbooks.cloudwatch.cost_controller import CostController, CostPreferences, OperationType
from services.cloudwatch_service import CloudWatchService, CloudWatchServiceConfig
from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestCostConstraintValidation:
    """Test cost constraint validation and enforcement."""
    
    @pytest.fixture
    def cost_validator(self):
        """Create cost constraint validator."""
        class CostConstraintValidator:
            def __init__(self):
                self.forbidden_operations = {
                    'logs_insights_queries', 'extensive_metric_retrieval',
                    'start_query', 'get_query_results', 'select_object_content'
                }
                self.free_operations = {
                    'list_metrics', 'describe_alarms', 'list_dashboards',
                    'describe_log_groups', 'get_dashboard', 'pricing_calculations'
                }
                self.paid_operations = {
                    'cost_explorer_analysis', 'aws_config_compliance',
                    'cloudtrail_usage_patterns', 'minimal_cost_metrics'
                }
                self.operation_calls = []
                self.cost_incurring_calls = []
            
            def validate_operation(self, operation_name: str, cost_preferences: dict = None):
                """Validate that an operation follows cost constraints."""
                self.operation_calls.append(operation_name)
                
                if operation_name in self.forbidden_operations:
                    raise ValueError(f"FORBIDDEN: Operation {operation_name} is never allowed")
                
                if operation_name in self.free_operations:
                    return True  # Always allowed
                
                if operation_name in self.paid_operations:
                    if not cost_preferences:
                        raise ValueError(f"PAID: Operation {operation_name} requires explicit consent")
                    
                    # Check specific cost flags
                    required_flag = self._get_required_flag(operation_name)
                    if not cost_preferences.get(required_flag, False):
                        raise ValueError(f"PAID: Operation {operation_name} requires {required_flag}=True")
                    
                    self.cost_incurring_calls.append(operation_name)
                    return True
                
                raise ValueError(f"UNKNOWN: Operation {operation_name} not defined")
            
            def _get_required_flag(self, operation_name: str) -> str:
                """Get required cost flag for paid operation."""
                flag_mapping = {
                    'cost_explorer_analysis': 'allow_cost_explorer',
                    'aws_config_compliance': 'allow_aws_config',
                    'cloudtrail_usage_patterns': 'allow_cloudtrail',
                    'minimal_cost_metrics': 'allow_minimal_cost_metrics'
                }
                return flag_mapping.get(operation_name, 'unknown_flag')
            
            def get_operation_summary(self):
                """Get summary of operations called."""
                return {
                    "total_operations": len(self.operation_calls),
                    "unique_operations": len(set(self.operation_calls)),
                    "operations": list(set(self.operation_calls)),
                    "cost_incurring_calls": self.cost_incurring_calls,
                    "forbidden_attempted": [op for op in self.operation_calls if op in self.forbidden_operations],
                    "free_operations_used": [op for op in self.operation_calls if op in self.free_operations],
                    "paid_operations_used": [op for op in self.operation_calls if op in self.paid_operations]
                }
        
        return CostConstraintValidator()
    
    def test_default_cost_preferences_prevent_charges(self, cost_validator):
        """Test that default cost preferences prevent all charges."""
        cost_controller = CostController()
        default_prefs = cost_controller.default_preferences
        
        # All paid operations should be blocked with default preferences
        paid_operations = [
            'cost_explorer_analysis', 'aws_config_compliance',
            'cloudtrail_usage_patterns', 'minimal_cost_metrics'
        ]
        
        for operation in paid_operations:
            is_allowed, reason = cost_controller.validate_operation(operation, default_prefs)
            assert is_allowed is False
            assert 'allow_' in reason  # Should mention required flag
        
        # Free operations should be allowed
        free_operations = [
            'list_metrics', 'describe_alarms', 'list_dashboards',
            'describe_log_groups', 'get_dashboard', 'pricing_calculations'
        ]
        
        for operation in free_operations:
            is_allowed, reason = cost_controller.validate_operation(operation, default_prefs)
            assert is_allowed is True
            assert 'Free operation' in reason
    
    def test_forbidden_operations_never_allowed(self, cost_validator):
        """Test that forbidden operations are never allowed regardless of preferences."""
        cost_controller = CostController()
        
        # Enable all paid features
        all_enabled_prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        
        forbidden_operations = ['logs_insights_queries', 'extensive_metric_retrieval']
        
        for operation in forbidden_operations:
            is_allowed, reason = cost_controller.validate_operation(operation, all_enabled_prefs)
            assert is_allowed is False
            assert 'Forbidden operation' in reason
    
    def test_cost_transparency_tracking(self):
        """Test that all cost-incurring operations are properly tracked."""
        cost_controller = CostController()
        
        # Enable some paid features
        prefs = CostPreferences(allow_cost_explorer=True, allow_minimal_cost_metrics=True)
        
        # Track cost decisions
        cost_decisions = []
        
        with patch('playbooks.cloudwatch.cost_controller.log_cloudwatch_operation') as mock_log:
            # Test various operations
            operations_to_test = [
                ('list_metrics', True, 'Free operation'),
                ('cost_explorer_analysis', True, 'allowed by allow_cost_explorer'),
                ('minimal_cost_metrics', True, 'allowed by allow_minimal_cost_metrics'),
                ('aws_config_compliance', False, 'requires allow_aws_config'),
                ('logs_insights_queries', False, 'Forbidden operation')
            ]
            
            for operation, expected_allowed, expected_reason_type in operations_to_test:
                is_allowed, reason = cost_controller.validate_operation(operation, prefs)
                
                cost_controller.log_cost_decision(operation, prefs, is_allowed, reason)
                
                assert is_allowed == expected_allowed
                assert expected_reason_type.lower() in reason.lower()
            
            # Verify all decisions were logged
            assert mock_log.call_count == len(operations_to_test)
            
            # Check that cost-incurring operations are identified
            cost_incurring_calls = [
                call for call in mock_log.call_args_list
                if call[1].get('is_allowed') and 'cost_explorer' in call[1].get('operation_name', '') or 'minimal_cost' in call[1].get('operation_name', '')
            ]
            
            assert len(cost_incurring_calls) == 2  # cost_explorer_analysis and minimal_cost_metrics
    
    def test_cost_estimation_accuracy(self):
        """Test accuracy of cost estimation calculations."""
        cost_controller = CostController()
        
        # Test with no paid features (should be $0)
        no_cost_prefs = CostPreferences()
        estimate = cost_controller.estimate_cost({}, no_cost_prefs)
        assert estimate.total_estimated_cost == 0.0
        
        # Test with all paid features
        all_paid_prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        
        # Test different scopes
        small_scope = {"lookback_days": 7, "log_group_names": ["group1"]}
        large_scope = {
            "lookback_days": 90,
            "log_group_names": [f"group{i}" for i in range(10)],
            "alarm_names": [f"alarm{i}" for i in range(20)]
        }
        
        small_estimate = cost_controller.estimate_cost(small_scope, all_paid_prefs)
        large_estimate = cost_controller.estimate_cost(large_scope, all_paid_prefs)
        
        # Large scope should cost more
        assert large_estimate.total_estimated_cost > small_estimate.total_estimated_cost
        
        # Both should have reasonable costs (not excessive)
        assert small_estimate.total_estimated_cost < 1.0  # Less than $1
        assert large_estimate.total_estimated_cost < 5.0  # Less than $5
        
        # Verify cost breakdown
        assert len(small_estimate.enabled_operations) > 0
        # Disabled operations should only include forbidden operations, not paid operations that are enabled
        forbidden_operations = ['logs_insights_queries', 'extensive_metric_retrieval']
        assert set(small_estimate.disabled_operations) == set(forbidden_operations)
        assert len(large_estimate.enabled_operations) > 0
    
    def test_runtime_cost_validation(self):
        """Test runtime validation prevents unauthorized operations."""
        cost_controller = CostController()
        
        # Simulate runtime validation during operation execution
        class MockOperationExecutor:
            def __init__(self, cost_controller, cost_preferences):
                self.cost_controller = cost_controller
                self.cost_preferences = cost_preferences
                self.executed_operations = []
            
            def execute_operation(self, operation_name):
                """Execute operation with runtime cost validation."""
                is_allowed, reason = self.cost_controller.validate_operation(
                    operation_name, self.cost_preferences
                )
                
                if not is_allowed:
                    raise PermissionError(f"Operation {operation_name} not allowed: {reason}")
                
                self.executed_operations.append(operation_name)
                return f"Executed {operation_name}"
        
        # Test with restrictive preferences
        restrictive_prefs = CostPreferences()  # All False
        executor = MockOperationExecutor(cost_controller, restrictive_prefs)
        
        # Free operations should work
        result = executor.execute_operation('list_metrics')
        assert 'Executed list_metrics' in result
        
        # Paid operations should be blocked
        with pytest.raises(PermissionError, match="not allowed"):
            executor.execute_operation('cost_explorer_analysis')
        
        # Forbidden operations should be blocked
        with pytest.raises(PermissionError, match="not allowed"):
            executor.execute_operation('logs_insights_queries')
        
        # Test with permissive preferences
        permissive_prefs = CostPreferences(allow_cost_explorer=True)
        executor_permissive = MockOperationExecutor(cost_controller, permissive_prefs)
        
        # Now cost explorer should work
        result = executor_permissive.execute_operation('cost_explorer_analysis')
        assert 'Executed cost_explorer_analysis' in result
        
        # But forbidden operations still blocked
        with pytest.raises(PermissionError, match="not allowed"):
            executor_permissive.execute_operation('logs_insights_queries')


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestServiceCostConstraints:
    """Test cost constraints at the service level."""
    
    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for testing."""
        with patch('services.cloudwatch_service.boto3') as mock_boto3:
            mock_cloudwatch = MagicMock()
            mock_logs = MagicMock()
            mock_ce = MagicMock()
            
            mock_boto3.client.side_effect = lambda service, **kwargs: {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce
            }[service]
            
            yield {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce
            }
    
    @pytest.fixture
    def cloudwatch_service(self, mock_boto3_clients):
        """Create CloudWatch service for testing."""
        return CloudWatchService(region='us-east-1')
    
    @pytest.mark.asyncio
    async def test_free_operations_no_cost_validation_required(self, cloudwatch_service, mock_boto3_clients):
        """Test that free operations work without cost validation."""
        # Mock successful responses
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{'Metrics': [], 'MetricAlarms': [], 'CompositeAlarms': []}]
        mock_boto3_clients['cloudwatch'].get_paginator.return_value = mock_paginator
        
        # Free operations should work with default (restrictive) cost preferences
        operations = [
            cloudwatch_service.list_metrics(),
            cloudwatch_service.describe_alarms(),
            cloudwatch_service.list_dashboards()
        ]
        
        results = await asyncio.gather(*operations)
        
        # All should succeed
        for result in results:
            assert result.success is True
            assert result.cost_incurred is False
            assert result.operation_type == 'free'
    
    @pytest.mark.asyncio
    async def test_paid_operations_require_explicit_consent(self, cloudwatch_service, mock_boto3_clients):
        """Test that paid operations require explicit consent."""
        # Mock successful metric statistics response
        mock_boto3_clients['cloudwatch'].get_metric_statistics.return_value = {
            'Datapoints': [{'Timestamp': datetime.now(timezone.utc), 'Sum': 1000000.0}]
        }
        
        # Paid operation should fail with default preferences
        result = await cloudwatch_service.get_log_group_incoming_bytes()
        assert result.success is False
        assert 'not allowed' in result.error_message.lower()
        assert result.cost_incurred is False
        
        # Enable minimal cost metrics and try again
        cloudwatch_service.update_cost_preferences({'allow_minimal_cost_metrics': True})
        
        result = await cloudwatch_service.get_log_group_incoming_bytes()
        assert result.success is True
        assert result.cost_incurred is True
        assert result.operation_type == 'paid'
    
    @pytest.mark.asyncio
    async def test_cost_tracking_in_service_operations(self, cloudwatch_service, mock_boto3_clients):
        """Test that cost-incurring operations are properly tracked."""
        # Enable paid operations
        cloudwatch_service.update_cost_preferences({
            'allow_minimal_cost_metrics': True
        })
        
        # Mock responses
        mock_boto3_clients['cloudwatch'].get_metric_statistics.return_value = {
            'Datapoints': [{'Timestamp': datetime.now(timezone.utc), 'Sum': 1000000.0}]
        }
        
        # Execute mix of free and paid operations
        await cloudwatch_service.list_metrics()  # Free
        await cloudwatch_service.get_log_group_incoming_bytes()  # Paid
        await cloudwatch_service.describe_alarms()  # Free
        
        # Check service statistics
        stats = cloudwatch_service.get_service_statistics()
        
        assert stats['total_operations'] == 4  # list_metrics + describe_log_groups (internal) + get_metric_statistics + describe_alarms
        assert stats['cost_incurred'] is True
        assert len(stats['cost_incurring_operations']) == 1
        assert 'minimal_cost_metrics' in stats['cost_incurring_operations']
    
    def test_cost_preference_validation_at_service_level(self, cloudwatch_service):
        """Test cost preference validation at service initialization."""
        # Test valid preferences
        valid_prefs = {
            'allow_cost_explorer': True,
            'allow_minimal_cost_metrics': False
        }
        
        cloudwatch_service.update_cost_preferences(valid_prefs)
        assert cloudwatch_service.cost_preferences.allow_cost_explorer is True
        assert cloudwatch_service.cost_preferences.allow_minimal_cost_metrics is False
        
        # Test invalid preferences (should handle gracefully)
        invalid_prefs = {
            'allow_cost_explorer': 'invalid_value',
            'unknown_preference': True
        }
        
        # Should not raise exception but log warnings
        with patch.object(cloudwatch_service.cost_controller, 'logger') as mock_logger:
            cloudwatch_service.update_cost_preferences(invalid_prefs)
            # Should have logged warnings for invalid values
            assert mock_logger.warning.called
            
            # Should have logged warnings about invalid values
            assert mock_logger.warning.called


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestOrchestratorCostConstraints:
    """Test cost constraints at the orchestrator level."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so, \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_ae:
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
            orchestrator.service_orchestrator = mock_so.return_value
            orchestrator.analysis_engine = mock_ae.return_value
            
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_cost_validation(self, mock_orchestrator):
        """Test cost validation in comprehensive analysis."""
        # Mock analysis engine to return different results based on analysis type
        async def mock_run_analysis(analysis_type, **kwargs):
            if analysis_type == "general_spend":
                return {
                    "status": "success",
                    "cost_incurred": False,
                    "cost_incurring_operations": []
                }
            elif analysis_type == "metrics_optimization":
                return {
                    "status": "success",
                    "cost_incurred": True,
                    "cost_incurring_operations": ["minimal_cost_metrics"]
                }
            elif analysis_type == "logs_optimization":
                return {
                    "status": "success",
                    "cost_incurred": False,
                    "cost_incurring_operations": []
                }
            elif analysis_type == "alarms_and_dashboards":
                return {
                    "status": "success",
                    "cost_incurred": False,
                    "cost_incurring_operations": []
                }
            else:
                return {
                    "status": "success",
                    "cost_incurred": False,
                    "cost_incurring_operations": []
                }
        
        mock_orchestrator.analysis_engine.run_analysis.side_effect = mock_run_analysis
        
        # Also mock the comprehensive analysis method
        async def mock_run_comprehensive_analysis(**kwargs):
            return {
                'analysis_results': {
                    'general_spend': {
                        'status': 'success',
                        'cost_incurred': False,
                        'cost_incurring_operations': []
                    },
                    'metrics_optimization': {
                        'status': 'success',
                        'cost_incurred': True,
                        'cost_incurring_operations': ['minimal_cost_metrics']
                    },
                    'logs_optimization': {
                        'status': 'success',
                        'cost_incurred': False,
                        'cost_incurring_operations': []
                    },
                    'alarms_and_dashboards': {
                        'status': 'success',
                        'cost_incurred': False,
                        'cost_incurring_operations': []
                    }
                },
                'status': 'success'
            }
        
        mock_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_run_comprehensive_analysis
        
        # Execute comprehensive analysis with mixed cost preferences
        result = await mock_orchestrator.execute_comprehensive_analysis(
            allow_minimal_cost_metrics=True  # Only enable minimal cost metrics
        )
        
        assert result['analysis_results']['general_spend']['status'] == 'success'
        assert result['analysis_results']['metrics_optimization']['cost_incurred'] is True
        assert 'minimal_cost_metrics' in result['analysis_results']['metrics_optimization']['cost_incurring_operations']
        
        # Verify cost transparency
        assert 'orchestrator_metadata' in result
        assert 'cost_estimate' in result['orchestrator_metadata']
        # The comprehensive analysis should show cost information in orchestrator metadata
        cost_estimate = result['orchestrator_metadata']['cost_estimate']
        assert cost_estimate['total_estimated_cost'] > 0
    
    def test_cost_estimate_before_execution(self, mock_orchestrator):
        """Test cost estimation before executing analysis."""
        analysis_scope = {
            'lookback_days': 30,
            'log_group_names': ['group1', 'group2'],
            'alarm_names': ['alarm1']
        }
        
        # Test with no paid features
        cost_prefs_free = CostPreferences()
        estimate_free = mock_orchestrator.get_cost_estimate(**analysis_scope, **cost_prefs_free.__dict__)
        
        assert estimate_free['cost_estimate']['total_estimated_cost'] == 0.0
        
        # Test with paid features
        cost_prefs_paid = CostPreferences(
            allow_cost_explorer=True,
            allow_minimal_cost_metrics=True
        )
        estimate_paid = mock_orchestrator.get_cost_estimate(**analysis_scope, **cost_prefs_paid.__dict__)
        
        assert estimate_paid['cost_estimate']['total_estimated_cost'] > 0.0
        
        # Verify cost transparency
        assert len(estimate_paid['cost_estimate']['enabled_operations']) > len(estimate_free['cost_estimate']['enabled_operations'])
        assert len(estimate_paid['cost_estimate']['disabled_operations']) < len(estimate_free['cost_estimate']['disabled_operations'])
    
    def test_cost_preference_validation_comprehensive(self, mock_orchestrator):
        """Test comprehensive cost preference validation."""
        # Test inputs that should fail validation
        invalid_inputs = [
            {'allow_cost_explorer': 'maybe'},  # Invalid boolean
            {'allow_cost_explorer': []},       # Wrong type
        ]
        
        for invalid_input in invalid_inputs:
            result = mock_orchestrator.validate_cost_preferences(**invalid_input)
            
            # These should fail validation
            assert result['valid'] is False
            assert len(result['errors']) > 0
        
        # Test inputs that should pass with warnings
        warning_inputs = [
            {'unknown_preference': True},   # Unknown preference (doesn't start with allow_)
            {'allow_minimal_cost_metrics': 'yes', 'allow_cost_explorer': 'no'}  # Valid string booleans
        ]
        
        for warning_input in warning_inputs:
            result = mock_orchestrator.validate_cost_preferences(**warning_input)
            
            # These should succeed with warnings
            assert result['valid'] is True
            assert len(result['warnings']) > 0
    
    @pytest.mark.asyncio
    async def test_cost_constraint_enforcement_during_execution(self, mock_orchestrator):
        """Test that cost constraints are enforced during execution."""
        # Mock analysis engine to simulate cost validation
        async def mock_run_analysis(analysis_type, **kwargs):
            # Check if cost-incurring operations are allowed based on the kwargs passed by orchestrator
            allow_cost_explorer = kwargs.get('allow_cost_explorer', False)
            
            if analysis_type == 'general_spend' and not allow_cost_explorer:
                return {
                    "status": "success",
                    "analysis_type": analysis_type,
                    "data": {"limited_analysis": True},
                    "cost_incurred": False,
                    "cost_incurring_operations": [],
                    "fallback_used": True,
                    "primary_data_source": "cloudwatch_config"
                }
            else:
                return {
                    "status": "success",
                    "analysis_type": analysis_type,
                    "data": {"full_analysis": True},
                    "cost_incurred": True,
                    "cost_incurring_operations": ["cost_explorer_analysis"],
                    "primary_data_source": "cost_explorer"
                }
        
        mock_orchestrator.analysis_engine.run_analysis.side_effect = mock_run_analysis
        
        # Test with restrictive preferences
        result_restrictive = await mock_orchestrator.execute_analysis(
            'general_spend',
            allow_cost_explorer=False
        )
        
        assert result_restrictive['status'] == 'success'
        assert result_restrictive['cost_incurred'] is False
        assert result_restrictive['fallback_used'] is True
        assert result_restrictive['data']['limited_analysis'] is True
        
        # Test with permissive preferences
        result_permissive = await mock_orchestrator.execute_analysis(
            'general_spend',
            allow_cost_explorer=True
        )
        
        assert result_permissive['status'] == 'success'
        assert result_permissive['cost_incurred'] is True
        assert result_permissive['data']['full_analysis'] is True


if __name__ == "__main__":
    pytest.main([__file__])