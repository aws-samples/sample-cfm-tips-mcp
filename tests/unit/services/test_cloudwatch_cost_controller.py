"""
Unit tests for CloudWatch CostController.

Tests cost preference validation, functionality coverage calculation,
cost estimation, and runtime validation functionality with strict
no-cost constraint validation.
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from playbooks.cloudwatch.cost_controller import (
    CostController, CostPreferences, OperationType, 
    OperationDefinition, CostEstimate
)


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestCostPreferences:
    """Test CostPreferences dataclass validation."""
    
    def test_default_preferences(self):
        """Test default cost preferences are all False."""
        prefs = CostPreferences()
        assert prefs.allow_cost_explorer is False
        assert prefs.allow_aws_config is False
        assert prefs.allow_cloudtrail is False
        assert prefs.allow_minimal_cost_metrics is False
    
    def test_valid_preferences(self):
        """Test valid preference initialization."""
        prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=False,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=False
        )
        assert prefs.allow_cost_explorer is True
        assert prefs.allow_aws_config is False
        assert prefs.allow_cloudtrail is True
        assert prefs.allow_minimal_cost_metrics is False
    
    def test_invalid_preference_types(self):
        """Test validation of invalid preference types."""
        with pytest.raises(ValueError, match="allow_cost_explorer must be a boolean"):
            CostPreferences(allow_cost_explorer="invalid")
        
        with pytest.raises(ValueError, match="allow_aws_config must be a boolean"):
            CostPreferences(allow_aws_config=123)


@pytest.mark.unit
class TestCostController:
    """Test CostController functionality."""
    
    @pytest.fixture
    def cost_controller(self):
        """Create a CostController instance for testing."""
        return CostController()
    
    def test_initialization(self, cost_controller):
        """Test CostController initialization."""
        assert cost_controller.default_preferences is not None
        assert isinstance(cost_controller.operation_definitions, dict)
        assert len(cost_controller.operation_definitions) > 0
        
        # Verify operation categories are present
        free_ops = [op for op in cost_controller.operation_definitions.values() 
                   if op.operation_type == OperationType.FREE]
        paid_ops = [op for op in cost_controller.operation_definitions.values() 
                   if op.operation_type == OperationType.PAID]
        forbidden_ops = [op for op in cost_controller.operation_definitions.values() 
                        if op.operation_type == OperationType.FORBIDDEN]
        
        assert len(free_ops) > 0, "Should have free operations"
        assert len(paid_ops) > 0, "Should have paid operations"
        assert len(forbidden_ops) > 0, "Should have forbidden operations"
    
    def test_validate_and_sanitize_preferences_valid(self, cost_controller):
        """Test validation of valid preferences."""
        input_prefs = {
            "allow_cost_explorer": True,
            "allow_aws_config": False,
            "allow_cloudtrail": "true",
            "allow_minimal_cost_metrics": 1
        }
        
        result = cost_controller.validate_and_sanitize_preferences(input_prefs)
        
        assert isinstance(result, CostPreferences)
        assert result.allow_cost_explorer is True
        assert result.allow_aws_config is False
        assert result.allow_cloudtrail is True
        assert result.allow_minimal_cost_metrics is True
    
    def test_validate_and_sanitize_preferences_string_conversion(self, cost_controller):
        """Test string to boolean conversion."""
        test_cases = [
            ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
            ("false", False), ("False", False), ("0", False), ("no", False), ("off", False)
        ]
        
        for string_val, expected_bool in test_cases:
            input_prefs = {"allow_cost_explorer": string_val}
            result = cost_controller.validate_and_sanitize_preferences(input_prefs)
            assert result.allow_cost_explorer == expected_bool
    
    def test_validate_and_sanitize_preferences_invalid(self, cost_controller):
        """Test validation of invalid preferences (now handles gracefully)."""
        # Invalid string value should be handled gracefully with warning
        result = cost_controller.validate_and_sanitize_preferences({
            "allow_cost_explorer": "invalid_value"
        })
        assert result.allow_cost_explorer is False  # Should use default
        
        # Invalid type should be handled gracefully with warning
        result = cost_controller.validate_and_sanitize_preferences({
            "allow_cost_explorer": []
        })
        assert result.allow_cost_explorer is False  # Should use default
    
    def test_validate_and_sanitize_preferences_unknown_keys(self, cost_controller):
        """Test handling of unknown preference keys."""
        input_prefs = {
            "allow_cost_explorer": True,
            "unknown_preference": True,
            "another_unknown": False
        }
        
        with patch('playbooks.cloudwatch.cost_controller.log_cloudwatch_operation') as mock_log_op:
            result = cost_controller.validate_and_sanitize_preferences(input_prefs)
            
            assert result.allow_cost_explorer is True
            # Check that log_cloudwatch_operation was called for unknown preferences
            calls = mock_log_op.call_args_list
            unknown_call = next((call for call in calls if len(call[0]) > 1 and call[0][1] == "unknown_preferences_ignored"), None)
            assert unknown_call is not None
            assert "unknown_keys" in unknown_call[1]
    
    def test_get_functionality_coverage_all_disabled(self, cost_controller):
        """Test functionality coverage with all paid features disabled."""
        prefs = CostPreferences()  # All False by default
        coverage = cost_controller.get_functionality_coverage(prefs)
        
        assert "overall_coverage" in coverage
        assert "by_category" in coverage
        assert "free_tier_coverage" in coverage
        
        # Should have some coverage from free operations
        assert coverage["overall_coverage"] > 0
        assert coverage["free_tier_coverage"] > 0
        assert coverage["by_category"]["cost_explorer"] == 0.0
        assert coverage["by_category"]["aws_config"] == 0.0
    
    def test_get_functionality_coverage_all_enabled(self, cost_controller):
        """Test functionality coverage with all paid features enabled."""
        prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        coverage = cost_controller.get_functionality_coverage(prefs)
        
        # Should have high coverage with all features enabled
        assert coverage["overall_coverage"] > 90  # Should be close to 100%
        assert coverage["by_category"]["cost_explorer"] > 0
        assert coverage["by_category"]["aws_config"] > 0
        assert coverage["by_category"]["cloudtrail"] > 0
        assert coverage["by_category"]["minimal_cost_metrics"] > 0
    
    def test_estimate_cost_no_paid_features(self, cost_controller):
        """Test cost estimation with no paid features enabled."""
        prefs = CostPreferences()  # All False
        analysis_scope = {"lookback_days": 30}
        
        estimate = cost_controller.estimate_cost(analysis_scope, prefs)
        
        assert isinstance(estimate, CostEstimate)
        assert estimate.total_estimated_cost == 0.0
        assert len(estimate.enabled_operations) > 0  # Should have free operations
        assert len(estimate.disabled_operations) > 0  # Should have disabled paid operations
    
    def test_estimate_cost_with_paid_features(self, cost_controller):
        """Test cost estimation with paid features enabled."""
        prefs = CostPreferences(allow_cost_explorer=True, allow_minimal_cost_metrics=True)
        analysis_scope = {
            "lookback_days": 60,
            "log_group_names": ["group1", "group2"],
            "alarm_names": ["alarm1", "alarm2", "alarm3"]
        }
        
        estimate = cost_controller.estimate_cost(analysis_scope, prefs)
        
        assert estimate.total_estimated_cost > 0.0
        assert "cost_explorer_analysis" in estimate.enabled_operations
        assert "minimal_cost_metrics" in estimate.enabled_operations
        assert "aws_config_compliance" in estimate.disabled_operations
    
    def test_validate_operation_free(self, cost_controller):
        """Test validation of free operations."""
        prefs = CostPreferences()  # All False
        
        is_allowed, reason = cost_controller.validate_operation("list_metrics", prefs)
        
        assert is_allowed is True
        assert "Free operation" in reason
    
    def test_validate_operation_paid_disabled(self, cost_controller):
        """Test validation of disabled paid operations."""
        prefs = CostPreferences()  # All False
        
        is_allowed, reason = cost_controller.validate_operation("cost_explorer_analysis", prefs)
        
        assert is_allowed is False
        assert "allow_cost_explorer" in reason
    
    def test_validate_operation_paid_enabled(self, cost_controller):
        """Test validation of enabled paid operations."""
        prefs = CostPreferences(allow_cost_explorer=True)
        
        is_allowed, reason = cost_controller.validate_operation("cost_explorer_analysis", prefs)
        
        assert is_allowed is True
        assert "allow_cost_explorer" in reason
    
    def test_validate_operation_forbidden(self, cost_controller):
        """Test validation of forbidden operations."""
        prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        
        is_allowed, reason = cost_controller.validate_operation("logs_insights_queries", prefs)
        
        assert is_allowed is False
        assert "Forbidden operation" in reason
    
    def test_validate_operation_unknown(self, cost_controller):
        """Test validation of unknown operations."""
        prefs = CostPreferences()
        
        is_allowed, reason = cost_controller.validate_operation("unknown_operation", prefs)
        
        assert is_allowed is False
        assert "Unknown operation" in reason
    
    def test_get_allowed_operations(self, cost_controller):
        """Test getting allowed and disallowed operations."""
        prefs = CostPreferences(allow_cost_explorer=True)
        
        operations = cost_controller.get_allowed_operations(prefs)
        
        assert "allowed" in operations
        assert "disallowed" in operations
        assert isinstance(operations["allowed"], list)
        assert isinstance(operations["disallowed"], list)
        
        # Should have free operations in allowed
        assert "list_metrics" in operations["allowed"]
        assert "describe_alarms" in operations["allowed"]
        
        # Should have cost explorer in allowed
        assert "cost_explorer_analysis" in operations["allowed"]
        
        # Should have disabled paid operations in disallowed
        assert "aws_config_compliance" in operations["disallowed"]
        
        # Should have forbidden operations in disallowed
        assert "logs_insights_queries" in operations["disallowed"]
    
    def test_log_cost_decision(self, cost_controller):
        """Test cost decision logging."""
        prefs = CostPreferences(allow_cost_explorer=True)
        
        with patch('playbooks.cloudwatch.cost_controller.log_cloudwatch_operation') as mock_log_op:
            cost_controller.log_cost_decision(
                "cost_explorer_analysis", prefs, True, "Test reason"
            )
            
            mock_log_op.assert_called_once()
            args, kwargs = mock_log_op.call_args
            assert args[1] == "cost_decision"  # Operation name
            assert kwargs["operation_name"] == "cost_explorer_analysis"
            assert kwargs["is_allowed"] is True
            assert kwargs["reason"] == "Test reason"
    
    def test_get_cost_summary(self, cost_controller):
        """Test comprehensive cost summary generation."""
        prefs = CostPreferences(allow_cost_explorer=True, allow_aws_config=True)
        
        summary = cost_controller.get_cost_summary(prefs)
        
        assert "preferences" in summary
        assert "functionality_coverage" in summary
        assert "allowed_operations" in summary
        assert "cost_range" in summary
        assert "enabled_paid_features" in summary
        assert "total_paid_features" in summary
        
        # Verify cost range structure
        cost_range = summary["cost_range"]
        assert "minimum_cost" in cost_range
        assert "maximum_possible_cost" in cost_range
        assert "current_maximum_cost" in cost_range
        
        assert cost_range["minimum_cost"] == 0.0
        assert cost_range["maximum_possible_cost"] > 0.0
        assert cost_range["current_maximum_cost"] > 0.0
        assert cost_range["current_maximum_cost"] <= cost_range["maximum_possible_cost"]


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestCostControllerOperationDefinitions:
    """Test operation definitions completeness and cost constraints."""
    
    @pytest.fixture
    def cost_controller(self):
        """Create a CostController instance for testing."""
        return CostController()
    
    def test_operation_definitions_completeness(self, cost_controller):
        """Test that operation definitions are complete and valid."""
        operations = cost_controller.operation_definitions
        
        # Check that we have the expected operations
        expected_free_ops = [
            "list_metrics", "describe_alarms", "list_dashboards", 
            "describe_log_groups", "get_dashboard", "pricing_calculations"
        ]
        expected_paid_ops = [
            "cost_explorer_analysis", "aws_config_compliance", 
            "cloudtrail_usage_patterns", "minimal_cost_metrics"
        ]
        expected_forbidden_ops = [
            "logs_insights_queries", "extensive_metric_retrieval"
        ]
        
        for op_name in expected_free_ops:
            assert op_name in operations
            assert operations[op_name].operation_type == OperationType.FREE
            assert operations[op_name].functionality_weight > 0
        
        for op_name in expected_paid_ops:
            assert op_name in operations
            assert operations[op_name].operation_type == OperationType.PAID
            assert operations[op_name].cost_flag is not None
            assert operations[op_name].estimated_cost > 0
            assert operations[op_name].functionality_weight > 0
        
        for op_name in expected_forbidden_ops:
            assert op_name in operations
            assert operations[op_name].operation_type == OperationType.FORBIDDEN
            assert operations[op_name].functionality_weight == 0.0
    
    def test_free_operations_coverage(self, cost_controller):
        """Test that free operations provide substantial coverage."""
        operations = cost_controller.operation_definitions
        
        free_weight = sum(
            op.functionality_weight for op in operations.values()
            if op.operation_type == OperationType.FREE
        )
        total_weight = sum(op.functionality_weight for op in operations.values())
        
        free_coverage = (free_weight / total_weight) * 100
        
        # Free operations should provide at least 50% of functionality
        assert free_coverage >= 50.0, f"Free operations only provide {free_coverage}% coverage"
    
    def test_cost_flag_mapping(self, cost_controller):
        """Test that all paid operations have valid cost flags."""
        operations = cost_controller.operation_definitions
        valid_cost_flags = {
            "allow_cost_explorer", "allow_aws_config", 
            "allow_cloudtrail", "allow_minimal_cost_metrics"
        }
        
        for op in operations.values():
            if op.operation_type == OperationType.PAID:
                assert op.cost_flag in valid_cost_flags, f"Invalid cost flag: {op.cost_flag}"
    
    def test_forbidden_operations_zero_weight(self, cost_controller):
        """Test that forbidden operations have zero functionality weight."""
        operations = cost_controller.operation_definitions
        
        for op in operations.values():
            if op.operation_type == OperationType.FORBIDDEN:
                assert op.functionality_weight == 0.0, f"Forbidden operation {op.name} has non-zero weight"
    
    def test_cost_estimates_reasonable(self, cost_controller):
        """Test that cost estimates are reasonable."""
        operations = cost_controller.operation_definitions
        
        for op in operations.values():
            if op.operation_type == OperationType.PAID:
                # Cost should be reasonable (between $0.001 and $1.00 per operation)
                assert 0.001 <= op.estimated_cost <= 1.0, f"Unreasonable cost for {op.name}: ${op.estimated_cost}"


@pytest.mark.unit
@pytest.mark.no_cost_validation
class TestCostControllerConstraintValidation:
    """Test cost constraint validation and enforcement."""
    
    @pytest.fixture
    def cost_controller(self):
        """Create a CostController instance for testing."""
        return CostController()
    
    def test_default_preferences_minimize_cost(self, cost_controller):
        """Test that default preferences minimize cost."""
        default_prefs = cost_controller.default_preferences
        
        # All paid features should be disabled by default
        assert default_prefs.allow_cost_explorer is False
        assert default_prefs.allow_aws_config is False
        assert default_prefs.allow_cloudtrail is False
        assert default_prefs.allow_minimal_cost_metrics is False
        
        # Verify zero cost with defaults
        estimate = cost_controller.estimate_cost({}, default_prefs)
        assert estimate.total_estimated_cost == 0.0
    
    def test_forbidden_operations_never_allowed(self, cost_controller):
        """Test that forbidden operations are never allowed regardless of preferences."""
        # Enable all paid features
        prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        
        forbidden_ops = [
            "logs_insights_queries", "extensive_metric_retrieval"
        ]
        
        for op_name in forbidden_ops:
            is_allowed, reason = cost_controller.validate_operation(op_name, prefs)
            assert is_allowed is False
            assert "Forbidden operation" in reason
    
    def test_cost_transparency_tracking(self, cost_controller):
        """Test that cost decisions are properly tracked and logged."""
        prefs = CostPreferences(allow_cost_explorer=True)
        
        with patch('playbooks.cloudwatch.cost_controller.log_cloudwatch_operation') as mock_log_op:
            # Test allowed operation
            cost_controller.log_cost_decision(
                "cost_explorer_analysis", prefs, True, "Enabled by user"
            )
            
            # Test disallowed operation
            cost_controller.log_cost_decision(
                "aws_config_compliance", prefs, False, "Disabled by user"
            )
            
            assert mock_log_op.call_count == 2
            
            # Check operation details
            calls = mock_log_op.call_args_list
            assert calls[0][1]["operation_name"] == "cost_explorer_analysis"
            assert calls[0][1]["is_allowed"] is True
            assert calls[1][1]["operation_name"] == "aws_config_compliance"
            assert calls[1][1]["is_allowed"] is False
    
    def test_functionality_coverage_calculation_accuracy(self, cost_controller):
        """Test accuracy of functionality coverage calculations."""
        # Test with no paid features
        prefs_none = CostPreferences()
        coverage_none = cost_controller.get_functionality_coverage(prefs_none)
        
        # Test with all paid features
        prefs_all = CostPreferences(
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=True
        )
        coverage_all = cost_controller.get_functionality_coverage(prefs_all)
        
        # Coverage with all features should be higher than with none
        assert coverage_all["overall_coverage"] > coverage_none["overall_coverage"]
        
        # Free tier coverage should be the same regardless of paid features
        assert coverage_all["free_tier_coverage"] == coverage_none["free_tier_coverage"]
        
        # All coverage percentages should be between 0 and 100
        for category_coverage in coverage_all["by_category"].values():
            assert 0.0 <= category_coverage <= 100.0
    
    def test_cost_estimation_scope_scaling(self, cost_controller):
        """Test that cost estimation scales properly with analysis scope."""
        prefs = CostPreferences(allow_cost_explorer=True, allow_minimal_cost_metrics=True)
        
        # Small scope
        small_scope = {"lookback_days": 7, "log_group_names": ["group1"]}
        small_estimate = cost_controller.estimate_cost(small_scope, prefs)
        
        # Large scope
        large_scope = {
            "lookback_days": 90,
            "log_group_names": ["group1", "group2", "group3", "group4"],
            "alarm_names": [f"alarm{i}" for i in range(20)]
        }
        large_estimate = cost_controller.estimate_cost(large_scope, prefs)
        
        # Large scope should cost more than small scope
        assert large_estimate.total_estimated_cost > small_estimate.total_estimated_cost
        
        # Both should have the same enabled/disabled operations
        assert set(large_estimate.enabled_operations) == set(small_estimate.enabled_operations)
        assert set(large_estimate.disabled_operations) == set(small_estimate.disabled_operations)


if __name__ == "__main__":
    pytest.main([__file__])