"""
Unit tests for BaseAnalyzer class.

Tests the abstract base class functionality including parameter validation,
error handling, recommendation creation, and performance monitoring integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, List, Any

from playbooks.s3.base_analyzer import BaseAnalyzer, AnalyzerRegistry, get_analyzer_registry


class MockAnalyzer(BaseAnalyzer):
    """Concrete implementation of BaseAnalyzer for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_type = "test_analyzer"
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """Test implementation of analyze method."""
        return {
            "status": "success",
            "data": {"test_metric": 100},
            "execution_time": 1.0
        }
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test implementation of get_recommendations method."""
        return [
            {
                "type": "test_recommendation",
                "priority": "high",
                "title": "Test Recommendation",
                "description": "This is a test recommendation"
            }
        ]


class MockFailingAnalyzer(BaseAnalyzer):
    """Analyzer that fails for testing error handling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_type = "failing_analyzer"
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """Failing implementation for testing."""
        raise ValueError("Test error for error handling")
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test implementation."""
        return []


@pytest.mark.unit
class TestBaseAnalyzer:
    """Test cases for BaseAnalyzer class."""
    
    def test_analyzer_initialization(self, mock_s3_service, mock_pricing_service, 
                                   mock_storage_lens_service, mock_performance_monitor, 
                                   mock_memory_manager):
        """Test analyzer initialization with all services."""
        analyzer = MockAnalyzer(
            s3_service=mock_s3_service,
            pricing_service=mock_pricing_service,
            storage_lens_service=mock_storage_lens_service,
            performance_monitor=mock_performance_monitor,
            memory_manager=mock_memory_manager
        )
        
        assert analyzer.s3_service == mock_s3_service
        assert analyzer.pricing_service == mock_pricing_service
        assert analyzer.storage_lens_service == mock_storage_lens_service
        assert analyzer.performance_monitor == mock_performance_monitor
        assert analyzer.memory_manager == mock_memory_manager
        assert analyzer.analysis_type == "test_analyzer"
        assert analyzer.version == "1.0.0"
        assert analyzer.execution_count == 0
        assert analyzer.last_execution is None
    
    def test_analyzer_initialization_minimal(self):
        """Test analyzer initialization with minimal parameters."""
        analyzer = MockAnalyzer()
        
        assert analyzer.s3_service is None
        assert analyzer.pricing_service is None
        assert analyzer.storage_lens_service is None
        assert analyzer.analysis_type == "test_analyzer"
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(
            region="us-east-1",
            lookback_days=30,
            bucket_names=["bucket1", "bucket2"],
            timeout_seconds=60
        )
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert len(validation["warnings"]) == 0
    
    def test_validate_parameters_invalid_region(self):
        """Test parameter validation with invalid region."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(region=123)
        
        assert validation["valid"] is False
        assert "Region must be a string" in validation["errors"]
    
    def test_validate_parameters_invalid_lookback_days(self):
        """Test parameter validation with invalid lookback_days."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(lookback_days=-5)
        
        assert validation["valid"] is False
        assert "lookback_days must be a positive integer" in validation["errors"]
    
    def test_validate_parameters_large_lookback_days(self):
        """Test parameter validation with large lookback_days."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(lookback_days=400)
        
        assert validation["valid"] is True
        assert "lookback_days > 365 may result in large datasets" in validation["warnings"]
    
    def test_validate_parameters_invalid_bucket_names(self):
        """Test parameter validation with invalid bucket_names."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(bucket_names="not_a_list")
        
        assert validation["valid"] is False
        assert "bucket_names must be a list" in validation["errors"]
    
    def test_validate_parameters_invalid_bucket_name_types(self):
        """Test parameter validation with invalid bucket name types."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(bucket_names=["bucket1", 123, "bucket2"])
        
        assert validation["valid"] is False
        assert "All bucket names must be strings" in validation["errors"]
    
    def test_validate_parameters_invalid_timeout(self):
        """Test parameter validation with invalid timeout."""
        analyzer = MockAnalyzer()
        
        validation = analyzer.validate_parameters(timeout_seconds=-10)
        
        assert validation["valid"] is False
        assert "timeout_seconds must be a positive number" in validation["errors"]
    
    def test_prepare_analysis_context(self):
        """Test analysis context preparation."""
        analyzer = MockAnalyzer()
        
        context = analyzer.prepare_analysis_context(
            region="us-west-2",
            session_id="test_session",
            lookback_days=14,
            bucket_names=["test-bucket"]
        )
        
        assert context["analysis_type"] == "test_analyzer"
        assert context["analyzer_version"] == "1.0.0"
        assert context["region"] == "us-west-2"
        assert context["session_id"] == "test_session"
        assert context["lookback_days"] == 14
        assert context["bucket_names"] == ["test-bucket"]
        assert context["timeout_seconds"] == 60  # default
        assert "started_at" in context
        assert "execution_id" in context
    
    def test_handle_analysis_error(self):
        """Test error handling functionality."""
        analyzer = MockAnalyzer()
        context = {"analysis_type": "test_analyzer", "session_id": "test"}
        error = ValueError("Test error message")
        
        error_result = analyzer.handle_analysis_error(error, context)
        
        assert error_result["status"] == "error"
        assert error_result["analysis_type"] == "test_analyzer"
        assert error_result["error_message"] == "Test error message"
        assert error_result["error_type"] == "ValueError"
        assert error_result["context"] == context
        assert "timestamp" in error_result
        assert len(error_result["recommendations"]) == 1
        assert error_result["recommendations"][0]["type"] == "error_resolution"
    
    def test_create_recommendation_minimal(self):
        """Test recommendation creation with minimal parameters."""
        analyzer = MockAnalyzer()
        
        recommendation = analyzer.create_recommendation(
            rec_type="cost_optimization",
            priority="high",
            title="Test Recommendation",
            description="Test description"
        )
        
        assert recommendation["type"] == "cost_optimization"
        assert recommendation["priority"] == "high"
        assert recommendation["title"] == "Test Recommendation"
        assert recommendation["description"] == "Test description"
        assert recommendation["implementation_effort"] == "medium"  # default
        assert recommendation["analyzer"] == "test_analyzer"
        assert "created_at" in recommendation
    
    def test_create_recommendation_full(self):
        """Test recommendation creation with all parameters."""
        analyzer = MockAnalyzer()
        
        recommendation = analyzer.create_recommendation(
            rec_type="governance",
            priority="medium",
            title="Full Recommendation",
            description="Full description",
            potential_savings=100.50,
            implementation_effort="low",
            affected_resources=["bucket1", "bucket2"],
            action_items=["Action 1", "Action 2"]
        )
        
        assert recommendation["type"] == "governance"
        assert recommendation["priority"] == "medium"
        assert recommendation["potential_savings"] == 100.50
        assert recommendation["potential_savings_formatted"] == "$100.50"
        assert recommendation["implementation_effort"] == "low"
        assert recommendation["affected_resources"] == ["bucket1", "bucket2"]
        assert recommendation["resource_count"] == 2
        assert recommendation["action_items"] == ["Action 1", "Action 2"]
    
    def test_log_analysis_start(self, mock_performance_monitor):
        """Test analysis start logging."""
        analyzer = MockAnalyzer(performance_monitor=mock_performance_monitor)
        context = {"analysis_type": "test_analyzer"}
        
        analyzer.log_analysis_start(context)
        
        assert analyzer.execution_count == 1
        assert analyzer.last_execution is not None
        mock_performance_monitor.record_metric.assert_called_once()
    
    def test_log_analysis_complete(self, mock_performance_monitor):
        """Test analysis completion logging."""
        analyzer = MockAnalyzer(performance_monitor=mock_performance_monitor)
        context = {"analysis_type": "test_analyzer"}
        result = {
            "status": "success",
            "execution_time": 5.0,
            "recommendations": [{"type": "test"}]
        }
        
        analyzer.log_analysis_complete(context, result)
        
        # Should record 3 metrics: completed, execution_time, recommendations
        assert mock_performance_monitor.record_metric.call_count == 3
    
    def test_get_analyzer_info(self, mock_s3_service, mock_pricing_service):
        """Test analyzer info retrieval."""
        analyzer = MockAnalyzer(
            s3_service=mock_s3_service,
            pricing_service=mock_pricing_service
        )
        analyzer.execution_count = 5
        analyzer.last_execution = datetime.now()
        
        info = analyzer.get_analyzer_info()
        
        assert info["analysis_type"] == "test_analyzer"
        assert info["class_name"] == "MockAnalyzer"
        assert info["version"] == "1.0.0"
        assert info["execution_count"] == 5
        assert info["last_execution"] is not None
        assert info["services"]["s3_service"] is True
        assert info["services"]["pricing_service"] is True
        assert info["services"]["storage_lens_service"] is False
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling_success(self, mock_memory_manager):
        """Test successful execution with error handling."""
        analyzer = MockAnalyzer(memory_manager=mock_memory_manager)
        
        result = await analyzer.execute_with_error_handling(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "test_analyzer"
        assert "timestamp" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) == 1
        
        # Verify memory tracking was called
        mock_memory_manager.start_memory_tracking.assert_called_once()
        mock_memory_manager.stop_memory_tracking.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling_validation_failure(self):
        """Test execution with parameter validation failure."""
        analyzer = MockAnalyzer()
        
        result = await analyzer.execute_with_error_handling(
            region=123,  # Invalid region
            lookback_days=-5  # Invalid lookback_days
        )
        
        assert result["status"] == "error"
        assert result["error_message"] == "Parameter validation failed"
        assert "validation_errors" in result
        assert len(result["validation_errors"]) == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling_analysis_failure(self):
        """Test execution with analysis failure."""
        analyzer = MockFailingAnalyzer()
        
        result = await analyzer.execute_with_error_handling(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result["status"] == "error"
        assert result["error_message"] == "Test error for error handling"
        assert result["error_type"] == "ValueError"
        assert "recommendations" in result
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["type"] == "error_resolution"


@pytest.mark.unit
class TestAnalyzerRegistry:
    """Test cases for AnalyzerRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AnalyzerRegistry()
        
        assert len(registry._analyzers) == 0
        assert registry.list_analyzers() == []
    
    def test_register_analyzer(self):
        """Test analyzer registration."""
        registry = AnalyzerRegistry()
        analyzer = MockAnalyzer()
        
        registry.register(analyzer)
        
        assert len(registry._analyzers) == 1
        assert "test_analyzer" in registry._analyzers
        assert registry.get("test_analyzer") == analyzer
    
    def test_register_multiple_analyzers(self):
        """Test registering multiple analyzers."""
        registry = AnalyzerRegistry()
        analyzer1 = MockAnalyzer()
        analyzer2 = MockFailingAnalyzer()
        
        registry.register(analyzer1)
        registry.register(analyzer2)
        
        assert len(registry._analyzers) == 2
        assert set(registry.list_analyzers()) == {"test_analyzer", "failing_analyzer"}
    
    def test_get_nonexistent_analyzer(self):
        """Test getting non-existent analyzer."""
        registry = AnalyzerRegistry()
        
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_get_analyzer_info(self):
        """Test getting analyzer info from registry."""
        registry = AnalyzerRegistry()
        analyzer = MockAnalyzer()
        registry.register(analyzer)
        
        info = registry.get_analyzer_info()
        
        assert "test_analyzer" in info
        assert info["test_analyzer"]["class_name"] == "MockAnalyzer"
        assert info["test_analyzer"]["analysis_type"] == "test_analyzer"
    
    def test_global_registry(self):
        """Test global registry access."""
        global_registry = get_analyzer_registry()
        
        assert isinstance(global_registry, AnalyzerRegistry)
        
        # Test that multiple calls return the same instance
        registry2 = get_analyzer_registry()
        assert global_registry is registry2


@pytest.mark.unit
class TestAnalyzerPerformanceIntegration:
    """Test performance monitoring integration in analyzers."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, mock_performance_monitor, 
                                                    mock_memory_manager):
        """Test that analyzers properly integrate with performance monitoring."""
        analyzer = MockAnalyzer(
            performance_monitor=mock_performance_monitor,
            memory_manager=mock_memory_manager
        )
        
        result = await analyzer.execute_with_error_handling(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result["status"] == "success"
        
        # Verify performance monitoring calls
        mock_performance_monitor.record_metric.assert_called()
        
        # Verify memory management calls
        mock_memory_manager.start_memory_tracking.assert_called_once()
        mock_memory_manager.stop_memory_tracking.assert_called_once()
        mock_memory_manager.register_large_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_stats_in_result(self, mock_memory_manager):
        """Test that memory statistics are included in results."""
        mock_memory_manager.stop_memory_tracking.return_value = {
            "peak_memory_mb": 75.5,
            "avg_memory_mb": 45.2
        }
        
        analyzer = MockAnalyzer(memory_manager=mock_memory_manager)
        
        result = await analyzer.execute_with_error_handling(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result["status"] == "success"
        assert "memory_usage" in result
        assert result["memory_usage"]["peak_memory_mb"] == 75.5
        assert result["memory_usage"]["avg_memory_mb"] == 45.2
    
    def test_analyzer_without_performance_components(self):
        """Test analyzer behavior without performance monitoring components."""
        analyzer = MockAnalyzer()
        
        # Should not raise errors
        context = {"analysis_type": "test"}
        analyzer.log_analysis_start(context)
        analyzer.log_analysis_complete(context, {"status": "success", "execution_time": 1.0, "recommendations": []})
        
        info = analyzer.get_analyzer_info()
        assert info["services"]["s3_service"] is False
        assert info["services"]["pricing_service"] is False
        assert info["services"]["storage_lens_service"] is False