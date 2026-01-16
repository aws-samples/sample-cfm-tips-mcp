#!/usr/bin/env python3
"""
Integration tests for CloudWatch comprehensive optimization tool.

Tests the unified comprehensive optimization tool with intelligent orchestration,
executive summary generation, and all 4 functionalities working together.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator


@pytest.mark.integration
class TestCloudWatchComprehensiveToolIntegration:
    """Test CloudWatch comprehensive optimization tool integration."""
    
    @pytest.fixture
    def comprehensive_orchestrator(self):
        """Create orchestrator for comprehensive tool testing."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so, \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_ae:
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
            orchestrator.service_orchestrator = mock_so.return_value
            orchestrator.analysis_engine = mock_ae.return_value
            
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_comprehensive_tool_basic_execution(self, comprehensive_orchestrator):
        """Test basic execution of comprehensive optimization tool."""
        # Mock the comprehensive analysis result
        expected_result = {
            "status": "success",
            "analysis_type": "comprehensive",
            "successful_analyses": 4,
            "total_analyses": 4,
            "results": {
                "general_spend": {"status": "success", "data": {"savings": 50.0}},
                "logs_optimization": {"status": "success", "data": {"savings": 30.0}},
                "metrics_optimization": {"status": "success", "data": {"savings": 25.0}},
                "alarms_and_dashboards": {"status": "success", "data": {"savings": 20.0}}
            },
            "orchestrator_metadata": {
                "session_id": "test_session",
                "region": "us-east-1",
                "total_orchestration_time": 2.5,
                "performance_optimizations": {
                    "intelligent_timeout": 120.0,
                    "cache_enabled": True,
                    "memory_management": True,
                    "performance_monitoring": True
                }
            }
        }
        
        # Mock the analysis engine as an async function
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis = AsyncMock(return_value=expected_result)
        
        # Execute comprehensive analysis
        result = await comprehensive_orchestrator.execute_comprehensive_analysis(
            region="us-east-1",
            lookback_days=30,
            allow_cost_explorer=False,
            allow_aws_config=False,
            allow_cloudtrail=False,
            allow_minimal_cost_metrics=False
        )
        
        # Verify results
        assert result["status"] == "success"
        assert "orchestrator_metadata" in result
        assert result["orchestrator_metadata"]["region"] == "us-east-1"
        assert result["orchestrator_metadata"]["orchestrator_version"] == "1.0.0"
        
        # Verify the analysis engine was called
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_tool_with_cost_constraints(self, comprehensive_orchestrator):
        """Test comprehensive tool respects cost constraints."""
        # Mock result with cost constraints applied
        expected_result = {
            "status": "success",
            "analysis_type": "comprehensive",
            "successful_analyses": 4,
            "total_analyses": 4,
            "results": {
                "general_spend": {"status": "success", "cost_incurred": False},
                "logs_optimization": {"status": "success", "cost_incurred": False},
                "metrics_optimization": {"status": "success", "cost_incurred": False},
                "alarms_and_dashboards": {"status": "success", "cost_incurred": False}
            },
            "orchestrator_metadata": {
                "cost_preferences": {
                    "allow_cost_explorer": False,
                    "allow_aws_config": False,
                    "allow_cloudtrail": False,
                    "allow_minimal_cost_metrics": False
                }
            }
        }
        
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis = AsyncMock(return_value=expected_result)
        
        # Execute with strict cost constraints
        result = await comprehensive_orchestrator.execute_comprehensive_analysis(
            allow_cost_explorer=False,
            allow_aws_config=False,
            allow_cloudtrail=False,
            allow_minimal_cost_metrics=False
        )
        
        # Verify cost constraints were respected
        assert result["status"] == "success"
        assert "cost_preferences" in result["orchestrator_metadata"]
        cost_prefs = result["orchestrator_metadata"]["cost_preferences"]
        assert cost_prefs["allow_cost_explorer"] is False
        assert cost_prefs["allow_aws_config"] is False
        assert cost_prefs["allow_cloudtrail"] is False
        assert cost_prefs["allow_minimal_cost_metrics"] is False
    
    @pytest.mark.asyncio
    async def test_comprehensive_tool_error_handling(self, comprehensive_orchestrator):
        """Test comprehensive tool handles errors gracefully."""
        # Mock a partial failure scenario
        expected_result = {
            "status": "partial",
            "analysis_type": "comprehensive",
            "successful_analyses": 3,
            "total_analyses": 4,
            "results": {
                "general_spend": {"status": "success", "data": {"savings": 50.0}},
                "logs_optimization": {"status": "error", "error_message": "Simulated failure"},
                "metrics_optimization": {"status": "success", "data": {"savings": 25.0}},
                "alarms_and_dashboards": {"status": "success", "data": {"savings": 20.0}}
            }
        }
        
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis = AsyncMock(return_value=expected_result)
        
        # Execute comprehensive analysis
        result = await comprehensive_orchestrator.execute_comprehensive_analysis()
        
        # Verify partial success is handled correctly
        assert result["status"] == "partial"
        # The orchestrator adds its own metadata, so we check the engine result was used
        assert "orchestrator_metadata" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_tool_timeout_handling(self, comprehensive_orchestrator):
        """Test comprehensive tool handles timeouts properly."""
        # Mock a timeout scenario
        async def mock_timeout_analysis(**kwargs):
            # Simulate a timeout by raising an exception
            raise Exception("Analysis comprehensive timed out after 60.0 seconds")
        
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_timeout_analysis
        
        # Execute with a short timeout
        result = await comprehensive_orchestrator.execute_comprehensive_analysis(
            timeout_seconds=1.0
        )
        
        # Verify timeout is handled gracefully
        assert result["status"] == "error"
        assert "timed out" in result["error_message"].lower()
    
    @pytest.mark.asyncio
    async def test_comprehensive_tool_caching_behavior(self, comprehensive_orchestrator):
        """Test comprehensive tool caching behavior."""
        # Mock successful result
        expected_result = {
            "status": "success",
            "analysis_type": "comprehensive",
            "from_cache": False,
            "successful_analyses": 4,
            "total_analyses": 4
        }
        
        comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis = AsyncMock(return_value=expected_result)
        
        # First execution - should not be from cache
        result1 = await comprehensive_orchestrator.execute_comprehensive_analysis(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result1["status"] == "success"
        # Note: The orchestrator adds its own metadata, so we check the engine result
        
        # Second execution - should call the same mock again
        result2 = await comprehensive_orchestrator.execute_comprehensive_analysis(
            region="us-east-1",
            lookback_days=30
        )
        
        assert result2["status"] == "success"
        # Verify both calls were made (caching is handled at lower levels)
        assert comprehensive_orchestrator.analysis_engine.run_comprehensive_analysis.call_count == 2


@pytest.mark.integration
class TestCloudWatchOrchestratorRealIntegration:
    """Test CloudWatch orchestrator with more realistic scenarios."""
    
    @pytest.fixture
    def real_orchestrator(self):
        """Create orchestrator with minimal mocking for realistic testing."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so:
            # Mock the service orchestrator but let other components work
            mock_so.return_value.session_id = "test_session_123"
            mock_so.return_value.get_stored_tables.return_value = ["test_table"]
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-west-2')
            return orchestrator
    
    def test_orchestrator_initialization(self, real_orchestrator):
        """Test orchestrator initializes correctly with all components."""
        assert real_orchestrator.region == 'us-west-2'
        assert real_orchestrator.session_id == "test_session_123"
        assert real_orchestrator.analysis_engine is not None
        assert real_orchestrator.cost_controller is not None
        assert real_orchestrator.aggregation_queries is not None
    
    def test_cost_preferences_validation(self, real_orchestrator):
        """Test cost preferences validation works correctly."""
        # Test valid preferences
        result = real_orchestrator.validate_cost_preferences(
            allow_cost_explorer=True,
            allow_aws_config=False,
            allow_cloudtrail=True,
            allow_minimal_cost_metrics=False
        )
        
        assert result["valid"] is True
        assert "validated_preferences" in result
        assert result["validated_preferences"]["allow_cost_explorer"] is True
        assert result["validated_preferences"]["allow_cloudtrail"] is True
        
        # Test invalid preferences (non-boolean values)
        result = real_orchestrator.validate_cost_preferences(
            allow_cost_explorer="invalid",
            allow_aws_config=123
        )
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_cost_estimation(self, real_orchestrator):
        """Test cost estimation functionality."""
        # Test with no paid features
        result = real_orchestrator.get_cost_estimate(
            allow_cost_explorer=False,
            allow_aws_config=False,
            allow_cloudtrail=False,
            allow_minimal_cost_metrics=False
        )
        
        assert "cost_estimate" in result
        assert result["cost_estimate"]["total_estimated_cost"] == 0.0
        
        # Test with paid features
        result = real_orchestrator.get_cost_estimate(
            allow_cost_explorer=True,
            allow_aws_config=True,
            lookback_days=30
        )
        
        assert "cost_estimate" in result
        # Should have some estimated cost for paid features
        assert result["cost_estimate"]["total_estimated_cost"] >= 0.0
    
    def test_stored_tables_access(self, real_orchestrator):
        """Test access to stored tables."""
        tables = real_orchestrator.get_stored_tables()
        assert isinstance(tables, list)
        assert "test_table" in tables
    
    def test_analysis_results_query(self, real_orchestrator):
        """Test querying analysis results."""
        # Mock the service orchestrator query execution
        with patch.object(real_orchestrator.service_orchestrator, 'query_session_data') as mock_query:
            mock_query.return_value = [
                {"analysis_type": "general_spend", "status": "success"},
                {"analysis_type": "logs_optimization", "status": "success"}
            ]
            
            results = real_orchestrator.get_analysis_results("SELECT * FROM analysis_results")
            
            assert len(results) == 2
            assert results[0]["analysis_type"] == "general_spend"
            assert results[1]["analysis_type"] == "logs_optimization"
            
            # Verify cost control info was added
            for result in results:
                assert "cost_control_info" in result
                assert "current_preferences" in result["cost_control_info"]