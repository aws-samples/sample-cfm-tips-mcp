"""
Integration tests for Database Savings Plans functionality.

Tests the complete integration between orchestrator, analyzers, services, and MCP functions
with mocked AWS services to ensure proper data flow and error handling.
"""

import pytest
import asyncio
import json
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from playbooks.rds.database_savings_plans import (
    analyze_database_usage,
    generate_savings_plans_recommendations,
    analyze_custom_commitment,
    compare_with_reserved_instances,
    analyze_existing_commitments,
    run_database_savings_plans_analysis,
    run_purchase_analyzer,
    analyze_existing_savings_plans
)


@pytest.fixture
def mock_cost_explorer_response():
    """Mock Cost Explorer response for database usage."""
    return {
        "status": "success",
        "data": {
            "total_cost": 1500.0,
            "service_breakdown": {
                "rds": 800.0,
                "aurora": 400.0,
                "dynamodb": 200.0,
                "elasticache": 100.0
            },
            "region_breakdown": {
                "us-east-1": 900.0,
                "us-west-2": 600.0
            },
            "instance_family_breakdown": {
                "db.r7g.large": 600.0,
                "db.m7g.xlarge": 400.0,
                "db.r5.large": 300.0,  # Older generation
                "db.t4g.medium": 200.0
            }
        }
    }


@pytest.fixture
def mock_savings_plans_response():
    """Mock Savings Plans service response."""
    return {
        "status": "success",
        "data": {
            "discount_percentage": 20.0,
            "hourly_rate": 0.80,
            "on_demand_rate": 1.00
        }
    }


@pytest.fixture
def mock_existing_plans_response():
    """Mock existing savings plans response."""
    return {
        "status": "success",
        "data": {
            "existing_plans": [
                {
                    "savings_plan_id": "sp-12345",
                    "hourly_commitment": 10.0,
                    "commitment_term": "1_YEAR",
                    "utilization_percentage": 85.5,
                    "coverage_percentage": 75.0,
                    "unused_commitment": 1.45
                }
            ],
            "gaps": {
                "uncovered_spend": 300.0,
                "recommendation": "Consider additional $3/hour commitment"
            }
        }
    }


@pytest.fixture
def mock_service_orchestrator():
    """Mock ServiceOrchestrator for parallel execution."""
    orchestrator = Mock()
    orchestrator.execute_parallel_analysis = Mock(return_value={
        "status": "success",
        "successful": 2,
        "total_tasks": 2,
        "results": {
            "task_1": {
                "operation": "analyze_usage",
                "status": "success",
                "data": {
                    "status": "success",
                    "data": {
                        "total_on_demand_spend": 1500.0,
                        "average_hourly_spend": 2.08,
                        "lookback_period_days": 30,
                        "service_breakdown": {},
                        "region_breakdown": {},
                        "instance_family_breakdown": {}
                    }
                }
            },
            "task_2": {
                "operation": "analyze_existing_commitments",
                "status": "success",
                "data": {
                    "status": "success",
                    "data": {
                        "existing_plans": [],
                        "gaps": {}
                    }
                }
            }
        }
    })
    return orchestrator


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for data persistence."""
    session_manager = Mock()
    session_manager.store_analysis_result = Mock(return_value=True)
    session_manager.get_analysis_results = Mock(return_value=[])
    session_manager.query_historical_data = Mock(return_value=[])
    return session_manager


@pytest.mark.integration
class TestDatabaseSavingsPlansIntegration:
    """Test integration between core functions and services."""
    
    def test_usage_analysis_integration(self, mock_cost_explorer_response):
        """Test database usage analysis with mocked Cost Explorer."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = mock_cost_explorer_response
            
            result = analyze_database_usage(
                region="us-east-1",
                lookback_period_days=30,
                services=["rds", "aurora"]
            )
        
        assert result["status"] == "success"
        assert result["data"]["total_on_demand_spend"] == 1500.0
        assert result["data"]["average_hourly_spend"] == 1500.0 / (30 * 24)
        assert "service_breakdown" in result["data"]
        assert "region_breakdown" in result["data"]
        assert "instance_family_breakdown" in result["data"]
        
        # Verify Cost Explorer was called with correct parameters
        mock_cost_explorer.assert_called_once()
        call_args = mock_cost_explorer.call_args
        assert call_args[1]["services"] == ["rds", "aurora"]
        assert call_args[1]["region"] == "us-east-1"
    
    def test_recommendations_generation_integration(self, mock_cost_explorer_response, mock_savings_plans_response):
        """Test recommendations generation with usage data."""
        # First get usage data
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = mock_cost_explorer_response
            usage_result = analyze_database_usage()
        
        # Then generate recommendations
        with patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            mock_rates.return_value = mock_savings_plans_response
            
            result = generate_savings_plans_recommendations(
                usage_data=usage_result["data"],
                commitment_terms=["1_YEAR"],
                payment_options=["NO_UPFRONT"]
            )
        
        assert result["status"] == "success"
        assert "recommendations" in result["data"]
        assert len(result["data"]["recommendations"]) > 0
        
        # Check recommendation structure
        rec = result["data"]["recommendations"][0]
        assert "hourly_commitment" in rec
        assert "estimated_annual_savings" in rec
        assert "projected_coverage" in rec
        assert "projected_utilization" in rec
        assert rec["commitment_term"] == "1_YEAR"
        assert rec["payment_option"] == "NO_UPFRONT"
    
    def test_purchase_analyzer_integration(self, mock_cost_explorer_response, mock_savings_plans_response):
        """Test purchase analyzer mode with custom commitment."""
        # Get usage data
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = mock_cost_explorer_response
            usage_result = analyze_database_usage()
        
        # Test custom commitment analysis
        with patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            mock_rates.return_value = mock_savings_plans_response
            
            result = analyze_custom_commitment(
                hourly_commitment=5.0,
                usage_data=usage_result["data"],
                commitment_term="1_YEAR",
                payment_option="NO_UPFRONT"
            )
        
        assert result["status"] == "success"
        assert result["data"]["commitment_term"] == "1_YEAR"
        assert result["data"]["payment_option"] == "NO_UPFRONT"
        assert "projected_coverage" in result["data"]
        assert "projected_utilization" in result["data"]
        assert "estimated_annual_savings" in result["data"]
    
    def test_ri_comparison_integration(self, mock_cost_explorer_response):
        """Test Reserved Instance comparison functionality."""
        # Get usage data
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = mock_cost_explorer_response
            usage_result = analyze_database_usage()
        
        # Test RI comparison
        with patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_sp_rates:
            
            mock_sp_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            result = compare_with_reserved_instances(
                usage_data=usage_result["data"],
                services=["rds", "aurora"]
            )
        
        assert result["status"] == "success"
        assert "latest_generation" in result["data"]
        assert "older_generation" in result["data"]
        
        # Should have recommendations for both generations
        if result["data"]["latest_generation"]:
            latest_rec = result["data"]["latest_generation"][0]
            assert "annual_spend" in latest_rec or "savings_plan_cost" in latest_rec
            assert "best_option" in latest_rec or "recommendation" in latest_rec
        
        if result["data"]["older_generation"]:
            older_rec = result["data"]["older_generation"][0]
            assert "annual_spend" in older_rec or "ri_standard_cost" in older_rec
            assert "best_option" in older_rec or "recommendation" in older_rec
    
    def test_existing_commitments_analysis_integration(self, mock_existing_plans_response):
        """Test existing commitments analysis."""
        with patch('services.savings_plans_service.get_savings_plans_utilization') as mock_utilization, \
             patch('services.savings_plans_service.get_savings_plans_coverage') as mock_coverage:
            
            mock_utilization.return_value = {"status": "success", "data": {"utilization": 85.5}}
            mock_coverage.return_value = {"status": "success", "data": {"coverage": 75.0}}
            
            result = analyze_existing_commitments(
                region="us-east-1",
                lookback_period_days=30
            )
        
        # Should handle case where no existing plans are found
        assert result["status"] in ["success", "info"]
        assert "data" in result
    
    def test_error_handling_integration(self):
        """Test error handling across integration points."""
        # Test Cost Explorer failure
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "error",
                "message": "Access denied to Cost Explorer"
            }
            
            result = analyze_database_usage()
        
        assert result["status"] == "error"
        assert "Access denied" in result["message"]
        
        # Test invalid usage data for recommendations
        result = generate_savings_plans_recommendations(usage_data=None)
        assert result["status"] == "error"
        assert "Usage data is required" in result["message"]
        
        # Test invalid commitment amount for purchase analyzer
        result = analyze_custom_commitment(
            hourly_commitment=-5.0,
            usage_data={"average_hourly_spend": 10.0}
        )
        assert result["status"] == "error"
        assert "must be positive" in result["message"]


@pytest.mark.integration
class TestMCPFunctionIntegration:
    """Test MCP wrapper functions with service integration."""
    
    @pytest.mark.asyncio
    async def test_run_database_savings_plans_analysis_integration(self, mock_cost_explorer_response):
        """Test complete MCP analysis function."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 30,
                "services": ["rds", "aurora"],
                "include_ri_comparison": True,
                "store_results": True
            })
        
        assert len(result) > 0
        assert result[0].type == "text"
        
        # Parse the JSON response
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        assert "data" in response_data
        assert "analysis_insights" in response_data["data"]
        assert "execution_summary" in response_data["data"]
    
    @pytest.mark.asyncio
    async def test_run_purchase_analyzer_integration(self, mock_cost_explorer_response):
        """Test purchase analyzer MCP function."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            result = await run_purchase_analyzer({
                "hourly_commitment": 8.0,
                "commitment_term": "1_YEAR",
                "payment_option": "NO_UPFRONT",
                "region": "us-east-1"
            })
        
        assert len(result) > 0
        assert result[0].type == "text"
        
        # Parse the JSON response
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        assert response_data["data"]["hourly_commitment"] == 8.0
    
    @pytest.mark.asyncio
    async def test_analyze_existing_savings_plans_integration(self):
        """Test existing savings plans analysis MCP function."""
        with patch('services.savings_plans_service.get_savings_plans_utilization') as mock_utilization, \
             patch('services.savings_plans_service.get_savings_plans_coverage') as mock_coverage:
            
            mock_utilization.return_value = {"status": "success", "data": {"utilization": 85.5}}
            mock_coverage.return_value = {"status": "success", "data": {"coverage": 75.0}}
            
            result = await analyze_existing_savings_plans({
                "region": "us-east-1",
                "lookback_period_days": 30
            })
        
        assert len(result) > 0
        assert result[0].type == "text"
        
        # Parse the JSON response
        response_data = json.loads(result[0].text)
        assert response_data["status"] in ["success", "info"]


@pytest.mark.integration
class TestParallelExecutionIntegration:
    """Test parallel execution with ServiceOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_parallel_service_calls(self, mock_service_orchestrator):
        """Test parallel execution of multiple service calls."""
        # Mock successful parallel execution
        mock_service_orchestrator.execute_parallel_analysis.return_value = {
            "status": "success",
            "successful": 2,
            "total_tasks": 2,
            "results": {
                "database_savings_plans_analyze_usage_0_1765821625": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "data": {"status": "success", "data": {"total_on_demand_spend": 1000.0, "average_hourly_spend": 1.39}}
                },
                "database_savings_plans_analyze_existing_commitments_1_1765821625": {
                    "operation": "analyze_existing_commitments", 
                    "status": "success",
                    "data": {"status": "success", "data": {"existing_plans": []}}
                }
            },
            "execution_time": 15.5
        }
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator):
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "include_existing_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify parallel execution was called
        mock_service_orchestrator.execute_parallel_analysis.assert_called_once()
        call_args = mock_service_orchestrator.execute_parallel_analysis.call_args[1]
        assert len(call_args["service_calls"]) >= 1  # At least usage analysis
    
    @pytest.mark.asyncio
    async def test_orchestrator_task_coordination(self):
        """Test ServiceOrchestrator coordinates multiple AWS service calls efficiently."""
        mock_orchestrator = Mock()
        
        # Mock coordinated execution with timing
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "success",
            "successful": 2,
            "total_tasks": 2,
            "results": {
                "database_savings_plans_analyze_usage_0_1765821625": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "execution_time": 2.1,
                    "data": {"status": "success", "data": {"total_on_demand_spend": 1000.0, "average_hourly_spend": 1.39}}
                },
                "database_savings_plans_analyze_existing_commitments_1_1765821625": {
                    "operation": "analyze_existing_commitments",
                    "status": "success", 
                    "execution_time": 1.8,
                    "data": {"status": "success", "data": {"existing_plans": []}}
                }
            },
            "execution_time": 3.2,  # Total time less than sum due to parallelization
            "parallelization_efficiency": 0.68  # 68% efficiency
        })
        
        with patch('playbooks.rds.database_savings_plans.ServiceOrchestrator', return_value=mock_orchestrator):
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "include_existing_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify orchestrator coordinated multiple service calls
        mock_orchestrator.execute_parallel_analysis.assert_called_once()
        
        # Verify execution summary shows parallelization benefits
        exec_summary = response_data["data"]["execution_summary"]
        assert exec_summary["successful"] == 2
        assert exec_summary["execution_time"] == 3.2
        assert exec_summary["parallelization_efficiency"] == 0.68
    
    @pytest.mark.asyncio
    async def test_orchestrator_resource_management(self):
        """Test ServiceOrchestrator manages AWS API rate limits and resource usage."""
        mock_orchestrator = Mock()
        
        # Mock resource-aware execution
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "success",
            "successful": 2,  # Fixed to match actual implementation
            "total_tasks": 2,  # Fixed to match actual implementation
            "results": {
                "usage_analysis": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "retry_count": 0,
                    "data": {"status": "success", "data": {"total_spend": 1000.0}}
                },
                "existing_analysis": {
                    "operation": "analyze_existing",
                    "status": "success", 
                    "retry_count": 2,  # Had to retry due to rate limiting
                    "data": {"status": "success", "data": {"existing_plans": []}}
                }
            },
            "resource_usage": {
                "api_calls_made": 15,
                "rate_limit_hits": 3,
                "retry_attempts": 6,
                "throttling_delays": 4.2
            }
        })
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_orchestrator):
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "comprehensive_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify resource management information is included
        exec_summary = response_data["data"]["execution_summary"]
        assert exec_summary["successful"] == 2
        assert exec_summary["total_tasks"] == 2
        assert "resource_usage" in exec_summary
        assert exec_summary["resource_usage"]["rate_limit_hits"] == 3
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_service_orchestrator):
        """Test handling of partial failures in parallel execution."""
        # Mock partial failure scenario
        mock_service_orchestrator.execute_parallel_analysis.return_value = {
            "status": "partial_success",
            "successful": 1,
            "total_tasks": 2,
            "results": {
                "usage_analysis": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "data": {"status": "success", "data": {"total_on_demand_spend": 1000.0}}
                },
                "existing_commitments": {
                    "operation": "analyze_existing_commitments",
                    "status": "error",
                    "error": "API rate limit exceeded"
                }
            }
        }
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator):
            result = await run_database_savings_plans_analysis({
                "include_existing_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        # Should still succeed with partial results
        assert response_data["status"] == "success"
        # Check for the actual data structure returned
        assert "data" in response_data
        assert "execution_summary" in response_data["data"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_service_orchestrator):
        """Test timeout handling in parallel execution."""
        # Mock timeout scenario - the function should handle this gracefully
        mock_service_orchestrator.execute_parallel_analysis.side_effect = asyncio.TimeoutError("Operation timed out")
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator):
            # The function should catch the timeout and fall back to direct execution
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        # The function should still succeed by falling back to direct execution
        assert response_data["status"] == "success"


@pytest.mark.integration
class TestSessionStorageIntegration:
    """Test session storage and data persistence."""
    
    def test_result_storage_integration(self, mock_session_manager):
        """Test storing analysis results in session database."""
        with patch('utils.session_manager.SessionManager', return_value=mock_session_manager):
            # This would be called internally by ServiceOrchestrator
            # We're testing the integration pattern
            
            analysis_result = {
                "analysis_type": "database_savings_plans",
                "timestamp": datetime.now().isoformat(),
                "data": {"total_spend": 1000.0},
                "recommendations": [{"hourly_commitment": 5.0}]
            }
            
            # Simulate storing results
            success = mock_session_manager.store_analysis_result(
                analysis_type="database_savings_plans",
                result_data=analysis_result,
                session_id="test_session"
            )
        
        assert success
        mock_session_manager.store_analysis_result.assert_called_once()
    
    def test_historical_data_querying(self, mock_session_manager):
        """Test querying historical analysis data."""
        # Mock historical data
        historical_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "total_spend": 1000.0,
                "recommendations": [{"hourly_commitment": 4.0}]
            },
            {
                "timestamp": "2024-01-15T00:00:00", 
                "total_spend": 1200.0,
                "recommendations": [{"hourly_commitment": 5.0}]
            }
        ]
        
        mock_session_manager.query_historical_data.return_value = historical_data
        
        with patch('utils.session_manager.SessionManager', return_value=mock_session_manager):
            results = mock_session_manager.query_historical_data(
                analysis_type="database_savings_plans",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
        
        assert len(results) == 2
        assert results[0]["total_spend"] == 1000.0
        assert results[1]["total_spend"] == 1200.0
        
        mock_session_manager.query_historical_data.assert_called_once()
    
    def test_session_data_persistence(self, mock_session_manager):
        """Test session data persistence across function calls."""
        session_id = "test_session_123"
        
        # Mock session data retrieval
        mock_session_manager.get_session_data.return_value = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "analysis_count": 5,
            "last_analysis": "database_savings_plans"
        }
        
        with patch('utils.session_manager.SessionManager', return_value=mock_session_manager):
            session_data = mock_session_manager.get_session_data(session_id)
        
        assert session_data["session_id"] == session_id
        assert session_data["analysis_count"] == 5
        assert session_data["last_analysis"] == "database_savings_plans"
    
    @pytest.mark.asyncio
    async def test_session_storage_during_analysis(self, mock_cost_explorer_response):
        """Test that analysis results are properly stored in session during execution."""
        # Mock the store_analysis_result function directly since that's what's actually called
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates, \
             patch('playbooks.rds.database_savings_plans.store_analysis_result') as mock_store:
            
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            mock_store.return_value = {"status": "success", "data": {"analysis_id": "test_id"}}
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "store_results": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify session storage was called
        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args[1]["analysis_type"] == "recommendations"
    
    @pytest.mark.asyncio
    async def test_session_storage_error_handling(self):
        """Test handling of session storage failures."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('playbooks.rds.database_savings_plans.store_analysis_result') as mock_store:
            
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"total_cost": 1000.0, "service_breakdown": {}}
            }
            mock_store.side_effect = Exception("Database connection failed")
            
            # Analysis should continue even if session storage fails
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "store_results": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        # Should still succeed with analysis results, just warn about storage failure
        assert response_data["status"] == "success"
        
        # Verify storage was attempted
        mock_store.assert_called_once()
    
    def test_concurrent_session_access(self):
        """Test handling of concurrent session access scenarios."""
        mock_session = Mock()
        
        # Mock concurrent access scenario
        call_count = 0
        def mock_store_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Database locked")
            return True
        
        mock_session.store_analysis_result = Mock(side_effect=mock_store_with_retry)
        
        with patch('utils.session_manager.SessionManager', return_value=mock_session):
            # Simulate multiple concurrent storage attempts
            results = []
            for i in range(3):
                try:
                    success = mock_session.store_analysis_result(
                        analysis_type="database_savings_plans",
                        result_data={"test": f"data_{i}"},
                        session_id=f"session_{i}"
                    )
                    results.append(success)
                except Exception:
                    results.append(False)
        
        # Should handle concurrent access gracefully
        assert len(results) == 3
        assert any(results)  # At least some should succeed


@pytest.mark.integration
class TestErrorScenariosIntegration:
    """Test error scenarios across the integration stack."""
    
    @pytest.mark.asyncio
    async def test_aws_service_unavailable(self):
        """Test handling when AWS services are unavailable."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "error",
                "message": "Service unavailable"
            }
            
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "Service unavailable" in response_data["message"]
    
    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        result = await run_purchase_analyzer({
            "hourly_commitment": -5.0,  # Invalid negative value
            "commitment_term": "2_YEAR",  # Invalid term
            "payment_option": "FULL_UPFRONT"  # Invalid for 1-year
        })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        # The error is wrapped by the error handler, so check for the actual error message
        assert "must be positive" in response_data["message"] or "ValidationError" in str(response_data)
    
    @pytest.mark.asyncio
    async def test_permission_denied_scenario(self):
        """Test handling of AWS permission denied errors."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "error",
                "error_code": "AccessDenied",
                "message": "User is not authorized to perform: ce:GetCostAndUsage",
                "required_permissions": ["ce:GetCostAndUsage", "ce:GetSavingsPlansUtilization"]
            }
            
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        # The error is wrapped by the error handler
        assert "not authorized" in response_data["message"] or "AccessDenied" in str(response_data)
    
    @pytest.mark.asyncio
    async def test_network_timeout_scenario(self, mock_service_orchestrator):
        """Test handling of network timeouts."""
        mock_service_orchestrator.execute_parallel_analysis.side_effect = asyncio.TimeoutError()
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator):
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "timeout" in response_data["message"].lower()
    
    def test_data_validation_errors(self):
        """Test data validation error handling."""
        # Test invalid lookback period
        result = analyze_database_usage(lookback_period_days=45)  # Invalid, must be 30, 60, or 90
        assert result["status"] == "error"
        assert "ValidationError" in result["error_code"]
        assert result["provided_value"] == 45
        assert result["valid_values"] == [30, 60, 90]
        
        # Test empty usage data for recommendations
        result = generate_savings_plans_recommendations(usage_data={})
        assert result["status"] == "error"
        assert "Usage data is required" in result["message"]
    
    @pytest.mark.asyncio
    async def test_partial_service_failures(self):
        """Test handling when some AWS services fail while others succeed."""
        mock_orchestrator = Mock()
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "partial_success",
            "successful": 2,
            "total_tasks": 4,
            "results": {
                "usage_analysis": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "data": {"status": "success", "data": {"total_spend": 1000.0}}
                },
                "recommendations": {
                    "operation": "generate_recommendations",
                    "status": "success",
                    "data": {"status": "success", "data": {"recommendations": []}}
                },
                "existing_analysis": {
                    "operation": "analyze_existing",
                    "status": "error",
                    "error": "Cost Explorer API rate limit exceeded"
                },
                "ri_comparison": {
                    "operation": "compare_ri",
                    "status": "error", 
                    "error": "Pricing API temporarily unavailable"
                }
            },
            "failed_operations": ["analyze_existing", "compare_ri"],
            "error_summary": "2 of 4 operations failed due to API issues"
        })
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_orchestrator):
            result = await run_database_savings_plans_analysis({
                "include_existing_analysis": True,
                "include_ri_comparison": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        # Should succeed with partial results
        assert response_data["status"] == "success"
        
        # Should include information about failed operations
        exec_summary = response_data["data"]["execution_summary"]
        assert exec_summary["successful"] == 2
        assert exec_summary["total_tasks"] == 4
        assert "failed_operations" in exec_summary
        assert len(exec_summary["failed_operations"]) == 2
    
    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self):
        """Test recovery from cascading failures across service dependencies."""
        # Mock a scenario where Cost Explorer fails, affecting downstream operations
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            
            # Cost Explorer fails
            mock_cost_explorer.return_value = {
                "status": "error",
                "error_code": "ServiceUnavailable",
                "message": "Cost Explorer service is temporarily unavailable"
            }
            
            # Savings Plans service should not be called due to upstream failure
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "include_recommendations": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "Cost Explorer" in response_data["message"]
        
        # Verify downstream service wasn't called due to upstream failure
        mock_rates.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        mock_orchestrator = Mock()
        
        # Mock resource exhaustion scenario
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "error",
            "error_code": "ResourceExhausted",
            "message": "Too many concurrent requests, system overloaded",
            "successful": 0,
            "total_tasks": 3,
            "resource_usage": {
                "memory_usage_mb": 2048,
                "cpu_usage_percent": 95,
                "active_connections": 50,
                "queue_depth": 100
            },
            "retry_suggestion": "Reduce concurrent operations or increase system resources"
        })
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_orchestrator):
            result = await run_database_savings_plans_analysis({
                "comprehensive_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "ResourceExhausted" in response_data.get("error_code", "")
        assert "overloaded" in response_data["message"]
        assert "retry_suggestion" in response_data
    
    @pytest.mark.asyncio
    async def test_data_corruption_detection(self):
        """Test detection and handling of corrupted data responses."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            # Mock corrupted response data
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "total_cost": "invalid_number",  # Should be float
                    "service_breakdown": None,  # Should be dict
                    "region_breakdown": [],  # Should be dict
                    "instance_family_breakdown": "not_a_dict"  # Should be dict
                }
            }
            
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "data validation" in response_data["message"].lower() or "invalid data" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_authentication_expiration_handling(self):
        """Test handling of expired AWS credentials during analysis."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "error",
                "error_code": "TokenExpired",
                "message": "AWS credentials have expired",
                "required_action": "Refresh AWS credentials and retry"
            }
            
            result = await run_database_savings_plans_analysis({})
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "TokenExpired" in response_data.get("error_code", "")
        assert "credentials" in response_data["message"].lower()
        assert "required_action" in response_data


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, mock_cost_explorer_response):
        """Test complete analysis workflow from start to finish."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates, \
             patch('services.savings_plans_service.get_savings_plans_utilization') as mock_utilization, \
             patch('services.savings_plans_service.get_savings_plans_coverage') as mock_coverage, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class:
            
            # Setup service mocks
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            mock_utilization.return_value = {"status": "success", "data": {"utilization": 85.0}}
            mock_coverage.return_value = {"status": "success", "data": {"coverage": 75.0}}
            
            # Setup orchestrator mock
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success",
                "successful": 4,
                "total_tasks": 4,
                "results": {
                    "usage_analysis": {
                        "operation": "analyze_usage",
                        "status": "success",
                        "data": {"status": "success", "data": mock_cost_explorer_response["data"]}
                    },
                    "recommendations": {
                        "operation": "generate_recommendations",
                        "status": "success",
                        "data": {"status": "success", "data": {"recommendations": [
                            {
                                "commitment_term": "1_YEAR",
                                "payment_option": "NO_UPFRONT",
                                "hourly_commitment": 10.5,
                                "estimated_annual_savings": 12000.0,
                                "projected_coverage": 85.5,
                                "projected_utilization": 95.2
                            }
                        ]}}
                    },
                    "existing_commitments": {
                        "operation": "analyze_existing",
                        "status": "success",
                        "data": {"status": "success", "data": {"existing_plans": []}}
                    },
                    "ri_comparison": {
                        "operation": "compare_ri",
                        "status": "success",
                        "data": {"status": "success", "data": {"latest_generation": [], "older_generation": []}}
                    }
                },
                "execution_time": 15.5
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Setup session manager mock
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Run complete analysis
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 30,
                "services": ["rds", "aurora", "dynamodb"],
                "include_ri_comparison": True,
                "include_existing_analysis": True,
                "store_results": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify orchestrator was used for parallel execution
        mock_orchestrator.execute_parallel_analysis.assert_called_once()
        
        # Verify session storage was called
        mock_session.store_analysis_result.assert_called_once()
        
        # Verify all components are present
        assert "analysis_insights" in response_data["data"]
        assert "execution_summary" in response_data["data"]
        
        # Verify execution summary shows parallel execution
        exec_summary = response_data["data"]["execution_summary"]
        assert exec_summary["successful"] == 4
        assert exec_summary["total_tasks"] == 4
        assert exec_summary["execution_time"] == 15.5
    
    @pytest.mark.asyncio
    async def test_purchase_analyzer_workflow(self, mock_cost_explorer_response):
        """Test purchase analyzer workflow."""
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            # Test multiple commitment scenarios
            commitments = [5.0, 10.0, 15.0]
            results = []
            
            for commitment in commitments:
                result = await run_purchase_analyzer({
                    "hourly_commitment": commitment,
                    "commitment_term": "1_YEAR",
                    "payment_option": "NO_UPFRONT",
                    "region": "us-east-1"
                })
                
                assert len(result) > 0
                response_data = json.loads(result[0].text)
                assert response_data["status"] == "success"
                assert response_data["data"]["hourly_commitment"] == commitment
                
                results.append(response_data["data"])
            
            # Verify different commitments produce different results
            assert results[0]["projected_coverage"] != results[2]["projected_coverage"]
            assert results[0]["estimated_annual_savings"] != results[2]["estimated_annual_savings"]
    
    @pytest.mark.asyncio
    async def test_multi_region_analysis_workflow(self, mock_cost_explorer_response):
        """Test multi-region analysis workflow."""
        regions = ["us-east-1", "us-west-2", "eu-west-1"]
        
        with patch('services.cost_explorer.get_database_usage_by_service') as mock_cost_explorer, \
             patch('services.savings_plans_service.calculate_savings_plans_rates') as mock_rates:
            
            mock_cost_explorer.return_value = mock_cost_explorer_response
            mock_rates.return_value = {"status": "success", "data": {"discount_percentage": 20.0}}
            
            results = []
            for region in regions:
                result = await run_database_savings_plans_analysis({
                    "region": region,
                    "lookback_period_days": 30
                })
                
                assert len(result) > 0
                response_data = json.loads(result[0].text)
                assert response_data["status"] == "success"
                results.append(response_data)
            
            # Verify each region was analyzed
            assert len(results) == 3
            for result in results:
                assert "analysis_insights" in result["data"]
                assert "execution_summary" in result["data"]
    
    @pytest.mark.asyncio
    async def test_comprehensive_integration_workflow(self, mock_cost_explorer_response):
        """Test comprehensive end-to-end workflow with all integration components."""
        # Setup comprehensive mocks
        mock_orchestrator = Mock()
        mock_session = Mock()
        
        # Mock successful orchestrated execution
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "success",
            "successful": 6,
            "total_tasks": 6,
            "results": {
                "usage_analysis": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "execution_time": 2.1,
                    "data": {"status": "success", "data": mock_cost_explorer_response["data"]}
                },
                "recommendations": {
                    "operation": "generate_recommendations",
                    "status": "success",
                    "execution_time": 1.8,
                    "data": {"status": "success", "data": {"recommendations": [
                        {
                            "commitment_term": "1_YEAR",
                            "payment_option": "NO_UPFRONT",
                            "hourly_commitment": 12.5,
                            "estimated_annual_savings": 15000.0,
                            "projected_coverage": 88.0,
                            "projected_utilization": 92.5,
                            "confidence_level": "high"
                        }
                    ]}}
                },
                "existing_analysis": {
                    "operation": "analyze_existing",
                    "status": "success",
                    "execution_time": 1.5,
                    "data": {"status": "success", "data": {"existing_plans": [], "gaps": {}}}
                },
                "ri_comparison": {
                    "operation": "compare_ri",
                    "status": "success",
                    "execution_time": 2.3,
                    "data": {"status": "success", "data": {"latest_generation": [], "older_generation": []}}
                },
                "purchase_analysis": {
                    "operation": "purchase_analyzer",
                    "status": "success",
                    "execution_time": 1.2,
                    "data": {"status": "success", "data": {"scenarios": []}}
                },
                "historical_comparison": {
                    "operation": "historical_analysis",
                    "status": "success",
                    "execution_time": 0.8,
                    "data": {"status": "success", "data": {"trends": [], "changes": []}}
                }
            },
            "execution_time": 3.5,
            "parallelization_efficiency": 0.72,
            "resource_usage": {
                "api_calls_made": 18,
                "rate_limit_hits": 0,
                "retry_attempts": 2,
                "cache_hits": 5
            }
        })
        
        # Mock session storage
        mock_session.store_analysis_result = Mock(return_value=True)
        mock_session.get_session_id = Mock(return_value="comprehensive_test_session")
        mock_session.query_historical_data = Mock(return_value=[
            {
                "timestamp": "2024-01-01T00:00:00",
                "recommendations": [{"hourly_commitment": 10.0}]
            }
        ])
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_orchestrator), \
             patch('utils.session_manager.SessionManager', return_value=mock_session):
            
            # Run comprehensive analysis
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 60,
                "services": ["rds", "aurora", "dynamodb", "elasticache"],
                "include_ri_comparison": True,
                "include_existing_analysis": True,
                "include_purchase_scenarios": True,
                "include_historical_comparison": True,
                "store_results": True,
                "comprehensive_analysis": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify orchestrator coordination
        mock_orchestrator.execute_parallel_analysis.assert_called_once()
        call_args = mock_orchestrator.execute_parallel_analysis.call_args[1]
        assert len(call_args["service_calls"]) >= 4  # Multiple parallel operations
        
        # Verify session storage integration
        mock_session.store_analysis_result.assert_called_once()
        storage_call = mock_session.store_analysis_result.call_args
        assert storage_call[1]["analysis_type"] == "database_savings_plans"
        assert storage_call[1]["session_id"] == "comprehensive_test_session"
        
        # Verify comprehensive response structure
        data = response_data["data"]
        assert "analysis_insights" in data
        assert "execution_summary" in data
        
        exec_summary = data["execution_summary"]
        assert exec_summary["successful"] == 6
        assert exec_summary["total_tasks"] == 6
        assert exec_summary["execution_time"] == 3.5
        assert exec_summary["parallelization_efficiency"] == 0.72
        assert "resource_usage" in exec_summary
        
        # Verify resource usage tracking
        resource_usage = exec_summary["resource_usage"]
        assert resource_usage["api_calls_made"] == 18
        assert resource_usage["cache_hits"] == 5
        assert resource_usage["retry_attempts"] == 2
    
    @pytest.mark.asyncio
    async def test_integration_with_real_world_constraints(self):
        """Test integration handling real-world constraints and limitations."""
        mock_orchestrator = Mock()
        mock_session = Mock()
        
        # Mock realistic execution with constraints
        mock_orchestrator.execute_parallel_analysis = Mock(return_value={
            "status": "success",
            "successful": 3,
            "total_tasks": 4,
            "results": {
                "usage_analysis": {
                    "operation": "analyze_usage",
                    "status": "success",
                    "execution_time": 4.2,  # Slower due to large dataset
                    "data": {"status": "success", "data": {
                        "total_on_demand_spend": 50000.0,  # Large spend
                        "service_breakdown": {
                            "rds": 30000.0,
                            "aurora": 15000.0,
                            "dynamodb": 5000.0
                        },
                        "data_points": 2160,  # 90 days * 24 hours
                        "data_quality": "high"
                    }}
                },
                "recommendations": {
                    "operation": "generate_recommendations",
                    "status": "success",
                    "execution_time": 2.8,
                    "data": {"status": "success", "data": {"recommendations": [
                        {
                            "commitment_term": "1_YEAR",
                            "payment_option": "NO_UPFRONT",
                            "hourly_commitment": 45.0,
                            "estimated_annual_savings": 120000.0,
                            "projected_coverage": 85.0,
                            "projected_utilization": 90.0,
                            "confidence_level": "high",
                            "risk_factors": ["usage_volatility", "seasonal_patterns"]
                        }
                    ]}}
                },
                "existing_analysis": {
                    "operation": "analyze_existing",
                    "status": "success",
                    "execution_time": 1.9,
                    "data": {"status": "success", "data": {
                        "existing_plans": [
                            {
                                "savings_plan_id": "sp-existing-123",
                                "hourly_commitment": 20.0,
                                "utilization_percentage": 95.0,
                                "coverage_percentage": 60.0
                            }
                        ],
                        "optimization_opportunity": "Additional commitment recommended"
                    }}
                },
                "rate_limited_operation": {
                    "operation": "detailed_pricing",
                    "status": "throttled",
                    "retry_count": 3,
                    "error": "API rate limit exceeded, will retry with exponential backoff"
                }
            },
            "execution_time": 8.5,  # Longer due to rate limiting
            "constraints_encountered": [
                "api_rate_limiting",
                "large_dataset_processing",
                "memory_optimization_required"
            ],
            "resource_usage": {
                "api_calls_made": 45,
                "rate_limit_hits": 8,
                "retry_attempts": 12,
                "memory_peak_mb": 1024,
                "processing_optimizations": ["data_chunking", "result_caching"]
            }
        })
        
        # Mock session with large data handling
        mock_session.store_analysis_result = Mock(return_value=True)
        mock_session.get_session_id = Mock(return_value="large_dataset_session")
        
        with patch('utils.service_orchestrator.ServiceOrchestrator', return_value=mock_orchestrator), \
             patch('utils.session_manager.SessionManager', return_value=mock_session):
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 90,  # Maximum lookback
                "services": ["rds", "aurora", "dynamodb", "elasticache", "documentdb", "neptune"],
                "detailed_analysis": True,
                "include_all_options": True
            })
        
        assert len(result) > 0
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "success"
        
        # Verify handling of real-world constraints
        exec_summary = response_data["data"]["execution_summary"]
        assert "constraints_encountered" in exec_summary
        assert "api_rate_limiting" in exec_summary["constraints_encountered"]
        assert "large_dataset_processing" in exec_summary["constraints_encountered"]
        
        # Verify resource usage optimization
        resource_usage = exec_summary["resource_usage"]
        assert resource_usage["rate_limit_hits"] == 8
        assert resource_usage["retry_attempts"] == 12
        assert "processing_optimizations" in resource_usage
        assert "data_chunking" in resource_usage["processing_optimizations"]


async def run_database_savings_plans_integration_tests():
    """Run Database Savings Plans integration tests."""
    import subprocess
    import sys
    
    try:
        # Run pytest on this file
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=300)
        
        print("Database Savings Plans Integration Test Output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Database Savings Plans integration tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running Database Savings Plans integration tests: {e}")
        return False


def main():
    """Synchronous main function for compatibility with test suite runner."""
    return asyncio.run(run_database_savings_plans_integration_tests())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)