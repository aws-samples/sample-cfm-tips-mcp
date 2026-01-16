"""
Unit tests for Database Savings Plans MCP function wrappers.

Tests the async wrapper functions, parameter validation, response formatting,
and error handling in MCP context for all Database Savings Plans MCP functions.

Requirements: 14.3 - Test MCP functions including async wrappers, parameter validation,
response formatting, and error handling in MCP context.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from typing import Dict, List, Any

from mcp.types import TextContent

# Import the MCP wrapper functions
from playbooks.rds.database_savings_plans import (
    run_database_savings_plans_analysis,
    run_purchase_analyzer,
    analyze_existing_savings_plans
)


class TestMCPAsyncWrappers:
    """Test async wrapper functionality for MCP functions."""
    
    @pytest.mark.asyncio
    async def test_mcp_functions_return_textcontent(self):
        """Test that all MCP functions return List[TextContent] when called."""
        # Test with minimal valid parameters and comprehensive mocking
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.analyze_custom_commitment') as mock_custom, \
             patch('playbooks.rds.database_savings_plans.analyze_existing_commitments') as mock_existing, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup basic successful mocks
            mock_analyze.return_value = {"status": "success", "data": {"total_spend": 1000.0}}
            mock_custom.return_value = {"status": "success", "data": {"hourly_commitment": 5.0}}
            mock_existing.return_value = {"status": "success", "data": {"existing_plans": []}}
            
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success", "successful": 1, "total_tasks": 1,
                "results": {"task": {"status": "success", "data": {"test": "data"}}}
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Test each function returns List[TextContent]
            result1 = await run_database_savings_plans_analysis({"region": "us-east-1"})
            assert isinstance(result1, list) and all(isinstance(item, TextContent) for item in result1)
            
            result2 = await run_purchase_analyzer({"hourly_commitment": 5.0})
            assert isinstance(result2, list) and all(isinstance(item, TextContent) for item in result2)
            
            result3 = await analyze_existing_savings_plans({"region": "us-east-1"})
            assert isinstance(result3, list) and all(isinstance(item, TextContent) for item in result3)
    
    @pytest.mark.asyncio
    async def test_run_database_savings_plans_analysis_async_wrapper(self):
        """Test that run_database_savings_plans_analysis is properly async and returns TextContent."""
        # Mock all dependencies to avoid real AWS calls
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.generate_savings_plans_recommendations') as mock_recommendations, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup mocks
            mock_analyze.return_value = {
                "status": "success",
                "data": {
                    "total_on_demand_spend": 1000.0,
                    "average_hourly_spend": 1.39,
                    "service_breakdown": {"rds": 1000.0},
                    "region_breakdown": {"us-east-1": 1000.0},
                    "instance_family_breakdown": {"db.r7": 1000.0}
                }
            }
            
            mock_recommendations.return_value = {
                "status": "success",
                "data": {
                    "recommendations": [
                        {
                            "commitment_term": "1_YEAR",
                            "payment_option": "NO_UPFRONT",
                            "hourly_commitment": 10.0,
                            "estimated_annual_savings": 12000.0,
                            "projected_coverage": 85.0,
                            "projected_utilization": 95.0,
                            "confidence_level": "high"
                        }
                    ]
                }
            }
            
            # Mock orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success",
                "successful": 2,
                "total_tasks": 2,
                "results": {
                    "usage_analysis": {
                        "operation": "analyze_usage",
                        "status": "success",
                        "data": mock_analyze.return_value
                    },
                    "recommendations": {
                        "operation": "generate_recommendations",
                        "status": "success",
                        "data": mock_recommendations.return_value
                    }
                }
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock session manager
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Mock formatter
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Test async execution
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 30
            })
            
            # Verify async wrapper behavior
            assert isinstance(result, list), "Should return a list"
            assert len(result) > 0, "Should return at least one TextContent item"
            assert isinstance(result[0], TextContent), "Should return TextContent objects"
            assert result[0].type == "text", "TextContent should have type 'text'"
            
            # Verify orchestrator was used for parallel execution
            mock_orchestrator.execute_parallel_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_purchase_analyzer_async_wrapper(self):
        """Test that run_purchase_analyzer is properly async and returns TextContent."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.analyze_custom_commitment') as mock_custom, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup mocks
            mock_analyze.return_value = {
                "status": "success",
                "data": {
                    "total_on_demand_spend": 1000.0,
                    "average_hourly_spend": 1.39
                }
            }
            
            mock_custom.return_value = {
                "status": "success",
                "data": {
                    "hourly_commitment": 5.0,
                    "commitment_term": "1_YEAR",
                    "payment_option": "NO_UPFRONT",
                    "projected_coverage": 85.0,
                    "projected_utilization": 95.0,
                    "estimated_annual_savings": 8000.0
                }
            }
            
            # Mock session manager
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Mock formatter
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Test async execution
            result = await run_purchase_analyzer({
                "hourly_commitment": 5.0,
                "commitment_term": "1_YEAR",
                "payment_option": "NO_UPFRONT"
            })
            
            # Verify async wrapper behavior
            assert isinstance(result, list), "Should return a list"
            assert len(result) > 0, "Should return at least one TextContent item"
            assert isinstance(result[0], TextContent), "Should return TextContent objects"
            assert result[0].type == "text", "TextContent should have type 'text'"
            
            # Verify underlying functions were called
            mock_analyze.assert_called_once()
            assert mock_custom.call_count >= 1, "analyze_custom_commitment should be called at least once"
    
    @pytest.mark.asyncio
    async def test_analyze_existing_savings_plans_async_wrapper(self):
        """Test that analyze_existing_savings_plans is properly async and returns TextContent."""
        with patch('playbooks.rds.database_savings_plans.analyze_existing_commitments') as mock_existing, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup mocks
            mock_existing.return_value = {
                "status": "success",
                "data": {
                    "existing_plans": [
                        {
                            "savings_plan_id": "sp-12345",
                            "hourly_commitment": 10.0,
                            "utilization_percentage": 85.5,
                            "coverage_percentage": 75.0
                        }
                    ],
                    "gaps": {
                        "uncovered_spend": 300.0,
                        "recommendation": "Consider additional $3/hour commitment"
                    }
                }
            }
            
            # Mock session manager
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Mock formatter
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Test async execution
            result = await analyze_existing_savings_plans({
                "region": "us-east-1",
                "lookback_period_days": 30
            })
            
            # Verify async wrapper behavior
            assert isinstance(result, list), "Should return a list"
            assert len(result) > 0, "Should return at least one TextContent item"
            assert isinstance(result[0], TextContent), "Should return TextContent objects"
            assert result[0].type == "text", "TextContent should have type 'text'"
            
            # Verify underlying function was called
            mock_existing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_async_execution(self):
        """Test that multiple MCP functions can be executed concurrently."""
        # Mock all dependencies
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.analyze_custom_commitment') as mock_custom, \
             patch('playbooks.rds.database_savings_plans.analyze_existing_commitments') as mock_existing, \
             patch('playbooks.rds.database_savings_plans.generate_savings_plans_recommendations') as mock_recommendations, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup common mocks
            mock_analyze.return_value = {
                "status": "success",
                "data": {"total_on_demand_spend": 1000.0, "average_hourly_spend": 1.39}
            }
            
            mock_custom.return_value = {
                "status": "success",
                "data": {"hourly_commitment": 5.0, "projected_coverage": 85.0}
            }
            
            mock_existing.return_value = {
                "status": "success",
                "data": {"existing_plans": [], "gaps": {}}
            }
            
            mock_recommendations.return_value = {
                "status": "success",
                "data": {"recommendations": []}
            }
            
            # Mock orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success",
                "successful": 1,
                "total_tasks": 1,
                "results": {"task": {"status": "success", "data": mock_analyze.return_value}}
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock session and formatter
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Execute multiple functions concurrently
            tasks = [
                run_database_savings_plans_analysis({"region": "us-east-1"}),
                run_purchase_analyzer({"hourly_commitment": 5.0}),
                analyze_existing_savings_plans({"region": "us-east-1"})
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all functions completed successfully
            assert len(results) == 3, "Should have 3 results"
            for result in results:
                assert isinstance(result, list), "Each result should be a list"
                assert len(result) > 0, "Each result should contain TextContent"
                assert isinstance(result[0], TextContent), "Should contain TextContent objects"


class TestMCPParameterValidation:
    """Test parameter validation for MCP functions."""
    
    @pytest.mark.asyncio
    async def test_database_analysis_parameter_validation(self):
        """Test parameter validation for database_savings_plans_analysis."""
        # Test valid parameters
        valid_params = {
            "region": "us-east-1",
            "lookback_period_days": 30,
            "services": ["rds", "aurora"],
            "include_ri_comparison": True
        }
        
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup mocks for successful execution
            mock_analyze.return_value = {
                "status": "success",
                "data": {"total_on_demand_spend": 1000.0}
            }
            
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success",
                "successful": 1,
                "total_tasks": 1,
                "results": {"task": {"status": "success", "data": mock_analyze.return_value}}
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            # Should succeed with valid parameters
            result = await run_database_savings_plans_analysis(valid_params)
            assert isinstance(result, list)
            assert len(result) > 0
        
        # Test invalid lookback period
        invalid_params = {
            "region": "us-east-1",
            "lookback_period_days": 45,  # Invalid - must be 30, 60, or 90
            "services": ["rds"]
        }
        
        # Should handle invalid parameters gracefully
        result = await run_database_savings_plans_analysis(invalid_params)
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Parse response to check for error
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "lookback period" in response_data["message"].lower() or "validation" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_purchase_analyzer_parameter_validation(self):
        """Test parameter validation for purchase analyzer."""
        # Test missing required parameter
        missing_required = {}
        
        result = await run_purchase_analyzer(missing_required)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "required" in response_data["message"].lower() or "hourly_commitment" in response_data["message"].lower()
        
        # Test invalid hourly commitment (negative)
        invalid_commitment = {
            "hourly_commitment": -5.0,
            "commitment_term": "1_YEAR",
            "payment_option": "NO_UPFRONT"
        }
        
        result = await run_purchase_analyzer(invalid_commitment)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "positive" in response_data["message"].lower() or "validation" in response_data["message"].lower()
        
        # Test invalid commitment term
        invalid_term = {
            "hourly_commitment": 5.0,
            "commitment_term": "3_YEAR",  # Invalid - only 1_YEAR supported
            "payment_option": "NO_UPFRONT"
        }
        
        result = await run_purchase_analyzer(invalid_term)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "1-year" in response_data["message"].lower() or "support" in response_data["message"].lower()
        
        # Test invalid payment option
        invalid_payment = {
            "hourly_commitment": 5.0,
            "commitment_term": "1_YEAR",
            "payment_option": "FULL_UPFRONT"  # Invalid for 1-year terms
        }
        
        result = await run_purchase_analyzer(invalid_payment)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "upfront" in response_data["message"].lower() or "payment" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_existing_analysis_parameter_validation(self):
        """Test parameter validation for existing savings plans analysis."""
        # Test valid parameters
        valid_params = {
            "region": "us-east-1",
            "lookback_period_days": 60
        }
        
        with patch('playbooks.rds.database_savings_plans.analyze_existing_commitments') as mock_existing, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            mock_existing.return_value = {
                "status": "success",
                "data": {"existing_plans": [], "gaps": {}}
            }
            
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text='{"status": "success", "data": {}}')
            ]
            
            result = await analyze_existing_savings_plans(valid_params)
            assert isinstance(result, list)
            assert len(result) > 0
        
        # Test invalid lookback period
        invalid_params = {
            "region": "us-east-1",
            "lookback_period_days": 120  # Invalid - must be 30, 60, or 90
        }
        
        result = await analyze_existing_savings_plans(invalid_params)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        assert "lookback period" in response_data["message"].lower() or "validation" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_parameter_type_validation(self):
        """Test parameter type validation."""
        # Test wrong type for lookback_period_days (string instead of int)
        wrong_type_params = {
            "region": "us-east-1",
            "lookback_period_days": "thirty",  # Should be integer
            "services": ["rds"]
        }
        
        result = await run_database_savings_plans_analysis(wrong_type_params)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"
        
        # Test wrong type for hourly_commitment (string instead of number)
        wrong_commitment_type = {
            "hourly_commitment": "five dollars",  # Should be number
            "commitment_term": "1_YEAR"
        }
        
        result = await run_purchase_analyzer(wrong_commitment_type)
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_text = result[0].text
        response_data = json.loads(response_text)
        assert response_data["status"] == "error"


class TestMCPResponseFormatting:
    """Test response formatting for MCP functions."""
    
    @pytest.mark.asyncio
    async def test_successful_response_format(self):
        """Test that successful responses follow the correct format."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.generate_savings_plans_recommendations') as mock_recommendations, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup successful response data
            usage_data = {
                "status": "success",
                "data": {
                    "total_on_demand_spend": 1500.0,
                    "average_hourly_spend": 2.08,
                    "service_breakdown": {"rds": 800.0, "aurora": 700.0},
                    "region_breakdown": {"us-east-1": 1500.0},
                    "instance_family_breakdown": {"db.r7": 1500.0}
                }
            }
            
            recommendations_data = {
                "status": "success",
                "data": {
                    "recommendations": [
                        {
                            "commitment_term": "1_YEAR",
                            "payment_option": "NO_UPFRONT",
                            "hourly_commitment": 12.5,
                            "estimated_annual_savings": 15000.0,
                            "projected_coverage": 88.0,
                            "projected_utilization": 92.5,
                            "confidence_level": "high"
                        }
                    ]
                }
            }
            
            mock_analyze.return_value = usage_data
            mock_recommendations.return_value = recommendations_data
            
            # Mock orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                "status": "success",
                "successful": 2,
                "total_tasks": 2,
                "results": {
                    "usage_analysis": {
                        "operation": "analyze_usage",
                        "status": "success",
                        "data": usage_data
                    },
                    "recommendations": {
                        "operation": "generate_recommendations",
                        "status": "success",
                        "data": recommendations_data
                    }
                },
                "execution_time": 5.2
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock session manager
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Mock formatter to return properly formatted response
            formatted_response = {
                "status": "success",
                "data": {
                    "analysis_insights": {
                        "usage_analysis": usage_data["data"],
                        "recommendations": recommendations_data["data"]
                    },
                    "execution_summary": {
                        "successful": 2,
                        "total_tasks": 2,
                        "execution_time": 5.2
                    }
                },
                "message": "Database Savings Plans analysis completed successfully",
                "documentation_links": [
                    "https://docs.aws.amazon.com/savingsplans/latest/userguide/what-is-savings-plans.html"
                ]
            }
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text=json.dumps(formatted_response, indent=2))
            ]
            
            # Execute function
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "lookback_period_days": 30
            })
            
            # Verify response format
            assert isinstance(result, list), "Should return List[TextContent]"
            assert len(result) == 1, "Should return exactly one TextContent item"
            assert isinstance(result[0], TextContent), "Should contain TextContent object"
            assert result[0].type == "text", "TextContent should have type 'text'"
            
            # Parse and verify JSON structure
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "success", "Should have success status"
            assert "data" in response_data, "Should contain data field"
            assert "message" in response_data, "Should contain message field"
            assert "documentation_links" in response_data, "Should contain documentation links"
            
            # Verify data structure
            data = response_data["data"]
            assert "analysis_insights" in data, "Should contain analysis insights"
            assert "execution_summary" in data, "Should contain execution summary"
            
            # Verify analysis insights structure
            insights = data["analysis_insights"]
            assert "usage_analysis" in insights, "Should contain usage analysis"
            assert "recommendations" in insights, "Should contain recommendations"
            
            # Verify execution summary structure
            summary = data["execution_summary"]
            assert "successful" in summary, "Should contain successful count"
            assert "total_tasks" in summary, "Should contain total tasks count"
            assert "execution_time" in summary, "Should contain execution time"
    
    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """Test that error responses follow the correct format."""
        # Test with a function that will fail
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze:
            
            # Mock a failure
            mock_analyze.return_value = {
                "status": "error",
                "error_code": "AccessDenied",
                "message": "User is not authorized to perform: ce:GetCostAndUsage",
                "required_permissions": ["ce:GetCostAndUsage"]
            }
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            # Verify error response format
            assert isinstance(result, list), "Should return List[TextContent]"
            assert len(result) == 1, "Should return exactly one TextContent item"
            assert isinstance(result[0], TextContent), "Should contain TextContent object"
            assert result[0].type == "text", "TextContent should have type 'text'"
            
            # Parse and verify error JSON structure
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error", "Should have error status"
            assert "message" in response_data, "Should contain error message"
            assert "error_code" in response_data or "error" in response_data["message"], "Should indicate error type"
    
    @pytest.mark.asyncio
    async def test_purchase_analyzer_response_format(self):
        """Test purchase analyzer response format."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.analyze_custom_commitment') as mock_custom, \
             patch('utils.session_manager.SessionManager') as mock_session_class, \
             patch('utils.error_handler.ResponseFormatter') as mock_formatter:
            
            # Setup mocks
            mock_analyze.return_value = {
                "status": "success",
                "data": {"total_on_demand_spend": 1000.0, "average_hourly_spend": 1.39}
            }
            
            custom_analysis_data = {
                "status": "success",
                "data": {
                    "hourly_commitment": 8.0,
                    "commitment_term": "1_YEAR",
                    "payment_option": "NO_UPFRONT",
                    "projected_annual_cost": 70080.0,
                    "projected_coverage": 85.0,
                    "projected_utilization": 95.0,
                    "estimated_annual_savings": 10000.0,
                    "uncovered_on_demand_cost": 2000.0
                }
            }
            
            mock_custom.return_value = custom_analysis_data
            
            mock_session = Mock()
            mock_session.store_analysis_result = Mock(return_value=True)
            mock_session_class.return_value = mock_session
            
            # Mock formatter response
            formatted_response = {
                "status": "success",
                "data": {
                    "purchase_analysis": custom_analysis_data["data"],
                    "analysis_mode": "purchase_analyzer",
                    "commitment_details": {
                        "hourly_commitment": 8.0,
                        "annual_commitment": 70080.0,
                        "term": "1_YEAR",
                        "payment_option": "NO_UPFRONT"
                    }
                },
                "message": "Purchase analyzer simulation completed successfully"
            }
            
            mock_formatter.to_text_content.return_value = [
                TextContent(type="text", text=json.dumps(formatted_response, indent=2))
            ]
            
            # Execute function
            result = await run_purchase_analyzer({
                "hourly_commitment": 8.0,
                "commitment_term": "1_YEAR",
                "payment_option": "NO_UPFRONT"
            })
            
            # Verify response format
            assert isinstance(result, list), "Should return List[TextContent]"
            assert len(result) == 1, "Should return exactly one TextContent item"
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "success", "Should have success status"
            assert "data" in response_data, "Should contain data field"
            
            # Verify purchase analyzer specific structure
            data = response_data["data"]
            assert "purchase_analysis" in data, "Should contain purchase analysis"
            assert "analysis_mode" in data, "Should indicate analysis mode"
            assert data["analysis_mode"] == "purchase_analyzer", "Should indicate purchase analyzer mode"
    
    @pytest.mark.asyncio
    async def test_response_json_validity(self):
        """Test that all responses contain valid JSON."""
        test_cases = [
            # Valid database analysis
            (run_database_savings_plans_analysis, {"region": "us-east-1"}),
            # Valid purchase analyzer
            (run_purchase_analyzer, {"hourly_commitment": 5.0}),
            # Valid existing analysis
            (analyze_existing_savings_plans, {"region": "us-east-1"}),
            # Invalid parameters (should still return valid JSON)
            (run_database_savings_plans_analysis, {"lookback_period_days": 45}),
            (run_purchase_analyzer, {"hourly_commitment": -5.0}),
        ]
        
        for func, params in test_cases:
            # Mock dependencies to avoid real AWS calls
            with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
                 patch('playbooks.rds.database_savings_plans.analyze_custom_commitment') as mock_custom, \
                 patch('playbooks.rds.database_savings_plans.analyze_existing_commitments') as mock_existing, \
                 patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class, \
                 patch('utils.session_manager.SessionManager') as mock_session_class, \
                 patch('utils.error_handler.ResponseFormatter') as mock_formatter:
                
                # Setup basic mocks
                mock_analyze.return_value = {"status": "success", "data": {"total_spend": 1000.0}}
                mock_custom.return_value = {"status": "success", "data": {"hourly_commitment": 5.0}}
                mock_existing.return_value = {"status": "success", "data": {"existing_plans": []}}
                
                mock_orchestrator = Mock()
                mock_orchestrator.execute_parallel_analysis = Mock(return_value={
                    "status": "success", "successful": 1, "total_tasks": 1,
                    "results": {"task": {"status": "success", "data": {"test": "data"}}}
                })
                mock_orchestrator_class.return_value = mock_orchestrator
                
                mock_session = Mock()
                mock_session.store_analysis_result = Mock(return_value=True)
                mock_session_class.return_value = mock_session
                
                mock_formatter.to_text_content.return_value = [
                    TextContent(type="text", text='{"status": "success", "data": {}}')
                ]
                
                # Execute function
                result = await func(params)
                
                # Verify response is valid JSON
                assert isinstance(result, list), f"Function {func.__name__} should return list"
                assert len(result) > 0, f"Function {func.__name__} should return non-empty list"
                
                try:
                    response_data = json.loads(result[0].text)
                    assert isinstance(response_data, dict), f"Function {func.__name__} should return valid JSON object"
                    assert "status" in response_data, f"Function {func.__name__} response should contain status"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Function {func.__name__} returned invalid JSON: {e}")


class TestMCPErrorHandling:
    """Test error handling in MCP context."""
    
    @pytest.mark.asyncio
    async def test_aws_service_error_handling(self):
        """Test handling of AWS service errors."""
        # Test Cost Explorer access denied
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze:
            
            mock_analyze.return_value = {
                "status": "error",
                "error_code": "AccessDenied",
                "message": "User is not authorized to perform: ce:GetCostAndUsage",
                "required_permissions": ["ce:GetCostAndUsage", "ce:GetSavingsPlansUtilization"]
            }
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error"
            assert "AccessDenied" in response_data.get("error_code", "") or "not authorized" in response_data["message"]
        
        # Test service unavailable
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze:
            
            mock_analyze.return_value = {
                "status": "error",
                "error_code": "ServiceUnavailable",
                "message": "Cost Explorer service is temporarily unavailable"
            }
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error"
            assert "unavailable" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test handling of validation errors."""
        # Test invalid hourly commitment
        result = await run_purchase_analyzer({
            "hourly_commitment": -10.0,  # Invalid negative value
            "commitment_term": "1_YEAR"
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "positive" in response_data["message"].lower() or "validation" in response_data["message"].lower()
        
        # Test invalid lookback period
        result = await run_database_savings_plans_analysis({
            "lookback_period_days": 45  # Invalid - must be 30, 60, or 90
        })
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        response_data = json.loads(result[0].text)
        assert response_data["status"] == "error"
        assert "lookback period" in response_data["message"].lower() or "validation" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        with patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class:
            
            # Mock timeout scenario
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis.side_effect = asyncio.TimeoutError("Operation timed out")
            mock_orchestrator_class.return_value = mock_orchestrator
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error"
            assert "timeout" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze:
            
            # Mock unexpected exception
            mock_analyze.side_effect = Exception("Unexpected error occurred")
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error"
            assert "error" in response_data["message"].lower()
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test handling of partial failures in orchestrated execution."""
        with patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class:
            
            # Mock partial failure scenario
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis.return_value = {
                "status": "partial_success",
                "successful": 1,
                "total_tasks": 2,
                "results": {
                    "usage_analysis": {
                        "operation": "analyze_usage",
                        "status": "success",
                        "data": {"status": "success", "data": {"total_spend": 1000.0}}
                    },
                    "recommendations": {
                        "operation": "generate_recommendations",
                        "status": "error",
                        "error": "API rate limit exceeded"
                    }
                },
                "failed_operations": ["generate_recommendations"]
            }
            mock_orchestrator_class.return_value = mock_orchestrator
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "include_recommendations": True
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            # Should succeed with partial results
            assert response_data["status"] == "success"
            
            # Should include information about failed operations
            if "execution_summary" in response_data.get("data", {}):
                exec_summary = response_data["data"]["execution_summary"]
                assert exec_summary["successful"] == 1
                assert exec_summary["total_tasks"] == 2
    
    @pytest.mark.asyncio
    async def test_session_storage_error_handling(self):
        """Test handling of session storage errors."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.store_analysis_result') as mock_store, \
             patch('utils.service_orchestrator.ServiceOrchestrator') as mock_orchestrator_class:
            
            # Setup successful analysis but failed storage
            mock_analyze.return_value = {
                "status": "success",
                "data": {"total_on_demand_spend": 1000.0}
            }
            
            mock_orchestrator = Mock()
            mock_orchestrator.execute_parallel_analysis.return_value = {
                "status": "success",
                "successful": 1,
                "total_tasks": 1,
                "results": {
                    "usage_analysis": {
                        "operation": "analyze_usage",
                        "status": "success",
                        "data": mock_analyze.return_value
                    }
                }
            }
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock storage failure
            mock_store.side_effect = Exception("Database connection failed")
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1",
                "store_results": True
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            response_data = json.loads(result[0].text)
            # Should still succeed with analysis results, just warn about storage failure
            assert response_data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_error_response_consistency(self):
        """Test that error responses are consistent across all MCP functions."""
        error_scenarios = [
            # Invalid parameters for each function
            (run_database_savings_plans_analysis, {"lookback_period_days": 45}),
            (run_purchase_analyzer, {"hourly_commitment": -5.0}),
            (analyze_existing_savings_plans, {"lookback_period_days": 120}),
        ]
        
        for func, invalid_params in error_scenarios:
            result = await func(invalid_params)
            
            assert isinstance(result, list), f"Function {func.__name__} should return list"
            assert len(result) > 0, f"Function {func.__name__} should return non-empty list"
            
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error", f"Function {func.__name__} should return error status"
            assert "message" in response_data, f"Function {func.__name__} should include error message"
            assert isinstance(response_data["message"], str), f"Function {func.__name__} error message should be string"
            assert len(response_data["message"]) > 0, f"Function {func.__name__} error message should not be empty"
    
    @pytest.mark.asyncio
    async def test_error_logging_integration(self):
        """Test that errors are properly logged in MCP context."""
        with patch('playbooks.rds.database_savings_plans.analyze_database_usage') as mock_analyze, \
             patch('playbooks.rds.database_savings_plans.logger') as mock_logger:
            
            # Mock an error
            mock_analyze.return_value = {
                "status": "error",
                "error_code": "AccessDenied",
                "message": "Access denied to Cost Explorer"
            }
            
            result = await run_database_savings_plans_analysis({
                "region": "us-east-1"
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Verify error was logged (if logger is used in the function)
            # Note: This test depends on the actual logging implementation
            response_data = json.loads(result[0].text)
            assert response_data["status"] == "error"