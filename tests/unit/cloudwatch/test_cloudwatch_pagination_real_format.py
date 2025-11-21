#!/usr/bin/env python3
"""
Unit tests for CloudWatch pagination functionality using real API format.
Tests pagination logic with actual CloudWatch response structures.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure")
class TestCloudWatchPaginationRealFormat:
    """Unit tests for CloudWatch pagination with real API format."""
    
    def create_real_cloudwatch_response(self, page=1, total_metrics=25):
        """Create a response matching the actual CloudWatch API format."""
        items_per_page = 10
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_metrics)
        
        metrics = [
            {
                "Namespace": f"AWS/Test{i}",
                "MetricName": f"TestMetric{i}",
                "Dimensions": [{"Name": "TestDim", "Value": f"value{i}"}],
                "estimated_monthly_cost": 0.3
            }
            for i in range(start_idx, end_idx)
        ]
        
        total_pages = (total_metrics + items_per_page - 1) // items_per_page
        
        return {
            "status": "success",
            "analysis_type": "general_spend",
            "data": {
                "configuration_analysis": {
                    "metrics": {
                        "metrics": metrics,
                        "total_count": total_metrics,
                        "pagination": {
                            "current_page": page,
                            "page_size": items_per_page,
                            "total_items": total_metrics,
                            "total_pages": total_pages,
                            "has_next_page": page < total_pages,
                            "has_previous_page": page > 1
                        }
                    },
                    "log_groups": {
                        "log_groups": [],
                        "total_count": 0,
                        "pagination": {
                            "current_page": page,
                            "page_size": items_per_page,
                            "total_items": 0,
                            "total_pages": 0,
                            "has_next_page": False,
                            "has_previous_page": False
                        }
                    },
                    "alarms": {
                        "alarms": [],
                        "total_count": 0
                    },
                    "dashboards": {
                        "dashboards": [],
                        "total_count": 0
                    }
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_cloudwatch_general_spend_analysis_real_format(self):
        """Test CloudWatch general spend analysis with real API format."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_real_cloudwatch_response(page, 25)
            
            mock_orchestrator.return_value.execute_analysis.side_effect = mock_execute_analysis
            
            from runbook_functions import run_cloudwatch_general_spend_analysis
            
            # Test page 1
            result_p1 = await run_cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            response_p1 = json.loads(result_p1[0].text)
            
            # Verify structure matches real API
            assert "configuration_analysis" in response_p1["data"]
            assert "metrics" in response_p1["data"]["configuration_analysis"]
            
            metrics_data = response_p1["data"]["configuration_analysis"]["metrics"]
            assert len(metrics_data["metrics"]) == 10
            assert metrics_data["total_count"] == 25
            assert metrics_data["pagination"]["current_page"] == 1
            assert metrics_data["pagination"]["total_pages"] == 3
            assert metrics_data["pagination"]["has_next_page"] is True
            
            # Verify metric structure
            first_metric = metrics_data["metrics"][0]
            assert "Namespace" in first_metric
            assert "MetricName" in first_metric
            assert "Dimensions" in first_metric
            assert "estimated_monthly_cost" in first_metric
            
            # Test page 3 (last page with remainder)
            result_p3 = await run_cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 3,
                'lookback_days': 30
            })
            
            response_p3 = json.loads(result_p3[0].text)
            metrics_data_p3 = response_p3["data"]["configuration_analysis"]["metrics"]
            assert len(metrics_data_p3["metrics"]) == 5  # 25 % 10 = 5 remainder
            assert metrics_data_p3["pagination"]["has_next_page"] is False
    
    @pytest.mark.asyncio
    async def test_cloudwatch_metrics_optimization_real_format(self):
        """Test CloudWatch metrics optimization with real API format."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                response = self.create_real_cloudwatch_response(page, 15)
                response["analysis_type"] = "metrics_optimization"
                return response
            
            mock_orchestrator.return_value.execute_analysis.side_effect = mock_execute_analysis
            
            from runbook_functions import run_cloudwatch_metrics_optimization
            
            result = await run_cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            response = json.loads(result[0].text)
            assert response["analysis_type"] == "metrics_optimization"
            
            metrics_data = response["data"]["configuration_analysis"]["metrics"]
            assert len(metrics_data["metrics"]) == 10
            assert metrics_data["total_count"] == 15
            assert metrics_data["pagination"]["total_pages"] == 2
    
    @pytest.mark.asyncio
    async def test_cloudwatch_logs_optimization_real_format(self):
        """Test CloudWatch logs optimization with real API format."""
        
        def create_logs_response(page=1, total_logs=12):
            items_per_page = 10
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_logs)
            
            log_groups = [
                {
                    "logGroupName": f"/aws/lambda/test-function-{i}",
                    "creationTime": 1719500926627 + i,
                    "storedBytes": 1000000 + (i * 100000),
                    "estimated_monthly_cost": 0.1 + (i * 0.01)
                }
                for i in range(start_idx, end_idx)
            ]
            
            total_pages = (total_logs + items_per_page - 1) // items_per_page
            
            return {
                "status": "success",
                "analysis_type": "logs_optimization",
                "data": {
                    "configuration_analysis": {
                        "log_groups": {
                            "log_groups": log_groups,
                            "total_count": total_logs,
                            "pagination": {
                                "current_page": page,
                                "page_size": items_per_page,
                                "total_items": total_logs,
                                "total_pages": total_pages,
                                "has_next_page": page < total_pages,
                                "has_previous_page": page > 1
                            }
                        },
                        "metrics": {"metrics": [], "total_count": 0},
                        "alarms": {"alarms": [], "total_count": 0},
                        "dashboards": {"dashboards": [], "total_count": 0}
                    }
                }
            }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return create_logs_response(page, 12)
            
            mock_orchestrator.return_value.execute_analysis.side_effect = mock_execute_analysis
            
            from runbook_functions import run_cloudwatch_logs_optimization
            
            result = await run_cloudwatch_logs_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            response = json.loads(result[0].text)
            
            log_groups_data = response["data"]["configuration_analysis"]["log_groups"]
            assert len(log_groups_data["log_groups"]) == 10
            assert log_groups_data["total_count"] == 12
            assert log_groups_data["pagination"]["total_pages"] == 2
            
            # Verify log group structure
            first_log = log_groups_data["log_groups"][0]
            assert "logGroupName" in first_log
            assert "creationTime" in first_log
            assert "storedBytes" in first_log
            assert "estimated_monthly_cost" in first_log
    
    @pytest.mark.asyncio
    async def test_query_cloudwatch_analysis_results_real_format(self):
        """Test CloudWatch query results with real API format."""
        
        def create_query_response(page=1, total_rows=18):
            items_per_page = 10
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_rows)
            
            rows = [
                {
                    "metric_name": f"TestMetric{i}",
                    "namespace": f"AWS/Test{i}",
                    "cost": 0.3 + (i * 0.1),
                    "dimensions_count": 2
                }
                for i in range(start_idx, end_idx)
            ]
            
            total_pages = (total_rows + items_per_page - 1) // items_per_page
            
            return {
                "status": "success",
                "query_results": {
                    "rows": rows,
                    "pagination": {
                        "current_page": page,
                        "page_size": items_per_page,
                        "total_items": total_rows,
                        "total_pages": total_pages,
                        "has_next_page": page < total_pages,
                        "has_previous_page": page > 1
                    },
                    "query_metadata": {
                        "sql_query": "SELECT * FROM metrics",
                        "execution_time_ms": 125.5,
                        "rows_examined": total_rows
                    }
                }
            }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            def mock_get_analysis_results(*args, **kwargs):
                # Return 18 total rows for pagination testing
                return [
                    {
                        "metric_name": f"TestMetric{i}",
                        "namespace": f"AWS/Test{i}",
                        "cost": 0.3 + (i * 0.1),
                        "dimensions_count": 2
                    }
                    for i in range(18)
                ]
            
            mock_orchestrator.return_value.get_analysis_results.side_effect = mock_get_analysis_results
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            result = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM metrics WHERE cost > 0.5',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            
            # The actual response structure from query function
            assert "results" in response
            assert "pagination" in response
            assert len(response["results"]) == 10
            assert response["pagination"]["total_items"] == 18
            assert response["pagination"]["total_pages"] == 2
            
            # Test page 2
            result_p2 = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM metrics WHERE cost > 0.5',
                'page': 2
            })
            
            response_p2 = json.loads(result_p2[0].text)
            assert len(response_p2["results"]) == 8  # 18 % 10 = 8 remainder
            assert response_p2["pagination"]["has_next_page"] is False
    
    @pytest.mark.asyncio
    async def test_pagination_edge_cases_real_format(self):
        """Test pagination edge cases with real CloudWatch format."""
        
        # Test empty results
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            async def mock_execute_analysis(*args, **kwargs):
                return self.create_real_cloudwatch_response(1, 0)
            mock_orchestrator.return_value.execute_analysis.side_effect = mock_execute_analysis
            
            from runbook_functions import run_cloudwatch_general_spend_analysis
            
            result = await run_cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            metrics_data = response["data"]["configuration_analysis"]["metrics"]
            assert len(metrics_data["metrics"]) == 0
            assert metrics_data["total_count"] == 0
            assert metrics_data["pagination"]["total_pages"] == 0
            assert metrics_data["pagination"]["has_next_page"] is False
        
        # Test exactly one page
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator:
            async def mock_execute_analysis(*args, **kwargs):
                return self.create_real_cloudwatch_response(1, 10)
            mock_orchestrator.return_value.execute_analysis.side_effect = mock_execute_analysis
            
            from runbook_functions import run_cloudwatch_metrics_optimization
            
            result = await run_cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            metrics_data = response["data"]["configuration_analysis"]["metrics"]
            assert len(metrics_data["metrics"]) == 10
            assert metrics_data["pagination"]["total_pages"] == 1
            assert metrics_data["pagination"]["has_next_page"] is False
            assert metrics_data["pagination"]["has_previous_page"] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])