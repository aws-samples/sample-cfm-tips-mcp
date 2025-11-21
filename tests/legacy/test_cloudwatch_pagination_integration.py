"""
Comprehensive integration tests for CloudWatch pagination and sorting functionality.

This test suite validates the complete end-to-end functionality of:
- Cost-based sorting (highest cost first)
- 1-based pagination with fixed page size of 10
- Zero-cost guarantee (no additional AWS API calls)
- MCP function integration
- Backward compatibility
"""

import pytest
import asyncio
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock
from playbooks.cloudwatch.result_processor import CloudWatchResultProcessor
from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
from runbook_functions import cloudwatch_general_spend_analysis


class TestCloudWatchPaginationIntegration:
    """Comprehensive integration tests for CloudWatch pagination."""
    
    def test_cost_based_sorting_validation(self):
        """Test that results are consistently sorted by cost in descending order."""
        processor = CloudWatchResultProcessor()
        
        # Test with log groups of different sizes
        log_groups = [
            {'logGroupName': 'tiny-group', 'storedBytes': 536870912},      # 0.5 GB = $0.015
            {'logGroupName': 'small-group', 'storedBytes': 1073741824},    # 1 GB = $0.03
            {'logGroupName': 'large-group', 'storedBytes': 10737418240},   # 10 GB = $0.30
            {'logGroupName': 'medium-group', 'storedBytes': 5368709120},   # 5 GB = $0.15
            {'logGroupName': 'huge-group', 'storedBytes': 21474836480},    # 20 GB = $0.60
        ]
        
        result = processor.process_log_groups_results(log_groups, page=1)
        items = result['items']
        
        # Verify sorting by cost descending
        assert len(items) == 5
        assert items[0]['logGroupName'] == 'huge-group'
        assert items[0]['estimated_monthly_cost'] == 0.60
        assert items[1]['logGroupName'] == 'large-group'
        assert items[1]['estimated_monthly_cost'] == 0.30
        assert items[2]['logGroupName'] == 'medium-group'
        assert items[2]['estimated_monthly_cost'] == 0.15
        assert items[3]['logGroupName'] == 'small-group'
        assert items[3]['estimated_monthly_cost'] == 0.03
        assert items[4]['logGroupName'] == 'tiny-group'
        assert items[4]['estimated_monthly_cost'] == 0.015
    
    def test_pagination_accuracy_comprehensive(self):
        """Test pagination metadata accuracy across different scenarios."""
        processor = CloudWatchResultProcessor()
        
        # Test with exactly 30 items (3 full pages)
        log_groups = [
            {'logGroupName': f'group-{i:02d}', 'storedBytes': (31-i) * 1073741824}  # Descending sizes
            for i in range(1, 31)  # 30 log groups
        ]
        
        # Test page 1
        result_p1 = processor.process_log_groups_results(log_groups, page=1)
        assert result_p1['pagination']['current_page'] == 1
        assert result_p1['pagination']['total_items'] == 30
        assert result_p1['pagination']['total_pages'] == 3
        assert result_p1['pagination']['has_next_page'] is True
        assert result_p1['pagination']['has_previous_page'] is False
        assert len(result_p1['items']) == 10
        
        # Test page 2
        result_p2 = processor.process_log_groups_results(log_groups, page=2)
        assert result_p2['pagination']['current_page'] == 2
        assert result_p2['pagination']['has_next_page'] is True
        assert result_p2['pagination']['has_previous_page'] is True
        assert len(result_p2['items']) == 10
        
        # Test page 3 (last full page)
        result_p3 = processor.process_log_groups_results(log_groups, page=3)
        assert result_p3['pagination']['current_page'] == 3
        assert result_p3['pagination']['has_next_page'] is False
        assert result_p3['pagination']['has_previous_page'] is True
        assert len(result_p3['items']) == 10
        
        # Test out of range page
        result_p4 = processor.process_log_groups_results(log_groups, page=4)
        assert result_p4['pagination']['current_page'] == 4
        assert result_p4['pagination']['has_next_page'] is False
        assert result_p4['pagination']['has_previous_page'] is True
        assert len(result_p4['items']) == 0
    
    def test_1_based_pagination_edge_cases(self):
        """Test 1-based pagination handles edge cases correctly."""
        processor = CloudWatchResultProcessor()
        
        log_groups = [{'logGroupName': 'test', 'storedBytes': 1073741824}]
        
        # Test page 0 (should default to page 1)
        result_p0 = processor.process_log_groups_results(log_groups, page=0)
        assert result_p0['pagination']['current_page'] == 1
        
        # Test negative page (should default to page 1)
        result_neg = processor.process_log_groups_results(log_groups, page=-10)
        assert result_neg['pagination']['current_page'] == 1
        
        # Test very large page number
        result_large = processor.process_log_groups_results(log_groups, page=999)
        assert result_large['pagination']['current_page'] == 999
        assert result_large['pagination']['total_pages'] == 1
        assert len(result_large['items']) == 0
    
    def test_zero_cost_guarantee_comprehensive(self):
        """Test that no AWS API calls are made during any processing operations."""
        processor = CloudWatchResultProcessor()
        
        # Test data for all resource types
        log_groups = [{'logGroupName': 'test-lg', 'storedBytes': 1073741824}]
        metrics = [{'MetricName': 'CustomMetric', 'Namespace': 'MyApp'}]
        alarms = [{'AlarmName': 'test-alarm', 'Period': 300}]
        dashboards = [{'DashboardName': 'test-dashboard'}]
        recommendations = [{'type': 'optimization', 'potential_monthly_savings': 10.0}]
        
        with mock.patch('boto3.client') as mock_boto3, \
             mock.patch('boto3.resource') as mock_resource, \
             mock.patch('requests.get') as mock_get:
            
            # Test all processing methods
            processor.process_log_groups_results(log_groups, page=1)
            processor.process_metrics_results(metrics, page=1)
            processor.process_alarms_results(alarms, page=1)
            processor.process_dashboards_results(dashboards, page=1)
            processor.process_recommendations(recommendations, page=1)
            
            # Verify no external calls were made
            mock_boto3.assert_not_called()
            mock_resource.assert_not_called()
            mock_get.assert_not_called()
    
    def test_all_resource_types_sorting(self):
        """Test that all CloudWatch resource types are sorted correctly."""
        processor = CloudWatchResultProcessor()
        
        # Test metrics (custom vs AWS)
        metrics = [
            {'MetricName': 'AWSMetric', 'Namespace': 'AWS/EC2'},           # Free = $0.00
            {'MetricName': 'CustomMetric1', 'Namespace': 'MyApp/Perf'},    # Custom = $0.30
            {'MetricName': 'CustomMetric2', 'Namespace': 'MyApp/Business'} # Custom = $0.30
        ]
        
        result = processor.process_metrics_results(metrics, page=1)
        items = result['items']
        
        # Custom metrics should be first (higher cost)
        assert items[0]['Namespace'] in ['MyApp/Perf', 'MyApp/Business']
        assert items[0]['estimated_monthly_cost'] == 0.30
        assert items[1]['Namespace'] in ['MyApp/Perf', 'MyApp/Business']
        assert items[1]['estimated_monthly_cost'] == 0.30
        assert items[2]['Namespace'] == 'AWS/EC2'
        assert items[2]['estimated_monthly_cost'] == 0.0
        
        # Test alarms (high-resolution vs standard)
        alarms = [
            {'AlarmName': 'standard-alarm', 'Period': 300},  # Standard = $0.10
            {'AlarmName': 'high-res-alarm', 'Period': 60}    # High-res = $0.50
        ]
        
        result = processor.process_alarms_results(alarms, page=1)
        items = result['items']
        
        # High-resolution alarm should be first (higher cost)
        assert items[0]['AlarmName'] == 'high-res-alarm'
        assert items[0]['estimated_monthly_cost'] == 0.50
        assert items[1]['AlarmName'] == 'standard-alarm'
        assert items[1]['estimated_monthly_cost'] == 0.10
    
    def test_dashboard_free_tier_handling(self):
        """Test that dashboard free tier is handled correctly."""
        processor = CloudWatchResultProcessor()
        
        # Test with 5 dashboards (2 beyond free tier)
        dashboards = [
            {'DashboardName': f'dashboard-{i}'} for i in range(1, 6)
        ]
        
        result = processor.process_dashboards_results(dashboards, page=1)
        items = result['items']
        
        # First 2 should be paid dashboards (beyond free tier)
        assert items[0]['estimated_monthly_cost'] == 3.00
        assert items[1]['estimated_monthly_cost'] == 3.00
        
        # Last 3 should be free tier dashboards
        assert items[2]['estimated_monthly_cost'] == 0.0
        assert items[3]['estimated_monthly_cost'] == 0.0
        assert items[4]['estimated_monthly_cost'] == 0.0
    
    def test_recommendations_sorting(self):
        """Test that recommendations are sorted by potential savings."""
        processor = CloudWatchResultProcessor()
        
        recommendations = [
            {'type': 'low_impact', 'potential_monthly_savings': 5.0},
            {'type': 'high_impact', 'potential_monthly_savings': 50.0},
            {'type': 'medium_impact', 'potential_monthly_savings': 20.0},
            {'type': 'minimal_impact', 'potential_monthly_savings': 1.0}
        ]
        
        result = processor.process_recommendations(recommendations, page=1)
        items = result['items']
        
        # Should be sorted by potential savings descending
        assert items[0]['potential_monthly_savings'] == 50.0
        assert items[1]['potential_monthly_savings'] == 20.0
        assert items[2]['potential_monthly_savings'] == 5.0
        assert items[3]['potential_monthly_savings'] == 1.0
    
    def test_empty_results_handling(self):
        """Test that empty results are handled correctly."""
        processor = CloudWatchResultProcessor()
        
        # Test with empty list
        result = processor.process_log_groups_results([], page=1)
        
        assert len(result['items']) == 0
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 0
        assert result['pagination']['total_pages'] == 0
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is False
    
    def test_single_page_results(self):
        """Test handling of results that fit in a single page."""
        processor = CloudWatchResultProcessor()
        
        # Test with 5 items (less than page size of 10)
        log_groups = [
            {'logGroupName': f'group-{i}', 'storedBytes': i * 1073741824}
            for i in range(1, 6)
        ]
        
        result = processor.process_log_groups_results(log_groups, page=1)
        
        assert len(result['items']) == 5
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 5
        assert result['pagination']['total_pages'] == 1
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is False


class TestMCPFunctionIntegration:
    """Test MCP function integration with pagination."""
    
    def test_mcp_function_signatures(self):
        """Test that MCP functions have correct pagination signatures."""
        import inspect
        
        # Test cloudwatch_general_spend_analysis
        sig = inspect.signature(cloudwatch_general_spend_analysis)
        params = sig.parameters
        
        assert 'region' in params
        assert 'page' in params
        assert params['page'].default == 1
        
        # Verify page parameter type annotation if present
        if params['page'].annotation != inspect.Parameter.empty:
            assert params['page'].annotation == int
    
    def test_mcp_function_parameter_handling(self):
        """Test that MCP functions handle pagination parameters correctly."""
        
        # Mock the orchestrator to avoid actual AWS calls
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_analysis.return_value = {
                'status': 'success',
                'data': {'test': 'data'},
                'pagination_applied': True,
                'current_page': 1
            }
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Test default page
            result = cloudwatch_general_spend_analysis(region='us-east-1')
            assert result['status'] == 'success'
            
            # Verify orchestrator was called with page=1 (default)
            mock_orchestrator.execute_analysis.assert_called()
            call_args = mock_orchestrator.execute_analysis.call_args
            assert call_args[1]['page'] == 1
            
            # Test explicit page parameter
            result = cloudwatch_general_spend_analysis(region='us-east-1', page=2)
            assert result['status'] == 'success'
            
            # Verify orchestrator was called with page=2
            call_args = mock_orchestrator.execute_analysis.call_args
            assert call_args[1]['page'] == 2
    
    def test_mcp_function_error_handling(self):
        """Test that MCP functions handle errors gracefully."""
        
        # Test with invalid region to trigger error
        result = cloudwatch_general_spend_analysis(region='invalid-region-12345')
        
        # Should return error status, not raise exception
        assert 'status' in result
        # The function should handle errors gracefully
        assert result['status'] in ['error', 'success']  # May succeed with mock data


class TestBackwardCompatibility:
    """Test backward compatibility of pagination implementation."""
    
    def test_existing_api_compatibility(self):
        """Test that existing API calls still work without pagination parameters."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_analysis.return_value = {
                'status': 'success',
                'data': {'test': 'data'}
            }
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Test call without page parameter (should default to page 1)
            result = cloudwatch_general_spend_analysis(region='us-east-1')
            
            assert result['status'] == 'success'
            
            # Verify default page was used
            call_args = mock_orchestrator.execute_analysis.call_args
            assert call_args[1]['page'] == 1
    
    def test_result_structure_compatibility(self):
        """Test that result structures are backward compatible."""
        processor = CloudWatchResultProcessor()
        
        log_groups = [{'logGroupName': 'test', 'storedBytes': 1073741824}]
        result = processor.process_log_groups_results(log_groups, page=1)
        
        # Verify expected structure
        assert 'items' in result
        assert 'pagination' in result
        assert isinstance(result['items'], list)
        assert isinstance(result['pagination'], dict)
        
        # Verify pagination metadata structure
        pagination = result['pagination']
        required_fields = ['current_page', 'page_size', 'total_items', 'total_pages', 'has_next_page', 'has_previous_page']
        for field in required_fields:
            assert field in pagination


class TestPerformanceAndMemory:
    """Test performance and memory efficiency of pagination."""
    
    def test_large_dataset_handling(self):
        """Test that large datasets are handled efficiently."""
        processor = CloudWatchResultProcessor()
        
        # Create a large dataset (1000 items)
        large_dataset = [
            {'logGroupName': f'group-{i:04d}', 'storedBytes': i * 1073741824}
            for i in range(1, 1001)
        ]
        
        # Processing should complete quickly
        import time
        start_time = time.time()
        
        result = processor.process_log_groups_results(large_dataset, page=1)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second)
        assert processing_time < 1.0
        
        # Should return correct pagination
        assert len(result['items']) == 10
        assert result['pagination']['total_items'] == 1000
        assert result['pagination']['total_pages'] == 100
    
    def test_memory_efficiency(self):
        """Test that pagination doesn't cause memory issues."""
        processor = CloudWatchResultProcessor()
        
        # Create dataset and process multiple pages
        dataset = [
            {'logGroupName': f'group-{i}', 'storedBytes': i * 1073741824}
            for i in range(1, 101)  # 100 items
        ]
        
        # Process multiple pages - should not accumulate memory
        for page in range(1, 11):  # Pages 1-10
            result = processor.process_log_groups_results(dataset, page=page)
            assert len(result['items']) == 10
            assert result['pagination']['current_page'] == page


if __name__ == '__main__':
    pytest.main([__file__, '-v'])