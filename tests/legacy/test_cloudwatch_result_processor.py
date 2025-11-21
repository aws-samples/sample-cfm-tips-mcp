"""
Tests for CloudWatch Result Processor - Zero-Cost Pagination and Sorting

These tests verify that the CloudWatch result processor provides cost-based sorting
and pagination without making any additional AWS API calls.
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
from playbooks.cloudwatch.result_processor import CloudWatchResultProcessor, PaginationMetadata


class TestCloudWatchResultProcessor:
    """Test suite for CloudWatch Result Processor with zero-cost guarantee."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock pricing service to avoid any API calls
        self.mock_pricing_service = MagicMock()
        self.mock_pricing_service.get_logs_pricing.return_value = {
            'status': 'success',
            'logs_pricing': {'storage_per_gb_month': 0.03}
        }
        self.mock_pricing_service.get_metrics_pricing.return_value = {
            'status': 'success',
            'metrics_pricing': {'custom_metric_per_month': 0.30}
        }
        self.mock_pricing_service.get_alarms_pricing.return_value = {
            'status': 'success',
            'alarms_pricing': {
                'standard_alarm_per_month': 0.10,
                'high_resolution_alarm_per_month': 0.50
            }
        }
        self.mock_pricing_service.get_dashboards_pricing.return_value = {
            'status': 'success',
            'dashboards_pricing': {'dashboard_per_month': 3.00}
        }
        
        self.processor = CloudWatchResultProcessor(pricing_service=self.mock_pricing_service)
    
    def test_initialization_zero_cost_guarantee(self):
        """Test that processor initializes with zero-cost guarantee."""
        processor = CloudWatchResultProcessor()
        assert processor.page_size == 10
        assert not hasattr(processor, '_aws_clients')
    
    def test_calculate_log_group_cost_no_api_calls(self):
        """Test log group cost calculation without API calls."""
        log_group = {
            'logGroupName': 'test-log-group',
            'storedBytes': 1073741824  # 1 GB
        }
        
        # Verify no AWS clients are created or called
        with patch('boto3.client') as mock_boto3:
            cost = self.processor.calculate_log_group_cost(log_group)
            mock_boto3.assert_not_called()
        
        # Should calculate cost based on stored bytes
        assert cost == 0.03  # 1 GB * $0.03/GB/month
    
    def test_calculate_log_group_cost_zero_bytes(self):
        """Test log group cost calculation with zero stored bytes."""
        log_group = {
            'logGroupName': 'empty-log-group',
            'storedBytes': 0
        }
        
        cost = self.processor.calculate_log_group_cost(log_group)
        assert cost == 0.0
    
    def test_calculate_custom_metric_cost_no_api_calls(self):
        """Test custom metric cost calculation without API calls."""
        custom_metric = {
            'MetricName': 'CustomMetric',
            'Namespace': 'MyApp/Performance'
        }
        
        with patch('boto3.client') as mock_boto3:
            cost = self.processor.calculate_custom_metric_cost(custom_metric)
            mock_boto3.assert_not_called()
        
        assert cost == 0.30  # Custom metric cost
    
    def test_calculate_aws_metric_cost_free(self):
        """Test that AWS metrics are correctly identified as free."""
        aws_metric = {
            'MetricName': 'CPUUtilization',
            'Namespace': 'AWS/EC2'
        }
        
        cost = self.processor.calculate_custom_metric_cost(aws_metric)
        assert cost == 0.0  # AWS metrics are free
    
    def test_calculate_alarm_cost_standard(self):
        """Test standard alarm cost calculation."""
        standard_alarm = {
            'AlarmName': 'test-alarm',
            'Period': 300  # 5 minutes = standard resolution
        }
        
        with patch('boto3.client') as mock_boto3:
            cost = self.processor.calculate_alarm_cost(standard_alarm)
            mock_boto3.assert_not_called()
        
        assert cost == 0.10  # Standard alarm cost
    
    def test_calculate_alarm_cost_high_resolution(self):
        """Test high-resolution alarm cost calculation."""
        high_res_alarm = {
            'AlarmName': 'test-alarm-hr',
            'Period': 60  # 1 minute = high resolution
        }
        
        cost = self.processor.calculate_alarm_cost(high_res_alarm)
        assert cost == 0.50  # High-resolution alarm cost
    
    def test_calculate_dashboard_cost_free_tier(self):
        """Test dashboard cost calculation within free tier."""
        dashboard = {'DashboardName': 'test-dashboard'}
        
        with patch('boto3.client') as mock_boto3:
            cost = self.processor.calculate_dashboard_cost(dashboard, total_dashboards=2)
            mock_boto3.assert_not_called()
        
        assert cost == 0.0  # Within free tier (3 dashboards)
    
    def test_calculate_dashboard_cost_beyond_free_tier(self):
        """Test dashboard cost calculation beyond free tier."""
        dashboard = {'DashboardName': 'test-dashboard'}
        
        cost = self.processor.calculate_dashboard_cost(dashboard, total_dashboards=5)
        assert cost == 3.00  # Beyond free tier
    
    def test_enrich_items_with_cost_estimates_no_api_calls(self):
        """Test enriching items with cost estimates without API calls."""
        log_groups = [
            {'logGroupName': 'group1', 'storedBytes': 1073741824},  # 1 GB
            {'logGroupName': 'group2', 'storedBytes': 2147483648}   # 2 GB
        ]
        
        with patch('boto3.client') as mock_boto3:
            enriched = self.processor.enrich_items_with_cost_estimates(log_groups, 'log_groups')
            mock_boto3.assert_not_called()
        
        assert len(enriched) == 2
        assert enriched[0]['estimated_monthly_cost'] == 0.03
        assert enriched[1]['estimated_monthly_cost'] == 0.06
    
    def test_sort_by_cost_descending_no_api_calls(self):
        """Test cost-based sorting without API calls."""
        items = [
            {'name': 'item1', 'estimated_monthly_cost': 10.0},
            {'name': 'item2', 'estimated_monthly_cost': 25.0},
            {'name': 'item3', 'estimated_monthly_cost': 5.0}
        ]
        
        with patch('boto3.client') as mock_boto3:
            sorted_items = self.processor.sort_by_cost_descending(items)
            mock_boto3.assert_not_called()
        
        assert len(sorted_items) == 3
        assert sorted_items[0]['estimated_monthly_cost'] == 25.0
        assert sorted_items[1]['estimated_monthly_cost'] == 10.0
        assert sorted_items[2]['estimated_monthly_cost'] == 5.0
    
    def test_sort_by_cost_missing_field(self):
        """Test sorting with missing cost field."""
        items = [
            {'name': 'item1', 'estimated_monthly_cost': 10.0},
            {'name': 'item2'},  # Missing cost field
            {'name': 'item3', 'estimated_monthly_cost': 5.0}
        ]
        
        sorted_items = self.processor.sort_by_cost_descending(items)
        
        # Items with missing cost should be treated as 0.0 and sorted last
        assert sorted_items[0]['estimated_monthly_cost'] == 10.0
        assert sorted_items[1]['estimated_monthly_cost'] == 5.0
        assert 'estimated_monthly_cost' not in sorted_items[2]
    
    def test_create_pagination_metadata_page_1(self):
        """Test pagination metadata creation for page 1."""
        metadata = self.processor.create_pagination_metadata(total_items=25, current_page=1)
        
        assert metadata.current_page == 1
        assert metadata.page_size == 10
        assert metadata.total_items == 25
        assert metadata.total_pages == 3
        assert metadata.has_next_page is True
        assert metadata.has_previous_page is False
    
    def test_create_pagination_metadata_middle_page(self):
        """Test pagination metadata creation for middle page."""
        metadata = self.processor.create_pagination_metadata(total_items=25, current_page=2)
        
        assert metadata.current_page == 2
        assert metadata.total_pages == 3
        assert metadata.has_next_page is True
        assert metadata.has_previous_page is True
    
    def test_create_pagination_metadata_last_page(self):
        """Test pagination metadata creation for last page."""
        metadata = self.processor.create_pagination_metadata(total_items=25, current_page=3)
        
        assert metadata.current_page == 3
        assert metadata.total_pages == 3
        assert metadata.has_next_page is False
        assert metadata.has_previous_page is True
    
    def test_create_pagination_metadata_invalid_page_correction(self):
        """Test pagination metadata with invalid page number correction."""
        # Test page 0 correction
        metadata = self.processor.create_pagination_metadata(total_items=25, current_page=0)
        assert metadata.current_page == 1
        
        # Test negative page correction
        metadata = self.processor.create_pagination_metadata(total_items=25, current_page=-5)
        assert metadata.current_page == 1
    
    def test_create_pagination_metadata_empty_results(self):
        """Test pagination metadata with empty results."""
        metadata = self.processor.create_pagination_metadata(total_items=0, current_page=1)
        
        assert metadata.current_page == 1
        assert metadata.page_size == 10
        assert metadata.total_items == 0
        assert metadata.total_pages == 0
        assert metadata.has_next_page is False
        assert metadata.has_previous_page is False
    
    def test_paginate_results_page_1(self):
        """Test pagination for page 1."""
        items = [{'id': i, 'estimated_monthly_cost': i} for i in range(25)]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.paginate_results(items, page=1)
            mock_boto3.assert_not_called()
        
        assert len(result['items']) == 10
        assert result['items'][0]['id'] == 0
        assert result['items'][9]['id'] == 9
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 25
        assert result['pagination']['total_pages'] == 3
        assert result['pagination']['has_next_page'] is True
        assert result['pagination']['has_previous_page'] is False
    
    def test_paginate_results_page_2(self):
        """Test pagination for page 2."""
        items = [{'id': i, 'estimated_monthly_cost': i} for i in range(25)]
        
        result = self.processor.paginate_results(items, page=2)
        
        assert len(result['items']) == 10
        assert result['items'][0]['id'] == 10
        assert result['items'][9]['id'] == 19
        assert result['pagination']['current_page'] == 2
    
    def test_paginate_results_last_page_partial(self):
        """Test pagination for last page with partial results."""
        items = [{'id': i, 'estimated_monthly_cost': i} for i in range(25)]
        
        result = self.processor.paginate_results(items, page=3)
        
        assert len(result['items']) == 5  # Only 5 items on last page
        assert result['items'][0]['id'] == 20
        assert result['items'][4]['id'] == 24
        assert result['pagination']['current_page'] == 3
        assert result['pagination']['has_next_page'] is False
    
    def test_paginate_results_out_of_range(self):
        """Test pagination for out-of-range page."""
        items = [{'id': i, 'estimated_monthly_cost': i} for i in range(25)]
        
        result = self.processor.paginate_results(items, page=10)
        
        assert len(result['items']) == 0  # No items for out-of-range page
        assert result['pagination']['current_page'] == 10
        assert result['pagination']['total_items'] == 25
        assert result['pagination']['total_pages'] == 3
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is True
    
    def test_paginate_results_single_page(self):
        """Test pagination with single page of results."""
        items = [{'id': i, 'estimated_monthly_cost': i} for i in range(7)]
        
        result = self.processor.paginate_results(items, page=1)
        
        assert len(result['items']) == 7
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_pages'] == 1
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is False
    
    def test_process_log_groups_results_integration(self):
        """Test complete log groups processing with sorting and pagination."""
        log_groups = [
            {'logGroupName': 'small-group', 'storedBytes': 1073741824},    # 1 GB = $0.03
            {'logGroupName': 'large-group', 'storedBytes': 10737418240},   # 10 GB = $0.30
            {'logGroupName': 'medium-group', 'storedBytes': 5368709120}    # 5 GB = $0.15
        ]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.process_log_groups_results(log_groups, page=1)
            mock_boto3.assert_not_called()
        
        # Should be sorted by cost descending
        items = result['items']
        assert len(items) == 3
        assert items[0]['logGroupName'] == 'large-group'
        assert items[0]['estimated_monthly_cost'] == 0.30
        assert items[1]['logGroupName'] == 'medium-group'
        assert items[1]['estimated_monthly_cost'] == 0.15
        assert items[2]['logGroupName'] == 'small-group'
        assert items[2]['estimated_monthly_cost'] == 0.03
        
        # Check pagination metadata
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 3
        assert result['pagination']['total_pages'] == 1
    
    def test_process_metrics_results_integration(self):
        """Test complete metrics processing with sorting and pagination."""
        metrics = [
            {'MetricName': 'AWSMetric', 'Namespace': 'AWS/EC2'},
            {'MetricName': 'CustomMetric1', 'Namespace': 'MyApp/Performance'},
            {'MetricName': 'CustomMetric2', 'Namespace': 'MyApp/Business'}
        ]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.process_metrics_results(metrics, page=1)
            mock_boto3.assert_not_called()
        
        # Should be sorted by cost descending (custom metrics first)
        items = result['items']
        assert len(items) == 3
        # Custom metrics should be first (cost = 0.30 each)
        assert items[0]['Namespace'] in ['MyApp/Performance', 'MyApp/Business']
        assert items[0]['estimated_monthly_cost'] == 0.30
        assert items[1]['Namespace'] in ['MyApp/Performance', 'MyApp/Business']
        assert items[1]['estimated_monthly_cost'] == 0.30
        # AWS metric should be last (cost = 0.0)
        assert items[2]['Namespace'] == 'AWS/EC2'
        assert items[2]['estimated_monthly_cost'] == 0.0
    
    def test_process_alarms_results_integration(self):
        """Test complete alarms processing with sorting and pagination."""
        alarms = [
            {'AlarmName': 'standard-alarm', 'Period': 300},  # Standard = $0.10
            {'AlarmName': 'high-res-alarm', 'Period': 60}    # High-res = $0.50
        ]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.process_alarms_results(alarms, page=1)
            mock_boto3.assert_not_called()
        
        # Should be sorted by cost descending
        items = result['items']
        assert len(items) == 2
        assert items[0]['AlarmName'] == 'high-res-alarm'
        assert items[0]['estimated_monthly_cost'] == 0.50
        assert items[1]['AlarmName'] == 'standard-alarm'
        assert items[1]['estimated_monthly_cost'] == 0.10
    
    def test_process_dashboards_results_integration(self):
        """Test complete dashboards processing with sorting and pagination."""
        dashboards = [
            {'DashboardName': 'dashboard1'},
            {'DashboardName': 'dashboard2'},
            {'DashboardName': 'dashboard3'},
            {'DashboardName': 'dashboard4'},  # Beyond free tier
            {'DashboardName': 'dashboard5'}   # Beyond free tier
        ]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.process_dashboards_results(dashboards, page=1)
            mock_boto3.assert_not_called()
        
        # Should be sorted by cost descending (paid dashboards first)
        items = result['items']
        assert len(items) == 5
        # First dashboards should be the paid ones (beyond free tier)
        assert items[0]['estimated_monthly_cost'] == 3.00
        assert items[1]['estimated_monthly_cost'] == 3.00
        # Last dashboards should be free tier
        assert items[2]['estimated_monthly_cost'] == 0.0
        assert items[3]['estimated_monthly_cost'] == 0.0
        assert items[4]['estimated_monthly_cost'] == 0.0
    
    def test_process_recommendations_integration(self):
        """Test complete recommendations processing with sorting and pagination."""
        recommendations = [
            {'type': 'low_impact', 'potential_monthly_savings': 5.0},
            {'type': 'high_impact', 'potential_monthly_savings': 50.0},
            {'type': 'medium_impact', 'potential_monthly_savings': 20.0}
        ]
        
        with patch('boto3.client') as mock_boto3:
            result = self.processor.process_recommendations(recommendations, page=1)
            mock_boto3.assert_not_called()
        
        # Should be sorted by potential savings descending
        items = result['items']
        assert len(items) == 3
        assert items[0]['type'] == 'high_impact'
        assert items[0]['potential_monthly_savings'] == 50.0
        assert items[1]['type'] == 'medium_impact'
        assert items[1]['potential_monthly_savings'] == 20.0
        assert items[2]['type'] == 'low_impact'
        assert items[2]['potential_monthly_savings'] == 5.0
    
    def test_error_handling_graceful_degradation(self):
        """Test that errors in processing don't break the system."""
        # Test with invalid log group data
        invalid_log_groups = [
            {'invalid': 'data'},
            {'logGroupName': 'valid-group', 'storedBytes': 1073741824}
        ]
        
        result = self.processor.process_log_groups_results(invalid_log_groups, page=1)
        
        # Should still process valid items and handle invalid ones gracefully
        assert len(result['items']) == 2
        assert result['items'][0]['estimated_monthly_cost'] == 0.03  # Valid item
        assert result['items'][1]['estimated_monthly_cost'] == 0.0   # Invalid item defaulted to 0
    
    def test_no_pricing_service_fallback(self):
        """Test that processor works without pricing service using fallback pricing."""
        processor_no_pricing = CloudWatchResultProcessor(pricing_service=None)
        
        log_group = {'logGroupName': 'test', 'storedBytes': 1073741824}  # 1 GB
        
        with patch('boto3.client') as mock_boto3:
            cost = processor_no_pricing.calculate_log_group_cost(log_group)
            mock_boto3.assert_not_called()
        
        # Should use fallback pricing
        assert cost == 0.03  # Default fallback price


class TestZeroCostGuarantee:
    """Specific tests to verify zero-cost guarantee."""
    
    def test_no_boto3_imports_in_processing_methods(self):
        """Test that processing methods don't import boto3."""
        processor = CloudWatchResultProcessor()
        
        # Mock boto3 to track if it's imported
        with patch('boto3.client') as mock_boto3, \
             patch('boto3.resource') as mock_resource, \
             patch('boto3.Session') as mock_session:
            
            # Run all processing methods
            items = [{'test': 'data', 'estimated_monthly_cost': 1.0}]
            
            processor.sort_by_cost_descending(items)
            processor.paginate_results(items)
            processor.create_pagination_metadata(10, 1)
            
            # Verify no boto3 calls were made
            mock_boto3.assert_not_called()
            mock_resource.assert_not_called()
            mock_session.assert_not_called()
    
    def test_no_external_network_calls(self):
        """Test that no external network calls are made during processing."""
        processor = CloudWatchResultProcessor()
        
        # Mock requests library to ensure no HTTP calls
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post, \
             patch('urllib.request.urlopen') as mock_urlopen:
            
            # Process sample data
            log_groups = [{'logGroupName': 'test', 'storedBytes': 1073741824}]
            result = processor.process_log_groups_results(log_groups)
            
            # Verify no network calls were made
            mock_get.assert_not_called()
            mock_post.assert_not_called()
            mock_urlopen.assert_not_called()
    
    def test_memory_only_operations(self):
        """Test that all operations are memory-only."""
        processor = CloudWatchResultProcessor()
        
        # Create test data
        large_dataset = [
            {'id': i, 'estimated_monthly_cost': i * 0.1} 
            for i in range(100)
        ]
        
        # All operations should complete without external dependencies
        sorted_data = processor.sort_by_cost_descending(large_dataset)
        paginated_data = processor.paginate_results(sorted_data, page=5)
        
        # Verify results are correct
        assert len(paginated_data['items']) == 10
        assert paginated_data['pagination']['current_page'] == 5
        assert paginated_data['pagination']['total_items'] == 100
        
        # Verify highest cost items are first
        assert sorted_data[0]['estimated_monthly_cost'] == 9.9
        assert sorted_data[-1]['estimated_monthly_cost'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__])