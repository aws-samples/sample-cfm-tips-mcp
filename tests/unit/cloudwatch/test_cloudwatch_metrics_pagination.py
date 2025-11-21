#!/usr/bin/env python3
"""
Unit tests for CloudWatch metrics pagination functionality.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from playbooks.cloudwatch.result_processor import CloudWatchResultProcessor


class TestCloudWatchMetricsPagination:
    """Unit tests for CloudWatch metrics pagination functionality."""
    
    def test_metrics_pagination_with_large_dataset(self):
        """Test that metrics pagination correctly limits results to 10 items per page."""
        processor = CloudWatchResultProcessor()
        
        # Create 25 metrics to test pagination (should result in 3 pages)
        metrics = []
        for i in range(25):
            metric = {
                'MetricName': f'CustomMetric{i:02d}',
                'Namespace': f'MyApp/Service{i % 3}',  # Mix of namespaces
                'Dimensions': [
                    {'Name': 'InstanceId', 'Value': f'i-{i:010d}'},
                    {'Name': 'Environment', 'Value': 'production' if i % 2 == 0 else 'staging'}
                ]
            }
            metrics.append(metric)
        
        # Test page 1 - should have exactly 10 items
        result_p1 = processor.process_metrics_results(metrics, page=1)
        
        assert len(result_p1['items']) == 10, f"Page 1 should have exactly 10 items, got {len(result_p1['items'])}"
        assert result_p1['pagination']['current_page'] == 1
        assert result_p1['pagination']['total_items'] == 25
        assert result_p1['pagination']['total_pages'] == 3
        assert result_p1['pagination']['has_next_page'] is True
        assert result_p1['pagination']['has_previous_page'] is False
        
        # Test page 2 - should have exactly 10 items
        result_p2 = processor.process_metrics_results(metrics, page=2)
        
        assert len(result_p2['items']) == 10, f"Page 2 should have exactly 10 items, got {len(result_p2['items'])}"
        assert result_p2['pagination']['current_page'] == 2
        assert result_p2['pagination']['has_next_page'] is True
        assert result_p2['pagination']['has_previous_page'] is True
        
        # Test page 3 - should have exactly 5 items (remainder)
        result_p3 = processor.process_metrics_results(metrics, page=3)
        
        assert len(result_p3['items']) == 5, f"Page 3 should have exactly 5 items, got {len(result_p3['items'])}"
        assert result_p3['pagination']['current_page'] == 3
        assert result_p3['pagination']['has_next_page'] is False
        assert result_p3['pagination']['has_previous_page'] is True
        
        # Verify dimensions are preserved (not truncated)
        for item in result_p1['items']:
            assert 'Dimensions' in item, "Dimensions should be preserved"
            assert len(item['Dimensions']) == 2, "All dimensions should be preserved"
    
    def test_metrics_cost_sorting(self):
        """Test that metrics are sorted by cost (custom metrics first)."""
        processor = CloudWatchResultProcessor()
        
        # Create mix of AWS and custom metrics
        metrics = [
            {'MetricName': 'CPUUtilization', 'Namespace': 'AWS/EC2'},  # Free
            {'MetricName': 'CustomMetric1', 'Namespace': 'MyApp/Performance'},  # $0.30
            {'MetricName': 'NetworkIn', 'Namespace': 'AWS/EC2'},  # Free
            {'MetricName': 'CustomMetric2', 'Namespace': 'MyApp/Business'},  # $0.30
        ]
        
        result = processor.process_metrics_results(metrics, page=1)
        items = result['items']
        
        # Custom metrics should be first (higher cost)
        assert items[0]['Namespace'] in ['MyApp/Performance', 'MyApp/Business']
        assert items[0]['estimated_monthly_cost'] == 0.30
        assert items[1]['Namespace'] in ['MyApp/Performance', 'MyApp/Business']
        assert items[1]['estimated_monthly_cost'] == 0.30
        
        # AWS metrics should be last (free)
        assert items[2]['Namespace'] == 'AWS/EC2'
        assert items[2]['estimated_monthly_cost'] == 0.0
        assert items[3]['Namespace'] == 'AWS/EC2'
        assert items[3]['estimated_monthly_cost'] == 0.0
    
    def test_metrics_dimensions_preservation(self):
        """Test that metric dimensions are fully preserved, not truncated."""
        processor = CloudWatchResultProcessor()
        
        # Create metric with many dimensions
        metrics = [{
            'MetricName': 'ComplexMetric',
            'Namespace': 'MyApp/Complex',
            'Dimensions': [
                {'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'},
                {'Name': 'Environment', 'Value': 'production'},
                {'Name': 'Service', 'Value': 'web-server'},
                {'Name': 'Region', 'Value': 'us-east-1'},
                {'Name': 'AZ', 'Value': 'us-east-1a'},
                {'Name': 'Version', 'Value': 'v2.1.3'},
                {'Name': 'Team', 'Value': 'platform-engineering'},
            ]
        }]
        
        result = processor.process_metrics_results(metrics, page=1)
        item = result['items'][0]
        
        # All dimensions should be preserved
        assert len(item['Dimensions']) == 7, f"Expected 7 dimensions, got {len(item['Dimensions'])}"
        
        # Verify specific dimensions are present
        dimension_names = [d['Name'] for d in item['Dimensions']]
        expected_names = ['InstanceId', 'Environment', 'Service', 'Region', 'AZ', 'Version', 'Team']
        
        for expected_name in expected_names:
            assert expected_name in dimension_names, f"Dimension {expected_name} should be preserved"
    
    def test_empty_metrics_pagination(self):
        """Test pagination with empty metrics list."""
        processor = CloudWatchResultProcessor()
        
        result = processor.process_metrics_results([], page=1)
        
        assert len(result['items']) == 0
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 0
        assert result['pagination']['total_pages'] == 0
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is False
    
    def test_single_page_metrics(self):
        """Test pagination with metrics that fit in a single page."""
        processor = CloudWatchResultProcessor()
        
        # Create 5 metrics (less than page size of 10)
        metrics = [
            {'MetricName': f'Metric{i}', 'Namespace': 'MyApp/Test'}
            for i in range(5)
        ]
        
        result = processor.process_metrics_results(metrics, page=1)
        
        assert len(result['items']) == 5
        assert result['pagination']['current_page'] == 1
        assert result['pagination']['total_items'] == 5
        assert result['pagination']['total_pages'] == 1
        assert result['pagination']['has_next_page'] is False
        assert result['pagination']['has_previous_page'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])