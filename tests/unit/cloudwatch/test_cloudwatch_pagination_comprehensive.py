#!/usr/bin/env python3
"""
Comprehensive unit tests for CloudWatch API pagination functionality.
Tests all CloudWatch MCP tools that support pagination with large datasets (23 items)
to verify cardinality of results (page 1, 10 items, 3 total pages).

NOTE: These tests are currently incompatible with the actual API structure.
The tests expect a flat data.items + data.pagination structure, but the actual
CloudWatch API returns nested structures (logs.log_groups + logs.pagination, etc.).
These tests need to be refactored to match the actual API structure.
See test_cloudwatch_metrics_pagination.py for the correct approach.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import the MCP functions that support pagination
# Note: These tests are outdated and need refactoring
# from playbooks.cloudwatch.cloudwatch_optimization import (
#     run_cloudwatch_general_spend_analysis_mcp,
#     run_cloudwatch_metrics_optimization_mcp,
#     run_cloudwatch_logs_optimization_mcp,
#     run_cloudwatch_alarms_and_dashboards_optimization_mcp,
#     run_cloudwatch_comprehensive_optimization_tool_mcp,
#     query_cloudwatch_analysis_results_mcp
# )


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure - see test_cloudwatch_metrics_pagination.py for correct approach")
class TestCloudWatchPaginationComprehensive:
    """Comprehensive unit tests for all CloudWatch APIs that support pagination."""
    
    @pytest.fixture
    def mock_large_dataset(self):
        """Create a large dataset with 23 items for pagination testing."""
        return [
            {
                'id': f'item-{i:03d}',
                'name': f'CloudWatchResource{i:02d}',
                'type': 'metric' if i % 3 == 0 else 'log_group' if i % 3 == 1 else 'alarm',
                'cost': round(10.50 + (i * 2.25), 2),
                'region': 'us-east-1' if i % 2 == 0 else 'us-west-2',
                'namespace': f'AWS/Service{i % 5}',
                'dimensions': [
                    {'Name': 'InstanceId', 'Value': f'i-{i:010d}'},
                    {'Name': 'Environment', 'Value': 'production' if i % 2 == 0 else 'staging'}
                ],
                'created_date': (datetime.now() - timedelta(days=i)).isoformat(),
                'monthly_cost': round(15.75 + (i * 1.5), 2),
                'estimated_savings': round(5.25 + (i * 0.75), 2)
            }
            for i in range(23)  # 23 items = 3 pages (10, 10, 3)
        ]
    
    @pytest.fixture
    def mock_pagination_response(self, mock_large_dataset):
        """Create a mock response with pagination metadata."""
        def create_paginated_response(items, page=1, items_per_page=10):
            total_items = len(items)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_items = items[start_idx:end_idx]
            
            return {
                'status': 'success',
                'data': {
                    'items': page_items,
                    'pagination': {
                        'current_page': page,
                        'total_items': total_items,
                        'total_pages': total_pages,
                        'items_per_page': items_per_page,
                        'has_next_page': page < total_pages,
                        'has_previous_page': page > 1
                    }
                },
                'analysis_metadata': {
                    'region': 'us-east-1',
                    'lookback_days': 30,
                    'timestamp': datetime.now().isoformat()
                }
            }
        return create_paginated_response
    
    @pytest.mark.asyncio
    async def test_cloudwatch_general_spend_analysis_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch general spend analysis pagination with 23 items."""
        
        # Mock the external MCP function call that the runbook function uses
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_general_spend_analysis', create=True) as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                return mock_pagination_response(mock_large_dataset, page)
            
            mock_api.side_effect = side_effect
            
            # Test page 1 - should have exactly 10 items
            result_p1 = await cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            assert response_p1['data']['pagination']['has_next_page'] is True
            assert response_p1['data']['pagination']['has_previous_page'] is False
            
            # Test page 2 - should have exactly 10 items
            result_p2 = await cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 2,
                'lookback_days': 30
            })
            
            response_p2 = json.loads(result_p2[0].text)
            assert response_p2['data']['pagination']['current_page'] == 2
            assert len(response_p2['data']['items']) == 10
            assert response_p2['data']['pagination']['has_next_page'] is True
            assert response_p2['data']['pagination']['has_previous_page'] is True
            
            # Test page 3 - should have exactly 3 items (remainder)
            result_p3 = await cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 3,
                'lookback_days': 30
            })
            
            response_p3 = json.loads(result_p3[0].text)
            assert response_p3['data']['pagination']['current_page'] == 3
            assert len(response_p3['data']['items']) == 3
            assert response_p3['data']['pagination']['has_next_page'] is False
            assert response_p3['data']['pagination']['has_previous_page'] is True
    
    @pytest.mark.asyncio
    async def test_cloudwatch_metrics_optimization_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch metrics optimization pagination with 23 items."""
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_metrics_optimization') as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                # Filter to metrics only for this test
                metrics_data = [item for item in mock_large_dataset if item['type'] == 'metric']
                # Pad to 23 items if needed
                while len(metrics_data) < 23:
                    metrics_data.extend(metrics_data[:23-len(metrics_data)])
                return mock_pagination_response(metrics_data[:23], page)
            
            mock_api.side_effect = side_effect
            
            # Test pagination cardinality
            result_p1 = await cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Verify all items are metrics
            for item in response_p1['data']['items']:
                assert item['type'] == 'metric'
    
    @pytest.mark.asyncio
    async def test_cloudwatch_logs_optimization_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch logs optimization pagination with 23 items."""
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_logs_optimization') as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                # Filter to log groups only for this test
                logs_data = [item for item in mock_large_dataset if item['type'] == 'log_group']
                # Pad to 23 items if needed
                while len(logs_data) < 23:
                    logs_data.extend(logs_data[:23-len(logs_data)])
                return mock_pagination_response(logs_data[:23], page)
            
            mock_api.side_effect = side_effect
            
            # Test pagination cardinality
            result_p1 = await cloudwatch_logs_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30,
                'log_group_names': ['/aws/lambda/test-function']
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Test page 3 for remainder
            result_p3 = await cloudwatch_logs_optimization({
                'region': 'us-east-1',
                'page': 3,
                'lookback_days': 30
            })
            
            response_p3 = json.loads(result_p3[0].text)
            assert len(response_p3['data']['items']) == 3
    
    @pytest.mark.asyncio
    async def test_cloudwatch_alarms_and_dashboards_optimization_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch alarms and dashboards optimization pagination with 23 items."""
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_alarms_and_dashboards_optimization') as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                # Filter to alarms only for this test
                alarms_data = [item for item in mock_large_dataset if item['type'] == 'alarm']
                # Pad to 23 items if needed
                while len(alarms_data) < 23:
                    alarms_data.extend(alarms_data[:23-len(alarms_data)])
                return mock_pagination_response(alarms_data[:23], page)
            
            mock_api.side_effect = side_effect
            
            # Test pagination cardinality
            result_p1 = await cloudwatch_alarms_and_dashboards_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30,
                'alarm_names': ['TestAlarm1', 'TestAlarm2']
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Verify all items are alarms
            for item in response_p1['data']['items']:
                assert item['type'] == 'alarm'
    
    @pytest.mark.asyncio
    async def test_cloudwatch_comprehensive_optimization_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch comprehensive optimization pagination with 23 items."""
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_comprehensive_optimization') as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                return mock_pagination_response(mock_large_dataset, page)
            
            mock_api.side_effect = side_effect
            
            # Test pagination cardinality
            result_p1 = await cloudwatch_comprehensive_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30,
                'functionalities': ['general_spend', 'metrics', 'logs', 'alarms']
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Test all three pages
            for page_num in [1, 2, 3]:
                result = await cloudwatch_comprehensive_optimization({
                    'region': 'us-east-1',
                    'page': page_num,
                    'lookback_days': 30
                })
                
                response = json.loads(result[0].text)
                expected_items = 10 if page_num < 3 else 3
                assert len(response['data']['items']) == expected_items
                assert response['data']['pagination']['current_page'] == page_num
    
    @pytest.mark.asyncio
    async def test_query_cloudwatch_analysis_results_pagination(self, mock_large_dataset, mock_pagination_response):
        """Test CloudWatch analysis results query pagination with 23 items."""
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_api:
            # Configure mock to return paginated responses
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                return mock_pagination_response(mock_large_dataset, page)
            
            mock_api.side_effect = side_effect
            
            # Test SQL query with pagination
            result_p1 = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM cloudwatch_analysis WHERE cost > 10',
                'page': 1,
                'session_id': 'test-session-123'
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Test complex SQL query pagination
            result_p2 = await query_cloudwatch_analysis_results({
                'query': 'SELECT name, cost, region FROM cloudwatch_analysis ORDER BY cost DESC',
                'page': 2
            })
            
            response_p2 = json.loads(result_p2[0].text)
            assert response_p2['data']['pagination']['current_page'] == 2
            assert len(response_p2['data']['items']) == 10
    
    @pytest.mark.asyncio
    async def test_pagination_edge_cases(self, mock_pagination_response):
        """Test pagination edge cases with various dataset sizes."""
        
        # Test with empty dataset
        empty_dataset = []
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_general_spend_analysis') as mock_api:
            mock_api.return_value = mock_pagination_response(empty_dataset, 1)
            
            result = await cloudwatch_general_spend_analysis({
                'region': 'us-east-1',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            assert len(response['data']['items']) == 0
            assert response['data']['pagination']['total_items'] == 0
            assert response['data']['pagination']['total_pages'] == 0
            assert response['data']['pagination']['has_next_page'] is False
            assert response['data']['pagination']['has_previous_page'] is False
        
        # Test with exactly 10 items (single page)
        single_page_dataset = [{'id': f'item-{i}', 'name': f'Item{i}'} for i in range(10)]
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_metrics_optimization') as mock_api:
            mock_api.return_value = mock_pagination_response(single_page_dataset, 1)
            
            result = await cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            assert len(response['data']['items']) == 10
            assert response['data']['pagination']['total_items'] == 10
            assert response['data']['pagination']['total_pages'] == 1
            assert response['data']['pagination']['has_next_page'] is False
            assert response['data']['pagination']['has_previous_page'] is False
        
        # Test with 21 items (3 pages: 10, 10, 1)
        three_page_dataset = [{'id': f'item-{i}', 'name': f'Item{i}'} for i in range(21)]
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_logs_optimization') as mock_api:
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                return mock_pagination_response(three_page_dataset, page)
            
            mock_api.side_effect = side_effect
            
            # Test page 3 with 1 item
            result = await cloudwatch_logs_optimization({
                'region': 'us-east-1',
                'page': 3
            })
            
            response = json.loads(result[0].text)
            assert len(response['data']['items']) == 1
            assert response['data']['pagination']['current_page'] == 3
            assert response['data']['pagination']['total_items'] == 21
            assert response['data']['pagination']['total_pages'] == 3
    
    @pytest.mark.asyncio
    async def test_pagination_data_integrity(self, mock_large_dataset, mock_pagination_response):
        """Test that pagination preserves data integrity across pages."""
        
        with patch('runbook_functions.mcp_cfm_tips_cloudwatch_comprehensive_optimization') as mock_api:
            def side_effect(*args, **kwargs):
                page = kwargs.get('page', 1)
                return mock_pagination_response(mock_large_dataset, page)
            
            mock_api.side_effect = side_effect
            
            # Collect all items across all pages
            all_items = []
            for page_num in [1, 2, 3]:
                result = await cloudwatch_comprehensive_optimization({
                    'region': 'us-east-1',
                    'page': page_num
                })
                
                response = json.loads(result[0].text)
                all_items.extend(response['data']['items'])
            
            # Verify we got all 23 items
            assert len(all_items) == 23
            
            # Verify no duplicates
            item_ids = [item['id'] for item in all_items]
            assert len(item_ids) == len(set(item_ids))
            
            # Verify all original items are present
            original_ids = {item['id'] for item in mock_large_dataset}
            collected_ids = {item['id'] for item in all_items}
            assert original_ids == collected_ids
    
    @pytest.mark.asyncio
    async def test_pagination_metadata_consistency(self, mock_large_dataset, mock_pagination_response):
        """Test that pagination metadata is consistent across all CloudWatch APIs."""
        
        apis_to_test = [
            ('general_spend', cloudwatch_general_spend_analysis, 'mcp_cfm_tips_cloudwatch_general_spend_analysis'),
            ('metrics', cloudwatch_metrics_optimization, 'mcp_cfm_tips_cloudwatch_metrics_optimization'),
            ('logs', cloudwatch_logs_optimization, 'mcp_cfm_tips_cloudwatch_logs_optimization'),
            ('alarms', cloudwatch_alarms_and_dashboards_optimization, 'mcp_cfm_tips_cloudwatch_alarms_dashboards'),
            ('comprehensive', cloudwatch_comprehensive_optimization, 'mcp_cfm_tips_cloudwatch_comprehensive')
        ]
        
        for api_name, api_func, mock_path in apis_to_test:
            with patch(f'runbook_functions.{mock_path}') as mock_api:
                def side_effect(*args, **kwargs):
                    page = kwargs.get('page', 1)
                    return mock_pagination_response(mock_large_dataset, page)
                
                mock_api.side_effect = side_effect
                
                # Test page 1 for each API
                result = await api_func({
                    'region': 'us-east-1',
                    'page': 1,
                    'lookback_days': 30
                })
                
                response = json.loads(result[0].text)
                
                # Verify consistent pagination metadata structure
                assert 'pagination' in response['data']
                pagination = response['data']['pagination']
                
                assert pagination['current_page'] == 1
                assert pagination['total_items'] == 23
                assert pagination['total_pages'] == 3
                assert pagination['items_per_page'] == 10
                assert pagination['has_next_page'] is True
                assert pagination['has_previous_page'] is False
                
                # Verify items count
                assert len(response['data']['items']) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])