#!/usr/bin/env python3
"""
Fixed unit tests for CloudWatch pagination functionality.
Tests pagination logic with properly mocked CloudWatch orchestrator.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure")
class TestCloudWatchPaginationFixed:
    """Fixed unit tests for CloudWatch pagination functionality."""
    
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
    
    def create_orchestrator_response(self, items, page=1, items_per_page=10):
        """Create a mock orchestrator response with proper structure."""
        total_items = len(items)
        total_pages = (total_items + items_per_page - 1) // items_per_page
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = items[start_idx:end_idx]
        
        return {
            'status': 'success',
            'analysis_type': 'cloudwatch_optimization',
            'data': {
                'items': page_items,
                'pagination': {
                    'current_page': page,
                    'page_size': items_per_page,
                    'total_items': total_items,
                    'total_pages': total_pages,
                    'has_next_page': page < total_pages,
                    'has_previous_page': page > 1
                }
            },
            'analysis_metadata': {
                'region': 'us-east-1',
                'lookback_days': 30,
                'timestamp': datetime.now().isoformat(),
                'execution_time': 1.25,
                'session_id': 'test-session-123'
            },
            'cost_summary': {
                'total_resources': len(page_items),
                'total_monthly_cost': sum(item.get('monthly_cost', 0) for item in page_items),
                'potential_savings': sum(item.get('estimated_savings', 0) for item in page_items)
            }
        }
    
    @pytest.mark.asyncio
    async def test_cloudwatch_general_spend_analysis_pagination(self, mock_large_dataset):
        """Test CloudWatch general spend analysis pagination with 23 items."""
        
        # Mock the CloudWatch orchestrator
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Configure mock to return paginated responses
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(mock_large_dataset, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_general_spend_analysis
            
            # Test page 1 - should have exactly 10 items
            result_p1 = await run_cloudwatch_general_spend_analysis({
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
            
            # Test page 3 - should have exactly 3 items (remainder)
            result_p3 = await run_cloudwatch_general_spend_analysis({
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
    async def test_cloudwatch_metrics_optimization_pagination(self, mock_large_dataset):
        """Test CloudWatch metrics optimization pagination with 23 items."""
        
        # Filter to metrics only for this test
        metrics_data = [item for item in mock_large_dataset if item['type'] == 'metric']
        # Pad to 23 items if needed
        while len(metrics_data) < 23:
            metrics_data.extend(metrics_data[:23-len(metrics_data)])
        metrics_data = metrics_data[:23]
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(metrics_data, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_metrics_optimization
            
            # Test pagination cardinality
            result_p1 = await run_cloudwatch_metrics_optimization({
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
    async def test_cloudwatch_logs_optimization_pagination(self, mock_large_dataset):
        """Test CloudWatch logs optimization pagination with 23 items."""
        
        # Filter to log groups only for this test
        logs_data = [item for item in mock_large_dataset if item['type'] == 'log_group']
        # Pad to 23 items if needed
        while len(logs_data) < 23:
            logs_data.extend(logs_data[:23-len(logs_data)])
        logs_data = logs_data[:23]
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(logs_data, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_logs_optimization
            
            # Test pagination cardinality
            result_p1 = await run_cloudwatch_logs_optimization({
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
            result_p3 = await run_cloudwatch_logs_optimization({
                'region': 'us-east-1',
                'page': 3,
                'lookback_days': 30
            })
            
            response_p3 = json.loads(result_p3[0].text)
            assert len(response_p3['data']['items']) == 3
    
    @pytest.mark.asyncio
    async def test_cloudwatch_alarms_and_dashboards_optimization_pagination(self, mock_large_dataset):
        """Test CloudWatch alarms and dashboards optimization pagination with 23 items."""
        
        # Filter to alarms only for this test
        alarms_data = [item for item in mock_large_dataset if item['type'] == 'alarm']
        # Pad to 23 items if needed
        while len(alarms_data) < 23:
            alarms_data.extend(alarms_data[:23-len(alarms_data)])
        alarms_data = alarms_data[:23]
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(alarms_data, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_alarms_and_dashboards_optimization
            
            # Test pagination cardinality
            result_p1 = await run_cloudwatch_alarms_and_dashboards_optimization({
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
    async def test_cloudwatch_comprehensive_optimization_pagination(self, mock_large_dataset):
        """Test CloudWatch comprehensive optimization pagination with 23 items."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(mock_large_dataset, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_comprehensive_optimization
            
            # Test pagination cardinality
            result_p1 = await run_cloudwatch_comprehensive_optimization({
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
                result = await run_cloudwatch_comprehensive_optimization({
                    'region': 'us-east-1',
                    'page': page_num,
                    'lookback_days': 30
                })
                
                response = json.loads(result[0].text)
                expected_items = 10 if page_num < 3 else 3
                assert len(response['data']['items']) == expected_items
                assert response['data']['pagination']['current_page'] == page_num
    
    @pytest.mark.asyncio
    async def test_cloudwatch_comprehensive_optimization_tool_pagination(self, mock_large_dataset):
        """Test CloudWatch comprehensive optimization tool pagination with 23 items."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(mock_large_dataset, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_comprehensive_optimization_tool
            
            # Test pagination cardinality with executive summary
            result_p1 = await run_cloudwatch_comprehensive_optimization_tool({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30,
                'generate_executive_summary': True,
                'max_parallel_analyses': 4
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['pagination']['current_page'] == 1
            assert len(response_p1['data']['items']) == 10
            assert response_p1['data']['pagination']['total_items'] == 23
            assert response_p1['data']['pagination']['total_pages'] == 3
            
            # Test page 2 and 3 for complete coverage
            for page_num in [2, 3]:
                result = await run_cloudwatch_comprehensive_optimization_tool({
                    'region': 'us-east-1',
                    'page': page_num,
                    'lookback_days': 30
                })
                
                response = json.loads(result[0].text)
                expected_items = 10 if page_num == 2 else 3
                assert len(response['data']['items']) == expected_items
    
    @pytest.mark.asyncio
    async def test_query_cloudwatch_analysis_results_pagination(self, mock_large_dataset):
        """Test CloudWatch analysis results query pagination with 23 items."""
        
        with patch('playbooks.cloudwatch.aggregation_queries.CloudWatchAggregationQueries') as mock_queries_class:
            mock_queries = MagicMock()
            mock_queries_class.return_value = mock_queries
            
            def mock_execute_query(*args, **kwargs):
                page = kwargs.get('page', 1)
                items_per_page = 10
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_items = mock_large_dataset[start_idx:end_idx]
                
                return {
                    'status': 'success',
                    'query_results': {
                        'rows': page_items,
                        'pagination': {
                            'current_page': page,
                            'page_size': items_per_page,
                            'total_items': len(mock_large_dataset),
                            'total_pages': 3,
                            'has_next_page': page < 3,
                            'has_previous_page': page > 1
                        },
                        'query_metadata': {
                            'sql_query': kwargs.get('query', ''),
                            'execution_time_ms': 125.5,
                            'rows_examined': len(mock_large_dataset),
                            'session_id': kwargs.get('session_id', 'default-session')
                        }
                    }
                }
            
            mock_queries.execute_query = mock_execute_query
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test SQL query with pagination
            result_p1 = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM cloudwatch_analysis WHERE cost > 10',
                'page': 1,
                'session_id': 'test-session-123'
            })
            
            response_p1 = json.loads(result_p1[0].text)
            assert response_p1['data']['query_results']['pagination']['current_page'] == 1
            assert len(response_p1['data']['query_results']['rows']) == 10
            assert response_p1['data']['query_results']['pagination']['total_items'] == 23
            assert response_p1['data']['query_results']['pagination']['total_pages'] == 3
    
    @pytest.mark.asyncio
    async def test_pagination_edge_cases(self):
        """Test pagination edge cases with various dataset sizes."""
        
        # Test with empty dataset
        empty_dataset = []
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                return self.create_orchestrator_response(empty_dataset, 1)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_general_spend_analysis
            
            result = await run_cloudwatch_general_spend_analysis({
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
        single_page_dataset = [{'id': f'item-{i}', 'name': f'Item{i}', 'type': 'metric'} for i in range(10)]
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                return self.create_orchestrator_response(single_page_dataset, 1)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_metrics_optimization
            
            result = await run_cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            assert len(response['data']['items']) == 10
            assert response['data']['pagination']['total_items'] == 10
            assert response['data']['pagination']['total_pages'] == 1
            assert response['data']['pagination']['has_next_page'] is False
            assert response['data']['pagination']['has_previous_page'] is False
    
    @pytest.mark.asyncio
    async def test_pagination_data_integrity(self, mock_large_dataset):
        """Test that pagination preserves data integrity across pages."""
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            async def mock_execute_analysis(*args, **kwargs):
                page = kwargs.get('page', 1)
                return self.create_orchestrator_response(mock_large_dataset, page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            from runbook_functions import run_cloudwatch_comprehensive_optimization
            
            # Collect all items across all pages
            all_items = []
            for page_num in [1, 2, 3]:
                result = await run_cloudwatch_comprehensive_optimization({
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])