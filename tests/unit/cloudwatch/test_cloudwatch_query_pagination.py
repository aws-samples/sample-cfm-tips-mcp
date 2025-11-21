#!/usr/bin/env python3
"""
Unit tests for CloudWatch query functionality with pagination.
Tests SQL query results with large datasets and pagination.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure")
class TestCloudWatchQueryPagination:
    """Test CloudWatch query functionality with pagination."""
    
    @pytest.fixture
    def mock_sql_query_results(self):
        """Create mock SQL query results with 23 rows."""
        return [
            {
                'id': i,
                'resource_name': f'cloudwatch-resource-{i:03d}',
                'resource_type': 'metric' if i % 3 == 0 else 'log_group' if i % 3 == 1 else 'alarm',
                'region': 'us-east-1' if i % 2 == 0 else 'us-west-2',
                'monthly_cost': round(5.25 + (i * 1.75), 2),
                'optimization_potential': round(2.10 + (i * 0.85), 2),
                'last_analyzed': (datetime.now() - timedelta(hours=i)).isoformat(),
                'namespace': f'AWS/Service{i % 4}',
                'dimensions_count': (i % 5) + 1,
                'state': 'active' if i % 4 != 3 else 'inactive'
            }
            for i in range(23)
        ]
    
    @pytest.mark.asyncio
    async def test_basic_sql_query_pagination(self, mock_sql_query_results):
        """Test basic SQL query with pagination."""
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_query:
            def mock_response(*args, **kwargs):
                page = kwargs.get('page', 1)
                items_per_page = 10
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = mock_sql_query_results[start_idx:end_idx]
                
                return {
                    'status': 'success',
                    'query_results': {
                        'rows': page_results,
                        'pagination': {
                            'current_page': page,
                            'total_rows': len(mock_sql_query_results),
                            'total_pages': 3,
                            'rows_per_page': items_per_page,
                            'has_next_page': page < 3,
                            'has_previous_page': page > 1
                        },
                        'query_metadata': {
                            'sql_query': kwargs.get('query', ''),
                            'execution_time_ms': 125.5,
                            'rows_examined': len(mock_sql_query_results),
                            'session_id': kwargs.get('session_id', 'default-session')
                        }
                    }
                }
            
            mock_query.side_effect = mock_response
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test SELECT * query with pagination
            result_p1 = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM cloudwatch_analysis ORDER BY monthly_cost DESC',
                'page': 1,
                'session_id': 'test-session-123'
            })
            
            response_p1 = json.loads(result_p1[0].text)
            
            # Verify query results structure
            assert response_p1['query_results']['pagination']['current_page'] == 1
            assert len(response_p1['query_results']['rows']) == 10
            assert response_p1['query_results']['pagination']['total_rows'] == 23
            assert response_p1['query_results']['pagination']['total_pages'] == 3
            
            # Verify query metadata
            assert response_p1['query_results']['query_metadata']['sql_query'] == 'SELECT * FROM cloudwatch_analysis ORDER BY monthly_cost DESC'
            assert response_p1['query_results']['query_metadata']['session_id'] == 'test-session-123'
            assert response_p1['query_results']['query_metadata']['execution_time_ms'] > 0
            
            # Test page 3 (remainder)
            result_p3 = await query_cloudwatch_analysis_results({
                'query': 'SELECT * FROM cloudwatch_analysis ORDER BY monthly_cost DESC',
                'page': 3
            })
            
            response_p3 = json.loads(result_p3[0].text)
            assert len(response_p3['query_results']['rows']) == 3
            assert response_p3['query_results']['pagination']['current_page'] == 3
            assert response_p3['query_results']['pagination']['has_next_page'] is False
    
    @pytest.mark.asyncio
    async def test_filtered_sql_query_pagination(self, mock_sql_query_results):
        """Test filtered SQL queries with pagination."""
        
        # Filter results to only metrics (should be ~8 items)
        filtered_results = [r for r in mock_sql_query_results if r['resource_type'] == 'metric']
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_query:
            def mock_response(*args, **kwargs):
                page = kwargs.get('page', 1)
                items_per_page = 10
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = filtered_results[start_idx:end_idx]
                
                return {
                    'status': 'success',
                    'query_results': {
                        'rows': page_results,
                        'pagination': {
                            'current_page': page,
                            'total_rows': len(filtered_results),
                            'total_pages': 1,  # All filtered results fit in one page
                            'rows_per_page': items_per_page,
                            'has_next_page': False,
                            'has_previous_page': False
                        },
                        'query_metadata': {
                            'sql_query': kwargs.get('query', ''),
                            'execution_time_ms': 89.2,
                            'rows_examined': len(mock_sql_query_results),
                            'rows_returned': len(filtered_results)
                        }
                    }
                }
            
            mock_query.side_effect = mock_response
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test filtered query
            result = await query_cloudwatch_analysis_results({
                'query': "SELECT * FROM cloudwatch_analysis WHERE resource_type = 'metric'",
                'page': 1
            })
            
            response = json.loads(result[0].text)
            
            # Verify filtered results
            assert len(response['query_results']['rows']) == len(filtered_results)
            assert response['query_results']['pagination']['total_pages'] == 1
            assert response['query_results']['pagination']['has_next_page'] is False
            
            # Verify all returned rows are metrics
            for row in response['query_results']['rows']:
                assert row['resource_type'] == 'metric'
            
            # Verify query metadata shows filtering
            assert response['query_results']['query_metadata']['rows_examined'] == 23
            assert response['query_results']['query_metadata']['rows_returned'] == len(filtered_results)
    
    @pytest.mark.asyncio
    async def test_aggregation_sql_query_pagination(self, mock_sql_query_results):
        """Test aggregation SQL queries with pagination."""
        
        # Create mock aggregation results
        aggregation_results = [
            {
                'resource_type': 'metric',
                'count': 8,
                'total_cost': 156.75,
                'avg_cost': 19.59,
                'max_cost': 45.25,
                'min_cost': 5.25
            },
            {
                'resource_type': 'log_group', 
                'count': 8,
                'total_cost': 142.50,
                'avg_cost': 17.81,
                'max_cost': 42.50,
                'min_cost': 7.00
            },
            {
                'resource_type': 'alarm',
                'count': 7,
                'total_cost': 98.25,
                'avg_cost': 14.04,
                'max_cost': 38.75,
                'min_cost': 8.75
            }
        ]
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_query:
            def mock_response(*args, **kwargs):
                return {
                    'status': 'success',
                    'query_results': {
                        'rows': aggregation_results,
                        'pagination': {
                            'current_page': 1,
                            'total_rows': len(aggregation_results),
                            'total_pages': 1,
                            'rows_per_page': 10,
                            'has_next_page': False,
                            'has_previous_page': False
                        },
                        'query_metadata': {
                            'sql_query': kwargs.get('query', ''),
                            'execution_time_ms': 45.8,
                            'rows_examined': 23,
                            'rows_returned': len(aggregation_results),
                            'query_type': 'aggregation'
                        }
                    }
                }
            
            mock_query.side_effect = mock_response
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test aggregation query
            result = await query_cloudwatch_analysis_results({
                'query': '''
                    SELECT 
                        resource_type,
                        COUNT(*) as count,
                        SUM(monthly_cost) as total_cost,
                        AVG(monthly_cost) as avg_cost,
                        MAX(monthly_cost) as max_cost,
                        MIN(monthly_cost) as min_cost
                    FROM cloudwatch_analysis 
                    GROUP BY resource_type
                    ORDER BY total_cost DESC
                ''',
                'page': 1
            })
            
            response = json.loads(result[0].text)
            
            # Verify aggregation results
            assert len(response['query_results']['rows']) == 3
            assert response['query_results']['pagination']['total_pages'] == 1
            
            # Verify aggregation data structure
            for row in response['query_results']['rows']:
                assert 'resource_type' in row
                assert 'count' in row
                assert 'total_cost' in row
                assert 'avg_cost' in row
                assert row['count'] > 0
                assert row['total_cost'] > 0
            
            # Verify query metadata indicates aggregation
            assert response['query_results']['query_metadata']['query_type'] == 'aggregation'
    
    @pytest.mark.asyncio
    async def test_complex_join_query_pagination(self, mock_sql_query_results):
        """Test complex JOIN queries with pagination."""
        
        # Create mock join results (simulating joins between multiple tables)
        join_results = [
            {
                'resource_id': f'res-{i:03d}',
                'resource_name': f'cloudwatch-resource-{i:03d}',
                'resource_type': mock_sql_query_results[i]['resource_type'],
                'monthly_cost': mock_sql_query_results[i]['monthly_cost'],
                'optimization_score': round(85.5 - (i * 2.1), 1),
                'recommendation': f'Optimize {mock_sql_query_results[i]["resource_type"]} configuration',
                'priority': 'high' if i < 8 else 'medium' if i < 16 else 'low',
                'estimated_savings': round(mock_sql_query_results[i]['monthly_cost'] * 0.25, 2)
            }
            for i in range(23)
        ]
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_query:
            def mock_response(*args, **kwargs):
                page = kwargs.get('page', 1)
                items_per_page = 10
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = join_results[start_idx:end_idx]
                
                return {
                    'status': 'success',
                    'query_results': {
                        'rows': page_results,
                        'pagination': {
                            'current_page': page,
                            'total_rows': len(join_results),
                            'total_pages': 3,
                            'rows_per_page': items_per_page,
                            'has_next_page': page < 3,
                            'has_previous_page': page > 1
                        },
                        'query_metadata': {
                            'sql_query': kwargs.get('query', ''),
                            'execution_time_ms': 234.7,
                            'rows_examined': 46,  # Simulating join across 2 tables
                            'rows_returned': len(join_results),
                            'query_type': 'join',
                            'tables_joined': ['cloudwatch_analysis', 'optimization_recommendations']
                        }
                    }
                }
            
            mock_query.side_effect = mock_response
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test complex JOIN query
            result_p1 = await query_cloudwatch_analysis_results({
                'query': '''
                    SELECT 
                        ca.resource_id,
                        ca.resource_name,
                        ca.resource_type,
                        ca.monthly_cost,
                        or.optimization_score,
                        or.recommendation,
                        or.priority,
                        or.estimated_savings
                    FROM cloudwatch_analysis ca
                    JOIN optimization_recommendations or ON ca.resource_id = or.resource_id
                    WHERE or.optimization_score > 50
                    ORDER BY or.optimization_score DESC, ca.monthly_cost DESC
                ''',
                'page': 1
            })
            
            response_p1 = json.loads(result_p1[0].text)
            
            # Verify JOIN query results
            assert len(response_p1['query_results']['rows']) == 10
            assert response_p1['query_results']['pagination']['total_rows'] == 23
            assert response_p1['query_results']['pagination']['total_pages'] == 3
            
            # Verify JOIN result structure
            for row in response_p1['query_results']['rows']:
                assert 'resource_id' in row
                assert 'optimization_score' in row
                assert 'recommendation' in row
                assert 'priority' in row
                assert 'estimated_savings' in row
                assert row['optimization_score'] > 0
            
            # Verify query metadata indicates JOIN
            assert response_p1['query_results']['query_metadata']['query_type'] == 'join'
            assert 'tables_joined' in response_p1['query_results']['query_metadata']
            assert len(response_p1['query_results']['query_metadata']['tables_joined']) == 2
            
            # Test pagination across all pages
            all_results = []
            for page_num in [1, 2, 3]:
                result = await query_cloudwatch_analysis_results({
                    'query': 'SELECT * FROM joined_results',
                    'page': page_num
                })
                
                response = json.loads(result[0].text)
                all_results.extend(response['query_results']['rows'])
            
            # Verify complete dataset
            assert len(all_results) == 23
            
            # Verify no duplicates
            resource_ids = [row['resource_id'] for row in all_results]
            assert len(resource_ids) == len(set(resource_ids))
    
    @pytest.mark.asyncio
    async def test_query_error_handling_with_pagination(self):
        """Test error handling in SQL queries with pagination context."""
        
        with patch('runbook_functions.mcp_cfm_tips_query_cloudwatch_analysis_results') as mock_query:
            def mock_error_response(*args, **kwargs):
                return {
                    'status': 'error',
                    'error_message': 'SQL syntax error: invalid column name "invalid_column"',
                    'error_code': 'SQL_SYNTAX_ERROR',
                    'query_context': {
                        'sql_query': kwargs.get('query', ''),
                        'page_requested': kwargs.get('page', 1),
                        'session_id': kwargs.get('session_id', 'default')
                    },
                    'suggestions': [
                        'Check column names in your SELECT statement',
                        'Verify table schema with PRAGMA table_info(table_name)',
                        'Use valid column names from the analysis results'
                    ]
                }
            
            mock_query.side_effect = mock_error_response
            
            from runbook_functions import query_cloudwatch_analysis_results
            
            # Test invalid SQL query
            result = await query_cloudwatch_analysis_results({
                'query': 'SELECT invalid_column FROM cloudwatch_analysis',
                'page': 1,
                'session_id': 'error-test-session'
            })
            
            response = json.loads(result[0].text)
            
            # Verify error response structure
            assert response['status'] == 'error'
            assert 'error_message' in response
            assert 'error_code' in response
            assert 'query_context' in response
            assert 'suggestions' in response
            
            # Verify query context includes pagination info
            assert response['query_context']['page_requested'] == 1
            assert response['query_context']['session_id'] == 'error-test-session'
            assert response['query_context']['sql_query'] == 'SELECT invalid_column FROM cloudwatch_analysis'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])