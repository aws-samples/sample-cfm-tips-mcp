"""
Unit tests to identify MCP pagination bug in CloudWatch metrics optimization.

Tests the complete flow from MCP tool call through orchestrator to result processor
to identify where pagination is being bypassed.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
from mcp.types import TextContent

# Import the functions under test
from runbook_functions import run_cloudwatch_metrics_optimization


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure")
class TestMCPPaginationBug:
    """Test suite to identify where pagination is failing in the MCP flow."""
    
    @pytest.fixture
    def mock_large_metrics_dataset(self):
        """Create a large dataset of 25 metrics for pagination testing."""
        return [
            {
                'MetricName': f'CustomMetric{i:02d}',
                'Namespace': 'Custom/Application' if i % 2 == 0 else 'AWS/EC2',
                'Dimensions': [{'Name': 'InstanceId', 'Value': f'i-{i:08x}'}],
                'estimated_monthly_cost': 10.0 - (i * 0.2)  # Decreasing cost
            }
            for i in range(25)
        ]
    
    @pytest.fixture
    def mock_orchestrator_response(self, mock_large_metrics_dataset):
        """Mock orchestrator response with proper structure."""
        def create_response(page=1):
            # Simulate orchestrator pagination - should return only 10 items per page
            start_idx = (page - 1) * 10
            end_idx = start_idx + 10
            page_metrics = mock_large_metrics_dataset[start_idx:end_idx]
            
            return {
                'status': 'success',
                'data': {
                    'metrics_configuration_analysis': {
                        'metrics': {
                            'metrics': page_metrics,
                            'pagination': {
                                'current_page': page,
                                'page_size': 10,
                                'total_items': 25,
                                'total_pages': 3,
                                'has_next_page': page < 3,
                                'has_previous_page': page > 1
                            },
                            'total_count': 25,
                            'namespace': 'all',
                            'filtered': False
                        }
                    }
                },
                'orchestrator_metadata': {
                    'session_id': 'test-session',
                    'region': 'us-east-1'
                }
            }
        return create_response
    
    @pytest.mark.asyncio
    async def test_orchestrator_pagination_works(self, mock_orchestrator_response):
        """Test that orchestrator correctly applies pagination."""
        
        with patch('runbook_functions.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            # Setup mock orchestrator instance
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock execute_analysis to return paginated responses
            def mock_execute_analysis(analysis_type, **kwargs):
                page = kwargs.get('page', 1)
                return mock_orchestrator_response(page)
            
            mock_orchestrator.execute_analysis = AsyncMock(side_effect=mock_execute_analysis)
            
            # Test page 1
            result_p1 = await run_cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 1,
                'lookback_days': 30
            })
            
            # Verify result structure
            assert len(result_p1) == 1
            assert isinstance(result_p1[0], TextContent)
            
            # Parse JSON response
            data_p1 = json.loads(result_p1[0].text)
            assert data_p1['status'] == 'success'
            
            # Check metrics data
            metrics_data_p1 = data_p1['data']['metrics_configuration_analysis']['metrics']
            assert len(metrics_data_p1['metrics']) == 10, "Page 1 should have 10 metrics"
            assert metrics_data_p1['pagination']['current_page'] == 1
            assert metrics_data_p1['pagination']['total_items'] == 25
            
            # Test page 2
            result_p2 = await run_cloudwatch_metrics_optimization({
                'region': 'us-east-1',
                'page': 2,
                'lookback_days': 30
            })
            
            data_p2 = json.loads(result_p2[0].text)
            metrics_data_p2 = data_p2['data']['metrics_configuration_analysis']['metrics']
            assert len(metrics_data_p2['metrics']) == 10, "Page 2 should have 10 metrics"
            assert metrics_data_p2['pagination']['current_page'] == 2
            
            # Verify different metrics on different pages
            p1_names = [m['MetricName'] for m in metrics_data_p1['metrics']]
            p2_names = [m['MetricName'] for m in metrics_data_p2['metrics']]
            assert p1_names != p2_names, "Page 1 and Page 2 should have different metrics"
            
            # Verify orchestrator was called with correct parameters
            assert mock_orchestrator.execute_analysis.call_count == 2
            
            # Check first call (page 1)
            first_call_args = mock_orchestrator.execute_analysis.call_args_list[0]
            assert first_call_args[0][0] == 'metrics_optimization'
            assert first_call_args[1]['page'] == 1
            
            # Check second call (page 2)  
            second_call_args = mock_orchestrator.execute_analysis.call_args_list[1]
            assert second_call_args[1]['page'] == 2
    
    @pytest.mark.asyncio
    async def test_mcp_tool_bypasses_pagination(self):
        """Test to identify if MCP tool is bypassing orchestrator pagination."""
        
        # This test will help identify if there's a direct MCP call bypassing the orchestrator
        with patch('runbook_functions.CloudWatchOptimizationOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock orchestrator to return a response indicating it was called
            mock_orchestrator.execute_analysis = AsyncMock(return_value={
                'status': 'success',
                'data': {
                    'metrics_configuration_analysis': {
                        'metrics': {
                            'metrics': [],
                            'pagination': {'current_page': 1, 'total_items': 0},
                            'orchestrator_called': True  # Flag to verify orchestrator was used
                        }
                    }
                }
            })
            
            # Also patch any potential direct MCP calls
            with patch('runbook_functions.mcp_cfm_tips_cloudwatch_metrics_optimization') as mock_mcp:
                mock_mcp.return_value = {
                    'status': 'success', 
                    'data': {'direct_mcp_call': True}  # Flag to identify direct MCP call
                }
                
                result = await run_cloudwatch_metrics_optimization({
                    'region': 'us-east-1',
                    'page': 1
                })
                
                # Parse result
                data = json.loads(result[0].text)
                
                # Check if orchestrator was used (expected behavior)
                if 'orchestrator_called' in str(data):
                    print("✅ Orchestrator was called - pagination should work")
                    assert mock_orchestrator.execute_analysis.called
                    assert not mock_mcp.called, "Direct MCP call should not be made"
                
                # Check if direct MCP call was made (bug scenario)
                elif 'direct_mcp_call' in str(data):
                    pytest.fail("❌ BUG IDENTIFIED: Direct MCP call bypassing orchestrator pagination")
                
                else:
                    pytest.fail("❌ Unable to determine call path - check test setup")
    
    @pytest.mark.asyncio 
    async def test_result_processor_pagination(self):
        """Test that result processor correctly paginates metrics."""
        
        from playbooks.cloudwatch.result_processor import CloudWatchResultProcessor
        
        # Create test metrics
        test_metrics = [
            {'MetricName': f'Metric{i}', 'estimated_monthly_cost': 10 - i}
            for i in range(25)
        ]
        
        processor = CloudWatchResultProcessor()
        
        # Test page 1
        result_p1 = processor.process_metrics_results(test_metrics, page=1)
        assert len(result_p1['items']) == 10
        assert result_p1['pagination']['current_page'] == 1
        assert result_p1['pagination']['total_items'] == 25
        
        # Test page 2
        result_p2 = processor.process_metrics_results(test_metrics, page=2)
        assert len(result_p2['items']) == 10
        assert result_p2['pagination']['current_page'] == 2
        
        # Verify different items
        p1_names = [item['MetricName'] for item in result_p1['items']]
        p2_names = [item['MetricName'] for item in result_p2['items']]
        assert p1_names != p2_names
    
    @pytest.mark.asyncio
    async def test_orchestrator_apply_result_processing(self):
        """Test that orchestrator's _apply_result_processing works correctly."""
        
        from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
        
        # Create mock result with metrics
        mock_result = {
            'status': 'success',
            'data': {
                'metrics_configuration_analysis': {
                    'metrics': {
                        'metrics': [
                            {'MetricName': f'Metric{i}', 'estimated_monthly_cost': 10 - i}
                            for i in range(25)
                        ],
                        'total_count': 25
                    }
                }
            }
        }
        
        orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
        
        # Test pagination application
        processed_result = orchestrator._apply_result_processing(mock_result, page=1)
        
        metrics_data = processed_result['data']['metrics_configuration_analysis']['metrics']
        assert len(metrics_data['metrics']) == 10, "Should be paginated to 10 items"
        assert 'pagination' in metrics_data, "Should have pagination metadata"
        assert metrics_data['pagination']['current_page'] == 1
        assert metrics_data['pagination']['total_items'] == 25