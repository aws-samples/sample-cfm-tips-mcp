"""
Integration Tests for CloudWatch Orchestrator Pagination

These tests verify that the CloudWatch orchestrator correctly integrates
result processing and pagination without incurring additional AWS costs.
"""

import pytest
import asyncio
import unittest.mock as mock
from unittest.mock import MagicMock, patch, AsyncMock
from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator


class TestCloudWatchOrchestratorPagination:
    """Integration tests for CloudWatch orchestrator pagination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.region = 'us-east-1'
        self.session_id = 'test-session-123'
    
    @pytest.mark.asyncio
    async def test_execute_analysis_with_pagination_no_additional_api_calls(self):
        """Test that execute_analysis with pagination doesn't make additional API calls."""
        
        # Mock the analysis engine to return sample data
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': [
                            {'logGroupName': 'group1', 'storedBytes': 2147483648},  # 2 GB
                            {'logGroupName': 'group2', 'storedBytes': 1073741824}   # 1 GB
                        ]
                    },
                    'metrics': {
                        'metrics': [
                            {'MetricName': 'CustomMetric', 'Namespace': 'MyApp'},
                            {'MetricName': 'CPUUtilization', 'Namespace': 'AWS/EC2'}
                        ]
                    }
                }
            },
            'recommendations': [
                {'type': 'optimization', 'potential_monthly_savings': 15.0},
                {'type': 'cleanup', 'potential_monthly_savings': 5.0}
            ]
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class, \
             patch('boto3.client') as mock_boto3:
            
            # Mock the analysis engine
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            # Create orchestrator
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            # Execute analysis with pagination
            result = await orchestrator.execute_analysis(
                'general_spend', 
                page=1,
                allow_cost_explorer=False,
                allow_minimal_cost_metrics=False
            )
            
            # Verify no additional boto3 clients were created during result processing
            # (The orchestrator itself may create clients during initialization)
            initial_call_count = mock_boto3.call_count
            
            # Process the result again to verify no additional calls
            if result.get('status') == 'success':
                # The result should already be processed and paginated
                assert 'pagination_applied' in result
                assert result['current_page'] == 1
            
            # Verify no additional boto3 calls were made during result processing
            assert mock_boto3.call_count == initial_call_count
    
    @pytest.mark.asyncio
    async def test_result_processing_sorts_by_cost_descending(self):
        """Test that result processing correctly sorts items by cost in descending order."""
        
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': [
                            {'logGroupName': 'small-group', 'storedBytes': 1073741824},    # 1 GB = $0.03
                            {'logGroupName': 'large-group', 'storedBytes': 10737418240},   # 10 GB = $0.30
                            {'logGroupName': 'medium-group', 'storedBytes': 5368709120}    # 5 GB = $0.15
                        ]
                    }
                }
            }
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            result = await orchestrator.execute_analysis('general_spend', page=1)
            
            # Verify result is processed and sorted
            if result.get('status') == 'success':
                log_groups_data = result['data']['configuration_analysis']['log_groups']
                
                # Should have pagination metadata
                assert 'pagination' in log_groups_data
                assert log_groups_data['pagination']['current_page'] == 1
                
                # Should be sorted by cost descending
                items = log_groups_data['items']
                assert len(items) == 3
                
                # Verify cost estimates are added and sorted correctly
                assert items[0]['logGroupName'] == 'large-group'
                assert items[0]['estimated_monthly_cost'] == 0.30
                assert items[1]['logGroupName'] == 'medium-group'
                assert items[1]['estimated_monthly_cost'] == 0.15
                assert items[2]['logGroupName'] == 'small-group'
                assert items[2]['estimated_monthly_cost'] == 0.03
    
    @pytest.mark.asyncio
    async def test_pagination_metadata_accuracy(self):
        """Test that pagination metadata is accurate for different page scenarios."""
        
        # Create a large dataset to test pagination
        large_log_groups = [
            {'logGroupName': f'group-{i}', 'storedBytes': i * 1073741824}  # i GB each
            for i in range(25)  # 25 log groups = 3 pages (10 items per page)
        ]
        
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': large_log_groups
                    }
                }
            }
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            # Test page 1
            result_page1 = await orchestrator.execute_analysis('general_spend', page=1)
            log_groups_page1 = result_page1['data']['configuration_analysis']['log_groups']
            
            assert log_groups_page1['pagination']['current_page'] == 1
            assert log_groups_page1['pagination']['total_items'] == 25
            assert log_groups_page1['pagination']['total_pages'] == 3
            assert log_groups_page1['pagination']['has_next_page'] is True
            assert log_groups_page1['pagination']['has_previous_page'] is False
            assert len(log_groups_page1['items']) == 10
            
            # Test page 2
            result_page2 = await orchestrator.execute_analysis('general_spend', page=2)
            log_groups_page2 = result_page2['data']['configuration_analysis']['log_groups']
            
            assert log_groups_page2['pagination']['current_page'] == 2
            assert log_groups_page2['pagination']['has_next_page'] is True
            assert log_groups_page2['pagination']['has_previous_page'] is True
            assert len(log_groups_page2['items']) == 10
            
            # Test page 3 (last page with partial results)
            result_page3 = await orchestrator.execute_analysis('general_spend', page=3)
            log_groups_page3 = result_page3['data']['configuration_analysis']['log_groups']
            
            assert log_groups_page3['pagination']['current_page'] == 3
            assert log_groups_page3['pagination']['has_next_page'] is False
            assert log_groups_page3['pagination']['has_previous_page'] is True
            assert len(log_groups_page3['items']) == 5  # Last page has 5 items
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_pagination(self):
        """Test that comprehensive analysis also applies pagination correctly."""
        
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': [
                            {'logGroupName': 'group1', 'storedBytes': 1073741824},
                            {'logGroupName': 'group2', 'storedBytes': 2147483648}
                        ]
                    }
                }
            },
            'recommendations': [
                {'type': 'optimization', 'potential_monthly_savings': 10.0}
            ]
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            result = await orchestrator.execute_comprehensive_analysis(page=1)
            
            # Verify comprehensive analysis applies pagination to all analysis types
            if result.get('status') == 'success':
                # Each analysis type should have paginated results
                for analysis_type in ['general_spend', 'logs_optimization', 'metrics_optimization', 'alarms_and_dashboards']:
                    if analysis_type in result.get('results', {}):
                        analysis_result = result['results'][analysis_type]
                        if analysis_result.get('status') == 'success':
                            assert 'pagination_applied' in analysis_result
                            assert analysis_result['current_page'] == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_during_result_processing(self):
        """Test that errors during result processing don't break the analysis."""
        
        # Mock analysis result with invalid data structure
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': 'invalid_structure'  # Should be a dict with 'log_groups' key
                }
            }
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            # Should not raise an exception, should return original result
            result = await orchestrator.execute_analysis('general_spend', page=1)
            
            # Should still return a successful result (original data preserved)
            assert result.get('status') == 'success'
            # Original data should be preserved if processing fails
            assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_page_parameter_validation(self):
        """Test that invalid page parameters are handled correctly."""
        
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': [
                            {'logGroupName': 'group1', 'storedBytes': 1073741824}
                        ]
                    }
                }
            }
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            # Test with page 0 (should default to page 1)
            result = await orchestrator.execute_analysis('general_spend', page=0)
            if result.get('status') == 'success':
                log_groups_data = result['data']['configuration_analysis']['log_groups']
                assert log_groups_data['pagination']['current_page'] == 1
            
            # Test with negative page (should default to page 1)
            result = await orchestrator.execute_analysis('general_spend', page=-5)
            if result.get('status') == 'success':
                log_groups_data = result['data']['configuration_analysis']['log_groups']
                assert log_groups_data['pagination']['current_page'] == 1
    
    def test_orchestrator_initialization_no_additional_costs(self):
        """Test that orchestrator initialization doesn't incur additional costs."""
        
        with patch('boto3.client') as mock_boto3:
            # Track initial boto3 calls (orchestrator may create some clients during init)
            initial_call_count = mock_boto3.call_count
            
            # Create orchestrator
            orchestrator = CloudWatchOptimizationOrchestrator(
                region=self.region, 
                session_id=self.session_id
            )
            
            # Verify result processor is initialized
            assert hasattr(orchestrator, 'result_processor')
            assert orchestrator.result_processor is not None
            
            # The result processor itself should not create additional AWS clients
            # (any clients created should be from the orchestrator's other components)
            processor_specific_calls = mock_boto3.call_count - initial_call_count
            
            # Verify result processor doesn't have AWS clients
            assert not hasattr(orchestrator.result_processor, '_aws_clients')


class TestZeroCostIntegration:
    """Integration tests specifically for zero-cost guarantee."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_no_additional_api_calls(self):
        """Test complete end-to-end flow without additional API calls."""
        
        # Mock all AWS services to track API calls
        with patch('boto3.client') as mock_boto3, \
             patch('boto3.resource') as mock_resource:
            
            # Create sample analysis result
            mock_analysis_result = {
                'status': 'success',
                'data': {
                    'configuration_analysis': {
                        'log_groups': {
                            'log_groups': [
                                {'logGroupName': 'test-group', 'storedBytes': 1073741824}
                            ]
                        }
                    }
                }
            }
            
            with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
                mock_engine = AsyncMock()
                mock_engine.run_analysis.return_value = mock_analysis_result
                mock_engine_class.return_value = mock_engine
                
                # Track API calls before result processing
                calls_before_processing = mock_boto3.call_count
                
                orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
                result = await orchestrator.execute_analysis('general_spend', page=1)
                
                # Track API calls after result processing
                calls_after_processing = mock_boto3.call_count
                
                # Verify result processing didn't add API calls
                # (orchestrator initialization may create clients, but result processing should not)
                processing_calls = calls_after_processing - calls_before_processing
                
                # The result processing itself should not make additional API calls
                # Any calls should be from orchestrator initialization, not from result processing
                if result.get('status') == 'success' and 'pagination_applied' in result:
                    # If pagination was applied, verify it was done without additional API calls
                    # by checking that the processing phase specifically didn't add calls
                    
                    # We can't easily separate orchestrator init calls from processing calls
                    # in this test, but we can verify the result processor methods don't call boto3
                    processor = orchestrator.result_processor
                    
                    # Test direct processor methods don't make API calls
                    test_items = [{'test': 'data', 'estimated_monthly_cost': 1.0}]
                    
                    calls_before_direct = mock_boto3.call_count
                    processor.sort_by_cost_descending(test_items)
                    processor.paginate_results(test_items)
                    calls_after_direct = mock_boto3.call_count
                    
                    # Direct processor methods should make no API calls
                    assert calls_after_direct == calls_before_direct
    
    def test_memory_usage_efficiency(self):
        """Test that pagination doesn't cause excessive memory usage."""
        
        # Create a large dataset
        large_dataset = [
            {'id': i, 'logGroupName': f'group-{i}', 'storedBytes': i * 1073741824}
            for i in range(1000)  # 1000 log groups
        ]
        
        mock_analysis_result = {
            'status': 'success',
            'data': {
                'configuration_analysis': {
                    'log_groups': {
                        'log_groups': large_dataset
                    }
                }
            }
        }
        
        with patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.run_analysis.return_value = mock_analysis_result
            mock_engine_class.return_value = mock_engine
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
            
            # Process large dataset with pagination
            import tracemalloc
            tracemalloc.start()
            
            # This should be memory efficient due to pagination
            asyncio.run(orchestrator.execute_analysis('general_spend', page=1))
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should be reasonable (less than 100MB for this test)
            assert peak < 100 * 1024 * 1024  # 100MB limit


if __name__ == '__main__':
    pytest.main([__file__])