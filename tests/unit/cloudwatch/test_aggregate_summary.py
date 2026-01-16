"""
Test aggregate summary functionality for CloudWatch comprehensive optimization tool.

This test verifies that the tool returns a token-efficient aggregate summary by default
and only includes full details when explicitly requested.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime


class TestAggregateSummary:
    """Test aggregate summary functionality."""
    
    @pytest.mark.asyncio
    async def test_default_returns_summary(self):
        """Test that default behavior returns aggregate summary without full resource lists."""
        from playbooks.cloudwatch.cloudwatch_optimization_tool import CloudWatchOptimizationTool
        
        tool = CloudWatchOptimizationTool(region='us-east-1')
        
        # Mock the orchestrator to return sample data
        with patch.object(tool.orchestrator, 'validate_cost_preferences') as mock_validate, \
             patch.object(tool.orchestrator, 'get_cost_estimate') as mock_estimate, \
             patch.object(tool.orchestrator, 'execute_analysis') as mock_execute:
            
            # Setup mocks
            mock_validate.return_value = {
                'status': 'success',
                'validation_status': 'success',
                'validated_preferences': {},
                'functionality_coverage': {'overall_coverage': 60}
            }
            
            mock_estimate.return_value = {
                'status': 'success',
                'cost_estimate': {'total_estimated_cost': 0.0}
            }
            
            # Mock analysis results with many resources
            mock_execute.return_value = {
                'status': 'success',
                'data': {
                    'resources': [{'id': f'resource-{i}'} for i in range(100)],  # 100 resources
                    'optimization_opportunities': [{'id': f'opp-{i}'} for i in range(50)],
                    'total_potential_savings': 500.0,
                    'key_findings': ['Finding 1', 'Finding 2', 'Finding 3'],
                    'recommendations': [{'priority': 'high', 'title': 'Recommendation 1'}]
                }
            }
            
            # Execute without detail_level (should default to summary)
            # Limit to one functionality to simplify test
            result = await tool.execute_comprehensive_optimization_analysis(
                region='us-east-1',
                lookback_days=30,
                functionalities=['general_spend']
            )
            
            # Verify result structure
            assert result['status'] == 'success'
            assert result['detail_level'] == 'summary'
            
            # Verify aggregate summary exists
            assert 'aggregate_summary' in result
            aggregate = result['aggregate_summary']
            
            # Verify aggregate contains counts, not full resource lists
            assert 'overview' in aggregate
            assert aggregate['overview']['total_resources_analyzed'] == 100
            assert aggregate['overview']['total_optimization_opportunities'] == 50
            assert aggregate['overview']['total_potential_monthly_savings'] == 500.0
            
            # Verify detailed_results is NOT included by default
            assert 'detailed_results' not in result or result['detailed_results'] is None
            
            # Verify drill-down guidance is provided
            assert 'drill_down_guidance' in result
            assert 'options' in result['drill_down_guidance']
    
    @pytest.mark.asyncio
    async def test_full_detail_level_includes_resources(self):
        """Test that detail_level='full' includes complete resource details."""
        from playbooks.cloudwatch.cloudwatch_optimization_tool import CloudWatchOptimizationTool
        
        tool = CloudWatchOptimizationTool(region='us-east-1')
        
        # Mock the orchestrator
        with patch.object(tool.orchestrator, 'validate_cost_preferences') as mock_validate, \
             patch.object(tool.orchestrator, 'get_cost_estimate') as mock_estimate, \
             patch.object(tool.orchestrator, 'execute_analysis') as mock_execute:
            
            # Setup mocks
            mock_validate.return_value = {
                'status': 'success',
                'validation_status': 'success',
                'validated_preferences': {},
                'functionality_coverage': {'overall_coverage': 60}
            }
            
            mock_estimate.return_value = {
                'status': 'success',
                'cost_estimate': {'total_estimated_cost': 0.0}
            }
            
            mock_execute.return_value = {
                'status': 'success',
                'data': {
                    'resources': [{'id': f'resource-{i}'} for i in range(100)],
                    'optimization_opportunities': [{'id': f'opp-{i}'} for i in range(50)],
                    'total_potential_savings': 500.0
                }
            }
            
            # Execute with detail_level='full'
            result = await tool.execute_comprehensive_optimization_analysis(
                region='us-east-1',
                lookback_days=30,
                detail_level='full'
            )
            
            # Verify result structure
            assert result['status'] == 'success'
            assert result['detail_level'] == 'full'
            
            # Verify detailed_results IS included
            assert 'detailed_results' in result
            assert result['detailed_results'] is not None
            
            # Verify analysis_configuration is included
            assert 'analysis_configuration' in result
            
            # Verify implementation_support is included
            assert 'implementation_support' in result
    
    @pytest.mark.asyncio
    async def test_aggregate_summary_structure(self):
        """Test that aggregate summary has correct structure."""
        from playbooks.cloudwatch.cloudwatch_optimization_tool import CloudWatchOptimizationTool
        
        tool = CloudWatchOptimizationTool(region='us-east-1')
        
        # Create sample parallel results
        parallel_results = {
            'status': 'success',
            'successful_analyses': 2,
            'failed_analyses': 0,
            'individual_results': {
                'logs_optimization': {
                    'status': 'success',
                    'data': {
                        'resources': [{'id': 'log-1'}, {'id': 'log-2'}],
                        'optimization_opportunities': [{'id': 'opp-1'}],
                        'total_potential_savings': 100.0,
                        'key_findings': ['Finding 1', 'Finding 2'],
                        'recommendations': [{'priority': 'high', 'title': 'Rec 1'}]
                    }
                },
                'metrics_optimization': {
                    'status': 'success',
                    'data': {
                        'resources': [{'id': 'metric-1'}],
                        'optimization_opportunities': [{'id': 'opp-2'}],
                        'total_potential_savings': 50.0,
                        'key_findings': ['Finding 3'],
                        'recommendations': [{'priority': 'medium', 'title': 'Rec 2'}]
                    }
                }
            }
        }
        
        # Create aggregate summary
        aggregate = tool._create_aggregate_summary(
            parallel_results=parallel_results,
            cross_analysis_insights={'status': 'success', 'insights': {}},
            executive_summary=None,
            execution_plan={'valid_functionalities': ['logs_optimization', 'metrics_optimization']},
            top_recommendations=[],
            successful_analyses=2,
            failed_analyses=0
        )
        
        # Verify structure
        assert 'overview' in aggregate
        assert aggregate['overview']['total_resources_analyzed'] == 3
        assert aggregate['overview']['total_optimization_opportunities'] == 2
        assert aggregate['overview']['total_potential_monthly_savings'] == 150.0
        
        assert 'functionality_summaries' in aggregate
        assert 'logs_optimization' in aggregate['functionality_summaries']
        assert 'metrics_optimization' in aggregate['functionality_summaries']
        
        # Verify functionality summaries contain counts, not full lists
        logs_summary = aggregate['functionality_summaries']['logs_optimization']
        assert logs_summary['resources_analyzed'] == 2
        assert logs_summary['optimization_opportunities'] == 1
        assert logs_summary['potential_monthly_savings'] == 100.0
        assert len(logs_summary['key_findings']) <= 3  # Limited to top 3
