"""
Unit tests for CloudWatch GeneralSpendAnalyzer.

Tests the comprehensive CloudWatch spend analysis functionality with cost control flags.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from playbooks.cloudwatch.general_spend_analyzer import GeneralSpendAnalyzer
from services.cloudwatch_service import CloudWatchOperationResult


@pytest.mark.unit
class TestCloudWatchGeneralSpendAnalyzer:
    """Test cases for CloudWatch GeneralSpendAnalyzer."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            'cost_explorer_service': Mock(),
            'config_service': Mock(),
            'metrics_service': Mock(),
            'cloudwatch_service': Mock(),
            'pricing_service': Mock(),
            'performance_monitor': Mock(),
            'memory_manager': Mock()
        }
    
    @pytest.fixture
    def analyzer(self, mock_services):
        """Create GeneralSpendAnalyzer instance with mocked services."""
        return GeneralSpendAnalyzer(**mock_services)
    
    @pytest.mark.asyncio
    async def test_analyze_free_operations_only(self, analyzer, mock_services):
        """Test analysis using only free operations (default cost preferences)."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_log_groups = AsyncMock(
            return_value=CloudWatchOperationResult(
                success=True,
                data={
                    'log_groups': [
                        {'logGroupName': '/aws/lambda/test', 'storedBytes': 1024000, 'retentionInDays': None}
                    ],
                    'total_count': 1,
                    'analysis': {'without_retention_policy': ['/aws/lambda/test']}
                }
            )
        )
        
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(
            return_value=CloudWatchOperationResult(
                success=True,
                data={
                    'alarms': [
                        {'AlarmName': 'test-alarm', 'Period': 300, 'AlarmActions': []}
                    ],
                    'total_count': 1,
                    'analysis': {'alarms_without_actions': ['test-alarm']}
                }
            )
        )
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(
            return_value=CloudWatchOperationResult(
                success=True,
                data={
                    'dashboards': [
                        {'DashboardName': 'test-dashboard'}
                    ],
                    'total_count': 1,
                    'analysis': {'exceeds_free_tier': False}
                }
            )
        )
        
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(
            return_value=CloudWatchOperationResult(
                success=True,
                data={
                    'metrics': [
                        {'Namespace': 'AWS/Lambda', 'MetricName': 'Duration'},
                        {'Namespace': 'Custom/App', 'MetricName': 'RequestCount'}
                    ],
                    'total_count': 2
                }
            )
        )
        
        # Mock pricing service
        mock_services['pricing_service'].get_logs_pricing.return_value = {
            'status': 'success',
            'logs_pricing': {
                'ingestion_per_gb': 0.50,
                'storage_per_gb_month': 0.03,
                'insights_per_gb_scanned': 0.005
            }
        }
        
        mock_services['pricing_service'].calculate_logs_cost.return_value = {
            'status': 'success',
            'total_monthly_cost': 5.25,
            'cost_breakdown': {
                'ingestion_cost': 2.50,
                'storage_cost': 2.75,
                'insights_cost': 0.0
            }
        }
        
        # Execute analysis with default cost preferences (all False)
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30
        )
        
        # Verify results
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'general_spend'
        assert result['cost_incurred'] == False
        assert result['cost_incurring_operations'] == []
        assert result['primary_data_source'] == 'cloudwatch_config'
        
        # Verify configuration analysis was performed
        assert 'configuration_analysis' in result['data']
        config_data = result['data']['configuration_analysis']
        assert 'log_groups' in config_data
        assert 'alarms' in config_data
        assert 'dashboards' in config_data
        assert 'metrics' in config_data
        
        # Verify cost breakdown was generated
        assert 'cost_breakdown' in result['data']
        cost_breakdown = result['data']['cost_breakdown']
        assert 'logs_costs' in cost_breakdown
        assert 'total_estimated_monthly' in cost_breakdown
        
        # Verify no Cost Explorer or minimal cost metrics were called
        assert 'cost_explorer_analysis' not in result['data']
        assert 'minimal_cost_metrics_analysis' not in result['data']
    
    @pytest.mark.asyncio
    async def test_analyze_with_cost_explorer(self, analyzer, mock_services):
        """Test analysis with Cost Explorer enabled."""
        # Mock CloudWatch service responses (same as above)
        mock_services['cloudwatch_service'].describe_log_groups = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'log_groups': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'alarms': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'dashboards': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'metrics': [], 'total_count': 0})
        )
        
        # Mock Cost Explorer responses
        with patch('playbooks.cloudwatch.general_spend_analyzer.get_cost_and_usage') as mock_cost_usage, \
             patch('playbooks.cloudwatch.general_spend_analyzer.get_cost_forecast') as mock_forecast:
            
            mock_cost_usage.return_value = {
                'status': 'success',
                'data': {
                    'ResultsByTime': [
                        {
                            'TimePeriod': {'Start': '2024-01-01'},
                            'Groups': [
                                {
                                    'Keys': ['DataIngestion-Bytes'],
                                    'Metrics': {
                                        'BlendedCost': {'Amount': '10.50'},
                                        'UsageQuantity': {'Amount': '21.0', 'Unit': 'GB'}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            mock_forecast.return_value = {
                'status': 'success',
                'data': {
                    'ForecastResultsByTime': [
                        {
                            'MeanValue': '12.00',
                            'TimePeriod': {'Start': '2024-02-01', 'End': '2024-03-01'}
                        }
                    ]
                }
            }
            
            # Execute analysis with Cost Explorer enabled
            result = await analyzer.analyze(
                region='us-east-1',
                lookback_days=30,
                allow_cost_explorer=True
            )
            
            # Verify results
            assert result['status'] == 'success'
            assert result['cost_incurred'] == True
            assert 'cost_explorer_analysis' in result['cost_incurring_operations']
            assert result['primary_data_source'] == 'cost_explorer'
            
            # Verify Cost Explorer analysis was performed
            assert 'cost_explorer_analysis' in result['data']
            cost_explorer_data = result['data']['cost_explorer_analysis']
            assert 'cloudwatch_costs' in cost_explorer_data
            assert 'cost_forecast' in cost_explorer_data
            
            # Verify Cost Explorer API was called
            mock_cost_usage.assert_called()
            mock_forecast.assert_called()
    
    @pytest.mark.asyncio
    async def test_analyze_with_minimal_cost_metrics(self, analyzer, mock_services):
        """Test analysis with minimal cost metrics enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_log_groups = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'log_groups': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'alarms': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'dashboards': [], 'total_count': 0})
        )
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(
            return_value=CloudWatchOperationResult(success=True, data={'metrics': [], 'total_count': 0})
        )
        
        # Mock minimal cost metrics response
        mock_services['cloudwatch_service'].get_log_group_incoming_bytes = AsyncMock(
            return_value=CloudWatchOperationResult(
                success=True,
                data={
                    'log_group_metrics': [
                        {
                            'log_group_name': '/aws/lambda/test',
                            'total_incoming_bytes': 1073741824,  # 1 GB
                            'stored_bytes': 2147483648  # 2 GB
                        }
                    ],
                    'total_log_groups': 1,
                    'total_incoming_bytes': 1073741824
                }
            )
        )
        
        # Execute analysis with minimal cost metrics enabled
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30,
            allow_minimal_cost_metrics=True
        )
        
        # Verify results
        assert result['status'] == 'success'
        assert result['cost_incurred'] == True
        assert 'minimal_cost_metrics' in result['cost_incurring_operations']
        
        # Verify minimal cost metrics analysis was performed
        assert 'minimal_cost_metrics_analysis' in result['data']
        metrics_data = result['data']['minimal_cost_metrics_analysis']
        assert 'log_ingestion_metrics' in metrics_data
        
        # Verify minimal cost metrics API was called
        mock_services['cloudwatch_service'].get_log_group_incoming_bytes.assert_called_once()
    
    def test_get_recommendations_logs_optimization(self, analyzer):
        """Test recommendation generation for logs optimization."""
        analysis_results = {
            'data': {
                'cost_breakdown': {
                    'logs_costs': {
                        'optimization_opportunities': [
                            {
                                'type': 'retention_policy',
                                'description': '5 log groups without retention policy',
                                'potential_savings': 15.50,
                                'affected_resources': ['/aws/lambda/test1', '/aws/lambda/test2']
                            }
                        ]
                    },
                    'metrics_costs': {'optimization_opportunities': []},
                    'alarms_costs': {'optimization_opportunities': []},
                    'dashboards_costs': {'optimization_opportunities': []}
                }
            },
            'cost_incurred': False,
            'cost_incurring_operations': []
        }
        
        # Set cost preferences for recommendation logic
        analyzer.cost_preferences = {
            'allow_cost_explorer': False,
            'allow_minimal_cost_metrics': False
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        # Verify recommendations were generated
        assert len(recommendations) > 0
        
        # Find logs retention recommendation
        logs_rec = next((r for r in recommendations if 'Log Retention' in r['title']), None)
        assert logs_rec is not None
        assert logs_rec['type'] == 'cost_optimization'
        assert logs_rec['priority'] == 'high'
        assert logs_rec['potential_savings'] == 15.50
        assert logs_rec['cloudwatch_component'] == 'logs'
        assert 'retention policy' in logs_rec['description']
        
        # Verify cost-aware recommendations
        cost_explorer_rec = next((r for r in recommendations if 'Cost Explorer' in r['title']), None)
        assert cost_explorer_rec is not None
        assert cost_explorer_rec['type'] == 'analysis_enhancement'
    
    def test_get_recommendations_alarms_optimization(self, analyzer):
        """Test recommendation generation for alarms optimization."""
        analysis_results = {
            'data': {
                'cost_breakdown': {
                    'logs_costs': {'optimization_opportunities': []},
                    'metrics_costs': {'optimization_opportunities': []},
                    'alarms_costs': {
                        'optimization_opportunities': [
                            {
                                'type': 'unused_alarms',
                                'description': '3 alarms without actions',
                                'potential_savings': 0.30,
                                'affected_resources': ['alarm1', 'alarm2', 'alarm3']
                            },
                            {
                                'type': 'high_resolution_optimization',
                                'description': '2 high-resolution alarms could be standard',
                                'potential_savings': 0.40,
                                'affected_resources': ['hr-alarm1', 'hr-alarm2']
                            }
                        ]
                    },
                    'dashboards_costs': {'optimization_opportunities': []}
                }
            },
            'cost_incurred': False,
            'cost_incurring_operations': []
        }
        
        analyzer.cost_preferences = {'allow_cost_explorer': False, 'allow_minimal_cost_metrics': False}
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        # Find unused alarms recommendation
        unused_rec = next((r for r in recommendations if 'Unused Alarms' in r['title']), None)
        assert unused_rec is not None
        assert unused_rec['type'] == 'cost_optimization'
        assert unused_rec['priority'] == 'high'
        assert unused_rec['potential_savings'] == 0.30
        assert unused_rec['cloudwatch_component'] == 'alarms'
        
        # Find high-resolution alarms recommendation
        hr_rec = next((r for r in recommendations if 'High-Resolution' in r['title']), None)
        assert hr_rec is not None
        assert hr_rec['type'] == 'cost_optimization'
        assert hr_rec['priority'] == 'medium'
        assert hr_rec['potential_savings'] == 0.40
        assert hr_rec['cloudwatch_component'] == 'alarms'
    
    def test_process_cost_explorer_response(self, analyzer):
        """Test Cost Explorer response processing."""
        response_data = {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01'},
                    'Groups': [
                        {
                            'Keys': ['DataIngestion-Bytes'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '5.25'},
                                'UsageQuantity': {'Amount': '10.5', 'Unit': 'GB'}
                            }
                        },
                        {
                            'Keys': ['DataStorage-ByteHrs'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '2.10'},
                                'UsageQuantity': {'Amount': '70.0', 'Unit': 'GB-Hours'}
                            }
                        }
                    ]
                },
                {
                    'TimePeriod': {'Start': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['DataIngestion-Bytes'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '6.00'},
                                'UsageQuantity': {'Amount': '12.0', 'Unit': 'GB'}
                            }
                        }
                    ]
                }
            ]
        }
        
        processed = analyzer._process_cost_explorer_response(response_data, 'logs')
        
        assert processed['service_type'] == 'logs'
        assert processed['total_cost'] == 13.35  # 5.25 + 2.10 + 6.00
        assert len(processed['daily_costs']) == 2
        assert 'DataIngestion-Bytes' in processed['usage_types']
        assert 'DataStorage-ByteHrs' in processed['usage_types']
        
        # Verify usage type aggregation
        ingestion_usage = processed['usage_types']['DataIngestion-Bytes']
        assert ingestion_usage['total_cost'] == 11.25  # 5.25 + 6.00
        assert ingestion_usage['total_usage'] == 22.5  # 10.5 + 12.0
        assert ingestion_usage['unit'] == 'GB'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer, mock_services):
        """Test error handling in analysis."""
        # Mock all services to raise exceptions
        mock_services['cloudwatch_service'].describe_log_groups = AsyncMock(
            side_effect=Exception("AWS API Error")
        )
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(
            side_effect=Exception("AWS API Error")
        )
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(
            side_effect=Exception("AWS API Error")
        )
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(
            side_effect=Exception("AWS API Error")
        )
        
        result = await analyzer.analyze(region='us-east-1')
        
        # Should return error status when all services fail
        assert result['status'] == 'error'
        assert 'message' in result or 'error' in result
    
    def test_validate_parameters(self, analyzer):
        """Test parameter validation."""
        # Valid parameters
        validation = analyzer.validate_parameters(
            region='us-east-1',
            lookback_days=30,
            log_group_names=['/aws/lambda/test'],
            timeout_seconds=60
        )
        assert validation['valid'] == True
        assert len(validation['errors']) == 0
        
        # Invalid parameters
        validation = analyzer.validate_parameters(
            region=123,  # Should be string
            lookback_days=-5,  # Should be positive
            log_group_names='not-a-list',  # Should be list
            timeout_seconds='invalid'  # Should be number
        )
        assert validation['valid'] == False
        assert len(validation['errors']) > 0
        
        # Warning for large lookback
        validation = analyzer.validate_parameters(lookback_days=400)
        assert validation['valid'] == True
        assert len(validation['warnings']) > 0
        assert 'lookback_days > 365' in validation['warnings'][0]


if __name__ == "__main__":
    pytest.main([__file__])