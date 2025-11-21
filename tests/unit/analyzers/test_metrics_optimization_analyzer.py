"""
Unit tests for MetricsOptimizationAnalyzer

Tests the CloudWatch Metrics optimization analyzer functionality including
cost analysis, configuration analysis, and recommendation generation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from playbooks.cloudwatch.metrics_optimization_analyzer import MetricsOptimizationAnalyzer
from services.cloudwatch_service import CloudWatchOperationResult


class TestMetricsOptimizationAnalyzer:
    """Test suite for MetricsOptimizationAnalyzer."""
    
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
        """Create MetricsOptimizationAnalyzer instance with mocked services."""
        return MetricsOptimizationAnalyzer(**mock_services)
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Sample metrics data for testing."""
        return {
            'metrics': [
                {
                    'Namespace': 'AWS/EC2',
                    'MetricName': 'CPUUtilization',
                    'Dimensions': [{'Name': 'InstanceId', 'Value': 'i-1234567890abcdef0'}]
                },
                {
                    'Namespace': 'MyApp/Custom',
                    'MetricName': 'RequestCount',
                    'Dimensions': [
                        {'Name': 'Environment', 'Value': 'prod'},
                        {'Name': 'Service', 'Value': 'api'},
                        {'Name': 'Region', 'Value': 'us-east-1'},
                        {'Name': 'AZ', 'Value': 'us-east-1a'},
                        {'Name': 'Instance', 'Value': 'i-abc123'},
                        {'Name': 'Version', 'Value': 'v1.2.3'}
                    ]
                },
                {
                    'Namespace': 'MyApp/Custom',
                    'MetricName': 'ErrorRate',
                    'Dimensions': [{'Name': 'Service', 'Value': 'api'}]
                }
            ],
            'total_count': 3
        }
    
    @pytest.mark.asyncio
    async def test_analyze_basic_functionality(self, analyzer, mock_services):
        """Test basic analyze functionality."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data={'metrics': [], 'total_count': 0}
        ))
        
        result = await analyzer.analyze(region='us-east-1', lookback_days=30)
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'metrics_optimization'
        assert 'data' in result
        assert 'recommendations' in result
        assert result['cost_incurred'] == False  # No paid operations by default
    
    @pytest.mark.asyncio
    async def test_analyze_with_cost_explorer(self, analyzer, mock_services):
        """Test analyze with Cost Explorer enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data={'metrics': [], 'total_count': 0}
        ))
        
        # Mock Cost Explorer response
        with patch('playbooks.cloudwatch.metrics_optimization_analyzer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                'status': 'success',
                'data': {
                    'ResultsByTime': [
                        {
                            'TimePeriod': {'Start': '2024-01-01'},
                            'Groups': [
                                {
                                    'Keys': ['CloudWatch-Metrics'],
                                    'Metrics': {
                                        'BlendedCost': {'Amount': '10.50', 'Unit': 'USD'},
                                        'UsageQuantity': {'Amount': '35', 'Unit': 'Count'}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer.analyze(
                region='us-east-1',
                lookback_days=30,
                allow_cost_explorer=True
            )
            
            assert result['status'] == 'success'
            assert result['cost_incurred'] == True
            assert 'cost_explorer_metrics_analysis' in result['cost_incurring_operations']
            assert result['primary_data_source'] == 'cost_explorer'    

    @pytest.mark.asyncio
    async def test_analyze_metrics_configuration(self, analyzer, mock_services, sample_metrics_data):
        """Test metrics configuration analysis."""
        # Mock CloudWatch service response
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data=sample_metrics_data
        ))
        
        result = await analyzer._analyze_metrics_configuration()
        
        assert result['status'] == 'success'
        assert 'metrics_configuration_analysis' in result['data']
        
        config_data = result['data']['metrics_configuration_analysis']
        assert 'metrics' in config_data
        assert 'metrics_analysis' in config_data
        
        metrics_analysis = config_data['metrics_analysis']
        assert metrics_analysis['total_metrics'] == 3
        assert metrics_analysis['custom_metrics_count'] == 2  # MyApp/Custom metrics
        assert metrics_analysis['aws_metrics_count'] == 1     # AWS/EC2 metric
    
    def test_analyze_metrics_metadata(self, analyzer, sample_metrics_data):
        """Test metrics metadata analysis."""
        metrics = sample_metrics_data['metrics']
        analysis = analyzer._analyze_metrics_metadata(metrics)
        
        assert analysis['total_metrics'] == 3
        assert analysis['custom_metrics_count'] == 2
        assert analysis['aws_metrics_count'] == 1
        assert 'MyApp/Custom' in analysis['custom_namespaces']
        assert len(analysis['high_cardinality_metrics']) == 1  # RequestCount has 6 dimensions
        
        # Check free tier analysis
        free_tier = analysis['free_tier_analysis']
        assert free_tier['free_tier_limit'] == 10
        assert free_tier['within_free_tier'] == True  # Only 2 custom metrics
    
    def test_categorize_metrics_usage_type(self, analyzer):
        """Test metrics usage type categorization."""
        processed_data = {'metrics_specific_costs': {}}
        
        # Test custom metrics categorization
        analyzer._categorize_metrics_usage_type('CloudWatch-CustomMetrics', 5.0, 10, processed_data)
        assert processed_data['metrics_specific_costs']['custom_metrics'] == 5.0
        
        # Test detailed monitoring categorization (needs 'metric' and 'detailed' in usage type)
        analyzer._categorize_metrics_usage_type('CloudWatch-MetricDetailedMonitoring', 3.0, 5, processed_data)
        assert processed_data['metrics_specific_costs']['detailed_monitoring'] == 3.0
        
        # Test API requests categorization
        analyzer._categorize_metrics_usage_type('CloudWatch-Requests', 1.0, 1000, processed_data)
        assert processed_data['metrics_specific_costs']['api_requests'] == 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_with_minimal_cost_metrics(self, analyzer, mock_services):
        """Test analyze with minimal cost metrics enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data={'metrics': [], 'total_count': 0}
        ))
        
        mock_services['cloudwatch_service'].get_targeted_metric_statistics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data={'datapoints': []}
        ))
        
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30,
            allow_minimal_cost_metrics=True
        )
        
        assert result['status'] == 'success'
        assert result['cost_incurred'] == True
        assert 'minimal_cost_metrics_analysis' in result['cost_incurring_operations']
    
    @pytest.mark.asyncio
    async def test_analyze_custom_metrics_patterns(self, analyzer, mock_services, sample_metrics_data):
        """Test custom metrics patterns analysis."""
        # Mock CloudWatch service response
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data=sample_metrics_data
        ))
        
        result = await analyzer._analyze_custom_metrics_patterns(30)
        
        assert result is not None
        assert result['total_custom_metrics'] == 2
        assert 'MyApp/Custom' in result['custom_namespaces']
        assert len(result['high_cardinality_metrics']) == 1
        assert len(result['optimization_opportunities']) == 1
        
        # Check optimization opportunity
        opportunity = result['optimization_opportunities'][0]
        assert opportunity['type'] == 'reduce_cardinality'
        assert 'MyApp/Custom/RequestCount' in opportunity['metric']
    
    def test_generate_optimization_priorities(self, analyzer):
        """Test optimization priorities generation."""
        optimization_analysis = {
            'custom_metrics_optimization': {
                'optimization_opportunities': [
                    {
                        'type': 'reduce_high_cardinality_metrics',
                        'potential_monthly_savings': 15.0,
                        'implementation_effort': 'medium'
                    }
                ]
            },
            'detailed_monitoring_optimization': {
                'optimization_opportunities': [
                    {
                        'type': 'disable_detailed_monitoring',
                        'potential_monthly_savings': 25.0,
                        'implementation_effort': 'low'
                    }
                ]
            }
        }
        
        priorities = analyzer._generate_optimization_priorities(optimization_analysis)
        
        assert len(priorities) == 2
        # Higher savings with lower effort should be first
        assert priorities[0]['potential_monthly_savings'] == 25.0
        assert priorities[0]['implementation_effort'] == 'low'
        assert priorities[1]['potential_monthly_savings'] == 15.0
        assert priorities[1]['implementation_effort'] == 'medium'
    
    def test_get_recommendations_basic(self, analyzer):
        """Test basic recommendations generation."""
        analysis_results = {
            'data': {
                'optimization_analysis': {
                    'custom_metrics_optimization': {
                        'optimization_opportunities': [
                            {
                                'type': 'reduce_high_cardinality_metrics',
                                'description': 'Reduce cardinality of 5 high-cardinality metrics',
                                'potential_monthly_savings': 15.0,
                                'implementation_effort': 'medium',
                                'affected_metrics': 5
                            }
                        ]
                    }
                },
                'metrics_configuration_analysis': {
                    'metrics_analysis': {
                        'free_tier_analysis': {
                            'within_free_tier': False,
                            'custom_metrics_beyond_free_tier': 25
                        },
                        'high_cardinality_metrics': [
                            {'namespace': 'MyApp', 'metric_name': 'RequestCount'}
                        ]
                    }
                }
            }
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        assert len(recommendations) >= 2
        
        # Check for custom metrics optimization recommendation
        custom_metrics_rec = next((r for r in recommendations if 'Custom Metrics Optimization' in r['title']), None)
        assert custom_metrics_rec is not None
        assert custom_metrics_rec['potential_savings'] == 15.0
        assert custom_metrics_rec['cloudwatch_component'] == 'metrics'
        
        # Check for free tier recommendation
        free_tier_rec = next((r for r in recommendations if 'Free Tier' in r['title']), None)
        assert free_tier_rec is not None
        assert free_tier_rec['potential_savings'] == 25 * 0.30  # 25 metrics * $0.30
    
    def test_get_recommendations_with_cost_preferences_disabled(self, analyzer):
        """Test recommendations when cost preferences are disabled."""
        analyzer.cost_preferences = {
            'allow_cost_explorer': False,
            'allow_minimal_cost_metrics': False
        }
        
        analysis_results = {
            'data': {
                'optimization_analysis': {},
                'metrics_configuration_analysis': {
                    'metrics_analysis': {
                        'free_tier_analysis': {'within_free_tier': True}
                    }
                }
            }
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        # Should include recommendations to enable cost analysis features
        cost_explorer_rec = next((r for r in recommendations if 'Cost Explorer' in r['title']), None)
        assert cost_explorer_rec is not None
        assert cost_explorer_rec['type'] == 'governance'
        
        minimal_cost_rec = next((r for r in recommendations if 'Minimal Cost Metrics' in r['title']), None)
        assert minimal_cost_rec is not None
        assert minimal_cost_rec['type'] == 'governance'
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer, mock_services):
        """Test error handling in analysis."""
        # Mock CloudWatch service to raise an exception
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(side_effect=Exception("API Error"))
        
        result = await analyzer.analyze(region='us-east-1')
        
        # The analyzer handles errors gracefully and continues with partial results
        assert result['status'] == 'success'  # Still succeeds with partial data
        assert result['fallback_used'] == True  # But marks that fallback was used
        assert result['cost_incurred'] == False
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, analyzer):
        """Test parameter validation."""
        # Test invalid lookback_days
        validation = analyzer.validate_parameters(lookback_days=-5)
        assert not validation['valid']
        assert any('positive integer' in error for error in validation['errors'])
        
        # Test invalid region type
        validation = analyzer.validate_parameters(region=123)
        assert not validation['valid']
        assert any('string' in error for error in validation['errors'])
        
        # Test valid parameters
        validation = analyzer.validate_parameters(region='us-east-1', lookback_days=30)
        assert validation['valid']
        assert len(validation['errors']) == 0
    
    def test_analyzer_info(self, analyzer):
        """Test analyzer information retrieval."""
        info = analyzer.get_analyzer_info()
        
        assert info['analysis_type'] == 'metrics_optimization'
        assert info['class_name'] == 'MetricsOptimizationAnalyzer'
        assert info['version'] == '1.0.0'
        assert 'services' in info
        assert 'cost_optimization' in info
        assert info['cost_optimization']['prioritizes_cost_explorer'] == True
        assert info['cost_optimization']['minimizes_api_costs'] == True
    
    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self, analyzer, mock_services):
        """Test execute_with_error_handling method."""
        # Mock successful analysis
        mock_services['cloudwatch_service'].list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True,
            data={'metrics': [], 'total_count': 0}
        ))
        
        result = await analyzer.execute_with_error_handling(region='us-east-1', lookback_days=30)
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'metrics_optimization'
        assert 'execution_time' in result
        assert 'timestamp' in result
        assert result['cost_incurred'] == False
        assert result['primary_data_source'] == 'cloudwatch_config'


if __name__ == '__main__':
    pytest.main([__file__])