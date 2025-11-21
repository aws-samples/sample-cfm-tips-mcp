"""
Unit tests for LogsOptimizationAnalyzer

Tests the CloudWatch Logs optimization analysis functionality including:
- Log groups configuration analysis
- Cost Explorer logs analysis
- Log ingestion metrics analysis
- Optimization recommendations generation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from playbooks.cloudwatch.logs_optimization_analyzer import LogsOptimizationAnalyzer


class TestLogsOptimizationAnalyzer:
    """Test suite for LogsOptimizationAnalyzer."""
    
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
        """Create LogsOptimizationAnalyzer instance with mocked services."""
        return LogsOptimizationAnalyzer(**mock_services)
    
    @pytest.fixture
    def sample_log_groups(self):
        """Sample log groups data for testing."""
        return [
            {
                'logGroupName': '/aws/lambda/production-api',
                'retentionInDays': 30,
                'storedBytes': 1024**3 + 1024,  # 1GB + 1KB (slightly over 1GB)
                'creationTime': int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp() * 1000)
            },
            {
                'logGroupName': '/aws/lambda/dev-test-function',
                'retentionInDays': None,  # No retention policy
                'storedBytes': 512 * 1024**2,  # 512MB
                'creationTime': int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
            },
            {
                'logGroupName': '/aws/ecs/production-service',
                'retentionInDays': 400,  # Long retention (> 365 days)
                'storedBytes': 5 * 1024**3,  # 5GB
                'creationTime': int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp() * 1000)
            },
            {
                'logGroupName': '/aws/lambda/unused-function',
                'retentionInDays': None,
                'storedBytes': 1024**2 - 1024,  # Just under 1MB (small, potentially unused)
                'creationTime': int((datetime.now(timezone.utc) - timedelta(days=100)).timestamp() * 1000)
            }
        ]
    
    @pytest.fixture
    def sample_cost_explorer_response(self):
        """Sample Cost Explorer response for logs."""
        return {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01', 'End': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['CloudWatchLogs-DataIngestion'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '5.50', 'Unit': 'USD'},
                                'UsageQuantity': {'Amount': '11.0', 'Unit': 'GB'}
                            }
                        },
                        {
                            'Keys': ['CloudWatchLogs-DataStorage'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '1.20', 'Unit': 'USD'},
                                'UsageQuantity': {'Amount': '40.0', 'Unit': 'GB-Month'}
                            }
                        }
                    ]
                },
                {
                    'TimePeriod': {'Start': '2024-01-02', 'End': '2024-01-03'},
                    'Groups': [
                        {
                            'Keys': ['CloudWatchLogs-DataIngestion'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '6.00', 'Unit': 'USD'},
                                'UsageQuantity': {'Amount': '12.0', 'Unit': 'GB'}
                            }
                        }
                    ]
                }
            ]
        }
    
    @pytest.fixture
    def sample_ingestion_metrics(self):
        """Sample log ingestion metrics."""
        return {
            'total_log_groups': 4,
            'total_incoming_bytes': 2 * 1024**3,  # 2GB
            'log_group_metrics': [
                {
                    'log_group_name': '/aws/lambda/production-api',
                    'incoming_bytes': 1024**3,  # 1GB
                    'stored_bytes': 1024**3
                },
                {
                    'log_group_name': '/aws/ecs/production-service',
                    'incoming_bytes': 800 * 1024**2,  # 800MB
                    'stored_bytes': 5 * 1024**3
                },
                {
                    'log_group_name': '/aws/lambda/dev-test-function',
                    'incoming_bytes': 200 * 1024**2,  # 200MB
                    'stored_bytes': 512 * 1024**2
                },
                {
                    'log_group_name': '/aws/lambda/unused-function',
                    'incoming_bytes': 1024**2,  # 1MB
                    'stored_bytes': 1024**2
                }
            ]
        }
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.analysis_type == "logs_optimization"
        assert analyzer.version == "1.0.0"
        assert analyzer.cost_preferences is None
        assert len(analyzer.retention_recommendations) == 5
    
    def test_validate_parameters_valid_input(self, analyzer):
        """Test parameter validation with valid input."""
        validation = analyzer.validate_parameters(
            region='us-east-1',
            lookback_days=30,
            log_group_names=['/aws/lambda/test'],
            timeout_seconds=60
        )
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
    
    def test_validate_parameters_invalid_input(self, analyzer):
        """Test parameter validation with invalid input."""
        validation = analyzer.validate_parameters(
            region=123,  # Should be string
            lookback_days=-5,  # Should be positive
            log_group_names='not_a_list',  # Should be list
            timeout_seconds=0  # Should be positive
        )
        
        assert validation['valid'] is False
        assert len(validation['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_log_groups_configuration_success(self, analyzer, sample_log_groups):
        """Test successful log groups configuration analysis."""
        # Mock CloudWatch service response
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {
            'log_groups': sample_log_groups,
            'total_count': len(sample_log_groups)
        }
        analyzer.cloudwatch_service.describe_log_groups = AsyncMock(return_value=mock_result)
        
        result = await analyzer._analyze_log_groups_configuration(
            log_group_names=['/aws/lambda/production-api']
        )
        
        assert result['status'] == 'success'
        assert 'log_groups_configuration_analysis' in result['data']
        
        config_data = result['data']['log_groups_configuration_analysis']
        assert 'log_groups' in config_data
        assert 'log_groups_analysis' in config_data
        
        # Verify analysis results
        analysis = config_data['log_groups_analysis']
        assert analysis['total_log_groups'] == 4
        assert len(analysis['without_retention_policy']) == 2  # dev-test-function and unused-function
        assert len(analysis['with_retention_policy']) == 2  # production-api and production-service
        assert len(analysis['large_log_groups']) == 2  # production-api (1GB) and production-service (5GB)
        assert len(analysis['unused_log_groups']) == 1  # unused-function
    
    @pytest.mark.asyncio
    async def test_analyze_log_groups_configuration_failure(self, analyzer):
        """Test log groups configuration analysis failure."""
        # Mock CloudWatch service failure
        mock_result = Mock()
        mock_result.success = False
        analyzer.cloudwatch_service.describe_log_groups = AsyncMock(return_value=mock_result)
        
        result = await analyzer._analyze_log_groups_configuration()
        
        assert result['status'] == 'success'  # Should still succeed with empty data
        config_data = result['data']['log_groups_configuration_analysis']
        assert config_data == {}
    
    @pytest.mark.asyncio
    async def test_analyze_cost_explorer_logs_success(self, analyzer, sample_cost_explorer_response):
        """Test successful Cost Explorer logs analysis."""
        with patch('playbooks.cloudwatch.logs_optimization_analyzer.get_cost_and_usage') as mock_cost, \
             patch('playbooks.cloudwatch.logs_optimization_analyzer.get_cost_forecast') as mock_forecast:
            
            # Mock Cost Explorer responses
            mock_cost.return_value = {
                'status': 'success',
                'data': sample_cost_explorer_response
            }
            mock_forecast.return_value = {
                'status': 'success',
                'data': {
                    'ForecastResultsByTime': [{
                        'MeanValue': '180.50',
                        'TimePeriod': {'Start': '2024-02-01', 'End': '2024-03-01'}
                    }]
                }
            }
            
            result = await analyzer._analyze_cost_explorer_logs(
                lookback_days=30,
                region='us-east-1'
            )
            
            assert result['status'] == 'success'
            assert 'cost_explorer_logs_analysis' in result['data']
            
            cost_data = result['data']['cost_explorer_logs_analysis']
            assert 'logs_costs' in cost_data
            assert 'logs_cost_forecast' in cost_data
            
            # Verify cost processing
            logs_costs = cost_data['logs_costs']
            assert logs_costs['service_type'] == 'logs'
            assert logs_costs['total_cost'] > 0
            assert 'logs_specific_costs' in logs_costs
            assert 'ingestion' in logs_costs['logs_specific_costs']
            assert 'storage' in logs_costs['logs_specific_costs']
    
    @pytest.mark.asyncio
    async def test_analyze_log_ingestion_metrics_success(self, analyzer, sample_ingestion_metrics):
        """Test successful log ingestion metrics analysis."""
        # Mock CloudWatch service response
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = sample_ingestion_metrics
        analyzer.cloudwatch_service.get_log_group_incoming_bytes = AsyncMock(return_value=mock_result)
        
        result = await analyzer._analyze_log_ingestion_metrics(lookback_days=30)
        
        assert result['status'] == 'success'
        assert 'log_ingestion_metrics_analysis' in result['data']
        
        ingestion_data = result['data']['log_ingestion_metrics_analysis']
        assert 'log_ingestion_metrics' in ingestion_data
        assert 'ingestion_analysis' in ingestion_data
        
        # Verify ingestion analysis
        analysis = ingestion_data['ingestion_analysis']
        assert analysis['total_ingestion_gb'] == 2.0  # 2GB total
        assert len(analysis['high_volume_log_groups']) >= 2  # Should identify high-volume groups
    
    def test_analyze_log_groups_metadata(self, analyzer, sample_log_groups):
        """Test log groups metadata analysis."""
        analysis = analyzer._analyze_log_groups_metadata(sample_log_groups)
        
        assert analysis['total_log_groups'] == 4
        assert len(analysis['without_retention_policy']) == 2
        assert len(analysis['with_retention_policy']) == 2
        assert len(analysis['large_log_groups']) == 2
        assert len(analysis['unused_log_groups']) == 1
        
        # Check retention policy distribution
        retention_dist = analysis['retention_policy_distribution']
        assert '30_days' in retention_dist
        assert '400_days' in retention_dist
        
        # Check optimization opportunities
        opportunities = analysis['optimization_opportunities']
        assert len(opportunities) >= 2  # Should have retention and large log group opportunities
        
        # Verify opportunity types
        opportunity_types = [opp['type'] for opp in opportunities]
        assert 'retention_policy_missing' in opportunity_types
        assert 'large_log_groups_long_retention' in opportunity_types
    
    def test_recommend_retention_policy(self, analyzer):
        """Test retention policy recommendations."""
        # Test development environment
        dev_retention = analyzer._recommend_retention_policy('/aws/lambda/dev-function')
        assert dev_retention == 7
        
        # Test production application
        app_retention = analyzer._recommend_retention_policy('/aws/lambda/production-api')
        assert app_retention == 90
        
        # Test system logs
        sys_retention = analyzer._recommend_retention_policy('/aws/ecs/system-logs')
        assert sys_retention == 365
        
        # Test compliance logs
        compliance_retention = analyzer._recommend_retention_policy('/compliance/audit-logs')
        assert compliance_retention == 2557
        
        # Test default case
        default_retention = analyzer._recommend_retention_policy('/custom/unknown-service')
        assert default_retention == 90
    
    def test_estimate_retention_savings(self, analyzer):
        """Test retention savings estimation."""
        # Test significant savings
        savings_90_days = analyzer._estimate_retention_savings(90)
        assert 50 < savings_90_days < 90
        
        # Test minimal savings
        savings_700_days = analyzer._estimate_retention_savings(700)
        assert 0 <= savings_700_days < 10
        
        # Test no savings
        savings_800_days = analyzer._estimate_retention_savings(800)
        assert savings_800_days == 0.0
    
    def test_analyze_ingestion_patterns(self, analyzer, sample_ingestion_metrics):
        """Test ingestion patterns analysis."""
        analysis = analyzer._analyze_ingestion_patterns(sample_ingestion_metrics)
        
        assert analysis['total_ingestion_gb'] == 2.0
        assert len(analysis['high_volume_log_groups']) >= 2
        
        # Check high volume groups are sorted by ingestion volume
        high_volume = analysis['high_volume_log_groups']
        if len(high_volume) > 1:
            assert high_volume[0]['daily_ingestion_gb'] >= high_volume[1]['daily_ingestion_gb']
        
        # Check optimization opportunities
        opportunities = analysis['optimization_opportunities']
        assert len(opportunities) >= 1
        
        # Verify opportunity structure
        for opp in opportunities:
            assert 'type' in opp
            assert 'priority' in opp
            assert 'description' in opp
            assert 'potential_actions' in opp
    
    def test_categorize_logs_usage_type(self, analyzer):
        """Test logs usage type categorization."""
        processed_data = {'logs_specific_costs': {}}
        
        # Test ingestion categorization
        analyzer._categorize_logs_usage_type('CloudWatchLogs-DataIngestion', 5.0, 10.0, processed_data)
        assert processed_data['logs_specific_costs']['ingestion'] == 5.0
        
        # Test storage categorization
        analyzer._categorize_logs_usage_type('CloudWatchLogs-DataStorage', 2.0, 40.0, processed_data)
        assert processed_data['logs_specific_costs']['storage'] == 2.0
        
        # Test insights categorization
        analyzer._categorize_logs_usage_type('CloudWatchLogs-Insights-Query', 1.0, 5.0, processed_data)
        assert processed_data['logs_specific_costs']['insights'] == 1.0
        
        # Test other categorization
        analyzer._categorize_logs_usage_type('CloudWatchLogs-Unknown', 0.5, 1.0, processed_data)
        assert processed_data['logs_specific_costs']['other'] == 0.5
    
    def test_process_logs_cost_explorer_response(self, analyzer, sample_cost_explorer_response):
        """Test Cost Explorer response processing for logs."""
        processed = analyzer._process_logs_cost_explorer_response(sample_cost_explorer_response, 'logs')
        
        assert processed['service_type'] == 'logs'
        assert processed['total_cost'] > 0
        assert len(processed['daily_costs']) == 2
        assert len(processed['usage_types']) >= 2
        
        # Check logs-specific costs categorization
        logs_costs = processed['logs_specific_costs']
        assert 'ingestion' in logs_costs
        assert 'storage' in logs_costs
        assert logs_costs['ingestion'] > 0
        assert logs_costs['storage'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_logs_optimization_analysis(self, analyzer, sample_log_groups, sample_ingestion_metrics):
        """Test comprehensive logs optimization analysis generation."""
        # Prepare analysis data
        analysis_data = {
            'log_groups_configuration_analysis': {
                'log_groups': {'log_groups': sample_log_groups},
                'log_groups_analysis': analyzer._analyze_log_groups_metadata(sample_log_groups)
            },
            'log_ingestion_metrics_analysis': {
                'log_ingestion_metrics': sample_ingestion_metrics,
                'ingestion_analysis': analyzer._analyze_ingestion_patterns(sample_ingestion_metrics)
            }
        }
        
        # Mock pricing service
        analyzer.pricing_service.get_logs_pricing = Mock(return_value={
            'status': 'success',
            'logs_pricing': {
                'ingestion_per_gb': 0.50,
                'storage_per_gb_month': 0.03
            }
        })
        
        optimization = await analyzer._generate_logs_optimization_analysis(analysis_data)
        
        assert 'retention_policy_optimization' in optimization
        assert 'ingestion_optimization' in optimization
        assert 'storage_optimization' in optimization
        assert 'cost_optimization_summary' in optimization
        assert 'recommendations' in optimization
        
        # Verify recommendations are generated
        recommendations = optimization['recommendations']
        assert len(recommendations) >= 2
        
        # Check recommendation structure
        for rec in recommendations:
            assert 'type' in rec
            assert 'priority' in rec
            assert 'title' in rec
            assert 'description' in rec
            assert 'cloudwatch_component' in rec
            assert rec['cloudwatch_component'] == 'logs'
    
    @pytest.mark.asyncio
    async def test_full_analyze_method_success(self, analyzer, sample_log_groups):
        """Test the full analyze method with successful execution."""
        # Mock all service calls
        mock_log_groups_result = Mock()
        mock_log_groups_result.success = True
        mock_log_groups_result.data = {
            'log_groups': sample_log_groups,
            'total_count': len(sample_log_groups)
        }
        analyzer.cloudwatch_service.describe_log_groups = AsyncMock(return_value=mock_log_groups_result)
        
        # Mock pricing service
        analyzer.pricing_service.get_logs_pricing = Mock(return_value={
            'status': 'success',
            'logs_pricing': {
                'ingestion_per_gb': 0.50,
                'storage_per_gb_month': 0.03
            }
        })
        
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30,
            allow_cost_explorer=False,
            allow_minimal_cost_metrics=False
        )
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'logs_optimization'
        assert result['cost_incurred'] is False
        assert result['primary_data_source'] == 'cloudwatch_logs_config'
        assert 'data' in result
        assert 'execution_time' in result
        
        # Verify data structure
        data = result['data']
        assert 'log_groups_configuration_analysis' in data
        assert 'optimization_analysis' in data
        
        # Verify optimization analysis
        optimization = data['optimization_analysis']
        assert 'retention_policy_optimization' in optimization
        assert 'storage_optimization' in optimization
        assert 'recommendations' in optimization
    
    @pytest.mark.asyncio
    async def test_analyze_with_cost_explorer_enabled(self, analyzer, sample_cost_explorer_response):
        """Test analyze method with Cost Explorer enabled."""
        # Mock CloudWatch service
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {'log_groups': [], 'total_count': 0}
        analyzer.cloudwatch_service.describe_log_groups = AsyncMock(return_value=mock_result)
        
        # Mock Cost Explorer
        with patch('playbooks.cloudwatch.logs_optimization_analyzer.get_cost_and_usage') as mock_cost:
            mock_cost.return_value = {
                'status': 'success',
                'data': sample_cost_explorer_response
            }
            
            result = await analyzer.analyze(
                region='us-east-1',
                lookback_days=30,
                allow_cost_explorer=True
            )
            
            assert result['status'] == 'success'
            assert result['cost_incurred'] is True
            assert 'cost_explorer_logs_analysis' in result['cost_incurring_operations']
            assert result['primary_data_source'] == 'cost_explorer'
    
    @pytest.mark.asyncio
    async def test_analyze_with_minimal_cost_metrics_enabled(self, analyzer, sample_ingestion_metrics):
        """Test analyze method with minimal cost metrics enabled."""
        # Mock CloudWatch service
        mock_log_groups_result = Mock()
        mock_log_groups_result.success = True
        mock_log_groups_result.data = {'log_groups': [], 'total_count': 0}
        analyzer.cloudwatch_service.describe_log_groups = AsyncMock(return_value=mock_log_groups_result)
        
        mock_ingestion_result = Mock()
        mock_ingestion_result.success = True
        mock_ingestion_result.data = sample_ingestion_metrics
        analyzer.cloudwatch_service.get_log_group_incoming_bytes = AsyncMock(return_value=mock_ingestion_result)
        
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30,
            allow_minimal_cost_metrics=True
        )
        
        assert result['status'] == 'success'
        assert result['cost_incurred'] is True
        assert 'minimal_cost_logs_metrics' in result['cost_incurring_operations']
    
    def test_get_recommendations_with_data(self, analyzer):
        """Test get_recommendations method with analysis data."""
        analysis_results = {
            'data': {
                'optimization_analysis': {
                    'recommendations': [
                        {
                            'type': 'retention_policy_optimization',
                            'priority': 'high',
                            'title': 'Test Recommendation'
                        }
                    ]
                }
            }
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        assert len(recommendations) == 1
        assert recommendations[0]['type'] == 'retention_policy_optimization'
        assert recommendations[0]['priority'] == 'high'
    
    def test_get_recommendations_fallback(self, analyzer):
        """Test get_recommendations method with fallback."""
        analysis_results = {'data': {}}
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        assert len(recommendations) >= 1
        assert recommendations[0]['type'] == 'general_logs_optimization'
        assert recommendations[0]['cloudwatch_component'] == 'logs'
    
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, analyzer):
        """Test error handling in analyze method."""
        # Mock the prepare_analysis_context method to raise an exception
        analyzer.prepare_analysis_context = Mock(side_effect=Exception("Test error"))
        
        # The exception should be raised and not caught since it happens before the main try-catch
        with pytest.raises(Exception, match="Test error"):
            await analyzer.analyze(region='us-east-1')
    
    def test_analyzer_info(self, analyzer):
        """Test analyzer info method."""
        info = analyzer.get_analyzer_info()
        
        assert info['analysis_type'] == 'logs_optimization'
        assert info['class_name'] == 'LogsOptimizationAnalyzer'
        assert info['version'] == '1.0.0'
        assert 'services' in info
        assert 'cost_optimization' in info
        
        # Verify cost optimization flags
        cost_opt = info['cost_optimization']
        assert cost_opt['prioritizes_cost_explorer'] is True
        assert cost_opt['minimizes_api_costs'] is True
        assert cost_opt['tracks_cost_operations'] is True


if __name__ == '__main__':
    pytest.main([__file__])