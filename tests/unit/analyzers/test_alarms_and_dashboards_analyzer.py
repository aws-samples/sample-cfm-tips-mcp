"""
Unit tests for AlarmsAndDashboardsAnalyzer.

Tests the CloudWatch alarms and dashboards efficiency analysis functionality
including alarm efficiency analysis, dashboard optimization, governance checks,
and cost-aware feature coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from playbooks.cloudwatch.alarms_and_dashboards_analyzer import AlarmsAndDashboardsAnalyzer


class TestAlarmsAndDashboardsAnalyzer:
    """Test suite for AlarmsAndDashboardsAnalyzer."""
    
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
        """Create AlarmsAndDashboardsAnalyzer instance with mocked services."""
        return AlarmsAndDashboardsAnalyzer(**mock_services)
    
    @pytest.fixture
    def sample_alarms(self):
        """Sample alarm data for testing."""
        return [
            {
                'AlarmName': 'test-alarm-1',
                'StateValue': 'OK',
                'AlarmActions': ['arn:aws:sns:us-east-1:123456789012:test-topic'],
                'MetricName': 'CPUUtilization',
                'Period': 300
            },
            {
                'AlarmName': 'unused-alarm-1',
                'StateValue': 'OK',
                'AlarmActions': [],
                'OKActions': [],
                'InsufficientDataActions': [],
                'MetricName': 'CPUUtilization',
                'Period': 300
            },
            {
                'AlarmName': 'insufficient-data-alarm',
                'StateValue': 'INSUFFICIENT_DATA',
                'AlarmActions': ['arn:aws:sns:us-east-1:123456789012:test-topic'],
                'StateReason': 'Insufficient Data',
                'MetricName': 'CPUUtilization',
                'Period': 300
            },
            {
                'AlarmName': 'high-res-alarm',
                'StateValue': 'OK',
                'AlarmActions': ['arn:aws:sns:us-east-1:123456789012:test-topic'],
                'MetricName': 'CPUUtilization',
                'Period': 60  # High resolution (1 minute)
            }
        ]
    
    @pytest.fixture
    def sample_dashboards(self):
        """Sample dashboard data for testing."""
        return [
            {
                'DashboardName': 'test-dashboard-1',
                'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/test-dashboard-1'
            },
            {
                'DashboardName': 'test-dashboard-2',
                'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/test-dashboard-2'
            },
            {
                'DashboardName': 'oversized-dashboard',
                'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/oversized-dashboard'
            }
        ]
    
    @pytest.fixture
    def sample_dashboard_config(self):
        """Sample dashboard configuration for testing."""
        return {
            'DashboardBody': '''{
                "widgets": [
                    {
                        "type": "metric",
                        "properties": {
                            "metrics": [
                                ["AWS/EC2", "CPUUtilization", "InstanceId", "i-1234567890abcdef0"],
                                ["AWS/EC2", "NetworkIn", "InstanceId", "i-1234567890abcdef0"]
                            ]
                        }
                    },
                    {
                        "type": "log",
                        "properties": {
                            "query": "fields @timestamp, @message"
                        }
                    }
                ]
            }''',
            'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/test-dashboard'
        }
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.analysis_type == "alarms_and_dashboards"
        assert analyzer.version == "1.0.0"
        assert analyzer.cost_preferences is None
    
    @pytest.mark.asyncio
    async def test_analyze_basic_functionality(self, analyzer, mock_services, sample_alarms, sample_dashboards):
        """Test basic analyze functionality with free operations only."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': sample_alarms, 'total_count': len(sample_alarms)}
        ))
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': sample_dashboards, 'total_count': len(sample_dashboards)}
        ))
        
        mock_services['cloudwatch_service'].get_dashboard = AsyncMock(return_value=Mock(
            success=True,
            data={
                'DashboardBody': '{"widgets": [{"type": "metric", "properties": {"metrics": [["AWS/EC2", "CPUUtilization"]]}}]}',
                'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/test'
            }
        ))
        
        # Execute analysis with default (free) settings
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30
        )
        
        # Verify result structure
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'alarms_and_dashboards'
        assert result['cost_incurred'] is False
        assert result['cost_incurring_operations'] == []
        assert result['primary_data_source'] == 'cloudwatch_config'
        
        # Verify data structure
        assert 'alarm_efficiency' in result['data']
        assert 'dashboard_efficiency' in result['data']
        assert 'efficiency_analysis' in result['data']
        
        # Verify alarm analysis
        alarm_data = result['data']['alarm_efficiency']
        assert alarm_data['total_alarms'] == 4
        assert alarm_data['unused_alarms_count'] == 1
        assert alarm_data['insufficient_data_count'] == 1
        assert alarm_data['high_resolution_count'] == 1
        
        # Verify dashboard analysis
        dashboard_data = result['data']['dashboard_efficiency']
        assert dashboard_data['total_dashboards'] == 3
    
    @pytest.mark.asyncio
    async def test_analyze_with_cost_explorer(self, analyzer, mock_services):
        """Test analysis with Cost Explorer enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': [], 'total_count': 0}
        ))
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': [], 'total_count': 0}
        ))
        
        # Mock Cost Explorer service
        mock_services['cost_explorer_service'].get_cloudwatch_spend_breakdown = AsyncMock(return_value={
            'cost_breakdown': {
                'CloudWatch-Alarms': 10.50,
                'CloudWatch-Dashboards': 15.00
            }
        })
        
        # Execute analysis with Cost Explorer enabled
        result = await analyzer.analyze(
            region='us-east-1',
            allow_cost_explorer=True,
            lookback_days=30
        )
        
        # Verify cost tracking
        assert result['cost_incurred'] is True
        assert 'cost_explorer_analysis' in result['cost_incurring_operations']
        assert result['primary_data_source'] == 'cost_explorer'
        
        # Verify cost data is included
        assert 'cost_analysis' in result['data']
    
    @pytest.mark.asyncio
    async def test_analyze_with_governance_checks(self, analyzer, mock_services, sample_alarms):
        """Test analysis with AWS Config governance checks enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': sample_alarms, 'total_count': len(sample_alarms)}
        ))
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': [], 'total_count': 0}
        ))
        
        # Execute analysis with AWS Config enabled
        result = await analyzer.analyze(
            region='us-east-1',
            allow_aws_config=True,
            lookback_days=30
        )
        
        # Verify cost tracking
        assert result['cost_incurred'] is True
        assert 'aws_config_governance' in result['cost_incurring_operations']
        
        # Verify governance data is included
        assert 'governance_compliance' in result['data']
        governance_data = result['data']['governance_compliance']
        assert 'compliance_score' in governance_data
        assert 'non_compliant_alarms' in governance_data
    
    @pytest.mark.asyncio
    async def test_analyze_with_cloudtrail(self, analyzer, mock_services):
        """Test analysis with CloudTrail usage patterns enabled."""
        # Mock CloudWatch service responses
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': [], 'total_count': 0}
        ))
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': [], 'total_count': 0}
        ))
        
        # Execute analysis with CloudTrail enabled
        result = await analyzer.analyze(
            region='us-east-1',
            allow_cloudtrail=True,
            lookback_days=30
        )
        
        # Verify cost tracking
        assert result['cost_incurred'] is True
        assert 'cloudtrail_usage_patterns' in result['cost_incurring_operations']
        
        # Verify usage patterns data is included
        assert 'usage_patterns' in result['data']
    
    def test_analyze_alarm_configurations(self, analyzer, sample_alarms):
        """Test alarm configuration analysis."""
        efficiency_metrics = analyzer._analyze_alarm_configurations(sample_alarms)
        
        # Verify alarm state counting
        assert efficiency_metrics['alarm_states']['OK'] == 3
        assert efficiency_metrics['alarm_states']['INSUFFICIENT_DATA'] == 1
        
        # Verify unused alarm detection
        assert efficiency_metrics['unused_alarms_count'] == 1
        assert len(efficiency_metrics['unused_alarms']) == 1
        assert efficiency_metrics['unused_alarms'][0]['alarm_name'] == 'unused-alarm-1'
        
        # Verify insufficient data alarm detection
        assert efficiency_metrics['insufficient_data_count'] == 1
        assert len(efficiency_metrics['insufficient_data_alarms']) == 1
        assert efficiency_metrics['insufficient_data_alarms'][0]['alarm_name'] == 'insufficient-data-alarm'
        
        # Verify high-resolution alarm detection
        assert efficiency_metrics['high_resolution_count'] == 1
        assert len(efficiency_metrics['high_resolution_alarms']) == 1
        assert efficiency_metrics['high_resolution_alarms'][0]['alarm_name'] == 'high-res-alarm'
        
        # Verify alarm type categorization
        assert efficiency_metrics['alarm_types']['metric'] == 4
    
    def test_analyze_dashboard_configuration(self, analyzer, sample_dashboard_config):
        """Test dashboard configuration analysis."""
        dashboard_analysis = analyzer._analyze_dashboard_configuration(
            'test-dashboard', sample_dashboard_config
        )
        
        assert dashboard_analysis['dashboard_name'] == 'test-dashboard'
        assert dashboard_analysis['widget_count'] == 2
        assert dashboard_analysis['total_metrics'] == 2
        assert 'metric' in dashboard_analysis['widget_types']
        assert 'log' in dashboard_analysis['widget_types']
    
    def test_calculate_dashboard_efficiency(self, analyzer):
        """Test dashboard efficiency calculation."""
        dashboard_details = [
            {'dashboard_name': 'dash1', 'total_metrics': 30, 'widget_count': 5},
            {'dashboard_name': 'dash2', 'total_metrics': 60, 'widget_count': 10},  # Oversized
            {'dashboard_name': 'dash3', 'total_metrics': 5, 'widget_count': 2},   # Undersized
            {'dashboard_name': 'dash4', 'total_metrics': 40, 'widget_count': 8},
            {'dashboard_name': 'dash5', 'total_metrics': 25, 'widget_count': 6}   # Paid (5th dashboard)
        ]
        
        efficiency_metrics = analyzer._calculate_dashboard_efficiency(dashboard_details)
        
        # Verify free tier calculations
        assert efficiency_metrics['free_tier_count'] == 3  # First 3 dashboards
        assert efficiency_metrics['paid_dashboards_count'] == 2  # Remaining 2 dashboards
        assert efficiency_metrics['free_tier_utilization'] == 100.0  # 3/3 * 100
        
        # Verify oversized dashboard detection
        assert len(efficiency_metrics['oversized_dashboards']) == 1
        assert efficiency_metrics['oversized_dashboards'][0]['dashboard_name'] == 'dash2'
        
        # Verify undersized dashboard detection
        assert len(efficiency_metrics['undersized_dashboards']) == 1
        assert efficiency_metrics['undersized_dashboards'][0]['dashboard_name'] == 'dash3'
        
        # Verify totals
        assert efficiency_metrics['total_dashboards'] == 5
        assert efficiency_metrics['total_metrics'] == 160
        assert efficiency_metrics['average_metrics_per_dashboard'] == 32.0
    
    def test_check_alarm_action_compliance(self, analyzer, sample_alarms):
        """Test alarm action compliance checking."""
        compliance_check = analyzer._check_alarm_action_compliance(sample_alarms)
        
        # Verify compliance calculations
        assert compliance_check['total_alarms'] == 4
        assert compliance_check['compliant_alarms'] == 3  # 3 alarms have actions
        assert len(compliance_check['non_compliant_alarms']) == 1
        assert compliance_check['compliance_score'] == 75.0  # 3/4 * 100
        assert compliance_check['compliance_status'] == 'non_compliant'
        
        # Verify non-compliant alarm details
        non_compliant = compliance_check['non_compliant_alarms'][0]
        assert non_compliant['alarm_name'] == 'unused-alarm-1'
        assert non_compliant['compliance_issue'] == 'No actions configured'
    
    def test_extract_alarms_costs(self, analyzer):
        """Test alarms cost extraction from CloudWatch cost data."""
        cloudwatch_costs = {
            'cost_breakdown': {
                'CloudWatch-Alarms': 10.50,
                'CloudWatch-Alarms-HighResolution': 5.25,
                'CloudWatch-Alarms-Composite': 2.10,
                'CloudWatch-Metrics': 15.00  # Should not be included
            }
        }
        
        alarms_costs = analyzer._extract_alarms_costs(cloudwatch_costs)
        
        assert alarms_costs['total'] == 17.85  # 10.50 + 5.25 + 2.10
        assert alarms_costs['standard_alarms'] == 10.50
        assert alarms_costs['high_resolution_alarms'] == 5.25
        assert alarms_costs['composite_alarms'] == 2.10
    
    def test_extract_dashboards_costs(self, analyzer):
        """Test dashboards cost extraction from CloudWatch cost data."""
        cloudwatch_costs = {
            'cost_breakdown': {
                'CloudWatch-Dashboards': 15.00,
                'CloudWatch-Dashboards-API': 3.50,
                'CloudWatch-Metrics': 10.00  # Should not be included
            }
        }
        
        dashboards_costs = analyzer._extract_dashboards_costs(cloudwatch_costs)
        
        assert dashboards_costs['total'] == 18.50  # 15.00 + 3.50
        assert dashboards_costs['custom_dashboards'] == 15.00
        assert dashboards_costs['dashboard_api_requests'] == 3.50
    
    def test_generate_key_findings(self, analyzer):
        """Test key findings generation."""
        analysis_data = {
            'alarm_efficiency': {
                'total_alarms': 10,
                'unused_alarms_count': 3,
                'insufficient_data_count': 2
            },
            'dashboard_efficiency': {
                'total_dashboards': 5,
                'paid_dashboards_count': 2,
                'free_tier_utilization': 100.0
            }
        }
        
        findings = analyzer._generate_key_findings(analysis_data)
        
        assert len(findings) >= 5
        assert "Found 10 total alarms" in findings
        assert "3 alarms have no actions configured (unused)" in findings
        assert "2 alarms are in INSUFFICIENT_DATA state" in findings
        assert "Found 5 total dashboards" in findings
        assert "2 dashboards exceed free tier (incurring costs)" in findings
        assert "Free tier utilization: 100.0%" in findings
    
    def test_generate_optimization_opportunities(self, analyzer):
        """Test optimization opportunities generation."""
        analysis_data = {
            'alarm_efficiency': {
                'unused_alarms_count': 5,
                'high_resolution_count': 3
            },
            'dashboard_efficiency': {
                'paid_dashboards_count': 2,
                'oversized_dashboards': [
                    {'dashboard_name': 'dash1'},
                    {'dashboard_name': 'dash2'}
                ]
            }
        }
        
        opportunities = analyzer._generate_optimization_opportunities(analysis_data)
        
        assert len(opportunities) == 4
        
        # Check alarm cleanup opportunity
        alarm_cleanup = next((opp for opp in opportunities if opp['type'] == 'alarm_cleanup'), None)
        assert alarm_cleanup is not None
        assert alarm_cleanup['potential_savings'] == 0.50  # 5 * 0.10
        
        # Check alarm optimization opportunity
        alarm_opt = next((opp for opp in opportunities if opp['type'] == 'alarm_optimization'), None)
        assert alarm_opt is not None
        assert abs(alarm_opt['potential_savings'] - 0.60) < 0.01  # 3 * 0.20 (allow for floating point precision)
        
        # Check dashboard consolidation opportunity
        dash_consolidation = next((opp for opp in opportunities if opp['type'] == 'dashboard_consolidation'), None)
        assert dash_consolidation is not None
        assert dash_consolidation['potential_savings'] == 6.00  # 2 * 3.00
        
        # Check dashboard optimization opportunity
        dash_opt = next((opp for opp in opportunities if opp['type'] == 'dashboard_optimization'), None)
        assert dash_opt is not None
        assert dash_opt['potential_savings'] == 2.00  # 2 * 1.00
    
    def test_get_recommendations_comprehensive(self, analyzer):
        """Test comprehensive recommendation generation."""
        analysis_results = {
            'data': {
                'alarm_efficiency': {
                    'unused_alarms': [
                        {'alarm_name': 'unused-1'},
                        {'alarm_name': 'unused-2'}
                    ],
                    'insufficient_data_alarms': [
                        {'alarm_name': 'insufficient-1'}
                    ],
                    'high_resolution_alarms': [
                        {'alarm_name': 'high-res-1'}
                    ]
                },
                'dashboard_efficiency': {
                    'paid_dashboards_count': 2,
                    'oversized_dashboards': [
                        {'dashboard_name': 'oversized-1'}
                    ]
                },
                'governance_compliance': {
                    'compliance_score': 80.0,
                    'non_compliant_alarms': [
                        {'alarm_name': 'non-compliant-1'}
                    ]
                },
                'cost_analysis': {
                    'total_monitoring_costs': 75.0
                },
                'efficiency_analysis': {
                    'overall_efficiency_score': 70.0,
                    'optimization_opportunities': [
                        {
                            'type': 'test_opportunity',
                            'title': 'Test Optimization',
                            'description': 'Test optimization opportunity',
                            'potential_savings': 10.0,
                            'effort': 'low'
                        }
                    ]
                }
            }
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        # Should have recommendations from all categories
        assert len(recommendations) >= 7
        
        # Check for alarm recommendations
        alarm_recs = [rec for rec in recommendations if rec.get('cloudwatch_component') == 'alarms']
        assert len(alarm_recs) >= 3
        
        # Check for dashboard recommendations
        dashboard_recs = [rec for rec in recommendations if rec.get('cloudwatch_component') == 'dashboards']
        assert len(dashboard_recs) >= 1
        
        # Check for governance recommendations
        governance_recs = [rec for rec in recommendations if rec.get('type') == 'governance']
        assert len(governance_recs) >= 1
        
        # Check for cost recommendations
        cost_recs = [rec for rec in recommendations if rec.get('type') == 'cost_optimization']
        assert len(cost_recs) >= 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer, mock_services):
        """Test error handling in analysis."""
        # Mock service to raise exception
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(side_effect=Exception("API Error"))
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': [], 'total_count': 0}
        ))
        
        result = await analyzer.analyze(region='us-east-1')
        
        # Should handle error gracefully
        assert result['status'] == 'success'  # Should still succeed with partial data
        assert result['fallback_used'] is True
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, analyzer, mock_services):
        """Test parameter validation."""
        # Setup mock services to avoid async issues
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': [], 'total_count': 0}
        ))
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': [], 'total_count': 0}
        ))
        
        # Test with invalid parameters using execute_with_error_handling
        result = await analyzer.execute_with_error_handling(
            region=123,  # Invalid type
            lookback_days=-5,  # Invalid value
            alarm_names="not-a-list"  # Invalid type
        )
        
        assert result['status'] == 'error'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) >= 3
    
    def test_cost_preferences_tracking(self, analyzer):
        """Test cost preferences are properly tracked."""
        # Test default preferences (all False)
        assert analyzer.cost_preferences is None
        
        # After analysis, preferences should be set
        # This would be tested in the actual analyze method calls above
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_flow(self, analyzer, mock_services, sample_alarms, sample_dashboards):
        """Test the complete analysis flow with all components."""
        # Setup comprehensive mocks
        mock_services['cloudwatch_service'].describe_alarms = AsyncMock(return_value=Mock(
            success=True,
            data={'alarms': sample_alarms, 'total_count': len(sample_alarms)}
        ))
        
        mock_services['cloudwatch_service'].list_dashboards = AsyncMock(return_value=Mock(
            success=True,
            data={'dashboards': sample_dashboards, 'total_count': len(sample_dashboards)}
        ))
        
        mock_services['cloudwatch_service'].get_dashboard = AsyncMock(return_value=Mock(
            success=True,
            data={
                'DashboardBody': '{"widgets": [{"type": "metric", "properties": {"metrics": [["AWS/EC2", "CPUUtilization"]]}}]}',
                'DashboardArn': 'arn:aws:cloudwatch:us-east-1:123456789012:dashboard/test'
            }
        ))
        
        mock_services['cost_explorer_service'].get_cloudwatch_spend_breakdown = AsyncMock(return_value={
            'cost_breakdown': {
                'CloudWatch-Alarms': 10.50,
                'CloudWatch-Dashboards': 15.00
            }
        })
        
        # Execute comprehensive analysis
        result = await analyzer.analyze(
            region='us-east-1',
            lookback_days=30,
            allow_cost_explorer=True,
            allow_aws_config=True,
            allow_cloudtrail=True,
            alarm_names=['test-alarm-1'],
            dashboard_names=['test-dashboard-1']
        )
        
        # Verify comprehensive result
        assert result['status'] == 'success'
        assert result['cost_incurred'] is True
        assert len(result['cost_incurring_operations']) == 3
        
        # Verify all data components are present
        assert 'alarm_efficiency' in result['data']
        assert 'dashboard_efficiency' in result['data']
        assert 'cost_analysis' in result['data']
        assert 'governance_compliance' in result['data']
        assert 'usage_patterns' in result['data']
        assert 'efficiency_analysis' in result['data']
        
        # Verify recommendations are generated
        recommendations = analyzer.get_recommendations(result)
        assert len(recommendations) > 0
        
        # Verify execution time is recorded
        assert 'execution_time' in result
        assert result['execution_time'] > 0


if __name__ == '__main__':
    pytest.main([__file__])