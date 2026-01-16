#!/usr/bin/env python3
"""
Live tests for CloudWatch MCP tools in PAID mode (with Cost Explorer enabled)

These tests require:
- Valid AWS credentials
- Cost Explorer API access
- IAM permissions for CloudWatch and Cost Explorer
"""

import pytest
import asyncio
from playbooks.cloudwatch.general_spend_analyzer import GeneralSpendAnalyzer
from playbooks.cloudwatch.metrics_optimization_analyzer import MetricsOptimizationAnalyzer
from playbooks.cloudwatch.logs_optimization_analyzer import LogsOptimizationAnalyzer
from playbooks.cloudwatch.alarms_and_dashboards_analyzer import AlarmsAndDashboardsAnalyzer


@pytest.mark.live
@pytest.mark.asyncio
async def test_general_spend_paid_mode():
    """Test general spend analysis with Cost Explorer enabled"""
    analyzer = GeneralSpendAnalyzer()
    
    result = await analyzer.analyze(
        region="us-east-1",
        lookback_days=30,
        allow_cost_explorer=True,
        allow_aws_config=False,
        allow_cloudtrail=False,
        allow_minimal_cost_metrics=False,
        page=1,
        page_size=10
    )
    
    assert result.get('status') == 'success'
    assert result.get('cost_incurred') is True, "Cost Explorer should incur costs"
    
    # Verify we got Cost Explorer data
    data = result.get('data', {})
    assert 'cost_explorer_analysis' in data, "Should have Cost Explorer analysis in paid mode"
    
    cost_explorer_data = data.get('cost_explorer_analysis', {})
    assert 'cloudwatch_costs' in cost_explorer_data or 'logs_costs' in cost_explorer_data


@pytest.mark.live
@pytest.mark.asyncio
async def test_metrics_optimization_paid_mode():
    """Test metrics optimization with Cost Explorer enabled"""
    analyzer = MetricsOptimizationAnalyzer()
    
    result = await analyzer.analyze(
        region="us-east-1",
        lookback_days=30,
        allow_cost_explorer=True,
        allow_aws_config=False,
        allow_cloudtrail=False,
        allow_minimal_cost_metrics=False,
        page=1,
        page_size=10
    )
    
    assert result.get('status') == 'success'
    
    # Check if Cost Explorer was used
    data = result.get('data', {})
    custom_metrics = data.get('custom_metrics', {})
    
    # In paid mode, should not use free_tier_sample
    if custom_metrics.get('data_source'):
        assert 'free_tier' not in custom_metrics.get('data_source', '').lower()


@pytest.mark.live
@pytest.mark.asyncio
async def test_logs_optimization_paid_mode():
    """Test logs optimization with Cost Explorer enabled"""
    analyzer = LogsOptimizationAnalyzer()
    
    result = await analyzer.analyze(
        region="us-east-1",
        lookback_days=30,
        allow_cost_explorer=True,
        allow_aws_config=False,
        allow_cloudtrail=False,
        allow_minimal_cost_metrics=False,
        page=1,
        page_size=10
    )
    
    assert result.get('status') == 'success'
    assert result.get('cost_incurred') is True, "Cost Explorer should incur costs"
    
    # Verify Cost Explorer logs data
    data = result.get('data', {})
    assert 'cost_explorer_logs_analysis' in data or 'optimization_analysis' in data


@pytest.mark.live
@pytest.mark.asyncio
async def test_alarms_dashboards_paid_mode():
    """Test alarms/dashboards optimization with Cost Explorer enabled"""
    analyzer = AlarmsAndDashboardsAnalyzer()
    
    result = await analyzer.analyze(
        region="us-east-1",
        lookback_days=30,
        allow_cost_explorer=True,
        allow_aws_config=False,
        allow_cloudtrail=False,
        allow_minimal_cost_metrics=False,
        page=1,
        page_size=10
    )
    
    assert result.get('status') == 'success'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
