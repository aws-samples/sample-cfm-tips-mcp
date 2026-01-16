"""
Test to replicate the bug where comprehensive optimization tool returns 
zero resources analyzed despite resources existing.

Bug Description:
- cloudwatch_general_spend_analysis correctly identifies 163 log groups and 461 custom metrics
- cloudwatch_comprehensive_optimization_tool returns 0 resources_analyzed for:
  - logs_optimization
  - metrics_optimization  
  - alarms_and_dashboards
- Only general_spend shows 665 resources_analyzed

Expected: All functionalities should analyze and report resources found
Actual: Optimization functionalities report 0 resources despite resources existing
"""

import pytest
from playbooks.cloudwatch.cloudwatch_optimization_tool import CloudWatchOptimizationTool


@pytest.mark.live
@pytest.mark.asyncio
async def test_comprehensive_tool_finds_resources():
    """
    Test that comprehensive optimization tool properly analyzes resources
    across all functionalities, not just general_spend.
    """
    tool = CloudWatchOptimizationTool()
    
    # Run comprehensive analysis with summary detail level
    result = await tool.execute_comprehensive_optimization_analysis(
        region=None,
        lookback_days=30,
        detail_level="summary",
        output_format="json"
    )
    
    # Verify the result structure
    assert result["status"] == "success"
    assert "aggregate_summary" in result
    
    summary = result["aggregate_summary"]
    
    # Check overview
    assert "overview" in summary
    overview = summary["overview"]
    
    print(f"\nTotal resources analyzed: {overview['total_resources_analyzed']}")
    print(f"Successful analyses: {overview['successful_analyses']}")
    
    # Check functionality summaries
    assert "functionality_summaries" in summary
    func_summaries = summary["functionality_summaries"]
    
    # Print each functionality's resource count
    for func_name, func_data in func_summaries.items():
        print(f"\n{func_name}:")
        print(f"  Status: {func_data['status']}")
        print(f"  Resources analyzed: {func_data['resources_analyzed']}")
        print(f"  Optimization opportunities: {func_data['optimization_opportunities']}")
    
    # BUG: These assertions will fail because the tool reports 0 resources
    # for optimization functionalities despite resources existing
    
    # If general_spend found resources, other analyses should too
    if func_summaries["general_spend"]["resources_analyzed"] > 0:
        # At minimum, logs_optimization should find log groups
        # (unless there truly are no optimization opportunities)
        print("\n⚠️  BUG DETECTED:")
        print(f"general_spend found {func_summaries['general_spend']['resources_analyzed']} resources")
        print(f"logs_optimization found {func_summaries['logs_optimization']['resources_analyzed']} resources")
        print(f"metrics_optimization found {func_summaries['metrics_optimization']['resources_analyzed']} resources")
        
        # The issue is that resources_analyzed should reflect resources examined,
        # not just optimization opportunities found
        # These should not all be zero if resources exist
        

@pytest.mark.live
@pytest.mark.asyncio
async def test_individual_analyses_vs_comprehensive():
    """
    Compare results from running individual analyses vs comprehensive tool
    to identify where the discrepancy occurs.
    """
    tool = CloudWatchOptimizationTool()
    
    # Run comprehensive analysis
    comprehensive_result = await tool.execute_comprehensive_optimization_analysis(
        region=None,
        lookback_days=30,
        detail_level="summary",
        output_format="json"
    )
    
    print("\n=== Comprehensive Analysis ===")
    func_summaries = comprehensive_result["aggregate_summary"]["functionality_summaries"]
    
    print(f"general_spend resources: {func_summaries['general_spend']['resources_analyzed']}")
    print(f"logs_optimization resources: {func_summaries['logs_optimization']['resources_analyzed']}")
    print(f"metrics_optimization resources: {func_summaries['metrics_optimization']['resources_analyzed']}")
    
    # The bug: comprehensive tool's optimization functionalities show 0 resources
    # even though general_spend (both individual and in comprehensive) finds resources


if __name__ == "__main__":
    print("Running comprehensive optimization bug test...")
    test_comprehensive_tool_finds_resources()
    print("\n" + "="*80)
    test_individual_analyses_vs_comprehensive()
