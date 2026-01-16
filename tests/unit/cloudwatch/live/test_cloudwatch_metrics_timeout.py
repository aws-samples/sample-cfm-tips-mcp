"""
Live test to validate CloudWatch metrics optimization performance fix.

This test validates that the metrics optimization correctly filters AWS/* namespaces
during pagination instead of loading all metrics first.
"""

import pytest
import asyncio
import time
from playbooks.cloudwatch.cloudwatch_optimization import run_cloudwatch_metrics_optimization_mcp


@pytest.mark.live
@pytest.mark.asyncio
async def test_cloudwatch_metrics_optimization_performance():
    """Test that metrics optimization completes within reasonable time by filtering during pagination."""
    
    arguments = {
        "region": "us-east-1",
        "lookback_days": 30,
        "output_format": "json",
        "page": 1,
        "timeout_seconds": 120  # Should complete well within this
    }
    
    start_time = time.time()
    result = await run_cloudwatch_metrics_optimization_mcp(arguments)
    execution_time = time.time() - start_time
    
    # Verify result structure
    assert result is not None
    assert len(result) > 0
    
    # Parse the result
    import json
    result_data = json.loads(result[0].text)
    
    # Print status for debugging
    print(f"\nStatus: {result_data['status']}")
    print(f"Execution time: {execution_time:.2f} seconds")
    
    if result_data["status"] == "error":
        print(f"Error code: {result_data.get('error_code')}")
        print(f"Error message: {result_data.get('message')}")
        if "traceback" in result_data:
            print(f"Traceback:\n{result_data['traceback']}")
        pytest.fail(f"Metrics optimization failed: {result_data.get('message')}")
    else:
        print(f"Success: {result_data.get('message')}")
        if "data" in result_data:
            data = result_data['data']
            print(f"\nData keys: {list(data.keys())}")
            
            # Check custom metrics summary
            if 'custom_metrics' in data:
                custom_summary = data['custom_metrics'].get('summary', {})
                print(f"\nCustom Metrics Summary:")
                print(f"  Total custom metrics: {custom_summary.get('total_custom_metrics', 0)}")
                print(f"  Recently active: {custom_summary.get('recently_active_metrics', 0)}")
                print(f"  Inactive: {custom_summary.get('inactive_metrics', 0)}")
                print(f"  Billable metrics: {custom_summary.get('billable_metrics', 0)}")
                print(f"  Monthly cost: ${custom_summary.get('total_estimated_monthly_cost', 0):.2f}")
                
                # Verify we're not counting AWS/* metrics
                total_custom = custom_summary.get('total_custom_metrics', 0)
                assert total_custom < 10000, f"Custom metrics count ({total_custom}) seems too high - may be including AWS/* metrics"
                
            # Verify execution time is reasonable (should be much faster than 600+ seconds)
            assert execution_time < 120, f"Execution took {execution_time:.2f}s, should be under 120s with the fix"
            print(f"\n✓ Performance test passed - completed in {execution_time:.2f}s")


@pytest.mark.live
@pytest.mark.asyncio  
async def test_cloudwatch_metrics_excludes_aws_namespaces():
    """Test that custom metrics list excludes AWS/* namespaces."""
    
    arguments = {
        "region": "us-east-1",
        "lookback_days": 30,
        "output_format": "json",
        "page": 1,
        "timeout_seconds": 120
    }
    
    result = await run_cloudwatch_metrics_optimization_mcp(arguments)
    
    # Parse the result
    import json
    result_data = json.loads(result[0].text)
    
    assert result_data["status"] == "success", f"Expected success, got: {result_data.get('message')}"
    
    # Check that no AWS/* metrics are in the custom metrics list
    if "data" in result_data and "custom_metrics" in result_data["data"]:
        custom_metrics_data = result_data["data"]["custom_metrics"]
        metrics_list = custom_metrics_data.get("custom_metrics", [])
        
        print(f"\nChecking {len(metrics_list)} metrics on page 1...")
        
        for metric in metrics_list:
            namespace = metric.get("namespace", "")
            assert not namespace.startswith("AWS/"), \
                f"Found AWS namespace in custom metrics: {namespace}"
        
        print(f"✓ All {len(metrics_list)} metrics are custom (non-AWS) metrics")


if __name__ == "__main__":
    # Run the tests
    print("="*80)
    print("Test 1: Performance Test")
    print("="*80)
    asyncio.run(test_cloudwatch_metrics_optimization_performance())
    
    print("\n" + "="*80)
    print("Test 2: AWS Namespace Exclusion Test")
    print("="*80)
    asyncio.run(test_cloudwatch_metrics_excludes_aws_namespaces())
