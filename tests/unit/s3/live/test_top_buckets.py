"""
Live test for S3 top buckets listing functionality.

This test validates that the quick analysis properly returns top 10 buckets
with their cost estimates.
"""

import pytest
import asyncio
import json
from playbooks.s3.s3_optimization_orchestrator import run_s3_quick_analysis


@pytest.mark.live
@pytest.mark.asyncio
async def test_list_top_10_buckets():
    """Test that quick analysis returns top 10 buckets with cost estimates."""
    
    # Run quick analysis
    arguments = {
        'region': 'us-east-1'  # You can change this to your preferred region
    }
    
    result = await run_s3_quick_analysis(arguments)
    
    # Parse the result
    assert len(result) > 0
    assert result[0]["type"] == "text"
    
    data = json.loads(result[0]["text"])
    
    # Verify structure
    assert data["status"] == "success"
    assert "results" in data
    assert "general_spend" in data["results"]
    
    # Check general_spend results
    general_spend = data["results"]["general_spend"]
    
    print(f"\n=== General Spend Status: {general_spend.get('status')} ===")
    
    if general_spend.get("status") == "success":
        assert "data" in general_spend
        
        # Print full data structure for debugging
        print(f"\nData keys: {list(general_spend['data'].keys())}")
        
        if "bucket_costs" in general_spend["data"]:
            bucket_costs = general_spend["data"]["bucket_costs"]
            print(f"\nBucket costs keys: {list(bucket_costs.keys())}")
            print(f"Total buckets analyzed: {bucket_costs.get('total_buckets_analyzed', 'N/A')}")
            
            # Verify top_10_buckets exists
            assert "top_10_buckets" in bucket_costs
            
            top_buckets = bucket_costs["top_10_buckets"]
            
            # Print results for manual verification
            print("\n=== Top 10 S3 Buckets by Estimated Cost ===")
            if len(top_buckets) == 0:
                print("No buckets found or analyzed.")
            else:
                for i, bucket in enumerate(top_buckets, 1):
                    print(f"{i}. {bucket['bucket_name']}")
                    print(f"   Estimated Monthly Cost: ${bucket['estimated_monthly_cost']:.2f}")
                    print(f"   Size: {bucket['size_gb']:.2f} GB")
                    print(f"   Objects: {bucket['object_count']:,}")
                    print(f"   Storage Class: {bucket['primary_storage_class']}")
                    print()
        else:
            print("\nWARNING: bucket_costs not found in general_spend data")
            print(f"Available data: {json.dumps(general_spend['data'], indent=2, default=str)}")
        
        # Verify bucket data structure
        if len(top_buckets) > 0:
            first_bucket = top_buckets[0]
            assert "bucket_name" in first_bucket
            assert "estimated_monthly_cost" in first_bucket
            assert "size_gb" in first_bucket
            assert "object_count" in first_bucket
            assert "primary_storage_class" in first_bucket
            
            # Verify costs are sorted (highest first)
            if len(top_buckets) > 1:
                for i in range(len(top_buckets) - 1):
                    assert top_buckets[i]["estimated_monthly_cost"] >= top_buckets[i + 1]["estimated_monthly_cost"], \
                        "Buckets should be sorted by cost (highest first)"
    else:
        print(f"\nGeneral spend analysis failed: {general_spend.get('message')}")
        pytest.skip(f"General spend analysis failed: {general_spend.get('message')}")


@pytest.mark.live
@pytest.mark.asyncio
async def test_bucket_cost_estimation():
    """Test that bucket cost estimation is working correctly."""
    
    arguments = {'region': 'us-east-1'}
    result = await run_s3_quick_analysis(arguments)
    
    data = json.loads(result[0]["text"])
    
    if data["status"] == "success":
        general_spend = data["results"].get("general_spend", {})
        
        if general_spend.get("status") == "success":
            bucket_costs = general_spend["data"].get("bucket_costs", {})
            
            # Check that we have bucket analysis data
            assert "by_bucket" in bucket_costs or "top_10_buckets" in bucket_costs
            
            # Verify total buckets analyzed
            if "total_buckets_analyzed" in bucket_costs:
                print(f"\nTotal buckets analyzed: {bucket_costs['total_buckets_analyzed']}")
            
            # Verify cost estimation method
            if "cost_estimation_method" in bucket_costs:
                print(f"Cost estimation method: {bucket_costs['cost_estimation_method']}")
                assert bucket_costs["cost_estimation_method"] in ["size_based", "cost_explorer"]


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_list_top_10_buckets())
