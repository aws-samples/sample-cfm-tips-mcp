#!/usr/bin/env python3
"""
Test to verify the CloudWatch pricing cache fix works.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.cloudwatch_pricing import CloudWatchPricing


def test_pricing_cache():
    """Test that pricing calls are cached and don't block."""
    print("Testing CloudWatch pricing cache fix...")
    
    # Initialize pricing service
    pricing = CloudWatchPricing(region='us-east-1')
    
    # First call - should use fallback pricing and cache it
    print("Making first pricing call...")
    start_time = time.time()
    result1 = pricing.get_metrics_pricing()
    first_call_time = time.time() - start_time
    
    print(f"First call took {first_call_time:.3f} seconds")
    print(f"Status: {result1.get('status')}")
    print(f"Source: {result1.get('source')}")
    
    # Second call - should use cache and be instant
    print("\nMaking second pricing call...")
    start_time = time.time()
    result2 = pricing.get_metrics_pricing()
    second_call_time = time.time() - start_time
    
    print(f"Second call took {second_call_time:.3f} seconds")
    print(f"Status: {result2.get('status')}")
    print(f"Source: {result2.get('source')}")
    
    # Verify caching worked
    if second_call_time < 0.001:  # Should be nearly instant
        print("✅ SUCCESS: Caching is working - second call was instant")
        return True
    else:
        print("❌ FAILURE: Caching not working - second call took too long")
        return False


if __name__ == "__main__":
    success = test_pricing_cache()
    if success:
        print("\n✅ Pricing cache fix verification PASSED")
    else:
        print("\n❌ Pricing cache fix verification FAILED")
        sys.exit(1)