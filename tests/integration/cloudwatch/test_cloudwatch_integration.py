#!/usr/bin/env python3
"""
Integration tests for CloudWatch functionality.
"""

import asyncio
import json
import sys
import os
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from runbook_functions import run_cloudwatch_general_spend_analysis


@pytest.mark.asyncio
async def test_cloudwatch_timeout():
    """Test CloudWatch general spend analysis to replicate timeout issue."""
    print("Testing CloudWatch general spend analysis timeout issue...")
    
    try:
        # Test with minimal parameters that should trigger the timeout
        arguments = {
            "region": "us-east-1",
            "lookback_days": 7,
            "page": 1
        }
        
        print(f"Calling run_cloudwatch_general_spend_analysis with: {arguments}")
        
        # This should timeout and we should get a full stack trace
        result = await run_cloudwatch_general_spend_analysis(arguments)
        
        print("Result received:")
        for content in result:
            print(content.text)
            
        return True
        
    except Exception as e:
        print(f"Exception caught in test: {str(e)}")
        print("Full stack trace:")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_stack_trace_capture():
    """Test that CloudWatch functions handle errors gracefully with structured responses."""
    print("Testing CloudWatch error handling...")
    
    # Test with invalid arguments that will cause an error
    arguments = {
        "region": "us-east-1",
        "lookback_days": "invalid_string",  # This should cause a type error
        "page": 1
    }
    
    print(f"Calling run_cloudwatch_general_spend_analysis with invalid lookback_days: {arguments}")
    
    try:
        result = await run_cloudwatch_general_spend_analysis(arguments)
        
        print("Result received:")
        for content in result:
            result_text = content.text
            print(result_text)
            
            # Parse the JSON response to check for proper error handling
            import json
            try:
                response_data = json.loads(result_text)
                
                # Check if it's a proper error response with structured format
                if (response_data.get('status') == 'error' and 
                    'error_message' in response_data and
                    'analysis_type' in response_data and
                    'timestamp' in response_data):
                    print("✅ SUCCESS: Structured error response found")
                    return True
                else:
                    print("❌ FAILURE: Invalid error response structure")
                    return False
                    
            except json.JSONDecodeError:
                print("❌ FAILURE: Response is not valid JSON")
                return False
                
    except Exception as e:
        print(f"❌ FAILURE: Exception not handled properly: {str(e)}")
        return False


def test_pricing_cache():
    """Test that pricing calls are cached and don't block."""
    print("Testing CloudWatch pricing cache fix...")
    
    try:
        from services.cloudwatch_pricing import CloudWatchPricing
        import time
        
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
            
    except Exception as e:
        print(f"❌ Error in pricing cache test: {str(e)}")
        return False


async def run_cloudwatch_integration_tests():
    """Run all CloudWatch integration tests."""
    print("Starting CloudWatch Integration Tests")
    print("=" * 50)
    
    tests = [
        ("CloudWatch Timeout Handling", test_cloudwatch_timeout),
        ("Error Handling", test_stack_trace_capture),
        ("Pricing Cache", test_pricing_cache),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                print(f"✓ PASS: {test_name}")
                passed += 1
            else:
                print(f"✗ FAIL: {test_name}")
                failed += 1
        except Exception as e:
            print(f"✗ FAIL: {test_name} - Exception: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"CloudWatch Integration Tests: {passed + failed} total, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL CLOUDWATCH INTEGRATION TESTS PASSED!")
        return True
    else:
        print(f"❌ {failed} CLOUDWATCH INTEGRATION TESTS FAILED")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_cloudwatch_integration_tests())
    sys.exit(0 if success else 1)