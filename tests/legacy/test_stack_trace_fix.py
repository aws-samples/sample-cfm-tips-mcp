#!/usr/bin/env python3
"""
Test to verify stack traces are now properly captured in CloudWatch functions.
"""

import asyncio
import json
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from playbooks.cloudwatch.cloudwatch_optimization import run_cloudwatch_general_spend_analysis_mcp as run_cloudwatch_general_spend_analysis


async def test_stack_trace_capture():
    """Test that CloudWatch functions now capture full stack traces."""
    print("Testing CloudWatch stack trace capture...")
    
    # Test with invalid region to trigger an error
    arguments = {
        "region": "invalid-region-12345",  # This should cause an error
        "lookback_days": 1,
        "page": 1
    }
    
    print(f"Calling run_cloudwatch_general_spend_analysis with invalid region: {arguments}")
    
    try:
        result = await run_cloudwatch_general_spend_analysis(arguments)
        
        print("Result received:")
        for content in result:
            result_text = content.text
            print(result_text)
            
            # Check if the result contains a full stack trace
            if "Full stack trace:" in result_text:
                print("✅ SUCCESS: Full stack trace found in error response")
                return True
            else:
                print("❌ FAILURE: No full stack trace found in error response")
                return False
                
    except Exception as e:
        print(f"❌ FAILURE: Exception not handled properly: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_stack_trace_capture())
    if success:
        print("\n✅ Stack trace fix verification PASSED")
    else:
        print("\n❌ Stack trace fix verification FAILED")
        sys.exit(1)