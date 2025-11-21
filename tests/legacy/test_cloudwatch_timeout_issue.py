#!/usr/bin/env python3
"""
Test to replicate the CloudWatch timeout issue and verify stack trace reporting.
"""

import asyncio
import json
import traceback
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runbook_functions import run_cloudwatch_general_spend_analysis


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
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_cloudwatch_timeout())
    if success:
        print("✅ Test completed successfully")
    else:
        print("❌ Test failed")
        sys.exit(1)