"""
Test to replicate the JSON parsing error in general spend analysis.

This test replicates the exact issue where cloudwatch_general_spend_analysis
returns "Expecting value: line 1 column 1 (char 0)" error.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock


@pytest.mark.asyncio
async def test_general_spend_analysis_json_error():
    """Test that general spend analysis returns valid JSON response with empty arguments."""
    
    # Import the MCP function
    from playbooks.cloudwatch.cloudwatch_optimization import run_cloudwatch_general_spend_analysis_mcp
    
    # Call with empty arguments - should succeed with defaults
    result = await run_cloudwatch_general_spend_analysis_mcp({})
    
    # Result should be a list of TextContent
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Extract the JSON from TextContent
    import json
    result_text = result[0].text
    result_dict = json.loads(result_text)
    
    # Should have success status (function now has proper defaults)
    assert result_dict['status'] == 'success'
    
    # Should have standard response structure
    assert 'data' in result_dict
    assert 'message' in result_dict
    
    print(f"Success response: {json.dumps(result_dict, indent=2)}")


@pytest.mark.asyncio
async def test_general_spend_analysis_with_mock_analyzer_error():
    """Test error handling when analyzer raises an exception."""
    
    from playbooks.cloudwatch.cloudwatch_optimization import run_cloudwatch_general_spend_analysis_mcp
    
    # Mock the analyzer to raise an exception
    with patch('playbooks.cloudwatch.cloudwatch_optimization_analyzer.CloudWatchOptimizationAnalyzer') as mock_analyzer_class:
        mock_analyzer = Mock()
        mock_analyzer.analyze_general_spend = AsyncMock(side_effect=ValueError("Test error message"))
        mock_analyzer_class.return_value = mock_analyzer
        
        # Call the function
        result = await run_cloudwatch_general_spend_analysis_mcp({})
        
        # Extract result
        import json
        result_text = result[0].text
        result_dict = json.loads(result_text)
        
        # Should have error with full traceback
        assert result_dict['status'] == 'error'
        assert 'traceback' in result_dict
        assert 'ValueError' in result_dict['traceback']
        assert 'Test error message' in result_dict['message']
        
        print(f"Error response with mock: {json.dumps(result_dict, indent=2)}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_general_spend_analysis_json_error())
    asyncio.run(test_general_spend_analysis_with_mock_analyzer_error())
