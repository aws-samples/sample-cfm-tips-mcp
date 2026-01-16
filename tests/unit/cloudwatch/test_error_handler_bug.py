"""Test to replicate error handler bug when exception is converted to string."""
import pytest
from utils.error_handler import AWSErrorHandler


def test_error_handler_with_string_exception():
    """Test that error handler can handle string exceptions without crashing."""
    # Simulate what happens in cloudwatch_optimization.py line 78
    # where str(e) is passed to error_response
    try:
        raise ValueError("Original exception message")
    except Exception as e:
        error_string = str(e)
        
        # This should not crash with AttributeError
        result = AWSErrorHandler.format_general_error(error_string, "test_context")
        
        assert result is not None
        assert result["status"] == "error"
        assert result["error_code"] == "GeneralError"
        assert result["message"] == "Original exception message"
        assert "traceback" in result


def test_error_handler_with_actual_exception():
    """Test that error handler works correctly with actual exception objects."""
    try:
        raise ValueError("Test exception")
    except Exception as e:
        # This should work fine
        result = AWSErrorHandler.format_general_error(e, "test_context")
        
        assert result is not None
        assert result["status"] == "error"
        assert result["error_code"] == "ValueError"
        assert result["message"] == "Test exception"
        assert "traceback" in result
        assert "ValueError: Test exception" in result["traceback"]
