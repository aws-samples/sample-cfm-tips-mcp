#!/usr/bin/env python3
"""
CloudWatch Unit Test Suite Runner

Runs all CloudWatch unit tests including pagination tests.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestCloudWatchUnitSuite:
    """CloudWatch unit test suite runner."""
    
    def test_run_all_cloudwatch_unit_tests(self):
        """Run all CloudWatch unit tests."""
        # Get the directory containing this file
        test_dir = os.path.dirname(__file__)
        
        # Run all test files in the cloudwatch unit test directory, excluding this suite runner
        exit_code = pytest.main([
            test_dir,
            '-v',
            '--tb=short',
            '--disable-warnings',
            '--ignore=' + __file__  # Exclude this suite runner to prevent recursion
        ])
        
        assert exit_code == 0, "CloudWatch unit tests failed"


if __name__ == '__main__':
    # Run the CloudWatch unit test suite
    test_dir = os.path.dirname(__file__)
    exit_code = pytest.main([
        test_dir,
        '-v',
        '--tb=short',
        '--ignore=' + __file__  # Exclude this suite runner to prevent recursion
    ])
    
    sys.exit(exit_code)