#!/usr/bin/env python3
"""
Unit Test Suite Runner - Second Level Suite
Runs all unit tests across all playbooks.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def run_unit_tests():
    """Run all unit tests across all playbooks."""
    print("🧪 Running Unit Test Suite")
    print("=" * 50)
    
    # Define test directories for each playbook (relative to tests directory)
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to tests directory
    test_dirs = [
        os.path.join(base_dir, "unit/cloudwatch/"),
        os.path.join(base_dir, "unit/ec2/"),
        os.path.join(base_dir, "unit/s3/"),
        # Add other playbooks as they are organized
    ]
    
    # Filter to only existing directories
    existing_dirs = [d for d in test_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        print("❌ No unit test directories found")
        return False
    
    print(f"Running unit tests from: {existing_dirs}")
    
    # Run pytest on all unit test directories
    exit_code = pytest.main([
        "-v",
        "--tb=short",
        "--color=yes",
        *existing_dirs
    ])
    
    success = exit_code == 0
    
    if success:
        print("\n🎉 ALL UNIT TESTS PASSED!")
    else:
        print(f"\n❌ UNIT TESTS FAILED (exit code: {exit_code})")
    
    return success


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)