#!/usr/bin/env python3
"""
Test runner for S3 optimization system.

This script provides a convenient way to run different test suites
with appropriate configurations and reporting.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=core",
            "--cov=services", 
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit"
        ])
    
    cmd.extend([
        "-m", "unit",
        "--tb=short"
    ])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "-m", "integration",
        "--tb=short"
    ])
    
    return run_command(cmd, "Integration Tests")


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/performance/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "-m", "performance",
        "--tb=short",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ])
    
    return run_command(cmd, "Performance Tests")


def run_cost_validation_tests(verbose: bool = False) -> int:
    """Run critical no-cost constraint validation tests."""
    cmd = ["python", "-m", "pytest", "tests/no_cost_validation/"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([
        "-m", "no_cost_validation",
        "--tb=long",  # More detailed output for critical tests
        "--strict-markers"
    ])
    
    return run_command(cmd, "No-Cost Constraint Validation Tests (CRITICAL)")


def run_all_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run all test suites."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=core",
            "--cov=services",
            "--cov-report=term-missing", 
            "--cov-report=html:htmlcov/all",
            "--cov-fail-under=80"
        ])
    
    cmd.extend([
        "--tb=short",
        "--durations=10"
    ])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path: str, verbose: bool = False) -> int:
    """Run a specific test file or directory."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--tb=short"])
    
    return run_command(cmd, f"Specific Test: {test_path}")


def check_test_environment() -> bool:
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__} is available")
    except ImportError:
        print("✗ pytest is not installed")
        return False
    
    # Check if moto is available for AWS mocking
    try:
        import moto
        print(f"✓ moto {moto.__version__} is available")
    except ImportError:
        print("✗ moto is not installed")
        return False
    
    # Check if core modules can be imported
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from playbooks.s3.base_analyzer import BaseAnalyzer
        from services.s3_service import S3Service
        print("✓ Core modules can be imported")
    except ImportError as e:
        print(f"✗ Cannot import core modules: {e}")
        return False
    
    print("✓ Test environment is ready")
    return True


def generate_test_report() -> int:
    """Generate comprehensive test report."""
    cmd = [
        "python", "-m", "pytest", "tests/",
        "--html=test_report.html",
        "--self-contained-html",
        "--json-report",
        "--json-report-file=test_report.json",
        "--cov=core",
        "--cov=services",
        "--cov-report=html:htmlcov/report",
        "--tb=short"
    ]
    
    return run_command(cmd, "Comprehensive Test Report Generation")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for S3 optimization system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests only
  python run_tests.py --integration             # Run integration tests only
  python run_tests.py --performance             # Run performance tests only
  python run_tests.py --cost-validation         # Run cost validation tests only
  python run_tests.py --all                     # Run all tests
  python run_tests.py --specific tests/unit/    # Run specific test directory
  python run_tests.py --report                  # Generate comprehensive report
  python run_tests.py --check                   # Check test environment
        """
    )
    
    # Test suite selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--cost-validation", action="store_true", help="Run cost validation tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--specific", type=str, help="Run specific test file or directory")
    
    # Utility options
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--check", action="store_true", help="Check test environment")
    
    # Test options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check:
        if check_test_environment():
            return 0
        else:
            return 1
    
    # Generate report if requested
    if args.report:
        return generate_test_report()
    
    # Determine which tests to run
    exit_code = 0
    coverage = not args.no_coverage
    
    if args.unit:
        exit_code = run_unit_tests(args.verbose, coverage)
    elif args.integration:
        exit_code = run_integration_tests(args.verbose)
    elif args.performance:
        exit_code = run_performance_tests(args.verbose)
    elif args.cost_validation:
        exit_code = run_cost_validation_tests(args.verbose)
    elif args.all:
        exit_code = run_all_tests(args.verbose, coverage)
    elif args.specific:
        exit_code = run_specific_test(args.specific, args.verbose)
    else:
        # Default: run unit and integration tests
        print("No specific test suite selected. Running unit and integration tests...")
        exit_code = run_unit_tests(args.verbose, coverage)
        if exit_code == 0:
            exit_code = run_integration_tests(args.verbose)
    
    # Summary
    if exit_code == 0:
        print(f"\n{'='*60}")
        print("✓ All tests passed successfully!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("✗ Some tests failed. Check the output above for details.")
        print(f"{'='*60}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())