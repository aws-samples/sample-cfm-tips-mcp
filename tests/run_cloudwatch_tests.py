#!/usr/bin/env python3
"""
Test runner for CloudWatch optimization comprehensive testing suite.

Runs all CloudWatch-related tests including unit tests, integration tests,
performance tests, and cost constraint validation tests.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PASSED")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run CloudWatch optimization tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--cost-validation", action="store_true", help="Run only cost validation tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Change to tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    if args.coverage:
        base_cmd.extend([
            "--cov=playbooks.cloudwatch",
            "--cov=services.cloudwatch_service",
            "--cov=services.cloudwatch_pricing",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Test categories
    test_categories = []
    
    if args.unit or not any([args.unit, args.integration, args.performance, args.cost_validation]):
        test_categories.append(("Unit Tests", [
            "unit/analyzers/test_cloudwatch_base_analyzer.py",
            "unit/analyzers/test_cloudwatch_general_spend_analyzer.py",
            "unit/analyzers/test_metrics_optimization_analyzer.py",
            "unit/analyzers/test_logs_optimization_analyzer.py",
            "unit/analyzers/test_alarms_and_dashboards_analyzer.py",
            "unit/services/test_cloudwatch_service.py",
            "unit/services/test_cloudwatch_cost_controller.py",
            "unit/services/test_cloudwatch_query_service.py"
        ]))
    
    if args.integration or not any([args.unit, args.integration, args.performance, args.cost_validation]):
        test_categories.append(("Integration Tests", [
            "integration/test_cloudwatch_orchestrator_integration.py",
            "integration/test_cloudwatch_comprehensive_tool_integration.py"
        ]))
    
    if args.performance or not any([args.unit, args.integration, args.performance, args.cost_validation]):
        test_categories.append(("Performance Tests", [
            "performance/test_cloudwatch_parallel_execution.py"
        ]))
    
    if args.cost_validation or not any([args.unit, args.integration, args.performance, args.cost_validation]):
        test_categories.append(("Cost Constraint Validation Tests", [
            "unit/test_cloudwatch_cost_constraints.py"
        ]))
    
    # Run test categories
    all_passed = True
    results = {}
    
    for category_name, test_files in test_categories:
        print(f"\n🧪 Running {category_name}")
        print("=" * 80)
        
        category_passed = True
        for test_file in test_files:
            if os.path.exists(test_file):
                cmd = base_cmd + [test_file]
                success = run_command(cmd, f"{category_name}: {test_file}")
                if not success:
                    category_passed = False
                    all_passed = False
            else:
                print(f"⚠️  Test file not found: {test_file}")
                category_passed = False
                all_passed = False
        
        results[category_name] = category_passed
    
    # Run specific CloudWatch marker tests
    print(f"\n🧪 Running CloudWatch-specific marker tests")
    print("=" * 80)
    
    marker_tests = [
        ("No-Cost Validation", ["-m", "no_cost_validation"]),
        ("CloudWatch Unit Tests", ["-m", "unit and cloudwatch"]),
        ("CloudWatch Integration Tests", ["-m", "integration and cloudwatch"]),
        ("CloudWatch Performance Tests", ["-m", "performance and cloudwatch"])
    ]
    
    for test_name, marker_args in marker_tests:
        cmd = base_cmd + marker_args
        success = run_command(cmd, test_name)
        if not success:
            all_passed = False
        results[test_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for category, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{category:<40} {status}")
    
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"\nOverall Result: {overall_status}")
    
    if args.coverage and all_passed:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())