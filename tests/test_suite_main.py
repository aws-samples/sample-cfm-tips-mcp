#!/usr/bin/env python3
"""
Main Test Suite Runner - Top Level Suite
Orchestrates all second-level test suites (unit, performance, integration).
"""

import asyncio
import sys
import os
import importlib.util
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_suite(suite_name: str, suite_path: str) -> Dict[str, Any]:
    """Run a test suite and return results."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {suite_name.upper()} SUITE")
    print(f"{'='*60}")
    
    if not os.path.exists(suite_path):
        return {
            'name': suite_name,
            'status': 'skipped',
            'reason': f'Suite file not found: {suite_path}'
        }
    
    try:
        # Load the suite module
        spec = importlib.util.spec_from_file_location("suite_module", suite_path)
        suite_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suite_module)
        
        # Determine the appropriate function to call
        if suite_name == 'Integration' and hasattr(suite_module, 'run_integration_tests'):
            # Integration tests are async
            success = asyncio.run(suite_module.run_integration_tests())
        elif hasattr(suite_module, f'run_{suite_name.lower()}_tests'):
            # Standard naming convention
            func = getattr(suite_module, f'run_{suite_name.lower()}_tests')
            success = func()
        elif hasattr(suite_module, 'main'):
            # Fallback to main function
            success = suite_module.main()
        else:
            return {
                'name': suite_name,
                'status': 'error',
                'reason': f'No suitable entry point found in {suite_path}'
            }
        
        return {
            'name': suite_name,
            'status': 'passed' if success else 'failed',
            'success': success
        }
        
    except Exception as e:
        return {
            'name': suite_name,
            'status': 'error',
            'reason': str(e)
        }


def main():
    """Run all test suites in order."""
    print("🎯 CFM Tips - Main Test Suite Runner")
    print("=" * 60)
    print("Running hierarchical test suite:")
    print("  📁 Top Level: Main Suite")
    print("  📁 Second Level: Unit, Performance, Integration")
    print("  📁 Third Level: Playbook-specific (CloudWatch, EC2, S3, etc.)")
    print("=" * 60)
    
    # Define the test suites in execution order
    suites = [
        ("Unit", "tests/unit/test_unit_suite.py"),
        ("Performance", "tests/performance/test_performance_suite.py"),
        ("Integration", "tests/integration/test_integration_suite.py"),
    ]
    
    results = []
    
    # Run each suite
    for suite_name, suite_path in suites:
        result = run_suite(suite_name, suite_path)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 MAIN TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    
    for result in results:
        status = result['status']
        name = result['name']
        
        if status == 'passed':
            print(f"✅ {name} Suite: PASSED")
            passed += 1
        elif status == 'failed':
            print(f"❌ {name} Suite: FAILED")
            failed += 1
        elif status == 'skipped':
            print(f"⏭️  {name} Suite: SKIPPED - {result['reason']}")
            skipped += 1
        elif status == 'error':
            print(f"💥 {name} Suite: ERROR - {result['reason']}")
            errors += 1
    
    total = len(results)
    print(f"\n📈 Results: {total} suites total")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   💥 Errors: {errors}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"   📊 Success Rate: {success_rate:.1f}%")
    
    overall_success = failed == 0 and errors == 0
    
    if overall_success:
        print(f"\n🎉 ALL TEST SUITES COMPLETED SUCCESSFULLY!")
        print("   🚀 CFM Tips is ready for deployment!")
    else:
        print(f"\n⚠️  SOME TEST SUITES FAILED")
        print("   🔧 Please review failed tests before deployment")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)