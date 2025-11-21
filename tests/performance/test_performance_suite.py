#!/usr/bin/env python3
"""
Performance Test Suite Runner - Second Level Suite
Runs all performance tests across all playbooks.
"""

import sys
import os
import importlib.util

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def run_performance_tests():
    """Run all performance tests across all playbooks."""
    print("⚡ Running Performance Test Suite")
    print("=" * 50)
    
    # Define performance test modules for each playbook (relative to tests directory)
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to tests directory
    test_modules = [
        ("CloudWatch Performance", os.path.join(base_dir, "performance/cloudwatch/test_cloudwatch_performance.py")),
        # Add other playbooks as they are organized
        # ("EC2 Performance", os.path.join(base_dir, "performance/ec2/test_ec2_performance.py")),
        # ("S3 Performance", os.path.join(base_dir, "performance/s3/test_s3_performance.py")),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name, test_path in test_modules:
        if not os.path.exists(test_path):
            print(f"⚠️  Skipping {test_name}: {test_path} not found")
            continue
            
        print(f"\n🔄 Running {test_name}...")
        
        try:
            # Load and run the test module
            spec = importlib.util.spec_from_file_location("test_module", test_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Run the main function if it exists
            if hasattr(test_module, 'main'):
                success = test_module.main()
                if success:
                    total_passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    total_failed += 1
                    print(f"❌ {test_name} FAILED")
            else:
                print(f"⚠️  {test_name}: No main() function found")
                
        except Exception as e:
            total_failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Performance Test Results: {total_passed + total_failed} total, {total_passed} passed, {total_failed} failed")
    
    success = total_failed == 0
    
    if success:
        print("🎉 ALL PERFORMANCE TESTS PASSED!")
    else:
        print(f"❌ {total_failed} PERFORMANCE TESTS FAILED")
    
    return success


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)