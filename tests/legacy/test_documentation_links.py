#!/usr/bin/env python3
"""
Test script for documentation links functionality
"""

import json
from utils.documentation_links import add_documentation_links, get_service_documentation, format_documentation_section

def test_documentation_links():
    """Test the documentation links functionality"""
    
    print("Testing documentation links functionality...\n")
    
    # Test 1: Basic result with EC2 service
    print("1. Testing EC2 service documentation links:")
    ec2_result = {
        "status": "success",
        "data": {
            "underutilized_instances": [],
            "count": 0,
            "total_monthly_savings": 0
        },
        "message": "Found 0 underutilized EC2 instances"
    }
    
    enhanced_result = add_documentation_links(ec2_result, "ec2")
    print(json.dumps(enhanced_result, indent=2))
    print()
    
    # Test 2: S3 service documentation
    print("2. Testing S3 service documentation links:")
    s3_result = {
        "status": "success",
        "data": {
            "buckets_analyzed": 5,
            "total_savings": 150.50
        }
    }
    
    enhanced_s3_result = add_documentation_links(s3_result, "s3")
    print(json.dumps(enhanced_s3_result, indent=2))
    print()
    
    # Test 3: General documentation (no specific service)
    print("3. Testing general documentation links:")
    general_result = {
        "status": "success",
        "message": "Cost analysis completed"
    }
    
    enhanced_general_result = add_documentation_links(general_result)
    print(json.dumps(enhanced_general_result, indent=2))
    print()
    
    # Test 4: Get service-specific documentation
    print("4. Testing service-specific documentation retrieval:")
    rds_docs = get_service_documentation("rds")
    print("RDS Documentation:")
    for title, url in rds_docs.items():
        print(f"  - {title}: {url}")
    print()
    
    # Test 5: Format standalone documentation section
    print("5. Testing standalone documentation section:")
    lambda_docs = format_documentation_section("lambda")
    print(json.dumps(lambda_docs, indent=2))
    print()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_documentation_links()