#!/usr/bin/env python3
"""
Example showing how documentation links appear in tool outputs
"""

import json
from utils.documentation_links import add_documentation_links

def show_example_outputs():
    """Show examples of how documentation links appear in different tool outputs"""
    
    print("CFM Tips - Documentation Links Feature Examples")
    print("=" * 60)
    print()
    
    # Example 1: EC2 Right-sizing Analysis
    print("1. EC2 Right-sizing Analysis Output:")
    print("-" * 40)
    ec2_result = {
        "status": "success",
        "data": {
            "underutilized_instances": [
                {
                    "instance_id": "i-1234567890abcdef0",
                    "instance_type": "m5.large",
                    "finding": "Overprovisioned",
                    "recommendation": {
                        "recommended_instance_type": "m5.medium",
                        "estimated_monthly_savings": 45.50
                    }
                }
            ],
            "count": 1,
            "total_monthly_savings": 45.50
        },
        "message": "Found 1 underutilized EC2 instances via Compute Optimizer"
    }
    
    enhanced_ec2 = add_documentation_links(ec2_result, "ec2")
    print(json.dumps(enhanced_ec2, indent=2))
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: S3 Optimization Analysis
    print("2. S3 Optimization Analysis Output:")
    print("-" * 40)
    s3_result = {
        "status": "success",
        "comprehensive_s3_optimization": {
            "overview": {
                "total_potential_savings": "$1,250.75",
                "analyses_completed": "6/6",
                "failed_analyses": 0,
                "execution_time": "45.2s"
            },
            "key_findings": [
                "Found 15 buckets with suboptimal storage classes",
                "Identified $800 in potential lifecycle savings",
                "Discovered 25 incomplete multipart uploads"
            ],
            "top_recommendations": [
                {
                    "type": "storage_class_optimization",
                    "bucket": "my-data-bucket",
                    "potential_savings": "$450.25/month",
                    "action": "Transition to IA after 30 days"
                }
            ]
        }
    }
    
    enhanced_s3 = add_documentation_links(s3_result, "s3")
    print(json.dumps(enhanced_s3, indent=2))
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: RDS Optimization Analysis
    print("3. RDS Optimization Analysis Output:")
    print("-" * 40)
    rds_result = {
        "status": "success",
        "data": {
            "underutilized_instances": [
                {
                    "db_instance_identifier": "prod-database-1",
                    "db_instance_class": "db.r5.xlarge",
                    "finding": "Underutilized",
                    "avg_cpu_utilization": 15.5,
                    "recommendation": {
                        "recommended_instance_class": "db.r5.large",
                        "estimated_monthly_savings": 180.00
                    }
                }
            ],
            "count": 1,
            "total_monthly_savings": 180.00
        },
        "message": "Found 1 underutilized RDS instances"
    }
    
    enhanced_rds = add_documentation_links(rds_result, "rds")
    print(json.dumps(enhanced_rds, indent=2))
    print("\n" + "=" * 60 + "\n")
    
    # Example 4: Lambda Optimization Analysis
    print("4. Lambda Optimization Analysis Output:")
    print("-" * 40)
    lambda_result = {
        "status": "success",
        "data": {
            "overprovisioned_functions": [
                {
                    "function_name": "data-processor",
                    "current_memory": 1024,
                    "avg_memory_utilization": 35.2,
                    "recommendation": {
                        "recommended_memory": 512,
                        "estimated_monthly_savings": 25.75
                    }
                }
            ],
            "count": 1,
            "total_monthly_savings": 25.75
        },
        "message": "Found 1 overprovisioned Lambda functions"
    }
    
    enhanced_lambda = add_documentation_links(lambda_result, "lambda")
    print(json.dumps(enhanced_lambda, indent=2))
    print("\n" + "=" * 60 + "\n")
    
    # Example 5: General Cost Analysis (no specific service)
    print("5. General Cost Analysis Output:")
    print("-" * 40)
    general_result = {
        "status": "success",
        "data": {
            "total_monthly_cost": 5420.75,
            "potential_savings": 1250.50,
            "services_analyzed": ["EC2", "EBS", "RDS", "Lambda", "S3"],
            "optimization_opportunities": 47
        },
        "message": "Comprehensive cost analysis completed"
    }
    
    enhanced_general = add_documentation_links(general_result)
    print(json.dumps(enhanced_general, indent=2))
    print("\n" + "=" * 60 + "\n")
    
    print("Key Benefits of Documentation Links:")
    print("• Provides immediate access to AWS best practices")
    print("• Links to CFM-TIPs guidance and workshops")
    print("• References AWS Well-Architected Framework")
    print("• Service-specific playbooks for detailed guidance")
    print("• Consistent across all tool outputs")
    print("• Helps users understand optimization recommendations")

if __name__ == "__main__":
    show_example_outputs()