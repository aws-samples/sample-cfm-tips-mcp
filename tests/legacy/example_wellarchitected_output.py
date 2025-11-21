#!/usr/bin/env python3
"""
Example showing enhanced tool outputs with Well-Architected Framework recommendations
"""

import json
from utils.documentation_links import add_documentation_links

def show_enhanced_examples():
    """Show examples of enhanced tool outputs with Well-Architected recommendations"""
    
    print("CFM Tips - Enhanced Output with Well-Architected Framework")
    print("=" * 70)
    print()
    
    # Example 1: EC2 Right-sizing with Well-Architected guidance
    print("1. EC2 Right-sizing Analysis - Enhanced Output:")
    print("-" * 50)
    ec2_result = {
        "status": "success",
        "data": {
            "underutilized_instances": [
                {
                    "instance_id": "i-1234567890abcdef0",
                    "instance_type": "m5.2xlarge",
                    "finding": "Overprovisioned",
                    "avg_cpu_utilization": 8.5,
                    "avg_memory_utilization": 12.3,
                    "recommendation": {
                        "recommended_instance_type": "m5.large",
                        "estimated_monthly_savings": 180.50,
                        "confidence": "High"
                    }
                },
                {
                    "instance_id": "i-0987654321fedcba0",
                    "instance_type": "c5.xlarge",
                    "finding": "Underprovisioned",
                    "avg_cpu_utilization": 85.2,
                    "recommendation": {
                        "recommended_instance_type": "c5.2xlarge",
                        "estimated_monthly_cost_increase": 120.00,
                        "performance_improvement": "40%"
                    }
                }
            ],
            "count": 2,
            "total_monthly_savings": 180.50,
            "analysis_period": "14 days",
            "data_source": "AWS Compute Optimizer"
        },
        "message": "Found 2 EC2 instances with optimization opportunities"
    }
    
    enhanced_ec2 = add_documentation_links(ec2_result, "ec2", "underutilized")
    
    # Show key sections
    print("Key Findings:")
    for instance in enhanced_ec2["data"]["underutilized_instances"]:
        print(f"  • {instance['instance_id']}: {instance['finding']} - Save ${instance['recommendation'].get('estimated_monthly_savings', 0)}/month")
    
    print(f"\nTotal Monthly Savings: ${enhanced_ec2['data']['total_monthly_savings']}")
    
    print("\nWell-Architected Framework Guidance:")
    wa_framework = enhanced_ec2["wellarchitected_framework"]
    print(f"  Cost Optimization Pillar: {wa_framework['cost_optimization_pillar']}")
    
    print("\n  Applicable Principles:")
    for principle in wa_framework["applicable_principles"]:
        print(f"    • {principle['title']}: {principle['description']}")
    
    print("\n  High Priority Recommendations:")
    for rec in wa_framework["implementation_priority"]["high"]:
        print(f"    • {rec}")
    
    print("\n  Service-Specific Best Practices:")
    for rec in wa_framework["service_specific_recommendations"][:2]:  # Show first 2
        print(f"    • {rec['practice']} ({rec['impact']})")
        print(f"      Implementation: {rec['implementation']}")
    
    print("\n" + "=" * 70)
    
    # Example 2: S3 Storage Optimization
    print("\n2. S3 Storage Optimization - Enhanced Output:")
    print("-" * 50)
    s3_result = {
        "status": "success",
        "comprehensive_s3_optimization": {
            "overview": {
                "total_potential_savings": "$2,450.75",
                "analyses_completed": "6/6",
                "buckets_analyzed": 25,
                "execution_time": "42.3s"
            },
            "key_findings": [
                "15 buckets using suboptimal storage classes",
                "Found 45 incomplete multipart uploads",
                "Identified $1,200 in lifecycle policy savings",
                "3 buckets with high request costs suitable for CloudFront"
            ],
            "top_recommendations": [
                {
                    "type": "storage_class_optimization",
                    "bucket": "analytics-data-lake",
                    "finding": "Standard storage for infrequently accessed data",
                    "recommendation": "Transition to Standard-IA after 30 days",
                    "potential_savings": "$850.25/month",
                    "priority": "High"
                },
                {
                    "type": "lifecycle_policy",
                    "bucket": "backup-archives",
                    "finding": "No lifecycle policy for old backups",
                    "recommendation": "Archive to Glacier Deep Archive after 90 days",
                    "potential_savings": "$650.50/month",
                    "priority": "High"
                }
            ]
        }
    }
    
    enhanced_s3 = add_documentation_links(s3_result, "s3", "storage_optimization")
    
    print("Key Findings:")
    for finding in enhanced_s3["comprehensive_s3_optimization"]["key_findings"]:
        print(f"  • {finding}")
    
    print(f"\nTotal Potential Savings: {enhanced_s3['comprehensive_s3_optimization']['overview']['total_potential_savings']}")
    
    print("\nTop Recommendations:")
    for rec in enhanced_s3["comprehensive_s3_optimization"]["top_recommendations"]:
        print(f"  • {rec['bucket']}: {rec['recommendation']} - {rec['potential_savings']}")
    
    print("\nWell-Architected Framework Guidance:")
    wa_s3 = enhanced_s3["wellarchitected_framework"]
    
    print("  High Priority Actions:")
    for action in wa_s3["implementation_priority"]["high"]:
        print(f"    • {action}")
    
    print("  Medium Priority Actions:")
    for action in wa_s3["implementation_priority"]["medium"][:2]:  # Show first 2
        print(f"    • {action}")
    
    print("\n" + "=" * 70)
    
    # Example 3: Multi-Service Comprehensive Analysis
    print("\n3. Multi-Service Comprehensive Analysis - Enhanced Output:")
    print("-" * 50)
    comprehensive_result = {
        "status": "success",
        "comprehensive_analysis": {
            "overview": {
                "total_monthly_cost": "$8,450.25",
                "total_potential_savings": "$2,180.75",
                "savings_percentage": "25.8%",
                "services_analyzed": ["EC2", "EBS", "RDS", "Lambda", "S3"]
            },
            "service_breakdown": {
                "ec2": {"current_cost": 3200, "potential_savings": 640, "optimization_opportunities": 12},
                "ebs": {"current_cost": 850, "potential_savings": 180, "optimization_opportunities": 8},
                "rds": {"current_cost": 2100, "potential_savings": 420, "optimization_opportunities": 3},
                "lambda": {"current_cost": 150, "potential_savings": 45, "optimization_opportunities": 15},
                "s3": {"current_cost": 2150, "potential_savings": 895, "optimization_opportunities": 22}
            },
            "top_opportunities": [
                {"service": "S3", "type": "Storage Class Optimization", "savings": 895, "effort": "Low"},
                {"service": "EC2", "type": "Right-sizing", "savings": 640, "effort": "Medium"},
                {"service": "RDS", "type": "Reserved Instances", "savings": 420, "effort": "Low"}
            ]
        }
    }
    
    enhanced_comprehensive = add_documentation_links(comprehensive_result, None, "comprehensive")
    
    print("Cost Overview:")
    overview = enhanced_comprehensive["comprehensive_analysis"]["overview"]
    print(f"  • Current Monthly Cost: ${overview['total_monthly_cost']}")
    print(f"  • Potential Savings: ${overview['total_potential_savings']} ({overview['savings_percentage']})")
    
    print("\nTop Optimization Opportunities:")
    for opp in enhanced_comprehensive["comprehensive_analysis"]["top_opportunities"]:
        print(f"  • {opp['service']} - {opp['type']}: ${opp['savings']}/month ({opp['effort']} effort)")
    
    print("\nWell-Architected Framework Principles:")
    wa_comp = enhanced_comprehensive["wellarchitected_framework"]
    for principle in wa_comp["principles"][:3]:  # Show first 3
        print(f"  • {principle['title']}")
        print(f"    {principle['description']}")
        print(f"    Key practices: {', '.join(principle['best_practices'][:2])}")
        print()
    
    print("=" * 70)
    print("\nEnhanced Features Summary:")
    print("✓ Documentation links to AWS best practices")
    print("✓ Well-Architected Framework Cost Optimization pillar mapping")
    print("✓ Service-specific implementation guidance")
    print("✓ Impact assessment and priority ranking")
    print("✓ Principle-based recommendations")
    print("✓ Actionable next steps with implementation details")

if __name__ == "__main__":
    show_enhanced_examples()