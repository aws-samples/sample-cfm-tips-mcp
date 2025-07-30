#!/usr/bin/env python3
"""
Diagnostic script for Cost Optimization Hub issues - Updated with correct permissions
"""

import boto3
import json
from botocore.exceptions import ClientError

def check_cost_optimization_hub():
    """Check Cost Optimization Hub status and common issues."""
    print("🔍 Diagnosing Cost Optimization Hub Issues")
    print("=" * 50)
    
    # Test different regions where Cost Optimization Hub is available
    regions_to_test = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    
    for region in regions_to_test:
        print(f"\n📍 Testing region: {region}")
        try:
            client = boto3.client('cost-optimization-hub', region_name=region)
            
            # Test 1: Check enrollment statuses (correct API call)
            print("   ✅ Testing enrollment statuses...")
            try:
                enrollment_response = client.list_enrollment_statuses()
                print(f"   📊 Enrollment Response: {json.dumps(enrollment_response, indent=2, default=str)}")
                
                # Check if any accounts are enrolled
                items = enrollment_response.get('items', [])
                if items:
                    active_enrollments = [item for item in items if item.get('status') == 'Active']
                    if active_enrollments:
                        print(f"   ✅ Found {len(active_enrollments)} active enrollments")
                        
                        # Test 2: Try to list recommendations
                        print("   ✅ Testing list recommendations...")
                        try:
                            recommendations = client.list_recommendations(maxResults=5)
                            print(f"   📊 Found {len(recommendations.get('items', []))} recommendations")
                            print("   ✅ Cost Optimization Hub is working correctly!")
                            return True
                            
                        except ClientError as rec_error:
                            print(f"   ❌ Error listing recommendations: {rec_error.response['Error']['Code']} - {rec_error.response['Error']['Message']}")
                    else:
                        print("   ⚠️  No active enrollments found")
                        print("   💡 You need to enable Cost Optimization Hub in the AWS Console")
                else:
                    print("   ⚠️  No enrollment information found")
                    print("   💡 Cost Optimization Hub may not be set up for this account")
                    
            except ClientError as enrollment_error:
                error_code = enrollment_error.response['Error']['Code']
                error_message = enrollment_error.response['Error']['Message']
                
                if error_code == 'AccessDeniedException':
                    print("   ❌ Access denied - check IAM permissions")
                    print("   💡 Required permissions: cost-optimization-hub:ListEnrollmentStatuses")
                elif error_code == 'ValidationException':
                    print(f"   ❌ Validation error: {error_message}")
                else:
                    print(f"   ❌ Error: {error_code} - {error_message}")
                    
        except Exception as e:
            print(f"   ❌ Failed to create client for region {region}: {str(e)}")
    
    return False

def check_iam_permissions():
    """Check IAM permissions for Cost Optimization Hub."""
    print("\n🔐 Checking IAM Permissions")
    print("=" * 30)
    
    try:
        # Get current user/role
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        print(f"Current identity: {identity.get('Arn', 'Unknown')}")
        
        # Correct required actions for Cost Optimization Hub
        required_actions = [
            'cost-optimization-hub:ListEnrollmentStatuses',
            'cost-optimization-hub:UpdateEnrollmentStatus',
            'cost-optimization-hub:GetPreferences',
            'cost-optimization-hub:UpdatePreferences',
            'cost-optimization-hub:GetRecommendation',
            'cost-optimization-hub:ListRecommendations',
            'cost-optimization-hub:ListRecommendationSummaries'
        ]
        
        print("\nRequired permissions for Cost Optimization Hub:")
        for action in required_actions:
            print(f"  - {action}")
            
        print("\nMinimal permissions for read-only access:")
        minimal_actions = [
            'cost-optimization-hub:ListEnrollmentStatuses',
            'cost-optimization-hub:ListRecommendations',
            'cost-optimization-hub:GetRecommendation',
            'cost-optimization-hub:ListRecommendationSummaries'
        ]
        for action in minimal_actions:
            print(f"  - {action}")
            
    except Exception as e:
        print(f"Error checking IAM: {str(e)}")

def test_individual_apis():
    """Test individual Cost Optimization Hub APIs."""
    print("\n🧪 Testing Individual APIs")
    print("=" * 30)
    
    try:
        client = boto3.client('cost-optimization-hub', region_name='us-east-1')
        
        # Test 1: List Enrollment Statuses
        print("\n1. Testing list_enrollment_statuses...")
        try:
            response = client.list_enrollment_statuses()
            print(f"   ✅ Success: Found {len(response.get('items', []))} enrollment records")
        except ClientError as e:
            print(f"   ❌ Failed: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        
        # Test 2: List Recommendations
        print("\n2. Testing list_recommendations...")
        try:
            response = client.list_recommendations(maxResults=5)
            print(f"   ✅ Success: Found {len(response.get('items', []))} recommendations")
        except ClientError as e:
            print(f"   ❌ Failed: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        
        # Test 3: List Recommendation Summaries
        print("\n3. Testing list_recommendation_summaries...")
        try:
            response = client.list_recommendation_summaries(maxResults=5)
            print(f"   ✅ Success: Found {len(response.get('items', []))} summaries")
        except ClientError as e:
            print(f"   ❌ Failed: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
            
    except Exception as e:
        print(f"Error testing APIs: {str(e)}")

def provide_correct_iam_policy():
    """Provide the correct IAM policy for Cost Optimization Hub."""
    print("\n📋 Correct IAM Policy")
    print("=" * 25)
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "cost-optimization-hub:ListEnrollmentStatuses",
                    "cost-optimization-hub:UpdateEnrollmentStatus",
                    "cost-optimization-hub:GetPreferences",
                    "cost-optimization-hub:UpdatePreferences",
                    "cost-optimization-hub:GetRecommendation",
                    "cost-optimization-hub:ListRecommendations",
                    "cost-optimization-hub:ListRecommendationSummaries"
                ],
                "Resource": "*"
            }
        ]
    }
    
    print("Full IAM Policy:")
    print(json.dumps(policy, indent=2))
    
    minimal_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "cost-optimization-hub:ListEnrollmentStatuses",
                    "cost-optimization-hub:ListRecommendations",
                    "cost-optimization-hub:GetRecommendation",
                    "cost-optimization-hub:ListRecommendationSummaries"
                ],
                "Resource": "*"
            }
        ]
    }
    
    print("\nMinimal Read-Only Policy:")
    print(json.dumps(minimal_policy, indent=2))

def provide_solutions():
    """Provide solutions for common Cost Optimization Hub issues."""
    print("\n🛠️  Updated Solutions")
    print("=" * 20)
    
    solutions = [
        {
            "issue": "AccessDeniedException",
            "solution": [
                "1. Add the correct IAM permissions (see policy above)",
                "2. The service uses different permission names than other AWS services",
                "3. Use 'cost-optimization-hub:ListEnrollmentStatuses' not 'GetEnrollmentStatus'",
                "4. Attach the policy to your IAM user/role"
            ]
        },
        {
            "issue": "No enrollment found",
            "solution": [
                "1. Go to AWS Console → Cost Optimization Hub",
                "2. Enable the service for your account",
                "3. Wait for enrollment to complete",
                "4. URL: https://console.aws.amazon.com/cost-optimization-hub/"
            ]
        },
        {
            "issue": "Service not available",
            "solution": [
                "1. Cost Optimization Hub is only available in specific regions",
                "2. Use us-east-1, us-west-2, eu-west-1, or ap-southeast-1",
                "3. The service may not be available in your region yet"
            ]
        },
        {
            "issue": "No recommendations found",
            "solution": [
                "1. Cost Optimization Hub needs time to analyze your resources",
                "2. Ensure you have resources running for at least 14 days",
                "3. The service needs sufficient usage data to generate recommendations",
                "4. Check if you have any EC2, RDS, or other supported resources"
            ]
        }
    ]
    
    for solution in solutions:
        print(f"\n🔧 {solution['issue']}:")
        for step in solution['solution']:
            print(f"   {step}")

def main():
    """Main diagnostic function."""
    print("AWS Cost Optimization Hub Diagnostic Tool v2")
    print("=" * 50)
    
    try:
        # Run diagnostics
        hub_working = check_cost_optimization_hub()
        check_iam_permissions()
        test_individual_apis()
        provide_correct_iam_policy()
        provide_solutions()
        
        print("\n" + "=" * 60)
        if hub_working:
            print("✅ DIAGNOSIS: Cost Optimization Hub appears to be working!")
        else:
            print("❌ DIAGNOSIS: Cost Optimization Hub needs to be set up.")
            
        print("\n📝 Next Steps:")
        print("1. Apply the correct IAM policy shown above")
        print("2. Enable Cost Optimization Hub in the AWS Console if needed")
        print("3. Use the updated MCP server (mcp_server_fixed_v3.py)")
        print("4. Test with the enrollment status tool first")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {str(e)}")
        print("Please check your AWS credentials and try again.")

if __name__ == "__main__":
    main()
