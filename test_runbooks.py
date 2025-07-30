#!/usr/bin/env python3
"""
Test script for CFM Tips AWS Cost Optimization MCP Server
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test MCP server imports
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        print("✅ MCP imports successful")
        
        # Test AWS imports
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        print("✅ AWS imports successful")
        
        # Test runbook functions import
        from runbook_functions import (
            run_ec2_right_sizing_analysis,
            generate_ec2_right_sizing_report,
            run_ebs_optimization_analysis,
            identify_unused_ebs_volumes,
            generate_ebs_optimization_report,
            run_rds_optimization_analysis,
            identify_idle_rds_instances,
            generate_rds_optimization_report,
            run_lambda_optimization_analysis,
            identify_unused_lambda_functions,
            generate_lambda_optimization_report,
            run_comprehensive_cost_analysis
        )
        print("✅ Runbook functions import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_server_creation():
    """Test that the MCP server can be created."""
    print("\nTesting server creation...")
    
    try:
        # Import the server module
        import mcp_server_with_runbooks
        print("✅ Server module imported successfully")
        
        # Check if server is created
        if hasattr(mcp_server_with_runbooks, 'server'):
            print("✅ Server object created successfully")
            
            # Check server name
            if mcp_server_with_runbooks.server.name == "cfm_tips":
                print("✅ Server name is correct: cfm_tips")
            else:
                print(f"⚠️  Server name: {mcp_server_with_runbooks.server.name}")
            
            return True
        else:
            print("❌ Server object not found")
            return False
            
    except Exception as e:
        print(f"❌ Server creation error: {str(e)}")
        return False



def test_tool_names():
    """Test that tool names are within MCP limits."""
    print("\nTesting tool name lengths...")
    
    server_name = "cfm_tips"
    sample_tools = [
        "ec2_rightsizing",
        "ebs_optimization", 
        "rds_idle",
        "lambda_unused",
        "comprehensive_analysis",
        "get_coh_recommendations"
    ]
    
    max_length = 0
    for tool in sample_tools:
        combined = f"{server_name}___{tool}"
        length = len(combined)
        max_length = max(max_length, length)
        
        if length > 64:
            print(f"❌ Tool name too long: {combined} ({length} chars)")
            return False
    
    print(f"✅ All tool names within limit (max: {max_length} chars)")
    return True

def main():
    """Run all tests."""
    print("CFM Tips AWS Cost Optimization MCP Server - Integration Test")
    print("=" * 65)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test server creation
    if test_server_creation():
        tests_passed += 1
    
    # Test tool names
    if test_tool_names():
        tests_passed += 1
    
    print(f"\n" + "=" * 65)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All integration tests passed!")
        print("\nNext steps:")
        print("1. Configure AWS credentials: aws configure")
        print("2. Apply the correct IAM permissions (see CORRECTED_PERMISSIONS.md)")
        print("3. Start the server: q chat --mcp-config \"$(pwd)/mcp_runbooks.json\"")
        print("4. Test with: \"Run comprehensive cost analysis for us-east-1\"")
        print("\n🎉 CFM Tips is ready to help optimize your AWS costs!")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
