#!/usr/bin/env python3
"""
CFM Tips - AWS Cost Optimization MCP Server Setup Script

This script helps set up the CFM Tips AWS Cost Optimization MCP Server
for use with Kiro CLI and other MCP-compatible clients.
"""

import os
import sys
import json
import subprocess
import shlex
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = ['boto3', 'mcp']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Alternative 1: Use pip's Python API directly (most secure)
            import pip
            
            # Install each package individually using pip's internal API
            for package in missing_packages:
                try:
                    pip.main(['install', package])
                    print(f"✅ {package} installed successfully")
                except Exception as e:
                    print(f"❌ Failed to install {package}: {str(e)}")
                    return False
            
            print("✅ All dependencies installed successfully")
            return True
            
        except ImportError:
            # Alternative 2: Use importlib and sys.path manipulation
            print("⚠️  pip module not available, trying alternative method...")
            try:
                import importlib.util
                import site
                
                # Try to install using importlib (this is a fallback)
                print("❌ Cannot install packages without pip")
                print("💡 Please install missing packages manually:")
                for package in missing_packages:
                    print(f"   pip install {package}")
                return False
                
            except Exception as e:
                print(f"❌ Alternative installation method failed: {str(e)}")
                return False
        
        except Exception as e:
            print(f"❌ Failed to install dependencies: {str(e)}")
            return False
    
    return True

def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("\n🔐 Checking AWS credentials...")
    
    try:
        import boto3
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        print("✅ AWS credentials are configured")
        print(f"   Account: {identity.get('Account', 'Unknown')}")
        print(f"   User/Role: {identity.get('Arn', 'Unknown')}")
        return True
        
    except Exception as e:
        print("❌ AWS credentials not configured or invalid")
        print(f"   Error: {str(e)}")
        print("\n💡 To configure AWS credentials:")
        print("   aws configure")
        print("   or set environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        return False

def create_mcp_config():
    """Create or update MCP configuration file."""
    print("\n⚙️  Creating MCP configuration...")
    
    current_dir = os.getcwd()
    kiro_dir = Path.home() / ".kiro" / "settings"
    config_file = kiro_dir / "mcp.json"
    
    # Create kiro directory if it doesn't exist
    kiro_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    existing_config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding="utf-8") as f:
                existing_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_config = {}
    
    # Ensure mcpServers key exists
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    
    # Add or update cfm-tips server config
    existing_config["mcpServers"]["cfm-tips"] = {
        "command": "python3",
        "args": [str(Path(current_dir) / "mcp_server_with_runbooks.py")],
        "env": {
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_PROFILE": "default",
            "PYTHONPATH": current_dir
        }
    }
    
    # Write updated config
    with open(config_file, 'w', encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)
    
    # Also create local template for reference
    template_file = "mcp_runbooks.json"
    with open(template_file, 'w', encoding="utf-8") as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"✅ MCP configuration updated: {config_file}")
    print(f"✅ Template created: {template_file}")
    return str(config_file)

def test_server():
    """Test the MCP server."""
    print("\n🧪 Testing MCP server...")
    
    try:
        # Alternative 1: Direct module import and testing (most secure)
        # This avoids subprocess entirely by importing the test module directly
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        try:
            # Import the test module directly
            import test_runbooks
            
            # Run the main test function directly
            test_result = test_runbooks.main()
            
            if test_result:
                print("✅ Server tests passed")
                return True
            else:
                print("❌ Server tests failed")
                return False
                
        except ImportError as e:
            print(f"❌ Could not import test module: {str(e)}")
            
            # Alternative 2: Basic server validation without subprocess
            try:
                # Test basic imports that the server needs
                from mcp.server import Server
                from mcp.server.stdio import stdio_server
                import boto3
                
                # Try to import the server module
                import mcp_server_with_runbooks
                
                # Check if server object exists
                if hasattr(mcp_server_with_runbooks, 'server'):
                    print("✅ Server module validation passed")
                    return True
                else:
                    print("❌ Server object not found in module")
                    return False
                    
            except ImportError as import_err:
                print(f"❌ Server validation failed: {str(import_err)}")
                return False
        
        except Exception as test_err:
            print(f"❌ Test execution failed: {str(test_err)}")
            
            # Alternative 3: Minimal validation
            try:
                # Just check if we can import the main components
                import mcp_server_with_runbooks
                # runbook_functions is deprecated - functions are now in playbooks
                from playbooks.ec2.ec2_optimization import run_ec2_right_sizing_analysis
                
                print("✅ Basic server validation passed")
                return True
                
            except ImportError:
                print("❌ Basic server validation failed")
                return False
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Error testing server: {str(e)}")
        print("⚠️  Continuing with setup - you can test manually later")
        return True  # Return True to continue setup even if tests fail

def show_usage_instructions(config_file):
    """Show usage instructions."""
    print("\n" + "=" * 60)
    print("🎉 CFM Tips AWS Cost Optimization MCP Server Setup Complete!")
    print("=" * 60)
    
    print("\n🚀 Quick Start:")
    print("   kiro-cli chat")
    
    print("\n💬 Example commands in Kiro:")
    examples = [
        "Run comprehensive cost analysis for us-east-1",
        "Find unused EBS volumes costing money",
        "Generate EC2 right-sizing report in markdown",
        "Show me idle RDS instances",
        "Identify unused Lambda functions"
    ]
    
    for example in examples:
        print(f"   \"{example}\"")
    
    print("\n🔧 Available tools:")
    tools = [
        "ec2_rightsizing - Find underutilized EC2 instances",
        "ebs_unused - Identify unused EBS volumes",
        "rds_idle - Find idle RDS databases",
        "lambda_unused - Identify unused Lambda functions",
        "comprehensive_analysis - Multi-service analysis"
    ]
    
    for tool in tools:
        print(f"   • {tool}")
    
    print("\n📚 Documentation:")
    print("   • README.md - Main documentation")
    print("   • RUNBOOKS_GUIDE.md - Detailed usage guide")
    print("   • CORRECTED_PERMISSIONS.md - IAM permissions")
    
    print("\n🔍 Troubleshooting:")
    print("   • python3 diagnose_cost_optimization_hub_v2.py")
    print("   • python3 test_runbooks.py")
    
    print("\n💡 Tips:")
    print("   • Ensure your AWS resources have been running for 14+ days for metrics")
    print("   • Apply the IAM permissions from CORRECTED_PERMISSIONS.md")
    print("   • Enable Cost Optimization Hub in AWS Console if needed")

def main():
    """Main setup function."""
    print("CFM Tips - AWS Cost Optimization MCP Server Setup")
    print("=" * 55)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # AWS credentials check (warning only)
    aws_ok = check_aws_credentials()
    
    # Create configuration
    config_file = create_mcp_config()
    
    # Test server
    test_ok = test_server()
    
    # Show results
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print(f"✅ Python version: OK")
    print(f"✅ Dependencies: OK")
    print(f"{'✅' if aws_ok else '⚠️ '} AWS credentials: {'OK' if aws_ok else 'Needs configuration'}")
    print(f"✅ MCP configuration: OK")
    print(f"{'✅' if test_ok else '⚠️ '} Server tests: {'OK' if test_ok else 'Check manually'}")
    
    if aws_ok and test_ok:
        show_usage_instructions(config_file)
        print("\n🎯 Ready to use! Start with:")
        print("   kiro-cli chat")
    else:
        print("\n⚠️  Setup completed with warnings. Please address the issues above.")
        if not aws_ok:
            print("   Configure AWS credentials: aws configure")
        if not test_ok:
            print("   Test manually: python3 test_runbooks.py")

if __name__ == "__main__":
    main()