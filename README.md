# CFM Tips - Cost Optimization MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS cost analysis and optimization recommendations, designed to work seamlessly with Kiro CLI and other MCP-compatible clients.

## ✅ Features

### Core AWS Services Integration
- **Cost Explorer** - Retrieve cost data and usage metrics
- **Cost Optimization Hub** - Get AWS cost optimization recommendations
- **Compute Optimizer** - Right-sizing recommendations for compute resources
- **Trusted Advisor** - Cost optimization checks and recommendations
- **Performance Insights** - RDS performance metrics and analysis

### Cost Optimization Playbooks
- 🔧 **EC2 Right Sizing** - Identify underutilized EC2 instances with 12 specialized analysis tools
- 💾 **EBS Optimization** - Find unused and underutilized volumes
- 🗄️ **RDS Optimization** - Identify idle and underutilized databases
- ⚡ **Lambda Optimization** - Find overprovisioned and unused functions
- 🪣 **S3 Optimization** - Comprehensive S3 cost analysis and storage class optimization with 11 specialized tools
- 📋 **CloudTrail Optimization** - Analyze and optimize CloudTrail configurations
- 📊 **CloudWatch Optimization** - Optimize monitoring costs across logs, metrics, alarms, and dashboards
- 💰 **Database Savings Plans** - Analyze and optimize database commitment plans for Aurora, RDS, DynamoDB, and more
- 🌐 **NAT Gateway Optimization** - Identify underutilized, redundant, and unused NAT Gateways
- 📈 **Comprehensive Analysis** - Multi-service cost analysis

### Advanced Features
- **Real CloudWatch Metrics** - Uses actual AWS metrics for analysis
- **Multiple Output Formats** - JSON and Markdown report generation
- **Cost Calculations** - Estimated savings and cost breakdowns
- **Actionable Recommendations** - Priority-based optimization suggestions

## 📁 Project Structure

```
sample-cfm-tips-mcp/
├── playbooks/                            # CFM Tips optimization playbooks engine
│   ├── ec2/                             # EC2 optimization with 12 specialized tools
│   │   └── ec2_optimization.py          # EC2 right-sizing and comprehensive analysis
│   ├── ebs/                             # EBS volume optimization
│   │   └── ebs_optimization.py          # EBS volume optimization playbook
│   ├── rds/                             # RDS database optimization
│   │   ├── rds_optimization.py          # RDS database optimization playbook
│   │   └── database_savings_plans.py    # Database Savings Plans analysis
│   ├── aws_lambda/                      # Lambda function optimization
│   │   └── lambda_optimization.py       # Lambda function optimization playbook
│   ├── s3/                              # S3 storage optimization with 11 tools
│   │   └── s3_optimization_orchestrator.py # S3 cost optimization orchestrator
│   ├── cloudtrail/                      # CloudTrail optimization
│   │   └── cloudtrail_optimization.py   # CloudTrail optimization playbook
│   ├── cloudwatch/                      # CloudWatch optimization with 8 tools
│   │   └── cloudwatch_optimization.py   # CloudWatch cost optimization playbook
│   ├── nat_gateway/                     # NAT Gateway optimization
│   │   └── nat_gateway_optimization.py  # NAT Gateway optimization playbook
│   └── comprehensive_optimization.py    # Multi-service analysis
├── services/                             # AWS Services as datasources for the cost optimization
│   ├── s3_service.py                    # S3 API interactions and metrics
│   ├── s3_pricing.py                    # S3 pricing calculations and cost modeling
│   ├── cost_explorer.py                 # Cost Explorer API integration
│   ├── compute_optimizer.py             # Compute Optimizer API integration
│   └── optimization_hub.py              # Cost Optimization Hub integration
├── utils/                               # Cross-cutting utilities and analyzers
│   ├── analyzers/                       # Analysis engines for different optimization types
│   ├── logging_config.py                # Centralized logging configuration
│   ├── session_manager.py               # Session management for analysis results
│   └── parallel_executor.py             # Parallel execution utilities
├── mcp_server_with_runbooks.py           # Main MCP server with 50+ tools
├── runbook_functions.py                  # NAT Gateway and additional optimization functions
├── mcp_runbooks.json                     # Template file for MCP configuration file
├── requirements.txt                      # Python dependencies
├── tests/                               # Comprehensive test suite
├── diagnose_cost_optimization_hub_v2.py  # Diagnostic utilities
├── RUNBOOKS_GUIDE.md                     # Detailed usage guide
└── README.md                             # Project ReadMe
```

## 🔐 Security and Permissions - Least Privileges

### Security Best Practices

The CFM Tips MCP server follows AWS security best practices and requires only read-only permissions. Here are the key security principles:

#### Core Security Principles
- **Read-Only Access**: All operations are read-only - no create, update, or delete permissions
- **Least Privilege**: Grant only the minimum permissions required for cost analysis
- **No Resource Modification**: The tool cannot modify, terminate, or create AWS resources
- **Audit Trail**: All API calls are logged via CloudTrail for security monitoring
- **Credential Security**: Supports multiple secure credential methods (IAM roles, profiles, etc.)

#### Recommended Security Setup

**1. Create Dedicated IAM Role (Recommended)**
```bash
# Create a dedicated role for CFM Tips
aws iam create-role --role-name CFMTipsCostAnalysis --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR-ACCOUNT-ID:user/YOUR-USERNAME"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach the CFM Tips policies to the role
aws iam attach-role-policy --role-name CFMTipsCostAnalysis --policy-arn arn:aws:iam::YOUR-ACCOUNT-ID:policy/CFMTipsComprehensiveReadOnly
```

**2. Enable CloudTrail Monitoring**
```bash
# Ensure CloudTrail is enabled for API monitoring
aws cloudtrail describe-trails --query 'trailList[*].[Name,IsLogging]'

# Create CloudTrail if not exists
aws cloudtrail create-trail --name cfm-tips-audit --s3-bucket-name your-cloudtrail-bucket
aws cloudtrail start-logging --name cfm-tips-audit
```

**3. Use IAM Profiles (Alternative)**
```bash
# Create AWS CLI profile for CFM Tips
aws configure --profile cfm-tips
export AWS_PROFILE=cfm-tips
```

#### Security Validation

**Verify Read-Only Access**
```bash
# Test that no write operations are possible
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::YOUR-ACCOUNT-ID:role/CFMTipsCostAnalysis \
  --action-names ec2:TerminateInstances s3:DeleteBucket rds:DeleteDBInstance \
  --resource-arns "*"
# Should return "implicitDeny" for all write operations
```

**Monitor API Usage**
```bash
# Monitor CFM Tips API calls via CloudTrail
aws logs filter-log-events \
  --log-group-name CloudTrail/CFMTips \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --filter-pattern "{ $.userIdentity.type = \"AssumedRole\" && $.userIdentity.arn = \"*CFMTipsCostAnalysis*\" }"
```

#### Network Security

**VPC Endpoint Configuration (Optional)**
For enhanced security in private networks:
```bash
# Create VPC endpoints for AWS services (optional)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.cost-optimization-hub \
  --route-table-ids rtb-12345678
```

**Firewall Rules**
```bash
# Required outbound HTTPS access to AWS APIs
# Allow outbound to: *.amazonaws.com on port 443
# No inbound connections required
```

#### Credential Management

**Environment Variables (Development)**
```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

**IAM Role (Production - Recommended)**
```bash
# No credentials needed when running on EC2/ECS/Lambda with IAM role
# Role automatically provides temporary credentials
```

**AWS CLI Profile (Multi-Account)**
```bash
# ~/.aws/config
[profile cfm-tips-prod]
role_arn = arn:aws:iam::PROD-ACCOUNT:role/CFMTipsCostAnalysis
source_profile = default

[profile cfm-tips-dev]
role_arn = arn:aws:iam::DEV-ACCOUNT:role/CFMTipsCostAnalysis
source_profile = default
```

#### Service-Specific Security Notes

**Cost Optimization Hub**
- Requires enrollment in AWS Cost Optimization Hub
- No additional security considerations beyond IAM permissions

**Trusted Advisor**
- Requires Business or Enterprise support plan
- Some checks may require additional permissions

**Performance Insights**
- Must be enabled on RDS instances
- Provides database performance metrics (no sensitive data)

**S3 Analysis**
- Only accesses bucket metadata and configuration
- Does not read object contents or data
- Respects bucket policies and ACLs

#### Compliance Considerations

**Data Privacy**
- Tool analyzes only AWS service metadata and metrics
- No application data or user content is accessed
- All analysis is based on AWS service configurations and usage patterns

**Audit Requirements**
- All API calls are logged via CloudTrail
- Tool provides detailed logging of all operations
- Supports compliance reporting through AWS Config integration

**Multi-Account Security**
```bash
# For cross-account analysis, use cross-account roles
aws iam create-role --role-name CFMTipsCrossAccountRole --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::MANAGEMENT-ACCOUNT:role/CFMTipsCostAnalysis"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'
```

### IAM Policy Requirements

The CFM Tips MCP server requires comprehensive read-only permissions across multiple AWS services. Below are the complete IAM policies needed:

#### Core Cost Optimization Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CostAnalysisServices",
      "Effect": "Allow",
      "Action": [
        "cost-optimization-hub:ListEnrollmentStatuses",
        "cost-optimization-hub:ListRecommendations",
        "cost-optimization-hub:GetRecommendation", 
        "cost-optimization-hub:ListRecommendationSummaries",
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "ce:GetUsageReport",
        "ce:GetCostCategories",
        "compute-optimizer:GetEC2InstanceRecommendations",
        "compute-optimizer:GetEBSVolumeRecommendations",
        "compute-optimizer:GetLambdaFunctionRecommendations",
        "compute-optimizer:GetAutoScalingGroupRecommendations",
        "compute-optimizer:GetECSServiceRecommendations",
        "support:DescribeTrustedAdvisorChecks",
        "support:DescribeTrustedAdvisorCheckResult",
        "pricing:GetProducts",
        "pricing:DescribeServices",
        "pricing:GetAttributeValues"
      ],
      "Resource": "*"
    }
  ]
}
```

#### EC2 and Network Optimization Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2AndNetworkAnalysis",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeVolumes",
        "ec2:DescribeSnapshots",
        "ec2:DescribeImages",
        "ec2:DescribeAddresses",
        "ec2:DescribeNatGateways",
        "ec2:DescribeRouteTables",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeReservedInstances",
        "ec2:DescribeReservedInstancesOfferings",
        "ec2:DescribeCapacityReservations",
        "ec2:DescribeSpotInstanceRequests",
        "ec2:DescribeSpotPriceHistory",
        "ec2:GetEbsEncryptionByDefault",
        "ec2:GetEbsDefaultKmsKeyId"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Storage and Database Services Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "StorageAndDatabaseAnalysis",
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets",
        "s3:ListBucket",
        "s3:ListObjectsV2",
        "s3:GetBucketLocation",
        "s3:GetBucketVersioning",
        "s3:GetBucketLifecycleConfiguration",
        "s3:GetBucketNotification",
        "s3:GetBucketTagging",
        "s3:GetBucketPolicy",
        "s3:GetBucketAcl",
        "s3:GetBucketEncryption",
        "s3:ListMultipartUploads",
        "s3:GetStorageLensConfiguration",
        "s3:GetAnalyticsConfiguration",
        "s3:GetInventoryConfiguration",
        "rds:DescribeDBInstances",
        "rds:DescribeDBClusters",
        "rds:DescribeDBSnapshots",
        "rds:DescribeDBClusterSnapshots",
        "rds:DescribeReservedDBInstances",
        "rds:DescribeReservedDBInstancesOfferings",
        "pi:GetResourceMetrics",
        "pi:DescribeDimensionKeys",
        "pi:GetDimensionKeyDetails"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Lambda and CloudWatch Policy
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaAndMonitoringAnalysis",
      "Effect": "Allow",
      "Action": [
        "lambda:ListFunctions",
        "lambda:GetFunction",
        "lambda:GetFunctionConfiguration",
        "lambda:ListTags",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:GetMetricData",
        "cloudwatch:ListMetrics",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DescribeAlarmsForMetric",
        "cloudwatch:GetDashboard",
        "cloudwatch:ListDashboards",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams",
        "logs:DescribeMetricFilters",
        "logs:DescribeRetentionPolicy",
        "cloudtrail:DescribeTrails",
        "cloudtrail:GetTrailStatus",
        "cloudtrail:GetEventSelectors",
        "cloudtrail:LookupEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Consolidated Single Policy (Alternative)
For simplified management, you can use this single comprehensive policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CFMTipsComprehensiveReadOnly",
      "Effect": "Allow",
      "Action": [
        "cost-optimization-hub:*",
        "ce:Get*",
        "ce:List*",
        "compute-optimizer:Get*",
        "support:DescribeTrustedAdvisor*",
        "pricing:*",
        "ec2:Describe*",
        "ec2:Get*",
        "s3:List*",
        "s3:Get*",
        "rds:Describe*",
        "lambda:List*",
        "lambda:Get*",
        "cloudwatch:Get*",
        "cloudwatch:List*",
        "cloudwatch:Describe*",
        "logs:Describe*",
        "pi:Get*",
        "pi:Describe*",
        "cloudtrail:Describe*",
        "cloudtrail:Get*",
        "cloudtrail:LookupEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🛠️ Installation

### System Requirements

#### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.11+ recommended for best performance
- **Network**: Outbound HTTPS access to AWS APIs (*.amazonaws.com:443)


#### Python Dependencies
```bash
# Core dependencies (automatically installed)
boto3>=1.28.0          # AWS SDK
botocore>=1.31.0        # AWS core library
mcp>=0.1.0              # Model Context Protocol
asyncio                 # Async support (built-in)
json                    # JSON handling (built-in)
logging                 # Logging (built-in)
```

### Quick Start

#### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/aws-samples/sample-cfm-tips-mcp.git
cd sample-cfm-tips-mcp

# Install dependencies
pip install -r requirements.txt
```

#### 2. AWS Configuration
Choose one of the following methods:

**Option A: AWS CLI Configuration**
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and default region
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option C: IAM Role (for EC2/ECS/Lambda)**
```bash
# No additional configuration needed if running on AWS compute with IAM role
```

#### 3. Apply IAM Permissions
Create and attach the IAM policies from the Security section above to your AWS user or role.

#### 4. Install MCP Configuration
```bash
python3 setup.py
```


### Integration Options

#### Option 1: Kiro CLI Integration
```bash
# Start Kiro CLI chat
kiro-cli chat

# Example queries
"Show me cost optimization recommendations"
"Analyze my EC2 instances for right-sizing opportunities"
```

#### Option 2: Kiro IDE Integration
1. Open Kiro IDE or Kiro Developer Plugin
2. Navigate to Chat → 🛠️ Configure MCP Servers → ➕ Add new MCP
3. Use the following configuration:
   ```
   - Scope: Global
   - Name: cfm-tips
   - Transport: stdio
   - Command: python3
   - Arguments: /full/path/to/cfm-tips-mcp/mcp_server_with_runbooks.py
   - Timeout: 60
   ```


## 🔧 Available Tools

### Cost Analysis Tools
- `get_cost_explorer_data` - Retrieve AWS cost and usage data
- `list_coh_enrollment` - Check Cost Optimization Hub enrollment
- `get_coh_recommendations` - Get cost optimization recommendations
- `get_coh_summaries` - Get recommendation summaries
- `get_compute_optimizer_recommendations` - Get compute optimization recommendations

### EC2 Optimization
- `ec2_rightsizing` - Analyze EC2 instances for right-sizing opportunities
- `ec2_report` - Generate detailed EC2 optimization reports
- `ec2_stopped_instances` - Identify stopped instances that could be terminated
- `ec2_unattached_eips` - Identify unattached Elastic IP addresses
- `ec2_old_generation` - Identify old generation instances for upgrade
- `ec2_detailed_monitoring` - Find instances without detailed monitoring enabled
- `ec2_graviton_compatible` - Identify instances compatible with Graviton processors
- `ec2_burstable_analysis` - Analyze burstable instances for credit usage optimization
- `ec2_spot_opportunities` - Identify instances suitable for Spot pricing
- `ec2_unused_reservations` - Identify unused On-Demand Capacity Reservations
- `ec2_scheduling_opportunities` - Find instances suitable for scheduling optimization
- `ec2_commitment_plans` - Analyze Reserved Instance and Savings Plans opportunities
- `ec2_governance_violations` - Detect governance violations and policy non-compliance
- `ec2_comprehensive_report` - Generate comprehensive EC2 optimization reports

### EBS Optimization
- `ebs_optimization` - Analyze EBS volumes for optimization
- `ebs_unused` - Identify unused EBS volumes
- `ebs_report` - Generate EBS optimization reports

### RDS Optimization
- `rds_optimization` - Analyze RDS instances for optimization
- `rds_idle` - Identify idle RDS instances
- `rds_report` - Generate RDS optimization reports

### Lambda Optimization
- `lambda_optimization` - Analyze Lambda functions for optimization
- `lambda_unused` - Identify unused Lambda functions
- `lambda_report` - Generate Lambda optimization reports

### S3 Optimization
- `s3_general_spend_analysis` - Analyze overall S3 spending patterns and usage
- `s3_storage_class_selection` - Get guidance on choosing cost-effective storage classes
- `s3_storage_class_validation` - Validate existing data storage class appropriateness
- `s3_archive_optimization` - Identify and optimize long-term archive data storage
- `s3_api_cost_minimization` - Minimize S3 API request charges through optimization
- `s3_multipart_cleanup` - Identify and clean up incomplete multipart uploads
- `s3_governance_check` - Implement S3 cost controls and governance compliance
- `s3_comprehensive_analysis` - Run comprehensive S3 cost optimization analysis
- `s3_comprehensive_optimization_tool` - Unified comprehensive S3 optimization with parallel execution
- `s3_quick_analysis` - Fast 30-second analysis for spending overview and quick wins
- `s3_bucket_analysis` - Analyze specific S3 buckets for optimization opportunities

### CloudTrail Optimization
- `get_management_trails` - Get CloudTrail management trails
- `run_cloudtrail_trails_analysis` - Run CloudTrail trails analysis for optimization
- `generate_cloudtrail_report` - Generate CloudTrail optimization reports

### CloudWatch Optimization
- `cloudwatch_general_spend_analysis` - Analyze CloudWatch spending patterns across logs, metrics, alarms, and dashboards
- `cloudwatch_metrics_optimization` - Identify custom metrics cost optimization opportunities
- `cloudwatch_logs_optimization` - Optimize log retention and ingestion costs
- `cloudwatch_alarms_and_dashboards_optimization` - Improve monitoring efficiency and reduce alarm costs
- `cloudwatch_comprehensive_optimization_tool` - Run comprehensive CloudWatch analysis with intelligent orchestration
- `query_cloudwatch_analysis_results` - Query stored CloudWatch analysis results using SQL
- `validate_cloudwatch_cost_preferences` - Validate cost preferences and get functionality coverage estimates
- `get_cloudwatch_cost_estimate` - Get detailed cost estimates for CloudWatch optimization analysis

### Database Savings Plans
- `database_savings_plans_analysis` - Comprehensive analysis for Aurora, RDS, DynamoDB, ElastiCache, DocumentDB, Neptune, Keyspaces, Timestream, and DMS
- `database_savings_plans_purchase_analyzer` - Model custom commitment scenarios with user-specified hourly amounts
- `database_savings_plans_existing_analysis` - Analyze existing Database Savings Plans utilization and coverage

### NAT Gateway Optimization
- `nat_gateway_optimization` - Comprehensive NAT Gateway optimization analysis for underutilized, redundant, and unused gateways
- `nat_gateway_underutilized` - Identify underutilized NAT Gateways based on data transfer metrics
- `nat_gateway_redundant` - Find potentially redundant NAT Gateways in the same availability zone
- `nat_gateway_unused` - Identify NAT Gateways not referenced by any route tables

### Comprehensive Analysis
- `comprehensive_analysis` - Multi-service cost analysis

### Additional Tools
- `get_trusted_advisor_checks` - Get Trusted Advisor recommendations
- `get_performance_insights_metrics` - Get RDS Performance Insights data

## 📊 Example Usage

### Basic Cost Analysis
```
"Get my AWS costs for the last month"
"Show me cost optimization recommendations"
"What are my biggest cost drivers?"
```

### Resource Optimization
```
"Find underutilized EC2 instances in us-east-1"
"Show me unused EBS volumes that I can delete"
"Identify idle RDS databases"
"Find unused Lambda functions"
"Analyze my S3 storage costs and recommend optimizations"
"Find incomplete multipart uploads in my S3 buckets"
"Recommend the best S3 storage class for my data"
"Identify stopped EC2 instances I can terminate"
"Find unattached Elastic IP addresses"
"Show me old generation instances that need upgrading"
"Identify instances compatible with Graviton processors"
"Find opportunities for EC2 Spot pricing"
"Analyze my NAT Gateway utilization and costs"
"Identify redundant NAT Gateways in my VPC"
```

### CloudWatch and Monitoring Optimization
```
"Analyze my CloudWatch spending and identify cost optimization opportunities"
"Find expensive custom metrics I can optimize"
"Optimize my CloudWatch log retention policies"
"Identify unused or inefficient CloudWatch alarms"
"Run comprehensive CloudWatch cost analysis"
"Show me CloudWatch cost estimates for different optimization scenarios"
```

### Database Cost Optimization
```
"Analyze my database costs for Savings Plans opportunities"
"Model a $10/hour Database Savings Plans commitment"
"Review my existing Database Savings Plans utilization"
"Find cost optimization opportunities for Aurora and RDS instances"
"Analyze DynamoDB and ElastiCache costs for commitment plans"
```

### Report Generation
```
"Generate a comprehensive cost optimization report"
"Create an EC2 right-sizing report in PDF format"
"Generate an EBS optimization report with cost savings"
```

### Multi-Service Analysis
```
"Run comprehensive cost analysis for all services in us-east-1"
"Analyze my AWS infrastructure for cost optimization opportunities"
"Show me immediate cost savings opportunities"
"Generate a comprehensive S3 optimization report"
"Analyze my S3 spending patterns and storage class efficiency"
"Run quick analysis to identify top cost optimization opportunities"
"Perform comprehensive CloudWatch optimization analysis"
"Analyze my network costs and NAT Gateway efficiency"
"Generate comprehensive EC2 optimization report covering all playbooks"
```

## 🔍 Troubleshooting

### Installation Issues

#### Python Version Compatibility
```bash
# Check Python version
python3 --version

# If Python 3.11+ not available, install via package manager
# macOS with Homebrew:
brew install python@3.11

# Ubuntu/Debian:
sudo apt update && sudo apt install python3.11 python3.11-venv

# CentOS/RHEL:
sudo yum install python3.11
```

#### Dependency Installation Failures
```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Install with verbose output for debugging
pip install -r requirements.txt -v

# If specific packages fail, install individually
pip install boto3 botocore mcp

# For M1/M2 Macs with architecture issues:
pip install --no-binary :all: boto3
```

### AWS Configuration Issues

#### Credentials Not Found
```bash
# Verify AWS credentials are configured
aws sts get-caller-identity

# If credentials missing, configure them:
aws configure

# Or check environment variables:
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
echo $AWS_DEFAULT_REGION
```

#### Region Configuration Problems
```bash
# Check current region
aws configure get region

# Set region if not configured
aws configure set region us-east-1

# Or use environment variable
export AWS_DEFAULT_REGION=us-east-1
```

#### IAM Permission Errors
```bash
# Test specific permissions
aws cost-optimization-hub list-enrollment-statuses
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity MONTHLY --metrics BlendedCost

# Common permission error solutions:
# 1. Ensure IAM policies are attached to correct user/role
# 2. Wait 5-10 minutes for IAM changes to propagate
# 3. Check if Cost Optimization Hub is enabled in AWS Console
```

### Service-Specific Issues

#### Cost Optimization Hub Not Working
```bash
# Run diagnostic script
python3 diagnose_cost_optimization_hub_v2.py

# Enable Cost Optimization Hub in AWS Console:
# 1. Go to AWS Cost Management Console
# 2. Navigate to Cost Optimization Hub
# 3. Click "Get Started" and enable the service
```

#### No CloudWatch Metrics Found
```bash
# Verify CloudWatch is enabled and has data
aws cloudwatch list-metrics --namespace AWS/EC2

# Common issues:
# - Resources must run for 14+ days to have sufficient metrics
# - Detailed monitoring must be enabled for some metrics
# - Check correct region is being analyzed
```

#### S3 Analysis Failures
```bash
# Test S3 permissions
aws s3 ls

# Common S3 issues:
# - Bucket policies may restrict access
# - Cross-region bucket access requires proper permissions
# - Large buckets may timeout - use bucket-specific analysis
```

#### RDS Performance Insights Errors
```bash
# Verify Performance Insights is enabled
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,PerformanceInsightsEnabled]'

# Enable Performance Insights in RDS Console if needed
```

### MCP Integration Issues

#### Kiro IDE Connection Problems
```bash
# Test MCP server directly
python3 mcp_server_with_runbooks.py

# Check MCP configuration file
cat ~/.kiro/settings/mcp.json

# Verify correct path in Kiro configuration
which python3
pwd  # Note full path to mcp_server_with_runbooks.py
```

#### Kiro CLI Issues
```bash
# Verify Kiro CLI MCP configuration
cat ~/.kiro/settings/mcp.json
kiro-cli --version
```

### Performance Issues

#### Slow Analysis Performance
```bash
# Reduce analysis scope
# - Specify specific regions instead of all regions
# - Use quick analysis tools for initial assessment
# - Analyze specific resources instead of comprehensive scans

# Increase timeouts for large accounts
export CFM_TIPS_TIMEOUT=600

# Reduce parallel threads if hitting API limits
export CFM_TIPS_MAX_THREADS=2
```

#### API Rate Limiting
```bash
# AWS API throttling solutions:
# 1. Implement exponential backoff (built into boto3)
# 2. Reduce concurrent requests
# 3. Use pagination for large result sets
# 4. Consider AWS Support case for rate limit increases

# Check CloudTrail for throttling events
aws logs filter-log-events --log-group-name CloudTrail/APIGateway --filter-pattern "throttle"
```

### Common Error Messages

#### "NoCredentialsError"
```bash
# Solution: Configure AWS credentials
aws configure
# Or set environment variables as shown above
```

#### "AccessDenied" or "UnauthorizedOperation"
```bash
# Solution: Check IAM permissions
# 1. Verify policies are attached to correct user/role
# 2. Check policy syntax and permissions
# 3. Ensure services are enabled (Cost Optimization Hub, etc.)
```

#### "EndpointConnectionError"
```bash
# Solution: Check network connectivity and region
# 1. Verify internet connection
# 2. Check if region supports the service
# 3. Verify no proxy/firewall blocking AWS APIs
```

#### "ServiceNotAvailable" or "OptInRequired"
```bash
# Solution: Enable required AWS services
# 1. Cost Optimization Hub: Enable in AWS Console
# 2. Compute Optimizer: Opt-in via AWS Console
# 3. Trusted Advisor: Requires Business or Enterprise support plan
```

### Getting Additional Help

#### Enable Debug Logging
```bash
export CFM_TIPS_LOG_LEVEL=DEBUG
python3 mcp_server_with_runbooks.py

# Check log files
tail -f logs/cfm_tips_mcp.log
tail -f logs/cfm_tips_mcp_errors.log
```

#### Run Diagnostic Tools
```bash
# Comprehensive diagnostics
python3 diagnose_cost_optimization_hub_v2.py

# Test individual components
python3 -c "import boto3; print('Boto3 version:', boto3.__version__)"
python3 -c "from mcp.server import Server; print('MCP imported successfully')"
```

#### Contact Support
- Check the [RUNBOOKS_GUIDE.md](RUNBOOKS_GUIDE.md) for detailed usage instructions
- Review logs in the `logs/` directory
- Run integration tests: `python3 test_runbooks.py`
- Create GitHub issue with error logs and system information

## 🧩 Add-on MCPs
Add-on AWS Pricing MCP Server MCP server for accessing real-time AWS pricing information and providing cost analysis capabilities
https://github.com/awslabs/mcp/tree/main/src/aws-pricing-mcp-server

```bash
# Example usage with Add-on AWS Pricing MCP Server:
"Review the CDK by comparing it to the actual spend from my AWS account's stackset. Suggest cost optimization opportunities for the app accordingly"
```

## 🎯 Key Benefits

- **Immediate Cost Savings** - Identify unused resources for deletion
- **Right-Sizing Opportunities** - Optimize overprovisioned resources across EC2, RDS, and Lambda
- **Real Metrics Analysis** - Uses actual CloudWatch data for accurate analysis
- **Actionable Reports** - Clear recommendations with cost estimates and priority rankings
- **Comprehensive Coverage** - Analyze EC2, EBS, RDS, Lambda, S3, CloudWatch, NAT Gateways, and more
- **Advanced EC2 Optimization** - 12 specialized tools covering Graviton, Spot, governance, and more
- **Intelligent S3 Analysis** - 11 tools for storage class optimization, lifecycle policies, and cost control
- **CloudWatch Cost Control** - 8 tools for optimizing monitoring, logging, and alerting costs
- **Database Commitment Optimization** - Comprehensive Savings Plans analysis for all database services
- **Network Cost Optimization** - NAT Gateway analysis and redundancy elimination
- **Easy Integration** - Works seamlessly with Kiro CLI and Amazon Q

## 📈 Expected Results

The CFM Tips cost optimization server can help you:

- **Identify cost savings** on average across all AWS services
- **Find unused resources** costing hundreds of dollars monthly
- **Right-size overprovisioned instances** for optimal performance/cost ratio
- **Optimize storage costs** through volume type and storage class recommendations
- **Eliminate idle resources** that provide no business value
- **Reduce S3 costs by 30-60%** through intelligent storage class transitions
- **Clean up storage waste** from incomplete multipart uploads and orphaned data
- **Optimize API request patterns** to minimize S3 request charges
- **Reduce CloudWatch costs** through log retention optimization and metrics analysis
- **Optimize monitoring efficiency** by identifying unused alarms and dashboards
- **Maximize Database Savings Plans** utilization across Aurora, RDS, DynamoDB, and more
- **Eliminate network waste** by optimizing NAT Gateway usage and removing redundant gateways
- **Upgrade to modern instances** including Graviton processors for better price-performance
- **Leverage Spot pricing** for suitable workloads to reduce costs by up to 90%
- **Optimize Reserved Instances** and Savings Plans commitments for maximum savings

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
