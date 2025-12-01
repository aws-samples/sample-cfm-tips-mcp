# CFM Tips - Cost Optimization MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS cost analysis and optimization recommendations, designed to work seamlessly with Amazon Q CLI and other MCP-compatible clients.

## ✅ Features

### Core AWS Services Integration
- **Cost Explorer** - Retrieve cost data and usage metrics
- **Cost Optimization Hub** - Get AWS cost optimization recommendations
- **Compute Optimizer** - Right-sizing recommendations for compute resources
- **Trusted Advisor** - Cost optimization checks and recommendations
- **Performance Insights** - RDS performance metrics and analysis

### Cost Optimization Playbooks
- 🔧 **EC2 Right Sizing** - Identify underutilized EC2 instances
- 💾 **EBS Optimization** - Find unused and underutilized volumes
- 🗄️ **RDS Optimization** - Identify idle and underutilized databases
- ⚡ **Lambda Optimization** - Find overprovisioned and unused functions
- 🪣 **S3 Optimization** - Comprehensive S3 cost analysis and storage class optimization
- 📊 **CloudWatch Optimization** - Analyze logs, metrics, alarms, and dashboards for cost efficiency
- 📋 **CloudTrail Optimization** - Analyze and optimize CloudTrail configurations
- 📊 **Comprehensive Analysis** - Multi-service cost analysis

### Advanced Features
- **Real CloudWatch Metrics** - Uses actual AWS metrics for analysis
- **Multiple Output Formats** - JSON and Markdown report generation
- **Cost Calculations** - Estimated savings and cost breakdowns
- **Actionable Recommendations** - Priority-based optimization suggestions

## 📁 Project Structure

```
sample-cfm-tips-mcp/
├── playbooks/                            # CFM Tips optimization playbooks engine
│   ├── s3_optimization.py               # S3 cost optimization playbook
│   ├── ec2_optimization.py              # EC2 right-sizing playbook
│   ├── ebs_optimization.py              # EBS volume optimization playbook
│   ├── rds_optimization.py              # RDS database optimization playbook
│   ├── lambda_optimization.py           # Lambda function optimization playbook
│   ├── cloudwatch_optimization.py       # CloudWatch optimization playbook
│   └── cloudtrail_optimization.py       # CloudTrail optimization playbook
├── services/                             # AWS Services as datasources for the cost optimization
│   ├── s3_service.py                    # S3 API interactions and metrics
│   ├── s3_pricing.py                    # S3 pricing calculations and cost modeling
│   ├── cost_explorer.py                 # Cost Explorer API integration
│   ├── compute_optimizer.py             # Compute Optimizer API integration
│   └── optimization_hub.py              # Cost Optimization Hub integration
├── mcp_server_with_runbooks.py           # Main MCP server
├── mcp_runbooks.json                     # Template file for MCP configuration file
├── requirements.txt                      # Python dependencies
├── test/                                 # Integration tests
├── diagnose_cost_optimization_hub_v2.py  # Diagnostic utilities
├── RUNBOOKS_GUIDE.md                     # Detailed usage guide
└── README.md                             # Project ReadMe
```

## 🔐 Security and Permissions - Least Privileges

The MCP tools require specific AWS permissions to function. 
-  **Create a read-only IAM role** - Restricts LLM agents from modifying AWS resources. This prevents unintended create, update, or delete actions. 
-  **Enable CloudTrail** - Tracks API activity across your AWS account for security monitoring. 
-  **Follow least-privilege principles** - Grant only essential read permissions (Describe*, List*, Get*) for required services.

The below creates an IAM policy with for list, read and describe actions only:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cost-optimization-hub:ListEnrollmentStatuses",
        "cost-optimization-hub:ListRecommendations",
        "cost-optimization-hub:GetRecommendation", 
        "cost-optimization-hub:ListRecommendationSummaries",
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "compute-optimizer:GetEC2InstanceRecommendations",
        "compute-optimizer:GetEBSVolumeRecommendations",
        "compute-optimizer:GetLambdaFunctionRecommendations",
        "ec2:DescribeInstances",
        "ec2:DescribeVolumes",
        "rds:DescribeDBInstances",
        "lambda:ListFunctions",
        "cloudwatch:GetMetricStatistics",
        "s3:ListBucket",
        "s3:ListObjectsV2",
        "s3:GetBucketLocation",
        "s3:GetBucketVersioning",
        "s3:GetBucketLifecycleConfiguration",
        "s3:GetBucketNotification",
        "s3:GetBucketTagging",
        "s3:ListMultipartUploads",
        "s3:GetStorageLensConfiguration",
        "support:DescribeTrustedAdvisorChecks",
        "support:DescribeTrustedAdvisorCheckResult",
        "pi:GetResourceMetrics",
        "cloudtrail:DescribeTrails",
        "cloudtrail:GetTrailStatus",
        "cloudtrail:GetEventSelectors",
        "pricing:GetProducts",
        "pricing:DescribeServices",
        "pricing:GetAttributeValues"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🛠️ Installation

### Prerequisites
- **Python 3.11** or higher
- AWS CLI configured with appropriate credentials
- Amazon Kiro CLI (for MCP integration) - https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aws-samples/sample-cfm-tips-mcp.git
   cd sample-cfm-tips-mcp
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS Credentials**
   ```bash
   aws configure
   # Or set environment variables:
   # export AWS_ACCESS_KEY_ID=your_access_key
   # export AWS_SECRET_ACCESS_KEY=your_secret_key
   # export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Apply IAM Permissions**
   - Create an IAM policy with the permissions listed above
   - Attach the policy to your IAM user or role

5. **Install the MCP Configurations**
   ```bash
   python3 setup.py
   ```

6. **Usage Option 1: Using the Kiro CLI Chat**
   ```bash
   kiro-cli
   Show me cost optimization recommendations
   ```

7. **Usage Option 2: Integrate with Amazon Q Developer Plugin or Kiro**
   - Open Amazon Q Developer Plugin on your IDE
   - Click on Chat -> 🛠️ Configure MCP Servers -> ➕ Add new MCP
   - Use the following configuration
   ```bash
   - Scope: Global
   - Name: cfm-tips
   - Transport: stdio
   - Command: python3
   - Arguments: <replace-with-path-to-folder>/mcp_server_with_runbooks.py
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

### CloudWatch Optimization
- `cloudwatch_general_spend_analysis` - Analyze CloudWatch cost breakdown across logs, metrics, alarms, and dashboards
- `cloudwatch_metrics_optimization` - Identify custom metrics cost optimization opportunities
- `cloudwatch_logs_optimization` - Analyze log retention and ingestion cost optimization
- `cloudwatch_alarms_and_dashboards_optimization` - Identify monitoring efficiency improvements
- `cloudwatch_comprehensive_optimization_tool` - Run comprehensive CloudWatch optimization with intelligent orchestration
- `get_cloudwatch_cost_estimate` - Get detailed cost estimate for CloudWatch optimization analysis

### CloudTrail Optimization
- `get_management_trails` - Get CloudTrail management trails
- `run_cloudtrail_trails_analysis` - Run CloudTrail trails analysis for optimization
- `generate_cloudtrail_report` - Generate CloudTrail optimization reports

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
"Analyze my CloudWatch logs and metrics for cost optimization"
"Show me CloudWatch alarms that can be optimized"
```

### Report Generation
```
"Generate a comprehensive cost optimization report"
"Create an EC2 right-sizing report in markdown format"
"Generate an EBS optimization report with cost savings"
```

### Multi-Service Analysis
```
"Run comprehensive cost analysis for all services in us-east-1"
"Analyze my AWS infrastructure for cost optimization opportunities"
"Show me immediate cost savings opportunities"
"Generate a comprehensive S3 optimization report"
"Analyze my S3 spending patterns and storage class efficiency"
```

## 🔍 Troubleshooting

### Common Issues

1. **Cost Optimization Hub Not Working**
   ```bash
   python3 diagnose_cost_optimization_hub_v2.py
   ```

2. **No Metrics Found**
   - Ensure resources have been running for at least 14 days
   - Verify CloudWatch metrics are enabled
   - Check that you're analyzing the correct region

3. **Permission Errors**
   - Verify IAM permissions are correctly applied
   - Check AWS credentials configuration
   - Ensure Cost Optimization Hub is enabled in AWS Console

4. **Import Errors**
   ```bash
   # Check Python path and dependencies
   python3 -c "import boto3, mcp; print('Dependencies OK')"
   ```

### Getting Help

- Check the [RUNBOOKS_GUIDE.md](RUNBOOKS_GUIDE.md) for detailed usage instructions
- Run the diagnostic script: `python3 diagnose_cost_optimization_hub_v2.py`
- Run integration tests: `python3 test_runbooks.py`

## 🧩 Add-on MCPs
Add-on AWS Pricing MCP Server MCP server for accessing real-time AWS pricing information and providing cost analysis capabilities
https://github.com/awslabs/mcp/tree/main/src/aws-pricing-mcp-server

```bash
# Example usage with Add-on AWS Pricing MCP Server:
"Review the CDK by comparing it to the actual spend from my AWS account's stackset. Suggest cost optimization opportunities for the app accordingly"
```

## 🪣 S3 Optimization Features

The S3 optimization module provides comprehensive cost analysis and optimization recommendations:

### Storage Class Optimization
- **Intelligent Storage Class Selection** - Get recommendations for the most cost-effective storage class based on access patterns
- **Storage Class Validation** - Analyze existing data to ensure optimal storage class usage
- **Cost Breakeven Analysis** - Calculate when to transition between storage classes
- **Archive Optimization** - Identify long-term data suitable for Glacier or Deep Archive

### Cost Analysis & Monitoring
- **General Spend Analysis** - Comprehensive S3 spending pattern analysis over 12 months
- **Bucket-Level Cost Ranking** - Identify highest-cost buckets and optimization opportunities
- **Usage Type Breakdown** - Analyze costs by storage, requests, and data transfer
- **Regional Cost Distribution** - Understand spending across AWS regions

### Operational Optimization
- **Multipart Upload Cleanup** - Identify and eliminate incomplete multipart uploads
- **API Cost Minimization** - Optimize request patterns to reduce API charges
- **Governance Compliance** - Implement cost controls and policy compliance checking
- **Lifecycle Policy Recommendations** - Automated suggestions for lifecycle transitions

### Advanced Analytics
- **Real-Time Pricing Integration** - Uses AWS Price List API for accurate cost calculations
- **Trend Analysis** - Identify spending growth patterns and anomalies
- **Efficiency Metrics** - Calculate cost per GB and storage efficiency ratios
- **Comprehensive Reporting** - Generate detailed optimization reports in JSON or Markdown

## 🎯 Key Benefits

- **Immediate Cost Savings** - Identify unused resources for deletion
- **Right-Sizing Opportunities** - Optimize overprovisioned resources
- **Real Metrics Analysis** - Uses actual CloudWatch data
- **Actionable Reports** - Clear recommendations with cost estimates
- **Comprehensive Coverage** - Analyze EC2, EBS, RDS, Lambda, S3, and more
- **Easy Integration** - Works seamlessly with Amazon Q CLI

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
- **Reduce CloudWatch costs** through log retention and metrics optimization
- **Eliminate unused alarms and dashboards** reducing monitoring overhead

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
