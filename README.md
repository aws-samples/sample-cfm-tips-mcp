# CFM Tips - Cost Optimization MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS cost analysis and optimization recommendations, designed to work seamlessly with Amazon Q CLI and other MCP-compatible clients.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/aws-samples/sample-cfm-tips-mcp.git

cd sample-cfm-tips-mcp

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Add the MCP server config to Amazon Q using the mcp_runbooks.json as a template
vi ~/.aws/amazonq/mcp.json

# Example usage in Q chat:
"Run comprehensive cost analysis for us-east-1"
"Find unused EBS volumes costing money"
"Generate EC2 right-sizing report in markdown"
```

## 🧩 Add-on MCPs
Add-on AWS Pricing MCP Server MCP server for accessing real-time AWS pricing information and providing cost analysis capabilities
https://github.com/awslabs/mcp/tree/main/src/aws-pricing-mcp-server

```bash
# Example usage with Add-on AWS Pricing MCP Server:
"Review the CDK by comparing it to the actual spend from my AWS account's stackset. Suggest cost optimization opportunities for the app accordingly"
```

## ✅ Features

### Core AWS Services Integration
- **Cost Explorer** - Retrieve cost data and usage metrics
- **Cost Optimization Hub** - Get AWS cost optimization recommendations
- **Compute Optimizer** - Right-sizing recommendations for compute resources
- **Trusted Advisor** - Cost optimization checks and recommendations
- **Performance Insights** - RDS performance metrics and analysis
- **CUR Reports** - Cost and Usage Report analysis from S3

### Cost Optimization Playbooks
- 🔧 **EC2 Right Sizing** - Identify underutilized EC2 instances
- 💾 **EBS Optimization** - Find unused and underutilized volumes
- 🗄️ **RDS Optimization** - Identify idle and underutilized databases
- ⚡ **Lambda Optimization** - Find overprovisioned and unused functions
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
├── services/                             # AWS Services as datasources for the cost optimization
├── mcp_server_with_runbooks.py           # Main MCP server
├── runbook_functions.py                  # Cost optimization runbook implementations
├── mcp_runbooks.json                     # Template file for MCP configuration file
├── requirements.txt                      # Python dependencies
├── test_runbooks.py                      # Integration tests
├── diagnose_cost_optimization_hub_v2.py  # Diagnostic utilities
├── RUNBOOKS_GUIDE.md                     # Detailed usage guide
└── README.md                             # Project ReadMe
```

## 🔐 Required AWS Permissions

Your AWS credentials need these permissions. The MCP server uses your role to perform actions, hence proceed with least privilege access only. The below creates an IAM policy with for list, read and describe actions only:

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
        "s3:ListObjectsV2",
        "support:DescribeTrustedAdvisorChecks",
        "support:DescribeTrustedAdvisorCheckResult",
        "pi:GetResourceMetrics"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- AWS CLI configured with appropriate credentials
- Amazon Q CLI (for MCP integration) - https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html

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

5. **Test the Installation**
   ```bash
   python3 test_runbooks.py
   ```

6. **Start the MCP Server**
   ```bash
   python3 mcp_server_with_runbooks.py &
   ```

7. **Start the Q Chat**
   ```bash
   q chat
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

### Comprehensive Analysis
- `comprehensive_analysis` - Multi-service cost analysis

### Additional Tools
- `list_cur_reports` - List Cost and Usage Reports in S3
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

## 🎯 Key Benefits

- **Immediate Cost Savings** - Identify unused resources for deletion
- **Right-Sizing Opportunities** - Optimize overprovisioned resources
- **Real Metrics Analysis** - Uses actual CloudWatch data
- **Actionable Reports** - Clear recommendations with cost estimates
- **Comprehensive Coverage** - Analyze EC2, EBS, RDS, Lambda, and more
- **Easy Integration** - Works seamlessly with Amazon Q CLI

## 📈 Expected Results

The CFM Tips cost optimization server can help you:

- **Identify cost savings** on average
- **Find unused resources** costing hundreds of dollars monthly
- **Right-size overprovisioned instances** for optimal performance/cost ratio
- **Optimize storage costs** through volume type recommendations
- **Eliminate idle resources** that provide no business value

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
