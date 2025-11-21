# AWS Cost Optimization Runbooks with MCP v3

This guide shows how to use the AWS Cost Optimization Runbooks with the MCP server that includes proper Cost Optimization Hub permissions.

## What's Included

### Core AWS Services
- ✅ **Cost Explorer** - Retrieve cost data and usage metrics
- ✅ **Cost Optimization Hub** - With correct permissions and API calls
- ✅ **Compute Optimizer** - Get right-sizing recommendations
- ✅ **Trusted Advisor** - Cost optimization checks
- ✅ **Performance Insights** - RDS performance metrics

### Cost Optimization Runbooks
- 🔧 **EC2 Right Sizing** - Identify underutilized EC2 instances
- 💾 **EBS Optimization** - Find unused and underutilized volumes
- 🗄️ **RDS Optimization** - Identify idle and underutilized databases
- ⚡ **Lambda Optimization** - Find overprovisioned and unused functions
- 📋 **CloudTrail Optimization** - Identify duplicate management event trails
- 📊 **Comprehensive Analysis** - Multi-service cost analysis

## Quick Start

### 1. Setup
```bash
cd <replace-with-project-folder>/

# Make sure all files are executable
chmod +x mcp_server_with_runbooks.py

# Test the server
python3 -m py_compile mcp_server_with_runbooks.py
python3 -c "from playbooks.ec2.ec2_optimization import run_ec2_right_sizing_analysis; print('Playbooks OK')"
```

### 2. Configure AWS Permissions

Apply the correct IAM policy for Cost Optimization Hub:

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
        "support:DescribeTrustedAdvisorChecks",
        "support:DescribeTrustedAdvisorCheckResult",
        "pi:GetResourceMetrics",
        "cloudtrail:DescribeTrails",
        "cloudtrail:GetTrailStatus",
        "cloudtrail:GetEventSelectors"
      ],
      "Resource": "*"
    }
  ]
}
```

### 3. Install dependencies
pip install -r requirements_fixed.txt

### 4. Configure AWS credentials
aws configure

### 5.  Add the MCP server config to Amazon Q using the mcp_runbooks.json as a template
vi ~/.aws/amazonq/mcp.json

## Available Runbook Tools

### EC2 Right Sizing Runbooks

#### 1. `ec2_rightsizing`
Analyze EC2 instances for right-sizing opportunities.

**Example Usage:**
```
"Run EC2 right-sizing analysis for us-east-1 region with 14-day lookback period"
```

**Parameters:**
- `region`: AWS region to analyze
- `lookback_period_days`: Days to analyze (default: 14)
- `cpu_threshold`: CPU utilization threshold % (default: 40.0)

#### 2. `ec2_report`
Generate comprehensive EC2 right-sizing report.

**Example Usage:**
```
"Generate an EC2 right-sizing report for us-east-1 in markdown format"
```

### EBS Optimization Runbooks

#### 1. `ebs_optimization`
Analyze EBS volumes for optimization opportunities.

**Example Usage:**
```
"Analyze EBS volumes in us-east-1 for optimization opportunities"
```

#### 2. `ebs_unused`
Find unused EBS volumes that can be deleted.

**Example Usage:**
```
"Find unused EBS volumes older than 30 days in us-east-1"
```

#### 3. `ebs_report`
Generate comprehensive EBS optimization report.

**Example Usage:**
```
"Generate a comprehensive EBS optimization report for us-east-1"
```

### RDS Optimization Runbooks

#### 1. `rds_optimization`
Analyze RDS instances for optimization opportunities.

**Example Usage:**
```
"Analyze RDS instances in us-east-1 for underutilization"
```

#### 2. `rds_idle`
Find idle RDS instances with minimal activity.

**Example Usage:**
```
"Find idle RDS instances with less than 1 connection in the last 7 days"
```

#### 3. `rds_report`
Generate comprehensive RDS optimization report.

**Example Usage:**
```
"Generate an RDS optimization report for us-east-1"
```

### Lambda Optimization Runbooks

#### 1. `lambda_optimization`
Analyze Lambda functions for optimization opportunities.

**Example Usage:**
```
"Analyze Lambda functions in us-east-1 for memory optimization"
```

#### 2. `lambda_unused`
Find unused Lambda functions.

**Example Usage:**
```
"Find Lambda functions with less than 5 invocations in the last 30 days"
```

#### 3. `lambda_report`
Generate comprehensive Lambda optimization report.

**Example Usage:**
```
"Generate a Lambda optimization report for us-east-1"
```

### CloudTrail Optimization Runbooks

#### 1. `get_management_trails`
Get CloudTrail trails that have management events enabled.

**Example Usage:**
```
"Show me all CloudTrail trails with management events enabled in us-east-1"
```

#### 2. `run_cloudtrail_trails_analysis`
Analyze CloudTrail trails to identify duplicate management event trails.

**Example Usage:**
```
"Analyze CloudTrail trails in us-east-1 for cost optimization opportunities"
```

**Parameters:**
- `region`: AWS region to analyze

#### 3. `generate_cloudtrail_report`
Generate comprehensive CloudTrail optimization report.

**Example Usage:**
```
"Generate a CloudTrail optimization report for us-east-1 in markdown format"
```

**Parameters:**
- `region`: AWS region to analyze
- `format`: "json" or "markdown" (default: "json")

### Comprehensive Analysis

#### `comprehensive_analysis`
Run analysis across all services (EC2, EBS, RDS, Lambda).

**Example Usage:**
```
"Run comprehensive cost analysis for us-east-1 covering all services"
```

**Parameters:**
- `region`: AWS region to analyze
- `services`: Array of services ["ec2", "ebs", "rds", "lambda"]
- `lookback_period_days`: Days to analyze (default: 14)
- `output_format`: "json" or "markdown"

### Cost Optimization Hub Tools (Shortened)

#### 1. `list_coh_enrollment`
Check Cost Optimization Hub enrollment status.

#### 2. `get_coh_recommendations`
Get cost optimization recommendations.

#### 3. `get_coh_summaries`
Get recommendation summaries.

#### 4. `get_coh_recommendation`
Get specific recommendation by ID.

## Sample Conversation Flow
**Configure AWS credentials**
```aws configure```

**Add the MCP server config to Amazon Q using the mcp_runbooks.json as a template**
```vi ~/.aws/amazonq/mcp.json```

```bash
# Start Q with runbooks
q chat
```

**User:** "What cost optimization tools are available?"

**Q:** "I can see several AWS cost optimization tools including Cost Optimization Hub, runbooks for EC2, EBS, RDS, and Lambda optimization..."

**User:** "Run a comprehensive cost analysis for us-east-1"

**Q:** "I'll run a comprehensive cost analysis across all services for the us-east-1 region..."
*[Uses comprehensive_analysis tool]*

**User:** "Show me unused EBS volumes that are costing money"

**Q:** "Let me identify unused EBS volumes in your account..."
*[Uses ebs_unused tool]*

**User:** "Generate an EC2 right-sizing report in markdown format"

**Q:** "I'll generate a detailed EC2 right-sizing report in markdown format..."
*[Uses ec2_report tool]*

## Tool Names (For Reference)

The tool names have been shortened to fit MCP's 64-character limit:

| Purpose | Tool Name |
|----------|----------|
| `Run EC2 right sizing analysis` | `ec2_rightsizing` |
| `Generate EC2 right sizing report` | `ec2_report` |
| `Run EBS optimization analysis` | `ebs_optimization` |
| `Identify unused EBS volumes` | `ebs_unused` |
| `Generate EBS optimization report` | `ebs_report` |
| `Run RDS optimization analysis` | `rds_optimization` |
| `Iidentify idle RDS instances` | `rds_idle` |
| `Generate RDS optimization report` | `rds_report` |
| `Run Lambda optimization analysis` | `lambda_optimization` |
| `Identify unused Lambda functions` | `lambda_unused` |
| `Generate Lambda optimization report` | `lambda_report` |
| `Run comprehensive cost analysis` | `comprehensive_analysis` |
| `Get CloudTrail management trails` | `get_management_trails` |
| `Run CloudTrail trails analysis` | `run_cloudtrail_trails_analysis` |
| `Generate CloudTrail optimization report` | `generate_cloudtrail_report` |
| `List Cost Optimization Hub enrollment statuses` | `list_coh_enrollment` |
| `Get Cost Optimization Hub recommendations` | `get_coh_recommendations` |
| `Get Cost Optimization Hub recommendation summaries` | `get_coh_summaries` |
| `Get a particular Cost Optimization Hub recommendation` | `get_coh_recommendation` |

## Troubleshooting

### Common Issues

1. **Import Error for playbook functions**
   ```bash
   # Make sure PYTHONPATH is set in mcp_runbooks.json
   export PYTHONPATH="<replace-with-project-folder>"
   ```

2. **Cost Optimization Hub Errors**
   ```bash
   # Run the diagnostic first
   python3 diagnose_cost_optimization_hub_v2.py
   ```