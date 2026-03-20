#!/usr/bin/env bash
#
# Deploy CFMTipsReadOnly role to all member accounts via CloudFormation StackSets.
#
# Prerequisites:
#   1. Run from the management account (or delegated StackSets admin)
#   2. AWS Organizations with all features enabled
#   3. Trusted access for StackSets enabled:
#      aws organizations enable-aws-service-access \
#        --service-principal member.org.stacksets.cloudformation.amazonaws.com
#   4. The CFM Tips Lambda must already be deployed (need its execution role ARN)
#
# Usage:
#   ./deploy-stackset.sh <lambda-role-arn> <ou-id-1>[,<ou-id-2>,...]
#
# Examples:
#   # Deploy to a specific OU
#   ./deploy-stackset.sh arn:aws:iam::123456789012:role/CFM-TIPs-Remote-MCP-CfmTipsMcpFunctionRole-xxx ou-abc1-12345678
#
#   # Deploy to the entire org (use root OU)
#   ./deploy-stackset.sh arn:aws:iam::123456789012:role/CFM-TIPs-Remote-MCP-CfmTipsMcpFunctionRole-xxx r-abc1
#
#   # Deploy to multiple OUs
#   ./deploy-stackset.sh arn:aws:iam::123456789012:role/CFM-TIPs-Remote-MCP-CfmTipsMcpFunctionRole-xxx ou-abc1-11111111,ou-abc1-22222222
#
# Scaling (1000+ accounts):
#   - MaxConcurrentPercentage: 25 → ~250 parallel deploys per batch
#   - Full rollout to 1000 accounts: ~4 batches, typically 5-15 minutes
#   - FailureTolerancePercentage: 10 → up to 100 accounts can fail without aborting
#   - AutoDeployment: new accounts joining the OU get the role automatically
#   - IAM is global: only deploys to us-east-1 (1 stack instance per account)
#
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <lambda-role-arn> <ou-ids-comma-separated> [delegated-admin-role-arn]"
  echo ""
  echo "  lambda-role-arn:    ARN of the CFM Tips Lambda execution role"
  echo "  ou-ids:             Comma-separated OU IDs (or root ID r-xxxx for entire org)"
  echo "  delegated-admin:    Optional ARN of CFMTipsReadOnly in delegated admin account"
  exit 1
fi

LAMBDA_ROLE_ARN="$1"
OU_IDS="$2"
DELEGATED_ADMIN_ARN="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== CFM Tips StackSet Deployment ==="
echo "Lambda Role:      ${LAMBDA_ROLE_ARN}"
echo "Target OUs:       ${OU_IDS}"
echo "Delegated Admin:  ${DELEGATED_ADMIN_ARN:-none}"
echo ""

# Check if trusted access is enabled
echo "Checking StackSets trusted access..."
TRUSTED=$(aws organizations list-aws-service-access-for-organization \
  --query "EnabledServicePrincipals[?ServicePrincipal=='member.org.stacksets.cloudformation.amazonaws.com'].ServicePrincipal" \
  --output text 2>/dev/null || echo "")

if [ -z "$TRUSTED" ]; then
  echo "Enabling trusted access for CloudFormation StackSets..."
  aws organizations enable-aws-service-access \
    --service-principal member.org.stacksets.cloudformation.amazonaws.com
  echo "Trusted access enabled."
else
  echo "Trusted access already enabled."
fi

# Check if CloudFormation organizations access is activated
echo "Checking CloudFormation organizations access..."
ORG_ACCESS=$(aws cloudformation describe-organizations-access --region us-east-1 \
  --query "Status" --output text 2>/dev/null || echo "DISABLED")

if [ "$ORG_ACCESS" != "ENABLED" ]; then
  echo "Activating CloudFormation organizations access..."
  aws cloudformation activate-organizations-access --region us-east-1
  echo "Organizations access activated."
else
  echo "Organizations access already active."
fi

echo ""
echo "Deploying StackSet..."
aws cloudformation deploy \
  --template-file "${SCRIPT_DIR}/stackset-deploy.yaml" \
  --stack-name CFM-Tips-StackSet-Deployer \
  --parameter-overrides \
    CfmTipsLambdaRoleArn="${LAMBDA_ROLE_ARN}" \
    TargetOUIds="${OU_IDS}" \
    DelegatedAdminRoleArn="${DELEGATED_ADMIN_ARN}" \
    CallAs=SELF \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --region us-east-1

echo ""
echo "=== Deployment initiated ==="
echo ""
echo "Monitor progress:"
echo "  aws cloudformation list-stack-instances \\"
echo "    --stack-set-name CFM-Tips-CrossAccount-Role \\"
echo "    --query 'Summaries[].{Account:Account,Status:StackInstanceStatus.DetailedStatus}' \\"
echo "    --output table"
echo ""
echo "Check for failures:"
echo "  aws cloudformation list-stack-instances \\"
echo "    --stack-set-name CFM-Tips-CrossAccount-Role \\"
echo "    --filters Name=DETAILED_STATUS,Values=FAILED \\"
echo "    --output table"
