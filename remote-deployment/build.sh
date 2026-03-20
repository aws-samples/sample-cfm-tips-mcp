#!/usr/bin/env bash
#
# Build and deploy CFM Tips Lambda MCP Server (Docker image on Graviton arm64).
#
# SAM builds the Docker image from the project root using the Dockerfile,
# pushes to ECR, and deploys via CloudFormation. No manual build step needed.
#
# Usage:
#   cd remote-deployment
#   sam build
#   sam deploy --stack-name CFM-TIPs-Remote-MCP \
#     --capabilities CAPABILITY_NAMED_IAM \
#     --resolve-s3 --resolve-image-repos \
#     --no-confirm-changeset --region us-east-1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building Docker image via SAM..."
sam build

echo ""
echo "Deploying to AWS..."
sam deploy \
  --stack-name CFM-TIPs-Remote-MCP \
  --capabilities CAPABILITY_NAMED_IAM \
  --resolve-s3 \
  --resolve-image-repos \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --region us-east-1
