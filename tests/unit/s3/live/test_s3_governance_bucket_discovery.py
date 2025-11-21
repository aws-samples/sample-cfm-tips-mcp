"""
Live test to debug S3 governance check bucket discovery issue.

This test will help identify why s3_governance_check returns 0 buckets
when there are actually 40+ buckets in the account.
"""

import asyncio
import logging
import pytest
from typing import Dict, Any

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.mark.live
async def test_s3_bucket_discovery_debug():
    """
    Debug the S3 bucket discovery mechanism used by governance check.
    
    This test will step through the bucket discovery process to identify
    where the silent failure is occurring.
    """
    
    # Test 1: Direct S3Service bucket listing
    logger.info("=== Test 1: Direct S3Service bucket listing ===")
    try:
        from services.s3_service import S3Service
        
        s3_service = S3Service(region='us-east-1')
        logger.info(f"S3Service initialized: {s3_service}")
        
        # Test the list_buckets method directly
        buckets_result = await s3_service.list_buckets()
        logger.info(f"S3Service.list_buckets() result: {buckets_result}")
        
        if buckets_result.get("status") == "success":
            buckets = buckets_result.get("data", {}).get("Buckets", [])
            logger.info(f"Found {len(buckets)} buckets via S3Service")
            for i, bucket in enumerate(buckets[:5]):  # Show first 5
                logger.info(f"  Bucket {i+1}: {bucket.get('Name')} (Region: {bucket.get('Region', 'unknown')})")
        else:
            logger.error(f"S3Service.list_buckets() failed: {buckets_result}")
            
    except Exception as e:
        logger.error(f"Error in S3Service test: {str(e)}")
    
    # Test 2: Direct boto3 client call
    logger.info("\n=== Test 2: Direct boto3 client call ===")
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3', region_name='us-east-1')
        logger.info(f"Boto3 S3 client created: {s3_client}")
        
        # Direct list_buckets call
        response = s3_client.list_buckets()
        buckets = response.get('Buckets', [])
        logger.info(f"Found {len(buckets)} buckets via direct boto3 call")
        
        for i, bucket in enumerate(buckets[:5]):  # Show first 5
            logger.info(f"  Bucket {i+1}: {bucket.get('Name')} (Created: {bucket.get('CreationDate')})")
            
    except ClientError as e:
        logger.error(f"AWS ClientError in direct boto3 test: {e}")
    except Exception as e:
        logger.error(f"Error in direct boto3 test: {str(e)}")
    
    # Test 3: GovernanceAnalyzer bucket discovery
    logger.info("\n=== Test 3: GovernanceAnalyzer bucket discovery ===")
    try:
        from playbooks.s3.analyzers.governance_analyzer import GovernanceAnalyzer
        from services.s3_service import S3Service
        
        s3_service = S3Service(region='us-east-1')
        analyzer = GovernanceAnalyzer(s3_service=s3_service)
        logger.info(f"GovernanceAnalyzer initialized: {analyzer}")
        
        # Test the _get_buckets_to_analyze method
        context = {'region': 'us-east-1'}
        buckets_to_analyze = await analyzer._get_buckets_to_analyze(context)
        logger.info(f"GovernanceAnalyzer._get_buckets_to_analyze() returned: {len(buckets_to_analyze)} buckets")
        
        for i, bucket_name in enumerate(buckets_to_analyze[:5]):  # Show first 5
            logger.info(f"  Bucket {i+1}: {bucket_name}")
            
    except Exception as e:
        logger.error(f"Error in GovernanceAnalyzer test: {str(e)}")
    
    # Test 4: Full governance analysis
    logger.info("\n=== Test 4: Full governance analysis ===")
    try:
        from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator
        
        orchestrator = S3OptimizationOrchestrator(region='us-east-1')
        logger.info(f"S3OptimizationOrchestrator initialized: {orchestrator}")
        
        # Execute governance analysis
        result = await orchestrator.execute_analysis("governance", region='us-east-1')
        logger.info(f"Governance analysis result status: {result.get('status')}")
        logger.info(f"Total buckets analyzed: {result.get('data', {}).get('total_buckets_analyzed', 0)}")
        
        if result.get('status') == 'error':
            logger.error(f"Governance analysis error: {result.get('message')}")
        
    except Exception as e:
        logger.error(f"Error in full governance analysis test: {str(e)}")
    
    # Test 5: Check AWS credentials and permissions
    logger.info("\n=== Test 5: AWS credentials and permissions check ===")
    try:
        import boto3
        
        # Check STS identity
        sts_client = boto3.client('sts', region_name='us-east-1')
        identity = sts_client.get_caller_identity()
        logger.info(f"AWS Identity: {identity}")
        
        # Test S3 permissions
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Test list_buckets permission
        try:
            response = s3_client.list_buckets()
            logger.info(f"list_buckets permission: OK ({len(response.get('Buckets', []))} buckets)")
        except ClientError as e:
            logger.error(f"list_buckets permission: DENIED - {e}")
        
        # Test get_bucket_location permission on first bucket
        try:
            response = s3_client.list_buckets()
            if response.get('Buckets'):
                first_bucket = response['Buckets'][0]['Name']
                location = s3_client.get_bucket_location(Bucket=first_bucket)
                logger.info(f"get_bucket_location permission: OK (tested on {first_bucket})")
            else:
                logger.warning("No buckets to test get_bucket_location permission")
        except ClientError as e:
            logger.error(f"get_bucket_location permission: DENIED - {e}")
            
    except Exception as e:
        logger.error(f"Error in credentials/permissions test: {str(e)}")

if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_s3_bucket_discovery_debug())