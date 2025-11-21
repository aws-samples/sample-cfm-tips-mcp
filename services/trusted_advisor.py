"""
AWS Trusted Advisor service module.

This module provides functions for interacting with the AWS Trusted Advisor API.
"""

import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

from utils.error_handler import AWSErrorHandler, ResponseFormatter
from utils.aws_client_factory import get_trusted_advisor_client

logger = logging.getLogger(__name__)

def get_trusted_advisor_checks(
    check_categories: Optional[List[str]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get AWS Trusted Advisor check results.
    
    Args:
        check_categories: List of check categories to filter
        region: AWS region (optional)
        
    Returns:
        Dictionary containing the Trusted Advisor check results
    """
    try:
        # Trusted Advisor is only available in us-east-1
        support_client = get_trusted_advisor_client()
        
        # Get available checks
        checks_response = support_client.describe_trusted_advisor_checks(language='en')
        checks = checks_response['checks']
        
        # Filter by categories if specified
        if check_categories:
            checks = [check for check in checks if check['category'] in check_categories]
            
        # Get results for each check
        results = []
        for check in checks:
            # Ensure check is a dictionary
            if not isinstance(check, dict):
                logger.warning(f"Unexpected check format in Trusted Advisor response: {type(check)}")
                continue
                
            check_id = check.get('id')
            check_name = check.get('name', 'Unknown')
            
            if not check_id:
                logger.warning(f"Check missing ID: {check_name}")
                continue
                
            try:
                result = support_client.describe_trusted_advisor_check_result(
                    checkId=check_id,
                    language='en'
                )
                
                # Validate result structure
                if 'result' in result and isinstance(result['result'], dict):
                    results.append({
                        'check_id': check_id,
                        'name': check_name,
                        'category': check.get('category', 'unknown'),
                        'result': result['result']
                    })
                else:
                    logger.warning(f"Invalid result structure for check {check_name}")
                    
            except Exception as check_error:
                logger.warning(f"Error getting result for check {check_name}: {str(check_error)}")
                
        return ResponseFormatter.success_response(
            data={"checks": results, "count": len(results)},
            message=f"Retrieved {len(results)} Trusted Advisor check results",
            analysis_type="trusted_advisor_checks"
        )
        
    except ClientError as e:
        return AWSErrorHandler.format_client_error(
            e, 
            "get_trusted_advisor_checks",
            ["support:DescribeTrustedAdvisorChecks", "support:DescribeTrustedAdvisorCheckResult"]
        )
        
    except Exception as e:
        return AWSErrorHandler.format_general_error(e, "get_trusted_advisor_checks")