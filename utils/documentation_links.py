"""
Documentation Links Utility

This module provides centralized documentation links for AWS cost optimization tools.
It adds relevant documentation and best practices links to tool outputs, including
AWS Well-Architected Framework recommendations.
"""

from typing import Dict, Any, List
# Removed wellarchitected_recommendations - let LLMs provide recommendations based on MCP output

# Documentation links mapping
DOCUMENTATION_LINKS = {
    "general": {
        "CFM-TIPs Guidance": "https://catalog.workshops.aws/awscff/en-US/introduction",
        "Cost Optimization Pillar of AWS Well Architected": "https://docs.aws.amazon.com/wellarchitected/latest/framework/cost-optimization.html"
    },
    "ec2": {
        "Best Practices Playbooks for EC2": "https://catalog.workshops.aws/awscff/en-US/playbooks/compute/ec2"
    },
    "ebs": {
        "Best Practices Playbooks for EBS": "https://catalog.workshops.aws/awscff/en-US/playbooks/storage/ebs"
    },
    "rds": {
        "Best Practices Playbooks for RDS": "https://catalog.workshops.aws/awscff/en-US/playbooks/databases/rds"
    },
    "lambda": {
        "Best Practices Playbooks for AWS Lambda": "https://catalog.workshops.aws/awscff/en-US/playbooks/compute/lambda"
    },
    "s3": {
        "Best Practices Playbooks for S3": "https://catalog.workshops.aws/awscff/en-US/playbooks/storage/s3"
    },
    "cloudtrail": {
        "Best Practices Playbooks for CloudTrail": "https://catalog.workshops.aws/awscff/en-US/playbooks/management-and-governance/cloudtrail"
    }
}

def add_documentation_links(result: Dict[str, Any], service_type: str = None, finding_type: str = None) -> Dict[str, Any]:
    """
    Add relevant documentation links and Well-Architected recommendations to a result dictionary.
    
    Args:
        result: The result dictionary from a cost optimization function
        service_type: The AWS service type (ec2, ebs, rds, lambda, s3, cloudtrail)
        finding_type: Type of optimization finding (underutilized, unused, overprovisioned, etc.)
    
    Returns:
        Enhanced result dictionary with documentation links and Well-Architected recommendations
    """
    if not isinstance(result, dict):
        return result
    
    # Create a copy to avoid modifying the original
    enhanced_result = result.copy()
    
    # Build documentation links
    docs = {}
    
    # Always include general documentation
    docs.update(DOCUMENTATION_LINKS["general"])
    
    # Add service-specific documentation if specified
    if service_type and service_type.lower() in DOCUMENTATION_LINKS:
        docs.update(DOCUMENTATION_LINKS[service_type.lower()])
    
    # Add documentation section to the result
    enhanced_result["documentation"] = {
        "description": "Suggested documentation and further reading",
        "links": docs
    }
    
    # Well-Architected recommendations now provided by LLMs analyzing MCP output
    
    return enhanced_result

def get_service_documentation(service_type: str) -> Dict[str, str]:
    """
    Get documentation links for a specific service.
    
    Args:
        service_type: The AWS service type
    
    Returns:
        Dictionary of documentation links
    """
    docs = DOCUMENTATION_LINKS["general"].copy()
    
    if service_type.lower() in DOCUMENTATION_LINKS:
        docs.update(DOCUMENTATION_LINKS[service_type.lower()])
    
    return docs

def format_documentation_section(service_type: str = None) -> Dict[str, Any]:
    """
    Format a standalone documentation section.
    
    Args:
        service_type: Optional service type for service-specific links
    
    Returns:
        Formatted documentation section
    """
    docs = DOCUMENTATION_LINKS["general"].copy()
    
    if service_type and service_type.lower() in DOCUMENTATION_LINKS:
        docs.update(DOCUMENTATION_LINKS[service_type.lower()])
    
    return {
        "documentation": {
            "description": "Suggested documentation and further reading",
            "links": docs
        }
    }