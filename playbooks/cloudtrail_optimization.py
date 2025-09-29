#!/usr/bin/env python3
"""
CloudTrail Optimization Playbook

This playbook checks for multiple management event trails in AWS CloudTrail,
which could represent a cost optimization opportunity.

Multiple trails capturing the same management events can lead to unnecessary costs.
"""

import boto3
import logging
from datetime import datetime
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CloudTrailOptimization:
    """
    CloudTrail optimization playbook to identify cost-saving opportunities
    related to duplicate management event trails.
    """
    
    def __init__(self, region=None):
        """
        Initialize the CloudTrail optimization playbook.
        
        Args:
            region (str, optional): AWS region to analyze. If None, uses the default region.
        """
        self.region = region
        self.client = boto3.client('cloudtrail', region_name=region) if region else boto3.client('cloudtrail')
        
    def analyze_trails(self):
        """
        Analyze CloudTrail trails to identify multiple management event trails.
        
        Returns:
            dict: Analysis results including optimization recommendations.
        """
        try:
            # Get all trails using pagination
            trails = []
            next_token = None
            
            while True:
                # Prepare pagination parameters
                params = {}
                if next_token:
                    params['NextToken'] = next_token
                    
                # Make the API call
                response = self.client.list_trails(**params)
                trails.extend(response.get('Trails', []))
                
                # Check if there are more results
                if 'NextToken' in response:
                    next_token = response['NextToken']
                else:
                    break
            
            # Get detailed information for each trail
            management_event_trails = []
            for trail in trails:
                trail_arn = trail.get('TrailARN')
                trail_name = trail.get('Name')
                
                # Get trail status and configuration
                trail_info = self.client.get_trail(Name=trail_arn)
                trail_status = self.client.get_trail_status(Name=trail_arn)
                
                # Check if the trail is logging management events
                event_selectors = self.client.get_event_selectors(TrailName=trail_arn)
                
                has_management_events = False
                for selector in event_selectors.get('EventSelectors', []):
                    # Only include if management events are explicitly enabled
                    if selector.get('IncludeManagementEvents') is True:
                        has_management_events = True
                        break
                
                # Only include trails that actually have management events enabled
                if has_management_events:
                    management_event_trails.append({
                        'name': trail_name,
                        'arn': trail_arn,
                        'is_multi_region': trail_info.get('Trail', {}).get('IsMultiRegionTrail', False),
                        'is_organization': trail_info.get('Trail', {}).get('IsOrganizationTrail', False),
                        'logging_enabled': trail_status.get('IsLogging', False),
                        'region': self.region or 'default'
                    })
            
            # Analyze results
            result = {
                'status': 'success',
                'analysis_type': 'CloudTrail Optimization',
                'timestamp': datetime.now().isoformat(),
                'region': self.region or 'default',
                'data': {
                    'total_trails': len(management_event_trails),
                    'management_event_trails': len(management_event_trails),
                    'trails_details': management_event_trails
                },
                'recommendations': []
            }
            
            # Generate recommendations based on findings
            if len(management_event_trails) > 1:
                # Multiple management event trails found - potential optimization opportunity
                estimated_savings = (len(management_event_trails) - 1) * 2  # $2 per trail per month after the first one
                
                result['message'] = f"Found {len(management_event_trails)} trails capturing management events. Consider consolidation."
                result['recommendations'] = [
                    "Consolidate multiple management event trails into a single trail to reduce costs",
                    f"Potential monthly savings: ${estimated_savings:.2f}",
                    "Ensure the consolidated trail captures all required events and regions",
                    "Consider using CloudTrail Lake for more cost-effective querying of events"
                ]
                result['optimization_opportunity'] = True
                result['estimated_monthly_savings'] = estimated_savings
            else:
                result['message'] = "No duplicate management event trails found."
                result['optimization_opportunity'] = False
                result['estimated_monthly_savings'] = 0
            
            return result
            
        except ClientError as e:
            logger.error(f"Error analyzing CloudTrail trails: {e}")
            return {
                'status': 'error',
                'message': f"Failed to analyze CloudTrail trails: {str(e)}",
                'error': str(e)
            }
    
    def generate_report(self, format='json'):
        """
        Generate a CloudTrail optimization report showing only trails with management events.
        
        Args:
            format (str): Output format ('json' or 'markdown')
            
        Returns:
            dict or str: Report in the specified format
        """
        analysis_result = self.analyze_trails()
        
        if format.lower() == 'markdown':
            # Generate markdown report
            md_report = f"# CloudTrail Optimization Report - Management Events Only\n\n"
            md_report += f"**Region**: {analysis_result.get('region', 'All regions')}\n"
            md_report += f"**Analysis Date**: {analysis_result.get('timestamp')}\n\n"
            
            # Only show trails with management events enabled
            management_trails = analysis_result.get('data', {}).get('trails_details', [])
            
            md_report += f"## Summary\n"
            md_report += f"- Trails with management events enabled: {len(management_trails)}\n"
            
            if analysis_result.get('optimization_opportunity', False):
                md_report += f"- Optimization opportunity: **YES**\n"
                md_report += f"- Estimated monthly savings: **${analysis_result.get('estimated_monthly_savings', 0):.2f}**\n"
            else:
                md_report += f"- Optimization opportunity: No\n"
            
            if management_trails:
                md_report += f"\n## Management Event Trails ({len(management_trails)})\n"
                for trail in management_trails:
                    md_report += f"\n### {trail.get('name')}\n"
                    md_report += f"- ARN: {trail.get('arn')}\n"
                    md_report += f"- Multi-region: {'Yes' if trail.get('is_multi_region') else 'No'}\n"
                    md_report += f"- Organization trail: {'Yes' if trail.get('is_organization') else 'No'}\n"
                    md_report += f"- Logging enabled: {'Yes' if trail.get('logging_enabled') else 'No'}\n"
                
                if len(management_trails) > 1:
                    md_report += f"\n## Recommendations\n"
                    for rec in analysis_result.get('recommendations', []):
                        md_report += f"- {rec}\n"
            else:
                md_report += f"\n## Management Event Trails\nNo trails with management events enabled found.\n"
            
            return md_report
        else:
            # Return JSON format with only management event trails
            filtered_result = analysis_result.copy()
            filtered_result['data']['trails_shown'] = 'management_events_only'
            return filtered_result


def run_cloudtrail_optimization(region=None):
    """
    Run the CloudTrail optimization playbook.
    
    Args:
        region (str, optional): AWS region to analyze
        
    Returns:
        dict: Analysis results
    """
    optimizer = CloudTrailOptimization(region=region)
    return optimizer.analyze_trails()


def generate_cloudtrail_report(region=None, format='json'):
    """
    Generate a CloudTrail optimization report.
    
    Args:
        region (str, optional): AWS region to analyze
        format (str): Output format ('json' or 'markdown')
        
    Returns:
        dict or str: Report in the specified format
    """
    optimizer = CloudTrailOptimization(region=region)
    return optimizer.generate_report(format=format)


def get_management_trails(region=None):
    """
    Get CloudTrail trails that have management events enabled.
    
    Args:
        region (str, optional): AWS region to analyze
        
    Returns:
        list: List of trails with management events enabled
    """
    try:
        client = boto3.client('cloudtrail', region_name=region) if region else boto3.client('cloudtrail')
        
        # Get all trails using pagination
        trails = []
        next_token = None
        
        while True:
            # Prepare pagination parameters
            params = {}
            if next_token:
                params['NextToken'] = next_token
                
            # Make the API call
            response = client.list_trails(**params)
            trails.extend(response.get('Trails', []))
            
            # Check if there are more results
            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break
        
        management_trails = []
        
        for trail in trails:
            trail_arn = trail.get('TrailARN')
            trail_name = trail.get('Name')
            
            try:
                # Get trail configuration
                trail_info = client.get_trail(Name=trail_arn)
                
                # Check event selectors to see if management events are enabled
                event_selectors = client.get_event_selectors(TrailName=trail_arn)
                
                has_management_events = False
                for selector in event_selectors.get('EventSelectors', []):
                    # Check if this selector explicitly includes management events
                    if selector.get('IncludeManagementEvents') is True:
                        has_management_events = True
                        break
                
                # Only include trails that actually have management events enabled
                if has_management_events:
                    management_trails.append({
                        'name': trail_name,
                        'arn': trail_arn,
                        'region': region or trail.get('HomeRegion', 'us-east-1'),
                        'is_multi_region': trail_info.get('Trail', {}).get('IsMultiRegionTrail', False),
                        'is_organization_trail': trail_info.get('Trail', {}).get('IsOrganizationTrail', False)
                    })
                    
            except ClientError as e:
                logger.warning(f"Could not get details for trail {trail_name}: {e}")
                continue
        print(management_trails)
        return management_trails
        
    except ClientError as e:
        logger.error(f"Error getting management trails: {e}")
        return []


if __name__ == "__main__":
    # Run the playbook directly if executed as a script
    result = run_cloudtrail_optimization()
    print(result)
