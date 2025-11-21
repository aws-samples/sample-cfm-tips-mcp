#!/usr/bin/env python3
"""
CloudTrail Optimization Playbook

This playbook checks for multiple management event trails in AWS CloudTrail,
which could represent a cost optimization opportunity.
Includes both core optimization functions and MCP runbook functions.

Multiple trails capturing the same management events can lead to unnecessary costs.
"""

import asyncio
import json
import boto3
import logging
import time
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Dict, List, Any, Optional
from mcp.types import TextContent

from utils.error_handler import ResponseFormatter, handle_aws_error
from utils.documentation_links import add_documentation_links
from utils.memory_manager import get_memory_manager
from utils.performance_monitor import get_performance_monitor

# Configure logging
logger = logging.getLogger(__name__)

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
        
        # Initialize performance optimization components
        self.memory_manager = get_memory_manager()
        self.performance_monitor = get_performance_monitor()
        
    def analyze_trails(self):
        """
        Analyze CloudTrail trails to identify multiple management event trails.
        
        Returns:
            dict: Analysis results including optimization recommendations.
        """
        # Start memory tracking
        memory_tracker = self.memory_manager.start_memory_tracking("cloudtrail_analysis")
        
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
            
            # Stop memory tracking and add stats to result
            memory_stats = self.memory_manager.stop_memory_tracking("cloudtrail_analysis")
            if memory_stats:
                result['memory_usage'] = memory_stats
            
            return result
            
        except ClientError as e:
            logger.error(f"Error analyzing CloudTrail trails: {e}")
            self.memory_manager.stop_memory_tracking("cloudtrail_analysis")
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
# MCP Runbook Functions
# These functions provide MCP-compatible interfaces for the CloudTrail optimization playbook

@handle_aws_error
async def get_management_trails_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get CloudTrail management trails."""
    start_time = time.time()
    
    try:
        region = arguments.get("region", "us-east-1")
        result = get_management_trails(region=region)
        
        # Add documentation links
        result = add_documentation_links(result, "cloudtrail")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=result,
                message=f"Retrieved {len(result.get('management_trails', []))} CloudTrail management trails",
                analysis_type="cloudtrail_management_trails",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error getting CloudTrail management trails: {str(e)}")
        raise


@handle_aws_error
async def run_cloudtrail_trails_analysis_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run CloudTrail trails analysis."""
    start_time = time.time()
    
    try:
        region = arguments.get("region", "us-east-1")
        result = run_cloudtrail_optimization(region=region)
        
        # Add documentation links
        result = add_documentation_links(result, "cloudtrail")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=result,
                message="CloudTrail trails analysis completed successfully",
                analysis_type="cloudtrail_analysis",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error in CloudTrail trails analysis: {str(e)}")
        raise


@handle_aws_error
async def generate_cloudtrail_report_mcp(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate CloudTrail optimization report."""
    start_time = time.time()
    
    try:
        region = arguments.get("region", "us-east-1")
        output_format = arguments.get("output_format", "json")
        
        result = generate_cloudtrail_report(region=region, format=output_format)
        
        # Add documentation links
        result = add_documentation_links(result, "cloudtrail")
        
        execution_time = time.time() - start_time
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.success_response(
                data=result,
                message="CloudTrail optimization report generated successfully",
                analysis_type="cloudtrail_report",
                execution_time=execution_time
            )
        )
        
    except Exception as e:
        logger.error(f"Error generating CloudTrail report: {str(e)}")
        raise