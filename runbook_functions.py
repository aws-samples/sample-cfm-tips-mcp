"""
Runbook Functions for AWS Cost Optimization

This module contains all the runbook/playbook functions for cost optimization analysis.
"""

import json
import logging
from typing import Dict, List, Any
from mcp.types import TextContent

# Import playbook modules
from playbooks.ec2_optimization import (
    get_underutilized_instances, get_right_sizing_recommendation, generate_right_sizing_report,
    get_stopped_instances, get_unattached_elastic_ips, get_old_generation_instances,
    get_instances_without_detailed_monitoring, get_graviton_compatible_instances,
    get_burstable_instances_analysis, get_spot_instance_opportunities,
    get_unused_capacity_reservations, get_scheduling_opportunities,
    get_commitment_plan_recommendations, get_governance_violations,
    generate_comprehensive_ec2_report
)
from playbooks.ebs_optimization import get_underutilized_volumes, identify_unused_volumes, generate_ebs_optimization_report
from playbooks.rds_optimization import get_underutilized_rds_instances, identify_idle_rds_instances
from playbooks.lambda_optimization import get_underutilized_lambda_functions, identify_unused_lambda_functions

logger = logging.getLogger(__name__)

# EC2 Right Sizing Runbook Functions
async def run_ec2_right_sizing_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive EC2 right-sizing analysis."""
    try:
        result = get_underutilized_instances(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14),
            cpu_threshold=arguments.get("cpu_threshold", 40.0),
            memory_threshold=arguments.get("memory_threshold"),
            network_threshold=arguments.get("network_threshold")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_ec2_right_sizing_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed EC2 right-sizing report."""
    try:
        # Get underutilized instances
        instances_result = get_underutilized_instances(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14),
            cpu_threshold=arguments.get("cpu_threshold", 40.0)
        )
        
        if instances_result["status"] != "success":
            return [TextContent(type="text", text=json.dumps(instances_result, indent=2))]
            
        # Generate report
        report_result = generate_right_sizing_report(
            instances_result["data"]["underutilized_instances"]
        )
        
        output_format = arguments.get("output_format", "json")
        if output_format == "markdown":
            # Convert to markdown format
            data = report_result["data"]
            report = f"""# EC2 Right Sizing Report

## Summary
- **Total Instances**: {data['total_instances']}
- **Monthly Savings**: ${data['total_monthly_savings']:.2f}

## Top Recommendations
"""
            for instance in data.get('top_recommendations', []):
                rec = instance.get('recommendation', {})
                report += f"""### {instance['instance_id']}
- **Current**: {rec.get('current_instance_type', 'N/A')}
- **Recommended**: {rec.get('recommended_instance_type', 'N/A')}
- **Monthly Savings**: ${rec.get('estimated_monthly_savings', 0):.2f}

"""
            return [TextContent(type="text", text=report)]
        else:
            return [TextContent(type="text", text=json.dumps(report_result, indent=2, default=str))]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# EBS Optimization Runbook Functions
async def run_ebs_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive EBS optimization analysis."""
    try:
        result = get_underutilized_volumes(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 30),
            iops_threshold=arguments.get("iops_threshold", 100.0),
            throughput_threshold=arguments.get("throughput_threshold", 1.0)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_ebs_volumes(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unused EBS volumes."""
    try:
        result = identify_unused_volumes(
            region=arguments.get("region"),
            min_age_days=arguments.get("min_age_days", 30)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_ebs_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed EBS optimization report."""
    try:
        region = arguments.get("region")
        
        # Get underutilized and unused volumes
        underutilized_result = get_underutilized_volumes(region=region)
        unused_result = identify_unused_volumes(region=region)
        
        if underutilized_result["status"] != "success" or unused_result["status"] != "success":
            return [TextContent(type="text", text="Error getting volume data")]
            
        # Generate comprehensive report
        report_result = generate_ebs_optimization_report(
            underutilized_result["data"]["underutilized_volumes"],
            unused_result["data"]["unused_volumes"]
        )
        
        output_format = arguments.get("output_format", "json")
        if output_format == "markdown":
            data = report_result["data"]
            report = f"""# EBS Optimization Report

## Summary
- **Total Volumes**: {data['total_volumes']}
- **Monthly Savings**: ${data['total_monthly_savings']:.2f}
- **Unused Savings**: ${data['unused_savings']:.2f}

## Top Unused Volumes
"""
            for volume in data.get('top_unused', []):
                report += f"""### {volume['volume_id']}
- **Size**: {volume.get('volume_size', 'N/A')} GB
- **Monthly Cost**: ${volume.get('monthly_cost', 0):.2f}

"""
            return [TextContent(type="text", text=report)]
        else:
            return [TextContent(type="text", text=json.dumps(report_result, indent=2, default=str))]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# RDS Optimization Runbook Functions
async def run_rds_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive RDS optimization analysis."""
    try:
        result = get_underutilized_rds_instances(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14),
            cpu_threshold=arguments.get("cpu_threshold", 40.0),
            connection_threshold=arguments.get("connection_threshold", 20.0)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_idle_rds_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify idle RDS instances."""
    try:
        from playbooks.rds_optimization import identify_idle_rds_instances as get_idle_rds
        result = get_idle_rds(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 7),
            connection_threshold=arguments.get("connection_threshold", 1.0)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_rds_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed RDS optimization report."""
    try:
        region = arguments.get("region")
        
        # Get data from playbooks
        from playbooks.rds_optimization import identify_idle_rds_instances as get_idle_rds
        underutilized_result = get_underutilized_rds_instances(region=region)
        idle_result = get_idle_rds(region=region)
        
        combined_report = {
            "status": "success",
            "report_type": "RDS Comprehensive Optimization Report",
            "region": region or "default",
            "optimization_analysis": underutilized_result,
            "idle_instances_analysis": idle_result,
            "summary": {
                "underutilized_instances": underutilized_result.get("data", {}).get("count", 0),
                "idle_instances": idle_result.get("data", {}).get("count", 0)
            }
        }
        
        return [TextContent(type="text", text=json.dumps(combined_report, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

# Lambda Optimization Runbook Functions
async def run_lambda_optimization_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive Lambda optimization analysis."""
    try:
        result = get_underutilized_lambda_functions(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14),
            memory_utilization_threshold=arguments.get("memory_utilization_threshold", 50.0),
            min_invocations=arguments.get("min_invocations", 100)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_lambda_functions(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unused Lambda functions."""
    try:
        from playbooks.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
        result = get_unused_lambda(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 30),
            max_invocations=arguments.get("max_invocations", 5)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_lambda_optimization_report(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate detailed Lambda optimization report."""
    try:
        region = arguments.get("region")
        
        # Get data from playbooks
        from playbooks.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
        optimization_result = get_underutilized_lambda_functions(region=region)
        unused_result = get_unused_lambda(region=region)
        
        combined_report = {
            "status": "success",
            "report_type": "Lambda Comprehensive Optimization Report",
            "region": region or "default",
            "optimization_analysis": optimization_result,
            "unused_functions_analysis": unused_result,
            "summary": {
                "functions_with_usage": optimization_result.get("data", {}).get("count", 0),
                "unused_functions": unused_result.get("data", {}).get("count", 0)
            }
        }
        
        return [TextContent(type="text", text=json.dumps(combined_report, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating report: {str(e)}")]

async def run_comprehensive_cost_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Run comprehensive cost analysis across all services."""
    try:
        region = arguments.get("region")
        services = arguments.get("services", ["ec2", "ebs", "rds", "lambda"])
        lookback_period_days = arguments.get("lookback_period_days", 14)
        
        comprehensive_report = {
            "status": "success",
            "report_type": "Comprehensive Cost Analysis Report",
            "region": region or "default",
            "services_analyzed": services,
            "analyses": {}
        }
        
        # Run analyses using playbook functions
        if "ec2" in services:
            try:
                comprehensive_report["analyses"]["ec2"] = get_underutilized_instances(
                    region=region, lookback_period_days=lookback_period_days
                )
            except Exception as e:
                comprehensive_report["analyses"]["ec2"] = {"error": str(e)}
        
        if "ebs" in services:
            try:
                comprehensive_report["analyses"]["ebs"] = {
                    "optimization": get_underutilized_volumes(region=region),
                    "unused_volumes": identify_unused_volumes(region=region)
                }
            except Exception as e:
                comprehensive_report["analyses"]["ebs"] = {"error": str(e)}
        
        if "rds" in services:
            try:
                from playbooks.rds_optimization import identify_idle_rds_instances as get_idle_rds
                comprehensive_report["analyses"]["rds"] = {
                    "optimization": get_underutilized_rds_instances(region=region),
                    "idle_instances": get_idle_rds(region=region)
                }
            except Exception as e:
                comprehensive_report["analyses"]["rds"] = {"error": str(e)}
        
        if "lambda" in services:
            try:
                from playbooks.lambda_optimization import identify_unused_lambda_functions as get_unused_lambda
                comprehensive_report["analyses"]["lambda"] = {
                    "optimization": get_underutilized_lambda_functions(region=region),
                    "unused_functions": get_unused_lambda(region=region)
                }
            except Exception as e:
                comprehensive_report["analyses"]["lambda"] = {"error": str(e)}
        
        return [TextContent(type="text", text=json.dumps(comprehensive_report, indent=2, default=str))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error running comprehensive analysis: {str(e)}")]

# Additional EC2 runbook functions
async def identify_stopped_ec2_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify stopped EC2 instances."""
    try:
        result = get_stopped_instances(
            region=arguments.get("region"),
            min_stopped_days=arguments.get("min_stopped_days", 7)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unattached_elastic_ips(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify unattached Elastic IPs."""
    try:
        result = get_unattached_elastic_ips(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_old_generation_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify old generation instances."""
    try:
        result = get_old_generation_instances(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_instances_without_monitoring(arguments: Dict[str, Any]) -> List[TextContent]:
    """Identify instances without detailed monitoring."""
    try:
        result = get_instances_without_detailed_monitoring(
            region=arguments.get("region")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
