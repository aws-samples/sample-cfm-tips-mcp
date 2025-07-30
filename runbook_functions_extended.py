# Extended EC2 runbook functions
import json
from typing import Dict, List, Any
from mcp.types import TextContent

from playbooks.ec2_optimization import (
    get_graviton_compatible_instances, get_burstable_instances_analysis,
    get_spot_instance_opportunities, get_unused_capacity_reservations,
    get_scheduling_opportunities, get_commitment_plan_recommendations,
    get_governance_violations, generate_comprehensive_ec2_report
)

async def identify_graviton_compatible_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_graviton_compatible_instances(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def analyze_burstable_instances(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_burstable_instances_analysis(
            region=arguments.get("region"),
            lookback_period_days=arguments.get("lookback_period_days", 14)
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_spot_opportunities(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_spot_instance_opportunities(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_unused_reservations(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_unused_capacity_reservations(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_scheduling_opportunities(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_scheduling_opportunities(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def analyze_commitment_plans(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_commitment_plan_recommendations(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def identify_governance_violations(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = get_governance_violations(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def generate_comprehensive_report(arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        result = generate_comprehensive_ec2_report(region=arguments.get("region"))
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]