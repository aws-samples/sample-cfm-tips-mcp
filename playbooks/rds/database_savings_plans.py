"""
Database Savings Plans Analysis Playbook

This module implements comprehensive Database Savings Plans analysis and recommendations
for AWS database services including Amazon Aurora, Amazon RDS, Amazon DynamoDB, 
Amazon ElastiCache, Amazon DocumentDB, Amazon Neptune, Amazon Keyspaces, 
Amazon Timestream, and AWS DMS.

Includes both core analysis functions and MCP runbook wrapper functions.
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ServiceUsage:
    """Usage data for a specific database service."""
    service_name: str
    total_spend: float
    average_hourly_spend: float
    instance_types: Dict[str, float] = field(default_factory=dict)
    regions: Dict[str, float] = field(default_factory=dict)


@dataclass
class DatabaseUsageData:
    """Represents database usage analysis results."""
    total_on_demand_spend: float
    average_hourly_spend: float
    lookback_period_days: int
    service_breakdown: Dict[str, ServiceUsage]
    region_breakdown: Dict[str, float]
    instance_family_breakdown: Dict[str, float]
    analysis_timestamp: datetime


@dataclass
class SavingsPlanRecommendation:
    """Represents a Database Savings Plan recommendation."""
    commitment_term: str  # "1_YEAR" or "3_YEAR"
    payment_option: str  # "ALL_UPFRONT", "PARTIAL_UPFRONT", "NO_UPFRONT"
    hourly_commitment: float
    estimated_annual_savings: float
    estimated_monthly_savings: float
    savings_percentage: float
    projected_coverage: float
    projected_utilization: float
    break_even_months: int
    confidence_level: str  # "high", "medium", "low"
    upfront_cost: float
    total_commitment_cost: float
    rationale: str


@dataclass
class PurchaseAnalyzerResult:
    """Results from purchase analyzer custom modeling."""
    hourly_commitment: float
    commitment_term: str
    payment_option: str
    projected_annual_cost: float
    projected_coverage: float
    projected_utilization: float
    estimated_annual_savings: float
    uncovered_on_demand_cost: float
    unused_commitment_cost: float


@dataclass
class CommitmentComparison:
    """Comparison between Savings Plans and Reserved Instances."""
    service: str
    instance_type: str
    savings_plan_cost: float
    ri_standard_cost: float
    ri_convertible_cost: float
    on_demand_cost: float
    recommended_option: str
    savings_plan_savings: float
    ri_standard_savings: float
    ri_convertible_savings: float
    flexibility_score: float  # 0-100, higher = more flexible
    rationale: str


@dataclass
class ExistingSavingsPlan:
    """Represents an existing Database Savings Plan."""
    savings_plan_id: str
    savings_plan_arn: str
    hourly_commitment: float
    commitment_term: str
    payment_option: str
    start_date: datetime
    end_date: datetime
    utilization_percentage: float
    coverage_percentage: float
    unused_commitment_hourly: float
    status: str


@dataclass
class MultiAccountUsageData:
    """Represents multi-account database usage analysis results."""
    organization_id: Optional[str]
    total_accounts: int
    account_level_usage: Dict[str, DatabaseUsageData]
    consolidated_usage: DatabaseUsageData
    shared_savings_potential: float
    cross_account_optimization_opportunities: List[Dict[str, Any]]
    analysis_timestamp: datetime


@dataclass
class MultiAccountRecommendation:
    """Represents multi-account Database Savings Plans recommendations."""
    organization_level: List[SavingsPlanRecommendation]
    account_level: Dict[str, List[SavingsPlanRecommendation]]
    shared_savings_plans: List[Dict[str, Any]]
    consolidated_savings: float
    individual_account_savings: Dict[str, float]
    optimization_strategy: str


# ============================================================================
# Core Analysis Functions
# ============================================================================

def analyze_database_usage(
    region: Optional[str] = None,
    lookback_period_days: int = 30,
    services: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze on-demand database usage across supported services.
    
    Args:
        region: AWS region to analyze (None for all regions)
        lookback_period_days: Number of days to analyze (30, 60, or 90)
        services: List of database services to analyze (None for all)
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Usage analysis results
            - message: Status message
            
    Example:
        >>> result = analyze_database_usage(region="us-east-1", lookback_period_days=30)
        >>> print(result["data"]["total_on_demand_spend"])
    """
    from datetime import datetime, timedelta
    from services.cost_explorer import get_database_usage_by_service
    
    logger.info(f"Starting database usage analysis: region={region}, "
                f"lookback_period={lookback_period_days}, services={services}")
    
    # Validate lookback period
    if lookback_period_days not in [30, 60, 90]:
        logger.error(f"Invalid lookback period: {lookback_period_days}")
        return {
            "status": "error",
            "message": f"Lookback period must be 30, 60, or 90 days",
            "error_code": "ValidationError",
            "provided_value": lookback_period_days,
            "valid_values": [30, 60, 90]
        }
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_period_days)
    
    # Format dates for Cost Explorer API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Querying database usage from {start_date_str} to {end_date_str}")
    
    # Retrieve database usage data from Cost Explorer
    usage_result = get_database_usage_by_service(
        start_date=start_date_str,
        end_date=end_date_str,
        services=services,
        region=region,
        granularity="DAILY"
    )
    
    # Check for errors
    if usage_result.get("status") == "error":
        logger.error(f"Failed to retrieve database usage: {usage_result.get('message')}")
        return usage_result
    
    # Extract data from Cost Explorer response
    usage_data = usage_result.get("data", {})
    total_on_demand_spend = usage_data.get("total_cost", 0.0)
    service_breakdown_raw = usage_data.get("service_breakdown", {})
    region_breakdown = usage_data.get("region_breakdown", {})
    instance_family_breakdown = usage_data.get("instance_family_breakdown", {})
    
    # Calculate average hourly spend
    total_hours = lookback_period_days * 24
    average_hourly_spend = total_on_demand_spend / total_hours if total_hours > 0 else 0.0
    
    # Build service breakdown with ServiceUsage objects
    service_breakdown = {}
    for service_name, service_cost in service_breakdown_raw.items():
        # Calculate average hourly spend for this service
        service_hourly_spend = service_cost / total_hours if total_hours > 0 else 0.0
        
        # Create ServiceUsage object
        service_usage = ServiceUsage(
            service_name=service_name,
            total_spend=service_cost,
            average_hourly_spend=service_hourly_spend,
            instance_types={},  # Will be populated from detailed data if needed
            regions={}  # Will be populated from detailed data if needed
        )
        
        service_breakdown[service_name] = service_usage
    
    # Create DatabaseUsageData object
    database_usage = DatabaseUsageData(
        total_on_demand_spend=total_on_demand_spend,
        average_hourly_spend=average_hourly_spend,
        lookback_period_days=lookback_period_days,
        service_breakdown=service_breakdown,
        region_breakdown=region_breakdown,
        instance_family_breakdown=instance_family_breakdown,
        analysis_timestamp=datetime.now()
    )
    
    logger.info(f"Database usage analysis complete: "
                f"total_spend=${total_on_demand_spend:.2f}, "
                f"avg_hourly=${average_hourly_spend:.2f}")
    
    # Return results in dictionary format for compatibility
    return {
        "status": "success",
        "data": {
            "total_on_demand_spend": database_usage.total_on_demand_spend,
            "average_hourly_spend": database_usage.average_hourly_spend,
            "lookback_period_days": database_usage.lookback_period_days,
            "service_breakdown": {
                name: {
                    "service_name": svc.service_name,
                    "total_spend": svc.total_spend,
                    "average_hourly_spend": svc.average_hourly_spend,
                    "instance_types": svc.instance_types,
                    "regions": svc.regions
                }
                for name, svc in database_usage.service_breakdown.items()
            },
            "region_breakdown": database_usage.region_breakdown,
            "instance_family_breakdown": database_usage.instance_family_breakdown,
            "analysis_timestamp": database_usage.analysis_timestamp.isoformat()
        },
        "message": f"Successfully analyzed database usage for {lookback_period_days} days"
    }


def generate_savings_plans_recommendations(
    usage_data: Dict[str, Any],
    commitment_terms: List[str] = None,
    payment_options: List[str] = None
) -> Dict[str, Any]:
    """
    Generate Database Savings Plans recommendations based on usage.
    
    Focus exclusively on latest-generation instance families (M7, R7, R8) eligible 
    for Database Savings Plans. Excludes older generation instances (M5, R5, R6g, T3, T4g) 
    that require Reserved Instances.
    
    Args:
        usage_data: Database usage analysis results
        commitment_terms: List of commitment terms to analyze (default: ["1_YEAR"])
        payment_options: List of payment options (default: ["NO_UPFRONT"])
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Recommendations list
            - message: Status message
            
    Example:
        >>> usage = analyze_database_usage()
        >>> recommendations = generate_savings_plans_recommendations(usage["data"])
        >>> for rec in recommendations["data"]["recommendations"]:
        ...     print(f"Commitment: ${rec['hourly_commitment']}/hour, "
        ...           f"Savings: ${rec['estimated_annual_savings']}/year")
    """
    from services.savings_plans_service import calculate_savings_plans_rates
    
    if commitment_terms is None:
        commitment_terms = ["1_YEAR"]  # Only 1-year terms currently supported
    if payment_options is None:
        # Only NO_UPFRONT supported for 1-year Database Savings Plans
        payment_options = ["NO_UPFRONT"]
    
    logger.info(f"Generating savings plans recommendations: "
                f"terms={commitment_terms}, payment_options={payment_options}")
    
    # Validate usage_data
    if not usage_data:
        logger.error("Usage data is empty or None")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Usage data is required",
            "guidance": "Call analyze_database_usage() first to get usage data"
        }
    
    # Filter for latest-generation instance families eligible for Database Savings Plans
    # Latest-generation families: M7, R7, R8, T4g (for some services), X2gd, etc.
    # Exclude older generations: M5, R5, R6g, T3, etc. that require Reserved Instances
    latest_generation_families = {
        # RDS/Aurora latest generation
        'db.m7g', 'db.m7i', 'db.r7g', 'db.r7i', 'db.r8g', 'db.t4g',
        'db.x2gd', 'db.x2iedn', 'db.x2iezn',
        # ElastiCache latest generation (Valkey engine only)
        'cache.m7g', 'cache.r7g', 'cache.t4g',
        # DynamoDB (serverless and on-demand are automatically eligible)
        # DocumentDB latest generation
        'docdb.r7g', 'docdb.t4g',
        # Neptune latest generation  
        'neptune.r7g', 'neptune.t4g'
    }
    
    # Older generation families that require Reserved Instances
    older_generation_families = {
        'db.m5', 'db.m5d', 'db.m5n', 'db.m5zn', 'db.m6g', 'db.m6gd', 'db.m6i', 'db.m6id', 'db.m6idn', 'db.m6in',
        'db.r5', 'db.r5b', 'db.r5d', 'db.r5dn', 'db.r5n', 'db.r6g', 'db.r6gd', 'db.r6i', 'db.r6id', 'db.r6idn', 'db.r6in',
        'db.t3', 'db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge', 'db.t3.2xlarge',
        'cache.m5', 'cache.m5n', 'cache.m6g', 'cache.m6gd', 'cache.m6i', 'cache.m6id', 'cache.m6idn', 'cache.m6in',
        'cache.r5', 'cache.r5n', 'cache.r6g', 'cache.r6gd', 'cache.r6i', 'cache.r6id', 'cache.r6idn', 'cache.r6in',
        'cache.t3'
    }
    
    # Extract instance family breakdown and filter for eligible families
    instance_family_breakdown = usage_data.get("instance_family_breakdown", {})
    eligible_spend = 0.0
    excluded_spend = 0.0
    
    for family, spend in instance_family_breakdown.items():
        # Check if this family is eligible for Database Savings Plans
        is_eligible = any(family.startswith(eligible_prefix) for eligible_prefix in latest_generation_families)
        is_excluded = any(family.startswith(excluded_prefix) for excluded_prefix in older_generation_families)
        
        if is_eligible and not is_excluded:
            eligible_spend += spend
            logger.debug(f"Including {family}: ${spend:.2f} (eligible for Database Savings Plans)")
        else:
            excluded_spend += spend
            logger.debug(f"Excluding {family}: ${spend:.2f} (requires Reserved Instances)")
    
    # If no instance family breakdown available, use total spend but log warning
    if not instance_family_breakdown:
        logger.warning("No instance family breakdown available - using total spend")
        # Use total spend as eligible since we can't determine instance families
        total_spend = usage_data.get("total_on_demand_spend", 0.0)
        eligible_spend = total_spend
    
    # Calculate eligible average hourly spend
    total_hours = usage_data.get("lookback_period_days", 30) * 24
    eligible_average_hourly_spend = eligible_spend / total_hours if total_hours > 0 else 0.0
    
    if eligible_average_hourly_spend <= 0:
        logger.warning("No eligible database usage found for Database Savings Plans")
        message = "No eligible database usage found for Database Savings Plans"
        if excluded_spend > 0:
            excluded_hourly = excluded_spend / total_hours if total_hours > 0 else 0.0
            message += f" (${excluded_hourly:.2f}/hour in older generation instances requires Reserved Instances)"
        
        return {
            "status": "info",
            "data": {
                "recommendations": [],
                "eligible_hourly_spend": eligible_average_hourly_spend,
                "excluded_hourly_spend": excluded_spend / total_hours if total_hours > 0 else 0.0,
                "eligible_families": list(latest_generation_families),
                "excluded_families": list(older_generation_families)
            },
            "message": message,
            "guidance": "Consider upgrading to latest-generation instances (M7, R7, R8) to benefit from Database Savings Plans"
        }
    
    logger.info(f"Eligible average hourly spend: ${eligible_average_hourly_spend:.2f}/hour "
                f"(excluded: ${excluded_spend / total_hours if total_hours > 0 else 0.0:.2f}/hour)")
    
    # Generate recommendations for each combination of term and payment option
    recommendations = []
    
    for commitment_term in commitment_terms:
        # Database Savings Plans currently only support 1-year terms with No Upfront payment
        if commitment_term == "1_YEAR":
            valid_payment_options = ["NO_UPFRONT"]
        else:
            # Skip unsupported terms
            continue
        
        for payment_option in valid_payment_options:
            # Calculate optimal hourly commitment targeting 85-90% coverage
            # Strategy: Recommend commitment that covers eligible usage while minimizing unused commitment
            # Account for inability to resell or exchange unused portions
            
            # Use conservative coverage targets to account for dynamic usage patterns
            # Database Savings Plans best practice: aim for 85-90% coverage with 10-15% buffer
            # Higher buffer for commitment risk since Database Savings Plans cannot be resold or exchanged
            if commitment_term == "1_YEAR":
                # Conservative for 1-year (85% coverage with 15% buffer for commitment risk)
                coverage_target = 0.85
            # Note: 3-year terms not currently available for Database Savings Plans
            # All recommendations use 1-year term with 85% coverage target
            
            # Calculate recommended hourly commitment based on eligible spend only
            hourly_commitment = eligible_average_hourly_spend * coverage_target
            
            # Get savings rates from service (use eligible spend for rate calculation)
            rates_result = calculate_savings_plans_rates(
                on_demand_rate=eligible_average_hourly_spend,
                commitment_term=commitment_term,
                payment_option=payment_option,
                service_code="AmazonRDS"  # Generic database service
            )
            
            if rates_result.get("status") != "success":
                logger.error(f"Failed to calculate rates for {commitment_term} {payment_option}")
                continue
            
            rates_data = rates_result.get("data", {})
            
            # Calculate savings based on commitment
            discount_percentage = rates_data.get("discount_percentage", 0.0) / 100.0
            hourly_savings = hourly_commitment * discount_percentage
            
            # Calculate annual metrics
            hours_per_year = 8760
            estimated_annual_savings = hourly_savings * hours_per_year
            estimated_monthly_savings = estimated_annual_savings / 12
            
            # Calculate total commitment cost
            if payment_option == "ALL_UPFRONT":
                upfront_cost = hourly_commitment * hours_per_year * (1 - discount_percentage)
                total_commitment_cost = upfront_cost
            elif payment_option == "PARTIAL_UPFRONT":
                upfront_cost = hourly_commitment * hours_per_year * (1 - discount_percentage) * 0.5
                recurring_cost = upfront_cost  # Other half paid over time
                total_commitment_cost = upfront_cost + recurring_cost
            else:  # NO_UPFRONT
                upfront_cost = 0.0
                total_commitment_cost = hourly_commitment * hours_per_year * (1 - discount_percentage)
            
            # Calculate projected coverage and utilization based on eligible spend
            projected_coverage = (hourly_commitment / eligible_average_hourly_spend) * 100.0
            # Calculate utilization based on actual commitment usage
            projected_utilization = min((eligible_average_hourly_spend / hourly_commitment) * 100.0, 100.0)
            
            # Calculate savings percentage based on eligible spend
            savings_percentage = (estimated_annual_savings / (eligible_average_hourly_spend * hours_per_year)) * 100.0
            
            # Calculate break-even timeline (months)
            # Break-even is when savings offset upfront cost
            if upfront_cost > 0 and estimated_monthly_savings > 0:
                break_even_months = int(upfront_cost / estimated_monthly_savings)
            else:
                break_even_months = 0  # No upfront cost means immediate savings
            
            # Assign confidence level based on coverage best practices and commitment risk
            # Account for inability to resell or exchange unused portions
            # High confidence: Coverage in 85-90% range (optimal for dynamic workloads with commitment risk buffer)
            # Medium confidence: Coverage 80-95% range
            # Low confidence: Coverage >95% (risky for dynamic usage) or <80% (low savings)
            
            if 85 <= projected_coverage <= 90:
                confidence_level = "high"
            elif 80 <= projected_coverage <= 95:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            # Generate rationale with optimization hierarchy and commitment risk
            rationale_parts = []
            rationale_parts.append("PREREQUISITE: Complete operational optimizations first (right-sizing, modernization, serverless)")
            rationale_parts.append(f"Covers {projected_coverage:.1f}% of eligible latest-generation instance usage")
            rationale_parts.append(f"~{rates_data.get('discount_percentage', 20):.0f}% discount rate (typically around 20%)")
            rationale_parts.append("1-year term with no upfront payment (only option for Database Savings Plans)")
            rationale_parts.append("Cannot be resold or exchanged - conservative coverage recommended")
            
            if excluded_spend > 0:
                excluded_hourly = excluded_spend / total_hours if total_hours > 0 else 0.0
                rationale_parts.append(f"Excludes ${excluded_hourly:.2f}/hour in older generation instances (use Reserved Instances)")
            
            rationale = "; ".join(rationale_parts)
            
            # Create recommendation object
            recommendation = SavingsPlanRecommendation(
                commitment_term=commitment_term,
                payment_option=payment_option,
                hourly_commitment=round(hourly_commitment, 2),
                estimated_annual_savings=round(estimated_annual_savings, 2),
                estimated_monthly_savings=round(estimated_monthly_savings, 2),
                savings_percentage=round(savings_percentage, 2),
                projected_coverage=round(projected_coverage, 2),
                projected_utilization=round(projected_utilization, 2),
                break_even_months=break_even_months,
                confidence_level=confidence_level,
                upfront_cost=round(upfront_cost, 2),
                total_commitment_cost=round(total_commitment_cost, 2),
                rationale=rationale
            )
            
            recommendations.append(recommendation)
            
            logger.debug(f"Generated recommendation: {commitment_term} {payment_option} - "
                        f"${hourly_commitment:.2f}/hour, ${estimated_annual_savings:.2f}/year savings")
    
    # Sort recommendations by estimated annual savings (descending)
    recommendations.sort(key=lambda r: r.estimated_annual_savings, reverse=True)
    
    # Convert to dictionary format for response
    recommendations_dict = [
        {
            "commitment_term": rec.commitment_term,
            "payment_option": rec.payment_option,
            "hourly_commitment": rec.hourly_commitment,
            "estimated_annual_savings": rec.estimated_annual_savings,
            "estimated_monthly_savings": rec.estimated_monthly_savings,
            "savings_percentage": rec.savings_percentage,
            "projected_coverage": rec.projected_coverage,
            "projected_utilization": rec.projected_utilization,
            "break_even_months": rec.break_even_months,
            "confidence_level": rec.confidence_level,
            "upfront_cost": rec.upfront_cost,
            "total_commitment_cost": rec.total_commitment_cost,
            "rationale": rec.rationale
        }
        for rec in recommendations
    ]
    
    logger.info(f"Generated {len(recommendations)} recommendations for eligible usage")
    
    # Calculate excluded spend for response
    excluded_hourly = excluded_spend / total_hours if total_hours > 0 else 0.0
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations_dict,
            "eligible_hourly_spend": eligible_average_hourly_spend,
            "excluded_hourly_spend": excluded_hourly,
            "total_hourly_spend": usage_data.get("average_hourly_spend", 0.0),
            "eligible_families": sorted(list(latest_generation_families)),
            "excluded_families": sorted(list(older_generation_families)),
            "coverage_strategy": "85-90% coverage targeting latest-generation instances with 15% buffer for commitment risk"
        },
        "message": f"Generated {len(recommendations)} Database Savings Plans recommendations for latest-generation instances"
    }


def analyze_custom_commitment(
    hourly_commitment: float,
    usage_data: Dict[str, Any],
    commitment_term: str = "1_YEAR",
    payment_option: str = "NO_UPFRONT",  # Changed default to comply with 1-year limitations
    adjusted_usage_projection: Optional[float] = None
) -> Dict[str, Any]:
    """
    Purchase analyzer mode: simulate custom commitment scenario.
    
    IMPORTANT LIMITATIONS:
    - Database Savings Plans apply to latest-generation instance families only (M7, R7, R8, etc.)
    - Currently only 1-year terms with No Upfront payment option are supported
    - Coverage recommendations target 85-90% to account for dynamic usage patterns
    - ElastiCache support limited to Valkey engine
    - Not available for OpenSearch, RedShift, Kendra, and some other services
    - Aurora Serverless and DMS capacity unit calculations may vary
    - Covers compute usage only; storage, backups, and data transfer billed separately
    
    Args:
        hourly_commitment: Custom hourly commitment amount in USD
        usage_data: Database usage analysis results
        commitment_term: "1_YEAR" or "3_YEAR"
        payment_option: Payment option (restrictions apply based on term)
        adjusted_usage_projection: Optional adjusted average hourly usage for future scenarios
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Purchase analyzer simulation results
            - message: Status message
            
    Example:
        >>> usage = analyze_database_usage()
        >>> result = analyze_custom_commitment(10.0, usage["data"], "1_YEAR", "NO_UPFRONT")
        >>> print(f"Coverage: {result['data']['projected_coverage']}%")
    """
    from services.savings_plans_service import calculate_savings_plans_rates
    
    logger.info(f"Analyzing custom commitment: ${hourly_commitment}/hour, "
                f"term={commitment_term}, payment={payment_option}")
    
    # Validate inputs
    import math
    if hourly_commitment <= 0 or math.isnan(hourly_commitment) or math.isinf(hourly_commitment):
        logger.error(f"Invalid hourly commitment: {hourly_commitment}")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Hourly commitment must be positive",
            "provided_value": hourly_commitment,
            "valid_range": "0.01 to 10000.00"
        }
    
    if commitment_term not in ["1_YEAR", "3_YEAR"]:
        logger.error(f"Invalid commitment term: {commitment_term}")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": f"Invalid commitment term: {commitment_term}",
            "valid_values": ["1_YEAR", "3_YEAR"]
        }
    
    # Validate payment option based on Database Savings Plans limitations
    if commitment_term == "1_YEAR" and payment_option != "NO_UPFRONT":
        logger.error(f"Invalid payment option for 1-year term: {payment_option}")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": f"Database Savings Plans currently only support 1-year terms with No Upfront payment option",
            "provided_value": payment_option,
            "valid_values": ["NO_UPFRONT"],
            "limitation": "Database Savings Plans currently only support 1-year terms with No Upfront payment option"
        }
    elif commitment_term != "1_YEAR":
        logger.error(f"Invalid commitment term: {commitment_term}")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": f"Database Savings Plans currently only support 1-year terms",
            "provided_value": commitment_term,
            "valid_values": ["1_YEAR"],
            "limitation": "Database Savings Plans currently only support 1-year terms"
        }
    
    # Validate usage_data
    if not usage_data:
        logger.error("Usage data is empty or None")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Usage data is required",
            "guidance": "Call analyze_database_usage() first to get usage data"
        }
    
    # Extract current usage data
    current_average_hourly_spend = usage_data.get("average_hourly_spend", 0.0)
    
    if current_average_hourly_spend <= 0:
        logger.warning("No database usage found in usage data")
        return {
            "status": "warning",
            "data": {
                "hourly_commitment": hourly_commitment,
                "commitment_term": commitment_term,
                "payment_option": payment_option,
                "projected_annual_cost": 0.0,
                "projected_coverage": 0.0,
                "projected_utilization": 0.0,
                "estimated_annual_savings": 0.0,
                "uncovered_on_demand_cost": 0.0,
                "unused_commitment_cost": 0.0
            },
            "message": "No database usage found - cannot calculate meaningful projections",
            "guidance": "Ensure database resources are running and have usage history"
        }
    
    # Use adjusted usage projection if provided, otherwise use current usage
    projected_average_hourly_spend = adjusted_usage_projection if adjusted_usage_projection is not None else current_average_hourly_spend
    
    if adjusted_usage_projection is not None:
        logger.info(f"Using adjusted usage projection: ${projected_average_hourly_spend:.2f}/hour "
                   f"(current: ${current_average_hourly_spend:.2f}/hour)")
    
    # Validate adjusted usage projection if provided
    if adjusted_usage_projection is not None and (adjusted_usage_projection <= 0 or math.isnan(adjusted_usage_projection) or math.isinf(adjusted_usage_projection)):
        logger.error(f"Invalid adjusted usage projection: {adjusted_usage_projection}")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Adjusted usage projection must be positive",
            "provided_value": adjusted_usage_projection,
            "valid_range": "0.01 to 10000.00"
        }
    
    # Get savings rates for the commitment
    rates_result = calculate_savings_plans_rates(
        on_demand_rate=projected_average_hourly_spend,
        commitment_term=commitment_term,
        payment_option=payment_option,
        service_code="AmazonRDS"  # Generic database service
    )
    
    if rates_result.get("status") != "success":
        logger.error(f"Failed to calculate savings rates: {rates_result.get('message')}")
        return rates_result
    
    rates_data = rates_result.get("data", {})
    
    # Calculate coverage and utilization
    # Coverage: What percentage of eligible usage is covered by the commitment
    projected_coverage = min((hourly_commitment / projected_average_hourly_spend) * 100.0, 100.0)
    
    # Utilization: What percentage of the commitment is actually used
    projected_utilization = min((projected_average_hourly_spend / hourly_commitment) * 100.0, 100.0)
    
    # Calculate costs and savings
    hours_per_year = 8760
    
    # Calculate how much of the commitment will be used
    used_commitment = min(hourly_commitment, projected_average_hourly_spend)
    unused_commitment = max(0.0, hourly_commitment - projected_average_hourly_spend)
    
    # Calculate uncovered on-demand usage
    uncovered_usage = max(0.0, projected_average_hourly_spend - hourly_commitment)
    
    # Get discount rate and savings plan rate
    discount_percentage = rates_data.get("discount_percentage", 0.0) / 100.0
    savings_plan_rate = rates_data.get("savings_plan_rate", projected_average_hourly_spend)
    
    # Calculate annual costs
    # Cost for the commitment (used portion at savings plan rate)
    annual_commitment_cost = used_commitment * savings_plan_rate * hours_per_year
    
    # Cost for unused commitment (still pay commitment rate)
    annual_unused_commitment_cost = unused_commitment * savings_plan_rate * hours_per_year
    
    # Cost for uncovered usage (at on-demand rate)
    annual_uncovered_on_demand_cost = uncovered_usage * projected_average_hourly_spend * hours_per_year
    
    # Total projected annual cost
    projected_annual_cost = annual_commitment_cost + annual_unused_commitment_cost + annual_uncovered_on_demand_cost
    
    # Calculate what the cost would be without any commitment (all on-demand)
    annual_on_demand_cost_without_commitment = projected_average_hourly_spend * hours_per_year
    
    # Calculate savings
    estimated_annual_savings = annual_on_demand_cost_without_commitment - projected_annual_cost
    
    # Calculate upfront costs based on payment option
    if payment_option == "ALL_UPFRONT":
        upfront_cost = hourly_commitment * hours_per_year * (1 - discount_percentage)
    elif payment_option == "PARTIAL_UPFRONT":
        upfront_cost = hourly_commitment * hours_per_year * (1 - discount_percentage) * 0.5
    else:  # NO_UPFRONT
        upfront_cost = 0.0
    
    # Create PurchaseAnalyzerResult object
    result = PurchaseAnalyzerResult(
        hourly_commitment=hourly_commitment,
        commitment_term=commitment_term,
        payment_option=payment_option,
        projected_annual_cost=round(projected_annual_cost, 2),
        projected_coverage=round(projected_coverage, 2),
        projected_utilization=round(projected_utilization, 2),
        estimated_annual_savings=round(estimated_annual_savings, 2),
        uncovered_on_demand_cost=round(annual_uncovered_on_demand_cost, 2),
        unused_commitment_cost=round(annual_unused_commitment_cost, 2)
    )
    
    # Generate analysis insights
    insights = []
    
    if projected_coverage < 100:
        uncovered_percentage = 100 - projected_coverage
        insights.append(f"{uncovered_percentage:.1f}% of usage remains uncovered (${uncovered_usage:.2f}/hour)")
    
    if projected_utilization < 100:
        unused_percentage = 100 - projected_utilization
        insights.append(f"{unused_percentage:.1f}% of commitment unused (${unused_commitment:.2f}/hour)")
    
    if projected_utilization > 100:
        insights.append("Commitment fully utilized with additional on-demand usage")
    
    if estimated_annual_savings > 0:
        savings_percentage = (estimated_annual_savings / annual_on_demand_cost_without_commitment) * 100
        insights.append(f"{savings_percentage:.1f}% total cost reduction")
    else:
        insights.append("No savings - commitment exceeds usage")
    
    # Add coverage guidance based on best practices
    if projected_coverage > 90:
        insights.append("⚠️  Coverage >90% may be risky due to dynamic usage patterns - consider 85-90% target")
    elif projected_coverage >= 85:
        insights.append("✅ Coverage in recommended 85-90% range for dynamic workloads")
    elif projected_coverage < 85:
        insights.append("💡 Coverage <85% - consider increasing commitment for better savings")
    
    # Determine recommendation
    if projected_utilization >= 80 and 85 <= projected_coverage <= 90:
        recommendation = "Optimal balance - good utilization within recommended coverage range"
    elif projected_coverage > 90:
        recommendation = "Consider reducing commitment - coverage >90% risky for dynamic workloads"
    elif projected_utilization < 70:
        recommendation = "Consider reducing commitment to improve utilization"
    elif projected_coverage < 85:
        recommendation = "Consider increasing commitment to reach 85-90% coverage target"
    else:
        recommendation = "Moderate efficiency - review usage patterns and consider adjustments"
    
    logger.info(f"Custom commitment analysis complete: "
                f"coverage={projected_coverage:.1f}%, utilization={projected_utilization:.1f}%, "
                f"savings=${estimated_annual_savings:.2f}/year")
    
    # Return results in dictionary format
    return {
        "status": "success",
        "data": {
            "hourly_commitment": result.hourly_commitment,
            "commitment_term": result.commitment_term,
            "payment_option": result.payment_option,
            "projected_annual_cost": result.projected_annual_cost,
            "projected_coverage": result.projected_coverage,
            "projected_utilization": result.projected_utilization,
            "estimated_annual_savings": result.estimated_annual_savings,
            "uncovered_on_demand_cost": result.uncovered_on_demand_cost,
            "unused_commitment_cost": result.unused_commitment_cost,
            "upfront_cost": upfront_cost,
            "current_usage": current_average_hourly_spend,
            "projected_usage": projected_average_hourly_spend,
            "used_commitment": used_commitment,
            "unused_commitment": unused_commitment,
            "uncovered_usage": uncovered_usage,
            "discount_percentage": rates_data.get("discount_percentage", 0.0),
            "insights": insights,
            "recommendation": recommendation
        },
        "message": f"Purchase analyzer simulation complete: "
                  f"{projected_coverage:.1f}% coverage, {projected_utilization:.1f}% utilization"
    }


def compare_with_reserved_instances(
    usage_data: Dict[str, Any],
    services: List[str] = None
) -> Dict[str, Any]:
    """
    Compare Database Savings Plans with Reserved Instance options based on instance generation.
    
    Database Savings Plans apply to latest-generation instances (M7, R7, R8) only.
    Reserved Instances are required for older generation instances (M5, R5, R6g, T3, T4g).
    
    Args:
        usage_data: Database usage analysis results
        services: List of services to compare (default: ["rds", "aurora"])
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Comparison analysis results with latest_generation and older_generation sections
            - message: Status message
            
    Example:
        >>> usage = analyze_database_usage()
        >>> comparison = compare_with_reserved_instances(usage["data"])
        >>> for item in comparison["data"]["latest_generation"]:
        ...     print(f"{item['instance_type']}: {item['recommendation']}")
        >>> for item in comparison["data"]["older_generation"]:
        ...     print(f"{item['instance_type']}: {item['recommendation']}")
    """
    from services.pricing import get_rds_pricing
    from services.savings_plans_service import calculate_savings_plans_rates
    
    if services is None:
        services = ["rds", "aurora"]
    
    logger.info(f"Comparing Database Savings Plans vs Reserved Instances for services: {services}")
    
    # Validate usage_data
    if not usage_data:
        logger.error("Usage data is empty or None")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Usage data is required",
            "guidance": "Call analyze_database_usage() first to get usage data"
        }
    
    # Define instance family classifications
    latest_generation_families = {
        # RDS/Aurora latest generation (eligible for Database Savings Plans)
        'db.m7g', 'db.m7i', 'db.r7g', 'db.r7i', 'db.r8g', 'db.t4g',
        'db.x2gd', 'db.x2iedn', 'db.x2iezn',
        # ElastiCache latest generation (Valkey engine only)
        'cache.m7g', 'cache.r7g', 'cache.t4g',
        # DocumentDB latest generation
        'docdb.r7g', 'docdb.t4g',
        # Neptune latest generation  
        'neptune.r7g', 'neptune.t4g'
    }
    
    # Older generation families (require Reserved Instances)
    older_generation_families = {
        'db.m5', 'db.m5d', 'db.m5n', 'db.m5zn', 'db.m6g', 'db.m6gd', 'db.m6i', 'db.m6id', 'db.m6idn', 'db.m6in',
        'db.r5', 'db.r5b', 'db.r5d', 'db.r5dn', 'db.r5n', 'db.r6g', 'db.r6gd', 'db.r6i', 'db.r6id', 'db.r6idn', 'db.r6in',
        'db.t3', 'db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge', 'db.t3.2xlarge',
        'cache.m5', 'cache.m5n', 'cache.m6g', 'cache.m6gd', 'cache.m6i', 'cache.m6id', 'cache.m6idn', 'cache.m6in',
        'cache.r5', 'cache.r5n', 'cache.r6g', 'cache.r6gd', 'cache.r6i', 'cache.r6id', 'cache.r6idn', 'cache.r6in',
        'cache.t3'
    }
    
    # Extract instance family breakdown from usage data
    instance_family_breakdown = usage_data.get("instance_family_breakdown", {})
    service_breakdown = usage_data.get("service_breakdown", {})
    
    if not instance_family_breakdown and not service_breakdown:
        logger.warning("No instance family or service breakdown available for comparison")
        return {
            "status": "warning",
            "data": {
                "latest_generation": [],
                "older_generation": [],
                "summary": {
                    "total_latest_generation_spend": 0.0,
                    "total_older_generation_spend": 0.0,
                    "database_savings_plans_eligible": 0.0,
                    "reserved_instances_required": 0.0,
                    "total_potential_savings": 0.0
                }
            },
            "message": "No instance family breakdown available - cannot perform detailed comparison",
            "guidance": "Ensure usage data includes instance family breakdown for accurate comparison"
        }
    
    # Analyze latest-generation instances (Database Savings Plans eligible)
    latest_generation_comparisons = []
    total_latest_generation_spend = 0.0
    
    # Analyze older generation instances (Reserved Instances required)
    older_generation_comparisons = []
    total_older_generation_spend = 0.0
    
    # Process instance families
    for family, spend in instance_family_breakdown.items():
        if spend <= 0:
            continue
            
        # Determine if this family is latest or older generation
        is_latest_generation = any(family.startswith(prefix) for prefix in latest_generation_families)
        is_older_generation = any(family.startswith(prefix) for prefix in older_generation_families)
        
        # Calculate hourly spend for this family
        total_hours = usage_data.get("lookback_period_days", 30) * 24
        hourly_spend = spend / total_hours if total_hours > 0 else 0.0
        
        if is_latest_generation and not is_older_generation:
            # Latest generation - Database Savings Plans only
            total_latest_generation_spend += spend
            
            # Create a representative instance type for pricing (use common size)
            instance_type = f"{family}.xlarge"  # Use xlarge as representative size
            
            # Get on-demand pricing
            pricing_result = get_rds_pricing(
                instance_class=instance_type,
                engine='mysql',  # Use MySQL as representative engine
                region=usage_data.get("region", "us-east-1")
            )
            
            if pricing_result.get("status") == "success":
                on_demand_rate = pricing_result.get("hourly_price", 0.0)
            else:
                # Use estimated rate based on spend
                on_demand_rate = hourly_spend
            
            # Calculate Database Savings Plans savings (1-year, No Upfront only)
            sp_rates_result = calculate_savings_plans_rates(
                on_demand_rate=on_demand_rate,
                commitment_term="1_YEAR",
                payment_option="NO_UPFRONT",
                service_code="AmazonRDS"
            )
            
            if sp_rates_result.get("status") == "success":
                sp_data = sp_rates_result.get("data", {})
                savings_plan_rate = sp_data.get("savings_plan_rate", on_demand_rate)
                savings_plan_savings = sp_data.get("annual_savings", 0.0)
                discount_percentage = sp_data.get("discount_percentage", 0.0)
            else:
                # Use default 20% discount for Database Savings Plans
                discount_percentage = 20.0
                savings_plan_rate = on_demand_rate * 0.8
                savings_plan_savings = (on_demand_rate - savings_plan_rate) * 8760
            
            # Calculate annual costs
            annual_on_demand_cost = on_demand_rate * 8760
            annual_savings_plan_cost = savings_plan_rate * 8760
            
            # Create comparison entry
            comparison = CommitmentComparison(
                service=family.split('.')[0],  # Extract service (db, cache, etc.)
                instance_type=instance_type,
                savings_plan_cost=annual_savings_plan_cost,
                ri_standard_cost=0.0,  # Not applicable for latest generation
                ri_convertible_cost=0.0,  # Not applicable for latest generation
                on_demand_cost=annual_on_demand_cost,
                recommended_option="Database Savings Plans",
                savings_plan_savings=savings_plan_savings,
                ri_standard_savings=0.0,  # Not applicable
                ri_convertible_savings=0.0,  # Not applicable
                flexibility_score=85.0,  # High flexibility - applies across instance families
                rationale=f"Latest-generation instance ({family}) - Database Savings Plans only option available. ~{discount_percentage}% discount with 1-year No Upfront commitment. Cannot be resold or exchanged."
            )
            
            latest_generation_comparisons.append({
                "service": comparison.service,
                "instance_type": comparison.instance_type,
                "instance_family": family,
                "annual_spend": spend * (365 / usage_data.get("lookback_period_days", 30)),  # Annualize spend
                "savings_plan_cost": comparison.savings_plan_cost,
                "on_demand_cost": comparison.on_demand_cost,
                "savings_plan_savings": comparison.savings_plan_savings,
                "discount_percentage": discount_percentage,
                "recommendation": comparison.recommended_option,
                "rationale": comparison.rationale,
                "flexibility_score": comparison.flexibility_score,
                "commitment_options": ["1-year Database Savings Plans (No Upfront only)"],
                "limitations": ["Cannot be resold or exchanged", "Covers compute only", "Latest-generation instances only"]
            })
            
        elif is_older_generation:
            # Older generation - Reserved Instances required
            total_older_generation_spend += spend
            
            # Create a representative instance type for pricing
            instance_type = f"{family}.xlarge"  # Use xlarge as representative size
            
            # Get on-demand pricing
            pricing_result = get_rds_pricing(
                instance_class=instance_type,
                engine='mysql',  # Use MySQL as representative engine
                region=usage_data.get("region", "us-east-1")
            )
            
            if pricing_result.get("status") == "success":
                on_demand_rate = pricing_result.get("hourly_price", 0.0)
            else:
                # Use estimated rate based on spend
                on_demand_rate = hourly_spend
            
            # Calculate Reserved Instance savings (approximate rates)
            # Standard RI: ~30% discount for 1-year, ~45% for 3-year
            # Convertible RI: ~25% discount for 1-year, ~40% for 3-year
            
            # 1-year Standard RI
            ri_standard_1yr_rate = on_demand_rate * 0.70  # 30% discount
            ri_standard_1yr_annual_cost = ri_standard_1yr_rate * 8760
            ri_standard_1yr_savings = (on_demand_rate - ri_standard_1yr_rate) * 8760
            
            # 1-year Convertible RI
            ri_convertible_1yr_rate = on_demand_rate * 0.75  # 25% discount
            ri_convertible_1yr_annual_cost = ri_convertible_1yr_rate * 8760
            ri_convertible_1yr_savings = (on_demand_rate - ri_convertible_1yr_rate) * 8760
            
            # 3-year Standard RI
            ri_standard_3yr_rate = on_demand_rate * 0.55  # 45% discount
            ri_standard_3yr_annual_cost = ri_standard_3yr_rate * 8760
            ri_standard_3yr_savings = (on_demand_rate - ri_standard_3yr_rate) * 8760
            
            # 3-year Convertible RI
            ri_convertible_3yr_rate = on_demand_rate * 0.60  # 40% discount
            ri_convertible_3yr_annual_cost = ri_convertible_3yr_rate * 8760
            ri_convertible_3yr_savings = (on_demand_rate - ri_convertible_3yr_rate) * 8760
            
            # Calculate annual costs
            annual_on_demand_cost = on_demand_rate * 8760
            
            # Determine best RI option (highest savings)
            ri_options = [
                ("1-year Standard RI", ri_standard_1yr_annual_cost, ri_standard_1yr_savings, 70.0),
                ("1-year Convertible RI", ri_convertible_1yr_annual_cost, ri_convertible_1yr_savings, 85.0),
                ("3-year Standard RI", ri_standard_3yr_annual_cost, ri_standard_3yr_savings, 60.0),
                ("3-year Convertible RI", ri_convertible_3yr_annual_cost, ri_convertible_3yr_savings, 75.0)
            ]
            
            # Find option with highest savings
            best_option = max(ri_options, key=lambda x: x[2])
            recommended_option, best_cost, best_savings, flexibility_score = best_option
            
            # Create comparison entry
            comparison = CommitmentComparison(
                service=family.split('.')[0],  # Extract service (db, cache, etc.)
                instance_type=instance_type,
                savings_plan_cost=0.0,  # Not applicable for older generation
                ri_standard_cost=ri_standard_1yr_annual_cost,  # Use 1-year as reference
                ri_convertible_cost=ri_convertible_1yr_annual_cost,  # Use 1-year as reference
                on_demand_cost=annual_on_demand_cost,
                recommended_option=recommended_option,
                savings_plan_savings=0.0,  # Not applicable
                ri_standard_savings=ri_standard_1yr_savings,
                ri_convertible_savings=ri_convertible_1yr_savings,
                flexibility_score=flexibility_score,
                rationale=f"Older generation instance ({family}) - Database Savings Plans not available. {recommended_option} provides best savings. Consider upgrading to latest-generation instances for Database Savings Plans eligibility."
            )
            
            older_generation_comparisons.append({
                "service": comparison.service,
                "instance_type": comparison.instance_type,
                "instance_family": family,
                "annual_spend": spend * (365 / usage_data.get("lookback_period_days", 30)),  # Annualize spend
                "ri_standard_1yr_cost": ri_standard_1yr_annual_cost,
                "ri_convertible_1yr_cost": ri_convertible_1yr_annual_cost,
                "ri_standard_3yr_cost": ri_standard_3yr_annual_cost,
                "ri_convertible_3yr_cost": ri_convertible_3yr_annual_cost,
                "on_demand_cost": annual_on_demand_cost,
                "ri_standard_1yr_savings": ri_standard_1yr_savings,
                "ri_convertible_1yr_savings": ri_convertible_1yr_savings,
                "ri_standard_3yr_savings": ri_standard_3yr_savings,
                "ri_convertible_3yr_savings": ri_convertible_3yr_savings,
                "best_option": recommended_option,
                "best_savings": best_savings,
                "recommendation": comparison.recommended_option,
                "rationale": comparison.rationale,
                "flexibility_score": comparison.flexibility_score,
                "commitment_options": ["1-year Standard RI", "1-year Convertible RI", "3-year Standard RI", "3-year Convertible RI"],
                "upgrade_recommendation": f"Consider upgrading to latest-generation instances (M7, R7, R8) for Database Savings Plans eligibility"
            })
    
    # Calculate summary metrics
    total_spend = total_latest_generation_spend + total_older_generation_spend
    database_savings_plans_eligible_percentage = (total_latest_generation_spend / total_spend * 100) if total_spend > 0 else 0.0
    reserved_instances_required_percentage = (total_older_generation_spend / total_spend * 100) if total_spend > 0 else 0.0
    
    # Calculate total potential savings
    total_sp_savings = sum(item.get("savings_plan_savings", 0.0) for item in latest_generation_comparisons)
    total_ri_savings = sum(item.get("best_savings", 0.0) for item in older_generation_comparisons)
    total_potential_savings = total_sp_savings + total_ri_savings
    
    logger.info(f"Comparison complete: {len(latest_generation_comparisons)} latest-generation, "
                f"{len(older_generation_comparisons)} older generation instances analyzed")
    
    return {
        "status": "success",
        "data": {
            "latest_generation": latest_generation_comparisons,
            "older_generation": older_generation_comparisons,
            "summary": {
                "total_latest_generation_spend": round(total_latest_generation_spend, 2),
                "total_older_generation_spend": round(total_older_generation_spend, 2),
                "total_spend": round(total_spend, 2),
                "database_savings_plans_eligible_percentage": round(database_savings_plans_eligible_percentage, 1),
                "reserved_instances_required_percentage": round(reserved_instances_required_percentage, 1),
                "total_database_savings_plans_savings": round(total_sp_savings, 2),
                "total_reserved_instances_savings": round(total_ri_savings, 2),
                "total_potential_savings": round(total_potential_savings, 2),
                "latest_generation_families_analyzed": len(latest_generation_comparisons),
                "older_generation_families_analyzed": len(older_generation_comparisons)
            },
            "recommendations": {
                "immediate_actions": [
                    f"Purchase Database Savings Plans for ${total_latest_generation_spend:.2f} in latest-generation instances" if total_latest_generation_spend > 0 else None,
                    f"Purchase Reserved Instances for ${total_older_generation_spend:.2f} in older generation instances" if total_older_generation_spend > 0 else None
                ],
                "long_term_strategy": [
                    "Consider upgrading older generation instances to latest-generation for Database Savings Plans eligibility",
                    "Database Savings Plans provide better flexibility than Reserved Instances",
                    "Monitor usage patterns - Database Savings Plans work best for stable workloads"
                ]
            },
            "limitations": {
                "database_savings_plans": [
                    "Latest-generation instances only (M7, R7, R8, etc.)",
                    "1-year terms with No Upfront payment only",
                    "Cannot be resold or exchanged",
                    "Covers compute usage only"
                ],
                "reserved_instances": [
                    "Instance type and region specific",
                    "Can be sold in Reserved Instance Marketplace",
                    "Convertible RIs allow instance family changes",
                    "Standard RIs offer higher discounts but less flexibility"
                ]
            }
        },
        "message": f"Analyzed {len(latest_generation_comparisons)} latest-generation and {len(older_generation_comparisons)} older generation instance families. Total potential savings: ${total_potential_savings:.2f}/year"
    }


def analyze_existing_commitments(
    region: Optional[str] = None,
    lookback_period_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze existing Database Savings Plans utilization and coverage.
    
    Args:
        region: AWS region to analyze (None for all regions)
        lookback_period_days: Number of days to analyze
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Existing commitments analysis
            - message: Status message
            
    Example:
        >>> result = analyze_existing_commitments(region="us-east-1")
        >>> for plan in result["data"]["existing_plans"]:
        ...     print(f"Plan {plan['savings_plan_id']}: "
        ...           f"Utilization {plan['utilization_percentage']}%")
    """
    from datetime import datetime, timedelta
    from services.savings_plans_service import (
        get_existing_savings_plans,
        get_savings_plans_utilization,
        get_savings_plans_coverage
    )
    
    logger.info(f"Analyzing existing commitments: region={region}, "
                f"lookback_period={lookback_period_days}")
    
    # Validate lookback period
    if lookback_period_days not in [30, 60, 90]:
        logger.error(f"Invalid lookback period: {lookback_period_days}")
        return {
            "status": "error",
            "message": f"Lookback period must be 30, 60, or 90 days",
            "error_code": "ValidationError",
            "provided_value": lookback_period_days,
            "valid_values": [30, 60, 90]
        }
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_period_days)
    
    # Format dates for Cost Explorer API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    time_period = {
        'Start': start_date_str,
        'End': end_date_str
    }
    
    logger.info(f"Analyzing existing commitments from {start_date_str} to {end_date_str}")
    
    # Step 1: Retrieve current Database Savings Plans
    existing_plans_result = get_existing_savings_plans(
        states=['active', 'payment-pending'],  # Only active plans
        region=region
    )
    
    if existing_plans_result.get("status") != "success":
        logger.error(f"Failed to retrieve existing savings plans: {existing_plans_result.get('message')}")
        return existing_plans_result
    
    existing_plans_data = existing_plans_result.get("data", {})
    savings_plans = existing_plans_data.get("savings_plans", [])
    
    # Filter for Database Savings Plans only
    database_savings_plans = [
        plan for plan in savings_plans 
        if plan.get('planType') == 'SavingsPlans' and 
           plan.get('productTypes', []) and 
           any('Database' in product_type for product_type in plan.get('productTypes', []))
    ]
    
    if not database_savings_plans:
        logger.info("No active Database Savings Plans found")
        return {
            "status": "info",
            "data": {
                "existing_plans": [],
                "gaps": {
                    "uncovered_spend": 0.0,
                    "recommendation": "No existing Database Savings Plans found - consider analyzing usage to identify opportunities"
                },
                "summary": {
                    "total_plans": 0,
                    "total_hourly_commitment": 0.0,
                    "average_utilization": 0.0,
                    "average_coverage": 0.0,
                    "total_unused_commitment": 0.0
                }
            },
            "message": "No active Database Savings Plans found",
            "guidance": "Run database usage analysis to identify Database Savings Plans opportunities"
        }
    
    logger.info(f"Found {len(database_savings_plans)} active Database Savings Plans")
    
    # Step 2: Get utilization data from Cost Explorer
    utilization_result = get_savings_plans_utilization(
        time_period=time_period,
        granularity="MONTHLY",
        region=region
    )
    
    if utilization_result.get("status") != "success":
        logger.error(f"Failed to retrieve utilization data: {utilization_result.get('message')}")
        # Continue with partial data - we can still analyze the plans themselves
        utilization_data = {}
    else:
        utilization_data = utilization_result.get("data", {})
    
    # Step 3: Get coverage data from Cost Explorer
    coverage_result = get_savings_plans_coverage(
        time_period=time_period,
        granularity="MONTHLY",
        group_by=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}],
        region=region
    )
    
    if coverage_result.get("status") != "success":
        logger.error(f"Failed to retrieve coverage data: {coverage_result.get('message')}")
        # Continue with partial data
        coverage_data = {}
    else:
        coverage_data = coverage_result.get("data", {})
    
    # Step 4: Analyze each Database Savings Plan
    analyzed_plans = []
    total_hourly_commitment = 0.0
    total_unused_commitment = 0.0
    utilization_values = []
    coverage_values = []
    
    # Extract utilization metrics from Cost Explorer response
    utilization_by_plan = {}
    if utilization_data:
        # Parse utilization data - Cost Explorer returns aggregated data
        results_by_time = utilization_data.get('ResultsByTime', [])
        for time_result in results_by_time:
            total_utilization = time_result.get('Total', {})
            utilization_percentage = float(total_utilization.get('UtilizationPercentage', '0'))
            used_commitment = float(total_utilization.get('UsedCommitment', '0'))
            total_commitment = float(total_utilization.get('TotalCommitment', '0'))
            unused_commitment = float(total_utilization.get('UnusedCommitment', '0'))
            
            # Store aggregated utilization data (Cost Explorer doesn't break down by individual plan)
            utilization_by_plan['aggregated'] = {
                'utilization_percentage': utilization_percentage,
                'used_commitment': used_commitment,
                'total_commitment': total_commitment,
                'unused_commitment': unused_commitment
            }
    
    # Extract coverage metrics from Cost Explorer response
    coverage_by_service = {}
    total_coverage_percentage = 0.0
    uncovered_spend = 0.0
    
    if coverage_data:
        results_by_time = coverage_data.get('ResultsByTime', [])
        for time_result in results_by_time:
            # Get total coverage
            total_coverage = time_result.get('Total', {})
            if total_coverage:
                coverage_percentage = float(total_coverage.get('CoveragePercentage', '0'))
                on_demand_cost = float(total_coverage.get('OnDemandCost', '0'))
                covered_cost = float(total_coverage.get('CoveredCost', '0'))
                
                total_coverage_percentage = coverage_percentage
                uncovered_spend = on_demand_cost - covered_cost
            
            # Get coverage by service
            groups = time_result.get('Groups', [])
            for group in groups:
                service_name = group.get('Keys', ['Unknown'])[0]
                coverage_metrics = group.get('Coverage', {})
                
                coverage_by_service[service_name] = {
                    'coverage_percentage': float(coverage_metrics.get('CoveragePercentage', '0')),
                    'on_demand_cost': float(coverage_metrics.get('OnDemandCost', '0')),
                    'covered_cost': float(coverage_metrics.get('CoveredCost', '0'))
                }
    
    # Analyze each individual savings plan
    for plan in database_savings_plans:
        savings_plan_id = plan.get('savingsPlansId', 'Unknown')
        savings_plan_arn = plan.get('savingsPlansArn', '')
        
        # Extract plan details
        hourly_commitment = float(plan.get('commitment', '0'))
        commitment_term = plan.get('termDurationInSeconds', 0)
        payment_option = plan.get('paymentOption', 'Unknown')
        start_date = plan.get('start', '')
        end_date = plan.get('end', '')
        state = plan.get('state', 'Unknown')
        
        # Convert term duration to readable format
        if commitment_term == 31536000:  # 1 year in seconds
            term_readable = "1_YEAR"
        elif commitment_term == 94608000:  # 3 years in seconds
            term_readable = "3_YEAR"
        else:
            term_readable = f"{commitment_term}_SECONDS"
        
        # Parse dates
        try:
            start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
            end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        except:
            start_datetime = None
            end_datetime = None
        
        # Calculate utilization and coverage for this plan
        # Note: Cost Explorer provides aggregated data, so we'll distribute it proportionally
        if utilization_by_plan.get('aggregated') and len(database_savings_plans) > 0:
            # Distribute aggregated utilization proportionally based on commitment size
            total_commitment_all_plans = sum(float(p.get('commitment', '0')) for p in database_savings_plans)
            if total_commitment_all_plans > 0:
                plan_proportion = hourly_commitment / total_commitment_all_plans
                
                aggregated_util = utilization_by_plan['aggregated']
                plan_utilization_percentage = aggregated_util['utilization_percentage']
                plan_unused_commitment = aggregated_util['unused_commitment'] * plan_proportion
            else:
                plan_utilization_percentage = 0.0
                plan_unused_commitment = 0.0
        else:
            # No utilization data available - estimate based on commitment
            plan_utilization_percentage = 0.0
            plan_unused_commitment = hourly_commitment  # Assume fully unused if no data
        
        # Coverage is harder to attribute to individual plans, use overall coverage
        plan_coverage_percentage = total_coverage_percentage
        
        # Create ExistingSavingsPlan object
        existing_plan = ExistingSavingsPlan(
            savings_plan_id=savings_plan_id,
            savings_plan_arn=savings_plan_arn,
            hourly_commitment=hourly_commitment,
            commitment_term=term_readable,
            payment_option=payment_option,
            start_date=start_datetime,
            end_date=end_datetime,
            utilization_percentage=round(plan_utilization_percentage, 2),
            coverage_percentage=round(plan_coverage_percentage, 2),
            unused_commitment_hourly=round(plan_unused_commitment, 4),
            status=state
        )
        
        analyzed_plans.append(existing_plan)
        
        # Accumulate totals
        total_hourly_commitment += hourly_commitment
        total_unused_commitment += plan_unused_commitment
        utilization_values.append(plan_utilization_percentage)
        coverage_values.append(plan_coverage_percentage)
        
        logger.debug(f"Analyzed plan {savings_plan_id}: "
                    f"${hourly_commitment}/hour, {plan_utilization_percentage}% utilization")
    
    # Calculate summary metrics
    average_utilization = sum(utilization_values) / len(utilization_values) if utilization_values else 0.0
    average_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    
    # Identify gaps and over-commitment
    gaps_analysis = {}
    
    # Gap identification: uncovered spend that could benefit from additional commitments
    if uncovered_spend > 0:
        # Convert to hourly rate
        hours_in_period = lookback_period_days * 24
        uncovered_hourly_spend = uncovered_spend / hours_in_period if hours_in_period > 0 else 0.0
        
        gaps_analysis["uncovered_spend"] = round(uncovered_spend, 2)
        gaps_analysis["uncovered_hourly_spend"] = round(uncovered_hourly_spend, 4)
        
        # Recommend additional commitment (conservative 85% coverage target)
        recommended_additional_commitment = uncovered_hourly_spend * 0.85
        gaps_analysis["recommended_additional_commitment"] = round(recommended_additional_commitment, 4)
        gaps_analysis["recommendation"] = f"Consider additional ${recommended_additional_commitment:.2f}/hour commitment to cover uncovered usage"
    else:
        gaps_analysis["uncovered_spend"] = 0.0
        gaps_analysis["recommendation"] = "No significant coverage gaps identified"
    
    # Over-commitment detection
    if total_unused_commitment > 0:
        gaps_analysis["over_commitment_detected"] = True
        gaps_analysis["total_unused_commitment_hourly"] = round(total_unused_commitment, 4)
        gaps_analysis["over_commitment_guidance"] = f"${total_unused_commitment:.2f}/hour in unused commitment - monitor usage patterns and consider adjusting future commitments"
    else:
        gaps_analysis["over_commitment_detected"] = False
        gaps_analysis["over_commitment_guidance"] = "No significant over-commitment detected"
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if average_utilization < 80:
        recommendations.append("⚠️  Low utilization detected - review usage patterns and consider reducing future commitments")
    elif average_utilization > 95:
        recommendations.append("✅ High utilization - consider additional commitments if usage is growing")
    else:
        recommendations.append("✅ Good utilization levels")
    
    if average_coverage < 85:
        recommendations.append("💡 Coverage below recommended 85-90% - consider additional commitments")
    elif average_coverage > 90:
        recommendations.append("⚠️  Coverage above 90% - monitor for over-commitment risk")
    else:
        recommendations.append("✅ Coverage in recommended range")
    
    if total_unused_commitment > total_hourly_commitment * 0.1:  # >10% unused
        recommendations.append("⚠️  Significant unused commitment detected - review commitment sizing")
    
    # Convert analyzed plans to dictionary format for response
    plans_dict = [
        {
            "savings_plan_id": plan.savings_plan_id,
            "savings_plan_arn": plan.savings_plan_arn,
            "hourly_commitment": plan.hourly_commitment,
            "commitment_term": plan.commitment_term,
            "payment_option": plan.payment_option,
            "start_date": plan.start_date.isoformat() if plan.start_date else None,
            "end_date": plan.end_date.isoformat() if plan.end_date else None,
            "utilization_percentage": plan.utilization_percentage,
            "coverage_percentage": plan.coverage_percentage,
            "unused_commitment_hourly": plan.unused_commitment_hourly,
            "status": plan.status
        }
        for plan in analyzed_plans
    ]
    
    logger.info(f"Existing commitments analysis complete: {len(analyzed_plans)} plans analyzed, "
                f"avg utilization={average_utilization:.1f}%, avg coverage={average_coverage:.1f}%")
    
    return {
        "status": "success",
        "data": {
            "existing_plans": plans_dict,
            "gaps": gaps_analysis,
            "summary": {
                "total_plans": len(analyzed_plans),
                "total_hourly_commitment": round(total_hourly_commitment, 2),
                "average_utilization": round(average_utilization, 2),
                "average_coverage": round(average_coverage, 2),
                "total_unused_commitment": round(total_unused_commitment, 4),
                "analysis_period": f"{start_date_str} to {end_date_str}",
                "lookback_period_days": lookback_period_days
            },
            "coverage_by_service": coverage_by_service,
            "recommendations": recommendations
        },
        "message": f"Analyzed {len(analyzed_plans)} Database Savings Plans: "
                  f"avg utilization {average_utilization:.1f}%, avg coverage {average_coverage:.1f}%"
    }


# ============================================================================
# Multi-Account Support Functions
# ============================================================================

def aggregate_multi_account_usage(
    account_ids: List[str],
    region: Optional[str] = None,
    lookback_period_days: int = 30,
    services: Optional[List[str]] = None,
    organization_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate usage data across AWS Organizations member accounts.
    
    This function retrieves and aggregates database usage data from multiple AWS accounts
    within an organization, providing both account-level and consolidated views for
    Database Savings Plans analysis.
    
    Args:
        account_ids: List of AWS account IDs to analyze
        region: AWS region to analyze (None for all regions)
        lookback_period_days: Number of days to analyze (30, 60, or 90)
        services: List of database services to analyze (None for all)
        organization_id: AWS Organizations ID (optional)
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Multi-account usage analysis results
            - message: Status message
            
    Example:
        >>> accounts = ["123456789012", "123456789013", "123456789014"]
        >>> result = aggregate_multi_account_usage(accounts, region="us-east-1")
        >>> print(f"Total accounts: {result['data']['total_accounts']}")
        >>> print(f"Consolidated spend: ${result['data']['consolidated_usage']['total_on_demand_spend']}")
    """
    from services.cost_explorer import get_database_usage_by_service
    from datetime import datetime, timedelta
    
    logger.info(f"Starting multi-account usage aggregation: {len(account_ids)} accounts, "
                f"region={region}, lookback_period={lookback_period_days}")
    
    # Validate inputs
    if not account_ids:
        logger.error("No account IDs provided")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "At least one account ID is required",
            "guidance": "Provide a list of AWS account IDs to analyze"
        }
    
    if lookback_period_days not in [30, 60, 90]:
        logger.error(f"Invalid lookback period: {lookback_period_days}")
        return {
            "status": "error",
            "message": f"Lookback period must be 30, 60, or 90 days",
            "error_code": "ValidationError",
            "provided_value": lookback_period_days,
            "valid_values": [30, 60, 90]
        }
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_period_days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Initialize aggregation variables
    account_level_usage = {}
    failed_accounts = []
    successful_accounts = []
    
    # Consolidated totals
    consolidated_total_spend = 0.0
    consolidated_service_breakdown = {}
    consolidated_region_breakdown = {}
    consolidated_instance_family_breakdown = {}
    
    # Process each account
    for account_id in account_ids:
        logger.info(f"Analyzing account {account_id}")
        
        try:
            # Get usage data for this account
            # Note: This requires cross-account permissions or assume role capability
            usage_result = get_database_usage_by_service(
                start_date=start_date_str,
                end_date=end_date_str,
                services=services,
                region=region,
                granularity="DAILY",
                account_id=account_id  # Pass account ID for cross-account analysis
            )
            
            if usage_result.get("status") != "success":
                logger.warning(f"Failed to retrieve usage for account {account_id}: {usage_result.get('message')}")
                failed_accounts.append({
                    "account_id": account_id,
                    "error": usage_result.get("message", "Unknown error"),
                    "error_code": usage_result.get("error_code", "UnknownError")
                })
                continue
            
            # Process usage data for this account
            usage_data = usage_result.get("data", {})
            account_total_spend = usage_data.get("total_cost", 0.0)
            account_service_breakdown = usage_data.get("service_breakdown", {})
            account_region_breakdown = usage_data.get("region_breakdown", {})
            account_instance_family_breakdown = usage_data.get("instance_family_breakdown", {})
            
            # Calculate account-level metrics
            total_hours = lookback_period_days * 24
            account_average_hourly_spend = account_total_spend / total_hours if total_hours > 0 else 0.0
            
            # Build account-level service breakdown
            account_service_usage = {}
            for service_name, service_cost in account_service_breakdown.items():
                service_hourly_spend = service_cost / total_hours if total_hours > 0 else 0.0
                account_service_usage[service_name] = ServiceUsage(
                    service_name=service_name,
                    total_spend=service_cost,
                    average_hourly_spend=service_hourly_spend,
                    instance_types={},
                    regions={}
                )
            
            # Create DatabaseUsageData for this account
            account_usage = DatabaseUsageData(
                total_on_demand_spend=account_total_spend,
                average_hourly_spend=account_average_hourly_spend,
                lookback_period_days=lookback_period_days,
                service_breakdown=account_service_usage,
                region_breakdown=account_region_breakdown,
                instance_family_breakdown=account_instance_family_breakdown,
                analysis_timestamp=datetime.now()
            )
            
            account_level_usage[account_id] = account_usage
            successful_accounts.append(account_id)
            
            # Aggregate into consolidated totals
            consolidated_total_spend += account_total_spend
            
            # Aggregate service breakdown
            for service_name, service_cost in account_service_breakdown.items():
                if service_name in consolidated_service_breakdown:
                    consolidated_service_breakdown[service_name] += service_cost
                else:
                    consolidated_service_breakdown[service_name] = service_cost
            
            # Aggregate region breakdown
            for region_name, region_cost in account_region_breakdown.items():
                if region_name in consolidated_region_breakdown:
                    consolidated_region_breakdown[region_name] += region_cost
                else:
                    consolidated_region_breakdown[region_name] = region_cost
            
            # Aggregate instance family breakdown
            for family, family_cost in account_instance_family_breakdown.items():
                if family in consolidated_instance_family_breakdown:
                    consolidated_instance_family_breakdown[family] += family_cost
                else:
                    consolidated_instance_family_breakdown[family] = family_cost
            
            logger.debug(f"Account {account_id}: ${account_total_spend:.2f} total spend")
            
        except Exception as e:
            logger.error(f"Error processing account {account_id}: {str(e)}")
            failed_accounts.append({
                "account_id": account_id,
                "error": str(e),
                "error_code": "ProcessingError"
            })
    
    # Check if we have any successful accounts
    if not successful_accounts:
        logger.error("No accounts processed successfully")
        return {
            "status": "error",
            "error_code": "NoDataError",
            "message": "Failed to retrieve usage data from any accounts",
            "failed_accounts": failed_accounts,
            "guidance": "Verify account IDs and cross-account permissions"
        }
    
    # Create consolidated usage data
    total_hours = lookback_period_days * 24
    consolidated_average_hourly_spend = consolidated_total_spend / total_hours if total_hours > 0 else 0.0
    
    # Build consolidated service breakdown
    consolidated_service_usage = {}
    for service_name, service_cost in consolidated_service_breakdown.items():
        service_hourly_spend = service_cost / total_hours if total_hours > 0 else 0.0
        consolidated_service_usage[service_name] = ServiceUsage(
            service_name=service_name,
            total_spend=service_cost,
            average_hourly_spend=service_hourly_spend,
            instance_types={},
            regions={}
        )
    
    consolidated_usage = DatabaseUsageData(
        total_on_demand_spend=consolidated_total_spend,
        average_hourly_spend=consolidated_average_hourly_spend,
        lookback_period_days=lookback_period_days,
        service_breakdown=consolidated_service_usage,
        region_breakdown=consolidated_region_breakdown,
        instance_family_breakdown=consolidated_instance_family_breakdown,
        analysis_timestamp=datetime.now()
    )
    
    # Calculate shared savings potential
    # Shared savings plans can benefit multiple accounts, potentially providing better rates
    shared_savings_potential = 0.0
    if consolidated_average_hourly_spend > 0:
        # Estimate 5-10% additional savings from shared commitment vs individual commitments
        individual_commitments_total = sum(
            account.average_hourly_spend * 0.85  # 85% coverage target per account
            for account in account_level_usage.values()
        )
        shared_commitment_optimal = consolidated_average_hourly_spend * 0.85
        
        # Shared savings plans may allow for better utilization and lower total commitment
        if individual_commitments_total > shared_commitment_optimal:
            shared_savings_potential = (individual_commitments_total - shared_commitment_optimal) * 8760 * 0.2  # 20% savings rate
    
    # Identify cross-account optimization opportunities
    cross_account_opportunities = []
    
    # Opportunity 1: Accounts with complementary usage patterns
    if len(successful_accounts) > 1:
        # Find accounts with different peak usage times (simplified analysis)
        high_usage_accounts = [
            account_id for account_id, usage in account_level_usage.items()
            if usage.average_hourly_spend > consolidated_average_hourly_spend * 0.8
        ]
        low_usage_accounts = [
            account_id for account_id, usage in account_level_usage.items()
            if usage.average_hourly_spend < consolidated_average_hourly_spend * 0.3
        ]
        
        if high_usage_accounts and low_usage_accounts:
            cross_account_opportunities.append({
                "type": "complementary_usage_patterns",
                "description": f"{len(high_usage_accounts)} high-usage and {len(low_usage_accounts)} low-usage accounts",
                "opportunity": "Shared savings plans can leverage usage diversity for better utilization",
                "potential_benefit": "5-15% additional savings through shared commitment optimization",
                "high_usage_accounts": high_usage_accounts,
                "low_usage_accounts": low_usage_accounts
            })
    
    # Opportunity 2: Service consolidation opportunities
    service_distribution = {}
    for account_id, usage in account_level_usage.items():
        for service_name, service_usage in usage.service_breakdown.items():
            if service_name not in service_distribution:
                service_distribution[service_name] = []
            service_distribution[service_name].append({
                "account_id": account_id,
                "spend": service_usage.total_spend
            })
    
    # Find services used across multiple accounts
    multi_account_services = {
        service: accounts for service, accounts in service_distribution.items()
        if len(accounts) > 1
    }
    
    if multi_account_services:
        cross_account_opportunities.append({
            "type": "service_consolidation",
            "description": f"{len(multi_account_services)} services used across multiple accounts",
            "opportunity": "Consolidate similar workloads for better savings plan utilization",
            "services": list(multi_account_services.keys()),
            "potential_benefit": "10-20% better utilization through workload consolidation"
        })
    
    # Create MultiAccountUsageData object
    multi_account_data = MultiAccountUsageData(
        organization_id=organization_id,
        total_accounts=len(successful_accounts),
        account_level_usage=account_level_usage,
        consolidated_usage=consolidated_usage,
        shared_savings_potential=shared_savings_potential,
        cross_account_optimization_opportunities=cross_account_opportunities,
        analysis_timestamp=datetime.now()
    )
    
    logger.info(f"Multi-account aggregation complete: {len(successful_accounts)} accounts, "
                f"${consolidated_total_spend:.2f} total spend, "
                f"${shared_savings_potential:.2f} shared savings potential")
    
    # Convert to dictionary format for response
    account_level_dict = {}
    for account_id, usage in account_level_usage.items():
        account_level_dict[account_id] = {
            "total_on_demand_spend": usage.total_on_demand_spend,
            "average_hourly_spend": usage.average_hourly_spend,
            "lookback_period_days": usage.lookback_period_days,
            "service_breakdown": {
                name: {
                    "service_name": svc.service_name,
                    "total_spend": svc.total_spend,
                    "average_hourly_spend": svc.average_hourly_spend,
                    "instance_types": svc.instance_types,
                    "regions": svc.regions
                }
                for name, svc in usage.service_breakdown.items()
            },
            "region_breakdown": usage.region_breakdown,
            "instance_family_breakdown": usage.instance_family_breakdown,
            "analysis_timestamp": usage.analysis_timestamp.isoformat()
        }
    
    consolidated_dict = {
        "total_on_demand_spend": consolidated_usage.total_on_demand_spend,
        "average_hourly_spend": consolidated_usage.average_hourly_spend,
        "lookback_period_days": consolidated_usage.lookback_period_days,
        "service_breakdown": {
            name: {
                "service_name": svc.service_name,
                "total_spend": svc.total_spend,
                "average_hourly_spend": svc.average_hourly_spend,
                "instance_types": svc.instance_types,
                "regions": svc.regions
            }
            for name, svc in consolidated_usage.service_breakdown.items()
        },
        "region_breakdown": consolidated_usage.region_breakdown,
        "instance_family_breakdown": consolidated_usage.instance_family_breakdown,
        "analysis_timestamp": consolidated_usage.analysis_timestamp.isoformat()
    }
    
    return {
        "status": "success",
        "data": {
            "organization_id": multi_account_data.organization_id,
            "total_accounts": multi_account_data.total_accounts,
            "successful_accounts": successful_accounts,
            "failed_accounts": failed_accounts,
            "account_level_usage": account_level_dict,
            "consolidated_usage": consolidated_dict,
            "shared_savings_potential": round(multi_account_data.shared_savings_potential, 2),
            "cross_account_optimization_opportunities": multi_account_data.cross_account_optimization_opportunities,
            "analysis_timestamp": multi_account_data.analysis_timestamp.isoformat(),
            "summary": {
                "total_consolidated_spend": round(consolidated_total_spend, 2),
                "average_spend_per_account": round(consolidated_total_spend / len(successful_accounts), 2) if successful_accounts else 0.0,
                "consolidated_hourly_spend": round(consolidated_average_hourly_spend, 2),
                "services_analyzed": list(consolidated_service_breakdown.keys()),
                "regions_analyzed": list(consolidated_region_breakdown.keys()),
                "cross_account_opportunities": len(cross_account_opportunities)
            }
        },
        "message": f"Successfully aggregated usage data from {len(successful_accounts)} accounts. "
                  f"Total spend: ${consolidated_total_spend:,.2f}, "
                  f"Shared savings potential: ${shared_savings_potential:,.2f}"
    }


def generate_multi_account_recommendations(
    multi_account_usage: Dict[str, Any],
    commitment_terms: List[str] = None,
    payment_options: List[str] = None
) -> Dict[str, Any]:
    """
    Generate both account-level and organization-level Database Savings Plans recommendations.
    
    This function creates recommendations at multiple levels:
    1. Individual account recommendations for account-specific workloads
    2. Organization-level shared savings plans for cross-account benefits
    3. Hybrid strategies combining both approaches
    
    Args:
        multi_account_usage: Multi-account usage analysis results
        commitment_terms: List of commitment terms (default: ["1_YEAR"])
        payment_options: List of payment options (default: ["NO_UPFRONT"])
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Multi-level recommendations
            - message: Status message
            
    Example:
        >>> usage = aggregate_multi_account_usage(["123456789012", "123456789013"])
        >>> recommendations = generate_multi_account_recommendations(usage["data"])
        >>> org_recs = recommendations["data"]["organization_level"]
        >>> account_recs = recommendations["data"]["account_level"]
    """
    logger.info("Generating multi-account Database Savings Plans recommendations")
    
    if commitment_terms is None:
        commitment_terms = ["1_YEAR"]
    if payment_options is None:
        payment_options = ["NO_UPFRONT"]
    
    # Validate input data
    if not multi_account_usage:
        logger.error("Multi-account usage data is empty")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Multi-account usage data is required",
            "guidance": "Call aggregate_multi_account_usage() first"
        }
    
    account_level_usage = multi_account_usage.get("account_level_usage", {})
    consolidated_usage = multi_account_usage.get("consolidated_usage", {})
    
    if not account_level_usage or not consolidated_usage:
        logger.error("Invalid multi-account usage data structure")
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Invalid multi-account usage data structure"
        }
    
    # Generate organization-level recommendations based on consolidated usage
    logger.info("Generating organization-level recommendations")
    org_recommendations_result = generate_savings_plans_recommendations(
        usage_data=consolidated_usage,
        commitment_terms=commitment_terms,
        payment_options=payment_options
    )
    
    if org_recommendations_result.get("status") != "success":
        logger.error(f"Failed to generate organization-level recommendations: {org_recommendations_result.get('message')}")
        return org_recommendations_result
    
    org_recommendations = org_recommendations_result.get("data", {}).get("recommendations", [])
    
    # Generate account-level recommendations for each account
    logger.info("Generating account-level recommendations")
    account_level_recommendations = {}
    total_account_level_savings = 0.0
    
    for account_id, account_usage in account_level_usage.items():
        logger.debug(f"Generating recommendations for account {account_id}")
        
        account_rec_result = generate_savings_plans_recommendations(
            usage_data=account_usage,
            commitment_terms=commitment_terms,
            payment_options=payment_options
        )
        
        if account_rec_result.get("status") == "success":
            account_recommendations = account_rec_result.get("data", {}).get("recommendations", [])
            account_level_recommendations[account_id] = account_recommendations
            
            # Sum up account-level savings
            if account_recommendations:
                best_account_rec = account_recommendations[0]  # Best recommendation
                total_account_level_savings += best_account_rec.get("estimated_annual_savings", 0.0)
        else:
            logger.warning(f"Failed to generate recommendations for account {account_id}")
            account_level_recommendations[account_id] = []
    
    # Calculate shared savings plans benefits
    shared_savings_plans = []
    consolidated_savings = 0.0
    
    if org_recommendations:
        best_org_rec = org_recommendations[0]
        consolidated_savings = best_org_rec.get("estimated_annual_savings", 0.0)
        
        # Calculate shared benefits
        shared_benefit = consolidated_savings - total_account_level_savings
        
        if shared_benefit > 0:
            shared_savings_plans.append({
                "type": "organization_shared_plan",
                "hourly_commitment": best_org_rec.get("hourly_commitment", 0.0),
                "commitment_term": best_org_rec.get("commitment_term", "1_YEAR"),
                "payment_option": best_org_rec.get("payment_option", "NO_UPFRONT"),
                "total_annual_savings": consolidated_savings,
                "shared_benefit": shared_benefit,
                "coverage_percentage": best_org_rec.get("projected_coverage", 0.0),
                "utilization_percentage": best_org_rec.get("projected_utilization", 0.0),
                "benefiting_accounts": list(account_level_usage.keys()),
                "rationale": f"Organization-level commitment provides ${shared_benefit:,.2f} additional annual savings compared to individual account commitments"
            })
    
    # Determine optimization strategy
    if consolidated_savings > total_account_level_savings * 1.1:  # >10% better
        optimization_strategy = "organization_level_preferred"
        strategy_rationale = f"Organization-level savings plans provide ${consolidated_savings - total_account_level_savings:,.2f} additional annual savings"
    elif total_account_level_savings > consolidated_savings * 1.05:  # >5% better
        optimization_strategy = "account_level_preferred"
        strategy_rationale = "Account-level commitments provide better optimization for diverse usage patterns"
    else:
        optimization_strategy = "hybrid_approach"
        strategy_rationale = "Consider hybrid approach with both organization-level and account-specific commitments"
    
    # Create MultiAccountRecommendation object
    multi_account_rec = MultiAccountRecommendation(
        organization_level=org_recommendations,
        account_level=account_level_recommendations,
        shared_savings_plans=shared_savings_plans,
        consolidated_savings=consolidated_savings,
        individual_account_savings={
            account_id: recs[0].get("estimated_annual_savings", 0.0) if recs else 0.0
            for account_id, recs in account_level_recommendations.items()
        },
        optimization_strategy=optimization_strategy
    )
    
    logger.info(f"Multi-account recommendations complete: "
                f"org savings=${consolidated_savings:.2f}, "
                f"account savings=${total_account_level_savings:.2f}, "
                f"strategy={optimization_strategy}")
    
    return {
        "status": "success",
        "data": {
            "organization_level": org_recommendations,
            "account_level": account_level_recommendations,
            "shared_savings_plans": shared_savings_plans,
            "consolidated_savings": round(consolidated_savings, 2),
            "individual_account_savings": {
                account_id: round(savings, 2)
                for account_id, savings in multi_account_rec.individual_account_savings.items()
            },
            "total_individual_savings": round(total_account_level_savings, 2),
            "shared_benefit": round(consolidated_savings - total_account_level_savings, 2),
            "optimization_strategy": optimization_strategy,
            "strategy_rationale": strategy_rationale,
            "recommendations_summary": {
                "total_accounts_analyzed": len(account_level_usage),
                "accounts_with_recommendations": len([recs for recs in account_level_recommendations.values() if recs]),
                "organization_recommendations_count": len(org_recommendations),
                "shared_savings_plans_count": len(shared_savings_plans),
                "best_approach": "organization_level" if consolidated_savings > total_account_level_savings else "account_level"
            }
        },
        "message": f"Generated multi-account recommendations: "
                  f"${consolidated_savings:,.2f} organization-level vs "
                  f"${total_account_level_savings:,.2f} account-level savings. "
                  f"Strategy: {optimization_strategy}"
    }


def calculate_shared_savings_benefits(
    multi_account_usage: Dict[str, Any],
    organization_recommendations: List[Dict[str, Any]],
    account_recommendations: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculate shared savings plans benefits across multiple accounts.
    
    Shared savings plans can provide benefits across multiple accounts within an
    AWS Organization, potentially offering better utilization and cost optimization
    than individual account commitments.
    
    Args:
        multi_account_usage: Multi-account usage analysis results
        organization_recommendations: Organization-level recommendations
        account_recommendations: Account-level recommendations by account ID
        
    Returns:
        Dictionary containing shared savings analysis
        
    Example:
        >>> shared_benefits = calculate_shared_savings_benefits(
        ...     usage_data, org_recs, account_recs
        ... )
        >>> print(f"Shared benefit: ${shared_benefits['data']['total_shared_benefit']}")
    """
    logger.info("Calculating shared savings plans benefits")
    
    # Extract data
    account_level_usage = multi_account_usage.get("account_level_usage", {})
    consolidated_usage = multi_account_usage.get("consolidated_usage", {})
    
    if not account_level_usage or not consolidated_usage:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Invalid multi-account usage data"
        }
    
    # Calculate individual account commitment totals
    individual_commitments = {}
    total_individual_commitment = 0.0
    total_individual_savings = 0.0
    
    for account_id, recommendations in account_recommendations.items():
        if recommendations:
            best_rec = recommendations[0]  # Best recommendation for this account
            commitment = best_rec.get("hourly_commitment", 0.0)
            savings = best_rec.get("estimated_annual_savings", 0.0)
            
            individual_commitments[account_id] = {
                "hourly_commitment": commitment,
                "annual_savings": savings,
                "coverage": best_rec.get("projected_coverage", 0.0),
                "utilization": best_rec.get("projected_utilization", 0.0)
            }
            
            total_individual_commitment += commitment
            total_individual_savings += savings
    
    # Calculate organization-level commitment
    org_commitment = 0.0
    org_savings = 0.0
    org_coverage = 0.0
    org_utilization = 0.0
    
    if organization_recommendations:
        best_org_rec = organization_recommendations[0]
        org_commitment = best_org_rec.get("hourly_commitment", 0.0)
        org_savings = best_org_rec.get("estimated_annual_savings", 0.0)
        org_coverage = best_org_rec.get("projected_coverage", 0.0)
        org_utilization = best_org_rec.get("projected_utilization", 0.0)
    
    # Calculate shared benefits
    commitment_efficiency = 0.0
    if total_individual_commitment > 0:
        commitment_efficiency = (total_individual_commitment - org_commitment) / total_individual_commitment * 100
    
    savings_benefit = org_savings - total_individual_savings
    
    # Calculate utilization benefits from shared commitment
    consolidated_hourly_spend = consolidated_usage.get("average_hourly_spend", 0.0)
    
    # Shared utilization can be better due to usage diversity across accounts
    individual_utilization_weighted = 0.0
    total_weight = 0.0
    
    for account_id, usage in account_level_usage.items():
        account_spend = usage.get("average_hourly_spend", 0.0)
        if account_spend > 0:
            account_commitment = individual_commitments.get(account_id, {}).get("hourly_commitment", 0.0)
            if account_commitment > 0:
                account_utilization = min(account_spend / account_commitment, 1.0) * 100
                individual_utilization_weighted += account_utilization * account_spend
                total_weight += account_spend
    
    avg_individual_utilization = individual_utilization_weighted / total_weight if total_weight > 0 else 0.0
    utilization_improvement = org_utilization - avg_individual_utilization
    
    # Identify accounts that benefit most from shared commitment
    benefiting_accounts = []
    for account_id, usage in account_level_usage.items():
        account_spend = usage.get("average_hourly_spend", 0.0)
        individual_rec = individual_commitments.get(account_id, {})
        individual_savings = individual_rec.get("annual_savings", 0.0)
        
        # Estimate this account's share of organization savings
        if consolidated_hourly_spend > 0:
            account_share = account_spend / consolidated_hourly_spend
            estimated_shared_savings = org_savings * account_share
            
            benefit = estimated_shared_savings - individual_savings
            
            benefiting_accounts.append({
                "account_id": account_id,
                "individual_savings": individual_savings,
                "estimated_shared_savings": estimated_shared_savings,
                "benefit": benefit,
                "account_share_percentage": account_share * 100
            })
    
    # Sort by benefit
    benefiting_accounts.sort(key=lambda x: x["benefit"], reverse=True)
    
    # Calculate risk factors for shared commitment
    risk_factors = []
    
    # Risk 1: Account usage volatility
    usage_volatility = []
    for account_id, usage in account_level_usage.items():
        account_spend = usage.get("average_hourly_spend", 0.0)
        if account_spend > 0:
            # Simplified volatility measure based on spend relative to average
            avg_spend_per_account = consolidated_hourly_spend / len(account_level_usage)
            volatility = abs(account_spend - avg_spend_per_account) / avg_spend_per_account
            usage_volatility.append(volatility)
    
    avg_volatility = sum(usage_volatility) / len(usage_volatility) if usage_volatility else 0.0
    
    if avg_volatility > 0.5:  # High volatility
        risk_factors.append({
            "type": "usage_volatility",
            "level": "high",
            "description": f"High usage volatility across accounts ({avg_volatility:.1%})",
            "mitigation": "Consider 85% coverage target with monitoring"
        })
    elif avg_volatility > 0.3:  # Medium volatility
        risk_factors.append({
            "type": "usage_volatility",
            "level": "medium",
            "description": f"Moderate usage volatility across accounts ({avg_volatility:.1%})",
            "mitigation": "Monitor usage patterns and adjust commitments quarterly"
        })
    
    # Risk 2: Account dependency
    if len(account_level_usage) < 3:
        risk_factors.append({
            "type": "account_dependency",
            "level": "medium",
            "description": "Few accounts in shared commitment increases dependency risk",
            "mitigation": "Ensure stable usage patterns before committing"
        })
    
    logger.info(f"Shared savings calculation complete: "
                f"${savings_benefit:.2f} benefit, "
                f"{commitment_efficiency:.1f}% commitment efficiency")
    
    return {
        "status": "success",
        "data": {
            "total_shared_benefit": round(savings_benefit, 2),
            "commitment_efficiency_percentage": round(commitment_efficiency, 2),
            "utilization_improvement": round(utilization_improvement, 2),
            "individual_vs_shared": {
                "individual_total_commitment": round(total_individual_commitment, 2),
                "individual_total_savings": round(total_individual_savings, 2),
                "individual_average_utilization": round(avg_individual_utilization, 2),
                "shared_commitment": round(org_commitment, 2),
                "shared_savings": round(org_savings, 2),
                "shared_utilization": round(org_utilization, 2)
            },
            "benefiting_accounts": benefiting_accounts,
            "risk_factors": risk_factors,
            "recommendation": {
                "approach": "shared" if savings_benefit > total_individual_savings * 0.1 else "individual",
                "rationale": f"Shared commitment provides ${savings_benefit:,.2f} additional benefit" if savings_benefit > 0 else "Individual commitments provide better optimization",
                "confidence": "high" if len(risk_factors) == 0 else "medium" if len(risk_factors) <= 2 else "low"
            }
        },
        "message": f"Shared savings analysis complete: ${savings_benefit:,.2f} benefit, "
                  f"{len(benefiting_accounts)} accounts analyzed, "
                  f"{len(risk_factors)} risk factors identified"
    }


# ============================================================================
# MCP Runbook Wrapper Functions
# ============================================================================

import asyncio
import time
from mcp.types import TextContent

from utils.error_handler import ResponseFormatter, handle_aws_error
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import create_task
from utils.documentation_links import add_documentation_links


@handle_aws_error
async def run_database_savings_plans_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Comprehensive Database Savings Plans analysis with recommendations.
    
    This function provides automated recommendations for Database Savings Plans based on
    historical on-demand database usage. It analyzes usage patterns across all supported
    database services and generates optimal hourly commitment recommendations for 1-year
    terms with no upfront payment.
    
    MCP Tool Parameters:
        - region (optional): AWS region to analyze (None for all regions)
        - lookback_period_days (default: 30): Analysis period (30, 60, or 90 days)
        - services (optional): List of database services to analyze (None for all)
        - include_ri_comparison (default: true): Compare with Reserved Instances
        - include_existing_analysis (default: true): Analyze existing commitments
        - store_results (default: true): Store results in session database
        
    Returns:
        List[TextContent]: Formatted analysis results with recommendations
        
    Example Usage:
        >>> # Basic analysis for all services in all regions
        >>> await run_database_savings_plans_analysis({})
        
        >>> # Focused analysis for specific region and services
        >>> await run_database_savings_plans_analysis({
        ...     "region": "us-east-1",
        ...     "services": ["rds", "aurora", "dynamodb"],
        ...     "lookback_period_days": 60
        ... })
    """
    start_time = time.time()
    
    # Extract and validate parameters
    region = arguments.get("region")
    lookback_period_days = arguments.get("lookback_period_days", 30)
    services = arguments.get("services")
    include_ri_comparison = arguments.get("include_ri_comparison", True)
    include_existing_analysis = arguments.get("include_existing_analysis", True)
    store_results = arguments.get("store_results", True)
    
    logger.info(f"Starting Database Savings Plans analysis: region={region}, "
                f"lookback_period={lookback_period_days}, services={services}")
    
    # Initialize ServiceOrchestrator for parallel execution
    orchestrator = ServiceOrchestrator()
    
    # Define parallel service calls for comprehensive analysis
    service_calls = []
    
    # 1. Database usage analysis (always required)
    service_calls.append({
        'service': 'database_savings_plans',
        'operation': 'analyze_usage',
        'function': analyze_database_usage,
        'kwargs': {
            'region': region,
            'lookback_period_days': lookback_period_days,
            'services': services
        },
        'timeout': 45.0,
        'priority': 1
    })
    
    # 2. Existing commitments analysis (if requested)
    if include_existing_analysis:
        service_calls.append({
            'service': 'database_savings_plans',
            'operation': 'analyze_existing_commitments',
            'function': analyze_existing_commitments,
            'kwargs': {
                'region': region,
                'lookback_period_days': lookback_period_days
            },
            'timeout': 30.0,
            'priority': 2
        })
    
    # Execute parallel analysis
    execution_results = orchestrator.execute_parallel_analysis(
        service_calls=service_calls,
        store_results=store_results,
        timeout=90.0
    )
    
    # Process results from parallel execution
    usage_analysis_result = None
    existing_commitments_result = None
    
    for task_id, task_result in execution_results['results'].items():
        if task_result['operation'] == 'analyze_usage' and task_result['status'] == 'success':
            # Get the actual data from the task result
            # The parallel executor stores the function return value in 'data' field
            # But we need to handle both direct data and nested data structures
            if 'data' in task_result and task_result['data']:
                usage_analysis_result = task_result['data']
            else:
                # Fallback: call the function directly if parallel execution didn't work
                usage_analysis_result = analyze_database_usage(
                    region=region,
                    lookback_period_days=lookback_period_days,
                    services=services
                )
        elif task_result['operation'] == 'analyze_existing_commitments' and task_result['status'] == 'success':
            if 'data' in task_result and task_result['data']:
                existing_commitments_result = task_result['data']
            else:
                # Fallback: call the function directly if parallel execution didn't work
                existing_commitments_result = analyze_existing_commitments(
                    region=region,
                    lookback_period_days=lookback_period_days
                )
    
    # If usage analysis failed, return error
    if not usage_analysis_result or usage_analysis_result.get('status') != 'success':
        error_message = "Failed to retrieve database usage data"
        if usage_analysis_result and usage_analysis_result.get('message'):
            error_message = usage_analysis_result['message']
        
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                Exception(error_message),
                "run_database_savings_plans_analysis"
            )
        )
    
    usage_data = usage_analysis_result.get('data', {})
    
    # Generate savings plans recommendations based on usage data
    logger.info("Generating Database Savings Plans recommendations")
    recommendations_result = generate_savings_plans_recommendations(
        usage_data=usage_data,
        commitment_terms=["1_YEAR"],  # Only 1-year terms supported
        payment_options=["NO_UPFRONT"]  # Only No Upfront supported
    )
    
    # Generate Reserved Instance comparison if requested
    ri_comparison_result = None
    if include_ri_comparison and usage_data.get('average_hourly_spend', 0) > 0:
        logger.info("Generating Reserved Instance comparison")
        ri_comparison_result = compare_with_reserved_instances(
            usage_data=usage_data,
            services=services or ["rds", "aurora"]
        )
    
    # Compile comprehensive results
    analysis_results = {
        'usage_analysis': usage_analysis_result,
        'recommendations': recommendations_result,
        'existing_commitments': existing_commitments_result,
        'ri_comparison': ri_comparison_result,
        'execution_summary': execution_results
    }
    
    # Generate summary insights with optimization hierarchy
    insights = []
    
    # Usage insights
    total_spend = usage_data.get('total_on_demand_spend', 0)
    avg_hourly_spend = usage_data.get('average_hourly_spend', 0)
    
    if total_spend > 0:
        insights.append(f"💰 Total on-demand database spend: ${total_spend:,.2f} over {lookback_period_days} days")
        insights.append(f"⏱️  Average hourly spend: ${avg_hourly_spend:.2f}/hour")
        
        # PRIORITY 1: Operational optimizations (immediate savings, no commitment)
        insights.append("🎯 OPTIMIZATION PRIORITY HIERARCHY:")
        insights.append("   1️⃣ RIGHT-SIZING: Review oversized instances for immediate 20-50% savings")
        insights.append("   2️⃣ MODERNIZATION: Upgrade to latest generation instances (better price/performance)")
        insights.append("   3️⃣ SERVERLESS: Consider Aurora Serverless v2 for variable workloads")
        insights.append("   4️⃣ STORAGE OPTIMIZATION: Review storage types and unused allocated storage")
        insights.append("   5️⃣ COMMITMENT SAVINGS: Database Savings Plans (after operational optimizations)")
        insights.append("")
        insights.append("⚠️  IMPORTANT: Complete operational optimizations BEFORE commitment purchases")
    
    # Recommendations insights (with optimization hierarchy reminder)
    if recommendations_result.get('status') == 'success':
        recommendations = recommendations_result.get('data', {}).get('recommendations', [])
        if recommendations:
            best_rec = recommendations[0]  # Sorted by savings
            insights.append("")
            insights.append("📋 DATABASE SAVINGS PLANS RECOMMENDATIONS:")
            insights.append(f"💡 Best recommendation: ${best_rec['hourly_commitment']:.2f}/hour commitment")
            insights.append(f"💵 Estimated annual savings: ${best_rec['estimated_annual_savings']:,.2f}")
            insights.append(f"📊 Projected coverage: {best_rec['projected_coverage']:.1f}%")
            insights.append("")
            insights.append("⚠️  BEFORE PURCHASING: Ensure operational optimizations are complete")
            insights.append("   • Right-size oversized instances (20-50% immediate savings)")
            insights.append("   • Modernize to latest generation instances")
            insights.append("   • Evaluate Aurora Serverless v2 for variable workloads")
            insights.append("   • Optimize storage allocation and types")
        else:
            insights.append("ℹ️  No Database Savings Plans recommendations available")
            eligible_spend = recommendations_result.get('data', {}).get('eligible_hourly_spend', 0)
            excluded_spend = recommendations_result.get('data', {}).get('excluded_hourly_spend', 0)
            if excluded_spend > eligible_spend:
                insights.append(f"⚠️  Most usage (${excluded_spend:.2f}/hour) requires Reserved Instances")
                insights.append("💡 Focus on operational optimizations first for immediate savings")
    
    # Existing commitments insights
    if existing_commitments_result and existing_commitments_result.get('status') == 'success':
        existing_data = existing_commitments_result.get('data', {})
        summary = existing_data.get('summary', {})
        total_plans = summary.get('total_plans', 0)
        
        if total_plans > 0:
            avg_utilization = summary.get('average_utilization', 0)
            avg_coverage = summary.get('average_coverage', 0)
            insights.append(f"📋 Existing plans: {total_plans} Database Savings Plans active")
            insights.append(f"📈 Average utilization: {avg_utilization:.1f}%")
            insights.append(f"🎯 Average coverage: {avg_coverage:.1f}%")
        else:
            insights.append("📋 No existing Database Savings Plans found")
    
    # RI comparison insights (with optimization hierarchy context)
    if ri_comparison_result and ri_comparison_result.get('status') == 'success':
        comparison_data = ri_comparison_result.get('data', {})
        summary = comparison_data.get('summary', {})
        total_sp_savings = summary.get('total_database_savings_plans_savings', 0)
        total_ri_savings = summary.get('total_reserved_instances_savings', 0)
        
        if total_sp_savings > 0 or total_ri_savings > 0:
            insights.append("")
            insights.append("📊 COMMITMENT SAVINGS POTENTIAL (after operational optimizations):")
            insights.append(f"   • Database Savings Plans: ${total_sp_savings:,.2f}/year")
            insights.append(f"   • Reserved Instances: ${total_ri_savings:,.2f}/year")
            insights.append("⚠️  These projections assume current usage - optimize workloads first")
    
    # Create final response
    execution_time = time.time() - start_time
    
    # Add documentation links
    result_with_docs = add_documentation_links(
        analysis_results,
        service_type="rds",
        finding_type="database_savings_plans"
    )
    
    # Add analysis insights
    result_with_docs['analysis_insights'] = insights
    result_with_docs['analysis_mode'] = 'automated_recommendations'
    result_with_docs['optimization_hierarchy'] = {
        'priority_order': [
            "1. Right-sizing (immediate savings, no commitment)",
            "2. Modernization to latest generation instances", 
            "3. Serverless adoption for variable workloads",
            "4. Storage optimization and cleanup",
            "5. Database Savings Plans (after operational optimizations)"
        ],
        'rationale': "Operational optimizations provide immediate savings without long-term commitments and should be completed before purchasing Database Savings Plans"
    }
    result_with_docs['limitations'] = {
        'database_savings_plans': [
            "Latest-generation instances only (M7, R7, R8, etc.)",
            "1-year terms with No Upfront payment only",
            "Cannot be resold or exchanged",
            "Covers compute usage only - storage, backups, data transfer billed separately",
            "ElastiCache support limited to Valkey engine"
        ],
        'coverage_strategy': "Recommendations target 85-90% coverage with 15% buffer for commitment risk",
        'prerequisite': "Complete right-sizing, modernization, and serverless adoption BEFORE commitment purchases"
    }
    
    # Store results in historical database if requested
    if store_results and orchestrator.session_id:
        try:
            storage_result = store_analysis_result(
                session_id=orchestrator.session_id,
                analysis_type="recommendations",
                analysis_data=recommendations_result.get('data', {}),
                region=region,
                lookback_period_days=lookback_period_days,
                metadata={
                    'services_analyzed': services or 'all',
                    'include_ri_comparison': include_ri_comparison,
                    'include_existing_analysis': include_existing_analysis,
                    'execution_time': execution_time,
                    'total_insights': len(insights)
                }
            )
            if storage_result.get('status') == 'success':
                logger.info(f"Stored recommendations analysis: {storage_result['data']['analysis_id']}")
            else:
                logger.warning(f"Failed to store recommendations analysis: {storage_result.get('message')}")
        except Exception as e:
            logger.warning(f"Failed to store recommendations analysis: {e}")
    
    return ResponseFormatter.to_text_content(
        ResponseFormatter.success_response(
            data=result_with_docs,
            message=f"Database Savings Plans analysis completed successfully. {len(insights)} key insights identified.",
            analysis_type="database_savings_plans_comprehensive",
            execution_time=execution_time,
            metadata={
                'session_id': orchestrator.session_id,
                'region': region,
                'lookback_period_days': lookback_period_days,
                'services_analyzed': services or 'all',
                'parallel_tasks': execution_results['total_tasks'],
                'successful_tasks': execution_results['successful']
            }
        )
    )


@handle_aws_error
async def run_purchase_analyzer(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Purchase analyzer mode: model custom commitment scenarios.
    
    This function allows users to specify custom hourly commitment amounts and simulate
    their projected impact on cost, coverage, and utilization. It's designed for
    incremental commitment strategies and future usage scenario modeling.
    
    IMPORTANT LIMITATIONS:
    - Database Savings Plans currently only support 1-year terms with No Upfront payment
    - Coverage recommendations target 85-90% to account for dynamic usage patterns
    - Cannot be resold or exchanged - commitment risk must be carefully considered
    - Applies to latest-generation instance families only (M7, R7, R8, etc.)
    
    MCP Tool Parameters:
        - hourly_commitment (required): Custom hourly commitment amount in USD
        - region (optional): AWS region to analyze (None for all regions)
        - lookback_period_days (default: 30): Analysis period for baseline usage
        - commitment_term (default: "1_YEAR"): Commitment term (only 1_YEAR supported)
        - payment_option (default: "NO_UPFRONT"): Payment option (only NO_UPFRONT supported)
        - adjusted_usage_projection (optional): Adjusted average hourly usage for future scenarios
        - include_usage_analysis (default: true): Include current usage analysis for context
        - store_results (default: true): Store results in session database
        
    Returns:
        List[TextContent]: Formatted purchase analyzer simulation results
        
    Example Usage:
        >>> # Basic purchase analyzer with $10/hour commitment
        >>> await run_purchase_analyzer({"hourly_commitment": 10.0})
        
        >>> # Advanced scenario with future usage projection
        >>> await run_purchase_analyzer({
        ...     "hourly_commitment": 15.0,
        ...     "region": "us-east-1",
        ...     "adjusted_usage_projection": 18.0,
        ...     "lookback_period_days": 60
        ... })
    """
    start_time = time.time()
    
    # Extract and validate parameters
    hourly_commitment = arguments.get("hourly_commitment")
    region = arguments.get("region")
    lookback_period_days = arguments.get("lookback_period_days", 30)
    commitment_term = arguments.get("commitment_term", "1_YEAR")
    payment_option = arguments.get("payment_option", "NO_UPFRONT")
    adjusted_usage_projection = arguments.get("adjusted_usage_projection")
    include_usage_analysis = arguments.get("include_usage_analysis", True)
    store_results = arguments.get("store_results", True)
    
    # Validate required parameters
    if hourly_commitment is None:
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                ValueError("hourly_commitment parameter is required"),
                "run_purchase_analyzer"
            )
        )
    
    logger.info(f"Starting purchase analyzer: commitment=${hourly_commitment}/hour, "
                f"term={commitment_term}, payment={payment_option}")
    
    # Initialize ServiceOrchestrator for parallel execution
    orchestrator = ServiceOrchestrator()
    
    # Get current usage data for baseline comparison
    usage_analysis_result = None
    if include_usage_analysis:
        logger.info("Retrieving current database usage for baseline comparison")
        usage_analysis_result = analyze_database_usage(
            region=region,
            lookback_period_days=lookback_period_days,
            services=None  # Analyze all services for comprehensive baseline
        )
        
        if usage_analysis_result.get('status') != 'success':
            return ResponseFormatter.to_text_content(
                ResponseFormatter.error_response(
                    Exception(f"Failed to retrieve baseline usage data: {usage_analysis_result.get('message')}"),
                    "run_purchase_analyzer"
                )
            )
    
    usage_data = usage_analysis_result.get('data', {}) if usage_analysis_result else {}
    
    # Run custom commitment analysis
    logger.info(f"Analyzing custom commitment scenario: ${hourly_commitment}/hour")
    purchase_analysis_result = analyze_custom_commitment(
        hourly_commitment=hourly_commitment,
        usage_data=usage_data,
        commitment_term=commitment_term,
        payment_option=payment_option,
        adjusted_usage_projection=adjusted_usage_projection
    )
    
    if purchase_analysis_result.get('status') != 'success':
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                Exception(f"Purchase analyzer simulation failed: {purchase_analysis_result.get('message')}"),
                "run_purchase_analyzer"
            )
        )
    
    # Generate comparison scenarios (multiple commitment levels for context)
    comparison_scenarios = []
    if usage_data.get('average_hourly_spend', 0) > 0:
        base_hourly_spend = usage_data['average_hourly_spend']
        
        # Generate scenarios at different coverage levels
        scenario_commitments = [
            base_hourly_spend * 0.7,   # 70% coverage
            base_hourly_spend * 0.85,  # 85% coverage (recommended)
            base_hourly_spend * 0.9,   # 90% coverage (recommended max)
            base_hourly_spend * 1.0,   # 100% coverage (risky)
        ]
        
        for scenario_commitment in scenario_commitments:
            if abs(scenario_commitment - hourly_commitment) > 0.01:  # Skip if too close to main scenario
                scenario_result = analyze_custom_commitment(
                    hourly_commitment=scenario_commitment,
                    usage_data=usage_data,
                    commitment_term=commitment_term,
                    payment_option=payment_option,
                    adjusted_usage_projection=adjusted_usage_projection
                )
                
                if scenario_result.get('status') == 'success':
                    scenario_data = scenario_result.get('data', {})
                    comparison_scenarios.append({
                        'hourly_commitment': scenario_commitment,
                        'projected_coverage': scenario_data.get('projected_coverage', 0),
                        'projected_utilization': scenario_data.get('projected_utilization', 0),
                        'estimated_annual_savings': scenario_data.get('estimated_annual_savings', 0),
                        'recommendation': scenario_data.get('recommendation', '')
                    })
    
    # Sort comparison scenarios by savings
    comparison_scenarios.sort(key=lambda x: x['estimated_annual_savings'], reverse=True)
    
    # Compile comprehensive results
    purchase_data = purchase_analysis_result.get('data', {})
    analysis_results = {
        'purchase_analysis': purchase_analysis_result,
        'baseline_usage': usage_analysis_result,
        'comparison_scenarios': comparison_scenarios[:3],  # Top 3 alternatives
        'analysis_mode': 'purchase_analyzer'
    }
    
    # Generate purchase analyzer insights with optimization hierarchy
    insights = []
    
    # Optimization hierarchy reminder
    insights.append("🎯 OPTIMIZATION HIERARCHY REMINDER:")
    insights.append("   ⚠️  COMPLETE THESE FIRST before purchasing commitments:")
    insights.append("   1️⃣ Right-size oversized instances (immediate 20-50% savings)")
    insights.append("   2️⃣ Modernize to latest generation instances")
    insights.append("   3️⃣ Evaluate Aurora Serverless v2 for variable workloads")
    insights.append("   4️⃣ Optimize storage allocation and cleanup unused storage")
    insights.append("")
    
    # Main scenario insights
    coverage = purchase_data.get('projected_coverage', 0)
    utilization = purchase_data.get('projected_utilization', 0)
    annual_savings = purchase_data.get('estimated_annual_savings', 0)
    unused_commitment = purchase_data.get('unused_commitment', 0)
    uncovered_usage = purchase_data.get('uncovered_usage', 0)
    
    insights.append("📋 PURCHASE ANALYZER SIMULATION:")
    insights.append(f"💰 Custom commitment: ${hourly_commitment:.2f}/hour (${hourly_commitment * 8760:,.0f}/year)")
    insights.append(f"📊 Projected coverage: {coverage:.1f}%")
    insights.append(f"📈 Projected utilization: {utilization:.1f}%")
    
    if annual_savings > 0:
        insights.append(f"💵 Estimated annual savings: ${annual_savings:,.2f}")
        savings_percentage = (annual_savings / (purchase_data.get('projected_annual_cost', 1) + annual_savings)) * 100
        insights.append(f"📉 Cost reduction: {savings_percentage:.1f}%")
    else:
        insights.append("⚠️  No savings projected - commitment exceeds usage")
    
    # Risk assessment insights
    if unused_commitment > 0:
        insights.append(f"⚠️  Unused commitment: ${unused_commitment:.2f}/hour (${unused_commitment * 8760:,.0f}/year)")
    
    if uncovered_usage > 0:
        insights.append(f"💡 Uncovered usage: ${uncovered_usage:.2f}/hour remains at on-demand rates")
    
    # Coverage guidance
    if coverage > 90:
        insights.append("🚨 Coverage >90% may be risky for dynamic workloads - consider 85-90% target")
    elif coverage >= 85:
        insights.append("✅ Coverage in recommended 85-90% range for dynamic workloads")
    elif coverage < 85:
        insights.append("💡 Coverage <85% - consider increasing commitment for better savings")
    
    # Utilization guidance
    if utilization < 70:
        insights.append("⚠️  Low utilization - consider reducing commitment to avoid waste")
    elif utilization > 95:
        insights.append("✅ High utilization - commitment well-matched to usage")
    
    # Usage projection insights
    current_usage = purchase_data.get('current_usage', 0)
    projected_usage = purchase_data.get('projected_usage', 0)
    
    if adjusted_usage_projection and abs(projected_usage - current_usage) > 0.01:
        change_percentage = ((projected_usage - current_usage) / current_usage) * 100
        if change_percentage > 0:
            insights.append(f"📈 Usage projection: {change_percentage:+.1f}% increase to ${projected_usage:.2f}/hour")
        else:
            insights.append(f"📉 Usage projection: {change_percentage:+.1f}% decrease to ${projected_usage:.2f}/hour")
    
    # Comparison scenarios insights
    if comparison_scenarios:
        best_alternative = comparison_scenarios[0]
        if best_alternative['estimated_annual_savings'] > annual_savings:
            insights.append(f"💡 Alternative: ${best_alternative['hourly_commitment']:.2f}/hour could save ${best_alternative['estimated_annual_savings']:,.2f}/year")
    
    # Add commitment risk warnings and optimization prerequisites
    insights.append("")
    insights.append("⚠️  COMMITMENT RISKS & PREREQUISITES:")
    insights.append("   • Database Savings Plans cannot be resold or exchanged")
    insights.append("   • 1-year commitment with No Upfront payment only")
    insights.append("   • Complete operational optimizations BEFORE purchasing")
    insights.append("   • Projections based on current usage - optimize workloads first")
    
    # Create final response
    execution_time = time.time() - start_time
    
    # Add documentation links
    result_with_docs = add_documentation_links(
        analysis_results,
        service_type="rds",
        finding_type="database_savings_plans_purchase_analyzer"
    )
    
    # Add analysis insights and metadata
    result_with_docs['analysis_insights'] = insights
    
    # Add key fields directly to data for backward compatibility with tests
    result_with_docs['hourly_commitment'] = hourly_commitment
    result_with_docs['commitment_term'] = commitment_term
    result_with_docs['payment_option'] = payment_option
    result_with_docs['projected_coverage'] = coverage
    result_with_docs['projected_utilization'] = utilization
    result_with_docs['estimated_annual_savings'] = annual_savings
    
    result_with_docs['purchase_analyzer_summary'] = {
        'hourly_commitment': hourly_commitment,
        'commitment_term': commitment_term,
        'payment_option': payment_option,
        'projected_coverage': coverage,
        'projected_utilization': utilization,
        'estimated_annual_savings': annual_savings,
        'recommendation': purchase_data.get('recommendation', ''),
        'risk_level': 'high' if coverage > 90 or utilization < 70 else 'medium' if coverage < 85 else 'low'
    }
    result_with_docs['optimization_hierarchy'] = {
        'priority_order': [
            "1. Right-sizing (immediate savings, no commitment)",
            "2. Modernization to latest generation instances", 
            "3. Serverless adoption for variable workloads",
            "4. Storage optimization and cleanup",
            "5. Database Savings Plans (after operational optimizations)"
        ],
        'prerequisite_warning': "Complete operational optimizations BEFORE purchasing commitments"
    }
    result_with_docs['limitations'] = {
        'database_savings_plans': [
            "1-year terms with No Upfront payment only",
            "Cannot be resold or exchanged - commitment risk",
            "Latest-generation instances only (M7, R7, R8, etc.)",
            "Covers compute usage only",
            "ElastiCache support limited to Valkey engine"
        ],
        'purchase_analyzer': [
            "Projections based on historical usage patterns",
            "Actual savings may vary with usage changes",
            "Consider 85-90% coverage target for dynamic workloads",
            "PREREQUISITE: Complete right-sizing and modernization first"
        ]
    }
    
    # Store results if requested
    if store_results:
        try:
            # Store purchase analyzer results in historical database
            storage_result = store_analysis_result(
                session_id=orchestrator.session_id,
                analysis_type="purchase_analyzer",
                analysis_data=purchase_analysis_result.get('data', {}),
                region=region,
                lookback_period_days=lookback_period_days,
                metadata={
                    'hourly_commitment': hourly_commitment,
                    'commitment_term': commitment_term,
                    'payment_option': payment_option,
                    'adjusted_usage_projection': adjusted_usage_projection,
                    'execution_time': execution_time
                }
            )
            if storage_result.get('status') == 'success':
                logger.info(f"Stored purchase analyzer analysis: {storage_result['data']['analysis_id']}")
            else:
                logger.warning(f"Failed to store purchase analyzer analysis: {storage_result.get('message')}")
            
            # Also store in session for backward compatibility
            orchestrator.session_manager.store_data(
                orchestrator.session_id,
                f"purchase_analyzer_{int(time.time())}",
                [{
                    'value': json.dumps(result_with_docs),
                    'service': 'database_savings_plans',
                    'operation': 'purchase_analyzer',
                    'hourly_commitment': hourly_commitment,
                    'analysis_timestamp': datetime.now().isoformat()
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to store purchase analyzer results: {e}")
    
    return ResponseFormatter.to_text_content(
        ResponseFormatter.success_response(
            data=result_with_docs,
            message=f"Purchase analyzer simulation completed: ${hourly_commitment:.2f}/hour commitment, {coverage:.1f}% coverage, {utilization:.1f}% utilization",
            analysis_type="database_savings_plans_purchase_analyzer",
            execution_time=execution_time,
            metadata={
                'session_id': orchestrator.session_id,
                'hourly_commitment': hourly_commitment,
                'commitment_term': commitment_term,
                'payment_option': payment_option,
                'region': region,
                'lookback_period_days': lookback_period_days,
                'adjusted_usage_projection': adjusted_usage_projection,
                'comparison_scenarios_count': len(comparison_scenarios)
            }
        )
    )


@handle_aws_error
async def analyze_existing_savings_plans(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Analyze existing Database Savings Plans utilization and coverage.
    
    This function provides detailed analysis of current Database Savings Plans commitments,
    including utilization rates, coverage percentages, and optimization recommendations.
    It helps identify over-commitment, under-utilization, and coverage gaps.
    
    MCP Tool Parameters:
        - region (optional): AWS region to analyze (None for all regions)
        - lookback_period_days (default: 30): Analysis period (30, 60, or 90 days)
        - include_recommendations (default: true): Include optimization recommendations
        - store_results (default: true): Store results in session database
        
    Returns:
        List[TextContent]: Formatted existing commitments analysis results
        
    Example Usage:
        >>> # Basic analysis of existing commitments
        >>> await analyze_existing_savings_plans({})
        
        >>> # Detailed analysis for specific region with longer lookback
        >>> await analyze_existing_savings_plans({
        ...     "region": "us-east-1",
        ...     "lookback_period_days": 60,
        ...     "include_recommendations": True
        ... })
    """
    start_time = time.time()
    
    # Extract and validate parameters
    region = arguments.get("region")
    lookback_period_days = arguments.get("lookback_period_days", 30)
    include_recommendations = arguments.get("include_recommendations", True)
    store_results = arguments.get("store_results", True)
    
    logger.info(f"Starting existing Database Savings Plans analysis: region={region}, "
                f"lookback_period={lookback_period_days}")
    
    # Initialize ServiceOrchestrator for session management
    orchestrator = ServiceOrchestrator()
    
    # Analyze existing commitments
    existing_analysis_result = analyze_existing_commitments(
        region=region,
        lookback_period_days=lookback_period_days
    )
    
    if existing_analysis_result.get('status') not in ['success', 'info']:
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                Exception(f"Failed to analyze existing commitments: {existing_analysis_result.get('message')}"),
                "analyze_existing_savings_plans"
            )
        )
    
    existing_data = existing_analysis_result.get('data', {})
    existing_plans = existing_data.get('existing_plans', [])
    gaps_analysis = existing_data.get('gaps', {})
    summary = existing_data.get('summary', {})
    coverage_by_service = existing_data.get('coverage_by_service', {})
    recommendations = existing_data.get('recommendations', [])
    
    # Generate additional optimization recommendations if requested
    optimization_recommendations = []
    if include_recommendations and existing_plans:
        
        # Analyze utilization patterns
        utilization_values = [plan.get('utilization_percentage', 0) for plan in existing_plans]
        coverage_values = [plan.get('coverage_percentage', 0) for plan in existing_plans]
        
        avg_utilization = sum(utilization_values) / len(utilization_values) if utilization_values else 0
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0
        
        # Generate specific recommendations based on patterns
        if avg_utilization < 70:
            optimization_recommendations.append({
                'type': 'utilization_optimization',
                'priority': 'high',
                'title': 'Low Utilization Detected',
                'description': f'Average utilization is {avg_utilization:.1f}%, indicating over-commitment',
                'action': 'Consider reducing future commitment amounts or review usage patterns',
                'potential_impact': 'Reduce unused commitment costs'
            })
        
        if avg_coverage > 95:
            optimization_recommendations.append({
                'type': 'coverage_optimization',
                'priority': 'medium',
                'title': 'High Coverage Risk',
                'description': f'Average coverage is {avg_coverage:.1f}%, which may be risky for dynamic workloads',
                'action': 'Consider targeting 85-90% coverage for better flexibility',
                'potential_impact': 'Reduce over-commitment risk'
            })
        
        # Check for plans with very low utilization
        low_utilization_plans = [plan for plan in existing_plans if plan.get('utilization_percentage', 0) < 60]
        if low_utilization_plans:
            optimization_recommendations.append({
                'type': 'individual_plan_optimization',
                'priority': 'high',
                'title': f'{len(low_utilization_plans)} Plans with Low Utilization',
                'description': 'Some individual plans have utilization below 60%',
                'action': 'Review specific plans and consider adjustments',
                'affected_plans': [plan.get('savings_plan_id') for plan in low_utilization_plans],
                'potential_impact': 'Optimize individual commitment efficiency'
            })
        
        # Check for significant unused commitment
        total_unused = summary.get('total_unused_commitment', 0)
        total_commitment = summary.get('total_hourly_commitment', 0)
        
        if total_unused > 0 and total_commitment > 0:
            unused_percentage = (total_unused / total_commitment) * 100
            if unused_percentage > 15:
                optimization_recommendations.append({
                    'type': 'unused_commitment_optimization',
                    'priority': 'high',
                    'title': f'{unused_percentage:.1f}% Unused Commitment',
                    'description': f'${total_unused:.2f}/hour in unused commitment detected',
                    'action': 'Review usage patterns and consider commitment adjustments',
                    'potential_impact': f'Potential savings of ${total_unused * 8760:,.0f}/year'
                })
        
        # Check for coverage gaps
        uncovered_spend = gaps_analysis.get('uncovered_spend', 0)
        if uncovered_spend > 0:
            optimization_recommendations.append({
                'type': 'coverage_gap_optimization',
                'priority': 'medium',
                'title': 'Coverage Gaps Identified',
                'description': f'${uncovered_spend:,.2f} in uncovered database spend',
                'action': gaps_analysis.get('recommendation', 'Consider additional commitments'),
                'potential_impact': 'Additional savings opportunity'
            })
    
    # Compile comprehensive results
    analysis_results = {
        'existing_commitments_analysis': existing_analysis_result,
        'optimization_recommendations': optimization_recommendations,
        'analysis_mode': 'existing_commitments'
    }
    
    # Generate insights with optimization context
    insights = []
    
    # Optimization hierarchy context for existing commitments
    insights.append("🎯 OPTIMIZATION OPPORTUNITIES FOR EXISTING COMMITMENTS:")
    insights.append("   1️⃣ Right-size instances to improve utilization")
    insights.append("   2️⃣ Modernize to latest generation for better efficiency")
    insights.append("   3️⃣ Consider Aurora Serverless v2 for variable workloads")
    insights.append("   4️⃣ Optimize storage to reduce uncovered costs")
    insights.append("")
    
    # Summary insights
    total_plans = summary.get('total_plans', 0)
    total_commitment = summary.get('total_hourly_commitment', 0)
    avg_utilization = summary.get('average_utilization', 0)
    avg_coverage = summary.get('average_coverage', 0)
    
    if total_plans > 0:
        insights.append("📋 EXISTING COMMITMENTS ANALYSIS:")
        insights.append(f"   • Active Database Savings Plans: {total_plans}")
        insights.append(f"   • Total hourly commitment: ${total_commitment:.2f}/hour (${total_commitment * 8760:,.0f}/year)")
        insights.append(f"   • Average utilization: {avg_utilization:.1f}%")
        insights.append(f"   • Average coverage: {avg_coverage:.1f}%")
        
        # Utilization assessment with optimization recommendations
        if avg_utilization >= 90:
            insights.append("✅ Excellent utilization - commitments well-matched to usage")
        elif avg_utilization >= 80:
            insights.append("✅ Good utilization - consider right-sizing for further optimization")
        elif avg_utilization >= 70:
            insights.append("⚠️  Moderate utilization - RIGHT-SIZE instances to improve efficiency")
        else:
            insights.append("🚨 Low utilization - URGENT: Right-size instances before additional commitments")
        
        # Coverage assessment
        if 85 <= avg_coverage <= 90:
            insights.append("✅ Optimal coverage range for dynamic workloads")
        elif avg_coverage > 90:
            insights.append("⚠️  High coverage - may be risky for usage variability")
        elif avg_coverage >= 80:
            insights.append("💡 Good coverage - consider increasing for better savings")
        else:
            insights.append("💡 Low coverage - opportunity for additional commitments")
        
    else:
        insights.append("📋 No active Database Savings Plans found")
        insights.append("💡 Consider running usage analysis to identify opportunities")
    
    # Unused commitment insights
    total_unused = summary.get('total_unused_commitment', 0)
    if total_unused > 0:
        annual_unused_cost = total_unused * 8760
        insights.append(f"⚠️  Unused commitment: ${total_unused:.2f}/hour (${annual_unused_cost:,.0f}/year)")
    
    # Coverage gaps insights
    uncovered_spend = gaps_analysis.get('uncovered_spend', 0)
    if uncovered_spend > 0:
        insights.append(f"💡 Uncovered spend: ${uncovered_spend:,.2f} opportunity for additional savings")
    
    # Service-specific insights
    if coverage_by_service:
        service_coverages = [(service, data.get('coverage_percentage', 0)) 
                           for service, data in coverage_by_service.items()]
        service_coverages.sort(key=lambda x: x[1], reverse=True)
        
        if service_coverages:
            best_service, best_coverage = service_coverages[0]
            worst_service, worst_coverage = service_coverages[-1]
            
            if len(service_coverages) > 1:
                insights.append(f"🏆 Best coverage: {best_service} ({best_coverage:.1f}%)")
                insights.append(f"📊 Lowest coverage: {worst_service} ({worst_coverage:.1f}%)")
    
    # Optimization recommendations insights
    if optimization_recommendations:
        high_priority_recs = [rec for rec in optimization_recommendations if rec.get('priority') == 'high']
        if high_priority_recs:
            insights.append(f"🚨 {len(high_priority_recs)} high-priority optimization opportunities identified")
    
    # Create final response
    execution_time = time.time() - start_time
    
    # Add documentation links
    result_with_docs = add_documentation_links(
        analysis_results,
        service_type="rds",
        finding_type="database_savings_plans_existing"
    )
    
    # Add analysis insights and metadata
    result_with_docs['analysis_insights'] = insights
    result_with_docs['existing_commitments_summary'] = {
        'total_plans': total_plans,
        'total_hourly_commitment': total_commitment,
        'average_utilization': avg_utilization,
        'average_coverage': avg_coverage,
        'total_unused_commitment': total_unused,
        'uncovered_spend': uncovered_spend,
        'optimization_opportunities': len(optimization_recommendations),
        'analysis_period': f"{summary.get('lookback_period_days', lookback_period_days)} days"
    }
    result_with_docs['limitations'] = {
        'analysis_scope': [
            "Based on Cost Explorer aggregated data",
            "Individual plan metrics may be estimated",
            "Utilization calculated from available commitment data"
        ],
        'recommendations': [
            "Consider usage pattern volatility when adjusting commitments",
            "Database Savings Plans cannot be resold or exchanged",
            "Monitor trends over multiple periods for better insights"
        ]
    }
    
    # Store results if requested
    if store_results:
        try:
            # Store existing commitments analysis in historical database
            storage_result = store_analysis_result(
                session_id=orchestrator.session_id,
                analysis_type="existing_commitments",
                analysis_data=existing_analysis_result.get('data', {}),
                region=region,
                lookback_period_days=lookback_period_days,
                metadata={
                    'total_plans': total_plans,
                    'total_hourly_commitment': total_commitment,
                    'average_utilization': avg_utilization,
                    'average_coverage': avg_coverage,
                    'execution_time': execution_time
                }
            )
            if storage_result.get('status') == 'success':
                logger.info(f"Stored existing commitments analysis: {storage_result['data']['analysis_id']}")
            else:
                logger.warning(f"Failed to store existing commitments analysis: {storage_result.get('message')}")
            
            # Also store in session for backward compatibility
            orchestrator.session_manager.store_data(
                orchestrator.session_id,
                f"existing_commitments_analysis_{int(time.time())}",
                [{
                    'value': json.dumps(result_with_docs),
                    'service': 'database_savings_plans',
                    'operation': 'existing_commitments_analysis',
                    'total_plans': total_plans,
                    'average_utilization': avg_utilization,
                    'analysis_timestamp': datetime.now().isoformat()
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to store existing commitments analysis: {e}")
    
    return ResponseFormatter.to_text_content(
        ResponseFormatter.success_response(
            data=result_with_docs,
            message=f"Existing Database Savings Plans analysis completed: {total_plans} plans analyzed, {avg_utilization:.1f}% avg utilization, {len(optimization_recommendations)} optimization opportunities",
            analysis_type="database_savings_plans_existing_commitments",
            execution_time=execution_time,
            metadata={
                'session_id': orchestrator.session_id,
                'region': region,
                'lookback_period_days': lookback_period_days,
                'total_plans_analyzed': total_plans,
                'optimization_recommendations_count': len(optimization_recommendations)
            }
        )
    )


@handle_aws_error
async def run_multi_account_savings_plans_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Multi-account Database Savings Plans analysis with organization-level recommendations.
    
    This function provides comprehensive Database Savings Plans analysis across multiple
    AWS accounts within an organization. It aggregates usage data, generates both
    account-level and organization-level recommendations, and calculates shared
    savings plans benefits.
    
    MCP Tool Parameters:
        - account_ids (required): List of AWS account IDs to analyze
        - organization_id (optional): AWS Organizations ID
        - region (optional): AWS region to analyze (None for all regions)
        - lookback_period_days (default: 30): Analysis period (30, 60, or 90 days)
        - services (optional): List of database services to analyze (None for all)
        - include_shared_analysis (default: true): Calculate shared savings benefits
        - store_results (default: true): Store results in session database
        
    Returns:
        List[TextContent]: Formatted multi-account analysis results with recommendations
        
    Example Usage:
        >>> # Basic multi-account analysis
        >>> await run_multi_account_savings_plans_analysis({
        ...     "account_ids": ["123456789012", "123456789013", "123456789014"]
        ... })
        
        >>> # Advanced multi-account analysis with organization context
        >>> await run_multi_account_savings_plans_analysis({
        ...     "account_ids": ["123456789012", "123456789013"],
        ...     "organization_id": "o-1234567890",
        ...     "region": "us-east-1",
        ...     "lookback_period_days": 60,
        ...     "services": ["rds", "aurora", "dynamodb"]
        ... })
    """
    start_time = time.time()
    
    # Extract and validate parameters
    account_ids = arguments.get("account_ids")
    organization_id = arguments.get("organization_id")
    region = arguments.get("region")
    lookback_period_days = arguments.get("lookback_period_days", 30)
    services = arguments.get("services")
    include_shared_analysis = arguments.get("include_shared_analysis", True)
    store_results = arguments.get("store_results", True)
    
    # Validate required parameters
    if not account_ids or not isinstance(account_ids, list):
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                ValueError("account_ids parameter is required and must be a list"),
                "run_multi_account_savings_plans_analysis"
            )
        )
    
    if len(account_ids) < 2:
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                ValueError("At least 2 account IDs are required for multi-account analysis"),
                "run_multi_account_savings_plans_analysis"
            )
        )
    
    logger.info(f"Starting multi-account Database Savings Plans analysis: "
                f"{len(account_ids)} accounts, region={region}, "
                f"lookback_period={lookback_period_days}")
    
    # Initialize ServiceOrchestrator for parallel execution and session management
    orchestrator = ServiceOrchestrator()
    
    # Step 1: Aggregate multi-account usage data
    logger.info("Aggregating usage data across accounts")
    usage_aggregation_result = aggregate_multi_account_usage(
        account_ids=account_ids,
        region=region,
        lookback_period_days=lookback_period_days,
        services=services,
        organization_id=organization_id
    )
    
    if usage_aggregation_result.get('status') != 'success':
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                Exception(f"Failed to aggregate multi-account usage: {usage_aggregation_result.get('message')}"),
                "run_multi_account_savings_plans_analysis"
            )
        )
    
    usage_data = usage_aggregation_result.get('data', {})
    successful_accounts = usage_data.get('successful_accounts', [])
    failed_accounts = usage_data.get('failed_accounts', [])
    
    # Step 2: Generate multi-account recommendations
    logger.info("Generating multi-account recommendations")
    recommendations_result = generate_multi_account_recommendations(
        multi_account_usage=usage_data,
        commitment_terms=["1_YEAR"],
        payment_options=["NO_UPFRONT"]
    )
    
    if recommendations_result.get('status') != 'success':
        return ResponseFormatter.to_text_content(
            ResponseFormatter.error_response(
                Exception(f"Failed to generate multi-account recommendations: {recommendations_result.get('message')}"),
                "run_multi_account_savings_plans_analysis"
            )
        )
    
    recommendations_data = recommendations_result.get('data', {})
    
    # Step 3: Calculate shared savings benefits if requested
    shared_benefits_result = None
    if include_shared_analysis:
        logger.info("Calculating shared savings benefits")
        shared_benefits_result = calculate_shared_savings_benefits(
            multi_account_usage=usage_data,
            organization_recommendations=recommendations_data.get('organization_level', []),
            account_recommendations=recommendations_data.get('account_level', {})
        )
    
    # Compile comprehensive results
    analysis_results = {
        'multi_account_usage': usage_aggregation_result,
        'multi_account_recommendations': recommendations_result,
        'shared_benefits_analysis': shared_benefits_result,
        'analysis_mode': 'multi_account'
    }
    
    # Generate multi-account insights with optimization hierarchy
    insights = []
    
    # Multi-account summary
    total_accounts = len(successful_accounts)
    consolidated_spend = usage_data.get('consolidated_usage', {}).get('total_on_demand_spend', 0)
    consolidated_hourly = usage_data.get('consolidated_usage', {}).get('average_hourly_spend', 0)
    
    insights.append("🏢 MULTI-ACCOUNT DATABASE SAVINGS PLANS ANALYSIS:")
    insights.append(f"   • Accounts analyzed: {total_accounts}")
    insights.append(f"   • Total consolidated spend: ${consolidated_spend:,.2f} over {lookback_period_days} days")
    insights.append(f"   • Average hourly spend: ${consolidated_hourly:.2f}/hour")
    
    if failed_accounts:
        insights.append(f"   ⚠️  Failed accounts: {len(failed_accounts)} (check permissions)")
    
    # Optimization hierarchy for multi-account
    insights.append("")
    insights.append("🎯 MULTI-ACCOUNT OPTIMIZATION HIERARCHY:")
    insights.append("   ⚠️  COMPLETE THESE FIRST across all accounts:")
    insights.append("   1️⃣ Right-size oversized instances (immediate 20-50% savings)")
    insights.append("   2️⃣ Standardize on latest generation instances across accounts")
    insights.append("   3️⃣ Consolidate similar workloads where possible")
    insights.append("   4️⃣ Implement Aurora Serverless v2 for variable workloads")
    insights.append("   5️⃣ Optimize storage allocation across all accounts")
    insights.append("   6️⃣ Database Savings Plans (after operational optimizations)")
    insights.append("")
    
    # Recommendations insights
    org_recommendations = recommendations_data.get('organization_level', [])
    account_recommendations = recommendations_data.get('account_level', {})
    consolidated_savings = recommendations_data.get('consolidated_savings', 0)
    total_individual_savings = recommendations_data.get('total_individual_savings', 0)
    optimization_strategy = recommendations_data.get('optimization_strategy', 'unknown')
    
    insights.append("📋 COMMITMENT STRATEGY RECOMMENDATIONS:")
    
    if org_recommendations:
        best_org_rec = org_recommendations[0]
        insights.append(f"🏢 Organization-level: ${best_org_rec['hourly_commitment']:.2f}/hour commitment")
        insights.append(f"   • Estimated annual savings: ${consolidated_savings:,.2f}")
        insights.append(f"   • Projected coverage: {best_org_rec['projected_coverage']:.1f}%")
    
    if account_recommendations:
        accounts_with_recs = len([recs for recs in account_recommendations.values() if recs])
        insights.append(f"🏦 Account-level: {accounts_with_recs} accounts with individual recommendations")
        insights.append(f"   • Combined individual savings: ${total_individual_savings:,.2f}")
    
    # Strategy recommendation
    shared_benefit = consolidated_savings - total_individual_savings
    if shared_benefit > 0:
        insights.append(f"💡 Shared benefit: ${shared_benefit:,.2f} additional savings from organization-level approach")
        insights.append(f"✅ Recommended strategy: {optimization_strategy.replace('_', ' ').title()}")
    else:
        insights.append("💡 Individual account commitments may provide better optimization")
        insights.append(f"✅ Recommended strategy: {optimization_strategy.replace('_', ' ').title()}")
    
    # Shared benefits insights
    if shared_benefits_result and shared_benefits_result.get('status') == 'success':
        shared_data = shared_benefits_result.get('data', {})
        commitment_efficiency = shared_data.get('commitment_efficiency_percentage', 0)
        utilization_improvement = shared_data.get('utilization_improvement', 0)
        risk_factors = shared_data.get('risk_factors', [])
        
        insights.append("")
        insights.append("📊 SHARED SAVINGS ANALYSIS:")
        insights.append(f"   • Commitment efficiency: {commitment_efficiency:.1f}% reduction vs individual")
        insights.append(f"   • Utilization improvement: {utilization_improvement:+.1f}%")
        
        if risk_factors:
            high_risk = [r for r in risk_factors if r.get('level') == 'high']
            if high_risk:
                insights.append(f"   ⚠️  High-risk factors: {len(high_risk)} identified")
                for risk in high_risk[:2]:  # Show top 2 risks
                    insights.append(f"      • {risk.get('description', 'Unknown risk')}")
    
    # Cross-account opportunities
    cross_account_opportunities = usage_data.get('cross_account_optimization_opportunities', [])
    if cross_account_opportunities:
        insights.append("")
        insights.append("🔄 CROSS-ACCOUNT OPTIMIZATION OPPORTUNITIES:")
        for opportunity in cross_account_opportunities[:3]:  # Show top 3
            insights.append(f"   • {opportunity.get('type', 'Unknown').replace('_', ' ').title()}")
            insights.append(f"     {opportunity.get('opportunity', 'No description')}")
    
    # Account-specific insights
    insights.append("")
    insights.append("🏦 ACCOUNT-SPECIFIC INSIGHTS:")
    
    account_level_usage = usage_data.get('account_level_usage', {})
    if account_level_usage:
        # Find highest and lowest spend accounts
        account_spends = [
            (account_id, usage.get('total_on_demand_spend', 0))
            for account_id, usage in account_level_usage.items()
        ]
        account_spends.sort(key=lambda x: x[1], reverse=True)
        
        if account_spends:
            highest_account, highest_spend = account_spends[0]
            lowest_account, lowest_spend = account_spends[-1]
            
            insights.append(f"   • Highest spend: Account {highest_account} (${highest_spend:,.2f})")
            insights.append(f"   • Lowest spend: Account {lowest_account} (${lowest_spend:,.2f})")
            
            # Show recommendations for top account
            if highest_account in account_recommendations and account_recommendations[highest_account]:
                top_rec = account_recommendations[highest_account][0]
                insights.append(f"   • Top account recommendation: ${top_rec['hourly_commitment']:.2f}/hour")
    
    # Add warnings and prerequisites
    insights.append("")
    insights.append("⚠️  MULTI-ACCOUNT PREREQUISITES & WARNINGS:")
    insights.append("   • Ensure cross-account Cost Explorer permissions are configured")
    insights.append("   • Complete operational optimizations BEFORE purchasing commitments")
    insights.append("   • Database Savings Plans cannot be resold or exchanged")
    insights.append("   • Monitor usage patterns across all accounts before committing")
    insights.append("   • Consider account lifecycle changes (closures, mergers)")
    
    # Create final response
    execution_time = time.time() - start_time
    
    # Add documentation links
    result_with_docs = add_documentation_links(
        analysis_results,
        service_type="rds",
        finding_type="database_savings_plans_multi_account"
    )
    
    # Add analysis insights and metadata
    result_with_docs['analysis_insights'] = insights
    result_with_docs['multi_account_summary'] = {
        'total_accounts_requested': len(account_ids),
        'successful_accounts': len(successful_accounts),
        'failed_accounts': len(failed_accounts),
        'organization_id': organization_id,
        'consolidated_spend': consolidated_spend,
        'consolidated_hourly_spend': consolidated_hourly,
        'optimization_strategy': optimization_strategy,
        'shared_benefit': shared_benefit,
        'cross_account_opportunities': len(cross_account_opportunities)
    }
    result_with_docs['optimization_hierarchy'] = {
        'priority_order': [
            "1. Right-sizing across all accounts (immediate savings, no commitment)",
            "2. Standardization on latest generation instances",
            "3. Workload consolidation opportunities",
            "4. Serverless adoption for variable workloads",
            "5. Storage optimization across accounts",
            "6. Database Savings Plans (after operational optimizations)"
        ],
        'multi_account_considerations': [
            "Coordinate optimizations across accounts for maximum benefit",
            "Standardize instance families for better shared commitment utilization",
            "Consider workload consolidation to reduce total commitment needs"
        ]
    }
    result_with_docs['limitations'] = {
        'multi_account_analysis': [
            "Requires cross-account Cost Explorer permissions",
            "Account-level data may be aggregated in some cases",
            "Shared savings calculations are estimates based on usage patterns"
        ],
        'database_savings_plans': [
            "1-year terms with No Upfront payment only",
            "Cannot be resold or exchanged - commitment risk",
            "Latest-generation instances only (M7, R7, R8, etc.)",
            "Covers compute usage only"
        ],
        'prerequisites': [
            "Complete right-sizing and modernization across all accounts FIRST",
            "Ensure stable usage patterns before shared commitments",
            "Monitor account lifecycle changes that may affect commitments"
        ]
    }
    
    # Store results if requested
    if store_results:
        try:
            # Store multi-account analysis results in session
            orchestrator.session_manager.store_data(
                orchestrator.session_id,
                f"multi_account_analysis_{int(time.time())}",
                [{
                    'value': json.dumps(result_with_docs),
                    'service': 'database_savings_plans',
                    'operation': 'multi_account_analysis',
                    'total_accounts': total_accounts,
                    'consolidated_savings': consolidated_savings,
                    'optimization_strategy': optimization_strategy,
                    'analysis_timestamp': datetime.now().isoformat()
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to store multi-account analysis results: {e}")
    
    return ResponseFormatter.to_text_content(
        ResponseFormatter.success_response(
            data=result_with_docs,
            message=f"Multi-account Database Savings Plans analysis completed: {total_accounts} accounts analyzed, ${consolidated_savings:,.2f} organization-level savings, strategy: {optimization_strategy}",
            analysis_type="database_savings_plans_multi_account",
            execution_time=execution_time,
            metadata={
                'session_id': orchestrator.session_id,
                'total_accounts_analyzed': total_accounts,
                'failed_accounts_count': len(failed_accounts),
                'organization_id': organization_id,
                'region': region,
                'lookback_period_days': lookback_period_days,
                'optimization_strategy': optimization_strategy,
                'shared_benefit': shared_benefit
            }
        )
    )


# ============================================================================
# Service-Specific Recommendation Functions
# ============================================================================

def generate_rds_specific_recommendations(
    usage_data: Dict[str, Any],
    service_usage: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate RDS-specific Database Savings Plans recommendations.
    
    Considers RDS-specific factors:
    - Instance families (M7, R7, R8 for Database Savings Plans vs M5, R5, R6g for Reserved Instances)
    - Deployment options (Single-AZ, Multi-AZ, Read Replicas)
    - Engine types (MySQL, PostgreSQL, MariaDB, Oracle, SQL Server)
    - Storage types (gp3, io1, io2) - not covered by Database Savings Plans
    
    Args:
        usage_data: Overall database usage analysis results
        service_usage: RDS-specific usage data
        
    Returns:
        Dictionary containing RDS-specific recommendations
    """
    logger.info("Generating RDS-specific Database Savings Plans recommendations")
    
    # Extract RDS usage metrics
    rds_total_spend = service_usage.get("total_spend", 0.0)
    rds_hourly_spend = service_usage.get("average_hourly_spend", 0.0)
    rds_instance_types = service_usage.get("instance_types", {})
    
    if rds_hourly_spend <= 0:
        return {
            "status": "info",
            "data": {
                "recommendations": [],
                "instance_family_analysis": {},
                "deployment_considerations": []
            },
            "message": "No RDS usage found for Database Savings Plans analysis"
        }
    
    # Analyze instance families for Database Savings Plans eligibility
    latest_generation_families = {
        'db.m7g', 'db.m7i', 'db.r7g', 'db.r7i', 'db.r8g', 'db.t4g',
        'db.x2gd', 'db.x2iedn', 'db.x2iezn'
    }
    
    older_generation_families = {
        'db.m5', 'db.m5d', 'db.m5n', 'db.m5zn', 'db.m6g', 'db.m6gd', 'db.m6i', 'db.m6id', 'db.m6idn', 'db.m6in',
        'db.r5', 'db.r5b', 'db.r5d', 'db.r5dn', 'db.r5n', 'db.r6g', 'db.r6gd', 'db.r6i', 'db.r6id', 'db.r6idn', 'db.r6in',
        'db.t3'
    }
    
    # Categorize instance usage by generation
    eligible_spend = 0.0
    ineligible_spend = 0.0
    instance_family_analysis = {}
    
    for instance_type, spend in rds_instance_types.items():
        # Extract instance family (e.g., 'db.r7g' from 'db.r7g.xlarge')
        family = '.'.join(instance_type.split('.')[:2]) if '.' in instance_type else instance_type
        
        is_eligible = any(family.startswith(eligible_prefix) for eligible_prefix in latest_generation_families)
        is_ineligible = any(family.startswith(ineligible_prefix) for ineligible_prefix in older_generation_families)
        
        if is_eligible and not is_ineligible:
            eligible_spend += spend
            instance_family_analysis[family] = {
                "spend": spend,
                "eligibility": "database_savings_plans",
                "recommendation": "Eligible for Database Savings Plans",
                "instance_types": [instance_type]
            }
        elif is_ineligible:
            ineligible_spend += spend
            instance_family_analysis[family] = {
                "spend": spend,
                "eligibility": "reserved_instances_only",
                "recommendation": "Requires Reserved Instances - consider upgrading to latest generation",
                "instance_types": [instance_type]
            }
        else:
            # Unknown family - treat as ineligible for safety
            ineligible_spend += spend
            instance_family_analysis[family] = {
                "spend": spend,
                "eligibility": "unknown",
                "recommendation": "Verify Database Savings Plans eligibility",
                "instance_types": [instance_type]
            }
    
    # Calculate eligible hourly spend
    total_hours = usage_data.get("lookback_period_days", 30) * 24
    eligible_hourly_spend = eligible_spend / total_hours if total_hours > 0 else 0.0
    ineligible_hourly_spend = ineligible_spend / total_hours if total_hours > 0 else 0.0
    
    # Ensure consistent total for percentage calculations
    # Use the sum of categorized spend rather than the input rds_hourly_spend
    # to avoid mismatches when instance types don't perfectly align
    total_categorized_hourly_spend = eligible_hourly_spend + ineligible_hourly_spend
    
    # Generate RDS-specific recommendations
    recommendations = []
    deployment_considerations = []
    
    if eligible_hourly_spend > 0:
        # Generate Database Savings Plans recommendation for eligible instances
        # Target 85% coverage for RDS workloads (conservative for database stability)
        coverage_target = 0.85
        recommended_commitment = eligible_hourly_spend * coverage_target
        
        # Calculate savings (typical 20% discount for Database Savings Plans)
        discount_rate = 0.20
        annual_savings = recommended_commitment * discount_rate * 8760
        
        recommendations.append({
            "commitment_type": "database_savings_plans",
            "hourly_commitment": round(recommended_commitment, 2),
            "eligible_hourly_spend": round(eligible_hourly_spend, 2),
            "projected_coverage": round(coverage_target * 100, 1),
            "estimated_annual_savings": round(annual_savings, 2),
            "discount_rate": discount_rate * 100,
            "rationale": f"Covers {coverage_target*100:.0f}% of latest-generation RDS instances with 20% discount",
            "instance_families": list(set(family for family, data in instance_family_analysis.items() 
                                        if data["eligibility"] == "database_savings_plans"))
        })
        
        # RDS-specific deployment considerations
        deployment_considerations.extend([
            "Multi-AZ deployments: Database Savings Plans apply to both primary and standby instances",
            "Read Replicas: Each replica instance is covered separately by Database Savings Plans",
            "Cross-region replicas: Require separate Database Savings Plans in each region",
            "Storage costs: Database Savings Plans cover compute only - storage (EBS) billed separately"
        ])
    
    if ineligible_hourly_spend > 0:
        # Generate Reserved Instance recommendation for older generation instances
        # Use higher coverage target for RIs since they're instance-specific
        ri_coverage_target = 0.90
        ri_recommended_commitment = ineligible_hourly_spend * ri_coverage_target
        
        # Calculate RI savings (typical 30% discount for 1-year Standard RI)
        ri_discount_rate = 0.30
        ri_annual_savings = ri_recommended_commitment * ri_discount_rate * 8760
        
        recommendations.append({
            "commitment_type": "reserved_instances",
            "hourly_commitment": round(ri_recommended_commitment, 2),
            "eligible_hourly_spend": round(ineligible_hourly_spend, 2),
            "projected_coverage": round(ri_coverage_target * 100, 1),
            "estimated_annual_savings": round(ri_annual_savings, 2),
            "discount_rate": ri_discount_rate * 100,
            "rationale": f"Older generation instances require Reserved Instances - consider upgrading to latest generation",
            "instance_families": list(set(family for family, data in instance_family_analysis.items() 
                                        if data["eligibility"] == "reserved_instances_only")),
            "upgrade_recommendation": "Consider upgrading to M7, R7, or R8 instances for Database Savings Plans eligibility"
        })
        
        # RI-specific deployment considerations
        deployment_considerations.extend([
            "Reserved Instances: Instance type and region specific - less flexible than Database Savings Plans",
            "Convertible RIs: Allow instance family changes but with lower discount rates",
            "Standard RIs: Higher discounts but no modification flexibility"
        ])
    
    # Engine-specific considerations
    engine_considerations = [
        "MySQL/PostgreSQL: Excellent Database Savings Plans compatibility with latest-generation instances",
        "Oracle/SQL Server: License costs separate - Database Savings Plans apply to compute only",
        "MariaDB: Full Database Savings Plans support for latest-generation instances",
        "Engine upgrades: Consider upgrading to latest engine versions for better performance"
    ]
    
    # Performance and sizing considerations
    performance_considerations = [
        "Right-sizing: Complete instance right-sizing BEFORE purchasing Database Savings Plans",
        "Performance Insights: Use RDS Performance Insights to validate instance sizing",
        "CPU utilization: Target 40-70% average CPU for optimal cost-performance balance",
        "Memory utilization: R-series instances for memory-intensive workloads, M-series for balanced",
        "Burstable instances: T4g instances suitable for variable workloads with Database Savings Plans"
    ]
    
    logger.info(f"RDS-specific analysis complete: ${eligible_hourly_spend:.2f}/hour eligible, "
                f"${ineligible_hourly_spend:.2f}/hour requires Reserved Instances, "
                f"${total_categorized_hourly_spend:.2f}/hour total categorized")
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "instance_family_analysis": instance_family_analysis,
            "deployment_considerations": deployment_considerations,
            "engine_considerations": engine_considerations,
            "performance_considerations": performance_considerations,
            "summary": {
                "total_rds_hourly_spend": total_categorized_hourly_spend,
                "eligible_hourly_spend": eligible_hourly_spend,
                "ineligible_hourly_spend": ineligible_hourly_spend,
                "database_savings_plans_eligible_percentage": (eligible_hourly_spend / total_categorized_hourly_spend * 100) if total_categorized_hourly_spend > 0 else 0.0,
                "reserved_instances_required_percentage": (ineligible_hourly_spend / total_categorized_hourly_spend * 100) if total_categorized_hourly_spend > 0 else 0.0,
                "total_potential_annual_savings": sum(rec.get("estimated_annual_savings", 0) for rec in recommendations)
            }
        },
        "message": f"RDS-specific recommendations generated: {len(recommendations)} commitment options identified"
    }


def generate_aurora_specific_recommendations(
    usage_data: Dict[str, Any],
    service_usage: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate Aurora-specific Database Savings Plans recommendations.
    
    Considers Aurora-specific factors:
    - Serverless v2 vs Provisioned configurations
    - Aurora MySQL vs Aurora PostgreSQL engines
    - Global Database configurations
    - Reader instance scaling patterns
    - Storage auto-scaling (not covered by Database Savings Plans)
    
    Args:
        usage_data: Overall database usage analysis results
        service_usage: Aurora-specific usage data
        
    Returns:
        Dictionary containing Aurora-specific recommendations
    """
    logger.info("Generating Aurora-specific Database Savings Plans recommendations")
    
    # Extract Aurora usage metrics
    aurora_total_spend = service_usage.get("total_spend", 0.0)
    aurora_hourly_spend = service_usage.get("average_hourly_spend", 0.0)
    aurora_instance_types = service_usage.get("instance_types", {})
    
    if aurora_hourly_spend <= 0:
        return {
            "status": "info",
            "data": {
                "recommendations": [],
                "configuration_analysis": {},
                "serverless_considerations": []
            },
            "message": "No Aurora usage found for Database Savings Plans analysis"
        }
    
    # Analyze Aurora configurations
    provisioned_spend = 0.0
    serverless_spend = 0.0
    configuration_analysis = {}
    
    # Categorize Aurora instance types
    for instance_type, spend in aurora_instance_types.items():
        if 'serverless' in instance_type.lower():
            serverless_spend += spend
            configuration_analysis['serverless'] = {
                "spend": spend,
                "configuration": "Aurora Serverless v2",
                "database_savings_plans_eligibility": "limited",
                "recommendation": "Evaluate provisioned instances for predictable workloads"
            }
        else:
            provisioned_spend += spend
            # Extract instance family for provisioned instances
            family = '.'.join(instance_type.split('.')[:2]) if '.' in instance_type else instance_type
            
            # Check Database Savings Plans eligibility for Aurora provisioned instances
            latest_generation_families = {'db.r7g', 'db.r7i', 'db.r8g', 'db.x2gd', 'db.x2iedn', 'db.x2iezn'}
            is_eligible = any(family.startswith(prefix) for prefix in latest_generation_families)
            
            if family not in configuration_analysis:
                configuration_analysis[family] = {
                    "spend": 0.0,
                    "configuration": "Aurora Provisioned",
                    "database_savings_plans_eligibility": "eligible" if is_eligible else "reserved_instances_only",
                    "recommendation": "Eligible for Database Savings Plans" if is_eligible else "Consider upgrading to latest generation or use Reserved Instances"
                }
            configuration_analysis[family]["spend"] += spend
    
    # Calculate hourly spend by configuration
    total_hours = usage_data.get("lookback_period_days", 30) * 24
    provisioned_hourly_spend = provisioned_spend / total_hours if total_hours > 0 else 0.0
    serverless_hourly_spend = serverless_spend / total_hours if total_hours > 0 else 0.0
    
    # Generate Aurora-specific recommendations
    recommendations = []
    serverless_considerations = []
    
    # Provisioned Aurora recommendations
    if provisioned_hourly_spend > 0:
        # Filter for Database Savings Plans eligible provisioned instances
        eligible_provisioned_spend = sum(
            data["spend"] for data in configuration_analysis.values()
            if data["configuration"] == "Aurora Provisioned" and data["database_savings_plans_eligibility"] == "eligible"
        )
        
        eligible_provisioned_hourly = eligible_provisioned_spend / total_hours if total_hours > 0 else 0.0
        
        if eligible_provisioned_hourly > 0:
            # Aurora provisioned instances - use 85% coverage target
            coverage_target = 0.85
            recommended_commitment = eligible_provisioned_hourly * coverage_target
            
            # Calculate savings (20% discount for Database Savings Plans)
            discount_rate = 0.20
            annual_savings = recommended_commitment * discount_rate * 8760
            
            recommendations.append({
                "commitment_type": "database_savings_plans",
                "configuration": "aurora_provisioned",
                "hourly_commitment": round(recommended_commitment, 2),
                "eligible_hourly_spend": round(eligible_provisioned_hourly, 2),
                "projected_coverage": round(coverage_target * 100, 1),
                "estimated_annual_savings": round(annual_savings, 2),
                "discount_rate": discount_rate * 100,
                "rationale": "Aurora provisioned instances with latest-generation families eligible for Database Savings Plans",
                "considerations": [
                    "Applies to both writer and reader instances",
                    "Global Database: Each region requires separate Database Savings Plans",
                    "Auto Scaling readers: Plan for maximum expected reader count"
                ]
            })
    
    # Serverless considerations and recommendations
    if serverless_hourly_spend > 0:
        serverless_considerations.extend([
            f"Aurora Serverless v2 usage: ${serverless_hourly_spend:.2f}/hour",
            "Database Savings Plans: Limited applicability to serverless configurations",
            "Cost optimization: Focus on ACU (Aurora Capacity Unit) right-sizing",
            "Scaling patterns: Analyze min/max ACU settings for cost efficiency",
            "Workload evaluation: Consider provisioned instances for predictable workloads"
        ])
        
        # Evaluate if provisioned instances might be more cost-effective
        if serverless_hourly_spend > 5.0:  # Threshold for provisioned consideration
            estimated_provisioned_cost = serverless_hourly_spend * 0.7  # Rough estimate
            potential_savings_with_provisioned = (serverless_hourly_spend - estimated_provisioned_cost) * 8760
            
            recommendations.append({
                "commitment_type": "configuration_optimization",
                "configuration": "serverless_to_provisioned",
                "current_serverless_cost": round(serverless_hourly_spend * 8760, 2),
                "estimated_provisioned_cost": round(estimated_provisioned_cost * 8760, 2),
                "potential_annual_savings": round(potential_savings_with_provisioned, 2),
                "rationale": "High serverless usage - evaluate provisioned instances + Database Savings Plans",
                "considerations": [
                    "Requires workload analysis to determine appropriate instance sizing",
                    "Provisioned instances eligible for Database Savings Plans",
                    "Consider Aurora Serverless v2 for truly variable workloads"
                ]
            })
    
    # Aurora-specific deployment considerations
    deployment_considerations = [
        "Global Database: Database Savings Plans required in each region separately",
        "Cross-region replicas: Each region needs independent Database Savings Plans coverage",
        "Reader instances: Auto Scaling readers should be planned for maximum expected count",
        "Backtrack: Available for Aurora MySQL - doesn't affect Database Savings Plans eligibility",
        "Parallel Query: Aurora MySQL feature - compute optimization can reduce Database Savings Plans needs"
    ]
    
    # Engine-specific considerations
    engine_considerations = [
        "Aurora MySQL: Full Database Savings Plans compatibility with latest-generation instances",
        "Aurora PostgreSQL: Full Database Savings Plans compatibility with latest-generation instances",
        "Engine versions: Keep engines updated for optimal performance and cost efficiency",
        "Performance Insights: Use for workload analysis and right-sizing validation"
    ]
    
    # Storage and I/O considerations
    storage_considerations = [
        "Aurora storage: Auto-scaling storage not covered by Database Savings Plans",
        "I/O optimization: Aurora I/O-Optimized configuration affects total cost but not Database Savings Plans eligibility",
        "Backup storage: Not covered by Database Savings Plans - optimize retention policies",
        "Snapshot sharing: Cross-account snapshot sharing doesn't affect Database Savings Plans"
    ]
    
    logger.info(f"Aurora-specific analysis complete: ${provisioned_hourly_spend:.2f}/hour provisioned, "
                f"${serverless_hourly_spend:.2f}/hour serverless")
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "configuration_analysis": configuration_analysis,
            "serverless_considerations": serverless_considerations,
            "deployment_considerations": deployment_considerations,
            "engine_considerations": engine_considerations,
            "storage_considerations": storage_considerations,
            "summary": {
                "total_aurora_hourly_spend": aurora_hourly_spend,
                "provisioned_hourly_spend": provisioned_hourly_spend,
                "serverless_hourly_spend": serverless_hourly_spend,
                "provisioned_percentage": (provisioned_hourly_spend / aurora_hourly_spend * 100) if aurora_hourly_spend > 0 else 0.0,
                "serverless_percentage": (serverless_hourly_spend / aurora_hourly_spend * 100) if aurora_hourly_spend > 0 else 0.0,
                "total_potential_annual_savings": sum(rec.get("estimated_annual_savings", 0) for rec in recommendations if "estimated_annual_savings" in rec)
            }
        },
        "message": f"Aurora-specific recommendations generated: {len(recommendations)} optimization opportunities identified"
    }


def generate_dynamodb_specific_recommendations(
    usage_data: Dict[str, Any],
    service_usage: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate DynamoDB-specific Database Savings Plans recommendations.
    
    Considers DynamoDB-specific factors:
    - On-Demand vs Provisioned capacity modes
    - Global Tables configurations
    - DynamoDB Accelerator (DAX) usage
    - Auto Scaling configurations
    - Reserved Capacity vs Database Savings Plans
    
    Args:
        usage_data: Overall database usage analysis results
        service_usage: DynamoDB-specific usage data
        
    Returns:
        Dictionary containing DynamoDB-specific recommendations
    """
    logger.info("Generating DynamoDB-specific Database Savings Plans recommendations")
    
    # Extract DynamoDB usage metrics
    dynamodb_total_spend = service_usage.get("total_spend", 0.0)
    dynamodb_hourly_spend = service_usage.get("average_hourly_spend", 0.0)
    
    if dynamodb_hourly_spend <= 0:
        return {
            "status": "info",
            "data": {
                "recommendations": [],
                "capacity_mode_analysis": {},
                "optimization_opportunities": []
            },
            "message": "No DynamoDB usage found for Database Savings Plans analysis"
        }
    
    # DynamoDB capacity mode analysis
    # Note: Detailed capacity mode breakdown would require additional Cost Explorer dimensions
    # For now, provide general recommendations based on total spend
    
    capacity_mode_analysis = {
        "total_spend": dynamodb_total_spend,
        "hourly_spend": dynamodb_hourly_spend,
        "database_savings_plans_eligibility": "eligible",
        "capacity_modes": {
            "on_demand": {
                "description": "Pay-per-request pricing - automatically eligible for Database Savings Plans",
                "optimization": "Database Savings Plans provide ~20% discount on on-demand usage"
            },
            "provisioned": {
                "description": "Pre-allocated read/write capacity units",
                "optimization": "Compare Database Savings Plans vs Reserved Capacity for provisioned tables"
            }
        }
    }
    
    # Generate DynamoDB-specific recommendations
    recommendations = []
    
    # Database Savings Plans recommendation for DynamoDB
    # DynamoDB is automatically eligible - no instance family restrictions
    coverage_target = 0.90  # Higher coverage for DynamoDB due to predictable usage patterns
    recommended_commitment = dynamodb_hourly_spend * coverage_target
    
    # Calculate savings (20% discount for Database Savings Plans)
    discount_rate = 0.20
    annual_savings = recommended_commitment * discount_rate * 8760
    
    recommendations.append({
        "commitment_type": "database_savings_plans",
        "service": "dynamodb",
        "hourly_commitment": round(recommended_commitment, 2),
        "eligible_hourly_spend": round(dynamodb_hourly_spend, 2),
        "projected_coverage": round(coverage_target * 100, 1),
        "estimated_annual_savings": round(annual_savings, 2),
        "discount_rate": discount_rate * 100,
        "rationale": "DynamoDB usage automatically eligible for Database Savings Plans - covers both on-demand and provisioned capacity",
        "considerations": [
            "Applies to both on-demand and provisioned capacity charges",
            "Global Tables: Each region requires separate Database Savings Plans",
            "DAX clusters: Separate compute charges eligible for Database Savings Plans",
            "Storage and data transfer: Not covered by Database Savings Plans"
        ]
    })
    
    # Optimization opportunities specific to DynamoDB
    optimization_opportunities = []
    
    # Capacity mode optimization
    if dynamodb_hourly_spend > 10.0:  # Significant DynamoDB usage
        optimization_opportunities.extend([
            {
                "category": "capacity_mode_optimization",
                "opportunity": "Evaluate on-demand vs provisioned capacity",
                "description": "Analyze traffic patterns to determine optimal capacity mode",
                "potential_savings": "10-30% depending on usage patterns",
                "implementation": "Use CloudWatch metrics to analyze read/write patterns"
            },
            {
                "category": "auto_scaling_optimization",
                "opportunity": "Optimize Auto Scaling settings",
                "description": "Fine-tune target utilization and scaling policies",
                "potential_savings": "5-15% through better capacity utilization",
                "implementation": "Review Auto Scaling metrics and adjust target utilization"
            }
        ])
    
    # Global Tables considerations
    optimization_opportunities.append({
        "category": "global_tables_optimization",
        "opportunity": "Global Tables cost optimization",
        "description": "Optimize Global Tables configuration and replication patterns",
        "considerations": [
            "Each region requires separate Database Savings Plans coverage",
            "Cross-region replication costs not covered by Database Savings Plans",
            "Consider regional usage patterns for commitment sizing"
        ]
    })
    
    # DAX optimization
    optimization_opportunities.append({
        "category": "dax_optimization",
        "opportunity": "DynamoDB Accelerator (DAX) optimization",
        "description": "DAX cluster compute costs eligible for Database Savings Plans",
        "considerations": [
            "DAX node hours covered by Database Savings Plans",
            "Evaluate DAX necessity - may reduce DynamoDB request volume",
            "Consider DAX node sizing and cluster configuration"
        ]
    })
    
    # Reserved Capacity comparison (for provisioned tables)
    reserved_capacity_comparison = {
        "database_savings_plans": {
            "flexibility": "High - applies across all DynamoDB usage",
            "discount": "~20% on all DynamoDB compute charges",
            "commitment": "1-year term, hourly commitment",
            "coverage": "On-demand and provisioned capacity, DAX clusters"
        },
        "reserved_capacity": {
            "flexibility": "Low - specific to provisioned tables and regions",
            "discount": "~25-50% on provisioned capacity only",
            "commitment": "1-year or 3-year terms, capacity-specific",
            "coverage": "Provisioned read/write capacity units only"
        },
        "recommendation": "Database Savings Plans preferred for mixed workloads; Reserved Capacity for stable provisioned tables"
    }
    
    # DynamoDB-specific best practices
    best_practices = [
        "Partition key design: Optimize for even distribution to avoid hot partitions",
        "Item size optimization: Smaller items reduce read/write capacity consumption",
        "Query vs Scan: Use Query operations instead of Scan for better cost efficiency",
        "Indexes: Optimize Global Secondary Index (GSI) usage and projection",
        "TTL: Use Time To Live for automatic item expiration and cost reduction",
        "Compression: Consider item compression for large items to reduce storage costs"
    ]
    
    logger.info(f"DynamoDB-specific analysis complete: ${dynamodb_hourly_spend:.2f}/hour eligible for Database Savings Plans")
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "capacity_mode_analysis": capacity_mode_analysis,
            "optimization_opportunities": optimization_opportunities,
            "reserved_capacity_comparison": reserved_capacity_comparison,
            "best_practices": best_practices,
            "summary": {
                "total_dynamodb_hourly_spend": dynamodb_hourly_spend,
                "database_savings_plans_eligible": True,
                "recommended_coverage": coverage_target * 100,
                "estimated_annual_savings": annual_savings,
                "optimization_opportunities_count": len(optimization_opportunities)
            }
        },
        "message": f"DynamoDB-specific recommendations generated: Database Savings Plans can provide ${annual_savings:,.2f}/year savings"
    }


def generate_elasticache_specific_recommendations(
    usage_data: Dict[str, Any],
    service_usage: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate ElastiCache-specific Database Savings Plans recommendations.
    
    IMPORTANT: Database Savings Plans for ElastiCache only support Valkey engine.
    Redis and Memcached workloads require Reserved Nodes for optimization.
    
    Considers ElastiCache-specific factors:
    - Engine types (Valkey vs Redis vs Memcached)
    - Node types and instance families
    - Cluster configurations (single-node vs multi-node)
    - Replication groups and sharding
    
    Args:
        usage_data: Overall database usage analysis results
        service_usage: ElastiCache-specific usage data
        
    Returns:
        Dictionary containing ElastiCache-specific recommendations
    """
    logger.info("Generating ElastiCache-specific Database Savings Plans recommendations")
    
    # Extract ElastiCache usage metrics
    elasticache_total_spend = service_usage.get("total_spend", 0.0)
    elasticache_hourly_spend = service_usage.get("average_hourly_spend", 0.0)
    elasticache_instance_types = service_usage.get("instance_types", {})
    
    if elasticache_hourly_spend <= 0:
        return {
            "status": "info",
            "data": {
                "recommendations": [],
                "engine_analysis": {},
                "reserved_nodes_required": []
            },
            "message": "No ElastiCache usage found for Database Savings Plans analysis"
        }
    
    # Analyze ElastiCache engines and instance types
    valkey_spend = 0.0
    redis_spend = 0.0
    memcached_spend = 0.0
    unknown_spend = 0.0
    
    engine_analysis = {}
    
    # Note: Without detailed engine information from Cost Explorer, we'll analyze by instance family
    # and provide guidance on Database Savings Plans eligibility
    
    for instance_type, spend in elasticache_instance_types.items():
        # Extract instance family (e.g., 'cache.r7g' from 'cache.r7g.large')
        family = '.'.join(instance_type.split('.')[:2]) if '.' in instance_type else instance_type
        
        # Check Database Savings Plans eligibility (latest-generation families)
        latest_generation_families = {'cache.m7g', 'cache.r7g', 'cache.t4g'}
        older_generation_families = {
            'cache.m5', 'cache.m5n', 'cache.m6g', 'cache.m6gd', 'cache.m6i', 'cache.m6id', 'cache.m6idn', 'cache.m6in',
            'cache.r5', 'cache.r5n', 'cache.r6g', 'cache.r6gd', 'cache.r6i', 'cache.r6id', 'cache.r6idn', 'cache.r6in',
            'cache.t3'
        }
        
        is_latest_generation = any(family.startswith(prefix) for prefix in latest_generation_families)
        is_older_generation = any(family.startswith(prefix) for prefix in older_generation_families)
        
        if family not in engine_analysis:
            engine_analysis[family] = {
                "spend": 0.0,
                "instance_types": [],
                "database_savings_plans_eligibility": "unknown",
                "recommendation": ""
            }
        
        engine_analysis[family]["spend"] += spend
        engine_analysis[family]["instance_types"].append(instance_type)
        
        if is_latest_generation:
            # Latest generation - potentially eligible for Database Savings Plans (Valkey only)
            engine_analysis[family]["database_savings_plans_eligibility"] = "valkey_only"
            engine_analysis[family]["recommendation"] = "Eligible for Database Savings Plans if using Valkey engine"
            # Assume unknown engine distribution for now
            unknown_spend += spend
        elif is_older_generation:
            # Older generation - requires Reserved Nodes
            engine_analysis[family]["database_savings_plans_eligibility"] = "reserved_nodes_only"
            engine_analysis[family]["recommendation"] = "Requires Reserved Nodes - consider upgrading to latest generation"
            # Assume Redis/Memcached for older generations
            redis_spend += spend * 0.7  # Rough estimate
            memcached_spend += spend * 0.3
        else:
            # Unknown family
            engine_analysis[family]["database_savings_plans_eligibility"] = "unknown"
            engine_analysis[family]["recommendation"] = "Verify Database Savings Plans eligibility and engine compatibility"
            unknown_spend += spend
    
    # Calculate hourly spend by category
    total_hours = usage_data.get("lookback_period_days", 30) * 24
    valkey_hourly_spend = valkey_spend / total_hours if total_hours > 0 else 0.0
    redis_hourly_spend = redis_spend / total_hours if total_hours > 0 else 0.0
    memcached_hourly_spend = memcached_spend / total_hours if total_hours > 0 else 0.0
    unknown_hourly_spend = unknown_spend / total_hours if total_hours > 0 else 0.0
    
    # Generate ElastiCache-specific recommendations
    recommendations = []
    reserved_nodes_required = []
    
    # Database Savings Plans recommendation (Valkey engine only)
    # Since we can't determine actual engine usage, provide conditional recommendation
    if unknown_hourly_spend > 0:  # Latest generation instances with unknown engine
        # Conservative recommendation assuming some Valkey usage
        estimated_valkey_percentage = 0.3  # Conservative estimate
        estimated_valkey_hourly = unknown_hourly_spend * estimated_valkey_percentage
        
        if estimated_valkey_hourly > 0.1:  # Minimum threshold
            coverage_target = 0.85
            recommended_commitment = estimated_valkey_hourly * coverage_target
            
            # Calculate savings (20% discount for Database Savings Plans)
            discount_rate = 0.20
            annual_savings = recommended_commitment * discount_rate * 8760
            
            recommendations.append({
                "commitment_type": "database_savings_plans",
                "engine": "valkey_only",
                "hourly_commitment": round(recommended_commitment, 2),
                "estimated_eligible_spend": round(estimated_valkey_hourly, 2),
                "projected_coverage": round(coverage_target * 100, 1),
                "estimated_annual_savings": round(annual_savings, 2),
                "discount_rate": discount_rate * 100,
                "rationale": "Database Savings Plans available for Valkey engine on latest-generation instances only",
                "requirements": [
                    "Must be using Valkey engine (not Redis or Memcached)",
                    "Latest-generation instance families only (M7g, R7g, T4g)",
                    "Verify engine compatibility before purchasing"
                ],
                "confidence": "conditional",
                "verification_needed": "Confirm Valkey engine usage and instance generation"
            })
    
    # Reserved Nodes recommendations (Redis and Memcached)
    redis_memcached_hourly = redis_hourly_spend + memcached_hourly_spend
    if redis_memcached_hourly > 0:
        coverage_target = 0.90  # Higher coverage for Reserved Nodes (instance-specific)
        recommended_reserved_commitment = redis_memcached_hourly * coverage_target
        
        # Calculate Reserved Nodes savings (typical 30% discount)
        reserved_discount_rate = 0.30
        reserved_annual_savings = recommended_reserved_commitment * reserved_discount_rate * 8760
        
        reserved_nodes_required.append({
            "commitment_type": "reserved_nodes",
            "engines": ["redis", "memcached"],
            "hourly_commitment": round(recommended_reserved_commitment, 2),
            "eligible_hourly_spend": round(redis_memcached_hourly, 2),
            "projected_coverage": round(coverage_target * 100, 1),
            "estimated_annual_savings": round(reserved_annual_savings, 2),
            "discount_rate": reserved_discount_rate * 100,
            "rationale": "Redis and Memcached engines require Reserved Nodes for cost optimization",
            "considerations": [
                "Node type and region specific commitments",
                "Consider upgrading to latest-generation instances with Valkey engine",
                "Reserved Nodes can be sold in Reserved Instance Marketplace"
            ]
        })
    
    # Engine migration recommendations
    migration_recommendations = []
    
    if redis_hourly_spend > 0 or memcached_hourly_spend > 0:
        migration_recommendations.append({
            "migration_type": "engine_modernization",
            "from_engines": ["redis", "memcached"],
            "to_engine": "valkey",
            "benefits": [
                "Database Savings Plans eligibility (20% discount)",
                "Better flexibility than Reserved Nodes",
                "Latest-generation instance compatibility",
                "Open-source Redis-compatible engine"
            ],
            "considerations": [
                "Application compatibility testing required",
                "Migration planning and downtime considerations",
                "Performance validation in new environment"
            ],
            "estimated_savings_improvement": "Switch from Reserved Nodes to Database Savings Plans flexibility"
        })
    
    # Cluster configuration considerations
    cluster_considerations = [
        "Replication Groups: Each node in replication group covered separately",
        "Cluster Mode: Sharded clusters - each shard node eligible for coverage",
        "Multi-AZ: Replica nodes in different AZs covered by same Database Savings Plans",
        "Auto Scaling: Plan Database Savings Plans for maximum expected node count",
        "Backup and restore: Snapshot storage not covered by Database Savings Plans"
    ]
    
    # Performance optimization recommendations
    performance_considerations = [
        "Node sizing: Right-size nodes based on memory and network requirements",
        "Connection pooling: Optimize client connections to reduce node resource usage",
        "Data structure optimization: Use appropriate Redis/Valkey data structures",
        "Eviction policies: Configure appropriate eviction policies for memory management",
        "Monitoring: Use CloudWatch metrics for performance and utilization analysis"
    ]
    
    logger.info(f"ElastiCache-specific analysis complete: ${unknown_hourly_spend:.2f}/hour potential Valkey, "
                f"${redis_memcached_hourly:.2f}/hour requires Reserved Nodes")
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "reserved_nodes_required": reserved_nodes_required,
            "engine_analysis": engine_analysis,
            "migration_recommendations": migration_recommendations,
            "cluster_considerations": cluster_considerations,
            "performance_considerations": performance_considerations,
            "summary": {
                "total_elasticache_hourly_spend": elasticache_hourly_spend,
                "estimated_valkey_hourly_spend": valkey_hourly_spend,
                "redis_memcached_hourly_spend": redis_memcached_hourly,
                "unknown_engine_hourly_spend": unknown_hourly_spend,
                "database_savings_plans_eligible_percentage": (valkey_hourly_spend / elasticache_hourly_spend * 100) if elasticache_hourly_spend > 0 else 0.0,
                "reserved_nodes_required_percentage": (redis_memcached_hourly / elasticache_hourly_spend * 100) if elasticache_hourly_spend > 0 else 0.0,
                "verification_required": unknown_hourly_spend > 0
            },
            "important_notes": [
                "Database Savings Plans for ElastiCache support Valkey engine ONLY",
                "Redis and Memcached workloads require Reserved Nodes for cost optimization",
                "Engine verification required for accurate recommendations",
                "Consider migrating to Valkey for Database Savings Plans eligibility"
            ]
        },
        "message": f"ElastiCache-specific recommendations generated with engine-specific guidance. Verification required for ${unknown_hourly_spend:.2f}/hour usage."
    }


def generate_other_services_recommendations(
    usage_data: Dict[str, Any],
    service_usage_breakdown: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate Database Savings Plans recommendations for other supported services.
    
    Covers:
    - Amazon DocumentDB (MongoDB-compatible)
    - Amazon Neptune (Graph database)
    - Amazon Keyspaces (Apache Cassandra-compatible)
    - Amazon Timestream (Time series database)
    - AWS Database Migration Service (DMS)
    
    Args:
        usage_data: Overall database usage analysis results
        service_usage_breakdown: Usage data for other database services
        
    Returns:
        Dictionary containing service-specific recommendations for other services
    """
    logger.info("Generating recommendations for other Database Savings Plans eligible services")
    
    # Extract usage for each service
    services_analysis = {}
    total_other_services_spend = 0.0
    
    # Define service configurations and eligibility
    service_configs = {
        "documentdb": {
            "name": "Amazon DocumentDB",
            "description": "MongoDB-compatible document database",
            "latest_generation_families": {'docdb.r7g', 'docdb.t4g'},
            "older_generation_families": {'docdb.r5', 'docdb.r6g', 'docdb.t3'},
            "database_savings_plans_eligible": True,
            "considerations": [
                "Cluster-based architecture - each instance covered separately",
                "Storage auto-scaling not covered by Database Savings Plans",
                "Cross-region backup storage not covered"
            ]
        },
        "neptune": {
            "name": "Amazon Neptune",
            "description": "Graph database for highly connected datasets",
            "latest_generation_families": {'neptune.r7g', 'neptune.t4g'},
            "older_generation_families": {'neptune.r5', 'neptune.r6g', 'neptune.t3'},
            "database_savings_plans_eligible": True,
            "considerations": [
                "Graph workloads often have predictable compute patterns",
                "Gremlin and SPARQL query optimization affects compute usage",
                "Cluster storage not covered by Database Savings Plans"
            ]
        },
        "keyspaces": {
            "name": "Amazon Keyspaces",
            "description": "Apache Cassandra-compatible database",
            "latest_generation_families": None,  # Serverless service
            "older_generation_families": None,
            "database_savings_plans_eligible": True,
            "considerations": [
                "Serverless service - automatically eligible for Database Savings Plans",
                "On-demand and provisioned capacity modes supported",
                "Multi-region replication requires separate coverage per region"
            ]
        },
        "timestream": {
            "name": "Amazon Timestream",
            "description": "Time series database for IoT and operational applications",
            "latest_generation_families": None,  # Serverless service
            "older_generation_families": None,
            "database_savings_plans_eligible": True,
            "considerations": [
                "Serverless service - automatically eligible for Database Savings Plans",
                "Separate compute and storage pricing - Database Savings Plans cover compute only",
                "Query performance optimization affects compute costs"
            ]
        },
        "dms": {
            "name": "AWS Database Migration Service",
            "description": "Database migration and replication service",
            "latest_generation_families": {'dms.r7g', 'dms.t4g'},
            "older_generation_families": {'dms.r5', 'dms.r6g', 'dms.t3'},
            "database_savings_plans_eligible": True,
            "considerations": [
                "Replication instance compute hours covered by Database Savings Plans",
                "Data transfer costs not covered",
                "Consider instance sizing based on migration workload patterns"
            ]
        }
    }
    
    # Analyze each service
    for service_key, service_config in service_configs.items():
        service_usage = service_usage_breakdown.get(service_key, {})
        service_spend = service_usage.get("total_spend", 0.0)
        
        if service_spend > 0:
            total_other_services_spend += service_spend
            
            # Calculate hourly spend
            total_hours = usage_data.get("lookback_period_days", 30) * 24
            service_hourly_spend = service_spend / total_hours if total_hours > 0 else 0.0
            
            # Analyze instance families if applicable
            instance_analysis = {}
            eligible_spend = 0.0
            ineligible_spend = 0.0
            
            instance_types = service_usage.get("instance_types", {})
            
            if instance_types and service_config["latest_generation_families"]:
                # Service has instance families to analyze
                for instance_type, spend in instance_types.items():
                    # Extract family from instance type (e.g., "documentdb.r7g.large" -> "docdb.r7g")
                    if '.' in instance_type:
                        parts = instance_type.split('.')
                        if len(parts) >= 2:
                            # Map service prefixes to AWS family prefixes
                            service_to_family_prefix = {
                                'documentdb': 'docdb',
                                'neptune': 'neptune',
                                'dms': 'dms'
                            }
                            
                            # Get the correct family prefix
                            family_prefix = service_to_family_prefix.get(service_key, parts[0])
                            family = f"{family_prefix}.{parts[1]}"
                        else:
                            family = instance_type
                    else:
                        family = instance_type
                    
                    is_latest = any(family.startswith(prefix) for prefix in service_config["latest_generation_families"])
                    is_older = any(family.startswith(prefix) for prefix in service_config["older_generation_families"])
                    
                    if is_latest and not is_older:
                        eligible_spend += spend
                        instance_analysis[family] = {
                            "spend": spend,
                            "eligibility": "database_savings_plans",
                            "recommendation": "Eligible for Database Savings Plans"
                        }
                    elif is_older:
                        ineligible_spend += spend
                        instance_analysis[family] = {
                            "spend": spend,
                            "eligibility": "reserved_instances_only",
                            "recommendation": "Consider upgrading to latest generation or use Reserved Instances"
                        }
            else:
                # Serverless service or no instance family data - all spend eligible
                eligible_spend = service_spend
            
            # Generate recommendation for this service
            if eligible_spend > 0:
                eligible_hourly_spend = eligible_spend / total_hours if total_hours > 0 else 0.0
                
                # Use service-appropriate coverage target
                if service_key in ["keyspaces", "timestream"]:
                    coverage_target = 0.90  # Higher for serverless services with predictable patterns
                else:
                    coverage_target = 0.85  # Conservative for instance-based services
                
                recommended_commitment = eligible_hourly_spend * coverage_target
                
                # Calculate savings (20% discount for Database Savings Plans)
                discount_rate = 0.20
                annual_savings = recommended_commitment * discount_rate * 8760
                
                services_analysis[service_key] = {
                    "service_name": service_config["name"],
                    "description": service_config["description"],
                    "total_spend": service_spend,
                    "hourly_spend": service_hourly_spend,
                    "eligible_spend": eligible_spend,
                    "ineligible_spend": ineligible_spend,
                    "instance_analysis": instance_analysis,
                    "recommendation": {
                        "commitment_type": "database_savings_plans",
                        "hourly_commitment": round(recommended_commitment, 2),
                        "eligible_hourly_spend": round(eligible_hourly_spend, 2),
                        "projected_coverage": round(coverage_target * 100, 1),
                        "estimated_annual_savings": round(annual_savings, 2),
                        "discount_rate": discount_rate * 100,
                        "rationale": f"{service_config['name']} usage eligible for Database Savings Plans"
                    },
                    "considerations": service_config["considerations"],
                    "database_savings_plans_eligible": service_config["database_savings_plans_eligible"]
                }
    
    # Generate consolidated recommendations
    recommendations = []
    total_eligible_hourly_spend = 0.0
    total_estimated_savings = 0.0
    
    for service_key, analysis in services_analysis.items():
        if "recommendation" in analysis:
            recommendations.append({
                "service": service_key,
                "service_name": analysis["service_name"],
                **analysis["recommendation"]
            })
            total_eligible_hourly_spend += analysis["recommendation"]["eligible_hourly_spend"]
            total_estimated_savings += analysis["recommendation"]["estimated_annual_savings"]
    
    # Service-specific optimization opportunities
    optimization_opportunities = []
    
    # DocumentDB optimizations
    if "documentdb" in services_analysis:
        optimization_opportunities.append({
            "service": "documentdb",
            "opportunity": "Cluster optimization",
            "description": "Optimize DocumentDB cluster configuration and instance sizing",
            "recommendations": [
                "Right-size instances based on workload patterns",
                "Consider read replica scaling for read-heavy workloads",
                "Optimize index usage to reduce compute requirements"
            ]
        })
    
    # Neptune optimizations
    if "neptune" in services_analysis:
        optimization_opportunities.append({
            "service": "neptune",
            "opportunity": "Graph query optimization",
            "description": "Optimize graph queries and traversals for better performance",
            "recommendations": [
                "Optimize Gremlin/SPARQL queries for efficiency",
                "Consider graph data modeling improvements",
                "Use appropriate instance types for graph workloads"
            ]
        })
    
    # Keyspaces optimizations
    if "keyspaces" in services_analysis:
        optimization_opportunities.append({
            "service": "keyspaces",
            "opportunity": "Capacity mode optimization",
            "description": "Optimize between on-demand and provisioned capacity",
            "recommendations": [
                "Analyze traffic patterns for capacity mode selection",
                "Optimize partition key design for even distribution",
                "Consider auto-scaling for provisioned capacity"
            ]
        })
    
    # Timestream optimizations
    if "timestream" in services_analysis:
        optimization_opportunities.append({
            "service": "timestream",
            "opportunity": "Query and storage optimization",
            "description": "Optimize time series queries and data lifecycle",
            "recommendations": [
                "Optimize query patterns for time series data",
                "Configure appropriate data retention policies",
                "Use scheduled queries efficiently"
            ]
        })
    
    # DMS optimizations
    if "dms" in services_analysis:
        optimization_opportunities.append({
            "service": "dms",
            "opportunity": "Migration instance optimization",
            "description": "Right-size DMS replication instances",
            "recommendations": [
                "Size instances based on migration workload",
                "Consider Multi-AZ for production migrations",
                "Optimize replication instance utilization"
            ]
        })
    
    # Cross-service considerations
    cross_service_considerations = [
        "Multi-region deployments: Each region requires separate Database Savings Plans",
        "Service integration: Consider data flow patterns between services",
        "Monitoring: Use CloudWatch for cross-service performance analysis",
        "Cost allocation: Use cost allocation tags for service-specific tracking"
    ]
    
    logger.info(f"Other services analysis complete: {len(services_analysis)} services analyzed, "
                f"${total_eligible_hourly_spend:.2f}/hour eligible for Database Savings Plans")
    
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "services_analysis": services_analysis,
            "optimization_opportunities": optimization_opportunities,
            "cross_service_considerations": cross_service_considerations,
            "summary": {
                "total_services_analyzed": len(services_analysis),
                "total_other_services_hourly_spend": total_other_services_spend / (usage_data.get("lookback_period_days", 30) * 24),
                "total_eligible_hourly_spend": total_eligible_hourly_spend,
                "total_estimated_annual_savings": total_estimated_savings,
                "services_with_recommendations": len(recommendations)
            }
        },
        "message": f"Generated recommendations for {len(services_analysis)} other database services with ${total_estimated_savings:,.2f}/year potential savings"
    }


# ============================================================================
# Historical Tracking Functions
# ============================================================================

@dataclass
class HistoricalAnalysisRecord:
    """Represents a historical analysis record with timestamp."""
    analysis_id: str
    analysis_type: str  # "recommendations", "purchase_analyzer", "existing_commitments"
    timestamp: datetime
    region: Optional[str]
    lookback_period_days: int
    analysis_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def store_analysis_result(
    session_id: str,
    analysis_type: str,
    analysis_data: Dict[str, Any],
    region: Optional[str] = None,
    lookback_period_days: int = 30,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store analysis results with timestamps in session database.
    
    Args:
        session_id: Session ID for data storage
        analysis_type: Type of analysis ("recommendations", "purchase_analyzer", "existing_commitments")
        analysis_data: Analysis results to store
        region: AWS region analyzed
        lookback_period_days: Analysis period
        metadata: Additional metadata to store
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Storage confirmation with analysis_id
            - message: Status message
            
    Example:
        >>> result = analyze_database_usage()
        >>> storage_result = store_analysis_result(
        ...     session_id="session_123",
        ...     analysis_type="recommendations", 
        ...     analysis_data=result["data"]
        ... )
        >>> print(storage_result["data"]["analysis_id"])
    """
    from utils.session_manager import get_session_manager
    import uuid
    
    logger.info(f"Storing analysis result: type={analysis_type}, session={session_id}")
    
    # Validate inputs
    if not session_id:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Session ID is required"
        }
    
    if analysis_type not in ["recommendations", "purchase_analyzer", "existing_commitments"]:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": f"Invalid analysis type: {analysis_type}",
            "valid_values": ["recommendations", "purchase_analyzer", "existing_commitments"]
        }
    
    if not analysis_data:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Analysis data is required"
        }
    
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create historical record
        record = HistoricalAnalysisRecord(
            analysis_id=analysis_id,
            analysis_type=analysis_type,
            timestamp=timestamp,
            region=region,
            lookback_period_days=lookback_period_days,
            analysis_data=analysis_data,
            metadata=metadata or {}
        )
        
        # Prepare data for storage
        storage_data = [{
            "analysis_id": record.analysis_id,
            "analysis_type": record.analysis_type,
            "timestamp": record.timestamp.isoformat(),
            "region": record.region,
            "lookback_period_days": record.lookback_period_days,
            "analysis_data": json.dumps(record.analysis_data),
            "metadata": json.dumps(record.metadata)
        }]
        
        # Store in session database
        session_manager = get_session_manager()
        table_name = "database_savings_plans_history"
        
        success = session_manager.store_data(
            session_id=session_id,
            table_name=table_name,
            data=storage_data,
            replace=False  # Append to existing data
        )
        
        if success:
            logger.info(f"Stored analysis result: {analysis_id} in session {session_id}")
            return {
                "status": "success",
                "data": {
                    "analysis_id": analysis_id,
                    "timestamp": timestamp.isoformat(),
                    "table_name": table_name,
                    "session_id": session_id
                },
                "message": f"Analysis result stored successfully with ID: {analysis_id}"
            }
        else:
            logger.error(f"Failed to store analysis result in session {session_id}")
            return {
                "status": "error",
                "error_code": "StorageError",
                "message": "Failed to store analysis result in session database"
            }
            
    except Exception as e:
        logger.error(f"Error storing analysis result: {e}")
        return {
            "status": "error",
            "error_code": "StorageError",
            "message": f"Error storing analysis result: {str(e)}"
        }


def query_historical_data(
    session_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analysis_type: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Support retrieval of historical analysis data by date range.
    
    Args:
        session_id: Session ID for data retrieval
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        analysis_type: Filter by analysis type
        region: Filter by AWS region
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: List of historical analysis records
            - message: Status message
            
    Example:
        >>> historical_data = query_historical_data(
        ...     session_id="session_123",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     analysis_type="recommendations"
        ... )
        >>> for record in historical_data["data"]["records"]:
        ...     print(f"{record['timestamp']}: {record['analysis_type']}")
    """
    from utils.session_manager import get_session_manager
    
    logger.info(f"Querying historical data: session={session_id}, "
                f"start_date={start_date}, end_date={end_date}, type={analysis_type}")
    
    # Validate inputs
    if not session_id:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Session ID is required"
        }
    
    try:
        session_manager = get_session_manager()
        table_name = "database_savings_plans_history"
        
        # Build query with filters
        query_parts = ["SELECT * FROM database_savings_plans_history WHERE 1=1"]
        params = []
        
        # Add date range filters
        if start_date:
            query_parts.append("AND timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            query_parts.append("AND timestamp <= ?")
            params.append(end_date + "T23:59:59")  # Include full end date
        
        # Add analysis type filter
        if analysis_type:
            if analysis_type not in ["recommendations", "purchase_analyzer", "existing_commitments"]:
                return {
                    "status": "error",
                    "error_code": "ValidationError",
                    "message": f"Invalid analysis type: {analysis_type}",
                    "valid_values": ["recommendations", "purchase_analyzer", "existing_commitments"]
                }
            query_parts.append("AND analysis_type = ?")
            params.append(analysis_type)
        
        # Add region filter
        if region:
            query_parts.append("AND region = ?")
            params.append(region)
        
        # Order by timestamp descending (most recent first)
        query_parts.append("ORDER BY timestamp DESC")
        
        query = " ".join(query_parts)
        
        # Execute query
        results = session_manager.execute_query(
            session_id=session_id,
            query=query,
            params=tuple(params) if params else None
        )
        
        # Process results
        records = []
        for row in results:
            try:
                # Parse JSON fields
                analysis_data = json.loads(row.get("analysis_data", "{}"))
                metadata = json.loads(row.get("metadata", "{}"))
                
                record = {
                    "analysis_id": row.get("analysis_id"),
                    "analysis_type": row.get("analysis_type"),
                    "timestamp": row.get("timestamp"),
                    "region": row.get("region"),
                    "lookback_period_days": row.get("lookback_period_days"),
                    "analysis_data": analysis_data,
                    "metadata": metadata
                }
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse record {row.get('analysis_id')}: {e}")
                continue
        
        logger.info(f"Retrieved {len(records)} historical records from session {session_id}")
        
        return {
            "status": "success",
            "data": {
                "records": records,
                "total_count": len(records),
                "filters_applied": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "analysis_type": analysis_type,
                    "region": region
                }
            },
            "message": f"Retrieved {len(records)} historical analysis records"
        }
        
    except Exception as e:
        logger.error(f"Error querying historical data: {e}")
        return {
            "status": "error",
            "error_code": "QueryError",
            "message": f"Error querying historical data: {str(e)}"
        }


def compare_historical_analyses(
    session_id: str,
    analysis_id_1: str,
    analysis_id_2: str
) -> Dict[str, Any]:
    """
    Calculate changes in commitments and savings between two historical analyses.
    
    Args:
        session_id: Session ID for data retrieval
        analysis_id_1: First analysis ID (typically older)
        analysis_id_2: Second analysis ID (typically newer)
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Comparison results with changes
            - message: Status message
            
    Example:
        >>> comparison = compare_historical_analyses(
        ...     session_id="session_123",
        ...     analysis_id_1="older_analysis_id",
        ...     analysis_id_2="newer_analysis_id"
        ... )
        >>> print(f"Savings change: {comparison['data']['savings_change']}")
    """
    from utils.session_manager import get_session_manager
    
    logger.info(f"Comparing historical analyses: {analysis_id_1} vs {analysis_id_2}")
    
    # Validate inputs
    if not session_id:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Session ID is required"
        }
    
    if not analysis_id_1 or not analysis_id_2:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Both analysis IDs are required"
        }
    
    try:
        session_manager = get_session_manager()
        
        # Retrieve both analyses
        query = "SELECT * FROM database_savings_plans_history WHERE analysis_id IN (?, ?)"
        results = session_manager.execute_query(
            session_id=session_id,
            query=query,
            params=(analysis_id_1, analysis_id_2)
        )
        
        if len(results) != 2:
            return {
                "status": "error",
                "error_code": "NotFound",
                "message": f"Could not find both analyses. Found {len(results)} of 2 required."
            }
        
        # Parse and organize analyses
        analyses = {}
        for row in results:
            analysis_id = row.get("analysis_id")
            try:
                analysis_data = json.loads(row.get("analysis_data", "{}"))
                analyses[analysis_id] = {
                    "analysis_id": analysis_id,
                    "analysis_type": row.get("analysis_type"),
                    "timestamp": row.get("timestamp"),
                    "region": row.get("region"),
                    "lookback_period_days": row.get("lookback_period_days"),
                    "analysis_data": analysis_data,
                    "metadata": json.loads(row.get("metadata", "{}"))
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis {analysis_id}: {e}")
                return {
                    "status": "error",
                    "error_code": "ParseError",
                    "message": f"Failed to parse analysis data for {analysis_id}"
                }
        
        analysis_1 = analyses[analysis_id_1]
        analysis_2 = analyses[analysis_id_2]
        
        # Ensure both analyses are the same type
        if analysis_1["analysis_type"] != analysis_2["analysis_type"]:
            return {
                "status": "error",
                "error_code": "ValidationError",
                "message": f"Cannot compare different analysis types: {analysis_1['analysis_type']} vs {analysis_2['analysis_type']}"
            }
        
        # Calculate changes based on analysis type
        analysis_type = analysis_1["analysis_type"]
        
        if analysis_type == "recommendations":
            comparison_result = _compare_recommendations(analysis_1, analysis_2)
        elif analysis_type == "purchase_analyzer":
            comparison_result = _compare_purchase_analyzer(analysis_1, analysis_2)
        elif analysis_type == "existing_commitments":
            comparison_result = _compare_existing_commitments(analysis_1, analysis_2)
        else:
            return {
                "status": "error",
                "error_code": "UnsupportedType",
                "message": f"Comparison not supported for analysis type: {analysis_type}"
            }
        
        # Add metadata to comparison
        comparison_result.update({
            "comparison_metadata": {
                "analysis_1": {
                    "id": analysis_1["analysis_id"],
                    "timestamp": analysis_1["timestamp"],
                    "region": analysis_1["region"]
                },
                "analysis_2": {
                    "id": analysis_2["analysis_id"],
                    "timestamp": analysis_2["timestamp"],
                    "region": analysis_2["region"]
                },
                "time_difference_hours": _calculate_time_difference(
                    analysis_1["timestamp"], 
                    analysis_2["timestamp"]
                )
            }
        })
        
        logger.info(f"Historical comparison complete: {analysis_type} analysis")
        
        return {
            "status": "success",
            "data": comparison_result,
            "message": f"Successfully compared {analysis_type} analyses"
        }
        
    except Exception as e:
        logger.error(f"Error comparing historical analyses: {e}")
        return {
            "status": "error",
            "error_code": "ComparisonError",
            "message": f"Error comparing historical analyses: {str(e)}"
        }


def identify_usage_trends(
    session_id: str,
    analysis_type: str = "recommendations",
    region: Optional[str] = None,
    min_records: int = 3
) -> Dict[str, Any]:
    """
    Identify increasing or decreasing usage patterns that affect recommendations.
    
    Args:
        session_id: Session ID for data retrieval
        analysis_type: Type of analysis to analyze trends for
        region: Filter by AWS region
        min_records: Minimum number of records required for trend analysis
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Trend analysis results
            - message: Status message
            
    Example:
        >>> trends = identify_usage_trends(
        ...     session_id="session_123",
        ...     analysis_type="recommendations",
        ...     min_records=5
        ... )
        >>> print(f"Usage trend: {trends['data']['usage_trend']}")
    """
    from utils.session_manager import get_session_manager
    
    logger.info(f"Identifying usage trends: session={session_id}, type={analysis_type}")
    
    # Validate inputs
    if not session_id:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Session ID is required"
        }
    
    if analysis_type not in ["recommendations", "purchase_analyzer", "existing_commitments"]:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": f"Invalid analysis type: {analysis_type}",
            "valid_values": ["recommendations", "purchase_analyzer", "existing_commitments"]
        }
    
    try:
        # Query historical data
        historical_result = query_historical_data(
            session_id=session_id,
            analysis_type=analysis_type,
            region=region
        )
        
        if historical_result.get("status") != "success":
            return historical_result
        
        records = historical_result["data"]["records"]
        
        if len(records) < min_records:
            return {
                "status": "warning",
                "data": {
                    "trend_analysis": "insufficient_data",
                    "records_found": len(records),
                    "min_records_required": min_records
                },
                "message": f"Insufficient data for trend analysis. Found {len(records)} records, need at least {min_records}"
            }
        
        # Sort records by timestamp (oldest first for trend analysis)
        records.sort(key=lambda x: x["timestamp"])
        
        # Extract time series data based on analysis type
        if analysis_type == "recommendations":
            trend_data = _extract_recommendations_trend_data(records)
        elif analysis_type == "purchase_analyzer":
            trend_data = _extract_purchase_analyzer_trend_data(records)
        elif analysis_type == "existing_commitments":
            trend_data = _extract_existing_commitments_trend_data(records)
        else:
            return {
                "status": "error",
                "error_code": "UnsupportedType",
                "message": f"Trend analysis not supported for type: {analysis_type}"
            }
        
        # Analyze trends
        trend_analysis = _analyze_trends(trend_data)
        
        # Generate insights and recommendations
        insights = _generate_trend_insights(trend_analysis, analysis_type)
        
        logger.info(f"Usage trend analysis complete: {trend_analysis['overall_trend']} trend detected")
        
        return {
            "status": "success",
            "data": {
                "trend_analysis": trend_analysis,
                "insights": insights,
                "records_analyzed": len(records),
                "time_period": {
                    "start": records[0]["timestamp"],
                    "end": records[-1]["timestamp"]
                }
            },
            "message": f"Analyzed trends across {len(records)} historical records"
        }
        
    except Exception as e:
        logger.error(f"Error identifying usage trends: {e}")
        return {
            "status": "error",
            "error_code": "TrendAnalysisError",
            "message": f"Error identifying usage trends: {str(e)}"
        }


def format_data_for_visualization(
    session_id: str,
    analysis_type: str = "recommendations",
    region: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Structure historical data for trend analysis and visualization.
    
    Args:
        session_id: Session ID for data retrieval
        analysis_type: Type of analysis to format
        region: Filter by AWS region
        start_date: Start date for data range
        end_date: End date for data range
        
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - data: Visualization-friendly data structure
            - message: Status message
            
    Example:
        >>> viz_data = format_data_for_visualization(
        ...     session_id="session_123",
        ...     analysis_type="recommendations"
        ... )
        >>> chart_data = viz_data["data"]["time_series"]
    """
    from utils.session_manager import get_session_manager
    
    logger.info(f"Formatting data for visualization: session={session_id}, type={analysis_type}")
    
    # Validate inputs
    if not session_id:
        return {
            "status": "error",
            "error_code": "ValidationError",
            "message": "Session ID is required"
        }
    
    try:
        # Query historical data
        historical_result = query_historical_data(
            session_id=session_id,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
            region=region
        )
        
        if historical_result.get("status") != "success":
            return historical_result
        
        records = historical_result["data"]["records"]
        
        if not records:
            return {
                "status": "warning",
                "data": {
                    "time_series": [],
                    "summary_stats": {},
                    "chart_config": {}
                },
                "message": "No historical data found for visualization"
            }
        
        # Sort records by timestamp
        records.sort(key=lambda x: x["timestamp"])
        
        # Format data based on analysis type
        if analysis_type == "recommendations":
            viz_data = _format_recommendations_for_visualization(records)
        elif analysis_type == "purchase_analyzer":
            viz_data = _format_purchase_analyzer_for_visualization(records)
        elif analysis_type == "existing_commitments":
            viz_data = _format_existing_commitments_for_visualization(records)
        else:
            return {
                "status": "error",
                "error_code": "UnsupportedType",
                "message": f"Visualization not supported for type: {analysis_type}"
            }
        
        # Add metadata
        viz_data.update({
            "metadata": {
                "analysis_type": analysis_type,
                "region": region,
                "records_count": len(records),
                "time_range": {
                    "start": records[0]["timestamp"],
                    "end": records[-1]["timestamp"]
                },
                "generated_at": datetime.now().isoformat()
            }
        })
        
        logger.info(f"Formatted {len(records)} records for visualization")
        
        return {
            "status": "success",
            "data": viz_data,
            "message": f"Formatted {len(records)} historical records for visualization"
        }
        
    except Exception as e:
        logger.error(f"Error formatting data for visualization: {e}")
        return {
            "status": "error",
            "error_code": "FormattingError",
            "message": f"Error formatting data for visualization: {str(e)}"
        }


# ============================================================================
# Helper Functions for Historical Analysis
# ============================================================================

def _compare_recommendations(analysis_1: Dict[str, Any], analysis_2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two recommendations analyses."""
    data_1 = analysis_1["analysis_data"]
    data_2 = analysis_2["analysis_data"]
    
    # Extract key metrics
    recommendations_1 = data_1.get("recommendations", [])
    recommendations_2 = data_2.get("recommendations", [])
    
    # Get best recommendation from each (highest savings)
    best_rec_1 = max(recommendations_1, key=lambda x: x.get("estimated_annual_savings", 0)) if recommendations_1 else None
    best_rec_2 = max(recommendations_2, key=lambda x: x.get("estimated_annual_savings", 0)) if recommendations_2 else None
    
    if not best_rec_1 or not best_rec_2:
        return {
            "comparison_type": "recommendations",
            "changes": {
                "recommendations_available": {
                    "before": len(recommendations_1),
                    "after": len(recommendations_2),
                    "change": len(recommendations_2) - len(recommendations_1)
                }
            },
            "summary": "Insufficient recommendation data for detailed comparison"
        }
    
    # Calculate changes
    hourly_commitment_change = best_rec_2.get("hourly_commitment", 0) - best_rec_1.get("hourly_commitment", 0)
    savings_change = best_rec_2.get("estimated_annual_savings", 0) - best_rec_1.get("estimated_annual_savings", 0)
    coverage_change = best_rec_2.get("projected_coverage", 0) - best_rec_1.get("projected_coverage", 0)
    utilization_change = best_rec_2.get("projected_utilization", 0) - best_rec_1.get("projected_utilization", 0)
    
    # Helper function to safely calculate percentage change
    def safe_percentage_change(change_value: float, before_value: float) -> float:
        """Calculate percentage change, handling division by zero."""
        if before_value == 0:
            return 0.0 if change_value == 0 else float('inf') if change_value > 0 else float('-inf')
        return (change_value / before_value) * 100
    
    return {
        "comparison_type": "recommendations",
        "changes": {
            "hourly_commitment": {
                "before": best_rec_1.get("hourly_commitment", 0),
                "after": best_rec_2.get("hourly_commitment", 0),
                "change": hourly_commitment_change,
                "change_percentage": safe_percentage_change(hourly_commitment_change, best_rec_1.get("hourly_commitment", 0))
            },
            "estimated_annual_savings": {
                "before": best_rec_1.get("estimated_annual_savings", 0),
                "after": best_rec_2.get("estimated_annual_savings", 0),
                "change": savings_change,
                "change_percentage": safe_percentage_change(savings_change, best_rec_1.get("estimated_annual_savings", 0))
            },
            "projected_coverage": {
                "before": best_rec_1.get("projected_coverage", 0),
                "after": best_rec_2.get("projected_coverage", 0),
                "change": coverage_change
            },
            "projected_utilization": {
                "before": best_rec_1.get("projected_utilization", 0),
                "after": best_rec_2.get("projected_utilization", 0),
                "change": utilization_change
            }
        },
        "summary": _generate_recommendations_comparison_summary(hourly_commitment_change, savings_change, coverage_change)
    }


def _compare_purchase_analyzer(analysis_1: Dict[str, Any], analysis_2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two purchase analyzer analyses."""
    data_1 = analysis_1["analysis_data"]
    data_2 = analysis_2["analysis_data"]
    
    # Calculate changes in key metrics
    coverage_change = data_2.get("projected_coverage", 0) - data_1.get("projected_coverage", 0)
    utilization_change = data_2.get("projected_utilization", 0) - data_1.get("projected_utilization", 0)
    savings_change = data_2.get("estimated_annual_savings", 0) - data_1.get("estimated_annual_savings", 0)
    cost_change = data_2.get("projected_annual_cost", 0) - data_1.get("projected_annual_cost", 0)
    
    return {
        "comparison_type": "purchase_analyzer",
        "changes": {
            "hourly_commitment": {
                "before": data_1.get("hourly_commitment", 0),
                "after": data_2.get("hourly_commitment", 0),
                "change": data_2.get("hourly_commitment", 0) - data_1.get("hourly_commitment", 0)
            },
            "projected_coverage": {
                "before": data_1.get("projected_coverage", 0),
                "after": data_2.get("projected_coverage", 0),
                "change": coverage_change
            },
            "projected_utilization": {
                "before": data_1.get("projected_utilization", 0),
                "after": data_2.get("projected_utilization", 0),
                "change": utilization_change
            },
            "estimated_annual_savings": {
                "before": data_1.get("estimated_annual_savings", 0),
                "after": data_2.get("estimated_annual_savings", 0),
                "change": savings_change
            },
            "projected_annual_cost": {
                "before": data_1.get("projected_annual_cost", 0),
                "after": data_2.get("projected_annual_cost", 0),
                "change": cost_change
            }
        },
        "summary": _generate_purchase_analyzer_comparison_summary(coverage_change, utilization_change, savings_change)
    }


def _compare_existing_commitments(analysis_1: Dict[str, Any], analysis_2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two existing commitments analyses."""
    data_1 = analysis_1["analysis_data"]
    data_2 = analysis_2["analysis_data"]
    
    plans_1 = data_1.get("existing_plans", [])
    plans_2 = data_2.get("existing_plans", [])
    
    # Calculate aggregate metrics
    total_commitment_1 = sum(plan.get("hourly_commitment", 0) for plan in plans_1)
    total_commitment_2 = sum(plan.get("hourly_commitment", 0) for plan in plans_2)
    
    avg_utilization_1 = sum(plan.get("utilization_percentage", 0) for plan in plans_1) / len(plans_1) if plans_1 else 0
    avg_utilization_2 = sum(plan.get("utilization_percentage", 0) for plan in plans_2) / len(plans_2) if plans_2 else 0
    
    avg_coverage_1 = sum(plan.get("coverage_percentage", 0) for plan in plans_1) / len(plans_1) if plans_1 else 0
    avg_coverage_2 = sum(plan.get("coverage_percentage", 0) for plan in plans_2) / len(plans_2) if plans_2 else 0
    
    return {
        "comparison_type": "existing_commitments",
        "changes": {
            "number_of_plans": {
                "before": len(plans_1),
                "after": len(plans_2),
                "change": len(plans_2) - len(plans_1)
            },
            "total_hourly_commitment": {
                "before": total_commitment_1,
                "after": total_commitment_2,
                "change": total_commitment_2 - total_commitment_1
            },
            "average_utilization": {
                "before": avg_utilization_1,
                "after": avg_utilization_2,
                "change": avg_utilization_2 - avg_utilization_1
            },
            "average_coverage": {
                "before": avg_coverage_1,
                "after": avg_coverage_2,
                "change": avg_coverage_2 - avg_coverage_1
            }
        },
        "summary": _generate_existing_commitments_comparison_summary(
            len(plans_2) - len(plans_1),
            avg_utilization_2 - avg_utilization_1,
            avg_coverage_2 - avg_coverage_1
        )
    }


def _calculate_time_difference(timestamp_1: str, timestamp_2: str) -> float:
    """Calculate time difference in hours between two timestamps."""
    try:
        dt1 = datetime.fromisoformat(timestamp_1.replace('Z', '+00:00'))
        dt2 = datetime.fromisoformat(timestamp_2.replace('Z', '+00:00'))
        return abs((dt2 - dt1).total_seconds() / 3600)
    except Exception:
        return 0.0


def _extract_recommendations_trend_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract trend data from recommendations records."""
    trend_data = []
    
    for record in records:
        data = record["analysis_data"]
        recommendations = data.get("recommendations", [])
        
        if recommendations:
            # Get best recommendation (highest savings)
            best_rec = max(recommendations, key=lambda x: x.get("estimated_annual_savings", 0))
            
            trend_data.append({
                "timestamp": record["timestamp"],
                "hourly_commitment": best_rec.get("hourly_commitment", 0),
                "estimated_annual_savings": best_rec.get("estimated_annual_savings", 0),
                "projected_coverage": best_rec.get("projected_coverage", 0),
                "projected_utilization": best_rec.get("projected_utilization", 0),
                "eligible_hourly_spend": data.get("eligible_hourly_spend", 0),
                "total_hourly_spend": data.get("total_hourly_spend", 0)
            })
    
    return trend_data


def _extract_purchase_analyzer_trend_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract trend data from purchase analyzer records."""
    trend_data = []
    
    for record in records:
        data = record["analysis_data"]
        
        trend_data.append({
            "timestamp": record["timestamp"],
            "hourly_commitment": data.get("hourly_commitment", 0),
            "projected_coverage": data.get("projected_coverage", 0),
            "projected_utilization": data.get("projected_utilization", 0),
            "estimated_annual_savings": data.get("estimated_annual_savings", 0),
            "projected_annual_cost": data.get("projected_annual_cost", 0),
            "current_usage": data.get("current_usage", 0),
            "projected_usage": data.get("projected_usage", 0)
        })
    
    return trend_data


def _extract_existing_commitments_trend_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract trend data from existing commitments records."""
    trend_data = []
    
    for record in records:
        data = record["analysis_data"]
        plans = data.get("existing_plans", [])
        
        # Calculate aggregate metrics
        total_commitment = sum(plan.get("hourly_commitment", 0) for plan in plans)
        avg_utilization = sum(plan.get("utilization_percentage", 0) for plan in plans) / len(plans) if plans else 0
        avg_coverage = sum(plan.get("coverage_percentage", 0) for plan in plans) / len(plans) if plans else 0
        total_unused = sum(plan.get("unused_commitment_hourly", 0) for plan in plans)
        
        trend_data.append({
            "timestamp": record["timestamp"],
            "number_of_plans": len(plans),
            "total_hourly_commitment": total_commitment,
            "average_utilization": avg_utilization,
            "average_coverage": avg_coverage,
            "total_unused_commitment": total_unused
        })
    
    return trend_data


def _analyze_trends(trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trends in time series data."""
    if len(trend_data) < 2:
        return {
            "overall_trend": "insufficient_data",
            "metrics": {}
        }
    
    # Calculate trends for each metric
    metrics = {}
    
    # Get all numeric keys from the first record
    numeric_keys = [k for k, v in trend_data[0].items() if k != "timestamp" and isinstance(v, (int, float))]
    
    for key in numeric_keys:
        values = [record[key] for record in trend_data]
        
        # Calculate simple linear trend
        if len(values) >= 2:
            # Simple trend calculation: compare first and last values
            first_value = values[0]
            last_value = values[-1]
            
            if first_value == 0 and last_value == 0:
                trend = "stable"
                change_percentage = 0.0
            elif first_value == 0:
                trend = "increasing"
                change_percentage = 100.0
            else:
                change_percentage = ((last_value - first_value) / first_value) * 100
                
                if abs(change_percentage) < 5:  # Less than 5% change
                    trend = "stable"
                elif change_percentage > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            
            metrics[key] = {
                "trend": trend,
                "first_value": first_value,
                "last_value": last_value,
                "change": last_value - first_value,
                "change_percentage": change_percentage,
                "values": values
            }
    
    # Determine overall trend based on key metrics
    key_metrics = ["hourly_commitment", "estimated_annual_savings", "total_hourly_commitment"]
    trend_scores = []
    
    for key in key_metrics:
        if key in metrics:
            if metrics[key]["trend"] == "increasing":
                trend_scores.append(1)
            elif metrics[key]["trend"] == "decreasing":
                trend_scores.append(-1)
            else:
                trend_scores.append(0)
    
    if not trend_scores:
        overall_trend = "stable"
    else:
        avg_score = sum(trend_scores) / len(trend_scores)
        if avg_score > 0.3:
            overall_trend = "increasing"
        elif avg_score < -0.3:
            overall_trend = "decreasing"
        else:
            overall_trend = "stable"
    
    return {
        "overall_trend": overall_trend,
        "metrics": metrics,
        "data_points": len(trend_data),
        "time_span_hours": _calculate_time_difference(trend_data[0]["timestamp"], trend_data[-1]["timestamp"])
    }


def _generate_trend_insights(trend_analysis: Dict[str, Any], analysis_type: str) -> List[str]:
    """Generate insights based on trend analysis."""
    insights = []
    overall_trend = trend_analysis.get("overall_trend", "stable")
    metrics = trend_analysis.get("metrics", {})
    
    # Overall trend insight
    if overall_trend == "increasing":
        insights.append("📈 Overall usage and commitment trends are increasing")
    elif overall_trend == "decreasing":
        insights.append("📉 Overall usage and commitment trends are decreasing")
    else:
        insights.append("📊 Usage and commitment patterns are relatively stable")
    
    # Specific metric insights
    if analysis_type == "recommendations":
        if "eligible_hourly_spend" in metrics:
            spend_trend = metrics["eligible_hourly_spend"]["trend"]
            if spend_trend == "increasing":
                insights.append("💰 Eligible database spending is increasing - consider larger commitments")
            elif spend_trend == "decreasing":
                insights.append("💡 Eligible database spending is decreasing - review commitment levels")
        
        if "estimated_annual_savings" in metrics:
            savings_trend = metrics["estimated_annual_savings"]["trend"]
            if savings_trend == "increasing":
                insights.append("🎯 Potential savings opportunities are growing")
            elif savings_trend == "decreasing":
                insights.append("⚠️ Potential savings opportunities are declining")
    
    elif analysis_type == "existing_commitments":
        if "average_utilization" in metrics:
            util_trend = metrics["average_utilization"]["trend"]
            if util_trend == "decreasing":
                insights.append("⚠️ Commitment utilization is declining - review usage patterns")
            elif util_trend == "increasing":
                insights.append("✅ Commitment utilization is improving")
        
        if "total_unused_commitment" in metrics:
            unused_trend = metrics["total_unused_commitment"]["trend"]
            if unused_trend == "increasing":
                insights.append("🔍 Unused commitment is growing - consider reducing future commitments")
    
    # Time-based insights
    time_span = trend_analysis.get("time_span_hours", 0)
    if time_span > 0:
        if time_span < 24:
            insights.append(f"⏱️ Analysis covers {time_span:.1f} hours - short-term trend")
        elif time_span < 168:  # 1 week
            insights.append(f"📅 Analysis covers {time_span/24:.1f} days - medium-term trend")
        else:
            insights.append(f"📊 Analysis covers {time_span/168:.1f} weeks - long-term trend")
    
    return insights


def _format_recommendations_for_visualization(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format recommendations data for visualization."""
    time_series = []
    
    for record in records:
        data = record["analysis_data"]
        recommendations = data.get("recommendations", [])
        
        if recommendations:
            best_rec = max(recommendations, key=lambda x: x.get("estimated_annual_savings", 0))
            
            time_series.append({
                "timestamp": record["timestamp"],
                "hourly_commitment": best_rec.get("hourly_commitment", 0),
                "estimated_annual_savings": best_rec.get("estimated_annual_savings", 0),
                "projected_coverage": best_rec.get("projected_coverage", 0),
                "projected_utilization": best_rec.get("projected_utilization", 0),
                "eligible_hourly_spend": data.get("eligible_hourly_spend", 0),
                "total_hourly_spend": data.get("total_hourly_spend", 0)
            })
    
    # Calculate summary statistics
    if time_series:
        savings_values = [point["estimated_annual_savings"] for point in time_series]
        commitment_values = [point["hourly_commitment"] for point in time_series]
        
        summary_stats = {
            "avg_annual_savings": sum(savings_values) / len(savings_values),
            "max_annual_savings": max(savings_values),
            "min_annual_savings": min(savings_values),
            "avg_hourly_commitment": sum(commitment_values) / len(commitment_values),
            "max_hourly_commitment": max(commitment_values),
            "min_hourly_commitment": min(commitment_values)
        }
    else:
        summary_stats = {}
    
    # Chart configuration
    chart_config = {
        "chart_type": "line",
        "x_axis": "timestamp",
        "y_axes": [
            {
                "name": "Hourly Commitment ($)",
                "field": "hourly_commitment",
                "color": "#1f77b4"
            },
            {
                "name": "Annual Savings ($)",
                "field": "estimated_annual_savings",
                "color": "#ff7f0e"
            },
            {
                "name": "Coverage (%)",
                "field": "projected_coverage",
                "color": "#2ca02c"
            },
            {
                "name": "Utilization (%)",
                "field": "projected_utilization",
                "color": "#d62728"
            }
        ],
        "title": "Database Savings Plans Recommendations Over Time"
    }
    
    return {
        "time_series": time_series,
        "summary_stats": summary_stats,
        "chart_config": chart_config
    }


def _format_purchase_analyzer_for_visualization(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format purchase analyzer data for visualization."""
    time_series = []
    
    for record in records:
        data = record["analysis_data"]
        
        time_series.append({
            "timestamp": record["timestamp"],
            "hourly_commitment": data.get("hourly_commitment", 0),
            "projected_coverage": data.get("projected_coverage", 0),
            "projected_utilization": data.get("projected_utilization", 0),
            "estimated_annual_savings": data.get("estimated_annual_savings", 0),
            "projected_annual_cost": data.get("projected_annual_cost", 0)
        })
    
    # Summary statistics
    if time_series:
        coverage_values = [point["projected_coverage"] for point in time_series]
        utilization_values = [point["projected_utilization"] for point in time_series]
        
        summary_stats = {
            "avg_coverage": sum(coverage_values) / len(coverage_values),
            "avg_utilization": sum(utilization_values) / len(utilization_values),
            "coverage_range": [min(coverage_values), max(coverage_values)],
            "utilization_range": [min(utilization_values), max(utilization_values)]
        }
    else:
        summary_stats = {}
    
    chart_config = {
        "chart_type": "line",
        "x_axis": "timestamp",
        "y_axes": [
            {
                "name": "Coverage (%)",
                "field": "projected_coverage",
                "color": "#2ca02c"
            },
            {
                "name": "Utilization (%)",
                "field": "projected_utilization",
                "color": "#d62728"
            },
            {
                "name": "Annual Savings ($)",
                "field": "estimated_annual_savings",
                "color": "#ff7f0e"
            }
        ],
        "title": "Purchase Analyzer Scenarios Over Time"
    }
    
    return {
        "time_series": time_series,
        "summary_stats": summary_stats,
        "chart_config": chart_config
    }


def _format_existing_commitments_for_visualization(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format existing commitments data for visualization."""
    time_series = []
    
    for record in records:
        data = record["analysis_data"]
        plans = data.get("existing_plans", [])
        
        # Calculate aggregate metrics
        total_commitment = sum(plan.get("hourly_commitment", 0) for plan in plans)
        avg_utilization = sum(plan.get("utilization_percentage", 0) for plan in plans) / len(plans) if plans else 0
        avg_coverage = sum(plan.get("coverage_percentage", 0) for plan in plans) / len(plans) if plans else 0
        total_unused = sum(plan.get("unused_commitment_hourly", 0) for plan in plans)
        
        time_series.append({
            "timestamp": record["timestamp"],
            "number_of_plans": len(plans),
            "total_hourly_commitment": total_commitment,
            "average_utilization": avg_utilization,
            "average_coverage": avg_coverage,
            "total_unused_commitment": total_unused
        })
    
    # Summary statistics
    if time_series:
        utilization_values = [point["average_utilization"] for point in time_series]
        coverage_values = [point["average_coverage"] for point in time_series]
        
        summary_stats = {
            "avg_utilization": sum(utilization_values) / len(utilization_values),
            "avg_coverage": sum(coverage_values) / len(coverage_values),
            "utilization_trend": "increasing" if utilization_values[-1] > utilization_values[0] else "decreasing",
            "coverage_trend": "increasing" if coverage_values[-1] > coverage_values[0] else "decreasing"
        }
    else:
        summary_stats = {}
    
    chart_config = {
        "chart_type": "line",
        "x_axis": "timestamp",
        "y_axes": [
            {
                "name": "Number of Plans",
                "field": "number_of_plans",
                "color": "#1f77b4"
            },
            {
                "name": "Total Commitment ($/hour)",
                "field": "total_hourly_commitment",
                "color": "#ff7f0e"
            },
            {
                "name": "Average Utilization (%)",
                "field": "average_utilization",
                "color": "#2ca02c"
            },
            {
                "name": "Average Coverage (%)",
                "field": "average_coverage",
                "color": "#d62728"
            }
        ],
        "title": "Existing Database Savings Plans Performance Over Time"
    }
    
    return {
        "time_series": time_series,
        "summary_stats": summary_stats,
        "chart_config": chart_config
    }


def _generate_recommendations_comparison_summary(
    hourly_commitment_change: float,
    savings_change: float,
    coverage_change: float
) -> str:
    """Generate summary for recommendations comparison."""
    summary_parts = []
    
    if abs(hourly_commitment_change) > 0.01:
        if hourly_commitment_change > 0:
            summary_parts.append(f"Recommended commitment increased by ${hourly_commitment_change:.2f}/hour")
        else:
            summary_parts.append(f"Recommended commitment decreased by ${abs(hourly_commitment_change):.2f}/hour")
    
    if abs(savings_change) > 1:
        if savings_change > 0:
            summary_parts.append(f"Potential savings increased by ${savings_change:.2f}/year")
        else:
            summary_parts.append(f"Potential savings decreased by ${abs(savings_change):.2f}/year")
    
    if abs(coverage_change) > 1:
        if coverage_change > 0:
            summary_parts.append(f"Coverage improved by {coverage_change:.1f}%")
        else:
            summary_parts.append(f"Coverage decreased by {abs(coverage_change):.1f}%")
    
    if not summary_parts:
        return "No significant changes in recommendations"
    
    return "; ".join(summary_parts)


def _generate_purchase_analyzer_comparison_summary(
    coverage_change: float,
    utilization_change: float,
    savings_change: float
) -> str:
    """Generate summary for purchase analyzer comparison."""
    summary_parts = []
    
    if abs(coverage_change) > 1:
        if coverage_change > 0:
            summary_parts.append(f"Coverage improved by {coverage_change:.1f}%")
        else:
            summary_parts.append(f"Coverage decreased by {abs(coverage_change):.1f}%")
    
    if abs(utilization_change) > 1:
        if utilization_change > 0:
            summary_parts.append(f"Utilization improved by {utilization_change:.1f}%")
        else:
            summary_parts.append(f"Utilization decreased by {abs(utilization_change):.1f}%")
    
    if abs(savings_change) > 1:
        if savings_change > 0:
            summary_parts.append(f"Savings increased by ${savings_change:.2f}/year")
        else:
            summary_parts.append(f"Savings decreased by ${abs(savings_change):.2f}/year")
    
    if not summary_parts:
        return "No significant changes in purchase analyzer results"
    
    return "; ".join(summary_parts)


def _generate_existing_commitments_comparison_summary(
    plans_change: int,
    utilization_change: float,
    coverage_change: float
) -> str:
    """Generate summary for existing commitments comparison."""
    summary_parts = []
    
    if plans_change != 0:
        if plans_change > 0:
            summary_parts.append(f"Added {plans_change} new savings plan(s)")
        else:
            summary_parts.append(f"Removed {abs(plans_change)} savings plan(s)")
    
    if abs(utilization_change) > 1:
        if utilization_change > 0:
            summary_parts.append(f"Utilization improved by {utilization_change:.1f}%")
        else:
            summary_parts.append(f"Utilization decreased by {abs(utilization_change):.1f}%")
    
    if abs(coverage_change) > 1:
        if coverage_change > 0:
            summary_parts.append(f"Coverage improved by {coverage_change:.1f}%")
        else:
            summary_parts.append(f"Coverage decreased by {abs(coverage_change):.1f}%")
    
    if not summary_parts:
        return "No significant changes in existing commitments"
    
    return "; ".join(summary_parts)