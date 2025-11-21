"""
Cost Controller for CloudWatch optimization operations.

This module manages cost control flags, validates user preferences, calculates
functionality coverage, and provides cost estimation for CloudWatch analysis operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from utils.logging_config import log_cloudwatch_operation


class OperationType(Enum):
    """Types of operations categorized by cost impact."""
    FREE = "free"
    PAID = "paid"
    FORBIDDEN = "forbidden"


@dataclass
class CostPreferences:
    """User cost control preferences with validation."""
    allow_cost_explorer: bool = False
    allow_aws_config: bool = False
    allow_cloudtrail: bool = False
    allow_minimal_cost_metrics: bool = False
    
    def __post_init__(self):
        """Validate cost preferences after initialization."""
        if not isinstance(self.allow_cost_explorer, bool):
            raise ValueError("allow_cost_explorer must be a boolean")
        if not isinstance(self.allow_aws_config, bool):
            raise ValueError("allow_aws_config must be a boolean")
        if not isinstance(self.allow_cloudtrail, bool):
            raise ValueError("allow_cloudtrail must be a boolean")
        if not isinstance(self.allow_minimal_cost_metrics, bool):
            raise ValueError("allow_minimal_cost_metrics must be a boolean")


@dataclass
class OperationDefinition:
    """Definition of a CloudWatch operation with cost implications."""
    name: str
    operation_type: OperationType
    cost_flag: Optional[str] = None
    estimated_cost: float = 0.0
    functionality_weight: float = 0.0
    description: str = ""


@dataclass
class CostEstimate:
    """Cost estimation result for analysis scope."""
    total_estimated_cost: float
    cost_breakdown: Dict[str, float]
    enabled_operations: List[str]
    disabled_operations: List[str]
    functionality_coverage: Dict[str, float]


class CostController:
    """
    Manages cost control and validation for CloudWatch operations.
    
    This class provides:
    - Cost preference validation and sanitization
    - Functionality coverage calculation based on enabled features
    - Cost estimation for analysis scope and preferences
    - Runtime validation to prevent unauthorized paid operations
    """
    
    def __init__(self):
        """Initialize the CostController with operation definitions."""
        self.logger = logging.getLogger(__name__)
        self.default_preferences = CostPreferences()
        self.operation_definitions = self._initialize_operation_definitions()
        
    def _initialize_operation_definitions(self) -> Dict[str, OperationDefinition]:
        """Initialize CloudWatch operation definitions with cost implications."""
        operations = {
            # FREE Operations (Always Enabled - 60% of functionality)
            "list_metrics": OperationDefinition(
                name="list_metrics",
                operation_type=OperationType.FREE,
                functionality_weight=10.0,
                description="List available metrics metadata"
            ),
            "describe_alarms": OperationDefinition(
                name="describe_alarms",
                operation_type=OperationType.FREE,
                functionality_weight=15.0,
                description="Get alarm configurations"
            ),
            "list_dashboards": OperationDefinition(
                name="list_dashboards",
                operation_type=OperationType.FREE,
                functionality_weight=10.0,
                description="List dashboard names and metadata"
            ),
            "describe_log_groups": OperationDefinition(
                name="describe_log_groups",
                operation_type=OperationType.FREE,
                functionality_weight=15.0,
                description="Get log group configurations"
            ),
            "get_dashboard": OperationDefinition(
                name="get_dashboard",
                operation_type=OperationType.FREE,
                functionality_weight=5.0,
                description="Get dashboard configuration"
            ),
            "pricing_calculations": OperationDefinition(
                name="pricing_calculations",
                operation_type=OperationType.FREE,
                functionality_weight=5.0,
                description="Cost calculations and modeling"
            ),
            
            # PAID Operations (User Controlled)
            "cost_explorer_analysis": OperationDefinition(
                name="cost_explorer_analysis",
                operation_type=OperationType.PAID,
                cost_flag="allow_cost_explorer",
                estimated_cost=0.01,
                functionality_weight=30.0,
                description="Historical cost and usage analysis via Cost Explorer"
            ),
            "aws_config_compliance": OperationDefinition(
                name="aws_config_compliance",
                operation_type=OperationType.PAID,
                cost_flag="allow_aws_config",
                estimated_cost=0.003,
                functionality_weight=5.0,
                description="Compliance checking and configuration history"
            ),
            "cloudtrail_usage_patterns": OperationDefinition(
                name="cloudtrail_usage_patterns",
                operation_type=OperationType.PAID,
                cost_flag="allow_cloudtrail",
                estimated_cost=0.001,
                functionality_weight=1.0,
                description="Usage pattern analysis via CloudTrail"
            ),
            "minimal_cost_metrics": OperationDefinition(
                name="minimal_cost_metrics",
                operation_type=OperationType.PAID,
                cost_flag="allow_minimal_cost_metrics",
                estimated_cost=0.01,
                functionality_weight=4.0,
                description="Targeted CloudWatch metrics for log analysis"
            ),
            
            # FORBIDDEN Operations (Never Enabled - 0% usage)
            "logs_insights_queries": OperationDefinition(
                name="logs_insights_queries",
                operation_type=OperationType.FORBIDDEN,
                estimated_cost=1.0,  # High cost per GB scanned
                functionality_weight=0.0,
                description="CloudWatch Logs Insights queries (FORBIDDEN - high cost)"
            ),
            "extensive_metric_retrieval": OperationDefinition(
                name="extensive_metric_retrieval",
                operation_type=OperationType.FORBIDDEN,
                estimated_cost=0.5,
                functionality_weight=0.0,
                description="Extensive metric data retrieval (FORBIDDEN - high cost)"
            ),
        }
        
        return operations
    
    def validate_and_sanitize_preferences(self, preferences: Dict[str, Any]) -> CostPreferences:
        """
        Validate and sanitize cost control preferences.
        
        Args:
            preferences: Raw preference dictionary from user input
            
        Returns:
            CostPreferences: Validated and sanitized preferences
            
        Raises:
            ValueError: If preferences contain invalid values
        """
        log_cloudwatch_operation(self.logger, "validate_preferences", 
                               preferences=str(preferences))
        
        # Start with defaults
        sanitized = {
            "allow_cost_explorer": False,
            "allow_aws_config": False,
            "allow_cloudtrail": False,
            "allow_minimal_cost_metrics": False,
        }
        
        # Sanitize and validate each preference
        for key, default_value in sanitized.items():
            if key in preferences:
                value = preferences[key]
                if isinstance(value, bool):
                    sanitized[key] = value
                elif isinstance(value, str):
                    # Convert string representations to boolean
                    if value.lower() in ('true', '1', 'yes', 'on'):
                        sanitized[key] = True
                    elif value.lower() in ('false', '0', 'no', 'off'):
                        sanitized[key] = False
                    else:
                        self.logger.warning(f"Invalid boolean value for {key}: {value}, using default: {default_value}")
                        sanitized[key] = default_value
                elif isinstance(value, int):
                    sanitized[key] = bool(value)
                else:
                    self.logger.warning(f"Invalid type for {key}: {type(value)}, using default: {default_value}")
                    sanitized[key] = default_value
        
        # Log any unknown preferences
        unknown_keys = set(preferences.keys()) - set(sanitized.keys())
        if unknown_keys:
            log_cloudwatch_operation(self.logger, "unknown_preferences_ignored", 
                                   unknown_keys=list(unknown_keys))
        
        validated_preferences = CostPreferences(**sanitized)
        log_cloudwatch_operation(self.logger, "preferences_validated", 
                               validated_preferences=str(validated_preferences))
        
        return validated_preferences
    
    def get_functionality_coverage(self, preferences: CostPreferences) -> Dict[str, float]:
        """
        Calculate functionality coverage percentage based on enabled features.
        
        Args:
            preferences: Validated cost preferences
            
        Returns:
            Dict containing coverage percentages by category and overall
        """
        total_weight = sum(op.functionality_weight for op in self.operation_definitions.values())
        enabled_weight = 0.0
        
        coverage_by_category = {
            "free_operations": 0.0,
            "cost_explorer": 0.0,
            "aws_config": 0.0,
            "cloudtrail": 0.0,
            "minimal_cost_metrics": 0.0,
            "forbidden_operations": 0.0,
        }
        
        for operation in self.operation_definitions.values():
            if operation.operation_type == OperationType.FREE:
                # Free operations are always enabled
                enabled_weight += operation.functionality_weight
                coverage_by_category["free_operations"] += operation.functionality_weight
            elif operation.operation_type == OperationType.PAID:
                # Check if the corresponding cost flag is enabled
                if operation.cost_flag and getattr(preferences, operation.cost_flag, False):
                    enabled_weight += operation.functionality_weight
                    if operation.cost_flag == "allow_cost_explorer":
                        coverage_by_category["cost_explorer"] += operation.functionality_weight
                    elif operation.cost_flag == "allow_aws_config":
                        coverage_by_category["aws_config"] += operation.functionality_weight
                    elif operation.cost_flag == "allow_cloudtrail":
                        coverage_by_category["cloudtrail"] += operation.functionality_weight
                    elif operation.cost_flag == "allow_minimal_cost_metrics":
                        coverage_by_category["minimal_cost_metrics"] += operation.functionality_weight
            # Forbidden operations are never enabled (0% coverage)
        
        # Calculate percentages
        overall_coverage = (enabled_weight / total_weight * 100) if total_weight > 0 else 0.0
        
        # Convert weights to percentages
        for category in coverage_by_category:
            coverage_by_category[category] = (coverage_by_category[category] / total_weight * 100) if total_weight > 0 else 0.0
        
        coverage_result = {
            "overall_coverage": round(overall_coverage, 2),
            "by_category": {k: round(v, 2) for k, v in coverage_by_category.items()},
            "total_possible": 100.0,
            "free_tier_coverage": round(coverage_by_category["free_operations"], 2),
        }
        
        log_cloudwatch_operation(self.logger, "functionality_coverage_calculated", 
                               overall_coverage=coverage_result['overall_coverage'])
        return coverage_result
    
    def estimate_cost(self, analysis_scope: Dict[str, Any], preferences: CostPreferences) -> CostEstimate:
        """
        Estimate analysis cost based on scope and enabled features.
        
        Args:
            analysis_scope: Dictionary containing analysis parameters
            preferences: Validated cost preferences
            
        Returns:
            CostEstimate: Detailed cost estimation
        """
        log_cloudwatch_operation(self.logger, "cost_estimation_start", 
                               analysis_scope=str(analysis_scope))
        
        cost_breakdown = {}
        enabled_operations = []
        disabled_operations = []
        total_cost = 0.0
        
        # Get scope multipliers
        lookback_days = analysis_scope.get("lookback_days", 30)
        num_log_groups = len(analysis_scope.get("log_group_names", [])) or 10  # Default estimate
        num_alarms = len(analysis_scope.get("alarm_names", [])) or 50  # Default estimate
        num_dashboards = len(analysis_scope.get("dashboard_names", [])) or 5  # Default estimate
        
        # Calculate cost multiplier based on scope (additive approach)
        lookback_factor = max(1.0, lookback_days / 30)
        log_group_factor = max(1.0, num_log_groups / 10)
        alarm_factor = max(1.0, num_alarms / 50)
        scope_multiplier = lookback_factor * log_group_factor * alarm_factor
        

        
        for operation in self.operation_definitions.values():
            if operation.operation_type == OperationType.FREE:
                # Free operations
                enabled_operations.append(operation.name)
                cost_breakdown[operation.name] = 0.0
            elif operation.operation_type == OperationType.PAID:
                # Check if enabled
                if operation.cost_flag and getattr(preferences, operation.cost_flag, False):
                    enabled_operations.append(operation.name)
                    operation_cost = operation.estimated_cost * scope_multiplier
                    cost_breakdown[operation.name] = operation_cost
                    total_cost += operation_cost
                else:
                    disabled_operations.append(operation.name)
            else:  # FORBIDDEN
                disabled_operations.append(operation.name)
        
        functionality_coverage = self.get_functionality_coverage(preferences)
        
        estimate = CostEstimate(
            total_estimated_cost=round(total_cost, 6),
            cost_breakdown={k: round(v, 6) for k, v in cost_breakdown.items()},
            enabled_operations=enabled_operations,
            disabled_operations=disabled_operations,
            functionality_coverage=functionality_coverage,
        )
        
        log_cloudwatch_operation(self.logger, "cost_estimation_complete", 
                               total_estimated_cost=estimate.total_estimated_cost,
                               enabled_operations_count=len(enabled_operations))
        return estimate
    
    def validate_operation(self, operation_name: str, preferences: CostPreferences) -> Tuple[bool, str]:
        """
        Validate if an operation is allowed based on cost preferences.
        
        Args:
            operation_name: Name of the operation to validate
            preferences: Validated cost preferences
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if operation_name not in self.operation_definitions:
            return False, f"Unknown operation: {operation_name}"
        
        operation = self.operation_definitions[operation_name]
        
        if operation.operation_type == OperationType.FREE:
            return True, "Free operation - always allowed"
        elif operation.operation_type == OperationType.FORBIDDEN:
            return False, f"Forbidden operation: {operation.description}"
        elif operation.operation_type == OperationType.PAID:
            if operation.cost_flag and getattr(preferences, operation.cost_flag, False):
                return True, f"Paid operation allowed by {operation.cost_flag}"
            else:
                return False, f"Paid operation disabled - requires {operation.cost_flag}=True"
        
        return False, "Unknown operation type"
    
    def get_allowed_operations(self, preferences: CostPreferences) -> Dict[str, List[str]]:
        """
        Get lists of allowed and disallowed operations based on preferences.
        
        Args:
            preferences: Validated cost preferences
            
        Returns:
            Dictionary with 'allowed' and 'disallowed' operation lists
        """
        allowed = []
        disallowed = []
        
        for operation_name, operation in self.operation_definitions.items():
            is_allowed, _ = self.validate_operation(operation_name, preferences)
            if is_allowed:
                allowed.append(operation_name)
            else:
                disallowed.append(operation_name)
        
        return {
            "allowed": allowed,
            "disallowed": disallowed,
        }
    
    def log_cost_decision(self, operation_name: str, preferences: CostPreferences, 
                         is_allowed: bool, reason: str) -> None:
        """
        Log cost control decisions for transparency and auditing.
        
        Args:
            operation_name: Name of the operation
            preferences: Cost preferences used
            is_allowed: Whether operation was allowed
            reason: Reason for the decision
        """
        log_cloudwatch_operation(self.logger, "cost_decision", 
                               operation_name=operation_name,
                               is_allowed=is_allowed,
                               reason=reason,
                               preferences=str(preferences))
    
    def get_cost_summary(self, preferences: CostPreferences) -> Dict[str, Any]:
        """
        Get a comprehensive cost control summary.
        
        Args:
            preferences: Validated cost preferences
            
        Returns:
            Dictionary containing cost control summary
        """
        functionality_coverage = self.get_functionality_coverage(preferences)
        allowed_operations = self.get_allowed_operations(preferences)
        
        # Calculate potential cost range
        min_cost = 0.0  # Free operations only
        max_cost = sum(
            op.estimated_cost for op in self.operation_definitions.values()
            if op.operation_type == OperationType.PAID
        )
        
        enabled_paid_ops = [
            op for op in self.operation_definitions.values()
            if (op.operation_type == OperationType.PAID and 
                op.cost_flag and getattr(preferences, op.cost_flag, False))
        ]
        current_max_cost = sum(op.estimated_cost for op in enabled_paid_ops)
        
        return {
            "preferences": preferences,
            "functionality_coverage": functionality_coverage,
            "allowed_operations": allowed_operations,
            "cost_range": {
                "minimum_cost": min_cost,
                "maximum_possible_cost": max_cost,
                "current_maximum_cost": current_max_cost,
            },
            "enabled_paid_features": len(enabled_paid_ops),
            "total_paid_features": len([
                op for op in self.operation_definitions.values()
                if op.operation_type == OperationType.PAID
            ]),
        }
    
    def get_execution_path_routing(self, preferences: CostPreferences) -> Dict[str, Any]:
        """
        Get execution path routing configuration based on consent preferences.
        
        This method determines which execution paths should be used for each
        analysis type based on user consent for paid operations.
        
        Args:
            preferences: Validated cost preferences
            
        Returns:
            Dictionary containing routing configuration for each analysis type
        """
        routing_config = {
            "general_spend_analysis": {
                "primary_path": "cost_explorer" if preferences.allow_cost_explorer else "free_apis",
                "fallback_path": "free_apis",
                "data_sources": []
            },
            "metrics_optimization": {
                "primary_path": "cost_explorer" if preferences.allow_cost_explorer else "free_apis",
                "fallback_path": "free_apis", 
                "data_sources": []
            },
            "logs_optimization": {
                "primary_path": "cost_explorer" if preferences.allow_cost_explorer else "free_apis",
                "fallback_path": "free_apis",
                "data_sources": []
            },
            "alarms_and_dashboards": {
                "primary_path": "free_apis",  # Always starts with free APIs
                "fallback_path": "free_apis",
                "data_sources": []
            }
        }
        
        # Configure data sources based on consent
        for analysis_type, config in routing_config.items():
            # Always include free APIs
            config["data_sources"].append("cloudwatch_config_apis")
            config["data_sources"].append("pricing_calculations")
            
            # Add paid data sources based on consent
            if preferences.allow_cost_explorer:
                config["data_sources"].append("cost_explorer")
            
            if preferences.allow_aws_config:
                config["data_sources"].append("aws_config")
            
            if preferences.allow_cloudtrail:
                config["data_sources"].append("cloudtrail")
            
            if preferences.allow_minimal_cost_metrics:
                config["data_sources"].append("minimal_cost_metrics")
        
        log_cloudwatch_operation(self.logger, "execution_path_routing_configured",
                               routing_config=str(routing_config))
        
        return routing_config
    
    def create_cost_tracking_context(self, preferences: CostPreferences) -> Dict[str, Any]:
        """
        Create a cost tracking context for monitoring paid operations during analysis.
        
        Args:
            preferences: Validated cost preferences
            
        Returns:
            Dictionary containing cost tracking context
        """
        context = {
            "session_start_time": datetime.now().isoformat(),
            "preferences": preferences.__dict__,
            "cost_incurring_operations": [],
            "free_operations": [],
            "blocked_operations": [],
            "total_estimated_cost": 0.0,
            "actual_cost_incurred": 0.0,
            "operation_count": 0,
            "routing_decisions": []
        }
        
        log_cloudwatch_operation(self.logger, "cost_tracking_context_created",
                               context_id=id(context))
        
        return context
    
    def track_operation_execution(self, context: Dict[str, Any], operation_name: str, 
                                 operation_type: str, cost_incurred: float = 0.0,
                                 routing_decision: str = None) -> None:
        """
        Track execution of an operation in the cost tracking context.
        
        Args:
            context: Cost tracking context
            operation_name: Name of the executed operation
            operation_type: Type of operation (free, paid, forbidden)
            cost_incurred: Actual cost incurred (if known)
            routing_decision: Description of routing decision made
        """
        context["operation_count"] += 1
        
        if operation_type == "free":
            context["free_operations"].append({
                "operation": operation_name,
                "timestamp": datetime.now().isoformat(),
                "routing_decision": routing_decision
            })
        elif operation_type == "paid":
            context["cost_incurring_operations"].append({
                "operation": operation_name,
                "timestamp": datetime.now().isoformat(),
                "estimated_cost": self.operation_definitions.get(operation_name, {}).estimated_cost,
                "actual_cost": cost_incurred,
                "routing_decision": routing_decision
            })
            context["actual_cost_incurred"] += cost_incurred
        elif operation_type == "blocked":
            context["blocked_operations"].append({
                "operation": operation_name,
                "timestamp": datetime.now().isoformat(),
                "reason": "Operation not consented to",
                "routing_decision": routing_decision
            })
        
        if routing_decision:
            context["routing_decisions"].append({
                "operation": operation_name,
                "decision": routing_decision,
                "timestamp": datetime.now().isoformat()
            })
        
        log_cloudwatch_operation(self.logger, "operation_tracked",
                               operation_name=operation_name,
                               operation_type=operation_type,
                               cost_incurred=cost_incurred)
    
    def generate_cost_transparency_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive cost transparency report from tracking context.
        
        Args:
            context: Cost tracking context
            
        Returns:
            Dictionary containing detailed cost transparency report
        """
        session_duration = (datetime.now() - datetime.fromisoformat(context["session_start_time"])).total_seconds()
        
        report = {
            "session_summary": {
                "session_duration_seconds": session_duration,
                "total_operations": context["operation_count"],
                "free_operations_count": len(context["free_operations"]),
                "paid_operations_count": len(context["cost_incurring_operations"]),
                "blocked_operations_count": len(context["blocked_operations"])
            },
            "cost_summary": {
                "total_estimated_cost": sum(
                    op.get("estimated_cost", 0) for op in context["cost_incurring_operations"]
                ),
                "total_actual_cost": context["actual_cost_incurred"],
                "cost_by_operation": {
                    op["operation"]: op.get("actual_cost", op.get("estimated_cost", 0))
                    for op in context["cost_incurring_operations"]
                }
            },
            "execution_paths": {
                "consent_based_routing": True,
                "fallback_usage": any(
                    "fallback" in decision.get("decision", "").lower()
                    for decision in context["routing_decisions"]
                )
            },
            "transparency_details": {
                "routing_decisions": context["routing_decisions"],
                "free_operations": context["free_operations"],
                "paid_operations": context["cost_incurring_operations"],
                "blocked_operations": context["blocked_operations"],
                "user_preferences": context["preferences"]
            },
            "recommendations": self._generate_cost_optimization_recommendations(context)
        }
        
        log_cloudwatch_operation(self.logger, "cost_transparency_report_generated",
                               total_cost=report["cost_summary"]["total_actual_cost"],
                               operations_count=report["session_summary"]["total_operations"])
        
        return report
    
    def _generate_cost_optimization_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations based on tracking context."""
        recommendations = []
        
        # Recommend enabling cost explorer if not enabled and many operations were blocked
        if not context["preferences"]["allow_cost_explorer"] and len(context["blocked_operations"]) > 0:
            recommendations.append({
                "type": "enable_feature",
                "priority": "medium",
                "title": "Consider enabling Cost Explorer for comprehensive analysis",
                "description": f"You had {len(context['blocked_operations'])} operations that could provide additional insights if Cost Explorer was enabled.",
                "estimated_additional_cost": 0.01,
                "functionality_improvement": "30% additional coverage"
            })
        
        # Recommend minimal cost metrics if logs analysis was limited
        logs_blocked = any(
            "logs" in op["operation"].lower() for op in context["blocked_operations"]
        )
        if not context["preferences"]["allow_minimal_cost_metrics"] and logs_blocked:
            recommendations.append({
                "type": "enable_feature", 
                "priority": "low",
                "title": "Enable minimal cost metrics for detailed log analysis",
                "description": "Minimal cost metrics can provide detailed log ingestion patterns for better optimization.",
                "estimated_additional_cost": 0.01,
                "functionality_improvement": "4% additional coverage"
            })
        
        # Recommend cost awareness if many paid operations were used
        if len(context["cost_incurring_operations"]) > 5:
            recommendations.append({
                "type": "cost_awareness",
                "priority": "high", 
                "title": "Monitor your CloudWatch analysis costs",
                "description": f"You executed {len(context['cost_incurring_operations'])} paid operations. Consider reviewing cost preferences for future analyses.",
                "estimated_cost_impact": context["actual_cost_incurred"]
            })
        
        return recommendations