"""
Comprehensive S3 Optimization Tool for CFM Tips MCP Server

Unified tool that executes all 8 S3 optimization functionalities in parallel with
intelligent orchestration, priority-based execution, and comprehensive reporting.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from .s3_optimization_orchestrator import S3OptimizationOrchestrator
from .s3_analysis_engine import S3AnalysisEngine
from utils.service_orchestrator import ServiceOrchestrator
from utils.session_manager import get_session_manager
from utils.performance_monitor import get_performance_monitor
from utils.memory_manager import get_memory_manager
from utils.progressive_timeout import get_timeout_handler

logger = logging.getLogger(__name__)


@dataclass
class S3OptimizationScope:
    """Configuration for S3 optimization analysis scope."""
    
    # Analysis selection
    enabled_analyses: Optional[Set[str]] = None  # None means all analyses
    disabled_analyses: Optional[Set[str]] = None
    
    # Resource filtering
    bucket_names: Optional[List[str]] = None
    bucket_patterns: Optional[List[str]] = None  # Regex patterns for bucket names
    regions: Optional[List[str]] = None
    
    # Time range configuration
    lookback_days: int = 30
    max_lookback_days: int = 90  # Safety limit
    
    # Cost filtering
    min_monthly_cost: Optional[float] = None  # Only analyze buckets above this cost
    focus_high_cost: bool = True  # Prioritize high-cost buckets
    
    # Analysis depth
    include_detailed_breakdown: bool = True
    include_cross_analysis: bool = True
    include_trends: bool = False  # Disabled by default for performance
    
    # Performance configuration
    max_parallel_analyses: int = 6  # Limit concurrent analyses
    timeout_seconds: int = 120  # Total timeout for comprehensive analysis
    individual_timeout: int = 45  # Timeout per individual analysis
    
    # Reporting configuration
    include_executive_summary: bool = True
    min_savings_threshold: float = 10.0  # Minimum savings to include in recommendations
    max_recommendations_per_type: int = 10


class S3ComprehensiveOptimizationTool:
    """
    Unified S3 optimization tool that executes all functionalities with intelligent orchestration.
    
    This tool provides:
    - Parallel execution of all 8 S3 optimization analyses
    - Priority-based task scheduling based on cost impact and execution time
    - Comprehensive result aggregation with cross-analysis insights
    - Executive summary generation with top recommendations
    - Configurable analysis scope and resource filtering
    - Performance optimization with intelligent caching and memory management
    """
    
    def __init__(self, region: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize comprehensive S3 optimization tool.
        
        Args:
            region: AWS region for S3 operations
            session_id: Optional session ID for data persistence
        """
        self.region = region
        self.session_id = session_id
        
        # Initialize core components
        self.orchestrator = S3OptimizationOrchestrator(region=region, session_id=session_id)
        self.session_id = self.orchestrator.session_id  # Get actual session ID
        
        # Performance optimization components
        self.performance_monitor = get_performance_monitor()
        self.memory_manager = get_memory_manager()
        self.timeout_handler = get_timeout_handler()
        
        # Analysis metadata
        self.available_analyses = self._get_available_analyses()
        self.execution_history = []
        
        logger.info(f"S3ComprehensiveOptimizationTool initialized for region: {region or 'default'}, session: {self.session_id}")
    
    def _get_available_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Get available S3 analyses with metadata."""
        return {
            "general_spend": {
                "name": "General Spend Analysis",
                "description": "Comprehensive S3 spending analysis across storage, transfer, and API costs",
                "priority": 5,
                "cost_impact": "high",
                "execution_time_estimate": 30,
                "dependencies": [],
                "category": "cost_analysis"
            },
            "storage_class": {
                "name": "Storage Class Optimization",
                "description": "Storage class appropriateness and optimization recommendations",
                "priority": 4,
                "cost_impact": "high",
                "execution_time_estimate": 45,
                "dependencies": ["general_spend"],
                "category": "optimization"
            },
            "multipart_cleanup": {
                "name": "Multipart Upload Cleanup",
                "description": "Identify and clean up incomplete multipart uploads",
                "priority": 4,
                "cost_impact": "medium",
                "execution_time_estimate": 20,
                "dependencies": [],
                "category": "cleanup"
            },
            "archive_optimization": {
                "name": "Archive Optimization",
                "description": "Long-term data archiving optimization for cost reduction",
                "priority": 3,
                "cost_impact": "medium",
                "execution_time_estimate": 35,
                "dependencies": ["storage_class"],
                "category": "optimization"
            },
            "api_cost": {
                "name": "API Cost Minimization",
                "description": "API request pattern optimization and cost reduction",
                "priority": 2,
                "cost_impact": "low",
                "execution_time_estimate": 25,
                "dependencies": [],
                "category": "optimization"
            },
            "governance": {
                "name": "Governance & Compliance",
                "description": "S3 governance policies and compliance validation",
                "priority": 1,
                "cost_impact": "low",
                "execution_time_estimate": 15,
                "dependencies": [],
                "category": "governance"
            }
        }
    
    async def execute_comprehensive_optimization(self, 
                                               scope: Optional[S3OptimizationScope] = None,
                                               **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive S3 optimization with all analyses in parallel.
        
        Args:
            scope: Analysis scope configuration
            **kwargs: Additional parameters for backward compatibility
            
        Returns:
            Dictionary containing comprehensive optimization results
        """
        start_time = time.time()
        
        # Initialize scope with defaults
        if scope is None:
            scope = S3OptimizationScope()
        
        # Override scope with kwargs for backward compatibility
        if kwargs:
            scope = self._merge_scope_with_kwargs(scope, kwargs)
        
        # Start comprehensive monitoring
        monitoring_session = self.performance_monitor.start_analysis_monitoring(
            "comprehensive_optimization",
            f"comprehensive_{int(start_time)}"
        )
        
        # Start memory tracking
        memory_tracker = self.memory_manager.start_memory_tracking("comprehensive_optimization")
        
        logger.info("Starting comprehensive S3 optimization with intelligent orchestration")
        
        try:
            # Validate and prepare scope
            validated_scope = self._validate_and_prepare_scope(scope)
            
            # Determine which analyses to run
            analyses_to_run = self._determine_analyses_to_run(validated_scope)
            
            logger.info(f"Executing {len(analyses_to_run)} analyses: {list(analyses_to_run.keys())}")
            
            # Execute analyses with intelligent orchestration
            execution_results = await self._execute_analyses_with_orchestration(
                analyses_to_run, validated_scope
            )
            
            # Generate comprehensive report
            comprehensive_report = await self._generate_comprehensive_report(
                execution_results, validated_scope, start_time
            )
            
            # Generate executive summary
            if validated_scope.include_executive_summary:
                executive_summary = self._generate_executive_summary(
                    comprehensive_report, validated_scope
                )
                comprehensive_report["executive_summary"] = executive_summary
            
            # Store results if requested
            if kwargs.get('store_results', True):
                self._store_comprehensive_results(comprehensive_report)
            
            execution_time = time.time() - start_time
            comprehensive_report["total_execution_time"] = execution_time
            comprehensive_report["session_id"] = self.session_id
            
            # Record execution history
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "analyses_count": len(analyses_to_run),
                "status": "success",
                "scope": validated_scope.__dict__
            })
            
            # End monitoring
            self.performance_monitor.end_analysis_monitoring(monitoring_session, success=True)
            memory_stats = self.memory_manager.stop_memory_tracking("comprehensive_optimization")
            
            if memory_stats:
                comprehensive_report["memory_usage"] = memory_stats
            
            logger.info(f"Completed comprehensive S3 optimization in {execution_time:.2f}s")
            
            return comprehensive_report
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in comprehensive S3 optimization: {str(e)}")
            
            # End monitoring with error
            self.performance_monitor.end_analysis_monitoring(
                monitoring_session, success=False, error_message=str(e)
            )
            self.memory_manager.stop_memory_tracking("comprehensive_optimization")
            
            return {
                "status": "error",
                "message": f"Comprehensive optimization failed: {str(e)}",
                "execution_time": execution_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _merge_scope_with_kwargs(self, scope: S3OptimizationScope, kwargs: Dict[str, Any]) -> S3OptimizationScope:
        """Merge scope configuration with kwargs for backward compatibility."""
        # Create a copy of the scope
        merged_scope = S3OptimizationScope(
            enabled_analyses=scope.enabled_analyses,
            disabled_analyses=scope.disabled_analyses,
            bucket_names=kwargs.get('bucket_names', scope.bucket_names),
            bucket_patterns=scope.bucket_patterns,
            regions=scope.regions,
            lookback_days=kwargs.get('lookback_days', scope.lookback_days),
            max_lookback_days=scope.max_lookback_days,
            min_monthly_cost=kwargs.get('min_monthly_cost', scope.min_monthly_cost),
            focus_high_cost=kwargs.get('focus_high_cost', scope.focus_high_cost),
            include_detailed_breakdown=kwargs.get('include_detailed_breakdown', scope.include_detailed_breakdown),
            include_cross_analysis=kwargs.get('include_cross_analysis', scope.include_cross_analysis),
            include_trends=kwargs.get('include_trends', scope.include_trends),
            max_parallel_analyses=kwargs.get('max_parallel_analyses', scope.max_parallel_analyses),
            timeout_seconds=kwargs.get('timeout_seconds', scope.timeout_seconds),
            individual_timeout=kwargs.get('individual_timeout', scope.individual_timeout),
            include_executive_summary=kwargs.get('include_executive_summary', scope.include_executive_summary),
            min_savings_threshold=kwargs.get('min_savings_threshold', scope.min_savings_threshold),
            max_recommendations_per_type=kwargs.get('max_recommendations_per_type', scope.max_recommendations_per_type)
        )
        
        return merged_scope
    
    def _validate_and_prepare_scope(self, scope: S3OptimizationScope) -> S3OptimizationScope:
        """Validate and prepare the analysis scope."""
        # Validate lookback days
        if scope.lookback_days > scope.max_lookback_days:
            logger.warning(f"Lookback days {scope.lookback_days} exceeds maximum {scope.max_lookback_days}, using maximum")
            scope.lookback_days = scope.max_lookback_days
        
        # Validate timeout settings
        if scope.timeout_seconds < scope.individual_timeout:
            logger.warning("Total timeout is less than individual timeout, adjusting")
            scope.timeout_seconds = scope.individual_timeout * 3
        
        # Validate parallel execution limits
        if scope.max_parallel_analyses > 8:  # We have 6 analyses max
            scope.max_parallel_analyses = 6
        
        # Set region if not specified
        if not scope.regions and self.region:
            scope.regions = [self.region]
        
        return scope
    
    def _determine_analyses_to_run(self, scope: S3OptimizationScope) -> Dict[str, Dict[str, Any]]:
        """Determine which analyses to run based on scope configuration."""
        analyses_to_run = {}
        
        # Start with all available analyses
        available = self.available_analyses.copy()
        
        # Apply enabled_analyses filter
        if scope.enabled_analyses:
            available = {k: v for k, v in available.items() if k in scope.enabled_analyses}
        
        # Apply disabled_analyses filter
        if scope.disabled_analyses:
            available = {k: v for k, v in available.items() if k not in scope.disabled_analyses}
        
        # Apply cost-based filtering
        if scope.min_monthly_cost is not None:
            # Prioritize cost-related analyses when cost filtering is enabled
            cost_analyses = ["general_spend", "storage_class", "archive_optimization"]
            for analysis in cost_analyses:
                if analysis in available:
                    analyses_to_run[analysis] = available[analysis]
        else:
            analyses_to_run = available
        
        # Sort by priority (highest first)
        analyses_to_run = dict(sorted(
            analyses_to_run.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        ))
        
        logger.info(f"Determined analyses to run: {list(analyses_to_run.keys())}")
        
        return analyses_to_run
    
    async def _execute_analyses_with_orchestration(self, 
                                                 analyses_to_run: Dict[str, Dict[str, Any]], 
                                                 scope: S3OptimizationScope) -> Dict[str, Any]:
        """Execute analyses with intelligent orchestration and priority-based scheduling."""
        
        # Prepare analysis parameters
        analysis_params = {
            'region': self.region,
            'session_id': self.session_id,
            'lookback_days': scope.lookback_days,
            'include_cost_analysis': scope.include_detailed_breakdown,
            'bucket_names': scope.bucket_names,
            'timeout_seconds': scope.individual_timeout,
            'store_results': True,
            'include_cross_analysis': scope.include_cross_analysis
        }
        
        # Create prioritized execution plan
        execution_plan = self._create_execution_plan(analyses_to_run, scope)
        
        # Execute analyses using the orchestrator
        logger.info(f"Executing {len(execution_plan)} analyses with intelligent orchestration")
        
        # Use the orchestrator's comprehensive analysis with our specific parameters
        orchestrator_result = await self.orchestrator.execute_comprehensive_analysis(**analysis_params)
        
        # Filter results to only include requested analyses
        if orchestrator_result.get("status") == "success":
            # Filter execution results to only include our requested analyses
            execution_summary = orchestrator_result.get("execution_summary", {})
            filtered_results = {}
            
            if "results" in execution_summary:
                for task_id, result in execution_summary["results"].items():
                    # Extract analysis type from result
                    if hasattr(result, 'data') and isinstance(result.data, dict):
                        analysis_type = result.data.get("analysis_type")
                    elif isinstance(result, dict):
                        analysis_type = result.get("analysis_type")
                    else:
                        continue
                    
                    if analysis_type in analyses_to_run:
                        filtered_results[task_id] = result
            
            # Update the orchestrator result with filtered results
            if "execution_summary" in orchestrator_result:
                orchestrator_result["execution_summary"]["results"] = filtered_results
                orchestrator_result["execution_summary"]["total_tasks"] = len(filtered_results)
                orchestrator_result["execution_summary"]["successful"] = sum(
                    1 for r in filtered_results.values() 
                    if (hasattr(r, 'status') and r.status == 'success') or 
                       (isinstance(r, dict) and r.get('status') == 'success')
                )
        
        return orchestrator_result
    
    def _create_execution_plan(self, 
                             analyses_to_run: Dict[str, Dict[str, Any]], 
                             scope: S3OptimizationScope) -> List[Dict[str, Any]]:
        """Create intelligent execution plan with dependency resolution and priority scheduling."""
        
        execution_plan = []
        
        # Resolve dependencies and create execution order
        resolved_order = self._resolve_dependencies(analyses_to_run)
        
        # Create execution plan with batching for parallel execution
        batch_size = min(scope.max_parallel_analyses, len(resolved_order))
        
        for i, analysis_type in enumerate(resolved_order):
            analysis_info = analyses_to_run[analysis_type]
            
            # Calculate dynamic timeout based on priority and estimated execution time
            base_timeout = analysis_info.get("execution_time_estimate", 30)
            priority_multiplier = 1.0 + (analysis_info.get("priority", 1) * 0.2)
            dynamic_timeout = min(int(base_timeout * priority_multiplier) + 15, scope.individual_timeout)
            
            execution_item = {
                "analysis_type": analysis_type,
                "analysis_info": analysis_info,
                "execution_order": i + 1,
                "batch": i // batch_size,
                "dynamic_timeout": dynamic_timeout,
                "priority_score": analysis_info.get("priority", 1),
                "estimated_execution_time": analysis_info.get("execution_time_estimate", 30)
            }
            
            execution_plan.append(execution_item)
        
        logger.info(f"Created execution plan with {len(execution_plan)} analyses in {max(item['batch'] for item in execution_plan) + 1} batches")
        
        return execution_plan
    
    def _resolve_dependencies(self, analyses_to_run: Dict[str, Dict[str, Any]]) -> List[str]:
        """Resolve analysis dependencies and create execution order."""
        
        resolved = []
        remaining = analyses_to_run.copy()
        
        # Simple dependency resolution (topological sort)
        while remaining:
            # Find analyses with no unresolved dependencies
            ready = []
            for analysis_type, analysis_info in remaining.items():
                dependencies = analysis_info.get("dependencies", [])
                if all(dep in resolved or dep not in analyses_to_run for dep in dependencies):
                    ready.append(analysis_type)
            
            if not ready:
                # No analyses are ready, break circular dependencies by priority
                ready = [max(remaining.keys(), key=lambda x: remaining[x].get("priority", 0))]
                logger.warning(f"Breaking potential circular dependency by prioritizing: {ready[0]}")
            
            # Sort ready analyses by priority (highest first)
            ready.sort(key=lambda x: remaining[x].get("priority", 0), reverse=True)
            
            # Add to resolved list and remove from remaining
            for analysis_type in ready:
                resolved.append(analysis_type)
                del remaining[analysis_type]
        
        logger.info(f"Resolved execution order: {resolved}")
        
        return resolved
    
    async def _generate_comprehensive_report(self, 
                                           execution_results: Dict[str, Any], 
                                           scope: S3OptimizationScope,
                                           start_time: float) -> Dict[str, Any]:
        """Generate comprehensive optimization report with aggregated insights."""
        
        report = {
            "status": "success",
            "analysis_type": "comprehensive_s3_optimization",
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - start_time,
            "session_id": self.session_id,
            "scope_configuration": scope.__dict__,
            "analysis_summary": {},
            "aggregated_insights": {},
            "optimization_opportunities": [],
            "cost_savings_summary": {
                "total_potential_savings": 0.0,
                "savings_by_category": {},
                "high_impact_recommendations": [],
                "quick_wins": []
            },
            "performance_metrics": {},
            "cross_analysis_insights": []
        }
        
        # Extract results from orchestrator response
        if execution_results.get("status") == "success":
            # Use the orchestrator's aggregated results
            aggregated_results = execution_results.get("aggregated_results", {})
            execution_summary = execution_results.get("execution_summary", {})
            
            # Copy aggregated insights
            report["aggregated_insights"] = aggregated_results
            report["analysis_summary"] = aggregated_results.get("analysis_summary", {})
            
            # Extract cost savings information
            total_savings = aggregated_results.get("total_potential_savings", 0)
            report["cost_savings_summary"]["total_potential_savings"] = total_savings
            
            # Categorize recommendations
            recommendations_by_priority = aggregated_results.get("recommendations_by_priority", {})
            
            # High impact recommendations (high priority with significant savings)
            high_impact = []
            for rec in recommendations_by_priority.get("high", []):
                if rec.get("potential_savings", 0) >= scope.min_savings_threshold:
                    high_impact.append(rec)
            
            report["cost_savings_summary"]["high_impact_recommendations"] = high_impact[:scope.max_recommendations_per_type]
            
            # Quick wins (low implementation effort with good savings)
            quick_wins = []
            for priority in ["high", "medium"]:
                for rec in recommendations_by_priority.get(priority, []):
                    if (rec.get("implementation_effort", "medium") in ["low", "very_low"] and 
                        rec.get("potential_savings", 0) >= scope.min_savings_threshold / 2):
                        quick_wins.append(rec)
            
            report["cost_savings_summary"]["quick_wins"] = quick_wins[:scope.max_recommendations_per_type]
            
            # Aggregate optimization opportunities
            all_recommendations = []
            for priority_recs in recommendations_by_priority.values():
                all_recommendations.extend(priority_recs)
            
            # Sort by potential savings (highest first)
            all_recommendations.sort(
                key=lambda x: x.get("potential_savings", 0), 
                reverse=True
            )
            
            report["optimization_opportunities"] = all_recommendations[:scope.max_recommendations_per_type * 3]
            
            # Add cross-analysis insights if available
            cross_analysis_data = execution_results.get("cross_analysis_data", {})
            if cross_analysis_data:
                report["cross_analysis_insights"] = self._generate_cross_analysis_insights(cross_analysis_data)
            
            # Performance metrics
            report["performance_metrics"] = {
                "total_analyses": execution_summary.get("total_tasks", 0),
                "successful_analyses": execution_summary.get("successful", 0),
                "failed_analyses": execution_summary.get("failed", 0),
                "average_execution_time": aggregated_results.get("execution_performance", {}).get("average_execution_time", 0),
                "parallel_execution": True,
                "intelligent_orchestration": True
            }
        
        else:
            # Handle error case
            report["status"] = execution_results.get("status", "error")
            report["error_message"] = execution_results.get("message", "Unknown error")
        
        return report
    
    def _generate_cross_analysis_insights(self, cross_analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from cross-analysis data."""
        insights = []
        
        # Example cross-analysis insights
        if "bucket_cost_efficiency" in cross_analysis_data:
            efficiency_data = cross_analysis_data["bucket_cost_efficiency"]
            if efficiency_data:
                insights.append({
                    "type": "cost_efficiency",
                    "title": "Bucket Cost Efficiency Analysis",
                    "description": "Analysis of cost efficiency across S3 buckets",
                    "data": efficiency_data,
                    "recommendations": [
                        "Focus optimization efforts on high-cost, low-efficiency buckets",
                        "Consider consolidating small, inefficient buckets",
                        "Review storage class distribution for top cost buckets"
                    ]
                })
        
        if "storage_class_opportunities" in cross_analysis_data:
            storage_data = cross_analysis_data["storage_class_opportunities"]
            if storage_data:
                insights.append({
                    "type": "storage_optimization",
                    "title": "Cross-Bucket Storage Class Opportunities",
                    "description": "Storage class optimization opportunities across multiple buckets",
                    "data": storage_data,
                    "recommendations": [
                        "Implement lifecycle policies for identified transition opportunities",
                        "Review access patterns for infrequently accessed data",
                        "Consider archive tiers for long-term retention data"
                    ]
                })
        
        return insights
    
    def _generate_executive_summary(self, 
                                  comprehensive_report: Dict[str, Any], 
                                  scope: S3OptimizationScope) -> Dict[str, Any]:
        """Generate executive summary with top recommendations and key insights."""
        
        cost_savings = comprehensive_report.get("cost_savings_summary", {})
        analysis_summary = comprehensive_report.get("analysis_summary", {})
        performance_metrics = comprehensive_report.get("performance_metrics", {})
        
        # Calculate key metrics
        total_savings = cost_savings.get("total_potential_savings", 0)
        successful_analyses = performance_metrics.get("successful_analyses", 0)
        total_analyses = performance_metrics.get("total_analyses", 0)
        
        # Determine overall health score
        health_score = self._calculate_s3_health_score(comprehensive_report)
        
        # Generate key findings
        key_findings = self._generate_key_findings(comprehensive_report)
        
        # Top recommendations (limited to top 5)
        top_recommendations = cost_savings.get("high_impact_recommendations", [])[:5]
        
        # Quick wins (limited to top 3)
        quick_wins = cost_savings.get("quick_wins", [])[:3]
        
        executive_summary = {
            "overview": {
                "total_potential_savings": total_savings,
                "total_potential_savings_formatted": f"${total_savings:.2f}",
                "analyses_completed": f"{successful_analyses}/{total_analyses}",
                "s3_health_score": health_score,
                "analysis_scope": {
                    "lookback_days": scope.lookback_days,
                    "buckets_analyzed": len(scope.bucket_names) if scope.bucket_names else "All buckets",
                    "regions": scope.regions or ["All regions"]
                }
            },
            "key_findings": key_findings,
            "top_recommendations": top_recommendations,
            "quick_wins": quick_wins,
            "next_steps": self._generate_next_steps(comprehensive_report, scope),
            "risk_assessment": self._generate_risk_assessment(comprehensive_report),
            "generated_at": datetime.now().isoformat(),
            "executive_summary_version": "1.0"
        }
        
        return executive_summary
    
    def _calculate_s3_health_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall S3 health score based on optimization opportunities."""
        
        # Scoring factors
        total_savings = report.get("cost_savings_summary", {}).get("total_potential_savings", 0)
        high_impact_count = len(report.get("cost_savings_summary", {}).get("high_impact_recommendations", []))
        successful_analyses = report.get("performance_metrics", {}).get("successful_analyses", 0)
        total_analyses = report.get("performance_metrics", {}).get("total_analyses", 1)
        
        # Calculate component scores (0-100)
        cost_efficiency_score = max(0, 100 - (total_savings / 100))  # Lower savings = higher score
        governance_score = 85 if successful_analyses == total_analyses else 70  # Based on analysis success
        optimization_score = max(0, 100 - (high_impact_count * 10))  # Fewer high-impact issues = higher score
        
        # Overall score (weighted average)
        overall_score = (
            cost_efficiency_score * 0.4 +
            governance_score * 0.3 +
            optimization_score * 0.3
        )
        
        # Determine health level
        if overall_score >= 90:
            health_level = "Excellent"
            health_color = "green"
        elif overall_score >= 75:
            health_level = "Good"
            health_color = "yellow"
        elif overall_score >= 60:
            health_level = "Fair"
            health_color = "orange"
        else:
            health_level = "Needs Attention"
            health_color = "red"
        
        return {
            "overall_score": round(overall_score, 1),
            "health_level": health_level,
            "health_color": health_color,
            "component_scores": {
                "cost_efficiency": round(cost_efficiency_score, 1),
                "governance": round(governance_score, 1),
                "optimization": round(optimization_score, 1)
            }
        }
    
    def _generate_key_findings(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate key findings from the comprehensive analysis."""
        
        findings = []
        
        # Cost-related findings
        total_savings = report.get("cost_savings_summary", {}).get("total_potential_savings", 0)
        if total_savings > 100:
            findings.append({
                "type": "cost_opportunity",
                "severity": "high",
                "title": f"Significant Cost Optimization Opportunity",
                "description": f"Identified ${total_savings:.2f} in potential monthly savings",
                "impact": "high"
            })
        elif total_savings > 50:
            findings.append({
                "type": "cost_opportunity",
                "severity": "medium",
                "title": f"Moderate Cost Optimization Opportunity",
                "description": f"Identified ${total_savings:.2f} in potential monthly savings",
                "impact": "medium"
            })
        
        # Analysis completion findings
        performance_metrics = report.get("performance_metrics", {})
        success_rate = (performance_metrics.get("successful_analyses", 0) / 
                       max(performance_metrics.get("total_analyses", 1), 1)) * 100
        
        if success_rate < 100:
            findings.append({
                "type": "analysis_incomplete",
                "severity": "medium",
                "title": "Some Analyses Incomplete",
                "description": f"Analysis completion rate: {success_rate:.1f}%",
                "impact": "medium"
            })
        
        # Quick wins findings
        quick_wins = report.get("cost_savings_summary", {}).get("quick_wins", [])
        if len(quick_wins) > 0:
            findings.append({
                "type": "quick_wins",
                "severity": "low",
                "title": f"{len(quick_wins)} Quick Win Opportunities",
                "description": "Low-effort optimizations with immediate impact available",
                "impact": "positive"
            })
        
        return findings
    
    def _generate_next_steps(self, report: Dict[str, Any], scope: S3OptimizationScope) -> List[str]:
        """Generate recommended next steps based on analysis results."""
        
        next_steps = []
        
        # High-impact recommendations
        high_impact = report.get("cost_savings_summary", {}).get("high_impact_recommendations", [])
        if high_impact:
            next_steps.append(f"Prioritize implementation of {len(high_impact)} high-impact recommendations")
        
        # Quick wins
        quick_wins = report.get("cost_savings_summary", {}).get("quick_wins", [])
        if quick_wins:
            next_steps.append(f"Implement {len(quick_wins)} quick win optimizations for immediate savings")
        
        # Analysis-specific next steps
        analysis_summary = report.get("analysis_summary", {})
        
        if "storage_class" in analysis_summary:
            next_steps.append("Review and implement storage class lifecycle policies")
        
        if "multipart_cleanup" in analysis_summary:
            next_steps.append("Set up automated cleanup for incomplete multipart uploads")
        
        if "governance" in analysis_summary:
            next_steps.append("Establish S3 governance policies and compliance monitoring")
        
        # General recommendations
        next_steps.extend([
            "Schedule regular S3 optimization reviews (monthly recommended)",
            "Set up cost monitoring and alerting for S3 spending",
            "Consider implementing AWS Config rules for S3 governance"
        ])
        
        return next_steps[:7]  # Limit to top 7 next steps
    
    def _generate_risk_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment based on findings."""
        
        risks = []
        
        # Cost risk assessment
        total_savings = report.get("cost_savings_summary", {}).get("total_potential_savings", 0)
        if total_savings > 500:
            risks.append({
                "type": "cost_risk",
                "level": "high",
                "description": "High potential cost waste identified",
                "mitigation": "Immediate action recommended to implement cost optimizations"
            })
        
        # Governance risk assessment
        analysis_summary = report.get("analysis_summary", {})
        if "governance" in analysis_summary:
            governance_status = analysis_summary["governance"].get("status")
            if governance_status != "success":
                risks.append({
                    "type": "governance_risk",
                    "level": "medium",
                    "description": "S3 governance compliance issues detected",
                    "mitigation": "Review and implement proper S3 governance policies"
                })
        
        # Data management risk
        if "multipart_cleanup" in analysis_summary:
            multipart_status = analysis_summary["multipart_cleanup"].get("status")
            if multipart_status == "success":
                # Check if there are significant multipart upload issues
                risks.append({
                    "type": "data_management_risk",
                    "level": "low",
                    "description": "Incomplete multipart uploads consuming storage",
                    "mitigation": "Implement automated cleanup policies"
                })
        
        # Overall risk level
        if any(risk["level"] == "high" for risk in risks):
            overall_risk = "high"
        elif any(risk["level"] == "medium" for risk in risks):
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk_level": overall_risk,
            "identified_risks": risks,
            "risk_mitigation_priority": "high" if overall_risk == "high" else "medium"
        }
    
    def _store_comprehensive_results(self, report: Dict[str, Any]):
        """Store comprehensive results in session database."""
        try:
            # Store the main report
            session_manager = get_session_manager()
            
            # Create a summary table for the comprehensive analysis
            summary_data = [{
                "analysis_type": "comprehensive_s3_optimization",
                "timestamp": report["timestamp"],
                "total_potential_savings": report.get("cost_savings_summary", {}).get("total_potential_savings", 0),
                "analyses_completed": report.get("performance_metrics", {}).get("successful_analyses", 0),
                "health_score": report.get("executive_summary", {}).get("overview", {}).get("s3_health_score", {}).get("overall_score", 0),
                "session_id": self.session_id
            }]
            
            session_manager.store_data(
                session_id=self.session_id,
                table_name="s3_comprehensive_optimization_summary",
                data=summary_data
            )
            
            logger.info("Stored comprehensive optimization results in session database")
            
        except Exception as e:
            logger.error(f"Error storing comprehensive results: {str(e)}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for this tool instance."""
        return self.execution_history.copy()
    
    def get_available_analyses_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available analyses."""
        return self.available_analyses.copy()
    
    def create_optimization_scope(self, **kwargs) -> S3OptimizationScope:
        """Create an optimization scope configuration."""
        return S3OptimizationScope(**kwargs)