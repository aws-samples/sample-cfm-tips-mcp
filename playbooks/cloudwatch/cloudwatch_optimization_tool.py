"""
Comprehensive CloudWatch Optimization Tool

Unified tool that executes all 4 CloudWatch optimization functionalities in parallel
with intelligent analysis orchestration, cost-aware priority-based execution,
and comprehensive reporting.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
from playbooks.cloudwatch.cost_controller import CostController, CostPreferences, CostEstimate
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class CloudWatchOptimizationTool:
    """
    Comprehensive CloudWatch optimization tool that executes all functionalities.
    
    This tool provides:
    - Unified execution of all 4 CloudWatch optimization functionalities in parallel
    - Intelligent analysis orchestration with cost-aware priority-based execution
    - Comprehensive reporting that aggregates results from all analyzers
    - Executive summary with top recommendations and functionality coverage metrics
    - Configurable analysis scope (specific functionalities, resource filtering, time ranges, cost preferences)
    - Cost estimation and user consent workflows for paid features
    """
    
    def __init__(self, region: str = None, session_id: str = None):
        """Initialize the comprehensive CloudWatch optimization tool."""
        self.region = region
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize orchestrator
        self.orchestrator = CloudWatchOptimizationOrchestrator(
            region=region,
            session_id=session_id
        )
        
        # Initialize cost controller
        self.cost_controller = CostController()
        
        # Tool configuration
        self.tool_version = "1.0.0"
        self.supported_functionalities = [
            'general_spend',
            'metrics_optimization', 
            'logs_optimization',
            'alarms_and_dashboards'
        ]
        
        log_cloudwatch_operation(self.logger, "optimization_tool_initialized",
                               region=region, session_id=session_id,
                               supported_functionalities=len(self.supported_functionalities))
    
    async def execute_comprehensive_optimization_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive CloudWatch optimization analysis with all 4 functionalities in parallel.
        
        This method implements intelligent analysis orchestration with cost-aware priority-based execution,
        configurable analysis scope, and comprehensive cost estimation and user consent workflows.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region
                - lookback_days: Number of days to analyze (default: 30)
                - allow_cost_explorer: Enable Cost Explorer analysis (default: False)
                - allow_aws_config: Enable AWS Config governance checks (default: False)
                - allow_cloudtrail: Enable CloudTrail usage pattern analysis (default: False)
                - allow_minimal_cost_metrics: Enable minimal cost metrics (default: False)
                - functionalities: List of specific functionalities to run (default: all)
                - resource_filters: Dict of resource filters (log_group_names, alarm_names, etc.)
                - time_range_filters: Dict of time range filters (start_date, end_date)
                - priority_mode: 'cost_impact' | 'execution_time' | 'balanced' (default: 'balanced')
                - store_results: Whether to store results in session (default: True)
                - generate_executive_summary: Whether to generate executive summary (default: True)
                - max_parallel_analyses: Maximum parallel analyses (default: 4)
                
        Returns:
            Dictionary containing comprehensive optimization analysis results
        """
        start_time = datetime.now()
        
        log_cloudwatch_operation(self.logger, "comprehensive_optimization_start",
                               session_id=self.session_id, 
                               priority_mode=kwargs.get('priority_mode', 'balanced'))
        
        try:
            # Step 1: Validate and prepare cost preferences with detailed validation
            cost_validation = await self._validate_cost_preferences_with_consent(**kwargs)
            if cost_validation['status'] != 'success':
                return {
                    'status': 'error',
                    'error_message': f"Cost preference validation failed: {cost_validation.get('error_message')}",
                    'timestamp': start_time.isoformat(),
                    'cost_validation_details': cost_validation
                }
            
            cost_preferences = cost_validation['validated_preferences']
            functionality_coverage = cost_validation['functionality_coverage']
            
            # Step 2: Get detailed cost estimate with user consent workflow
            cost_estimate_result = await self._get_detailed_cost_estimate(**kwargs)
            if cost_estimate_result['status'] != 'success':
                return {
                    'status': 'error',
                    'error_message': f"Cost estimation failed: {cost_estimate_result.get('error_message')}",
                    'timestamp': start_time.isoformat(),
                    'cost_estimate_details': cost_estimate_result
                }
            
            cost_estimate = cost_estimate_result['cost_estimate']
            
            # Step 3: Configure analysis scope with intelligent filtering
            analysis_scope = self._configure_analysis_scope(**kwargs)
            
            # Step 4: Determine functionalities with priority-based ordering
            execution_plan = self._create_intelligent_execution_plan(
                requested_functionalities=kwargs.get('functionalities', self.supported_functionalities),
                cost_preferences=cost_preferences,
                priority_mode=kwargs.get('priority_mode', 'balanced'),
                analysis_scope=analysis_scope
            )
            
            if not execution_plan['valid_functionalities']:
                return {
                    'status': 'error',
                    'error_message': 'No valid functionalities specified or enabled',
                    'supported_functionalities': self.supported_functionalities,
                    'execution_plan': execution_plan,
                    'timestamp': start_time.isoformat()
                }
            
            # Step 5: Execute parallel analysis with intelligent orchestration
            parallel_results = await self._execute_parallel_analyses_with_orchestration(
                execution_plan=execution_plan,
                analysis_scope=analysis_scope,
                cost_preferences=cost_preferences,
                **kwargs
            )
            
            # Step 6: Execute cross-analysis insights and correlations
            cross_analysis_insights = await self._execute_cross_analysis_insights(
                parallel_results, analysis_scope, **kwargs
            )
            
            # Step 7: Generate executive summary with actionable insights
            executive_summary = None
            if kwargs.get('generate_executive_summary', True):
                executive_summary = await self._generate_executive_summary(
                    parallel_results, cross_analysis_insights, execution_plan, **kwargs
                )
            
            # Step 8: Compile comprehensive optimization report
            optimization_report = self._compile_comprehensive_report(
                parallel_results=parallel_results,
                cross_analysis_insights=cross_analysis_insights,
                executive_summary=executive_summary,
                execution_plan=execution_plan,
                analysis_scope=analysis_scope,
                cost_preferences=cost_preferences,
                functionality_coverage=functionality_coverage,
                cost_estimate=cost_estimate,
                start_time=start_time,
                **kwargs
            )
            
            log_cloudwatch_operation(self.logger, "comprehensive_optimization_complete",
                                   session_id=self.session_id,
                                   status=optimization_report.get('status'),
                                   total_execution_time=optimization_report.get('total_execution_time'),
                                   analyses_executed=len(execution_plan['valid_functionalities']))
            
            return optimization_report
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Comprehensive optimization analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'error_type': e.__class__.__name__,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_location': self._extract_error_location(full_traceback)
                },
                'timestamp': start_time.isoformat(),
                'session_id': self.session_id,
                'tool_version': self.tool_version,
                'execution_context': {
                    'kwargs_keys': list(kwargs.keys()),
                    'supported_functionalities': self.supported_functionalities
                }
            }
    
    def _compile_comprehensive_report(self, parallel_results: Dict[str, Any],
                                    cross_analysis_insights: Dict[str, Any],
                                    executive_summary: Optional[Dict[str, Any]],
                                    execution_plan: Dict[str, Any],
                                    analysis_scope: Dict[str, Any],
                                    cost_preferences: Dict[str, Any],
                                    functionality_coverage: Dict[str, Any],
                                    cost_estimate: Dict[str, Any],
                                    start_time: datetime,
                                    **kwargs) -> Dict[str, Any]:
        """
        Compile comprehensive optimization report from all analysis results.
        
        This method creates a unified report that includes:
        - Execution summary with intelligent orchestration metrics
        - Cost-aware analysis results with transparency
        - Cross-analysis insights and correlations
        - Executive summary with actionable recommendations
        - Detailed configuration and scope information
        """
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine overall status based on parallel execution results
        overall_status = 'success'
        if parallel_results.get('status') == 'error':
            overall_status = 'error'
        elif parallel_results.get('failed_analyses', 0) > 0:
            if parallel_results.get('successful_analyses', 0) > 0:
                overall_status = 'partial'
            else:
                overall_status = 'error'
        elif cross_analysis_insights.get('status') == 'error':
            overall_status = 'partial'
        
        # Extract key metrics from parallel execution
        successful_analyses = parallel_results.get('successful_analyses', 0)
        failed_analyses = parallel_results.get('failed_analyses', 0)
        total_analyses = successful_analyses + failed_analyses
        
        # Compile top recommendations from all sources
        top_recommendations = self._extract_top_recommendations_enhanced(
            parallel_results, cross_analysis_insights, executive_summary
        )
        
        # Create comprehensive report with enhanced structure
        report = {
            'status': overall_status,
            'report_type': 'comprehensive_cloudwatch_optimization_v2',
            'tool_version': self.tool_version,
            'generated_at': datetime.now().isoformat(),
            'analysis_started_at': start_time.isoformat(),
            'total_execution_time': total_execution_time,
            'session_id': self.session_id,
            'region': self.region,
            
            # Enhanced Analysis Configuration
            'analysis_configuration': {
                'execution_plan': execution_plan,
                'analysis_scope': analysis_scope,
                'cost_preferences': cost_preferences,
                'functionality_coverage': functionality_coverage,
                'cost_estimate': cost_estimate,
                'intelligent_orchestration': {
                    'priority_mode': execution_plan.get('priority_mode', 'balanced'),
                    'execution_batches': len(execution_plan.get('execution_batches', [])),
                    'max_parallel_analyses': analysis_scope.get('performance_constraints', {}).get('max_parallel_analyses', 4)
                }
            },
            
            # Enhanced Execution Summary
            'execution_summary': {
                'parallel_execution_metrics': {
                    'total_functionalities_requested': len(execution_plan.get('valid_functionalities', [])),
                    'successful_analyses': successful_analyses,
                    'failed_analyses': failed_analyses,
                    'success_rate': (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0,
                    'total_execution_time': parallel_results.get('total_execution_time', 0),
                    'average_analysis_time': (parallel_results.get('total_execution_time', 0) / total_analyses) if total_analyses > 0 else 0,
                    'execution_efficiency': self._calculate_execution_efficiency(parallel_results, execution_plan)
                },
                'cost_transparency': {
                    'cost_incurred': self._extract_cost_incurred(parallel_results),
                    'cost_incurring_operations': self._extract_cost_operations(parallel_results),
                    'primary_data_sources': self._extract_primary_data_sources(parallel_results),
                    'fallback_usage': self._extract_fallback_usage(parallel_results)
                },
                'batch_execution_details': parallel_results.get('batch_summaries', [])
            },
            
            # Enhanced Key Findings and Recommendations
            'key_findings': self._extract_key_findings_enhanced(parallel_results, cross_analysis_insights, executive_summary),
            'top_recommendations': top_recommendations,
            'optimization_priorities': self._determine_optimization_priorities_enhanced(top_recommendations, cross_analysis_insights),
            
            # Detailed Results with Enhanced Structure
            'detailed_results': {
                'individual_analyses': parallel_results.get('individual_results', {}),
                'cross_analysis_insights': cross_analysis_insights,
                'executive_summary': executive_summary,
                'parallel_execution_summary': parallel_results
            },
            
            # Enhanced Session and Data Information
            'session_metadata': {
                'session_id': self.session_id,
                'stored_tables': self._extract_stored_tables(parallel_results),
                'query_capabilities': 'Full SQL querying available on all stored analysis data',
                'data_retention': 'Session data available for 24 hours',
                'cross_analysis_correlations': cross_analysis_insights.get('correlation_strength', 'unknown')
            },
            
            # Enhanced Next Steps with Implementation Guidance
            'recommended_next_steps': self._generate_next_steps_enhanced(
                top_recommendations, functionality_coverage, cost_estimate, cross_analysis_insights, executive_summary
            ),
            
            # Implementation Support
            'implementation_support': {
                'cost_impact_analysis': self._generate_cost_impact_analysis(top_recommendations),
                'risk_assessment': self._generate_risk_assessment(top_recommendations),
                'timeline_recommendations': self._generate_timeline_recommendations(top_recommendations),
                'monitoring_recommendations': self._generate_monitoring_recommendations(parallel_results)
            }
        }
        
        return report
    
    def _extract_top_recommendations_enhanced(self, parallel_results: Dict[str, Any],
                                            cross_analysis_insights: Dict[str, Any],
                                            executive_summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and prioritize top recommendations from all analysis sources."""
        all_recommendations = []
        
        # Extract recommendations from individual parallel analyses
        individual_results = parallel_results.get('individual_results', {})
        for analysis_type, analysis_data in individual_results.items():
            if analysis_data.get('status') == 'success':
                recommendations = analysis_data.get('recommendations', [])
                for rec in recommendations:
                    rec['source'] = f'individual_analysis_{analysis_type}'
                    rec['analysis_type'] = analysis_type
                    all_recommendations.append(rec)
        
        # Extract recommendations from cross-analysis insights
        if cross_analysis_insights.get('status') == 'success':
            insights = cross_analysis_insights.get('insights', {})
            
            # Priority recommendations from cross-analysis
            priority_recs = insights.get('priority_recommendations', [])
            for rec in priority_recs:
                rec['source'] = 'cross_analysis_priority'
                all_recommendations.append(rec)
            
            # Synergy-based recommendations
            synergies = insights.get('optimization_synergies', [])
            for synergy in synergies:
                all_recommendations.append({
                    'type': 'optimization_synergy',
                    'priority': 'high',
                    'title': f"Synergistic Optimization: {synergy.get('type', 'Unknown')}",
                    'description': synergy.get('description', ''),
                    'potential_savings': 0,  # Synergies often have compound benefits
                    'implementation_effort': 'medium',
                    'source': 'cross_analysis_synergy',
                    'synergy_details': synergy
                })
        
        # Extract recommendations from executive summary
        if executive_summary and executive_summary.get('status') == 'success':
            exec_summary = executive_summary.get('executive_summary', {})
            immediate_actions = exec_summary.get('immediate_actions', [])
            for action in immediate_actions:
                all_recommendations.append({
                    'type': 'immediate_action',
                    'priority': 'critical',
                    'title': action.get('action', 'Immediate Action Required'),
                    'description': f"Source: {action.get('source', 'Unknown')}",
                    'potential_savings': action.get('impact', 0),
                    'implementation_effort': action.get('effort', 'medium'),
                    'source': 'executive_summary_immediate',
                    'timeline': action.get('timeline', 'Within 1 week')
                })
        
        # Enhanced sorting with multiple criteria
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        effort_order = {'low': 3, 'medium': 2, 'high': 1}
        
        sorted_recommendations = sorted(
            all_recommendations,
            key=lambda x: (
                priority_order.get(x.get('priority', 'low'), 1),
                x.get('potential_savings', 0),
                effort_order.get(x.get('implementation_effort', 'medium'), 2)
            ),
            reverse=True
        )
        
        # Return top 15 recommendations with enhanced metadata
        top_recommendations = sorted_recommendations[:15]
        
        # Add ranking and context to each recommendation
        for i, rec in enumerate(top_recommendations):
            rec['recommendation_rank'] = i + 1
            rec['recommendation_context'] = {
                'total_recommendations_analyzed': len(all_recommendations),
                'ranking_criteria': ['priority', 'potential_savings', 'implementation_effort'],
                'confidence_score': self._calculate_recommendation_confidence(rec)
            }
        
        return top_recommendations
    
    def _extract_key_findings(self, comprehensive_result: Dict[str, Any],
                            cross_analysis_insights: Dict[str, Any]) -> List[str]:
        """Extract key findings from all analyses."""
        findings = []
        
        # Analysis execution findings
        analysis_summary = comprehensive_result.get('analysis_summary', {})
        if analysis_summary:
            successful = analysis_summary.get('successful_analyses', 0)
            total = analysis_summary.get('total_analyses', 0)
            findings.append(f"Successfully completed {successful} of {total} CloudWatch optimization analyses")
        
        # Cost findings
        if comprehensive_result.get('cost_incurred', False):
            cost_ops = comprehensive_result.get('cost_incurring_operations', [])
            findings.append(f"Analysis used {len(cost_ops)} cost-incurring operations: {', '.join(cost_ops)}")
        else:
            findings.append("Analysis completed using only free AWS operations")
        
        # Data source findings
        primary_source = comprehensive_result.get('primary_data_source', 'unknown')
        findings.append(f"Primary data source: {primary_source}")
        
        # Cross-analysis findings
        insight_summary = cross_analysis_insights.get('summary', {})
        key_insights = insight_summary.get('key_findings', [])
        findings.extend(key_insights)
        
        # Cost impact findings
        cost_impact = insight_summary.get('cost_impact_analysis', {})
        if cost_impact.get('potential_monthly_savings'):
            savings = cost_impact['potential_monthly_savings']
            percentage = cost_impact.get('savings_percentage', 0)
            findings.append(f"Potential monthly savings identified: ${savings:.2f} ({percentage:.1f}% of current spend)")
        
        return findings
    
    def _determine_optimization_priorities(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine optimization priorities based on recommendations."""
        priorities = []
        
        # Group recommendations by priority level
        critical_recs = [r for r in recommendations if r.get('priority') == 'critical']
        high_recs = [r for r in recommendations if r.get('priority') == 'high']
        
        if critical_recs:
            priorities.append({
                'priority_level': 'critical',
                'title': 'Immediate Action Required',
                'description': f'{len(critical_recs)} critical optimization opportunities identified',
                'recommended_timeline': 'Within 1 week',
                'recommendations': critical_recs[:3]  # Top 3 critical
            })
        
        if high_recs:
            priorities.append({
                'priority_level': 'high',
                'title': 'High-Impact Optimizations',
                'description': f'{len(high_recs)} high-impact optimization opportunities identified',
                'recommended_timeline': 'Within 1 month',
                'recommendations': high_recs[:5]  # Top 5 high
            })
        
        # Add general implementation strategy
        priorities.append({
            'priority_level': 'strategic',
            'title': 'Implementation Strategy',
            'description': 'Recommended approach for CloudWatch optimization',
            'recommended_timeline': 'Ongoing',
            'strategy': [
                '1. Address critical issues immediately (cost impact >$100/month)',
                '2. Implement high-impact optimizations (cost impact >$50/month)',
                '3. Clean up unused resources (alarms, metrics, log groups)',
                '4. Establish governance policies to prevent future waste',
                '5. Schedule regular optimization reviews (quarterly)'
            ]
        })
        
        return priorities
    
    def _generate_next_steps(self, recommendations: List[Dict[str, Any]],
                           functionality_coverage: Dict[str, Any],
                           cost_estimate: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps based on analysis results."""
        next_steps = []
        
        # Based on functionality coverage
        overall_coverage = functionality_coverage.get('overall_coverage', 0)
        if overall_coverage < 70:
            next_steps.append(
                f"Consider enabling additional cost features for more comprehensive analysis "
                f"(current coverage: {overall_coverage:.1f}%)"
            )
        
        # Based on cost estimate
        if cost_estimate.get('total_estimated_cost', 0) == 0:
            next_steps.append(
                "All analysis completed with free operations - consider enabling Cost Explorer "
                "for historical cost data and trends"
            )
        
        # Based on recommendations
        if recommendations:
            critical_count = len([r for r in recommendations if r.get('priority') == 'critical'])
            if critical_count > 0:
                next_steps.append(f"Address {critical_count} critical optimization opportunities immediately")
            
            high_count = len([r for r in recommendations if r.get('priority') == 'high'])
            if high_count > 0:
                next_steps.append(f"Plan implementation of {high_count} high-impact optimizations")
        
        # General next steps
        next_steps.extend([
            "Review detailed analysis results for specific resource optimization opportunities",
            "Use session SQL queries to explore data relationships and patterns",
            "Implement monitoring for optimization progress tracking",
            "Schedule follow-up analysis after implementing changes"
        ])
        
        return next_steps
    
    async def execute_specific_functionalities(self, functionalities: List[str], **kwargs) -> Dict[str, Any]:
        """
        Execute specific CloudWatch optimization functionalities with intelligent orchestration.
        
        This method allows users to run a subset of the available functionalities with the same
        intelligent orchestration, cost-aware execution, and comprehensive reporting as the
        comprehensive analysis.
        
        Args:
            functionalities: List of specific functionalities to execute
            **kwargs: Analysis parameters (same as comprehensive analysis)
            
        Returns:
            Dictionary containing results for the specified functionalities
        """
        # Validate requested functionalities
        valid_functionalities = [f for f in functionalities if f in self.supported_functionalities]
        
        if not valid_functionalities:
            return {
                'status': 'error',
                'error_message': 'No valid functionalities specified',
                'requested_functionalities': functionalities,
                'supported_functionalities': self.supported_functionalities
            }
        
        # Execute comprehensive analysis with filtered functionalities
        kwargs['functionalities'] = valid_functionalities
        
        log_cloudwatch_operation(self.logger, "specific_functionalities_execution",
                               requested=functionalities,
                               valid=valid_functionalities,
                               session_id=self.session_id)
        
        return await self.execute_comprehensive_optimization_analysis(**kwargs)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about the optimization tool."""
        return {
            'tool_name': 'CloudWatch Optimization Tool',
            'version': self.tool_version,
            'supported_functionalities': self.supported_functionalities,
            'session_id': self.session_id,
            'region': self.region,
            'capabilities': [
                'Comprehensive CloudWatch cost analysis with all 4 functionalities',
                'Intelligent parallel execution with cost-aware priority-based orchestration',
                'Configurable analysis scope (specific functionalities, resource filtering, time ranges)',
                'Cost-aware analysis with user consent workflows and detailed cost estimation',
                'Cross-analysis insights and correlations for optimization synergies',
                'Executive summary generation with actionable recommendations',
                'SQL-queryable session data storage with comprehensive metadata',
                'Enhanced reporting with implementation guidance and risk assessment'
            ],
            'cost_control_features': [
                'Free operations prioritized (60% functionality coverage)',
                'Granular user-controlled paid operations with explicit consent',
                'Detailed cost estimation before analysis execution',
                'Transparent cost tracking with operation-level visibility',
                'Functionality coverage reporting with cost justification',
                'Graceful degradation to free operations when paid features disabled'
            ],
            'intelligent_orchestration_features': [
                'Priority-based execution ordering (cost_impact, execution_time, balanced)',
                'Parallel batch execution with configurable concurrency',
                'Intelligent resource filtering and scope configuration',
                'Cross-analysis correlation and synergy identification',
                'Executive summary with implementation roadmap',
                'Enhanced error handling with partial result recovery'
            ]
        }
    
    async def _validate_cost_preferences_with_consent(self, **kwargs) -> Dict[str, Any]:
        """
        Validate cost preferences with detailed consent workflow.
        
        Args:
            **kwargs: Analysis parameters including cost preferences
            
        Returns:
            Dictionary containing validation results and consent details
        """
        try:
            # Validate basic cost preferences
            cost_validation = self.orchestrator.validate_cost_preferences(**kwargs)
            
            if cost_validation.get('validation_status') != 'success':
                # Convert validation_status to status for consistency
                cost_validation['status'] = cost_validation.get('validation_status', 'error')
                return cost_validation
            
            # Add consent workflow details
            cost_validation['consent_workflow'] = {
                'consent_required': any([
                    kwargs.get('allow_cost_explorer', False),
                    kwargs.get('allow_aws_config', False),
                    kwargs.get('allow_cloudtrail', False),
                    kwargs.get('allow_minimal_cost_metrics', False)
                ]),
                'consent_timestamp': datetime.now().isoformat(),
                'consent_details': {
                    'cost_explorer_consent': kwargs.get('allow_cost_explorer', False),
                    'aws_config_consent': kwargs.get('allow_aws_config', False),
                    'cloudtrail_consent': kwargs.get('allow_cloudtrail', False),
                    'minimal_cost_metrics_consent': kwargs.get('allow_minimal_cost_metrics', False)
                }
            }
            
            # Ensure status field is set for consistency
            cost_validation['status'] = cost_validation.get('validation_status', 'success')
            
            return cost_validation
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Cost preference validation failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': f"Cost preference validation failed: {error_message}",
                'error_type': e.__class__.__name__,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_location': self._extract_error_location(full_traceback)
                },
                'validation_context': kwargs
            }
    
    async def _get_detailed_cost_estimate(self, **kwargs) -> Dict[str, Any]:
        """
        Get detailed cost estimate with user consent workflow.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing detailed cost estimation
        """
        try:
            # Get basic cost estimate
            cost_estimate_result = self.orchestrator.get_cost_estimate(**kwargs)
            
            if cost_estimate_result.get('status') == 'error':
                return cost_estimate_result
            
            # Add detailed cost breakdown and consent workflow
            cost_estimate_result['detailed_breakdown'] = {
                'free_operations_cost': 0.0,
                'paid_operations_estimate': cost_estimate_result.get('cost_estimate', {}).get('total_estimated_cost', 0.0),
                'cost_by_service': {
                    'cost_explorer': 0.01 if kwargs.get('allow_cost_explorer', False) else 0.0,
                    'aws_config': 0.003 if kwargs.get('allow_aws_config', False) else 0.0,
                    'cloudtrail': 0.10 if kwargs.get('allow_cloudtrail', False) else 0.0,
                    'minimal_cost_metrics': 0.01 if kwargs.get('allow_minimal_cost_metrics', False) else 0.0
                },
                'cost_justification': self._generate_cost_justification(**kwargs)
            }
            
            cost_estimate_result['status'] = 'success'
            return cost_estimate_result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Cost estimation failed: {str(e)}",
                'estimation_context': kwargs
            }
    
    def _generate_cost_justification(self, **kwargs) -> List[Dict[str, Any]]:
        """Generate cost justification for enabled paid features."""
        justifications = []
        
        if kwargs.get('allow_cost_explorer', False):
            justifications.append({
                'service': 'cost_explorer',
                'cost': 0.01,
                'justification': 'Provides historical cost trends and detailed spend analysis',
                'functionality_gain': '30% additional analysis coverage'
            })
        
        if kwargs.get('allow_aws_config', False):
            justifications.append({
                'service': 'aws_config',
                'cost': 0.003,
                'justification': 'Enables compliance checking and governance analysis',
                'functionality_gain': '5% additional analysis coverage'
            })
        
        if kwargs.get('allow_cloudtrail', False):
            justifications.append({
                'service': 'cloudtrail',
                'cost': 0.10,
                'justification': 'Provides usage pattern analysis and access insights',
                'functionality_gain': '1% additional analysis coverage'
            })
        
        if kwargs.get('allow_minimal_cost_metrics', False):
            justifications.append({
                'service': 'minimal_cost_metrics',
                'cost': 0.01,
                'justification': 'Enables detailed log ingestion pattern analysis',
                'functionality_gain': '4% additional analysis coverage'
            })
        
        return justifications
    
    def _configure_analysis_scope(self, **kwargs) -> Dict[str, Any]:
        """
        Configure intelligent analysis scope with resource filtering and time ranges.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary containing configured analysis scope
        """
        # Base scope configuration
        scope = {
            'temporal_scope': {
                'lookback_days': kwargs.get('lookback_days', 30),
                'start_date': kwargs.get('start_date'),
                'end_date': kwargs.get('end_date'),
                'analysis_granularity': kwargs.get('analysis_granularity', 'daily')
            },
            'resource_filters': {
                'log_group_names': kwargs.get('log_group_names', []),
                'log_group_patterns': kwargs.get('log_group_patterns', []),
                'alarm_names': kwargs.get('alarm_names', []),
                'alarm_patterns': kwargs.get('alarm_patterns', []),
                'dashboard_names': kwargs.get('dashboard_names', []),
                'metric_namespaces': kwargs.get('metric_namespaces', []),
                'exclude_patterns': kwargs.get('exclude_patterns', [])
            },
            'analysis_depth': {
                'include_detailed_metrics': kwargs.get('include_detailed_metrics', True),
                'include_cost_analysis': kwargs.get('include_cost_analysis', True),
                'include_governance_checks': kwargs.get('include_governance_checks', True),
                'include_optimization_recommendations': kwargs.get('include_optimization_recommendations', True)
            },
            'performance_constraints': {
                'max_parallel_analyses': kwargs.get('max_parallel_analyses', 4),
                'timeout_per_analysis': kwargs.get('timeout_per_analysis', 120),
                'memory_limit_mb': kwargs.get('memory_limit_mb', 1024),
                'enable_caching': kwargs.get('enable_caching', True)
            }
        }
        
        # Validate and sanitize scope
        scope = self._validate_analysis_scope(scope)
        
        return scope
    
    def _validate_analysis_scope(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize analysis scope parameters."""
        # Validate temporal scope
        lookback_days = scope['temporal_scope']['lookback_days']
        if lookback_days < 1 or lookback_days > 365:
            scope['temporal_scope']['lookback_days'] = min(max(lookback_days, 1), 365)
        
        # Validate performance constraints
        max_parallel = scope['performance_constraints']['max_parallel_analyses']
        if max_parallel < 1 or max_parallel > 8:
            scope['performance_constraints']['max_parallel_analyses'] = min(max(max_parallel, 1), 8)
        
        timeout = scope['performance_constraints']['timeout_per_analysis']
        if timeout < 30 or timeout > 600:
            scope['performance_constraints']['timeout_per_analysis'] = min(max(timeout, 30), 600)
        
        return scope
    
    def _create_intelligent_execution_plan(self, requested_functionalities: List[str], 
                                         cost_preferences: Dict[str, Any],
                                         priority_mode: str,
                                         analysis_scope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create intelligent execution plan with cost-aware priority-based ordering.
        
        Args:
            requested_functionalities: List of requested analysis functionalities
            cost_preferences: Validated cost preferences
            priority_mode: Priority mode ('cost_impact', 'execution_time', 'balanced')
            analysis_scope: Configured analysis scope
            
        Returns:
            Dictionary containing execution plan with prioritized functionalities
        """
        # Validate requested functionalities
        valid_functionalities = [f for f in requested_functionalities if f in self.supported_functionalities]
        
        # Define functionality metadata for prioritization
        functionality_metadata = {
            'general_spend': {
                'cost_impact_score': 10,  # High cost impact
                'execution_time_score': 3,  # Medium execution time
                'data_requirements': ['cost_explorer', 'cloudwatch_apis'],
                'dependencies': []
            },
            'logs_optimization': {
                'cost_impact_score': 9,  # High cost impact
                'execution_time_score': 4,  # Medium-high execution time
                'data_requirements': ['cost_explorer', 'cloudwatch_logs_apis', 'minimal_cost_metrics'],
                'dependencies': []
            },
            'metrics_optimization': {
                'cost_impact_score': 8,  # High cost impact
                'execution_time_score': 5,  # High execution time
                'data_requirements': ['cost_explorer', 'cloudwatch_apis', 'minimal_cost_metrics'],
                'dependencies': []
            },
            'alarms_and_dashboards': {
                'cost_impact_score': 6,  # Medium cost impact
                'execution_time_score': 2,  # Low execution time
                'data_requirements': ['cloudwatch_apis', 'aws_config', 'cloudtrail'],
                'dependencies': []
            }
        }
        
        # Calculate priority scores based on mode
        prioritized_functionalities = []
        for functionality in valid_functionalities:
            metadata = functionality_metadata.get(functionality, {})
            
            if priority_mode == 'cost_impact':
                priority_score = metadata.get('cost_impact_score', 5)
            elif priority_mode == 'execution_time':
                priority_score = 10 - metadata.get('execution_time_score', 5)  # Invert for faster first
            else:  # balanced
                cost_score = metadata.get('cost_impact_score', 5)
                time_score = 10 - metadata.get('execution_time_score', 5)
                priority_score = (cost_score * 0.7) + (time_score * 0.3)
            
            # Adjust score based on data availability
            data_requirements = metadata.get('data_requirements', [])
            availability_multiplier = self._calculate_data_availability_multiplier(
                data_requirements, cost_preferences
            )
            
            final_score = priority_score * availability_multiplier
            
            prioritized_functionalities.append({
                'functionality': functionality,
                'priority_score': final_score,
                'metadata': metadata,
                'data_availability_multiplier': availability_multiplier
            })
        
        # Sort by priority score (highest first)
        prioritized_functionalities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        execution_plan = {
            'valid_functionalities': [f['functionality'] for f in prioritized_functionalities],
            'prioritized_execution_order': prioritized_functionalities,
            'priority_mode': priority_mode,
            'total_functionalities': len(valid_functionalities),
            'execution_batches': self._create_execution_batches(
                prioritized_functionalities, analysis_scope['performance_constraints']['max_parallel_analyses']
            )
        }
        
        return execution_plan
    
    def _calculate_data_availability_multiplier(self, data_requirements: List[str], 
                                              cost_preferences: Dict[str, Any]) -> float:
        """Calculate data availability multiplier based on enabled cost preferences."""
        if not data_requirements:
            return 1.0
        
        available_sources = 0
        total_sources = len(data_requirements)
        
        for requirement in data_requirements:
            if requirement == 'cost_explorer' and cost_preferences.get('allow_cost_explorer', False):
                available_sources += 1
            elif requirement == 'aws_config' and cost_preferences.get('allow_aws_config', False):
                available_sources += 1
            elif requirement == 'cloudtrail' and cost_preferences.get('allow_cloudtrail', False):
                available_sources += 1
            elif requirement == 'minimal_cost_metrics' and cost_preferences.get('allow_minimal_cost_metrics', False):
                available_sources += 1
            elif requirement in ['cloudwatch_apis', 'cloudwatch_logs_apis']:
                available_sources += 1  # Always available (free)
        
        return available_sources / total_sources if total_sources > 0 else 1.0
    
    def _create_execution_batches(self, prioritized_functionalities: List[Dict[str, Any]], 
                                max_parallel: int) -> List[List[str]]:
        """Create execution batches for parallel processing."""
        batches = []
        current_batch = []
        
        for functionality_info in prioritized_functionalities:
            functionality = functionality_info['functionality']
            
            if len(current_batch) < max_parallel:
                current_batch.append(functionality)
            else:
                batches.append(current_batch)
                current_batch = [functionality]
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _execute_parallel_analyses_with_orchestration(self, execution_plan: Dict[str, Any],
                                                          analysis_scope: Dict[str, Any],
                                                          cost_preferences: Dict[str, Any],
                                                          **kwargs) -> Dict[str, Any]:
        """
        Execute analyses in parallel with intelligent orchestration.
        
        Args:
            execution_plan: Intelligent execution plan
            analysis_scope: Configured analysis scope
            cost_preferences: Validated cost preferences
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing parallel execution results
        """
        parallel_start_time = datetime.now()
        
        log_cloudwatch_operation(self.logger, "parallel_execution_start",
                               total_functionalities=len(execution_plan['valid_functionalities']),
                               execution_batches=len(execution_plan['execution_batches']))
        
        try:
            # Execute analyses in batches for optimal parallel processing
            all_results = {}
            batch_results = []
            
            for batch_index, batch in enumerate(execution_plan['execution_batches']):
                batch_start_time = datetime.now()
                
                log_cloudwatch_operation(self.logger, "executing_batch",
                                       batch_index=batch_index,
                                       batch_functionalities=batch)
                
                # Execute batch in parallel
                batch_tasks = []
                for functionality in batch:
                    task_kwargs = {
                        **kwargs,
                        'analysis_scope': analysis_scope,
                        'cost_preferences': cost_preferences,
                        'execution_context': {
                            'batch_index': batch_index,
                            'functionality': functionality,
                            'priority_score': next(
                                f['priority_score'] for f in execution_plan['prioritized_execution_order']
                                if f['functionality'] == functionality
                            )
                        }
                    }
                    
                    task = asyncio.create_task(
                        self.orchestrator.execute_analysis(functionality, **task_kwargs)
                    )
                    batch_tasks.append((functionality, task))
                
                # Wait for batch completion with timeout
                timeout = analysis_scope['performance_constraints']['timeout_per_analysis']
                batch_results_dict = {}
                
                for functionality, task in batch_tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=timeout)
                        batch_results_dict[functionality] = result
                        all_results[functionality] = result
                    except asyncio.TimeoutError:
                        error_result = {
                            'status': 'timeout',
                            'error_message': f'Analysis timed out after {timeout} seconds',
                            'functionality': functionality,
                            'batch_index': batch_index
                        }
                        batch_results_dict[functionality] = error_result
                        all_results[functionality] = error_result
                    except Exception as e:
                        error_result = {
                            'status': 'error',
                            'error_message': str(e),
                            'functionality': functionality,
                            'batch_index': batch_index
                        }
                        batch_results_dict[functionality] = error_result
                        all_results[functionality] = error_result
                
                batch_execution_time = (datetime.now() - batch_start_time).total_seconds()
                
                batch_summary = {
                    'batch_index': batch_index,
                    'functionalities': batch,
                    'execution_time': batch_execution_time,
                    'successful_analyses': len([r for r in batch_results_dict.values() if r.get('status') == 'success']),
                    'failed_analyses': len([r for r in batch_results_dict.values() if r.get('status') in ['error', 'timeout']]),
                    'results': batch_results_dict
                }
                
                batch_results.append(batch_summary)
                
                log_cloudwatch_operation(self.logger, "batch_execution_complete",
                                       batch_index=batch_index,
                                       execution_time=batch_execution_time,
                                       successful=batch_summary['successful_analyses'],
                                       failed=batch_summary['failed_analyses'])
            
            total_parallel_time = (datetime.now() - parallel_start_time).total_seconds()
            
            # Compile parallel execution summary
            parallel_summary = {
                'status': 'success',
                'total_execution_time': total_parallel_time,
                'total_functionalities': len(execution_plan['valid_functionalities']),
                'successful_analyses': len([r for r in all_results.values() if r.get('status') == 'success']),
                'failed_analyses': len([r for r in all_results.values() if r.get('status') in ['error', 'timeout']]),
                'batch_summaries': batch_results,
                'individual_results': all_results,
                'execution_plan': execution_plan,
                'analysis_scope': analysis_scope
            }
            
            log_cloudwatch_operation(self.logger, "parallel_execution_complete",
                                   total_time=total_parallel_time,
                                   successful=parallel_summary['successful_analyses'],
                                   failed=parallel_summary['failed_analyses'])
            
            return parallel_summary
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'execution_time': (datetime.now() - parallel_start_time).total_seconds(),
                'execution_plan': execution_plan
            }
    
    async def _execute_cross_analysis_insights(self, parallel_results: Dict[str, Any],
                                             analysis_scope: Dict[str, Any],
                                             **kwargs) -> Dict[str, Any]:
        """
        Execute cross-analysis insights and correlations.
        
        Args:
            parallel_results: Results from parallel analysis execution
            analysis_scope: Configured analysis scope
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing cross-analysis insights
        """
        try:
            # Extract successful results for cross-analysis
            successful_results = {
                k: v for k, v in parallel_results.get('individual_results', {}).items()
                if v.get('status') == 'success'
            }
            
            if len(successful_results) < 2:
                return {
                    'status': 'insufficient_data',
                    'message': 'Cross-analysis requires at least 2 successful analyses',
                    'available_analyses': list(successful_results.keys())
                }
            
            # Perform cross-analysis correlations
            cross_insights = {
                'cost_correlations': self._analyze_cost_correlations(successful_results),
                'resource_overlaps': self._analyze_resource_overlaps(successful_results),
                'optimization_synergies': self._identify_optimization_synergies(successful_results),
                'priority_recommendations': self._generate_priority_recommendations(successful_results)
            }
            
            return {
                'status': 'success',
                'insights': cross_insights,
                'analyses_included': list(successful_results.keys()),
                'correlation_strength': self._calculate_correlation_strength(cross_insights)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Cross-analysis failed: {str(e)}",
                'available_results': list(parallel_results.get('individual_results', {}).keys())
            }
    
    def _analyze_cost_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost correlations across different CloudWatch components."""
        correlations = {
            'high_cost_components': [],
            'cost_drivers': [],
            'optimization_impact': {}
        }
        
        # Extract cost data from each analysis
        for analysis_type, result in results.items():
            if 'cost_analysis' in result.get('data', {}):
                cost_data = result['data']['cost_analysis']
                if cost_data.get('monthly_cost', 0) > 100:  # High cost threshold
                    correlations['high_cost_components'].append({
                        'component': analysis_type,
                        'monthly_cost': cost_data.get('monthly_cost', 0),
                        'cost_trend': cost_data.get('cost_trend', 'unknown')
                    })
        
        return correlations
    
    def _analyze_resource_overlaps(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource overlaps and dependencies."""
        overlaps = {
            'shared_resources': [],
            'dependency_chains': [],
            'optimization_conflicts': []
        }
        
        # Identify shared log groups, metrics, alarms across analyses
        resource_usage = {}
        for analysis_type, result in results.items():
            resources = result.get('data', {}).get('resources_analyzed', [])
            for resource in resources:
                if resource not in resource_usage:
                    resource_usage[resource] = []
                resource_usage[resource].append(analysis_type)
        
        # Find resources used by multiple analyses
        for resource, analyses in resource_usage.items():
            if len(analyses) > 1:
                overlaps['shared_resources'].append({
                    'resource': resource,
                    'used_by_analyses': analyses,
                    'optimization_coordination_needed': True
                })
        
        return overlaps
    
    def _identify_optimization_synergies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization synergies across analyses."""
        synergies = []
        
        # Look for complementary optimizations
        if 'logs_optimization' in results and 'metrics_optimization' in results:
            synergies.append({
                'type': 'logs_metrics_synergy',
                'description': 'Optimizing log retention can reduce both log storage costs and related custom metrics',
                'combined_impact': 'high',
                'implementation_order': ['logs_optimization', 'metrics_optimization']
            })
        
        if 'alarms_and_dashboards' in results and 'metrics_optimization' in results:
            synergies.append({
                'type': 'alarms_metrics_synergy',
                'description': 'Removing unused metrics can eliminate related alarms and dashboard widgets',
                'combined_impact': 'medium',
                'implementation_order': ['metrics_optimization', 'alarms_and_dashboards']
            })
        
        return synergies
    
    def _generate_priority_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate priority recommendations based on cross-analysis."""
        priorities = []
        
        # Aggregate all recommendations and prioritize
        all_recommendations = []
        for analysis_type, result in results.items():
            recommendations = result.get('recommendations', [])
            for rec in recommendations:
                rec['source_analysis'] = analysis_type
                all_recommendations.append(rec)
        
        # Sort by priority and potential savings
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        sorted_recs = sorted(
            all_recommendations,
            key=lambda x: (
                priority_order.get(x.get('priority', 'low'), 1),
                x.get('potential_savings', 0)
            ),
            reverse=True
        )
        
        # Take top 5 cross-analysis priorities
        return sorted_recs[:5]
    
    def _calculate_correlation_strength(self, insights: Dict[str, Any]) -> str:
        """Calculate overall correlation strength between analyses."""
        correlation_factors = 0
        
        if insights.get('cost_correlations', {}).get('high_cost_components'):
            correlation_factors += 1
        
        if insights.get('resource_overlaps', {}).get('shared_resources'):
            correlation_factors += 1
        
        if insights.get('optimization_synergies'):
            correlation_factors += 1
        
        if correlation_factors >= 3:
            return 'strong'
        elif correlation_factors >= 2:
            return 'moderate'
        elif correlation_factors >= 1:
            return 'weak'
        else:
            return 'minimal'
    
    async def _generate_executive_summary(self, parallel_results: Dict[str, Any],
                                        cross_analysis_insights: Dict[str, Any],
                                        execution_plan: Dict[str, Any],
                                        **kwargs) -> Dict[str, Any]:
        """
        Generate executive summary with actionable insights.
        
        Args:
            parallel_results: Results from parallel analysis execution
            cross_analysis_insights: Cross-analysis insights
            execution_plan: Execution plan used
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing executive summary
        """
        try:
            successful_results = {
                k: v for k, v in parallel_results.get('individual_results', {}).items()
                if v.get('status') == 'success'
            }
            
            # Calculate key metrics
            total_potential_savings = 0
            critical_issues = 0
            high_priority_recommendations = 0
            
            for result in successful_results.values():
                recommendations = result.get('recommendations', [])
                for rec in recommendations:
                    total_potential_savings += rec.get('potential_savings', 0)
                    if rec.get('priority') == 'critical':
                        critical_issues += 1
                    elif rec.get('priority') == 'high':
                        high_priority_recommendations += 1
            
            # Generate executive summary
            executive_summary = {
                'analysis_overview': {
                    'total_analyses_requested': len(execution_plan['valid_functionalities']),
                    'successful_analyses': parallel_results.get('successful_analyses', 0),
                    'failed_analyses': parallel_results.get('failed_analyses', 0),
                    'total_execution_time': parallel_results.get('total_execution_time', 0),
                    'analysis_coverage': f"{(parallel_results.get('successful_analyses', 0) / len(execution_plan['valid_functionalities']) * 100):.1f}%"
                },
                'key_findings': {
                    'total_potential_monthly_savings': total_potential_savings,
                    'critical_issues_identified': critical_issues,
                    'high_priority_recommendations': high_priority_recommendations,
                    'cross_analysis_correlation': cross_analysis_insights.get('correlation_strength', 'unknown'),
                    'optimization_synergies_found': len(cross_analysis_insights.get('insights', {}).get('optimization_synergies', []))
                },
                'immediate_actions': self._generate_immediate_actions(successful_results, cross_analysis_insights),
                'strategic_recommendations': self._generate_strategic_recommendations(successful_results, cross_analysis_insights),
                'implementation_roadmap': self._generate_implementation_roadmap(successful_results, cross_analysis_insights),
                'cost_impact_summary': {
                    'estimated_monthly_savings': total_potential_savings,
                    'roi_timeline': '1-3 months',
                    'implementation_effort': self._assess_implementation_effort(successful_results),
                    'risk_level': 'low'
                }
            }
            
            return {
                'status': 'success',
                'executive_summary': executive_summary,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Executive summary generation failed: {str(e)}",
                'fallback_summary': {
                    'message': 'Executive summary generation failed, but detailed analysis results are available',
                    'successful_analyses': parallel_results.get('successful_analyses', 0),
                    'total_analyses': len(execution_plan.get('valid_functionalities', []))
                }
            }
    
    def _generate_immediate_actions(self, results: Dict[str, Any], 
                                  cross_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate immediate action items."""
        actions = []
        
        # Extract critical recommendations
        for analysis_type, result in results.items():
            recommendations = result.get('recommendations', [])
            critical_recs = [r for r in recommendations if r.get('priority') == 'critical']
            
            for rec in critical_recs[:2]:  # Top 2 critical per analysis
                actions.append({
                    'action': rec.get('title', 'Critical optimization'),
                    'source': analysis_type,
                    'timeline': 'Within 1 week',
                    'impact': rec.get('potential_savings', 0),
                    'effort': rec.get('implementation_effort', 'medium')
                })
        
        return actions[:5]  # Top 5 immediate actions
    
    def _generate_strategic_recommendations(self, results: Dict[str, Any],
                                          cross_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic recommendations."""
        strategic = []
        
        # Add synergy-based recommendations
        synergies = cross_insights.get('insights', {}).get('optimization_synergies', [])
        for synergy in synergies:
            strategic.append({
                'recommendation': synergy.get('description', 'Synergistic optimization'),
                'type': 'synergy',
                'impact': synergy.get('combined_impact', 'medium'),
                'timeline': '1-3 months'
            })
        
        # Add governance recommendations
        strategic.append({
            'recommendation': 'Establish CloudWatch cost governance policies',
            'type': 'governance',
            'impact': 'high',
            'timeline': '2-4 months',
            'description': 'Implement automated policies to prevent future CloudWatch cost waste'
        })
        
        return strategic
    
    def _generate_implementation_roadmap(self, results: Dict[str, Any],
                                       cross_insights: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate implementation roadmap."""
        return {
            'phase_1_immediate': [
                'Address critical cost optimization opportunities',
                'Remove unused alarms and dashboards',
                'Optimize log retention policies'
            ],
            'phase_2_short_term': [
                'Implement metrics optimization recommendations',
                'Establish monitoring for optimization progress',
                'Create cost governance policies'
            ],
            'phase_3_long_term': [
                'Implement automated cost optimization workflows',
                'Establish regular optimization review cycles',
                'Integrate with broader cloud cost management strategy'
            ]
        }
    
    def _assess_implementation_effort(self, results: Dict[str, Any]) -> str:
        """Assess overall implementation effort."""
        total_recommendations = sum(
            len(result.get('recommendations', [])) for result in results.values()
        )
        
        if total_recommendations > 20:
            return 'high'
        elif total_recommendations > 10:
            return 'medium'
        else:
            return 'low'
    
    def cleanup(self):
        """Clean up tool resources."""
        try:
            self.orchestrator.cleanup_session()
            log_cloudwatch_operation(self.logger, "optimization_tool_cleanup_complete",
                                   session_id=self.session_id)
        except Exception as e:
            self.logger.error(f"Error during tool cleanup: {str(e)}")
    
    def _calculate_recommendation_confidence(self, recommendation: Dict[str, Any]) -> float:
        """Calculate confidence score for a recommendation."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on source
        source = recommendation.get('source', '')
        if 'individual_analysis' in source:
            confidence += 0.3
        elif 'cross_analysis' in source:
            confidence += 0.4
        elif 'executive_summary' in source:
            confidence += 0.2
        
        # Increase confidence based on priority
        priority = recommendation.get('priority', 'low')
        if priority == 'critical':
            confidence += 0.3
        elif priority == 'high':
            confidence += 0.2
        elif priority == 'medium':
            confidence += 0.1
        
        # Increase confidence if potential savings are specified
        if recommendation.get('potential_savings', 0) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_execution_efficiency(self, parallel_results: Dict[str, Any], 
                                      execution_plan: Dict[str, Any]) -> float:
        """Calculate execution efficiency based on parallel performance."""
        total_time = parallel_results.get('total_execution_time', 0)
        total_analyses = len(execution_plan.get('valid_functionalities', []))
        
        if total_time == 0 or total_analyses == 0:
            return 0.0
        
        # Theoretical sequential time (assuming 60s per analysis)
        theoretical_sequential_time = total_analyses * 60
        
        # Efficiency is the ratio of theoretical time to actual time
        efficiency = min(theoretical_sequential_time / total_time, 1.0)
        
        return efficiency
    
    def _extract_cost_incurred(self, parallel_results: Dict[str, Any]) -> bool:
        """Extract whether any cost was incurred during analysis."""
        individual_results = parallel_results.get('individual_results', {})
        
        for result in individual_results.values():
            if result.get('cost_incurred', False):
                return True
        
        return False
    
    def _extract_cost_operations(self, parallel_results: Dict[str, Any]) -> List[str]:
        """Extract all cost-incurring operations from parallel results."""
        all_operations = []
        individual_results = parallel_results.get('individual_results', {})
        
        for result in individual_results.values():
            operations = result.get('cost_incurring_operations', [])
            all_operations.extend(operations)
        
        return list(set(all_operations))  # Remove duplicates
    
    def _extract_primary_data_sources(self, parallel_results: Dict[str, Any]) -> List[str]:
        """Extract primary data sources used across analyses."""
        sources = []
        individual_results = parallel_results.get('individual_results', {})
        
        for result in individual_results.values():
            source = result.get('primary_data_source')
            if source and source not in sources:
                sources.append(source)
        
        return sources
    
    def _extract_fallback_usage(self, parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fallback usage information."""
        fallback_info = {
            'fallback_used': False,
            'analyses_with_fallback': [],
            'fallback_reasons': []
        }
        
        individual_results = parallel_results.get('individual_results', {})
        
        for analysis_type, result in individual_results.items():
            if result.get('fallback_used', False):
                fallback_info['fallback_used'] = True
                fallback_info['analyses_with_fallback'].append(analysis_type)
                
                fallback_reason = result.get('fallback_reason', 'Unknown')
                if fallback_reason not in fallback_info['fallback_reasons']:
                    fallback_info['fallback_reasons'].append(fallback_reason)
        
        return fallback_info
    
    def _extract_stored_tables(self, parallel_results: Dict[str, Any]) -> List[str]:
        """Extract stored table names from parallel results."""
        tables = []
        individual_results = parallel_results.get('individual_results', {})
        
        for result in individual_results.values():
            session_tables = result.get('session_tables', [])
            tables.extend(session_tables)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_key_findings_enhanced(self, parallel_results: Dict[str, Any],
                                     cross_analysis_insights: Dict[str, Any],
                                     executive_summary: Optional[Dict[str, Any]]) -> List[str]:
        """Extract enhanced key findings from all analysis sources."""
        findings = []
        
        # Parallel execution findings
        successful = parallel_results.get('successful_analyses', 0)
        failed = parallel_results.get('failed_analyses', 0)
        total = successful + failed
        
        if total > 0:
            findings.append(f"Executed {total} CloudWatch optimization analyses with {successful} successful completions")
            
            if failed > 0:
                findings.append(f"{failed} analyses encountered issues but partial results may be available")
        
        # Cost transparency findings
        cost_incurred = self._extract_cost_incurred(parallel_results)
        if cost_incurred:
            cost_ops = self._extract_cost_operations(parallel_results)
            findings.append(f"Analysis used {len(cost_ops)} cost-incurring operations: {', '.join(cost_ops)}")
        else:
            findings.append("Analysis completed using only free AWS operations with no additional charges")
        
        # Execution efficiency findings
        efficiency = self._calculate_execution_efficiency(parallel_results, parallel_results.get('execution_plan', {}))
        if efficiency > 0.8:
            findings.append(f"Parallel execution achieved high efficiency ({efficiency:.1%}) through intelligent orchestration")
        elif efficiency > 0.5:
            findings.append(f"Parallel execution achieved moderate efficiency ({efficiency:.1%})")
        
        # Cross-analysis findings
        if cross_analysis_insights.get('status') == 'success':
            correlation = cross_analysis_insights.get('correlation_strength', 'unknown')
            findings.append(f"Cross-analysis correlation strength: {correlation}")
            
            insights = cross_analysis_insights.get('insights', {})
            synergies = insights.get('optimization_synergies', [])
            if synergies:
                findings.append(f"Identified {len(synergies)} optimization synergies for coordinated implementation")
        
        # Executive summary findings
        if executive_summary and executive_summary.get('status') == 'success':
            exec_data = executive_summary.get('executive_summary', {})
            key_findings = exec_data.get('key_findings', {})
            
            total_savings = key_findings.get('total_potential_monthly_savings', 0)
            if total_savings > 0:
                findings.append(f"Identified potential monthly savings of ${total_savings:.2f}")
            
            critical_issues = key_findings.get('critical_issues_identified', 0)
            if critical_issues > 0:
                findings.append(f"Found {critical_issues} critical issues requiring immediate attention")
        
        return findings
    
    def _determine_optimization_priorities_enhanced(self, recommendations: List[Dict[str, Any]],
                                                  cross_analysis_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine enhanced optimization priorities with cross-analysis context."""
        priorities = []
        
        # Critical priority tier
        critical_recs = [r for r in recommendations if r.get('priority') == 'critical']
        if critical_recs:
            priorities.append({
                'priority_level': 'critical',
                'title': 'Immediate Action Required',
                'description': f'{len(critical_recs)} critical optimization opportunities requiring immediate attention',
                'recommended_timeline': 'Within 1 week',
                'estimated_impact': sum(r.get('potential_savings', 0) for r in critical_recs[:3]),
                'recommendations': critical_recs[:3],
                'implementation_notes': [
                    'Address highest cost impact items first',
                    'Coordinate with stakeholders before implementation',
                    'Monitor impact after each change'
                ]
            })
        
        # High priority tier with synergies
        high_recs = [r for r in recommendations if r.get('priority') == 'high']
        if high_recs:
            synergy_note = ""
            if cross_analysis_insights.get('status') == 'success':
                synergies = cross_analysis_insights.get('insights', {}).get('optimization_synergies', [])
                if synergies:
                    synergy_note = f" Consider {len(synergies)} identified synergies for coordinated implementation."
            
            priorities.append({
                'priority_level': 'high',
                'title': 'High-Impact Optimizations',
                'description': f'{len(high_recs)} high-impact optimization opportunities identified.{synergy_note}',
                'recommended_timeline': 'Within 1 month',
                'estimated_impact': sum(r.get('potential_savings', 0) for r in high_recs[:5]),
                'recommendations': high_recs[:5],
                'synergy_opportunities': cross_analysis_insights.get('insights', {}).get('optimization_synergies', []),
                'implementation_notes': [
                    'Group related optimizations for batch implementation',
                    'Test changes in non-production environments first',
                    'Document changes for future reference'
                ]
            })
        
        # Strategic implementation guidance
        priorities.append({
            'priority_level': 'strategic',
            'title': 'CloudWatch Optimization Strategy',
            'description': 'Comprehensive approach to CloudWatch cost optimization and governance',
            'recommended_timeline': 'Ongoing',
            'strategy_components': [
                '1. Immediate: Address critical cost drivers (>$100/month impact)',
                '2. Short-term: Implement high-impact optimizations (>$50/month impact)',
                '3. Medium-term: Establish governance policies and automation',
                '4. Long-term: Integrate with broader cloud cost management strategy',
                '5. Continuous: Regular optimization reviews and monitoring'
            ],
            'governance_recommendations': [
                'Implement CloudWatch cost budgets and alerts',
                'Establish log retention policies by environment',
                'Create metrics naming conventions and lifecycle policies',
                'Implement automated cleanup for unused resources'
            ]
        })
        
        return priorities
    
    def _generate_next_steps_enhanced(self, recommendations: List[Dict[str, Any]],
                                    functionality_coverage: Dict[str, Any],
                                    cost_estimate: Dict[str, Any],
                                    cross_analysis_insights: Dict[str, Any],
                                    executive_summary: Optional[Dict[str, Any]]) -> List[str]:
        """Generate enhanced next steps with implementation guidance."""
        next_steps = []
        
        # Based on analysis completeness
        overall_coverage = functionality_coverage.get('overall_coverage', 0)
        if overall_coverage < 70:
            next_steps.append(
                f"Consider enabling additional cost features for more comprehensive analysis "
                f"(current coverage: {overall_coverage:.1f}%). Cost Explorer provides the highest value addition."
            )
        
        # Based on cost optimization opportunities
        if recommendations:
            critical_count = len([r for r in recommendations if r.get('priority') == 'critical'])
            high_count = len([r for r in recommendations if r.get('priority') == 'high'])
            
            if critical_count > 0:
                next_steps.append(f"URGENT: Address {critical_count} critical optimization opportunities within 1 week")
            
            if high_count > 0:
                next_steps.append(f"Plan implementation of {high_count} high-impact optimizations within 1 month")
        
        # Based on cross-analysis insights
        if cross_analysis_insights.get('status') == 'success':
            insights = cross_analysis_insights.get('insights', {})
            synergies = insights.get('optimization_synergies', [])
            
            if synergies:
                next_steps.append(f"Coordinate implementation of {len(synergies)} identified optimization synergies for maximum impact")
        
        # Based on executive summary
        if executive_summary and executive_summary.get('status') == 'success':
            exec_data = executive_summary.get('executive_summary', {})
            roadmap = exec_data.get('implementation_roadmap', {})
            
            phase_1 = roadmap.get('phase_1_immediate', [])
            if phase_1:
                next_steps.append(f"Execute Phase 1 immediate actions: {', '.join(phase_1[:2])}")
        
        # General implementation guidance
        next_steps.extend([
            "Review detailed analysis results and prioritize by cost impact and implementation effort",
            "Use session SQL queries to explore data relationships and identify additional optimization patterns",
            "Establish monitoring dashboards to track optimization progress and prevent regression",
            "Schedule quarterly CloudWatch optimization reviews to maintain cost efficiency"
        ])
        
        # Cost-specific guidance
        total_estimated_cost = cost_estimate.get('total_estimated_cost', 0)
        if total_estimated_cost == 0:
            next_steps.append(
                "All analysis completed with free operations. Consider enabling Cost Explorer "
                "for historical trends and more detailed cost attribution analysis."
            )
        
        return next_steps
    
    def _generate_cost_impact_analysis(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cost impact analysis for implementation planning."""
        total_potential_savings = sum(r.get('potential_savings', 0) for r in recommendations)
        
        # Categorize by impact level
        high_impact = [r for r in recommendations if r.get('potential_savings', 0) > 100]
        medium_impact = [r for r in recommendations if 50 <= r.get('potential_savings', 0) <= 100]
        low_impact = [r for r in recommendations if 0 < r.get('potential_savings', 0) < 50]
        
        return {
            'total_potential_monthly_savings': total_potential_savings,
            'annual_savings_projection': total_potential_savings * 12,
            'impact_distribution': {
                'high_impact_items': len(high_impact),
                'medium_impact_items': len(medium_impact),
                'low_impact_items': len(low_impact)
            },
            'roi_analysis': {
                'estimated_implementation_time': '2-4 weeks',
                'break_even_timeline': '1-2 months',
                'confidence_level': 'high' if total_potential_savings > 200 else 'medium'
            }
        }
    
    def _generate_risk_assessment(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate risk assessment for optimization implementations."""
        # Assess implementation risks
        high_effort_count = len([r for r in recommendations if r.get('implementation_effort') == 'high'])
        critical_count = len([r for r in recommendations if r.get('priority') == 'critical'])
        
        risk_level = 'low'
        if high_effort_count > 5 or critical_count > 3:
            risk_level = 'medium'
        if high_effort_count > 10 or critical_count > 5:
            risk_level = 'high'
        
        return {
            'overall_risk_level': risk_level,
            'risk_factors': {
                'implementation_complexity': 'medium' if high_effort_count > 3 else 'low',
                'business_impact': 'low',  # CloudWatch optimizations are generally low-risk
                'rollback_difficulty': 'low'  # Most changes are easily reversible
            },
            'mitigation_strategies': [
                'Test changes in non-production environments first',
                'Implement changes incrementally with monitoring',
                'Maintain backup configurations before making changes',
                'Coordinate with application teams for log retention changes'
            ]
        }
    
    def _generate_timeline_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate timeline recommendations for implementation."""
        critical_recs = [r for r in recommendations if r.get('priority') == 'critical']
        high_recs = [r for r in recommendations if r.get('priority') == 'high']
        medium_recs = [r for r in recommendations if r.get('priority') == 'medium']
        
        return {
            'week_1': [
                f"Address {len(critical_recs)} critical optimization opportunities",
                "Remove unused alarms and dashboards (quick wins)",
                "Optimize log retention policies for high-volume log groups"
            ],
            'month_1': [
                f"Implement {min(len(high_recs), 5)} high-impact optimizations",
                "Establish CloudWatch cost monitoring and alerting",
                "Create governance policies for new CloudWatch resources"
            ],
            'quarter_1': [
                f"Complete remaining {len(medium_recs)} medium-priority optimizations",
                "Implement automated cleanup workflows",
                "Conduct optimization impact review and adjust strategies"
            ],
            'ongoing': [
                "Monthly CloudWatch cost review and optimization",
                "Quarterly comprehensive optimization analysis",
                "Continuous monitoring of optimization effectiveness"
            ]
        }
    
    def _generate_monitoring_recommendations(self, parallel_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate monitoring recommendations based on analysis results."""
        recommendations = []
        
        # Cost monitoring
        recommendations.append({
            'type': 'cost_monitoring',
            'title': 'CloudWatch Cost Monitoring',
            'description': 'Implement comprehensive cost monitoring for CloudWatch services',
            'implementation': [
                'Set up CloudWatch cost budgets with alerts',
                'Create cost anomaly detection for CloudWatch spending',
                'Implement daily cost tracking dashboards'
            ]
        })
        
        # Resource monitoring
        recommendations.append({
            'type': 'resource_monitoring',
            'title': 'Resource Utilization Monitoring',
            'description': 'Monitor CloudWatch resource utilization and efficiency',
            'implementation': [
                'Track log ingestion rates and patterns',
                'Monitor alarm state changes and effectiveness',
                'Track dashboard usage and access patterns'
            ]
        })
        
        # Optimization tracking
        recommendations.append({
            'type': 'optimization_tracking',
            'title': 'Optimization Impact Tracking',
            'description': 'Track the impact of implemented optimizations',
            'implementation': [
                'Measure cost reduction from implemented changes',
                'Track resource cleanup and governance compliance',
                'Monitor for optimization regression'
            ]
        })
        
        return recommendations
    
    def _extract_error_location(self, traceback_str: str) -> str:
        """
        Extract error location from traceback string.
        
        Args:
            traceback_str: Full traceback string
            
        Returns:
            String describing the error location
        """
        try:
            lines = traceback_str.strip().split('\n')
            
            # Look for the last "File" line which usually contains the actual error location
            for line in reversed(lines):
                if line.strip().startswith('File "') and ', line ' in line:
                    # Extract file and line number
                    parts = line.strip().split(', line ')
                    if len(parts) >= 2:
                        file_part = parts[0].replace('File "', '').replace('"', '')
                        line_part = parts[1].split(',')[0]
                        
                        # Get just the filename, not the full path
                        filename = file_part.split('/')[-1] if '/' in file_part else file_part
                        
                        return f"{filename}:line {line_part}"
            
            # Fallback: look for any line with file information
            for line in lines:
                if 'File "' in line and 'line ' in line:
                    return line.strip()
            
            return "Error location not found in traceback"
            
        except Exception as e:
            return f"Error parsing traceback: {str(e)}"