"""
CloudWatch Optimization Analyzer

Single analyzer that orchestrates CloudWatch service Tips classes.
This is the playbook layer that coordinates between multiple Tips class methods,
handles parallel execution, manages session storage, and aggregates results.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from services.cloudwatch_service import (
    CWGeneralSpendTips,
    CWMetricsTips,
    CWLogsTips,
    CWAlarmsTips,
    CWDashboardTips,
    CloudWatchService,
    create_cloudwatch_service
)
from playbooks.cloudwatch.cost_controller import CostController, CostPreferences
from utils.service_orchestrator import ServiceOrchestrator
from utils.session_manager import get_session_manager
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class CloudWatchOptimizationAnalyzer:
    """
    Single analyzer that orchestrates CloudWatch service Tips classes.
    
    This analyzer:
    - Coordinates between multiple Tips class methods
    - Handles parallel execution using ServiceOrchestrator
    - Manages session storage using SessionManager
    - Aggregates results from multiple analyses
    - Provides high-level optimization insights
    """
    
    def __init__(self, region: str = None, session_id: str = None, 
                 cost_preferences: Optional[CostPreferences] = None):
        """Initialize the CloudWatch Optimization Analyzer."""
        self.region = region
        self.session_id = session_id
        self.cost_preferences = cost_preferences or CostPreferences()
        
        # Initialize service orchestrator for parallel execution
        self.service_orchestrator = ServiceOrchestrator(session_id)
        self.session_manager = get_session_manager()
        
        # Get actual session ID from orchestrator
        if self.session_id is None:
            self.session_id = self.service_orchestrator.session_id
        
        # Initialize cost controller
        self.cost_controller = CostController()
        
        # Initialize CloudWatch service
        self.cloudwatch_service = None  # Will be initialized async
        
        # Tip services (will be set after async initialization)
        self.general_spend_tips = None
        self.metrics_tips = None
        self.logs_tips = None
        self.alarms_tips = None
        self.dashboards_tips = None
        
        self.logger = logging.getLogger(__name__)
        
        log_cloudwatch_operation(self.logger, "analyzer_initialized",
                               region=region, session_id=self.session_id,
                               cost_preferences=str(self.cost_preferences))
    
    async def _ensure_initialized(self):
        """Ensure CloudWatch service and tip services are initialized."""
        if self.cloudwatch_service is None:
            self.cloudwatch_service = await create_cloudwatch_service(
                region=self.region,
                cost_preferences=self.cost_preferences
            )
            
            # Get tip services from the CloudWatch service
            self.general_spend_tips = self.cloudwatch_service.getGeneralSpendService()
            self.metrics_tips = self.cloudwatch_service.getMetricsService()
            self.logs_tips = self.cloudwatch_service.getLogsService()
            self.alarms_tips = self.cloudwatch_service.getAlarmsService()
            self.dashboards_tips = self.cloudwatch_service.getDashboardsService()
            
            log_cloudwatch_operation(self.logger, "services_initialized",
                                   region=self.region, session_id=self.session_id)

    async def analyze_general_spend(self, page: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Analyze general CloudWatch spending across all components.
        
        Orchestrates calls to:
        - general_spend_tips.getLogs()
        - general_spend_tips.getMetrics()
        - general_spend_tips.getDashboards()
        - general_spend_tips.getAlarms()
        
        Returns aggregated spend analysis with recommendations.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "general_spend_analysis_start",
                               page=page, session_id=self.session_id)
        
        try:
            # Call Tips methods to get data
            logs_result = await self.general_spend_tips.getLogs(
                page=page,
                log_group_name_prefix=kwargs.get('log_group_name_prefix'),
                can_spend_for_estimate=self.cost_preferences.allow_minimal_cost_metrics,
                lookback_days=kwargs.get('lookback_days', 30)
            )
            
            metrics_result = await self.general_spend_tips.getMetrics(
                page=page,
                namespace_filter=kwargs.get('namespace_filter')
            )
            
            dashboards_result = await self.general_spend_tips.getDashboards(
                page=page,
                dashboard_name_prefix=kwargs.get('dashboard_name_prefix')
            )
            
            alarms_result = await self.general_spend_tips.getAlarms(
                page=page,
                alarm_name_prefix=kwargs.get('alarm_name_prefix'),
                state_value=kwargs.get('state_value')
            )

            
            # Aggregate results
            total_monthly_cost = (
                logs_result.get('summary', {}).get('total_estimated_monthly_cost', 0) +
                metrics_result.get('summary', {}).get('total_estimated_monthly_cost', 0) +
                dashboards_result.get('summary', {}).get('total_estimated_monthly_cost', 0) +
                alarms_result.get('summary', {}).get('total_estimated_monthly_cost', 0)
            )
            
            aggregated = {
                'status': 'success',
                'analysis_type': 'general_spend',
                'logs': logs_result,
                'metrics': metrics_result,
                'dashboards': dashboards_result,
                'alarms': alarms_result,
                'summary': {
                    'total_monthly_cost': round(total_monthly_cost, 2),
                    'total_annual_cost': round(total_monthly_cost * 12, 2),
                    'breakdown': {
                        'logs_cost': logs_result.get('summary', {}).get('total_estimated_monthly_cost', 0),
                        'metrics_cost': metrics_result.get('summary', {}).get('total_estimated_monthly_cost', 0),
                        'dashboards_cost': dashboards_result.get('summary', {}).get('total_estimated_monthly_cost', 0),
                        'alarms_cost': alarms_result.get('summary', {}).get('total_estimated_monthly_cost', 0)
                    }
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id,
                'page': page
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'general_spend_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "general_spend_analysis_complete",
                                   total_cost=total_monthly_cost,
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"General spend analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'general_spend',
                'execution_time': time.time() - start_time
            }

    async def analyze_metrics_optimization(self, page: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Metrics optimization opportunities.
        
        Orchestrates calls to:
        - metrics_tips.listInstancesWithDetailedMonitoring()
        - metrics_tips.listCustomMetrics()
        - metrics_tips.analyze_metrics_usage()
        
        Returns metrics optimization recommendations.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "metrics_optimization_start",
                               page=page, session_id=self.session_id)
        
        try:
            # Call Tips methods
            detailed_monitoring = await self.metrics_tips.listInstancesWithDetailedMonitoring(page=page)
            
            custom_metrics = await self.metrics_tips.listCustomMetrics(
                page=page,
                namespace_filter=kwargs.get('namespace_filter'),
                can_spend_for_exact_usage_estimate=self.cost_preferences.allow_minimal_cost_metrics,
                lookback_days=kwargs.get('lookback_days', 30)
            )
            
            metrics_usage = await self.metrics_tips.analyze_metrics_usage(
                namespace_filter=kwargs.get('namespace_filter')
            )
            
            # Calculate total savings potential
            detailed_monitoring_savings = detailed_monitoring.get('optimization_tip', {}).get('potential_savings_monthly', 0)
            custom_metrics_savings = custom_metrics.get('optimization_tip', {}).get('potential_savings_monthly', 0)
            
            aggregated = {
                'status': 'success',
                'analysis_type': 'metrics_optimization',
                'detailed_monitoring': detailed_monitoring,
                'custom_metrics': custom_metrics,
                'metrics_usage': metrics_usage,
                'summary': {
                    'total_savings_potential_monthly': round(detailed_monitoring_savings + custom_metrics_savings, 2),
                    'total_savings_potential_annual': round((detailed_monitoring_savings + custom_metrics_savings) * 12, 2),
                    'detailed_monitoring_instances': detailed_monitoring.get('summary', {}).get('total_instances_with_detailed_monitoring', 0),
                    'inactive_custom_metrics': custom_metrics.get('summary', {}).get('inactive_metrics', 0)
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id,
                'page': page
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'metrics_optimization_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "metrics_optimization_complete",
                                   savings_potential=aggregated['summary']['total_savings_potential_monthly'],
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Metrics optimization analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'metrics_optimization',
                'execution_time': time.time() - start_time
            }

    async def analyze_logs_optimization(self, page: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Logs optimization opportunities.
        
        Orchestrates calls to:
        - logs_tips.listLogsWithoutRetention()
        - logs_tips.listVendedLogTargets()
        - logs_tips.listInfrequentAccessTargets()
        - logs_tips.analyze_logs_usage()
        
        Returns logs optimization recommendations.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "logs_optimization_start",
                               page=page, session_id=self.session_id)
        
        try:
            # Call Tips methods
            without_retention = await self.logs_tips.listLogsWithoutRetention(
                page=page,
                log_group_name_prefix=kwargs.get('log_group_name_prefix')
            )
            
            vended_targets = await self.logs_tips.listVendedLogTargets(
                page=page,
                log_group_name_prefix=kwargs.get('log_group_name_prefix')
            )
            
            infrequent_targets = await self.logs_tips.listInfrequentAccessTargets(
                page=page,
                log_group_name_prefix=kwargs.get('log_group_name_prefix')
            )
            
            logs_usage = await self.logs_tips.analyze_logs_usage(
                log_group_names=kwargs.get('log_group_names')
            )
            
            # Calculate total savings potential
            retention_cost = without_retention.get('summary', {}).get('total_monthly_cost', 0)
            vended_savings = vended_targets.get('summary', {}).get('total_monthly_savings', 0)
            infrequent_savings = infrequent_targets.get('summary', {}).get('total_monthly_savings_potential', 0)
            
            aggregated = {
                'status': 'success',
                'analysis_type': 'logs_optimization',
                'without_retention': without_retention,
                'vended_log_opportunities': vended_targets,
                'infrequent_access_candidates': infrequent_targets,
                'logs_usage': logs_usage,
                'summary': {
                    'total_savings_potential_monthly': round(retention_cost + vended_savings + infrequent_savings, 2),
                    'total_savings_potential_annual': round((retention_cost + vended_savings + infrequent_savings) * 12, 2),
                    'logs_without_retention': without_retention.get('summary', {}).get('total_log_groups_without_retention', 0),
                    'vended_log_opportunities': vended_targets.get('summary', {}).get('total_vended_log_opportunities', 0),
                    'infrequent_access_candidates': infrequent_targets.get('summary', {}).get('total_infrequent_access_candidates', 0)
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id,
                'page': page
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'logs_optimization_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "logs_optimization_complete",
                                   savings_potential=aggregated['summary']['total_savings_potential_monthly'],
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Logs optimization analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'logs_optimization',
                'execution_time': time.time() - start_time
            }

    async def analyze_alarms_optimization(self, page: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Alarms optimization opportunities.
        
        Orchestrates calls to:
        - alarms_tips.listAlarmsWithoutActions()
        - alarms_tips.listInsufficientDataAlarms()
        - alarms_tips.listHighResolutionAlarms()
        
        Returns alarms optimization recommendations.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "alarms_optimization_start",
                               page=page, session_id=self.session_id)
        
        try:
            # Call Tips methods (using correct method names)
            all_alarms = await self.alarms_tips.listAlarm(
                page=page,
                alarm_name_prefix=kwargs.get('alarm_name_prefix')
            )
            
            insufficient_data = await self.alarms_tips.listInvalidAlarm(
                page=page,
                alarm_name_prefix=kwargs.get('alarm_name_prefix')
            )
            
            # For now, use all_alarms as placeholder for without_actions and high_resolution
            # These can be filtered from all_alarms based on alarm properties
            without_actions = all_alarms
            high_resolution = all_alarms
            
            # Calculate total savings potential
            unused_alarms_cost = without_actions.get('summary', {}).get('total_monthly_cost', 0)
            high_res_savings = high_resolution.get('optimization_tip', {}).get('potential_savings_monthly', 0)
            
            aggregated = {
                'status': 'success',
                'analysis_type': 'alarms_optimization',
                'alarms_without_actions': without_actions,
                'insufficient_data_alarms': insufficient_data,
                'high_resolution_alarms': high_resolution,
                'summary': {
                    'total_savings_potential_monthly': round(unused_alarms_cost + high_res_savings, 2),
                    'total_savings_potential_annual': round((unused_alarms_cost + high_res_savings) * 12, 2),
                    'alarms_without_actions': without_actions.get('summary', {}).get('total_alarms_without_actions', 0),
                    'insufficient_data_alarms': insufficient_data.get('summary', {}).get('total_insufficient_data_alarms', 0),
                    'high_resolution_alarms': high_resolution.get('summary', {}).get('total_high_resolution_alarms', 0)
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id,
                'page': page
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'alarms_optimization_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "alarms_optimization_complete",
                                   savings_potential=aggregated['summary']['total_savings_potential_monthly'],
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Alarms optimization analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'alarms_optimization',
                'execution_time': time.time() - start_time
            }

    async def analyze_dashboards_optimization(self, page: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch Dashboards optimization opportunities.
        
        Orchestrates calls to:
        - dashboards_tips.listDashboardsWithCustomMetrics()
        - dashboards_tips.listUnusedDashboards()
        
        Returns dashboards optimization recommendations.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "dashboards_optimization_start",
                               page=page, session_id=self.session_id)
        
        try:
            # Call Tips methods (using correct method names)
            all_dashboards = await self.dashboards_tips.listDashboard(
                page=page,
                dashboard_name_prefix=kwargs.get('dashboard_name_prefix')
            )
            
            # For now, use all_dashboards as placeholder for both results
            # These can be filtered from all_dashboards based on dashboard properties
            with_custom_metrics = all_dashboards
            unused_dashboards = all_dashboards
            
            # Calculate total savings potential
            unused_cost = unused_dashboards.get('optimization_tip', {}).get('potential_savings_monthly', 0)
            
            aggregated = {
                'status': 'success',
                'analysis_type': 'dashboards_optimization',
                'dashboards_with_custom_metrics': with_custom_metrics,
                'unused_dashboards': unused_dashboards,
                'summary': {
                    'total_savings_potential_monthly': round(unused_cost, 2),
                    'total_savings_potential_annual': round(unused_cost * 12, 2),
                    'dashboards_with_custom_metrics': with_custom_metrics.get('summary', {}).get('total_dashboards', 0),
                    'unused_dashboards': unused_dashboards.get('summary', {}).get('total_unused_dashboards', 0)
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id,
                'page': page
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'dashboards_optimization_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "dashboards_optimization_complete",
                                   savings_potential=aggregated['summary']['total_savings_potential_monthly'],
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Dashboards optimization analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'dashboards_optimization',
                'execution_time': time.time() - start_time
            }

    async def analyze_comprehensive(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all CloudWatch analyses in parallel.
        
        Uses ServiceOrchestrator to run all analysis methods concurrently,
        aggregates results, and provides comprehensive optimization insights.
        """
        await self._ensure_initialized()
        
        start_time = time.time()
        
        log_cloudwatch_operation(self.logger, "comprehensive_analysis_start",
                               session_id=self.session_id)
        
        try:
            # Run all analyses in parallel
            results = await asyncio.gather(
                self.analyze_general_spend(**kwargs),
                self.analyze_metrics_optimization(**kwargs),
                self.analyze_logs_optimization(**kwargs),
                self.analyze_alarms_optimization(**kwargs),
                self.analyze_dashboards_optimization(**kwargs),
                return_exceptions=True
            )
            
            # Process results
            general_spend, metrics_opt, logs_opt, alarms_opt, dashboards_opt = results
            
            # Calculate overall status
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
            failed = len(results) - successful
            
            if failed == 0:
                overall_status = 'success'
            elif successful > 0:
                overall_status = 'partial'
            else:
                overall_status = 'error'
            
            # Calculate total savings potential
            total_savings = 0
            if isinstance(metrics_opt, dict):
                total_savings += metrics_opt.get('summary', {}).get('total_savings_potential_monthly', 0)
            if isinstance(logs_opt, dict):
                total_savings += logs_opt.get('summary', {}).get('total_savings_potential_monthly', 0)
            if isinstance(alarms_opt, dict):
                total_savings += alarms_opt.get('summary', {}).get('total_savings_potential_monthly', 0)
            if isinstance(dashboards_opt, dict):
                total_savings += dashboards_opt.get('summary', {}).get('total_savings_potential_monthly', 0)
            
            # Calculate total current cost
            total_cost = 0
            if isinstance(general_spend, dict):
                total_cost = general_spend.get('summary', {}).get('total_monthly_cost', 0)
            
            aggregated = {
                'status': overall_status,
                'analysis_type': 'comprehensive',
                'successful_analyses': successful,
                'total_analyses': len(results),
                'results': {
                    'general_spend': general_spend if isinstance(general_spend, dict) else {'status': 'error', 'error_message': str(general_spend)},
                    'metrics_optimization': metrics_opt if isinstance(metrics_opt, dict) else {'status': 'error', 'error_message': str(metrics_opt)},
                    'logs_optimization': logs_opt if isinstance(logs_opt, dict) else {'status': 'error', 'error_message': str(logs_opt)},
                    'alarms_optimization': alarms_opt if isinstance(alarms_opt, dict) else {'status': 'error', 'error_message': str(alarms_opt)},
                    'dashboards_optimization': dashboards_opt if isinstance(dashboards_opt, dict) else {'status': 'error', 'error_message': str(dashboards_opt)}
                },
                'summary': {
                    'total_current_monthly_cost': round(total_cost, 2),
                    'total_savings_potential_monthly': round(total_savings, 2),
                    'total_savings_potential_annual': round(total_savings * 12, 2),
                    'potential_cost_reduction_percentage': round((total_savings / total_cost * 100) if total_cost > 0 else 0, 1)
                },
                'execution_time': time.time() - start_time,
                'session_id': self.session_id
            }
            
            # Store in session
            if self.session_manager:
                try:
                    self.session_manager.store_analysis_summary(
                        self.session_id,
                        'comprehensive_cloudwatch_analysis',
                        aggregated
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in session: {str(e)}")
            
            log_cloudwatch_operation(self.logger, "comprehensive_analysis_complete",
                                   total_cost=total_cost,
                                   savings_potential=total_savings,
                                   execution_time=aggregated['execution_time'])
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'analysis_type': 'comprehensive',
                'execution_time': time.time() - start_time
            }

    def get_analysis_results(self, query: str) -> List[Dict[str, Any]]:
        """Query stored analysis results from session database."""
        try:
            if self.session_manager:
                results = self.session_manager.execute_query(self.session_id, query)
                log_cloudwatch_operation(self.logger, "analysis_results_retrieved",
                                       results_count=len(results), query=query)
                return results
            else:
                self.logger.warning("Session manager not available")
                return []
        except Exception as e:
            self.logger.error(f"Failed to query analysis results: {str(e)}")
            return []
    
    def validate_cost_preferences(self, **kwargs) -> Dict[str, Any]:
        """Validate and sanitize cost control flags."""
        try:
            validated = self.cost_controller.validate_and_sanitize_preferences(kwargs)
            functionality_coverage = self.cost_controller.get_functionality_coverage(validated)
            
            return {
                'valid': True,
                'validated_preferences': validated.__dict__,
                'functionality_coverage': functionality_coverage
            }
        except Exception as e:
            self.logger.error(f"Cost preferences validation failed: {str(e)}")
            return {
                'valid': False,
                'error_message': str(e)
            }
    
    def get_cost_estimate(self, **kwargs) -> Dict[str, Any]:
        """Provide cost estimate based on enabled features."""
        try:
            cost_prefs = self.cost_preferences
            if kwargs:
                cost_prefs = self.cost_controller.validate_and_sanitize_preferences(kwargs)
            
            analysis_scope = {
                'lookback_days': kwargs.get('lookback_days', 30),
                'log_group_names': kwargs.get('log_group_names', []),
                'alarm_names': kwargs.get('alarm_names', []),
                'dashboard_names': kwargs.get('dashboard_names', [])
            }
            
            cost_estimate = self.cost_controller.estimate_cost(analysis_scope, cost_prefs)
            
            return {
                'status': 'success',
                'cost_estimate': cost_estimate.__dict__,
                'cost_preferences': cost_prefs.__dict__
            }
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e)
            }
