"""
Alarms and Dashboards Analyzer for CloudWatch Optimization

Implements CloudWatch alarms and dashboards efficiency analysis using free CloudWatch APIs
as primary source with optional AWS Config and CloudTrail integration for governance checks.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from playbooks.cloudwatch.base_analyzer import BaseAnalyzer
from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class AlarmsAndDashboardsAnalyzer(BaseAnalyzer):
    """
    Alarms and dashboards analyzer for monitoring efficiency optimization.
    
    This analyzer provides:
    - Alarm efficiency analysis using free CloudWatch APIs as primary source
    - Dashboard usage analysis and free tier utilization tracking
    - Governance checks using AWS Config integration (allow_aws_config)
    - Usage pattern analysis using CloudTrail integration (allow_cloudtrail)
    - Efficiency recommendations with cost-aware feature coverage
    """
    
    def __init__(self, cost_explorer_service=None, config_service=None, 
                 metrics_service=None, cloudwatch_service=None, pricing_service=None,
                 performance_monitor=None, memory_manager=None):
        """Initialize AlarmsAndDashboardsAnalyzer with CloudWatch services."""
        super().__init__(
            cost_explorer_service=cost_explorer_service,
            config_service=config_service,
            metrics_service=metrics_service,
            cloudwatch_service=cloudwatch_service,
            pricing_service=pricing_service,
            performance_monitor=performance_monitor,
            memory_manager=memory_manager
        )
        
        # Analysis configuration
        self.analysis_type = "alarms_and_dashboards"
        self.version = "1.0.0"
        
        # Cost control flags
        self.cost_preferences = None
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive alarms and dashboards efficiency analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region
                - lookback_days: Number of days to analyze (default: 30)
                - allow_cost_explorer: Enable Cost Explorer analysis (default: False)
                - allow_aws_config: Enable AWS Config governance checks (default: False)
                - allow_cloudtrail: Enable CloudTrail usage pattern analysis (default: False)
                - alarm_names: Specific alarms to analyze
                - dashboard_names: Specific dashboards to analyze
                
        Returns:
            Dictionary containing comprehensive alarms and dashboards analysis results
        """
        start_time = datetime.now()
        context = self.prepare_analysis_context(**kwargs)
        
        # Extract cost preferences
        self.cost_preferences = {
            'allow_cost_explorer': kwargs.get('allow_cost_explorer', False),
            'allow_aws_config': kwargs.get('allow_aws_config', False),
            'allow_cloudtrail': kwargs.get('allow_cloudtrail', False),
            'allow_minimal_cost_metrics': kwargs.get('allow_minimal_cost_metrics', False)
        }
        
        log_cloudwatch_operation(self.logger, "alarms_dashboards_analysis_start",
                               cost_preferences=str(self.cost_preferences),
                               lookback_days=kwargs.get('lookback_days', 30))
        
        try:
            # Initialize result structure
            analysis_result = {
                'status': 'success',
                'analysis_type': self.analysis_type,
                'timestamp': start_time.isoformat(),
                'cost_incurred': False,
                'cost_incurring_operations': [],
                'primary_data_source': 'cloudwatch_config',
                'fallback_used': False,
                'data': {},
                'recommendations': []
            }
            
            # Execute analysis components in parallel
            analysis_tasks = []
            
            # 1. Alarm Efficiency Analysis (FREE - Always enabled)
            analysis_tasks.append(self._analyze_alarm_efficiency(**kwargs))
            
            # 2. Dashboard Analysis (FREE - Always enabled)
            analysis_tasks.append(self._analyze_dashboard_efficiency(**kwargs))
            
            # 3. Cost Explorer Analysis (PAID - User controlled)
            if self.cost_preferences['allow_cost_explorer']:
                analysis_tasks.append(self._analyze_cost_explorer_data(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('cost_explorer_analysis')
                analysis_result['primary_data_source'] = 'cost_explorer'
            
            # 4. AWS Config Governance Checks (PAID - User controlled)
            if self.cost_preferences['allow_aws_config']:
                analysis_tasks.append(self._analyze_governance_compliance(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('aws_config_governance')
            
            # 5. CloudTrail Usage Pattern Analysis (PAID - User controlled)
            if self.cost_preferences['allow_cloudtrail']:
                analysis_tasks.append(self._analyze_usage_patterns(**kwargs))
                analysis_result['cost_incurred'] = True
                analysis_result['cost_incurring_operations'].append('cloudtrail_usage_patterns')
            
            # Execute all analysis tasks
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Analysis task {i} failed: {str(result)}")
                    analysis_result['fallback_used'] = True
                elif isinstance(result, dict):
                    if result.get('status') == 'success':
                        # Merge successful results
                        analysis_result['data'].update(result.get('data', {}))
                    elif result.get('status') == 'error':
                        # Mark fallback used for error results
                        self.logger.warning(f"Analysis task {i} returned error: {result.get('error_message', 'Unknown error')}")
                        analysis_result['fallback_used'] = True
            
            # Generate efficiency analysis
            efficiency_analysis = await self._generate_efficiency_analysis(analysis_result['data'], **kwargs)
            analysis_result['data']['efficiency_analysis'] = efficiency_analysis
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            analysis_result['execution_time'] = execution_time
            
            log_cloudwatch_operation(self.logger, "alarms_dashboards_analysis_complete",
                                   execution_time=execution_time,
                                   cost_incurred=analysis_result['cost_incurred'],
                                   primary_data_source=analysis_result['primary_data_source'])
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Alarms and dashboards analysis failed: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_alarm_efficiency(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze alarm efficiency using free CloudWatch APIs.
        
        This provides comprehensive alarm analysis using only free operations.
        """
        log_cloudwatch_operation(self.logger, "alarm_efficiency_analysis_start", component="alarms")
        
        try:
            alarm_data = {}
            
            # Get all alarms configuration (FREE)
            if self.cloudwatch_service:
                alarms_result = await self.cloudwatch_service.describe_alarms(
                    alarm_names=kwargs.get('alarm_names')
                )
                if alarms_result.success:
                    alarms = alarms_result.data.get('alarms', [])
                    alarm_data['total_alarms'] = len(alarms)
                    
                    # Analyze alarm efficiency
                    efficiency_metrics = self._analyze_alarm_configurations(alarms)
                    alarm_data.update(efficiency_metrics)
                    
                    log_cloudwatch_operation(self.logger, "alarms_efficiency_analyzed",
                                           total_alarms=len(alarms),
                                           unused_alarms=efficiency_metrics.get('unused_alarms_count', 0),
                                           insufficient_data_alarms=efficiency_metrics.get('insufficient_data_count', 0))
            
            return {
                'status': 'success',
                'data': {'alarm_efficiency': alarm_data}
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Alarm efficiency analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    def _analyze_alarm_configurations(self, alarms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze alarm configurations for efficiency issues."""
        efficiency_metrics = {
            'unused_alarms': [],
            'unused_alarms_count': 0,
            'insufficient_data_alarms': [],
            'insufficient_data_count': 0,
            'high_resolution_alarms': [],
            'high_resolution_count': 0,
            'alarms_without_actions': [],
            'alarms_without_actions_count': 0,
            'alarm_states': {
                'OK': 0,
                'ALARM': 0,
                'INSUFFICIENT_DATA': 0
            },
            'alarm_types': {
                'metric': 0,
                'composite': 0,
                'anomaly_detector': 0
            }
        }
        
        for alarm in alarms:
            alarm_name = alarm.get('AlarmName', 'Unknown')
            alarm_state = alarm.get('StateValue', 'UNKNOWN')
            
            # Count alarm states
            if alarm_state in efficiency_metrics['alarm_states']:
                efficiency_metrics['alarm_states'][alarm_state] += 1
            
            # Identify alarms without actions (unused alarms)
            actions = (alarm.get('AlarmActions', []) + 
                      alarm.get('OKActions', []) + 
                      alarm.get('InsufficientDataActions', []))
            if not actions:
                efficiency_metrics['unused_alarms'].append({
                    'alarm_name': alarm_name,
                    'state': alarm_state,
                    'reason': 'No actions configured'
                })
                efficiency_metrics['unused_alarms_count'] += 1
                efficiency_metrics['alarms_without_actions'].append(alarm_name)
                efficiency_metrics['alarms_without_actions_count'] += 1
            
            # Identify alarms in INSUFFICIENT_DATA state
            if alarm_state == 'INSUFFICIENT_DATA':
                efficiency_metrics['insufficient_data_alarms'].append({
                    'alarm_name': alarm_name,
                    'state_reason': alarm.get('StateReason', 'Unknown'),
                    'state_updated': alarm.get('StateUpdatedTimestamp', 'Unknown')
                })
                efficiency_metrics['insufficient_data_count'] += 1
            
            # Identify high-resolution alarms (potential cost optimization)
            metric_name = alarm.get('MetricName')
            if metric_name and alarm.get('Period', 300) < 300:  # Less than 5 minutes
                efficiency_metrics['high_resolution_alarms'].append({
                    'alarm_name': alarm_name,
                    'period': alarm.get('Period'),
                    'metric_name': metric_name
                })
                efficiency_metrics['high_resolution_count'] += 1
            
            # Categorize alarm types
            if alarm.get('MetricName'):
                efficiency_metrics['alarm_types']['metric'] += 1
            elif 'Expression' in alarm:
                efficiency_metrics['alarm_types']['composite'] += 1
            elif 'AnomalyDetector' in alarm:
                efficiency_metrics['alarm_types']['anomaly_detector'] += 1
        
        return efficiency_metrics
    
    async def _analyze_dashboard_efficiency(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze dashboard efficiency and free tier utilization.
        
        This provides comprehensive dashboard analysis using only free operations.
        """
        log_cloudwatch_operation(self.logger, "dashboard_efficiency_analysis_start", component="dashboards")
        
        try:
            dashboard_data = {}
            
            # Get all dashboards (FREE)
            if self.cloudwatch_service:
                dashboards_result = await self.cloudwatch_service.list_dashboards(
                    dashboard_name_prefix=kwargs.get('dashboard_name_prefix')
                )
                if dashboards_result.success:
                    dashboards = dashboards_result.data.get('dashboards', [])
                    dashboard_data['total_dashboards'] = len(dashboards)
                    
                    # Analyze dashboard configurations
                    dashboard_details = []
                    for dashboard in dashboards:
                        dashboard_name = dashboard.get('DashboardName')
                        if dashboard_name:
                            # Get dashboard configuration (FREE)
                            config_result = await self.cloudwatch_service.get_dashboard(dashboard_name)
                            if config_result.success:
                                dashboard_config = config_result.data
                                dashboard_analysis = self._analyze_dashboard_configuration(
                                    dashboard_name, dashboard_config
                                )
                                dashboard_details.append(dashboard_analysis)
                    
                    # Calculate dashboard efficiency metrics
                    efficiency_metrics = self._calculate_dashboard_efficiency(dashboard_details)
                    dashboard_data.update(efficiency_metrics)
                    dashboard_data['dashboard_details'] = dashboard_details
                    
                    log_cloudwatch_operation(self.logger, "dashboards_efficiency_analyzed",
                                           total_dashboards=len(dashboards),
                                           free_tier_dashboards=efficiency_metrics.get('free_tier_count', 0),
                                           paid_dashboards=efficiency_metrics.get('paid_dashboards_count', 0))
            
            return {
                'status': 'success',
                'data': {'dashboard_efficiency': dashboard_data}
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Dashboard efficiency analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    def _analyze_dashboard_configuration(self, dashboard_name: str, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual dashboard configuration for efficiency."""
        dashboard_body = dashboard_config.get('DashboardBody', '{}')
        
        # Parse dashboard body to count widgets and metrics
        import json
        try:
            body = json.loads(dashboard_body) if isinstance(dashboard_body, str) else dashboard_body
        except json.JSONDecodeError:
            body = {}
        
        widgets = body.get('widgets', [])
        widget_count = len(widgets)
        
        # Count metrics across all widgets
        total_metrics = 0
        widget_types = {}
        
        for widget in widgets:
            widget_type = widget.get('type', 'unknown')
            widget_types[widget_type] = widget_types.get(widget_type, 0) + 1
            
            # Count metrics in widget properties
            properties = widget.get('properties', {})
            metrics = properties.get('metrics', [])
            if isinstance(metrics, list):
                total_metrics += len(metrics)
        
        return {
            'dashboard_name': dashboard_name,
            'widget_count': widget_count,
            'total_metrics': total_metrics,
            'widget_types': widget_types,
            'size': dashboard_config.get('DashboardArn', '').split('/')[-1] if dashboard_config.get('DashboardArn') else 'unknown'
        }
    
    def _calculate_dashboard_efficiency(self, dashboard_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dashboard efficiency metrics including free tier utilization."""
        # AWS CloudWatch free tier: 3 dashboards with up to 50 metrics each
        FREE_TIER_DASHBOARDS = 3
        FREE_TIER_METRICS_PER_DASHBOARD = 50
        
        efficiency_metrics = {
            'free_tier_count': 0,
            'paid_dashboards_count': 0,
            'free_tier_utilization': 0.0,
            'total_widgets': 0,
            'total_metrics': 0,
            'average_metrics_per_dashboard': 0.0,
            'oversized_dashboards': [],
            'undersized_dashboards': [],
            'widget_type_distribution': {}
        }
        
        total_dashboards = len(dashboard_details)
        if total_dashboards == 0:
            return efficiency_metrics
        
        # Sort dashboards by metric count (largest first) to optimize free tier usage
        sorted_dashboards = sorted(dashboard_details, key=lambda x: x.get('total_metrics', 0), reverse=True)
        
        total_widgets = sum(d.get('widget_count', 0) for d in dashboard_details)
        total_metrics = sum(d.get('total_metrics', 0) for d in dashboard_details)
        
        efficiency_metrics['total_dashboards'] = total_dashboards
        efficiency_metrics['total_widgets'] = total_widgets
        efficiency_metrics['total_metrics'] = total_metrics
        efficiency_metrics['average_metrics_per_dashboard'] = total_metrics / total_dashboards if total_dashboards > 0 else 0
        
        # Calculate free tier utilization
        free_tier_dashboards = min(total_dashboards, FREE_TIER_DASHBOARDS)
        efficiency_metrics['free_tier_count'] = free_tier_dashboards
        efficiency_metrics['paid_dashboards_count'] = max(0, total_dashboards - FREE_TIER_DASHBOARDS)
        efficiency_metrics['free_tier_utilization'] = (free_tier_dashboards / FREE_TIER_DASHBOARDS) * 100
        
        # Identify oversized and undersized dashboards
        for dashboard in dashboard_details:
            metrics_count = dashboard.get('total_metrics', 0)
            dashboard_name = dashboard.get('dashboard_name', 'Unknown')
            
            if metrics_count > FREE_TIER_METRICS_PER_DASHBOARD:
                efficiency_metrics['oversized_dashboards'].append({
                    'dashboard_name': dashboard_name,
                    'metrics_count': metrics_count,
                    'excess_metrics': metrics_count - FREE_TIER_METRICS_PER_DASHBOARD
                })
            elif metrics_count < 10:  # Arbitrary threshold for undersized
                efficiency_metrics['undersized_dashboards'].append({
                    'dashboard_name': dashboard_name,
                    'metrics_count': metrics_count,
                    'widget_count': dashboard.get('widget_count', 0)
                })
            
            # Aggregate widget types
            widget_types = dashboard.get('widget_types', {})
            for widget_type, count in widget_types.items():
                efficiency_metrics['widget_type_distribution'][widget_type] = (
                    efficiency_metrics['widget_type_distribution'].get(widget_type, 0) + count
                )
        
        return efficiency_metrics
    
    async def _analyze_cost_explorer_data(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze CloudWatch alarms and dashboards costs using Cost Explorer.
        
        This is a PAID operation that requires allow_cost_explorer=True.
        """
        log_cloudwatch_operation(self.logger, "cost_explorer_analysis_start", 
                               component="alarms_dashboards", operation_type="PAID")
        
        try:
            cost_data = {}
            lookback_days = kwargs.get('lookback_days', 30)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get CloudWatch costs filtered by alarms and dashboards
            if self.cost_explorer_service:
                # Get overall CloudWatch costs
                cloudwatch_costs = await self.cost_explorer_service.get_cloudwatch_spend_breakdown(
                    start_date.isoformat(), end_date.isoformat()
                )
                
                if cloudwatch_costs:
                    # Extract alarms and dashboards specific costs
                    cost_data['alarms_costs'] = self._extract_alarms_costs(cloudwatch_costs)
                    cost_data['dashboards_costs'] = self._extract_dashboards_costs(cloudwatch_costs)
                    cost_data['total_monitoring_costs'] = (
                        cost_data['alarms_costs'].get('total', 0) + 
                        cost_data['dashboards_costs'].get('total', 0)
                    )
            
            return {
                'status': 'success',
                'data': {'cost_analysis': cost_data}
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Cost Explorer analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    def _extract_alarms_costs(self, cloudwatch_costs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract alarms-specific costs from CloudWatch cost data."""
        alarms_costs = {
            'total': 0.0,
            'standard_alarms': 0.0,
            'high_resolution_alarms': 0.0,
            'composite_alarms': 0.0
        }
        
        # Parse cost data to extract alarms costs
        # This would need to be implemented based on actual Cost Explorer response format
        cost_breakdown = cloudwatch_costs.get('cost_breakdown', {})
        
        # Look for alarms-related usage types
        for usage_type, cost in cost_breakdown.items():
            if 'alarm' in usage_type.lower():
                alarms_costs['total'] += cost
                usage_lower = usage_type.lower()
                if 'high-resolution' in usage_lower or 'highresolution' in usage_lower:
                    alarms_costs['high_resolution_alarms'] += cost
                elif 'composite' in usage_lower:
                    alarms_costs['composite_alarms'] += cost
                else:
                    alarms_costs['standard_alarms'] += cost
        
        return alarms_costs
    
    def _extract_dashboards_costs(self, cloudwatch_costs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dashboards-specific costs from CloudWatch cost data."""
        dashboards_costs = {
            'total': 0.0,
            'custom_dashboards': 0.0,
            'dashboard_api_requests': 0.0
        }
        
        # Parse cost data to extract dashboards costs
        cost_breakdown = cloudwatch_costs.get('cost_breakdown', {})
        
        # Look for dashboards-related usage types
        for usage_type, cost in cost_breakdown.items():
            if 'dashboard' in usage_type.lower():
                dashboards_costs['total'] += cost
                if 'api' in usage_type.lower():
                    dashboards_costs['dashboard_api_requests'] += cost
                else:
                    dashboards_costs['custom_dashboards'] += cost
        
        return dashboards_costs
    
    async def _analyze_governance_compliance(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze governance compliance using AWS Config.
        
        This is a PAID operation that requires allow_aws_config=True.
        """
        log_cloudwatch_operation(self.logger, "governance_analysis_start", 
                               component="compliance", operation_type="PAID")
        
        try:
            governance_data = {
                'compliance_status': 'unknown',
                'config_rules': [],
                'non_compliant_alarms': [],
                'compliance_score': 0.0
            }
            
            # This would integrate with AWS Config to check compliance rules
            # For now, we'll simulate the analysis based on alarm configurations
            
            # Check alarm action compliance (simulated)
            if self.cloudwatch_service:
                alarms_result = await self.cloudwatch_service.describe_alarms()
                if alarms_result.success:
                    alarms = alarms_result.data.get('alarms', [])
                    compliance_check = self._check_alarm_action_compliance(alarms)
                    governance_data.update(compliance_check)
            
            return {
                'status': 'success',
                'data': {'governance_compliance': governance_data}
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Governance compliance analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    def _check_alarm_action_compliance(self, alarms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check alarm action compliance based on AWS Config rule cloudwatch-alarm-action-check."""
        total_alarms = len(alarms)
        compliant_alarms = 0
        non_compliant_alarms = []
        
        for alarm in alarms:
            alarm_name = alarm.get('AlarmName', 'Unknown')
            
            # Check if alarm has at least one action
            actions = (alarm.get('AlarmActions', []) + 
                      alarm.get('OKActions', []) + 
                      alarm.get('InsufficientDataActions', []))
            
            if actions:
                compliant_alarms += 1
            else:
                non_compliant_alarms.append({
                    'alarm_name': alarm_name,
                    'compliance_issue': 'No actions configured',
                    'state': alarm.get('StateValue', 'Unknown')
                })
        
        compliance_score = (compliant_alarms / total_alarms * 100) if total_alarms > 0 else 100
        
        return {
            'compliance_status': 'compliant' if compliance_score == 100 else 'non_compliant',
            'compliance_score': compliance_score,
            'total_alarms': total_alarms,
            'compliant_alarms': compliant_alarms,
            'non_compliant_alarms': non_compliant_alarms,
            'config_rules': ['cloudwatch-alarm-action-check']
        }
    
    async def _analyze_usage_patterns(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze usage patterns using CloudTrail.
        
        This is a PAID operation that requires allow_cloudtrail=True.
        """
        log_cloudwatch_operation(self.logger, "usage_patterns_analysis_start", 
                               component="usage_patterns", operation_type="PAID")
        
        try:
            usage_data = {
                'dashboard_access_patterns': {},
                'alarm_modification_patterns': {},
                'api_usage_summary': {},
                'inactive_resources': []
            }
            
            # This would integrate with CloudTrail to analyze usage patterns
            # For now, we'll provide a placeholder structure
            
            lookback_days = kwargs.get('lookback_days', 30)
            usage_data['analysis_period'] = f"{lookback_days} days"
            usage_data['note'] = "CloudTrail integration would provide detailed usage patterns"
            
            return {
                'status': 'success',
                'data': {'usage_patterns': usage_data}
            }
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            self.logger.error(f"Usage patterns analysis failed: {error_message}")
            self.logger.error(f"Full traceback: {full_traceback}")
            return {
                'status': 'error',
                'error_message': error_message,
                'full_exception_details': {
                    'traceback': full_traceback,
                    'error_type': e.__class__.__name__,
                    'error_location': self._extract_error_location(full_traceback) if hasattr(self, '_extract_error_location') else 'unknown'
                },
                'data': {}
            }
    
    async def _generate_efficiency_analysis(self, analysis_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate comprehensive efficiency analysis from all collected data."""
        efficiency_analysis = {
            'overall_efficiency_score': 0.0,
            'alarm_efficiency_score': 0.0,
            'dashboard_efficiency_score': 0.0,
            'cost_efficiency_score': 0.0,
            'governance_score': 0.0,
            'key_findings': [],
            'optimization_opportunities': []
        }
        
        # Calculate alarm efficiency score
        alarm_data = analysis_data.get('alarm_efficiency', {})
        if alarm_data:
            total_alarms = alarm_data.get('total_alarms', 0)
            unused_alarms = alarm_data.get('unused_alarms_count', 0)
            insufficient_data_alarms = alarm_data.get('insufficient_data_count', 0)
            
            if total_alarms > 0:
                efficiency_analysis['alarm_efficiency_score'] = max(0, 
                    100 - ((unused_alarms + insufficient_data_alarms) / total_alarms * 100)
                )
        
        # Calculate dashboard efficiency score
        dashboard_data = analysis_data.get('dashboard_efficiency', {})
        if dashboard_data:
            free_tier_utilization = dashboard_data.get('free_tier_utilization', 0)
            paid_dashboards = dashboard_data.get('paid_dashboards_count', 0)
            total_dashboards = dashboard_data.get('total_dashboards', 0)
            
            # Higher score for better free tier utilization and fewer paid dashboards
            if total_dashboards > 0:
                efficiency_analysis['dashboard_efficiency_score'] = (
                    free_tier_utilization * 0.7 + 
                    max(0, 100 - (paid_dashboards / total_dashboards * 100)) * 0.3
                )
        
        # Calculate governance score
        governance_data = analysis_data.get('governance_compliance', {})
        if governance_data:
            efficiency_analysis['governance_score'] = governance_data.get('compliance_score', 0)
        
        # Calculate overall efficiency score
        scores = [
            efficiency_analysis['alarm_efficiency_score'],
            efficiency_analysis['dashboard_efficiency_score'],
            efficiency_analysis['governance_score']
        ]
        valid_scores = [score for score in scores if score > 0]
        if valid_scores:
            efficiency_analysis['overall_efficiency_score'] = sum(valid_scores) / len(valid_scores)
        
        # Generate key findings
        efficiency_analysis['key_findings'] = self._generate_key_findings(analysis_data)
        
        # Generate optimization opportunities
        efficiency_analysis['optimization_opportunities'] = self._generate_optimization_opportunities(analysis_data)
        
        return efficiency_analysis
    
    def _generate_key_findings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate key findings from analysis data."""
        findings = []
        
        # Alarm findings
        alarm_data = analysis_data.get('alarm_efficiency', {})
        if alarm_data:
            total_alarms = alarm_data.get('total_alarms', 0)
            unused_alarms = alarm_data.get('unused_alarms_count', 0)
            insufficient_data = alarm_data.get('insufficient_data_count', 0)
            
            if total_alarms > 0:
                findings.append(f"Found {total_alarms} total alarms")
                if unused_alarms > 0:
                    findings.append(f"{unused_alarms} alarms have no actions configured (unused)")
                if insufficient_data > 0:
                    findings.append(f"{insufficient_data} alarms are in INSUFFICIENT_DATA state")
        
        # Dashboard findings
        dashboard_data = analysis_data.get('dashboard_efficiency', {})
        if dashboard_data:
            total_dashboards = dashboard_data.get('total_dashboards', 0)
            paid_dashboards = dashboard_data.get('paid_dashboards_count', 0)
            free_tier_util = dashboard_data.get('free_tier_utilization', 0)
            
            if total_dashboards > 0:
                findings.append(f"Found {total_dashboards} total dashboards")
                if paid_dashboards > 0:
                    findings.append(f"{paid_dashboards} dashboards exceed free tier (incurring costs)")
                findings.append(f"Free tier utilization: {free_tier_util:.1f}%")
        
        return findings
    
    def _generate_optimization_opportunities(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization opportunities from analysis data."""
        opportunities = []
        
        # Alarm optimization opportunities
        alarm_data = analysis_data.get('alarm_efficiency', {})
        if alarm_data:
            unused_alarms = alarm_data.get('unused_alarms_count', 0)
            if unused_alarms > 0:
                opportunities.append({
                    'type': 'alarm_cleanup',
                    'title': f'Remove {unused_alarms} unused alarms',
                    'description': 'Delete alarms without actions to reduce costs',
                    'potential_savings': unused_alarms * 0.10,  # $0.10 per alarm per month
                    'effort': 'low'
                })
            
            high_res_alarms = alarm_data.get('high_resolution_count', 0)
            if high_res_alarms > 0:
                opportunities.append({
                    'type': 'alarm_optimization',
                    'title': f'Convert {high_res_alarms} high-resolution alarms to standard',
                    'description': 'Reduce alarm costs by using standard resolution where appropriate',
                    'potential_savings': high_res_alarms * 0.20,  # Additional cost for high-res
                    'effort': 'medium'
                })
        
        # Dashboard optimization opportunities
        dashboard_data = analysis_data.get('dashboard_efficiency', {})
        if dashboard_data:
            paid_dashboards = dashboard_data.get('paid_dashboards_count', 0)
            if paid_dashboards > 0:
                opportunities.append({
                    'type': 'dashboard_consolidation',
                    'title': f'Consolidate {paid_dashboards} paid dashboards',
                    'description': 'Merge dashboards to stay within free tier limits',
                    'potential_savings': paid_dashboards * 3.00,  # $3 per dashboard per month
                    'effort': 'medium'
                })
            
            oversized_dashboards = dashboard_data.get('oversized_dashboards', [])
            if oversized_dashboards:
                opportunities.append({
                    'type': 'dashboard_optimization',
                    'title': f'Optimize {len(oversized_dashboards)} oversized dashboards',
                    'description': 'Reduce metrics per dashboard to optimize costs',
                    'potential_savings': len(oversized_dashboards) * 1.00,
                    'effort': 'medium'
                })
        
        return opportunities
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from alarms and dashboards analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        data = analysis_results.get('data', {})
        
        # Alarm efficiency recommendations
        alarm_data = data.get('alarm_efficiency', {})
        if alarm_data:
            recommendations.extend(self._generate_alarm_recommendations(alarm_data))
        
        # Dashboard efficiency recommendations
        dashboard_data = data.get('dashboard_efficiency', {})
        if dashboard_data:
            recommendations.extend(self._generate_dashboard_recommendations(dashboard_data))
        
        # Governance recommendations
        governance_data = data.get('governance_compliance', {})
        if governance_data:
            recommendations.extend(self._generate_governance_recommendations(governance_data))
        
        # Cost optimization recommendations
        cost_data = data.get('cost_analysis', {})
        if cost_data:
            recommendations.extend(self._generate_cost_recommendations(cost_data))
        
        # Efficiency analysis recommendations
        efficiency_data = data.get('efficiency_analysis', {})
        if efficiency_data:
            recommendations.extend(self._generate_efficiency_recommendations(efficiency_data))
        
        return recommendations
    
    def _generate_alarm_recommendations(self, alarm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alarm-specific recommendations."""
        recommendations = []
        
        # Unused alarms recommendation
        unused_alarms = alarm_data.get('unused_alarms', [])
        if unused_alarms:
            potential_savings = len(unused_alarms) * 0.10 * 12  # $0.10 per alarm per month
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="high",
                title=f"Remove {len(unused_alarms)} unused CloudWatch alarms",
                description=f"Found {len(unused_alarms)} alarms without any actions configured. These alarms provide no operational value and incur unnecessary costs.",
                potential_savings=potential_savings,
                implementation_effort="low",
                affected_resources=[alarm['alarm_name'] for alarm in unused_alarms],
                action_items=[
                    "Review each alarm to confirm it's truly unused",
                    "Delete alarms that have no actions and serve no monitoring purpose",
                    "Consider adding actions to alarms that should be monitoring critical metrics",
                    "Document the cleanup process for future reference"
                ],
                cloudwatch_component="alarms"
            ))
        
        # INSUFFICIENT_DATA alarms recommendation
        insufficient_data_alarms = alarm_data.get('insufficient_data_alarms', [])
        if insufficient_data_alarms:
            recommendations.append(self.create_recommendation(
                rec_type="performance",
                priority="medium",
                title=f"Fix {len(insufficient_data_alarms)} alarms in INSUFFICIENT_DATA state",
                description=f"Found {len(insufficient_data_alarms)} alarms in INSUFFICIENT_DATA state, indicating potential configuration issues or missing metrics.",
                implementation_effort="medium",
                affected_resources=[alarm['alarm_name'] for alarm in insufficient_data_alarms],
                action_items=[
                    "Review alarm configurations for correct metric names and dimensions",
                    "Verify that the monitored resources are still active and generating metrics",
                    "Check if the alarm period and evaluation periods are appropriate",
                    "Consider adjusting alarm thresholds or evaluation criteria",
                    "Update or delete alarms for resources that no longer exist"
                ],
                cloudwatch_component="alarms"
            ))
        
        # High-resolution alarms recommendation
        high_res_alarms = alarm_data.get('high_resolution_alarms', [])
        if high_res_alarms:
            potential_savings = len(high_res_alarms) * 0.20 * 12  # Additional cost for high-res
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="medium",
                title=f"Optimize {len(high_res_alarms)} high-resolution alarms",
                description=f"Found {len(high_res_alarms)} high-resolution alarms that could potentially use standard resolution to reduce costs.",
                potential_savings=potential_savings,
                implementation_effort="medium",
                affected_resources=[alarm['alarm_name'] for alarm in high_res_alarms],
                action_items=[
                    "Review if sub-5-minute resolution is truly necessary for each alarm",
                    "Convert alarms to standard resolution (5-minute periods) where appropriate",
                    "Consider the business impact of slightly delayed alerting",
                    "Keep high-resolution only for critical real-time monitoring needs"
                ],
                cloudwatch_component="alarms"
            ))
        
        return recommendations
    
    def _generate_dashboard_recommendations(self, dashboard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard-specific recommendations."""
        recommendations = []
        
        # Paid dashboards recommendation
        paid_dashboards_count = dashboard_data.get('paid_dashboards_count', 0)
        if paid_dashboards_count > 0:
            potential_savings = paid_dashboards_count * 3.00 * 12  # $3 per dashboard per month
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="high",
                title=f"Consolidate {paid_dashboards_count} paid dashboards to reduce costs",
                description=f"You have {paid_dashboards_count} dashboards beyond the free tier (3 dashboards). Consider consolidating to stay within free tier limits.",
                potential_savings=potential_savings,
                implementation_effort="medium",
                action_items=[
                    "Review dashboard usage patterns to identify consolidation opportunities",
                    "Merge related dashboards into comprehensive views",
                    "Remove duplicate or rarely used dashboards",
                    "Optimize widget layouts to fit more information in fewer dashboards",
                    "Consider using dashboard folders for organization instead of separate dashboards"
                ],
                cloudwatch_component="dashboards"
            ))
        
        # Oversized dashboards recommendation
        oversized_dashboards = dashboard_data.get('oversized_dashboards', [])
        if oversized_dashboards:
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="medium",
                title=f"Optimize {len(oversized_dashboards)} oversized dashboards",
                description=f"Found {len(oversized_dashboards)} dashboards with more than 50 metrics, which may incur additional costs.",
                implementation_effort="medium",
                affected_resources=[dash['dashboard_name'] for dash in oversized_dashboards],
                action_items=[
                    "Review metrics in oversized dashboards for relevance",
                    "Remove redundant or rarely viewed metrics",
                    "Split large dashboards into focused, smaller dashboards",
                    "Use dashboard filters to reduce the number of displayed metrics",
                    "Consider using CloudWatch Insights for ad-hoc analysis instead of permanent dashboard widgets"
                ],
                cloudwatch_component="dashboards"
            ))
        
        # Undersized dashboards recommendation
        undersized_dashboards = dashboard_data.get('undersized_dashboards', [])
        if len(undersized_dashboards) > 1:
            recommendations.append(self.create_recommendation(
                rec_type="efficiency",
                priority="low",
                title=f"Consider consolidating {len(undersized_dashboards)} undersized dashboards",
                description=f"Found {len(undersized_dashboards)} dashboards with very few metrics that could potentially be consolidated.",
                implementation_effort="low",
                affected_resources=[dash['dashboard_name'] for dash in undersized_dashboards],
                action_items=[
                    "Review if small dashboards serve distinct purposes",
                    "Merge related small dashboards into comprehensive views",
                    "Add relevant metrics to underutilized dashboards",
                    "Consider if some dashboards can be replaced with CloudWatch alarms"
                ],
                cloudwatch_component="dashboards"
            ))
        
        # Free tier utilization recommendation
        free_tier_utilization = dashboard_data.get('free_tier_utilization', 0)
        if free_tier_utilization < 100 and paid_dashboards_count == 0:
            recommendations.append(self.create_recommendation(
                rec_type="efficiency",
                priority="low",
                title="Optimize free tier dashboard utilization",
                description=f"You're using {free_tier_utilization:.1f}% of your free tier dashboard allocation. Consider maximizing free tier usage before creating paid dashboards.",
                implementation_effort="low",
                action_items=[
                    "Create additional dashboards within the free tier limit",
                    "Add more comprehensive monitoring to existing dashboards",
                    "Utilize the full 50 metrics per dashboard allowance",
                    "Plan dashboard strategy to maximize free tier benefits"
                ],
                cloudwatch_component="dashboards"
            ))
        
        return recommendations
    
    def _generate_governance_recommendations(self, governance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate governance and compliance recommendations."""
        recommendations = []
        
        compliance_score = governance_data.get('compliance_score', 100)
        non_compliant_alarms = governance_data.get('non_compliant_alarms', [])
        
        if compliance_score < 100 and non_compliant_alarms:
            recommendations.append(self.create_recommendation(
                rec_type="governance",
                priority="high",
                title=f"Fix compliance issues in {len(non_compliant_alarms)} alarms",
                description=f"Compliance score is {compliance_score:.1f}%. Found {len(non_compliant_alarms)} alarms that don't meet governance requirements.",
                implementation_effort="medium",
                affected_resources=[alarm['alarm_name'] for alarm in non_compliant_alarms],
                action_items=[
                    "Add appropriate actions to alarms without actions",
                    "Ensure all critical alarms have notification actions",
                    "Review alarm action configurations for completeness",
                    "Implement standardized alarm action policies",
                    "Document alarm governance requirements"
                ],
                cloudwatch_component="alarms"
            ))
        
        return recommendations
    
    def _generate_cost_recommendations(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cost-specific recommendations."""
        recommendations = []
        
        total_monitoring_costs = cost_data.get('total_monitoring_costs', 0)
        alarms_costs = cost_data.get('alarms_costs', {})
        dashboards_costs = cost_data.get('dashboards_costs', {})
        
        if total_monitoring_costs > 50:  # Arbitrary threshold for high costs
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="high",
                title="High CloudWatch monitoring costs detected",
                description=f"Monthly monitoring costs are ${total_monitoring_costs:.2f}. Consider optimization opportunities.",
                potential_savings=total_monitoring_costs * 0.3,  # Potential 30% savings
                implementation_effort="medium",
                action_items=[
                    "Review alarm and dashboard usage patterns",
                    "Identify and remove unused monitoring resources",
                    "Optimize high-resolution monitoring where standard resolution is sufficient",
                    "Consolidate dashboards to stay within free tier limits",
                    "Implement cost monitoring and alerting for CloudWatch usage"
                ],
                cloudwatch_component="alarms"
            ))
        
        return recommendations
    
    def _generate_efficiency_recommendations(self, efficiency_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate efficiency-based recommendations."""
        recommendations = []
        
        overall_score = efficiency_data.get('overall_efficiency_score', 0)
        optimization_opportunities = efficiency_data.get('optimization_opportunities', [])
        
        if overall_score < 80:  # Below 80% efficiency
            recommendations.append(self.create_recommendation(
                rec_type="efficiency",
                priority="medium",
                title="Improve overall monitoring efficiency",
                description=f"Overall efficiency score is {overall_score:.1f}%. Multiple optimization opportunities identified.",
                implementation_effort="medium",
                action_items=[
                    "Address unused and misconfigured alarms",
                    "Optimize dashboard usage and consolidation",
                    "Improve governance compliance",
                    "Implement regular monitoring hygiene practices",
                    "Set up automated monitoring optimization reviews"
                ],
                cloudwatch_component="alarms"
            ))
        
        # Add specific optimization opportunities as recommendations
        for opportunity in optimization_opportunities:
            recommendations.append(self.create_recommendation(
                rec_type="cost_optimization",
                priority="medium",
                title=opportunity.get('title', 'Optimization opportunity'),
                description=opportunity.get('description', 'Optimization opportunity identified'),
                potential_savings=opportunity.get('potential_savings'),
                implementation_effort=opportunity.get('effort', 'medium'),
                cloudwatch_component="alarms"
            ))
        
        return recommendations