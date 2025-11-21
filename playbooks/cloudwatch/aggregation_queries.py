"""
CloudWatch Aggregation Queries for cross-analysis insights and session-sql integration.

This module provides pre-defined SQL queries for analyzing CloudWatch optimization
data across multiple analyses and generating comprehensive insights.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logging_config import log_cloudwatch_operation

logger = logging.getLogger(__name__)


class CloudWatchAggregationQueries:
    """
    Provides SQL queries for cross-analysis of CloudWatch optimization data.
    
    This class contains pre-defined queries for:
    - Cross-analysis queries for logs, metrics, alarms, and dashboards
    - Cost correlation queries and optimization opportunity identification
    - Executive summary queries for comprehensive reporting
    """
    
    def __init__(self):
        """Initialize the aggregation queries class."""
        self.logger = logging.getLogger(__name__)
        
    def get_cost_correlation_queries(self) -> List[Dict[str, str]]:
        """
        Get queries for analyzing cost correlations across CloudWatch components.
        
        Returns:
            List of query definitions for cost correlation analysis
        """
        return [
            {
                'name': 'logs_vs_total_cost_correlation',
                'description': 'Analyze correlation between logs costs and total CloudWatch spend',
                'query': '''
                    SELECT 
                        'logs_cost_analysis' as analysis_type,
                        COALESCE(
                            JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly'), 0
                        ) as logs_monthly_cost,
                        COALESCE(
                            JSON_EXTRACT(data, '$.cost_breakdown.total_estimated_monthly'), 0
                        ) as total_monthly_cost,
                        CASE 
                            WHEN JSON_EXTRACT(data, '$.cost_breakdown.total_estimated_monthly') > 0 
                            THEN (JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') * 100.0) / 
                                 JSON_EXTRACT(data, '$.cost_breakdown.total_estimated_monthly')
                            ELSE 0 
                        END as logs_cost_percentage,
                        timestamp
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%general_spend%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                '''
            },
            {
                'name': 'high_cost_components_identification',
                'description': 'Identify CloudWatch components with highest cost impact',
                'query': '''
                    SELECT 
                        'cost_component_ranking' as analysis_type,
                        'logs' as component,
                        COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly'), 0) as monthly_cost,
                        'Logs ingestion, storage, and retention costs' as description
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%general_spend%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    
                    UNION ALL
                    
                    SELECT 
                        'cost_component_ranking' as analysis_type,
                        'metrics' as component,
                        COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly'), 0) as monthly_cost,
                        'Custom metrics, API requests, and detailed monitoring costs' as description
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%general_spend%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    
                    UNION ALL
                    
                    SELECT 
                        'cost_component_ranking' as analysis_type,
                        'alarms' as component,
                        COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly'), 0) as monthly_cost,
                        'Standard and high-resolution alarm costs' as description
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%general_spend%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    
                    ORDER BY monthly_cost DESC
                '''
            },
            {
                'name': 'optimization_opportunity_prioritization',
                'description': 'Prioritize optimization opportunities by potential cost savings',
                'query': '''
                    SELECT 
                        'optimization_priority' as analysis_type,
                        component,
                        monthly_cost,
                        CASE 
                            WHEN monthly_cost > 100 THEN 'Critical'
                            WHEN monthly_cost > 50 THEN 'High'
                            WHEN monthly_cost > 20 THEN 'Medium'
                            ELSE 'Low'
                        END as priority,
                        CASE 
                            WHEN component = 'logs' THEN 'Review retention policies, log group cleanup, ingestion optimization'
                            WHEN component = 'metrics' THEN 'Custom metrics optimization, detailed monitoring review'
                            WHEN component = 'alarms' THEN 'Remove unused alarms, optimize high-resolution alarms'
                            WHEN component = 'dashboards' THEN 'Dashboard consolidation, free tier optimization'
                            ELSE 'General optimization'
                        END as recommended_actions
                    FROM (
                        SELECT 
                            'logs' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly'), 0) as monthly_cost
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'metrics' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly'), 0) as monthly_cost
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'alarms' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly'), 0) as monthly_cost
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    )
                    ORDER BY monthly_cost DESC
                '''
            }
        ]
    
    def get_resource_relationship_queries(self) -> List[Dict[str, str]]:
        """
        Get queries for analyzing relationships between CloudWatch resources.
        
        Returns:
            List of query definitions for resource relationship analysis
        """
        return [
            {
                'name': 'unused_resources_correlation',
                'description': 'Find correlations between unused alarms and custom metrics',
                'query': '''
                    SELECT 
                        'unused_resources_analysis' as analysis_type,
                        alarms.unused_alarms_count,
                        metrics.custom_metrics_count,
                        CASE 
                            WHEN alarms.unused_alarms_count > 5 AND metrics.custom_metrics_count > 10 
                            THEN 'High cleanup potential - coordinate alarms and metrics cleanup'
                            WHEN alarms.unused_alarms_count > 0 OR metrics.custom_metrics_count > 10
                            THEN 'Medium cleanup potential - focus on primary waste area'
                            ELSE 'Low cleanup potential - resources appear well-managed'
                        END as cleanup_recommendation,
                        (alarms.unused_alarms_count * 0.10) + (CASE WHEN metrics.custom_metrics_count > 10 THEN (metrics.custom_metrics_count - 10) * 0.30 ELSE 0 END) as estimated_monthly_savings
                    FROM (
                        SELECT 
                            COALESCE(JSON_EXTRACT(data, '$.alarm_efficiency.unused_alarms_count'), 0) as unused_alarms_count
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%alarms_and_dashboards%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'alarms_and_dashboards'
                    ) alarms
                    CROSS JOIN (
                        SELECT 
                            COALESCE(JSON_EXTRACT(data, '$.metrics_configuration_analysis.metrics_analysis.custom_metrics_count'), 0) as custom_metrics_count
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%metrics_optimization%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'metrics_optimization'
                    ) metrics
                '''
            },
            {
                'name': 'log_groups_without_retention_analysis',
                'description': 'Analyze log groups without retention policies and their cost impact',
                'query': '''
                    SELECT 
                        'log_retention_analysis' as analysis_type,
                        COALESCE(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.total_log_groups'), 0) as total_log_groups,
                        COALESCE(JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')), 0) as log_groups_without_retention,
                        CASE 
                            WHEN JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')) > 0
                            THEN ROUND((JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')) * 100.0) / 
                                      JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.total_log_groups'), 2)
                            ELSE 0
                        END as percentage_without_retention,
                        CASE 
                            WHEN JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')) > 10
                            THEN 'Critical - High storage cost risk'
                            WHEN JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')) > 5
                            THEN 'High - Moderate storage cost risk'
                            WHEN JSON_LENGTH(JSON_EXTRACT(data, '$.log_groups_configuration_analysis.log_groups_analysis.without_retention_policy')) > 0
                            THEN 'Medium - Some storage cost risk'
                            ELSE 'Low - Good retention policy coverage'
                        END as risk_level
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%logs_optimization%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'logs_optimization'
                '''
            },
            {
                'name': 'dashboard_free_tier_utilization',
                'description': 'Analyze dashboard usage relative to AWS free tier limits',
                'query': '''
                    SELECT 
                        'dashboard_free_tier_analysis' as analysis_type,
                        COALESCE(JSON_EXTRACT(data, '$.dashboard_efficiency.total_dashboards'), 0) as total_dashboards,
                        COALESCE(JSON_EXTRACT(data, '$.dashboard_efficiency.free_tier_count'), 0) as free_tier_dashboards,
                        COALESCE(JSON_EXTRACT(data, '$.dashboard_efficiency.paid_dashboards_count'), 0) as paid_dashboards,
                        COALESCE(JSON_EXTRACT(data, '$.dashboard_efficiency.free_tier_utilization'), 0) as free_tier_utilization_percent,
                        CASE 
                            WHEN JSON_EXTRACT(data, '$.dashboard_efficiency.paid_dashboards_count') > 0
                            THEN JSON_EXTRACT(data, '$.dashboard_efficiency.paid_dashboards_count') * 3.0
                            ELSE 0
                        END as estimated_monthly_dashboard_cost,
                        CASE 
                            WHEN JSON_EXTRACT(data, '$.dashboard_efficiency.free_tier_utilization') < 100 AND JSON_EXTRACT(data, '$.dashboard_efficiency.paid_dashboards_count') > 0
                            THEN 'Optimize by consolidating dashboards to maximize free tier usage'
                            WHEN JSON_EXTRACT(data, '$.dashboard_efficiency.paid_dashboards_count') > 5
                            THEN 'Consider dashboard consolidation to reduce costs'
                            ELSE 'Dashboard usage appears optimized'
                        END as optimization_recommendation
                    FROM sqlite_master sm
                    JOIN (
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name LIKE '%alarms_and_dashboards%'
                        ORDER BY name DESC LIMIT 1
                    ) latest ON sm.name = latest.name
                    CROSS JOIN (SELECT * FROM latest.name) data
                    WHERE JSON_EXTRACT(data, '$.analysis_type') = 'alarms_and_dashboards'
                '''
            }
        ]
    
    def get_executive_summary_queries(self) -> List[Dict[str, str]]:
        """
        Get queries for generating executive summary reports.
        
        Returns:
            List of query definitions for executive summary generation
        """
        return [
            {
                'name': 'overall_optimization_summary',
                'description': 'High-level summary of all CloudWatch optimization opportunities',
                'query': '''
                    SELECT 
                        'executive_summary' as report_type,
                        COUNT(DISTINCT sm.name) as total_analyses_completed,
                        SUM(CASE WHEN JSON_EXTRACT(data, '$.status') = 'success' THEN 1 ELSE 0 END) as successful_analyses,
                        SUM(CASE WHEN JSON_EXTRACT(data, '$.cost_incurred') = 1 THEN 1 ELSE 0 END) as analyses_with_cost,
                        GROUP_CONCAT(DISTINCT JSON_EXTRACT(data, '$.primary_data_source')) as data_sources_used,
                        AVG(JSON_EXTRACT(data, '$.execution_time')) as avg_execution_time,
                        MAX(JSON_EXTRACT(data, '$.timestamp')) as latest_analysis_time
                    FROM sqlite_master sm
                    CROSS JOIN (SELECT * FROM sm.name) data
                    WHERE sm.type = 'table' 
                      AND sm.name LIKE '%cloudwatch%'
                      AND sm.name NOT LIKE '%summary%'
                      AND JSON_EXTRACT(data, '$.analysis_type') IS NOT NULL
                '''
            },
            {
                'name': 'top_cost_optimization_opportunities',
                'description': 'Top 5 cost optimization opportunities across all analyses',
                'query': '''
                    SELECT 
                        'top_opportunities' as report_type,
                        component,
                        monthly_cost,
                        priority,
                        recommended_actions,
                        RANK() OVER (ORDER BY monthly_cost DESC) as priority_rank
                    FROM (
                        SELECT 
                            'logs' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 100 THEN 'Critical'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 50 THEN 'High'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority,
                            'Optimize log retention policies and reduce log ingestion volume' as recommended_actions
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'metrics' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly') > 50 THEN 'High'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority,
                            'Reduce custom metrics and optimize detailed monitoring' as recommended_actions
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'alarms' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority,
                            'Remove unused alarms and optimize high-resolution alarms' as recommended_actions
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    )
                    WHERE monthly_cost > 0
                    ORDER BY monthly_cost DESC
                    LIMIT 5
                '''
            },
            {
                'name': 'cost_savings_potential_summary',
                'description': 'Summary of total potential cost savings across all components',
                'query': '''
                    SELECT 
                        'savings_potential' as report_type,
                        SUM(monthly_cost) as total_monthly_cost,
                        SUM(CASE WHEN priority IN ('Critical', 'High') THEN monthly_cost * 0.6 ELSE monthly_cost * 0.3 END) as estimated_savings_potential,
                        COUNT(*) as optimization_opportunities,
                        SUM(CASE WHEN priority = 'Critical' THEN 1 ELSE 0 END) as critical_opportunities,
                        SUM(CASE WHEN priority = 'High' THEN 1 ELSE 0 END) as high_opportunities,
                        SUM(CASE WHEN priority = 'Medium' THEN 1 ELSE 0 END) as medium_opportunities,
                        ROUND((SUM(CASE WHEN priority IN ('Critical', 'High') THEN monthly_cost * 0.6 ELSE monthly_cost * 0.3 END) * 100.0) / SUM(monthly_cost), 2) as savings_percentage
                    FROM (
                        SELECT 
                            'logs' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 100 THEN 'Critical'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 50 THEN 'High'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.logs_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'metrics' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly') > 50 THEN 'High'
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.metrics_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                        
                        UNION ALL
                        
                        SELECT 
                            'alarms' as component,
                            COALESCE(JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly'), 0) as monthly_cost,
                            CASE 
                                WHEN JSON_EXTRACT(data, '$.cost_breakdown.alarms_costs.estimated_monthly') > 20 THEN 'Medium'
                                ELSE 'Low'
                            END as priority
                        FROM sqlite_master sm
                        JOIN (
                            SELECT name FROM sqlite_master 
                            WHERE type='table' AND name LIKE '%general_spend%'
                            ORDER BY name DESC LIMIT 1
                        ) latest ON sm.name = latest.name
                        CROSS JOIN (SELECT * FROM latest.name) data
                        WHERE JSON_EXTRACT(data, '$.analysis_type') = 'general_spend'
                    )
                    WHERE monthly_cost > 0
                '''
            }
        ]
    
    def get_all_aggregation_queries(self) -> List[Dict[str, str]]:
        """
        Get all aggregation queries for comprehensive analysis.
        
        Returns:
            List of all available query definitions
        """
        all_queries = []
        all_queries.extend(self.get_cost_correlation_queries())
        all_queries.extend(self.get_resource_relationship_queries())
        all_queries.extend(self.get_executive_summary_queries())
        
        log_cloudwatch_operation(self.logger, "aggregation_queries_retrieved",
                               total_queries=len(all_queries))
        
        return all_queries
    
    def get_query_by_name(self, query_name: str) -> Optional[Dict[str, str]]:
        """
        Get a specific query by name.
        
        Args:
            query_name: Name of the query to retrieve
            
        Returns:
            Query definition or None if not found
        """
        all_queries = self.get_all_aggregation_queries()
        
        for query in all_queries:
            if query['name'] == query_name:
                return query
        
        return None
    
    def get_queries_by_category(self, category: str) -> List[Dict[str, str]]:
        """
        Get queries by category.
        
        Args:
            category: Category name ('cost_correlation', 'resource_relationship', 'executive_summary')
            
        Returns:
            List of queries in the specified category
        """
        category_map = {
            'cost_correlation': self.get_cost_correlation_queries,
            'resource_relationship': self.get_resource_relationship_queries,
            'executive_summary': self.get_executive_summary_queries
        }
        
        if category in category_map:
            return category_map[category]()
        else:
            return []