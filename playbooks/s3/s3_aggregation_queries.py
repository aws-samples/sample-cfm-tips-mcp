"""
S3 Aggregation Queries for Cross-Analysis Insights

Provides predefined SQL queries for aggregating S3 optimization analysis results
stored in session databases to generate comprehensive insights and recommendations.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class S3AggregationQueries:
    """
    Provides predefined SQL queries for S3 optimization result aggregation.
    
    These queries work with the structured data stored by S3OptimizationOrchestrator
    to provide cross-analysis insights and comprehensive reporting.
    """
    
    @staticmethod
    def get_all_recommendations_by_priority() -> Dict[str, str]:
        """
        Query to get all recommendations grouped by priority across all analyses.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "recommendations_by_priority",
            "query": """
                SELECT 
                    priority,
                    analysis_type,
                    title,
                    description,
                    potential_savings,
                    implementation_effort,
                    COUNT(*) as recommendation_count,
                    SUM(potential_savings) as total_potential_savings
                FROM (
                    SELECT * FROM sqlite_master WHERE type='table' AND name LIKE 's3_%'
                ) tables
                JOIN (
                    SELECT name as table_name FROM sqlite_master WHERE type='table' AND name LIKE 's3_%'
                ) table_list
                CROSS JOIN (
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_general_spend_* 
                    WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_storage_class_*
                    WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_archive_optimization_*
                    WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_api_cost_*
                    WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_multipart_cleanup_*
                    WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        priority,
                        analysis_type,
                        title,
                        description,
                        potential_savings,
                        implementation_effort
                    FROM s3_governance_*
                    WHERE record_type = 'recommendation'
                ) recommendations
                GROUP BY priority, analysis_type
                ORDER BY 
                    CASE priority 
                        WHEN 'high' THEN 3 
                        WHEN 'medium' THEN 2 
                        WHEN 'low' THEN 1 
                        ELSE 0 
                    END DESC,
                    total_potential_savings DESC
            """
        }
    
    @staticmethod
    def get_top_cost_optimization_opportunities() -> Dict[str, str]:
        """
        Query to get top cost optimization opportunities across all analyses.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "top_cost_opportunities",
            "query": """
                SELECT 
                    title,
                    description,
                    potential_savings,
                    source_analysis,
                    implementation_effort,
                    priority,
                    rank
                FROM s3_comprehensive_*
                WHERE record_type = 'optimization_opportunity'
                ORDER BY potential_savings DESC, rank ASC
                LIMIT 10
            """
        }
    
    @staticmethod
    def get_analysis_execution_summary() -> Dict[str, str]:
        """
        Query to get execution summary across all analyses.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "analysis_execution_summary",
            "query": """
                SELECT 
                    analysis_type,
                    status,
                    execution_time,
                    recommendations_count,
                    data_sources,
                    timestamp
                FROM (
                    SELECT 
                        analysis_type,
                        status,
                        execution_time,
                        recommendations_count,
                        data_sources,
                        timestamp,
                        ROW_NUMBER() OVER (PARTITION BY analysis_type ORDER BY timestamp DESC) as rn
                    FROM (
                        SELECT * FROM sqlite_master WHERE type='table' AND name LIKE 's3_%'
                    ) tables
                    JOIN (
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_general_spend_* WHERE record_type = 'metadata'
                        UNION ALL
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_storage_class_* WHERE record_type = 'metadata'
                        UNION ALL
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_archive_optimization_* WHERE record_type = 'metadata'
                        UNION ALL
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_api_cost_* WHERE record_type = 'metadata'
                        UNION ALL
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_multipart_cleanup_* WHERE record_type = 'metadata'
                        UNION ALL
                        SELECT 
                            analysis_type,
                            status,
                            execution_time,
                            recommendations_count,
                            data_sources,
                            timestamp
                        FROM s3_governance_* WHERE record_type = 'metadata'
                    ) metadata
                )
                WHERE rn = 1
                ORDER BY 
                    CASE status 
                        WHEN 'success' THEN 1 
                        WHEN 'error' THEN 2 
                        ELSE 3 
                    END,
                    execution_time ASC
            """
        }
    
    @staticmethod
    def get_cross_analysis_insights() -> Dict[str, str]:
        """
        Query to get cross-analysis insights from comprehensive results.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "cross_analysis_insights",
            "query": """
                SELECT 
                    insight_type,
                    title,
                    description,
                    recommendation,
                    analyses_involved,
                    timestamp
                FROM s3_comprehensive_*
                WHERE record_type = 'cross_analysis_insight'
                ORDER BY timestamp DESC
            """
        }
    
    @staticmethod
    def get_total_potential_savings_by_analysis() -> Dict[str, str]:
        """
        Query to calculate total potential savings by analysis type.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "total_savings_by_analysis",
            "query": """
                SELECT 
                    analysis_type,
                    COUNT(*) as recommendation_count,
                    SUM(CASE WHEN potential_savings > 0 THEN potential_savings ELSE 0 END) as total_potential_savings,
                    AVG(CASE WHEN potential_savings > 0 THEN potential_savings ELSE NULL END) as avg_potential_savings,
                    MAX(potential_savings) as max_potential_savings,
                    COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority_count,
                    COUNT(CASE WHEN priority = 'medium' THEN 1 END) as medium_priority_count,
                    COUNT(CASE WHEN priority = 'low' THEN 1 END) as low_priority_count
                FROM (
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_general_spend_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_storage_class_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_archive_optimization_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_api_cost_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_multipart_cleanup_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        analysis_type,
                        potential_savings,
                        priority
                    FROM s3_governance_* WHERE record_type = 'recommendation'
                ) all_recommendations
                GROUP BY analysis_type
                ORDER BY total_potential_savings DESC
            """
        }
    
    @staticmethod
    def get_implementation_effort_analysis() -> Dict[str, str]:
        """
        Query to analyze recommendations by implementation effort.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "implementation_effort_analysis",
            "query": """
                SELECT 
                    implementation_effort,
                    priority,
                    COUNT(*) as recommendation_count,
                    SUM(CASE WHEN potential_savings > 0 THEN potential_savings ELSE 0 END) as total_potential_savings,
                    AVG(CASE WHEN potential_savings > 0 THEN potential_savings ELSE NULL END) as avg_potential_savings
                FROM (
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_general_spend_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_storage_class_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_archive_optimization_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_api_cost_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_multipart_cleanup_* WHERE record_type = 'recommendation'
                    UNION ALL
                    SELECT 
                        implementation_effort,
                        priority,
                        potential_savings
                    FROM s3_governance_* WHERE record_type = 'recommendation'
                ) all_recommendations
                GROUP BY implementation_effort, priority
                ORDER BY 
                    CASE implementation_effort 
                        WHEN 'low' THEN 1 
                        WHEN 'medium' THEN 2 
                        WHEN 'high' THEN 3 
                        ELSE 4 
                    END,
                    CASE priority 
                        WHEN 'high' THEN 1 
                        WHEN 'medium' THEN 2 
                        WHEN 'low' THEN 3 
                        ELSE 4 
                    END
            """
        }
    
    @staticmethod
    def get_comprehensive_analysis_summary() -> Dict[str, str]:
        """
        Query to get comprehensive analysis summary from latest comprehensive run.
        
        Returns:
            Dictionary with query name and SQL
        """
        return {
            "name": "comprehensive_analysis_summary",
            "query": """
                SELECT 
                    total_analyses,
                    successful_analyses,
                    failed_analyses,
                    total_potential_savings,
                    aggregated_at
                FROM s3_comprehensive_*
                WHERE record_type = 'comprehensive_metadata'
                ORDER BY aggregated_at DESC
                LIMIT 1
            """
        }
    
    @staticmethod
    def get_all_standard_queries() -> List[Dict[str, str]]:
        """
        Get all standard aggregation queries for S3 optimization results.
        
        Returns:
            List of query dictionaries
        """
        return [
            S3AggregationQueries.get_all_recommendations_by_priority(),
            S3AggregationQueries.get_top_cost_optimization_opportunities(),
            S3AggregationQueries.get_analysis_execution_summary(),
            S3AggregationQueries.get_cross_analysis_insights(),
            S3AggregationQueries.get_total_potential_savings_by_analysis(),
            S3AggregationQueries.get_implementation_effort_analysis(),
            S3AggregationQueries.get_comprehensive_analysis_summary()
        ]
    
    @staticmethod
    def create_custom_query(query_name: str, 
                          analysis_types: List[str] = None,
                          priority_filter: str = None,
                          min_savings: float = None) -> Dict[str, str]:
        """
        Create a custom aggregation query with filters.
        
        Args:
            query_name: Name for the custom query
            analysis_types: List of analysis types to include
            priority_filter: Priority level to filter by ('high', 'medium', 'low')
            min_savings: Minimum potential savings to include
            
        Returns:
            Dictionary with query name and SQL
        """
        # Base query for recommendations
        base_query = """
            SELECT 
                analysis_type,
                rec_type,
                priority,
                title,
                description,
                potential_savings,
                implementation_effort,
                timestamp
            FROM (
        """
        
        # Add UNION ALL clauses for each analysis type
        union_clauses = []
        if not analysis_types:
            analysis_types = ['general_spend', 'storage_class', 'archive_optimization', 
                            'api_cost', 'multipart_cleanup', 'governance']
        
        for analysis_type in analysis_types:
            union_clauses.append(f"""
                SELECT 
                    analysis_type,
                    rec_type,
                    priority,
                    title,
                    description,
                    potential_savings,
                    implementation_effort,
                    timestamp
                FROM s3_{analysis_type}_* 
                WHERE record_type = 'recommendation'
            """)
        
        query = base_query + " UNION ALL ".join(union_clauses) + "\n) all_recommendations"
        
        # Add WHERE clauses for filters
        where_clauses = []
        
        if priority_filter:
            where_clauses.append(f"priority = '{priority_filter}'")
        
        if min_savings is not None:
            where_clauses.append(f"potential_savings >= {min_savings}")
        
        if where_clauses:
            query += "\nWHERE " + " AND ".join(where_clauses)
        
        query += "\nORDER BY potential_savings DESC, timestamp DESC"
        
        return {
            "name": query_name,
            "query": query
        }


def get_dynamic_table_query(base_query: str) -> str:
    """
    Convert a static query to work with dynamic table names.
    
    This function helps handle the dynamic nature of session table names
    by using SQLite's metadata tables to find matching tables.
    
    Args:
        base_query: Base SQL query with placeholder table names
        
    Returns:
        Modified query that works with dynamic table names
    """
    # This is a simplified version - in practice, you might need more sophisticated
    # table name resolution based on the session's actual table names
    return base_query


class S3QueryExecutor:
    """
    Executes S3 aggregation queries against session databases.
    """
    
    def __init__(self, service_orchestrator):
        """
        Initialize query executor.
        
        Args:
            service_orchestrator: ServiceOrchestrator instance for query execution
        """
        self.service_orchestrator = service_orchestrator
        self.logger = logging.getLogger(__name__)
    
    def execute_standard_queries(self) -> Dict[str, Any]:
        """
        Execute all standard aggregation queries.
        
        Returns:
            Dictionary containing results from all standard queries
        """
        try:
            queries = S3AggregationQueries.get_all_standard_queries()
            return self.service_orchestrator.aggregate_results(queries)
        except Exception as e:
            self.logger.error(f"Error executing standard queries: {str(e)}")
            return {"error": str(e)}
    
    def execute_custom_query(self, query_def: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Execute a custom aggregation query.
        
        Args:
            query_def: Query definition with 'name' and 'query' keys
            
        Returns:
            List of query results
        """
        try:
            return self.service_orchestrator.query_session_data(query_def["query"])
        except Exception as e:
            self.logger.error(f"Error executing custom query '{query_def.get('name', 'unknown')}': {str(e)}")
            return []
    
    def get_available_tables(self) -> List[str]:
        """
        Get list of available tables in the session.
        
        Returns:
            List of table names
        """
        try:
            return self.service_orchestrator.get_stored_tables()
        except Exception as e:
            self.logger.error(f"Error getting available tables: {str(e)}")
            return []