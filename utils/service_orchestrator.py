"""
Service Orchestrator for CFM Tips MCP Server

Coordinates parallel execution of AWS service calls with session storage.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .session_manager import get_session_manager
from .parallel_executor import get_parallel_executor, create_task, TaskResult

logger = logging.getLogger(__name__)

class ServiceOrchestrator:
    """Orchestrates AWS service calls with parallel execution and session storage."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_manager = get_session_manager()
        self.parallel_executor = get_parallel_executor()
        self.session_id = session_id or self.session_manager.create_session()
        
        logger.info(f"ServiceOrchestrator initialized with session {self.session_id}")
    
    def execute_parallel_analysis(self, 
                                 service_calls: List[Dict[str, Any]], 
                                 store_results: bool = True,
                                 timeout: float = 60.0) -> Dict[str, Any]:
        """
        Execute multiple AWS service calls in parallel and optionally store results.
        
        Args:
            service_calls: List of service call definitions
            store_results: Whether to store results in session database
            timeout: Maximum time to wait for all tasks
            
        Returns:
            Dictionary containing execution results and summary
        """
        start_time = time.time()
        
        # Create parallel tasks
        tasks = []
        for i, call_def in enumerate(service_calls):
            task_id = f"{call_def['service']}_{call_def['operation']}_{i}_{int(time.time())}"
            
            task = create_task(
                task_id=task_id,
                service=call_def['service'],
                operation=call_def['operation'],
                function=call_def['function'],
                args=call_def.get('args', ()),
                kwargs=call_def.get('kwargs', {}),
                timeout=call_def.get('timeout', 30.0),
                priority=call_def.get('priority', 1)
            )
            tasks.append(task)
        
        # Submit tasks for parallel execution
        task_ids = self.parallel_executor.submit_batch(tasks)
        
        # Wait for completion
        results = self.parallel_executor.wait_for_tasks(task_ids, timeout)
        
        # Process results
        execution_summary = {
            'total_tasks': len(task_ids),
            'successful': 0,
            'failed': 0,
            'timeout': 0,
            'total_execution_time': time.time() - start_time,
            'results': {}
        }
        
        stored_tables = []
        
        for task_id, result in results.items():
            execution_summary['results'][task_id] = {
                'service': result.service,
                'operation': result.operation,
                'status': result.status,
                'execution_time': result.execution_time,
                'error': result.error
            }
            
            # Update counters
            if result.status == 'success':
                execution_summary['successful'] += 1
                
                # Store successful results in session database
                if store_results and result.data:
                    # Create a safe table name by removing special characters
                    safe_service = ''.join(c for c in result.service if c.isalnum() or c == '_')
                    safe_operation = ''.join(c for c in result.operation if c.isalnum() or c == '_')
                    table_name = f"{safe_service}_{safe_operation}_{int(result.timestamp.timestamp())}"
                    
                    # Convert result data to list of dictionaries if needed
                    data_to_store = self._prepare_data_for_storage(result.data, result.service, result.operation)
                    
                    if data_to_store:
                        success = self.session_manager.store_data(
                            self.session_id, 
                            table_name, 
                            data_to_store
                        )
                        
                        if success:
                            stored_tables.append(table_name)
                            execution_summary['results'][task_id]['stored_table'] = table_name
                        else:
                            logger.warning(f"Failed to store data for task {task_id}")
                
            elif result.status == 'timeout':
                execution_summary['timeout'] += 1
            else:
                execution_summary['failed'] += 1
        
        execution_summary['stored_tables'] = stored_tables
        
        logger.info(f"Parallel analysis completed: {execution_summary['successful']}/{execution_summary['total_tasks']} successful")
        
        return execution_summary
    
    def _prepare_data_for_storage(self, data: Any, service: str, operation: str) -> List[Dict[str, Any]]:
        """Prepare data for storage in session database with consistent 'value' column format."""
        try:
            import json
            
            # Always store data in a consistent format with 'value' column containing JSON
            if isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], dict):
                    # Check if data contains lists of items (like underutilized_instances)
                    data_dict = data['data']
                    result = []
                    
                    # Look for common list fields that should be stored as individual records
                    list_fields = ['underutilized_instances', 'unused_volumes', 'idle_instances', 'underutilized_functions']
                    
                    for field in list_fields:
                        if field in data_dict and isinstance(data_dict[field], list):
                            # Store each item in the list as a separate record
                            for item in data_dict[field]:
                                result.append({
                                    'value': json.dumps(item),
                                    'service': service,
                                    'operation': operation,
                                    'item_type': field
                                })
                            break  # Only process the first matching field
                    
                    # If no list fields found, store the entire data dict
                    if not result:
                        result.append({
                            'value': json.dumps(data_dict),
                            'service': service,
                            'operation': operation
                        })
                    
                    return result
                    
                elif 'data' in data and isinstance(data['data'], list):
                    # Standard service response format - store each item as JSON in value column
                    result = []
                    for item in data['data']:
                        result.append({
                            'value': json.dumps(item),
                            'service': service,
                            'operation': operation
                        })
                    return result
                else:
                    # Store entire dictionary as JSON
                    return [{
                        'value': json.dumps(data),
                        'service': service,
                        'operation': operation
                    }]
            
            elif isinstance(data, list):
                # Store each list item as JSON in value column
                result = []
                for i, item in enumerate(data):
                    result.append({
                        'value': json.dumps(item),
                        'service': service,
                        'operation': operation,
                        'index': i
                    })
                return result
            
            else:
                # Single value - store as JSON
                return [{
                    'value': json.dumps(data),
                    'service': service,
                    'operation': operation
                }]
                
        except Exception as e:
            logger.error(f"Error preparing data for storage: {e}")
            return []
    
    def query_session_data(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SQL query on session data."""
        try:
            return self.session_manager.execute_query(self.session_id, query, params)
        except Exception as e:
            logger.error(f"Error executing session query: {e}")
            return []
    
    def get_stored_tables(self) -> List[str]:
        """Get list of tables stored in the session."""
        try:
            results = self.query_session_data(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            return [row['name'] for row in results]
        except Exception as e:
            logger.error(f"Error getting stored tables: {e}")
            return []
    
    def _fix_wildcard_query(self, query_sql: str) -> str:
        """Fix wildcard table names in SQL queries by replacing with actual table names."""
        try:
            # Get all table names in the session
            stored_tables = self.get_stored_tables()
            
            # Common wildcard patterns to replace
            wildcard_patterns = [
                'ec2_underutilized_instances_*',
                'ebs_underutilized_volumes_*', 
                'ebs_unused_volumes_*',
                'rds_underutilized_instances_*',
                'rds_idle_instances_*',
                'lambda_underutilized_functions_*',
                'lambda_unused_functions_*'
            ]
            
            fixed_query = query_sql
            
            for pattern in wildcard_patterns:
                if pattern in fixed_query:
                    # Find matching table names
                    prefix = pattern.replace('_*', '_')
                    matching_tables = [table for table in stored_tables if table.startswith(prefix)]
                    
                    if matching_tables:
                        # Use the first matching table (most recent)
                        actual_table = matching_tables[0]
                        fixed_query = fixed_query.replace(pattern, f'"{actual_table}"')
                        logger.debug(f"Replaced {pattern} with {actual_table}")
                    else:
                        # If no matching table, create a dummy query that returns empty results
                        logger.warning(f"No matching table found for pattern {pattern}")
                        # Replace the entire FROM clause with a dummy table
                        fixed_query = fixed_query.replace(
                            f"FROM {pattern}",
                            "FROM (SELECT NULL as value WHERE 1=0)"
                        )
            
            return fixed_query
            
        except Exception as e:
            logger.error(f"Error fixing wildcard query: {e}")
            return query_sql
    
    def aggregate_results(self, aggregation_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute aggregation queries across stored data."""
        aggregated_results = {}
        
        for query_def in aggregation_queries:
            query_name = query_def['name']
            query_sql = query_def['query']
            
            try:
                # Fix wildcard patterns in the query
                fixed_query = self._fix_wildcard_query(query_sql)
                
                results = self.query_session_data(fixed_query)
                aggregated_results[query_name] = {
                    'status': 'success',
                    'data': results,
                    'row_count': len(results)
                }
            except Exception as e:
                aggregated_results[query_name] = {
                    'status': 'error',
                    'error': str(e),
                    'data': []
                }
                logger.error(f"Error executing aggregation query '{query_name}': {e}")
        
        return aggregated_results
    
    def create_comprehensive_report(self, 
                                   service_calls: List[Dict[str, Any]], 
                                   aggregation_queries: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Create a comprehensive report with parallel execution and aggregation."""
        
        # Execute parallel analysis
        execution_results = self.execute_parallel_analysis(service_calls)
        
        # Execute aggregation queries if provided
        aggregated_data = {}
        if aggregation_queries:
            aggregated_data = self.aggregate_results(aggregation_queries)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'session_id': self.session_id,
                'generated_at': datetime.now().isoformat(),
                'execution_summary': execution_results,
                'stored_tables': self.get_stored_tables()
            },
            'service_results': execution_results['results'],
            'aggregated_analysis': aggregated_data,
            'recommendations': self._generate_recommendations(execution_results, aggregated_data)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                 execution_results: Dict[str, Any], 
                                 aggregated_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Analyze execution results for patterns
        successful_services = []
        failed_services = []
        
        for task_id, result in execution_results['results'].items():
            if result['status'] == 'success':
                successful_services.append(result['service'])
            else:
                failed_services.append(result['service'])
        
        # Generate recommendations based on successful analyses
        if successful_services:
            recommendations.append({
                'type': 'analysis_success',
                'priority': 'info',
                'title': 'Successful Service Analysis',
                'description': f"Successfully analyzed {len(set(successful_services))} AWS services",
                'services': list(set(successful_services))
            })
        
        # Generate recommendations for failed analyses
        if failed_services:
            recommendations.append({
                'type': 'analysis_failure',
                'priority': 'warning',
                'title': 'Failed Service Analysis',
                'description': f"Failed to analyze {len(set(failed_services))} AWS services",
                'services': list(set(failed_services)),
                'action': 'Review service permissions and configuration'
            })
        
        # Add aggregation-based recommendations
        for query_name, query_result in aggregated_data.items():
            if query_result['status'] == 'success' and query_result['row_count'] > 0:
                recommendations.append({
                    'type': 'data_insight',
                    'priority': 'medium',
                    'title': f'Data Available: {query_name}',
                    'description': f"Found {query_result['row_count']} records for analysis",
                    'query': query_name
                })
        
        return recommendations
    
    def cleanup_session(self):
        """Clean up the current session."""
        try:
            self.session_manager.close_session(self.session_id)
            logger.info(f"Cleaned up session {self.session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {self.session_id}: {e}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return self.session_manager.get_session_info(self.session_id)