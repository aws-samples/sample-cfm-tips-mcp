"""
Integration tests for CloudWatch optimization orchestrator and service interactions.

Tests end-to-end workflows, service coordination, parallel execution,
and session management integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta, timezone
import json

from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
from playbooks.cloudwatch.analysis_engine import CloudWatchAnalysisEngine
from services.cloudwatch_service import CloudWatchService, CloudWatchOperationResult
from playbooks.cloudwatch.cost_controller import CostPreferences
from utils.service_orchestrator import ServiceOrchestrator
from utils.session_manager import SessionManager


@pytest.mark.integration
@pytest.mark.asyncio
class TestCloudWatchOrchestratorIntegration:
    """Integration tests for CloudWatch orchestrator."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager for testing."""
        manager = Mock(spec=SessionManager)
        manager.session_id = "test_session_123"
        manager.store_data = AsyncMock(return_value=True)
        manager.query_data = Mock(return_value=[])
        manager.get_stored_tables = Mock(return_value=["test_table_1", "test_table_2"])
        return manager
    
    @pytest.fixture
    def mock_service_orchestrator(self, mock_session_manager):
        """Mock service orchestrator for testing."""
        orchestrator = Mock(spec=ServiceOrchestrator)
        orchestrator.session_manager = mock_session_manager
        orchestrator.session_id = "test_session_123"  # Add session_id attribute
        orchestrator.get_stored_tables.return_value = ["test_table_1", "test_table_2"]
        orchestrator.execute_parallel_analysis = AsyncMock(return_value={
            "status": "success",
            "successful": 4,
            "total_tasks": 4,
            "results": {
                "general_spend": {"status": "success", "data": {"total_cost": 100.0}},
                "metrics_optimization": {"status": "success", "data": {"optimization_opportunities": 3}},
                "logs_optimization": {"status": "success", "data": {"retention_savings": 25.0}},
                "alarms_dashboards": {"status": "success", "data": {"unused_alarms": 5}}
            },
            "stored_tables": ["general_spend_results", "metrics_results", "logs_results", "alarms_results"]
        })
        return orchestrator
    
    @pytest.fixture
    def mock_analysis_engine(self):
        """Mock analysis engine for testing."""
        engine = Mock(spec=CloudWatchAnalysisEngine)
        
        # Mock individual analysis methods
        engine.run_analysis = AsyncMock(return_value={
            "status": "success",
            "analysis_type": "general_spend",
            "data": {"total_cost": 100.0, "optimization_opportunities": 3},
            "recommendations": [
                {
                    "type": "cost_optimization",
                    "priority": "high",
                    "title": "Optimize Log Retention",
                    "potential_savings": 25.0
                }
            ],
            "execution_time": 5.0,
            "cost_incurred": False,
            "cost_incurring_operations": []
        })
        
        # Mock comprehensive analysis method
        engine.run_comprehensive_analysis = AsyncMock(return_value={
            "status": "success",
            "analysis_type": "comprehensive",
            "successful_analyses": 4,
            "total_analyses": 4,
            "analysis_summary": {
                "total_analyses": 4,
                "successful_analyses": 4,
                "failed_analyses": 0
            },
            "results": {
                "general_spend": {"status": "success", "data": {"total_cost": 100.0}},
                "metrics_optimization": {"status": "success", "data": {"optimization_opportunities": 3}},
                "logs_optimization": {"status": "success", "data": {"retention_savings": 25.0}},
                "alarms_and_dashboards": {"status": "success", "data": {"unused_alarms": 5}}
            },
            "stored_tables": ["general_spend_results", "metrics_results", "logs_results", "alarms_results"],
            "execution_time": 15.0,
            "cost_incurred": False
        })
        
        return engine
    
    @pytest.fixture
    def orchestrator(self, mock_service_orchestrator, mock_analysis_engine):
        """Create CloudWatch orchestrator with mocked dependencies."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so_class, \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_ae_class:
            
            mock_so_class.return_value = mock_service_orchestrator
            mock_ae_class.return_value = mock_analysis_engine
            
            return CloudWatchOptimizationOrchestrator(
                region='us-east-1',
                session_id='test_session_123'
            )
    
    async def test_execute_single_analysis_success(self, orchestrator, mock_analysis_engine):
        """Test successful execution of single analysis."""
        result = await orchestrator.execute_analysis(
            analysis_type='general_spend',
            region='us-east-1',
            lookback_days=30
        )
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'general_spend'
        assert result['data']['total_cost'] == 100.0
        assert len(result['recommendations']) > 0
        assert result['cost_incurred'] is False
        
        # Verify analysis engine was called with correct parameters
        mock_analysis_engine.run_analysis.assert_called_once()
        call_args = mock_analysis_engine.run_analysis.call_args
        assert call_args[0][0] == 'general_spend'  # First positional arg
        assert call_args[1]['region'] == 'us-east-1'  # Keyword args
        assert call_args[1]['lookback_days'] == 30
    
    async def test_execute_single_analysis_with_cost_preferences(self, orchestrator, mock_analysis_engine):
        """Test single analysis with cost preferences."""
        cost_prefs = {
            'allow_cost_explorer': True,
            'allow_minimal_cost_metrics': True
        }
        
        # Mock analysis result with cost incurred
        mock_analysis_engine.run_analysis.return_value = {
            "status": "success",
            "analysis_type": "general_spend",
            "data": {"total_cost": 150.0},
            "recommendations": [],
            "execution_time": 8.0,
            "cost_incurred": True,
            "cost_incurring_operations": ["cost_explorer_analysis", "minimal_cost_metrics"]
        }
        
        result = await orchestrator.execute_analysis(
            analysis_type='general_spend',
            **cost_prefs
        )
        
        assert result['status'] == 'success'
        assert result['cost_incurred'] is True
        assert 'cost_explorer_analysis' in result['cost_incurring_operations']
        assert 'minimal_cost_metrics' in result['cost_incurring_operations']
    
    async def test_execute_comprehensive_analysis_success(self, orchestrator, mock_service_orchestrator):
        """Test successful comprehensive analysis execution."""
        result = await orchestrator.execute_comprehensive_analysis(
            region='us-east-1',
            lookback_days=30,
            parallel_execution=True
        )
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'comprehensive'
        assert result['successful_analyses'] == 4
        assert result['total_analyses'] == 4
        assert 'results' in result
        assert 'stored_tables' in result
        
        # Verify analysis engine was called for comprehensive analysis
        # Note: The orchestrator calls the analysis engine directly for comprehensive analysis
    
    async def test_execute_comprehensive_analysis_with_failures(self, orchestrator, mock_analysis_engine):
        """Test comprehensive analysis with some failures."""
        # Reset the mock and set up new behavior
        mock_analysis_engine.reset_mock()
        
        # Mock the comprehensive analysis method to return partial failure
        mock_analysis_engine.run_comprehensive_analysis = AsyncMock(return_value={
            "status": "partial",
            "analysis_type": "comprehensive",
            "successful_analyses": 3,
            "total_analyses": 4,
            "analysis_summary": {
                "total_analyses": 4,
                "successful_analyses": 3,
                "failed_analyses": 1
            },
            "results": {
                "general_spend": {"status": "success", "data": {"total_cost": 100.0}},
                "metrics_optimization": {"status": "success", "data": {"optimization_opportunities": 3}},
                "logs_optimization": {"status": "error", "error": "API timeout"},
                "alarms_and_dashboards": {"status": "success", "data": {"unused_alarms": 5}}
            },
            "stored_tables": ["general_spend_results", "metrics_results", "alarms_results"],
            "execution_time": 15.0,
            "cost_incurred": False
        })
        
        result = await orchestrator.execute_comprehensive_analysis()
        
        assert result['status'] == 'partial'
        assert result['successful_analyses'] == 3
        assert result['total_analyses'] == 4
        assert len(result['stored_tables']) == 3
    
    async def test_execute_analysis_with_timeout(self, orchestrator, mock_analysis_engine):
        """Test analysis execution with timeout."""
        # Mock timeout scenario
        mock_analysis_engine.run_analysis.side_effect = asyncio.TimeoutError("Analysis timed out")
        
        result = await orchestrator.execute_analysis(
            analysis_type='general_spend',
            timeout_seconds=5
        )
        
        assert result['status'] == 'error'
        assert 'timed out' in result['error_message'].lower()
    
    def test_validate_cost_preferences_valid(self, orchestrator):
        """Test cost preferences validation with valid input."""
        input_prefs = {
            'allow_cost_explorer': True,
            'allow_aws_config': 'false',
            'allow_cloudtrail': 1,
            'allow_minimal_cost_metrics': 0
        }
        
        result = orchestrator.validate_cost_preferences(**input_prefs)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['sanitized_preferences']['allow_cost_explorer'] is True
        assert result['sanitized_preferences']['allow_aws_config'] is False
        assert result['sanitized_preferences']['allow_cloudtrail'] is True
        assert result['sanitized_preferences']['allow_minimal_cost_metrics'] is False
    
    def test_validate_cost_preferences_invalid(self, orchestrator):
        """Test cost preferences validation with invalid input."""
        input_prefs = {
            'allow_cost_explorer': 'invalid_value',
            'unknown_preference': True
        }
        
        result = orchestrator.validate_cost_preferences(**input_prefs)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('Invalid boolean value' in error for error in result['errors'])
    
    def test_get_cost_estimate_no_paid_features(self, orchestrator):
        """Test cost estimation with no paid features."""
        analysis_scope = {
            'lookback_days': 30,
            'log_group_names': ['group1', 'group2']
        }
        cost_prefs = CostPreferences()  # All False
        
        result = orchestrator.get_cost_estimate(analysis_scope, cost_prefs)
        
        assert result['total_estimated_cost'] == 0.0
        assert result['cost_breakdown']['free_operations'] > 0
        assert result['cost_breakdown']['paid_operations'] == 0.0
        assert len(result['enabled_operations']) > 0
        assert len(result['disabled_operations']) > 0
    
    def test_get_cost_estimate_with_paid_features(self, orchestrator):
        """Test cost estimation with paid features enabled."""
        analysis_scope = {
            'lookback_days': 60,
            'log_group_names': ['group1', 'group2', 'group3']
        }
        cost_prefs = CostPreferences(
            allow_cost_explorer=True,
            allow_minimal_cost_metrics=True
        )
        
        result = orchestrator.get_cost_estimate(analysis_scope, cost_prefs)
        
        assert result['total_estimated_cost'] > 0.0
        assert result['cost_breakdown']['paid_operations'] > 0.0
        assert 'cost_explorer_analysis' in result['enabled_operations']
        assert 'minimal_cost_metrics' in result['enabled_operations']
        assert 'aws_config_compliance' in result['disabled_operations']
    
    def test_get_analysis_results_query(self, orchestrator, mock_service_orchestrator):
        """Test querying stored analysis results."""
        # Mock query results
        mock_service_orchestrator.query_session_data.return_value = [
            {
                'analysis_type': 'general_spend',
                'total_cost': 100.0,
                'timestamp': '2024-01-01T00:00:00Z'
            },
            {
                'analysis_type': 'logs_optimization',
                'potential_savings': 25.0,
                'timestamp': '2024-01-01T00:05:00Z'
            }
        ]
        
        query = "SELECT analysis_type, total_cost FROM results WHERE analysis_type = 'general_spend'"
        results = orchestrator.get_analysis_results(query)
        
        assert len(results) == 2
        assert results[0]['analysis_type'] == 'general_spend'
        
        # Verify session manager was called
        mock_service_orchestrator.query_session_data.assert_called_once_with(query)
    
    def test_get_stored_tables(self, orchestrator, mock_service_orchestrator):
        """Test getting list of stored tables."""
        tables = orchestrator.get_stored_tables()
        
        assert len(tables) == 2
        assert 'test_table_1' in tables
        assert 'test_table_2' in tables
        
        mock_service_orchestrator.get_stored_tables.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
class TestCloudWatchServiceIntegration:
    """Integration tests for CloudWatch service interactions."""
    
    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for integration testing."""
        with patch('services.cloudwatch_service.boto3') as mock_boto3:
            mock_cloudwatch = MagicMock()
            mock_logs = MagicMock()
            mock_ce = MagicMock()
            mock_pricing = MagicMock()
            
            mock_boto3.client.side_effect = lambda service, **kwargs: {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }[service]
            
            yield {
                'cloudwatch': mock_cloudwatch,
                'logs': mock_logs,
                'ce': mock_ce,
                'pricing': mock_pricing
            }
    
    @pytest.fixture
    def cloudwatch_service(self, mock_boto3_clients):
        """Create CloudWatch service for integration testing."""
        return CloudWatchService(region='us-east-1')
    
    async def test_service_coordination_workflow(self, cloudwatch_service, mock_boto3_clients):
        """Test coordinated service operations workflow."""
        # Mock paginator responses for different operations
        mock_paginator = MagicMock()
        
        # Mock list_metrics response
        mock_paginator.paginate.return_value = [
            {
                'Metrics': [
                    {'Namespace': 'AWS/EC2', 'MetricName': 'CPUUtilization'},
                    {'Namespace': 'Custom/App', 'MetricName': 'RequestCount'}
                ]
            }
        ]
        mock_boto3_clients['cloudwatch'].get_paginator.return_value = mock_paginator
        
        # Execute coordinated operations
        metrics_result = await cloudwatch_service.list_metrics()
        alarms_result = await cloudwatch_service.describe_alarms()
        dashboards_result = await cloudwatch_service.list_dashboards()
        log_groups_result = await cloudwatch_service.describe_log_groups()
        
        # Verify all operations succeeded
        assert metrics_result.success is True
        assert alarms_result.success is True
        assert dashboards_result.success is True
        assert log_groups_result.success is True
        
        # Verify no cost was incurred
        assert not any([
            metrics_result.cost_incurred,
            alarms_result.cost_incurred,
            dashboards_result.cost_incurred,
            log_groups_result.cost_incurred
        ])
    
    async def test_service_error_handling_and_fallback(self, cloudwatch_service, mock_boto3_clients):
        """Test service error handling and fallback mechanisms."""
        # Mock first operation to fail, second to succeed
        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = [
            Exception("Network error"),  # First call fails
            [{'Metrics': []}]  # Second call succeeds (retry)
        ]
        mock_boto3_clients['cloudwatch'].get_paginator.return_value = mock_paginator
        
        result = await cloudwatch_service.list_metrics()
        
        # Should succeed after retry
        assert result.success is True
        assert mock_paginator.paginate.call_count >= 2
    
    async def test_cost_constraint_enforcement(self, cloudwatch_service, mock_boto3_clients):
        """Test that cost constraints are properly enforced."""
        # Try to execute paid operation without permission
        result = await cloudwatch_service.get_log_group_incoming_bytes()
        
        # Should fail due to cost constraints
        assert result.success is False
        assert 'not allowed' in result.error_message.lower()
        assert result.cost_incurred is False
        
        # Enable minimal cost metrics and try again
        cloudwatch_service.update_cost_preferences({'allow_minimal_cost_metrics': True})
        
        # Mock successful response
        mock_boto3_clients['cloudwatch'].get_metric_statistics.return_value = {
            'Datapoints': [{'Timestamp': datetime.now(timezone.utc), 'Sum': 1000000.0}]
        }
        
        result = await cloudwatch_service.get_log_group_incoming_bytes()
        
        # Should succeed with cost incurred
        assert result.success is True
        assert result.cost_incurred is True
        assert result.operation_type == 'paid'
    
    async def test_parallel_operation_execution(self, cloudwatch_service, mock_boto3_clients):
        """Test parallel execution of multiple operations."""
        # Mock responses for all operations
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                'Metrics': [],
                'MetricAlarms': [],
                'CompositeAlarms': [],
                'DashboardEntries': [],
                'logGroups': []
            }
        ]
        mock_boto3_clients['cloudwatch'].get_paginator.return_value = mock_paginator
        mock_boto3_clients['logs'].get_paginator.return_value = mock_paginator
        
        # Execute operations in parallel
        tasks = [
            cloudwatch_service.list_metrics(),
            cloudwatch_service.describe_alarms(),
            cloudwatch_service.list_dashboards(),
            cloudwatch_service.describe_log_groups()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        assert len(results) == 4
        for result in results:
            assert not isinstance(result, Exception)
            assert result.success is True
    
    def test_service_statistics_tracking(self, cloudwatch_service):
        """Test that service statistics are properly tracked."""
        # Initial statistics should be zero
        stats = cloudwatch_service.get_service_statistics()
        assert stats['total_operations'] == 0
        assert stats['total_execution_time'] == 0.0
        assert stats['cost_incurred'] is False
        
        # Simulate some operations
        cloudwatch_service.operation_count = 5
        cloudwatch_service.total_execution_time = 15.0
        cloudwatch_service.cost_incurring_operations = ['minimal_cost_metrics']
        
        stats = cloudwatch_service.get_service_statistics()
        assert stats['total_operations'] == 5
        assert stats['total_execution_time'] == 15.0
        assert stats['average_execution_time'] == 3.0
        assert stats['cost_incurred'] is True
        
        # Reset statistics
        cloudwatch_service.reset_statistics()
        stats = cloudwatch_service.get_service_statistics()
        assert stats['total_operations'] == 0
        assert stats['cost_incurred'] is False


@pytest.mark.integration
@pytest.mark.asyncio
class TestAnalysisEngineIntegration:
    """Integration tests for analysis engine coordination."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock all services for analysis engine testing."""
        services = {}
        
        # Mock CloudWatch service
        cloudwatch_service = Mock()
        cloudwatch_service.list_metrics = AsyncMock(return_value=CloudWatchOperationResult(
            success=True, data={'metrics': [], 'total_count': 0}
        ))
        cloudwatch_service.describe_alarms = AsyncMock(return_value=CloudWatchOperationResult(
            success=True, data={'alarms': [], 'total_count': 0, 'analysis': {}}
        ))
        cloudwatch_service.list_dashboards = AsyncMock(return_value=CloudWatchOperationResult(
            success=True, data={'dashboards': [], 'total_count': 0, 'analysis': {}}
        ))
        cloudwatch_service.describe_log_groups = AsyncMock(return_value=CloudWatchOperationResult(
            success=True, data={'log_groups': [], 'total_count': 0, 'analysis': {}}
        ))
        services['cloudwatch_service'] = cloudwatch_service
        
        # Mock other services
        services['cost_explorer_service'] = Mock()
        services['config_service'] = Mock()
        services['metrics_service'] = Mock()
        services['pricing_service'] = Mock()
        services['performance_monitor'] = Mock()
        services['memory_manager'] = Mock()
        
        return services
    
    @pytest.fixture
    def analysis_engine(self, mock_services):
        """Create analysis engine with mocked services."""
        with patch('services.cloudwatch_service.CloudWatchService') as mock_cw_service, \
             patch('services.cloudwatch_pricing.CloudWatchPricing') as mock_pricing:
            
            mock_cw_service.return_value = mock_services['cloudwatch_service']
            mock_pricing.return_value = mock_services['pricing_service']
            
            return CloudWatchAnalysisEngine(region='us-east-1')
    
    async def test_analysis_engine_initialization(self, analysis_engine):
        """Test analysis engine initialization and analyzer registration."""
        assert analysis_engine.region == 'us-east-1'
        assert len(analysis_engine.analyzers) == 4  # Four analyzer types
        
        expected_analyzers = [
            'general_spend', 'metrics_optimization', 
            'logs_optimization', 'alarms_and_dashboards'
        ]
        
        for analyzer_name in expected_analyzers:
            assert analyzer_name in analysis_engine.analyzers
    
    async def test_run_single_analysis(self, analysis_engine, mock_services):
        """Test running a single analysis through the engine."""
        # Mock pricing service response
        mock_services['pricing_service'].get_logs_pricing.return_value = {
            'status': 'success',
            'logs_pricing': {'ingestion_per_gb': 0.50}
        }
        
        result = await analysis_engine.run_analysis(
            'general_spend',
            region='us-east-1',
            lookback_days=30
        )
        
        assert result['status'] == 'success'
        assert result['analysis_type'] == 'general_spend'
        assert 'data' in result
        assert 'recommendations' in result
        assert result['cost_incurred'] is False
    
    async def test_run_analysis_with_invalid_type(self, analysis_engine):
        """Test running analysis with invalid analysis type."""
        result = await analysis_engine.run_analysis('invalid_analysis_type')
        
        assert result['status'] == 'error'
        assert 'Unknown analysis type' in result['error_message']
    
    async def test_analyzer_coordination(self, analysis_engine, mock_services):
        """Test coordination between different analyzers."""
        # Mock responses for different analyzers
        mock_services['pricing_service'].get_logs_pricing.return_value = {
            'status': 'success', 'logs_pricing': {}
        }
        mock_services['pricing_service'].get_metrics_pricing.return_value = {
            'status': 'success', 'metrics_pricing': {}
        }
        mock_services['pricing_service'].get_alarms_pricing.return_value = {
            'status': 'success', 'alarms_pricing': {}
        }
        
        # Run multiple analyses
        results = []
        for analysis_type in ['general_spend', 'metrics_optimization', 'logs_optimization']:
            result = await analysis_engine.run_analysis(analysis_type)
            results.append(result)
        
        # All should succeed
        assert all(result['status'] == 'success' for result in results)
        
        # Each should have different analysis types
        analysis_types = [result['analysis_type'] for result in results]
        assert len(set(analysis_types)) == 3  # All unique


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests for performance monitoring and optimization."""
    
    @pytest.fixture
    def performance_test_orchestrator(self):
        """Create orchestrator for performance testing."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator'), \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine'):
            return CloudWatchOptimizationOrchestrator(region='us-east-1')
    
    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, performance_test_orchestrator, performance_tracker):
        """Test performance of parallel execution."""
        # Mock analysis engine for performance testing
        mock_engine = Mock()
        
        # Mock the comprehensive analysis method
        async def mock_comprehensive_analysis(**kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "status": "success",
                "analysis_type": "comprehensive",
                "successful_analyses": 4,
                "total_analyses": 4,
                "analysis_summary": {
                    "total_analyses": 4,
                    "successful_analyses": 4,
                    "failed_analyses": 0
                },
                "results": {
                    "general_spend": {"status": "success", "data": {"total_cost": 100.0}},
                    "metrics_optimization": {"status": "success", "data": {}},
                    "logs_optimization": {"status": "success", "data": {}},
                    "alarms_and_dashboards": {"status": "success", "data": {}}
                },
                "stored_tables": ["general_spend_results", "metrics_results", "logs_results", "alarms_results"],
                "execution_time": 0.1,
                "cost_incurred": False
            }
        
        mock_engine.run_comprehensive_analysis = AsyncMock(side_effect=mock_comprehensive_analysis)
        performance_test_orchestrator.analysis_engine = mock_engine
        
        # Test parallel execution performance
        performance_tracker.start_timer('comprehensive_analysis')
        
        result = await performance_test_orchestrator.execute_comprehensive_analysis()
        
        execution_time = performance_tracker.end_timer('comprehensive_analysis')
        
        assert result['status'] == 'success'
        assert execution_time < 1.0  # Should complete quickly with mocked operations
        
        performance_tracker.assert_performance('comprehensive_analysis', 1.0)
    
    @pytest.mark.asyncio
    async def test_timeout_handling_performance(self, performance_test_orchestrator):
        """Test timeout handling performance."""
        # Mock analysis engine with slow operation
        mock_engine = Mock()
        
        async def slow_analysis(*args, **kwargs):
            await asyncio.sleep(2.0)  # Simulate slow operation
            return {"status": "success"}
        
        mock_engine.run_analysis = slow_analysis
        performance_test_orchestrator.analysis_engine = mock_engine
        
        # Test with short timeout
        start_time = datetime.now()
        
        result = await performance_test_orchestrator.execute_analysis(
            'general_spend',
            timeout_seconds=0.5
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should timeout quickly
        assert result['status'] == 'error'
        assert 'timed out' in result['error_message'].lower()
        assert execution_time < 1.0  # Should not wait for full 2 seconds


if __name__ == "__main__":
    pytest.main([__file__])