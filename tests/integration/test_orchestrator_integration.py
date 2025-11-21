"""
Integration tests for S3OptimizationOrchestrator and service interactions.

Tests the complete integration between orchestrator, analyzers, and services
with mocked AWS services to ensure proper data flow and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator


@pytest.mark.integration
class TestOrchestratorServiceIntegration:
    """Test integration between orchestrator and services."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_services(self, mock_service_orchestrator,
                                                           mock_performance_monitor,
                                                           mock_memory_manager,
                                                           mock_timeout_handler,
                                                           mock_cache):
        """Test orchestrator initialization with all performance components."""
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor', return_value=mock_performance_monitor), \
             patch('core.s3_optimization_orchestrator.get_memory_manager', return_value=mock_memory_manager), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler', return_value=mock_timeout_handler), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache', return_value=mock_cache):
            
            orchestrator = S3OptimizationOrchestrator(
                region="us-east-1",
                session_id="test_session"
            )
        
        assert orchestrator.region == "us-east-1"
        assert orchestrator.session_id == "test_session"
        assert orchestrator.service_orchestrator == mock_service_orchestrator
        assert orchestrator.performance_monitor == mock_performance_monitor
        assert orchestrator.memory_manager == mock_memory_manager
        
        # Verify performance components are properly integrated
        mock_memory_manager.add_cache_reference.assert_called()
        mock_cache.set_performance_monitor.assert_called()
    
    @pytest.mark.asyncio
    async def test_single_analysis_execution_flow(self, mock_service_orchestrator,
                                                 mock_performance_monitor,
                                                 mock_memory_manager,
                                                 mock_timeout_handler,
                                                 mock_cache):
        """Test complete flow of single analysis execution."""
        # Setup mocks
        mock_analysis_engine = Mock()
        mock_analysis_engine.run_analysis = AsyncMock(return_value={
            "status": "success",
            "analysis_type": "general_spend",
            "data": {"total_cost": 100.0},
            "recommendations": [{"type": "cost_optimization", "priority": "high"}],
            "execution_time": 5.0
        })
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor', return_value=mock_performance_monitor), \
             patch('core.s3_optimization_orchestrator.get_memory_manager', return_value=mock_memory_manager), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler', return_value=mock_timeout_handler), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.S3AnalysisEngine', return_value=mock_analysis_engine):
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Mock cache miss
            mock_cache.get.return_value = None
            
            result = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1",
                lookback_days=30
            )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "general_spend"
        assert result["from_cache"] is False
        assert "orchestrator_execution_time" in result
        assert "session_id" in result
        
        # Verify performance monitoring integration
        mock_performance_monitor.start_analysis_monitoring.assert_called_once()
        mock_performance_monitor.end_analysis_monitoring.assert_called_once()
        mock_memory_manager.start_memory_tracking.assert_called_once()
        mock_memory_manager.stop_memory_tracking.assert_called_once()
        
        # Verify caching
        mock_cache.put.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_execution_flow(self, mock_service_orchestrator,
                                                       mock_performance_monitor,
                                                       mock_memory_manager,
                                                       mock_timeout_handler,
                                                       mock_cache):
        """Test complete flow of comprehensive analysis execution."""
        # Setup analysis engine mock
        mock_analysis_engine = Mock()
        mock_analysis_engine.get_available_analyses.return_value = [
            {"analysis_type": "general_spend", "priority": 1},
            {"analysis_type": "storage_class", "priority": 2},
            {"analysis_type": "archive_optimization", "priority": 3}
        ]
        mock_analysis_engine.create_parallel_analysis_tasks.return_value = [
            {"analysis_type": "general_spend", "params": {}},
            {"analysis_type": "storage_class", "params": {}},
            {"analysis_type": "archive_optimization", "params": {}}
        ]
        
        # Setup service orchestrator mock
        mock_service_orchestrator.execute_parallel_analysis.return_value = {
            "status": "success",
            "successful": 3,
            "total_tasks": 3,
            "results": {
                "general_spend": {"status": "success", "data": {"cost": 100}},
                "storage_class": {"status": "success", "data": {"optimization": 25}},
                "archive_optimization": {"status": "success", "data": {"savings": 50}}
            },
            "stored_tables": ["general_spend_results", "storage_class_results"]
        }
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor', return_value=mock_performance_monitor), \
             patch('core.s3_optimization_orchestrator.get_memory_manager', return_value=mock_memory_manager), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler', return_value=mock_timeout_handler), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.S3AnalysisEngine', return_value=mock_analysis_engine):
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Mock cache miss
            mock_cache.get.return_value = None
            
            # Mock aggregation methods
            orchestrator.aggregate_results_with_insights = Mock(return_value={
                "total_potential_savings": 75.0,
                "top_recommendations": []
            })
            orchestrator._execute_cross_analysis_queries = Mock(return_value={
                "cross_analysis_insights": []
            })
            
            result = await orchestrator.execute_comprehensive_analysis(
                region="us-east-1",
                lookback_days=30,
                store_results=True
            )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "comprehensive"
        assert result["from_cache"] is False
        assert "execution_summary" in result
        assert "aggregated_results" in result
        assert "analysis_metadata" in result
        
        # Verify parallel execution was called
        mock_service_orchestrator.execute_parallel_analysis.assert_called_once()
        
        # Verify performance monitoring
        mock_performance_monitor.start_analysis_monitoring.assert_called_once()
        mock_performance_monitor.end_analysis_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, mock_service_orchestrator,
                                    mock_performance_monitor,
                                    mock_memory_manager,
                                    mock_timeout_handler,
                                    mock_cache):
        """Test analysis execution with cache hit."""
        cached_result = {
            "status": "success",
            "analysis_type": "general_spend",
            "data": {"cached": True},
            "execution_time": 2.0
        }
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor', return_value=mock_performance_monitor), \
             patch('core.s3_optimization_orchestrator.get_memory_manager', return_value=mock_memory_manager), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler', return_value=mock_timeout_handler), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache', return_value=mock_cache):
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Mock cache hit
            mock_cache.get.return_value = cached_result
            
            result = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1"
            )
        
        assert result["status"] == "success"
        assert result["from_cache"] is True
        assert result["data"]["cached"] is True
        
        # Verify cache hit was recorded
        mock_performance_monitor.record_cache_hit.assert_called_once()
        mock_performance_monitor.record_cache_miss.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_service_orchestrator,
                                            mock_performance_monitor,
                                            mock_memory_manager,
                                            mock_timeout_handler,
                                            mock_cache):
        """Test error handling integration across components."""
        # Setup analysis engine to fail
        mock_analysis_engine = Mock()
        mock_analysis_engine.run_analysis = AsyncMock(side_effect=Exception("Analysis engine error"))
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor', return_value=mock_performance_monitor), \
             patch('core.s3_optimization_orchestrator.get_memory_manager', return_value=mock_memory_manager), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler', return_value=mock_timeout_handler), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache', return_value=mock_cache), \
             patch('core.s3_optimization_orchestrator.S3AnalysisEngine', return_value=mock_analysis_engine):
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Mock cache miss
            mock_cache.get.return_value = None
            
            result = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1"
            )
        
        assert result["status"] == "error"
        assert "Analysis engine error" in result["message"]
        assert "session_id" in result
        
        # Verify error monitoring
        mock_performance_monitor.end_analysis_monitoring.assert_called_with(
            mock_performance_monitor.start_analysis_monitoring.return_value,
            success=False,
            error_message="Analysis engine error"
        )
    
    def test_session_data_querying(self, mock_service_orchestrator):
        """Test session data querying integration."""
        mock_service_orchestrator.query_session_data.return_value = [
            {"table": "results", "count": 100},
            {"table": "recommendations", "count": 25}
        ]
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor'), \
             patch('core.s3_optimization_orchestrator.get_memory_manager'), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler'), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache'), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache'), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache'):
            
            orchestrator = S3OptimizationOrchestrator()
            
            results = orchestrator.get_analysis_results("SELECT * FROM results")
        
        assert len(results) == 2
        assert results[0]["table"] == "results"
        mock_service_orchestrator.query_session_data.assert_called_once_with("SELECT * FROM results")
    
    def test_stored_tables_retrieval(self, mock_service_orchestrator):
        """Test stored tables retrieval integration."""
        mock_service_orchestrator.get_stored_tables.return_value = [
            "general_spend_results",
            "storage_class_results",
            "recommendations"
        ]
        
        with patch('core.s3_optimization_orchestrator.ServiceOrchestrator', return_value=mock_service_orchestrator), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor'), \
             patch('core.s3_optimization_orchestrator.get_memory_manager'), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler'), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache'), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache'), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache'):
            
            orchestrator = S3OptimizationOrchestrator()
            
            tables = orchestrator.get_stored_tables()
        
        assert len(tables) == 3
        assert "general_spend_results" in tables
        mock_service_orchestrator.get_stored_tables.assert_called_once()


@pytest.mark.integration
class TestAnalyzerServiceIntegration:
    """Test integration between analyzers and services."""
    
    @pytest.mark.asyncio
    async def test_analyzer_with_all_services(self, mock_s3_service, mock_storage_lens_service, 
                                            mock_pricing_service):
        """Test analyzer integration with all required services."""
        from playbooks.s3.analyzers.general_spend_analyzer import GeneralSpendAnalyzer
        
        # Setup service mocks
        mock_storage_lens_service.get_storage_metrics = AsyncMock(return_value={
            "status": "success",
            "data": {"ConfigurationId": "test", "StorageMetricsEnabled": True}
        })
        
        mock_pricing_service.get_storage_pricing.return_value = {
            "status": "success",
            "storage_pricing": {"STANDARD": 0.023}
        }
        
        analyzer = GeneralSpendAnalyzer(
            s3_service=mock_s3_service,
            storage_lens_service=mock_storage_lens_service,
            pricing_service=mock_pricing_service
        )
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            result = await analyzer.analyze(region="us-east-1", lookback_days=30)
        
        assert result["status"] == "success"
        assert "data_sources" in result
        
        # Verify services were called
        mock_storage_lens_service.get_storage_metrics.assert_called()
        mock_pricing_service.get_storage_pricing.assert_called()
    
    @pytest.mark.asyncio
    async def test_analyzer_service_fallback_chain(self, mock_s3_service):
        """Test analyzer fallback chain when primary services fail."""
        from playbooks.s3.analyzers.general_spend_analyzer import GeneralSpendAnalyzer
        
        # No storage lens service provided
        analyzer = GeneralSpendAnalyzer(s3_service=mock_s3_service)
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            result = await analyzer.analyze(region="us-east-1", lookback_days=30)
        
        assert result["status"] == "success"
        # Should fallback to Cost Explorer
        assert "cost_explorer" in result["data_sources"]
        assert "storage_lens" not in result["data_sources"]
    
    @pytest.mark.asyncio
    async def test_analyzer_error_propagation(self, mock_s3_service):
        """Test error propagation from services to analyzers."""
        from playbooks.s3.analyzers.general_spend_analyzer import GeneralSpendAnalyzer
        
        analyzer = GeneralSpendAnalyzer(s3_service=mock_s3_service)
        
        # Mock all Cost Explorer calls to fail
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.side_effect = Exception("Cost Explorer service unavailable")
            
            result = await analyzer.analyze(region="us-east-1", lookback_days=30)
        
        assert result["status"] == "error"
        assert "Cost Explorer service unavailable" in result["message"]


@pytest.mark.integration
class TestEndToEndAnalysisFlow:
    """End-to-end integration tests for complete analysis flows."""
    
    @pytest.mark.asyncio
    async def test_complete_single_analysis_flow(self, mock_aws_credentials):
        """Test complete flow from orchestrator to services for single analysis."""
        # This test uses real service classes but mocked AWS clients
        with patch('boto3.client') as mock_boto_client:
            # Mock AWS clients
            mock_s3_client = Mock()
            mock_cloudwatch_client = Mock()
            mock_ce_client = Mock()
            mock_s3control_client = Mock()
            mock_sts_client = Mock()
            
            mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
            
            def client_factory(service_name, **kwargs):
                if service_name == 's3':
                    return mock_s3_client
                elif service_name == 'cloudwatch':
                    return mock_cloudwatch_client
                elif service_name == 'ce':
                    return mock_ce_client
                elif service_name == 's3control':
                    return mock_s3control_client
                elif service_name == 'sts':
                    return mock_sts_client
                return Mock()
            
            mock_boto_client.side_effect = client_factory
            
            # Mock AWS API responses
            mock_s3_client.list_buckets.return_value = {
                "Buckets": [{"Name": "test-bucket", "CreationDate": datetime.now()}],
                "Owner": {"ID": "owner-id"}
            }
            mock_s3_client.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}
            
            mock_cloudwatch_client.get_metric_statistics.return_value = {
                "Datapoints": [{"Timestamp": datetime.now(), "Average": 1000}]
            }
            
            # Mock Cost Explorer
            with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                mock_cost_explorer.return_value = {
                    "status": "success",
                    "data": {"ResultsByTime": []}
                }
                
                # Create orchestrator and run analysis
                orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                
                result = await orchestrator.execute_analysis(
                    analysis_type="general_spend",
                    region="us-east-1",
                    lookback_days=30
                )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "general_spend"
        assert "data" in result
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_with_partial_failures(self, mock_aws_credentials):
        """Test comprehensive analysis handling partial service failures."""
        with patch('boto3.client') as mock_boto_client:
            # Setup client factory
            def client_factory(service_name, **kwargs):
                client = Mock()
                if service_name == 'sts':
                    client.get_caller_identity.return_value = {"Account": "123456789012"}
                return client
            
            mock_boto_client.side_effect = client_factory
            
            # Mock some services to fail
            with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                # First call succeeds, second fails, third succeeds
                mock_cost_explorer.side_effect = [
                    {"status": "success", "data": {"ResultsByTime": []}},  # Storage costs
                    Exception("API rate limit exceeded"),                   # Transfer costs fail
                    {"status": "success", "data": {"ResultsByTime": []}},  # API costs
                    {"status": "success", "data": {"ResultsByTime": []}}   # Retrieval costs
                ]
                
                orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                
                result = await orchestrator.execute_comprehensive_analysis(
                    region="us-east-1",
                    lookback_days=30
                )
        
        # Should still succeed overall despite partial failures
        assert result["status"] == "success"
        assert "execution_summary" in result
        # Some analyses may have failed but others succeeded
        assert result["execution_summary"]["successful"] >= 0
        assert result["execution_summary"]["total_tasks"] > 0


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance monitoring and optimization."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, performance_tracker):
        """Test that performance monitoring works across all components."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            performance_tracker.start_timer("full_analysis")
            
            result = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1",
                lookback_days=30
            )
            
            execution_time = performance_tracker.end_timer("full_analysis")
        
        assert result["status"] == "success"
        assert execution_time > 0
        
        # Should have performance metrics in result
        if "memory_usage" in result:
            assert "peak_memory_mb" in result["memory_usage"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self):
        """Test timeout handling across the integration stack."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            # Mock very slow Cost Explorer response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(5)  # Longer than timeout
                return {"status": "success", "data": {"ResultsByTime": []}}
            
            mock_cost_explorer.side_effect = slow_response
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Set short timeout
            result = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1",
                timeout_seconds=1.0
            )
        
        # Should handle timeout gracefully
        assert "status" in result
        # May be success with partial data or error, but should not hang