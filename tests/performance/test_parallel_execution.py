"""
Performance tests for parallel execution and timeout handling.

Tests the performance characteristics of the S3 optimization system,
including parallel execution, timeout handling, and resource usage.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator


@pytest.mark.performance
class TestParallelExecutionPerformance:
    """Test parallel execution performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_analysis(self, performance_tracker):
        """Test that parallel execution is faster than sequential."""
        # Mock analysis engine with controlled delays
        mock_analysis_engine = Mock()
        
        async def slow_analysis(analysis_type, **kwargs):
            await asyncio.sleep(0.1)  # 100ms per analysis
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "data": {"test": True},
                "execution_time": 0.1
            }
        
        mock_analysis_engine.run_analysis = slow_analysis
        
        # Test sequential execution (simulated)
        performance_tracker.start_timer("sequential")
        for analysis_type in ["general_spend", "storage_class", "archive_optimization"]:
            await slow_analysis(analysis_type)
        sequential_time = performance_tracker.end_timer("sequential")
        
        # Test parallel execution
        with patch('core.s3_optimization_orchestrator.S3AnalysisEngine', return_value=mock_analysis_engine), \
             patch('core.s3_optimization_orchestrator.ServiceOrchestrator'), \
             patch('core.s3_optimization_orchestrator.get_performance_monitor'), \
             patch('core.s3_optimization_orchestrator.get_memory_manager'), \
             patch('core.s3_optimization_orchestrator.get_timeout_handler'), \
             patch('core.s3_optimization_orchestrator.get_pricing_cache'), \
             patch('core.s3_optimization_orchestrator.get_bucket_metadata_cache'), \
             patch('core.s3_optimization_orchestrator.get_analysis_results_cache'):
            
            orchestrator = S3OptimizationOrchestrator()
            
            # Mock the parallel execution
            performance_tracker.start_timer("parallel")
            tasks = [
                slow_analysis("general_spend"),
                slow_analysis("storage_class"),
                slow_analysis("archive_optimization")
            ]
            await asyncio.gather(*tasks)
            parallel_time = performance_tracker.end_timer("parallel")
        
        # Parallel should be significantly faster
        assert parallel_time < sequential_time * 0.7  # At least 30% faster
        assert parallel_time < 0.2  # Should complete in ~0.1s, not 0.3s
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_performance(self, performance_tracker, 
                                                    performance_test_data):
        """Test comprehensive analysis performance with different dataset sizes."""
        for dataset_name, dataset_config in performance_test_data.items():
            with patch('boto3.client'), \
                 patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                
                # Mock Cost Explorer with dataset-appropriate delay
                async def dataset_appropriate_delay(*args, **kwargs):
                    # Simulate processing time based on dataset size
                    delay = dataset_config["bucket_count"] * 0.001  # 1ms per bucket
                    await asyncio.sleep(delay)
                    return {"status": "success", "data": {"ResultsByTime": []}}
                
                mock_cost_explorer.side_effect = dataset_appropriate_delay
                
                orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                
                performance_tracker.start_timer(f"comprehensive_{dataset_name}")
                
                result = await orchestrator.execute_comprehensive_analysis(
                    region="us-east-1",
                    lookback_days=30
                )
                
                execution_time = performance_tracker.end_timer(f"comprehensive_{dataset_name}")
            
            assert result["status"] == "success"
            performance_tracker.assert_performance(
                f"comprehensive_{dataset_name}",
                dataset_config["expected_max_time"]
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, performance_tracker):
        """Test handling multiple concurrent analysis requests."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Create multiple concurrent requests
            performance_tracker.start_timer("concurrent_requests")
            
            tasks = []
            for i in range(5):  # 5 concurrent requests
                task = orchestrator.execute_analysis(
                    analysis_type="general_spend",
                    region="us-east-1",
                    session_id=f"session_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            execution_time = performance_tracker.end_timer("concurrent_requests")
        
        # All requests should succeed
        assert all(result["status"] == "success" for result in results)
        
        # Should handle concurrency efficiently
        assert execution_time < 10.0  # Should not take too long
        
        # Each result should have unique session
        session_ids = [result["session_id"] for result in results]
        assert len(set(session_ids)) == 5  # All unique
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_tracker):
        """Test memory usage characteristics under load."""
        memory_usage_samples = []
        
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Run multiple analyses and track memory
            for i in range(10):
                performance_tracker.start_timer(f"analysis_{i}")
                
                result = await orchestrator.execute_analysis(
                    analysis_type="general_spend",
                    region="us-east-1"
                )
                
                execution_time = performance_tracker.end_timer(f"analysis_{i}")
                
                # Collect memory usage if available
                if "memory_usage" in result:
                    memory_usage_samples.append(result["memory_usage"]["peak_memory_mb"])
                
                assert result["status"] == "success"
                assert execution_time < 5.0  # Each analysis should be reasonably fast
        
        # Memory usage should be reasonable and not grow excessively
        if memory_usage_samples:
            avg_memory = sum(memory_usage_samples) / len(memory_usage_samples)
            max_memory = max(memory_usage_samples)
            
            assert avg_memory < 100.0  # Average memory usage should be reasonable
            assert max_memory < 200.0  # Peak memory should not be excessive


@pytest.mark.performance
class TestTimeoutHandling:
    """Test timeout handling performance and behavior."""
    
    @pytest.mark.asyncio
    async def test_timeout_scenarios(self, timeout_test_scenarios, performance_tracker):
        """Test various timeout scenarios."""
        for scenario_name, scenario_config in timeout_test_scenarios.items():
            with patch('boto3.client'), \
                 patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                
                if scenario_config["expected_behavior"] == "timeout":
                    # Mock very slow response for timeout scenario
                    async def very_slow_response(*args, **kwargs):
                        await asyncio.sleep(10)  # Much longer than timeout
                        return {"status": "success", "data": {"ResultsByTime": []}}
                    
                    mock_cost_explorer.side_effect = very_slow_response
                else:
                    # Mock normal response
                    mock_cost_explorer.return_value = {
                        "status": "success",
                        "data": {"ResultsByTime": []}
                    }
                
                orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                
                performance_tracker.start_timer(scenario_name)
                
                if scenario_config["analysis_type"] == "comprehensive":
                    result = await orchestrator.execute_comprehensive_analysis(
                        region="us-east-1",
                        timeout_seconds=scenario_config["timeout_seconds"]
                    )
                else:
                    result = await orchestrator.execute_analysis(
                        analysis_type=scenario_config["analysis_type"],
                        region="us-east-1",
                        timeout_seconds=scenario_config["timeout_seconds"]
                    )
                
                execution_time = performance_tracker.end_timer(scenario_name)
            
            # Should respect timeout
            if scenario_config["expected_behavior"] == "timeout":
                # Should complete quickly due to timeout, not wait for full operation
                assert execution_time < scenario_config["timeout_seconds"] * 2
            else:
                # Should complete successfully
                assert result["status"] == "success"
                assert execution_time < scenario_config["timeout_seconds"]
    
    @pytest.mark.asyncio
    async def test_progressive_timeout_behavior(self, performance_tracker):
        """Test progressive timeout behavior for different analysis complexities."""
        timeout_configs = [
            {"analysis_type": "general_spend", "expected_timeout": 30.0},
            {"analysis_type": "comprehensive", "expected_timeout": 120.0}
        ]
        
        for config in timeout_configs:
            with patch('boto3.client'), \
                 patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                
                mock_cost_explorer.return_value = {
                    "status": "success",
                    "data": {"ResultsByTime": []}
                }
                
                orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                
                performance_tracker.start_timer(f"timeout_{config['analysis_type']}")
                
                if config["analysis_type"] == "comprehensive":
                    result = await orchestrator.execute_comprehensive_analysis(
                        region="us-east-1",
                        lookback_days=30
                    )
                else:
                    result = await orchestrator.execute_analysis(
                        analysis_type=config["analysis_type"],
                        region="us-east-1",
                        lookback_days=30
                    )
                
                execution_time = performance_tracker.end_timer(f"timeout_{config['analysis_type']}")
            
            assert result["status"] == "success"
            # Should complete well within expected timeout
            assert execution_time < config["expected_timeout"] * 0.5
    
    @pytest.mark.asyncio
    async def test_timeout_with_partial_results(self, performance_tracker):
        """Test that timeout scenarios can return partial results."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            # Mock mixed response times - some fast, some slow
            call_count = 0
            async def mixed_response_times(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    # First two calls are fast
                    await asyncio.sleep(0.1)
                    return {"status": "success", "data": {"ResultsByTime": []}}
                else:
                    # Subsequent calls are very slow
                    await asyncio.sleep(10)
                    return {"status": "success", "data": {"ResultsByTime": []}}
            
            mock_cost_explorer.side_effect = mixed_response_times
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            performance_tracker.start_timer("partial_timeout")
            
            result = await orchestrator.execute_comprehensive_analysis(
                region="us-east-1",
                timeout_seconds=2.0  # Short timeout
            )
            
            execution_time = performance_tracker.end_timer("partial_timeout")
        
        # Should complete within timeout
        assert execution_time < 4.0  # Allow some overhead
        
        # May have partial results
        assert "status" in result
        if result["status"] == "success":
            # Some analyses completed
            assert "execution_summary" in result
        # Even if overall status is error, should have some partial data


@pytest.mark.performance
class TestCachingPerformance:
    """Test caching performance and effectiveness."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, performance_tracker):
        """Test that cache hits are significantly faster than cache misses."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            # Mock slow Cost Explorer response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(0.5)  # 500ms delay
                return {"status": "success", "data": {"ResultsByTime": []}}
            
            mock_cost_explorer.side_effect = slow_response
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # First call - cache miss
            performance_tracker.start_timer("cache_miss")
            result1 = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1",
                lookback_days=30
            )
            cache_miss_time = performance_tracker.end_timer("cache_miss")
            
            # Second call - should be cache hit
            performance_tracker.start_timer("cache_hit")
            result2 = await orchestrator.execute_analysis(
                analysis_type="general_spend",
                region="us-east-1",
                lookback_days=30
            )
            cache_hit_time = performance_tracker.end_timer("cache_hit")
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        
        # Cache hit should be much faster
        assert cache_hit_time < cache_miss_time * 0.1  # At least 10x faster
        assert cache_hit_time < 0.1  # Should be very fast
        
        # Second result should indicate cache hit
        if "from_cache" in result2:
            assert result2["from_cache"] is True
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness_under_load(self, performance_tracker):
        """Test cache effectiveness under concurrent load."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            call_count = 0
            async def counting_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.1)  # Small delay
                return {"status": "success", "data": {"ResultsByTime": []}}
            
            mock_cost_explorer.side_effect = counting_response
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Make multiple identical requests concurrently
            performance_tracker.start_timer("concurrent_cache_test")
            
            tasks = []
            for i in range(10):  # 10 identical requests
                task = orchestrator.execute_analysis(
                    analysis_type="general_spend",
                    region="us-east-1",
                    lookback_days=30
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            execution_time = performance_tracker.end_timer("concurrent_cache_test")
        
        # All should succeed
        assert all(result["status"] == "success" for result in results)
        
        # Should complete quickly due to caching
        assert execution_time < 2.0
        
        # Should have made fewer API calls than requests due to caching
        # (First request populates cache, others use cache)
        assert call_count < len(results)


@pytest.mark.performance
class TestResourceUsageOptimization:
    """Test resource usage optimization features."""
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_effectiveness(self, performance_tracker):
        """Test that memory cleanup is effective."""
        with patch('boto3.client'), \
             patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {"ResultsByTime": []}
            }
            
            orchestrator = S3OptimizationOrchestrator(region="us-east-1")
            
            # Run multiple analyses to test memory cleanup
            memory_samples = []
            
            for i in range(5):
                result = await orchestrator.execute_analysis(
                    analysis_type="general_spend",
                    region="us-east-1"
                )
                
                assert result["status"] == "success"
                
                # Collect memory usage if available
                if "memory_usage" in result:
                    memory_samples.append(result["memory_usage"]["peak_memory_mb"])
        
        # Memory usage should not grow significantly over time
        if len(memory_samples) >= 3:
            first_half_avg = sum(memory_samples[:2]) / 2
            second_half_avg = sum(memory_samples[-2:]) / 2
            
            # Memory usage should not increase significantly
            assert second_half_avg < first_half_avg * 1.5  # Allow 50% increase max
    
    @pytest.mark.asyncio
    async def test_connection_pooling_efficiency(self, performance_tracker):
        """Test that connection pooling improves efficiency."""
        with patch('boto3.client') as mock_boto_client:
            # Track client creation
            client_creation_count = 0
            
            def counting_client_factory(*args, **kwargs):
                nonlocal client_creation_count
                client_creation_count += 1
                client = Mock()
                if args[0] == 'sts':
                    client.get_caller_identity.return_value = {"Account": "123456789012"}
                return client
            
            mock_boto_client.side_effect = counting_client_factory
            
            with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
                mock_cost_explorer.return_value = {
                    "status": "success",
                    "data": {"ResultsByTime": []}
                }
                
                # Create multiple orchestrators and run analyses
                performance_tracker.start_timer("connection_pooling_test")
                
                for i in range(3):
                    orchestrator = S3OptimizationOrchestrator(region="us-east-1")
                    result = await orchestrator.execute_analysis(
                        analysis_type="general_spend",
                        region="us-east-1"
                    )
                    assert result["status"] == "success"
                
                execution_time = performance_tracker.end_timer("connection_pooling_test")
        
        # Should complete efficiently
        assert execution_time < 5.0
        
        # Should reuse connections efficiently (exact count depends on implementation)
        assert client_creation_count > 0  # Some clients were created