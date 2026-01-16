"""
Performance tests for CloudWatch optimization parallel execution and timeout handling.

Tests parallel execution efficiency, timeout behavior, memory usage,
and performance under various load conditions.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import psutil
import os

from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
from playbooks.cloudwatch.analysis_engine import CloudWatchAnalysisEngine
from utils.service_orchestrator import ServiceOrchestrator
from utils.parallel_executor import ParallelExecutor


@pytest.mark.performance
@pytest.mark.asyncio
class TestParallelExecutionPerformance:
    """Performance tests for parallel execution."""
    
    @pytest.fixture
    def mock_analysis_tasks(self):
        """Create mock analysis tasks with different execution times."""
        async def fast_task():
            await asyncio.sleep(0.1)
            return {"status": "success", "execution_time": 0.1, "data": {"result": "fast"}}
        
        async def medium_task():
            await asyncio.sleep(0.3)
            return {"status": "success", "execution_time": 0.3, "data": {"result": "medium"}}
        
        async def slow_task():
            await asyncio.sleep(0.5)
            return {"status": "success", "execution_time": 0.5, "data": {"result": "slow"}}
        
        async def variable_task(delay=0.2):
            await asyncio.sleep(delay)
            return {"status": "success", "execution_time": delay, "data": {"result": f"variable_{delay}"}}
        
        return {
            "fast": fast_task,
            "medium": medium_task,
            "slow": slow_task,
            "variable": variable_task
        }
    
    @pytest.fixture
    def performance_orchestrator(self):
        """Create orchestrator for performance testing."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so, \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_ae:
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
            orchestrator.service_orchestrator = mock_so.return_value
            orchestrator.analysis_engine = mock_ae.return_value
            
            return orchestrator
    
    async def test_parallel_vs_sequential_execution_performance(self, performance_orchestrator, mock_analysis_tasks, performance_tracker):
        """Test performance difference between parallel and sequential execution."""
        # Mock service orchestrator for parallel execution
        async def mock_parallel_execution(**kwargs):
            # Simulate parallel execution - all tasks run concurrently
            results = await asyncio.gather(*[task() for task in tasks])
            return {
                "status": "success",
                "successful": len(results),
                "total_tasks": len(tasks),
                "results": {f"task_{i}": result for i, result in enumerate(results)}
            }
        
        async def mock_sequential_execution(**kwargs):
            # Simulate sequential execution - tasks run one after another
            results = []
            for task in tasks:
                result = await task()
                results.append(result)
            return {
                "status": "success",
                "successful": len(results),
                "total_tasks": len(tasks),
                "results": {f"task_{i}": result for i, result in enumerate(results)}
            }
        
        tasks = [
            mock_analysis_tasks["fast"],
            mock_analysis_tasks["medium"],
            mock_analysis_tasks["slow"],
            mock_analysis_tasks["fast"]
        ]
        
        # Test parallel execution
        performance_tracker.start_timer('parallel_execution')
        performance_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_parallel_execution
        
        parallel_result = await performance_orchestrator.execute_comprehensive_analysis(
            parallel_execution=True
        )
        
        parallel_time = performance_tracker.end_timer('parallel_execution')
        
        # Test sequential execution
        performance_tracker.start_timer('sequential_execution')
        performance_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_sequential_execution
        
        sequential_result = await performance_orchestrator.execute_comprehensive_analysis(
            parallel_execution=False
        )
        
        sequential_time = performance_tracker.end_timer('sequential_execution')
        
        # Verify both succeeded
        assert parallel_result['status'] == 'success'
        assert sequential_result['status'] == 'success'
        
        # Parallel should be significantly faster
        # Expected: parallel ~0.5s (max task time), sequential ~0.9s (sum of task times)
        assert parallel_time < sequential_time
        assert parallel_time < 0.7  # Should be close to slowest task time
        assert sequential_time > 0.8  # Should be close to sum of task times
        
        print(f"Parallel execution: {parallel_time:.3f}s")
        print(f"Sequential execution: {sequential_time:.3f}s")
        print(f"Performance improvement: {sequential_time / parallel_time:.2f}x")
    
    async def test_parallel_execution_scalability(self, performance_orchestrator, mock_analysis_tasks, performance_tracker):
        """Test parallel execution scalability with increasing task count."""
        async def create_variable_tasks(count, base_delay=0.1):
            """Create multiple variable tasks with slight delay variations."""
            tasks = []
            for i in range(count):
                delay = base_delay + (i * 0.01)  # Slight variation in delays
                tasks.append(lambda d=delay: mock_analysis_tasks["variable"](d))
            return tasks
        
        # Test with different numbers of parallel tasks
        task_counts = [2, 4, 8, 16]
        execution_times = {}
        
        for count in task_counts:
            tasks = await create_variable_tasks(count)
            
            async def mock_parallel_execution_scalability(**kwargs):
                # Execute all tasks in parallel
                results = await asyncio.gather(*[task() for task in tasks])
                return {
                    "status": "success",
                    "successful": len(results),
                    "total_tasks": len(tasks),
                    "results": {f"task_{i}": result for i, result in enumerate(results)}
                }
            
            performance_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_parallel_execution_scalability
            
            performance_tracker.start_timer(f'parallel_{count}_tasks')
            
            result = await performance_orchestrator.execute_comprehensive_analysis(
                parallel_execution=True
            )
            
            execution_time = performance_tracker.end_timer(f'parallel_{count}_tasks')
            execution_times[count] = execution_time
            
            assert result['status'] == 'success'
            assert result['successful'] == count
            
            print(f"Parallel execution with {count} tasks: {execution_time:.3f}s")
        
        # Verify scalability - execution time should not increase linearly with task count
        # With proper parallel execution, time should remain relatively constant
        max_time = max(execution_times.values())
        min_time = min(execution_times.values())
        
        # Time difference should be reasonable (not more than 3x)
        assert max_time / min_time < 3.0, f"Poor scalability: {max_time:.3f}s vs {min_time:.3f}s"
    
    async def test_memory_usage_during_parallel_execution(self, performance_orchestrator, mock_analysis_tasks):
        """Test memory usage during parallel execution."""
        process = psutil.Process(os.getpid())
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-intensive mock tasks
        async def memory_intensive_task(data_size_mb=10):
            # Simulate memory usage by creating large data structures
            large_data = [0] * (data_size_mb * 1024 * 256)  # Approximate MB of integers
            await asyncio.sleep(0.2)
            return {
                "status": "success",
                "data": {"size": len(large_data)},
                "memory_used_mb": data_size_mb
            }
        
        # Create multiple memory-intensive tasks
        tasks = [lambda: memory_intensive_task(5) for _ in range(8)]
        
        async def mock_memory_parallel_execution(**kwargs):
            results = await asyncio.gather(*[task() for task in tasks])
            return {
                "status": "success",
                "successful": len(results),
                "total_tasks": len(tasks),
                "results": {f"task_{i}": result for i, result in enumerate(results)}
            }
        
        performance_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_memory_parallel_execution
        
        # Monitor memory during execution
        peak_memory = baseline_memory
        
        async def memory_monitor():
            nonlocal peak_memory
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                await asyncio.sleep(0.05)
        
        # Start memory monitoring
        monitor_task = asyncio.create_task(memory_monitor())
        
        try:
            result = await performance_orchestrator.execute_comprehensive_analysis(
                parallel_execution=True
            )
            
            # Wait a bit for memory monitoring
            await asyncio.sleep(0.1)
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        
        assert result['status'] == 'success'
        
        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")
        
        # Memory should not increase excessively (reasonable limit for test environment)
        memory_increase = peak_memory - baseline_memory
        assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.2f} MB"
    
    async def test_error_handling_in_parallel_execution(self, performance_orchestrator, performance_tracker):
        """Test error handling performance in parallel execution."""
        async def successful_task():
            await asyncio.sleep(0.1)
            return {"status": "success", "data": {"result": "success"}}
        
        async def failing_task():
            await asyncio.sleep(0.1)
            raise Exception("Simulated task failure")
        
        async def timeout_task():
            await asyncio.sleep(2.0)  # Will timeout
            return {"status": "success", "data": {"result": "timeout"}}
        
        # Mix of successful, failing, and timeout tasks
        tasks = [successful_task, failing_task, successful_task, timeout_task, successful_task]
        
        async def mock_error_parallel_execution(**kwargs):
            results = []
            for i, task in enumerate(tasks):
                try:
                    # Apply timeout to each task
                    result = await asyncio.wait_for(task(), timeout=0.5)
                    results.append({"status": "success", "data": result})
                except asyncio.TimeoutError:
                    results.append({"status": "timeout", "error": "Task timed out"})
                except Exception as e:
                    results.append({"status": "error", "error": str(e)})
            
            successful = sum(1 for r in results if r["status"] == "success")
            return {
                "status": "partial" if successful < len(results) else "success",
                "successful": successful,
                "total_tasks": len(results),
                "results": {f"task_{i}": result for i, result in enumerate(results)}
            }
        
        performance_orchestrator.analysis_engine.run_comprehensive_analysis.side_effect = mock_error_parallel_execution
        
        performance_tracker.start_timer('error_handling')
        
        result = await performance_orchestrator.execute_comprehensive_analysis(
            parallel_execution=True,
            timeout_seconds=1.0
        )
        
        execution_time = performance_tracker.end_timer('error_handling')
        
        # Should handle errors gracefully and complete quickly
        assert result['status'] in ['success', 'partial']
        assert result['successful'] >= 3  # At least 3 successful tasks
        assert execution_time < 1.0  # Should not wait for full timeout
        
        print(f"Error handling execution time: {execution_time:.3f}s")
        print(f"Successful tasks: {result['successful']}/{result['total_tasks']}")


@pytest.mark.performance
@pytest.mark.asyncio
class TestTimeoutHandlingPerformance:
    """Performance tests for timeout handling."""
    
    @pytest.fixture
    def timeout_orchestrator(self):
        """Create orchestrator for timeout testing."""
        with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator') as mock_so, \
             patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine') as mock_ae:
            
            orchestrator = CloudWatchOptimizationOrchestrator(region='us-east-1')
            orchestrator.service_orchestrator = mock_so.return_value
            orchestrator.analysis_engine = mock_ae.return_value
            
            return orchestrator
    
    async def test_timeout_accuracy(self, timeout_orchestrator, performance_tracker, timeout_test_scenarios):
        """Test timeout accuracy and responsiveness."""
        for scenario_name, scenario in timeout_test_scenarios.items():
            async def mock_analysis_with_delay(analysis_type, **kwargs):
                # Simulate analysis that takes longer than timeout
                if scenario["expected_behavior"] == "timeout":
                    await asyncio.sleep(scenario["timeout_seconds"] * 2)
                else:
                    await asyncio.sleep(scenario["timeout_seconds"] * 0.5)
                
                return {
                    "status": "success",
                    "analysis_type": analysis_type,
                    "data": {"completed": True}
                }
            
            timeout_orchestrator.analysis_engine.run_analysis.side_effect = mock_analysis_with_delay
            
            performance_tracker.start_timer(f'timeout_{scenario_name}')
            
            result = await timeout_orchestrator.execute_analysis(
                analysis_type=scenario["analysis_type"],
                timeout_seconds=scenario["timeout_seconds"]
            )
            
            execution_time = performance_tracker.end_timer(f'timeout_{scenario_name}')
            
            if scenario["expected_behavior"] == "timeout":
                assert result['status'] == 'error'
                assert 'timed out' in result['error_message'].lower()
                # Should timeout close to specified time (within 20% tolerance)
                assert execution_time <= scenario["timeout_seconds"] * 1.2
            else:
                assert result['status'] == 'success'
                assert execution_time < scenario["timeout_seconds"]
            
            print(f"Scenario {scenario_name}: {execution_time:.3f}s (timeout: {scenario['timeout_seconds']}s)")
    
    async def test_progressive_timeout_handling(self, timeout_orchestrator, performance_tracker):
        """Test progressive timeout handling for different analysis complexities."""
        # Mock different analysis complexities
        complexity_scenarios = {
            "simple": {"base_time": 0.1, "timeout_multiplier": 2.0},
            "medium": {"base_time": 0.3, "timeout_multiplier": 1.5},
            "complex": {"base_time": 0.8, "timeout_multiplier": 1.2}
        }
        
        for complexity, config in complexity_scenarios.items():
            async def mock_complex_analysis(analysis_type, **kwargs):
                await asyncio.sleep(config["base_time"])
                return {
                    "status": "success",
                    "complexity": complexity,
                    "data": {"analysis_time": config["base_time"]}
                }
            
            timeout_orchestrator.analysis_engine.run_analysis.side_effect = mock_complex_analysis
            
            # Calculate progressive timeout
            timeout_seconds = config["base_time"] * config["timeout_multiplier"]
            
            performance_tracker.start_timer(f'progressive_{complexity}')
            
            result = await timeout_orchestrator.execute_analysis(
                analysis_type='general_spend',
                timeout_seconds=timeout_seconds
            )
            
            execution_time = performance_tracker.end_timer(f'progressive_{complexity}')
            
            assert result['status'] == 'success'
            assert execution_time < timeout_seconds
            assert execution_time >= config["base_time"] * 0.9  # Should take at least base time
            
            print(f"Progressive timeout {complexity}: {execution_time:.3f}s (timeout: {timeout_seconds:.3f}s)")
    
    async def test_timeout_with_cleanup(self, timeout_orchestrator, performance_tracker):
        """Test timeout handling with proper resource cleanup."""
        cleanup_called = []
        
        async def mock_analysis_with_cleanup(analysis_type, **kwargs):
            try:
                # Simulate long-running analysis
                await asyncio.sleep(2.0)
                return {"status": "success", "data": {}}
            except asyncio.CancelledError:
                # Simulate cleanup operations
                cleanup_called.append(True)
                await asyncio.sleep(0.1)  # Cleanup time
                raise
        
        timeout_orchestrator.analysis_engine.run_analysis.side_effect = mock_analysis_with_cleanup
        
        performance_tracker.start_timer('timeout_with_cleanup')
        
        result = await timeout_orchestrator.execute_analysis(
            analysis_type='general_spend',
            timeout_seconds=0.5
        )
        
        execution_time = performance_tracker.end_timer('timeout_with_cleanup')
        
        assert result['status'] == 'error'
        assert 'timed out' in result['error_message'].lower()
        assert len(cleanup_called) > 0  # Cleanup should have been called
        
        # Should complete within reasonable time including cleanup
        assert execution_time < 1.0
        
        print(f"Timeout with cleanup: {execution_time:.3f}s")


@pytest.mark.performance
class TestConcurrencyLimits:
    """Test performance under various concurrency limits."""
    
    def test_concurrent_orchestrator_instances(self, performance_tracker):
        """Test performance with multiple concurrent orchestrator instances."""
        async def run_concurrent_orchestrators(count):
            orchestrators = []
            
            for i in range(count):
                with patch('playbooks.cloudwatch.optimization_orchestrator.ServiceOrchestrator'), \
                     patch('playbooks.cloudwatch.optimization_orchestrator.CloudWatchAnalysisEngine'):
                    
                    orchestrator = CloudWatchOptimizationOrchestrator(
                        region='us-east-1',
                        session_id=f'session_{i}'
                    )
                    
                    # Mock quick analysis
                    orchestrator.analysis_engine.run_analysis = AsyncMock(return_value={
                        "status": "success",
                        "analysis_type": "general_spend",
                        "data": {"instance": i}
                    })
                    
                    orchestrators.append(orchestrator)
            
            # Run all orchestrators concurrently
            tasks = [
                orchestrator.execute_analysis('general_spend')
                for orchestrator in orchestrators
            ]
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for level in concurrency_levels:
            performance_tracker.start_timer(f'concurrent_{level}')
            
            results = asyncio.run(run_concurrent_orchestrators(level))
            
            execution_time = performance_tracker.end_timer(f'concurrent_{level}')
            
            assert len(results) == level
            assert all(result['status'] == 'success' for result in results)
            
            print(f"Concurrent orchestrators ({level}): {execution_time:.3f}s")
            
            # Performance should not degrade significantly with reasonable concurrency
            if level <= 10:
                performance_tracker.assert_performance(f'concurrent_{level}', 2.0)


if __name__ == "__main__":
    pytest.main([__file__])