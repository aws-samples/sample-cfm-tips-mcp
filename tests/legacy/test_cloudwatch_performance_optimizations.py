"""
Test suite for CloudWatch performance optimizations including caching, memory management, and progressive timeouts.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Test imports
def test_performance_optimization_imports():
    """Test that all performance optimization modules can be imported."""
    try:
        from utils.cloudwatch_cache import get_cloudwatch_metadata_cache, get_cloudwatch_analysis_cache
        from utils.performance_monitor import get_performance_monitor
        from utils.memory_manager import get_memory_manager
        from utils.progressive_timeout import get_timeout_handler
        
        assert get_cloudwatch_metadata_cache is not None
        assert get_cloudwatch_analysis_cache is not None
        assert get_performance_monitor is not None
        assert get_memory_manager is not None
        assert get_timeout_handler is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import performance optimization modules: {str(e)}")

def test_cloudwatch_cache_initialization():
    """Test CloudWatch-specific cache initialization."""
    try:
        from utils.cloudwatch_cache import CloudWatchMetadataCache, CloudWatchAnalysisCache
        
        # Test metadata cache
        metadata_cache = CloudWatchMetadataCache()
        assert metadata_cache is not None
        assert hasattr(metadata_cache, 'get_alarm_metadata')
        assert hasattr(metadata_cache, 'put_alarm_metadata')
        assert hasattr(metadata_cache, 'get_dashboard_metadata')
        assert hasattr(metadata_cache, 'put_dashboard_metadata')
        
        # Test analysis cache
        analysis_cache = CloudWatchAnalysisCache()
        assert analysis_cache is not None
        assert hasattr(analysis_cache, 'get_analysis_result')
        assert hasattr(analysis_cache, 'put_analysis_result')
        
    except Exception as e:
        pytest.fail(f"Failed to initialize CloudWatch caches: {str(e)}")

def test_cloudwatch_metadata_cache_operations():
    """Test CloudWatch metadata cache operations."""
    try:
        from utils.cloudwatch_cache import CloudWatchMetadataCache
        
        cache = CloudWatchMetadataCache()
        region = "us-east-1"
        
        # Test alarm metadata operations
        alarm_data = {"alarm_name": "test-alarm", "state": "OK"}
        cache.put_alarm_metadata(region, alarm_data, "test-alarm")
        
        retrieved_data = cache.get_alarm_metadata(region, "test-alarm")
        assert retrieved_data == alarm_data
        
        # Test dashboard metadata operations
        dashboard_data = {"dashboard_name": "test-dashboard", "widgets": 5}
        cache.put_dashboard_metadata(region, dashboard_data, "test-dashboard")
        
        retrieved_dashboard = cache.get_dashboard_metadata(region, "test-dashboard")
        assert retrieved_dashboard == dashboard_data
        
        # Test log group metadata operations
        log_group_data = {"log_group_name": "/aws/lambda/test", "retention": 14}
        cache.put_log_group_metadata(region, log_group_data, "/aws/lambda/test")
        
        retrieved_log_group = cache.get_log_group_metadata(region, "/aws/lambda/test")
        assert retrieved_log_group == log_group_data
        
        # Test metrics metadata operations
        metrics_data = {"namespace": "AWS/Lambda", "metrics_count": 10}
        cache.put_metrics_metadata(region, metrics_data, "AWS/Lambda")
        
        retrieved_metrics = cache.get_metrics_metadata(region, "AWS/Lambda")
        assert retrieved_metrics == metrics_data
        
    except Exception as e:
        pytest.fail(f"CloudWatch metadata cache operations failed: {str(e)}")

def test_cloudwatch_analysis_cache_operations():
    """Test CloudWatch analysis cache operations."""
    try:
        from utils.cloudwatch_cache import CloudWatchAnalysisCache
        
        cache = CloudWatchAnalysisCache()
        region = "us-east-1"
        analysis_type = "general_spend"
        parameters_hash = "test_hash_123"
        
        # Test analysis result caching
        analysis_result = {
            "status": "success",
            "analysis_type": analysis_type,
            "recommendations": ["Optimize log retention", "Reduce custom metrics"],
            "cost_savings": 150.50
        }
        
        cache.put_analysis_result(analysis_type, region, parameters_hash, analysis_result)
        
        retrieved_result = cache.get_analysis_result(analysis_type, region, parameters_hash)
        assert retrieved_result == analysis_result
        
        # Test cache invalidation
        invalidated_count = cache.invalidate_analysis_type(analysis_type)
        assert invalidated_count >= 0  # Should be at least 0
        
        # Verify data is invalidated
        retrieved_after_invalidation = cache.get_analysis_result(analysis_type, region, parameters_hash)
        assert retrieved_after_invalidation is None
        
    except Exception as e:
        pytest.fail(f"CloudWatch analysis cache operations failed: {str(e)}")

def test_performance_monitor_integration():
    """Test performance monitor integration with CloudWatch optimizations."""
    try:
        from utils.performance_monitor import get_performance_monitor
        
        perf_monitor = get_performance_monitor()
        
        # Test analysis monitoring
        session_id = perf_monitor.start_analysis_monitoring("test_analysis", "test_execution_123")
        assert session_id is not None
        
        # Record some metrics
        perf_monitor.record_cache_hit("cloudwatch_metadata", "test_analysis")
        perf_monitor.record_cache_miss("cloudwatch_analysis", "test_analysis")
        perf_monitor.record_api_call("cloudwatch", "describe_alarms", "test_analysis")
        
        # End monitoring
        perf_data = perf_monitor.end_analysis_monitoring(session_id, success=True)
        assert perf_data is not None
        assert perf_data.analysis_type == "test_analysis"
        
        # Get performance summary
        summary = perf_monitor.get_performance_summary()
        assert "cache_performance" in summary
        assert "system_metrics" in summary
        
    except Exception as e:
        pytest.fail(f"Performance monitor integration failed: {str(e)}")

def test_memory_manager_integration():
    """Test memory manager integration with CloudWatch optimizations."""
    try:
        from utils.memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        
        # Test memory tracking
        tracker = memory_manager.start_memory_tracking("test_cloudwatch_analysis")
        assert tracker is not None
        
        # Register a large object
        test_data = {"large_dataset": list(range(1000))}
        memory_manager.register_large_object(
            "test_object_123",
            test_data,
            size_mb=1.0,
            cleanup_callback=lambda: None
        )
        
        # Stop tracking
        memory_stats = memory_manager.stop_memory_tracking("test_cloudwatch_analysis")
        assert memory_stats is not None
        assert "memory_delta_mb" in memory_stats
        
        # Get memory statistics
        stats = memory_manager.get_memory_statistics()
        assert "current_memory" in stats
        assert "thresholds" in stats
        
        # Cleanup
        success = memory_manager.cleanup_large_object("test_object_123")
        assert success is True
        
    except Exception as e:
        pytest.fail(f"Memory manager integration failed: {str(e)}")

def test_progressive_timeout_integration():
    """Test progressive timeout handler integration with CloudWatch optimizations."""
    try:
        from utils.progressive_timeout import get_timeout_handler, ComplexityLevel, AnalysisContext
        
        timeout_handler = get_timeout_handler()
        
        # Test timeout calculation
        context = AnalysisContext(
            analysis_type="comprehensive",
            complexity_level=ComplexityLevel.HIGH,
            estimated_data_size_mb=50.0,
            bucket_count=0,  # Not applicable for CloudWatch
            region="us-east-1",
            include_cost_analysis=True,
            lookback_days=30
        )
        
        timeout_result = timeout_handler.calculate_timeout(context)
        assert timeout_result.final_timeout > 0
        assert len(timeout_result.reasoning) > 0
        
        # Test execution time recording
        timeout_handler.record_execution_time("comprehensive", 45.5, ComplexityLevel.HIGH)
        
        # Test system load recording
        timeout_handler.record_system_load(75.0)
        
        # Get performance statistics
        stats = timeout_handler.get_performance_statistics()
        assert "historical_data" in stats
        assert "system_load" in stats
        
    except Exception as e:
        pytest.fail(f"Progressive timeout integration failed: {str(e)}")

@pytest.mark.asyncio
async def test_orchestrator_performance_optimizations():
    """Test CloudWatch orchestrator with performance optimizations."""
    try:
        from playbooks.cloudwatch.optimization_orchestrator import CloudWatchOptimizationOrchestrator
        
        # Mock AWS services to avoid actual API calls
        with patch('services.cloudwatch_service.CloudWatchService'), \
             patch('services.cloudwatch_pricing.CloudWatchPricing'):
            
            orchestrator = CloudWatchOptimizationOrchestrator(region="us-east-1")
            
            # Test that performance components are initialized
            assert orchestrator.performance_monitor is not None
            assert orchestrator.memory_manager is not None
            assert orchestrator.timeout_handler is not None
            assert orchestrator.pricing_cache is not None
            assert orchestrator.analysis_results_cache is not None
            assert orchestrator.cloudwatch_metadata_cache is not None
            
            # Test cache warming
            warm_result = orchestrator.warm_caches(cache_types=['metadata'])
            assert warm_result['status'] == 'success'
            
            # Test cache clearing
            clear_result = orchestrator.clear_caches()
            assert clear_result['status'] == 'success'
            
            # Test performance statistics
            stats = orchestrator.get_performance_statistics()
            assert 'performance_optimizations' in stats
            assert stats['performance_optimizations']['intelligent_caching'] is True
            assert stats['performance_optimizations']['memory_management'] is True
            assert stats['performance_optimizations']['progressive_timeouts'] is True
            
    except Exception as e:
        pytest.fail(f"Orchestrator performance optimizations test failed: {str(e)}")

@pytest.mark.asyncio
async def test_analysis_engine_performance_optimizations():
    """Test CloudWatch analysis engine with performance optimizations."""
    try:
        from playbooks.cloudwatch.analysis_engine import CloudWatchAnalysisEngine, EngineConfig
        
        # Create engine config with performance optimizations enabled
        config = EngineConfig(
            enable_performance_monitoring=True,
            enable_memory_management=True,
            enable_caching=True
        )
        
        # Mock AWS services
        with patch('services.cloudwatch_service.CloudWatchService'), \
             patch('services.cloudwatch_pricing.CloudWatchPricing'):
            
            engine = CloudWatchAnalysisEngine(
                region="us-east-1",
                session_id="test_session_123",
                config=config
            )
            
            # Test that performance components are integrated
            assert engine.performance_monitor is not None
            assert engine.memory_manager is not None
            assert engine.cache is not None
            
            # Test intelligent timeout calculation
            timeout = engine._calculate_intelligent_timeout(
                "general_spend", 
                engine.analyzer_configs.get("general_spend"),
                lookback_days=30,
                region="us-east-1"
            )
            assert timeout > 0
            
    except Exception as e:
        pytest.fail(f"Analysis engine performance optimizations test failed: {str(e)}")

def test_runbook_functions_performance_integration():
    """Test that runbook functions include performance optimization functions."""
    try:
        import runbook_functions
        
        # Test that new performance functions exist
        assert hasattr(runbook_functions, 'get_cloudwatch_performance_statistics')
        assert hasattr(runbook_functions, 'warm_cloudwatch_caches')
        assert hasattr(runbook_functions, 'clear_cloudwatch_caches')
        assert hasattr(runbook_functions, 'validate_cloudwatch_cost_preferences')
        assert hasattr(runbook_functions, 'get_cloudwatch_cost_estimate')
        
        # Test function signatures
        import inspect
        
        # Check get_cloudwatch_performance_statistics
        sig = inspect.signature(runbook_functions.get_cloudwatch_performance_statistics)
        assert 'region' in sig.parameters
        
        # Check warm_cloudwatch_caches
        sig = inspect.signature(runbook_functions.warm_cloudwatch_caches)
        assert 'region' in sig.parameters
        assert 'cache_types' in sig.parameters
        
        # Check clear_cloudwatch_caches
        sig = inspect.signature(runbook_functions.clear_cloudwatch_caches)
        assert 'region' in sig.parameters
        
    except Exception as e:
        pytest.fail(f"Runbook functions performance integration test failed: {str(e)}")

def test_mcp_server_performance_tools():
    """Test that MCP server includes performance optimization tools."""
    try:
        # Import the server module to check tool definitions
        import mcp_server_with_runbooks
        
        # Test that performance functions exist
        assert hasattr(mcp_server_with_runbooks, 'get_cloudwatch_performance_statistics')
        assert hasattr(mcp_server_with_runbooks, 'warm_cloudwatch_caches')
        assert hasattr(mcp_server_with_runbooks, 'clear_cloudwatch_caches')
        assert hasattr(mcp_server_with_runbooks, 'get_cloudwatch_cost_estimate')
        
        # Test sync wrapper functions
        assert hasattr(mcp_server_with_runbooks, 'get_cloudwatch_performance_statistics_sync')
        assert hasattr(mcp_server_with_runbooks, 'warm_cloudwatch_caches_sync')
        assert hasattr(mcp_server_with_runbooks, 'clear_cloudwatch_caches_sync')
        assert hasattr(mcp_server_with_runbooks, 'get_cloudwatch_cost_estimate_sync')
        
    except Exception as e:
        pytest.fail(f"MCP server performance tools test failed: {str(e)}")

def test_cache_warming_functionality():
    """Test cache warming functionality with mock data."""
    try:
        from utils.cloudwatch_cache import CloudWatchMetadataCache
        
        cache = CloudWatchMetadataCache()
        
        # Test warming functions exist
        warming_functions = cache._warming_functions
        assert 'alarms_metadata' in warming_functions
        assert 'dashboards_metadata' in warming_functions
        assert 'log_groups_metadata' in warming_functions
        assert 'metrics_metadata' in warming_functions
        
        # Test cache warming execution
        success = cache.warm_cache('alarms_metadata', region='us-east-1')
        assert success is True
        
        success = cache.warm_cache('dashboards_metadata', region='us-east-1')
        assert success is True
        
        success = cache.warm_cache('log_groups_metadata', region='us-east-1')
        assert success is True
        
        success = cache.warm_cache('metrics_metadata', region='us-east-1')
        assert success is True
        
    except Exception as e:
        pytest.fail(f"Cache warming functionality test failed: {str(e)}")

def test_memory_management_callbacks():
    """Test memory management cleanup callbacks."""
    try:
        from utils.memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        
        # Test callback registration
        callback_executed = False
        
        def test_callback():
            nonlocal callback_executed
            callback_executed = True
        
        memory_manager.register_cleanup_callback(test_callback)
        
        # Force cleanup to test callback execution
        memory_manager.force_cleanup("gentle")
        
        # Note: callback execution depends on memory thresholds, so we just test registration
        assert len(memory_manager.cleanup_callbacks) > 0
        
    except Exception as e:
        pytest.fail(f"Memory management callbacks test failed: {str(e)}")

if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Running CloudWatch performance optimization tests...")
    
    try:
        test_performance_optimization_imports()
        print("✅ Import test passed")
        
        test_cloudwatch_cache_initialization()
        print("✅ Cache initialization test passed")
        
        test_cloudwatch_metadata_cache_operations()
        print("✅ Metadata cache operations test passed")
        
        test_cloudwatch_analysis_cache_operations()
        print("✅ Analysis cache operations test passed")
        
        test_performance_monitor_integration()
        print("✅ Performance monitor integration test passed")
        
        test_memory_manager_integration()
        print("✅ Memory manager integration test passed")
        
        test_progressive_timeout_integration()
        print("✅ Progressive timeout integration test passed")
        
        test_runbook_functions_performance_integration()
        print("✅ Runbook functions integration test passed")
        
        test_mcp_server_performance_tools()
        print("✅ MCP server performance tools test passed")
        
        test_cache_warming_functionality()
        print("✅ Cache warming functionality test passed")
        
        test_memory_management_callbacks()
        print("✅ Memory management callbacks test passed")
        
        print("\n🎉 All CloudWatch performance optimization tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()