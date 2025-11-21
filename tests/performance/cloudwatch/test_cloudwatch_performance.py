#!/usr/bin/env python3
"""
Performance tests for CloudWatch optimization components.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

def test_core_performance_components():
    """Test core performance optimization components."""
    print("Testing CloudWatch performance optimization components...")
    
    try:
        # Test imports only - don't initialize global caches
        from utils.cloudwatch_cache import CloudWatchMetadataCache, CloudWatchAnalysisCache
        from utils.performance_monitor import get_performance_monitor
        from utils.memory_manager import get_memory_manager
        from utils.progressive_timeout import get_timeout_handler
        
        print("✅ All performance optimization imports successful")
        
        # Test that classes can be instantiated (but don't keep instances)
        print(f"✅ CloudWatch metadata cache class available: {CloudWatchMetadataCache.__name__}")
        print(f"✅ CloudWatch analysis cache class available: {CloudWatchAnalysisCache.__name__}")
        
        # Test performance components (these should be stateless or properly managed)
        perf_monitor = get_performance_monitor()
        memory_manager = get_memory_manager()
        timeout_handler = get_timeout_handler()
        
        print(f"✅ Performance monitor initialized: {type(perf_monitor).__name__}")
        print(f"✅ Memory manager initialized: {type(memory_manager).__name__}")
        print(f"✅ Timeout handler initialized: {type(timeout_handler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in core components test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cloudwatch_cache_operations():
    """Test CloudWatch cache operations with isolated instances."""
    print("\nTesting CloudWatch cache operations...")
    
    metadata_cache = None
    analysis_cache = None
    
    try:
        from utils.cloudwatch_cache import CloudWatchMetadataCache, CloudWatchAnalysisCache
        
        # Create isolated cache instances for testing
        metadata_cache = CloudWatchMetadataCache()
        region = "us-east-1"
        
        # Test alarm metadata
        alarm_data = {"alarm_name": "test-alarm", "state": "OK", "region": region}
        metadata_cache.put_alarm_metadata(region, alarm_data, "test-alarm")
        
        retrieved_data = metadata_cache.get_alarm_metadata(region, "test-alarm")
        assert retrieved_data == alarm_data, "Alarm metadata retrieval failed"
        
        print("✅ Alarm metadata cache operations working")
        
        # Test dashboard metadata
        dashboard_data = {"dashboard_name": "test-dashboard", "widgets": 5}
        metadata_cache.put_dashboard_metadata(region, dashboard_data, "test-dashboard")
        
        retrieved_dashboard = metadata_cache.get_dashboard_metadata(region, "test-dashboard")
        assert retrieved_dashboard == dashboard_data, "Dashboard metadata retrieval failed"
        
        print("✅ Dashboard metadata cache operations working")
        
        # Test analysis cache
        analysis_cache = CloudWatchAnalysisCache()
        analysis_type = "general_spend"
        parameters_hash = "test_hash_123"
        
        analysis_result = {
            "status": "success",
            "analysis_type": analysis_type,
            "recommendations": ["Optimize log retention", "Reduce custom metrics"],
            "cost_savings": 150.50
        }
        
        analysis_cache.put_analysis_result(analysis_type, region, parameters_hash, analysis_result)
        retrieved_result = analysis_cache.get_analysis_result(analysis_type, region, parameters_hash)
        assert retrieved_result == analysis_result, "Analysis result retrieval failed"
        
        print("✅ Analysis cache operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in cache operations test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up isolated cache instances
        if metadata_cache and hasattr(metadata_cache, 'shutdown'):
            metadata_cache.shutdown()
        if analysis_cache and hasattr(analysis_cache, 'shutdown'):
            analysis_cache.shutdown()

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting performance monitoring...")
    
    try:
        from utils.performance_monitor import get_performance_monitor
        
        perf_monitor = get_performance_monitor()
        
        # Test analysis monitoring
        session_id = perf_monitor.start_analysis_monitoring("test_analysis", "test_execution_123")
        assert session_id is not None, "Failed to start analysis monitoring"
        
        # Record some metrics
        perf_monitor.record_cache_hit("cloudwatch_metadata", "test_analysis")
        perf_monitor.record_cache_miss("cloudwatch_analysis", "test_analysis")
        perf_monitor.record_api_call("cloudwatch", "describe_alarms", "test_analysis")
        
        # End monitoring
        perf_data = perf_monitor.end_analysis_monitoring(session_id, success=True)
        assert perf_data is not None, "Failed to end analysis monitoring"
        assert perf_data.analysis_type == "test_analysis", "Analysis type mismatch"
        
        # Get performance summary
        summary = perf_monitor.get_performance_summary()
        assert "cache_performance" in summary, "Cache performance not in summary"
        assert "system_metrics" in summary, "System metrics not in summary"
        
        print("✅ Performance monitoring working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance monitoring test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management functionality."""
    print("\nTesting memory management...")
    
    try:
        from utils.memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        
        # Test memory tracking
        tracker = memory_manager.start_memory_tracking("test_cloudwatch_analysis")
        assert tracker is not None, "Failed to start memory tracking"
        
        # Test with a class that supports weak references
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject({"large_dataset": list(range(1000))})
        
        # Register the object
        memory_manager.register_large_object(
            "test_object_123",
            test_obj,
            size_mb=1.0,
            cleanup_callback=lambda: None
        )
        
        # Stop tracking
        memory_stats = memory_manager.stop_memory_tracking("test_cloudwatch_analysis")
        assert memory_stats is not None, "Failed to stop memory tracking"
        assert "memory_delta_mb" in memory_stats, "Memory delta not in stats"
        
        # Get memory statistics
        stats = memory_manager.get_memory_statistics()
        assert "current_memory" in stats, "Current memory not in stats"
        assert "thresholds" in stats, "Thresholds not in stats"
        
        # Cleanup
        success = memory_manager.cleanup_large_object("test_object_123")
        assert success is True, "Failed to cleanup large object"
        
        print("✅ Memory management working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in memory management test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_progressive_timeouts():
    """Test progressive timeout functionality."""
    print("\nTesting progressive timeouts...")
    
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
        assert timeout_result.final_timeout > 0, "Invalid timeout calculated"
        assert len(timeout_result.reasoning) > 0, "No reasoning provided"
        
        # Test execution time recording
        timeout_handler.record_execution_time("comprehensive", 45.5, ComplexityLevel.HIGH)
        
        # Test system load recording
        timeout_handler.record_system_load(75.0)
        
        # Get performance statistics
        stats = timeout_handler.get_performance_statistics()
        assert "historical_data" in stats, "Historical data not in stats"
        assert "system_load" in stats, "System load not in stats"
        
        print("✅ Progressive timeouts working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in progressive timeouts test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_warming():
    """Test cache warming functionality with isolated instance."""
    print("\nTesting cache warming...")
    
    cache = None
    
    try:
        from utils.cloudwatch_cache import CloudWatchMetadataCache
        
        # Create isolated cache instance for testing
        cache = CloudWatchMetadataCache()
        
        # Test warming functions exist
        warming_functions = cache._warming_functions
        assert 'alarms_metadata' in warming_functions, "Alarms warming function not found"
        assert 'dashboards_metadata' in warming_functions, "Dashboards warming function not found"
        assert 'log_groups_metadata' in warming_functions, "Log groups warming function not found"
        assert 'metrics_metadata' in warming_functions, "Metrics warming function not found"
        
        # Test cache warming execution (these should be mocked in real tests)
        success = cache.warm_cache('alarms_metadata', region='us-east-1')
        assert success is True, "Failed to warm alarms cache"
        
        success = cache.warm_cache('dashboards_metadata', region='us-east-1')
        assert success is True, "Failed to warm dashboards cache"
        
        print("✅ Cache warming working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in cache warming test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up isolated cache instance
        if cache and hasattr(cache, 'shutdown'):
            cache.shutdown()

def main():
    """Run all performance tests."""
    print("🚀 Starting CloudWatch Performance Tests\n")
    
    tests = [
        test_core_performance_components,
        test_cloudwatch_cache_operations,
        test_performance_monitoring,
        test_memory_management,
        test_progressive_timeouts,
        test_cache_warming
    ]
    
    passed = 0
    failed = 0
    
    try:
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
                failed += 1
        
        print(f"\n📊 CloudWatch Performance Test Results:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
        
        if failed == 0:
            print("\n🎉 All CloudWatch performance tests passed!")
            return True
        else:
            print(f"\n⚠️ {failed} CloudWatch performance test(s) failed.")
            return False
    
    except Exception as e:
        print(f"❌ Test suite failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)