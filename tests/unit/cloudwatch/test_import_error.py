"""
Test to replicate the CloudWatchServiceFactory import error.

This test verifies that the import error occurs when trying to import
CloudWatchServiceFactory from services.cloudwatch_service.
"""

import pytest


def test_cloudwatch_service_factory_import_error():
    """Test that CloudWatchServiceFactory import fails as expected."""
    with pytest.raises(ImportError, match="cannot import name 'CloudWatchServiceFactory'"):
        from services.cloudwatch_service import CloudWatchServiceFactory


def test_cloudwatch_optimization_analyzer_import_success():
    """Test that CloudWatchOptimizationAnalyzer import now works after fix."""
    from playbooks.cloudwatch.cloudwatch_optimization_analyzer import CloudWatchOptimizationAnalyzer
    assert CloudWatchOptimizationAnalyzer is not None


def test_correct_imports_work():
    """Test that correct imports from cloudwatch_service work."""
    from services.cloudwatch_service import (
        CWGeneralSpendTips,
        CWMetricsTips,
        CWLogsTips,
        CWAlarmsTips,
        CWDashboardTips,
        CloudWatchService,
        create_cloudwatch_service
    )
    
    # Verify all imports are classes/functions
    assert CWGeneralSpendTips is not None
    assert CWMetricsTips is not None
    assert CWLogsTips is not None
    assert CWAlarmsTips is not None
    assert CWDashboardTips is not None
    assert CloudWatchService is not None
    assert create_cloudwatch_service is not None
