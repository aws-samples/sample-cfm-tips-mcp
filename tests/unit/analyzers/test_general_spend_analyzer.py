"""
Unit tests for GeneralSpendAnalyzer class.

Tests the comprehensive S3 spending pattern analysis including storage costs,
data transfer costs, API costs, and retrieval costs with mocked AWS services.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from playbooks.s3.analyzers.general_spend_analyzer import GeneralSpendAnalyzer


@pytest.mark.unit
class TestGeneralSpendAnalyzer:
    """Test cases for GeneralSpendAnalyzer class."""
    
    def test_analyzer_initialization(self, mock_s3_service, mock_pricing_service, 
                                   mock_storage_lens_service):
        """Test analyzer initialization."""
        analyzer = GeneralSpendAnalyzer(
            s3_service=mock_s3_service,
            pricing_service=mock_pricing_service,
            storage_lens_service=mock_storage_lens_service
        )
        
        assert analyzer.s3_service == mock_s3_service
        assert analyzer.pricing_service == mock_pricing_service
        assert analyzer.storage_lens_service == mock_storage_lens_service
        assert analyzer.analysis_type == "general_spend"
    
    @pytest.mark.asyncio
    async def test_analyze_success_with_all_sources(self, mock_s3_service, 
                                                  mock_pricing_service, 
                                                  mock_storage_lens_service):
        """Test successful analysis with all data sources available."""
        # Mock storage lens service
        mock_storage_lens_service.get_storage_metrics = AsyncMock(return_value={
            "status": "success",
            "data": {
                "ConfigurationId": "test-config",
                "CostOptimizationMetrics": True,
                "StorageMetricsEnabled": True
            }
        })
        
        # Mock pricing service
        mock_pricing_service.get_storage_pricing.return_value = {
            "status": "success",
            "storage_pricing": {
                "STANDARD": 0.023,
                "STANDARD_IA": 0.0125
            }
        }
        
        analyzer = GeneralSpendAnalyzer(
            s3_service=mock_s3_service,
            pricing_service=mock_pricing_service,
            storage_lens_service=mock_storage_lens_service
        )
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-Storage-Standard"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "10.50", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "1000", "Unit": "GB-Mo"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30,
                include_cost_analysis=True
            )
        
        assert result["status"] == "success"
        assert result["analysis_type"] == "general_spend"
        assert "data" in result
        assert "storage_costs" in result["data"]
        assert "data_transfer_costs" in result["data"]
        assert "api_costs" in result["data"]
        assert "retrieval_costs" in result["data"]
        assert "total_costs" in result["data"]
        assert "cost_breakdown" in result["data"]
        assert "optimization_opportunities" in result["data"]
        assert "data_sources" in result
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_analyze_storage_lens_fallback_to_cost_explorer(self, mock_s3_service, 
                                                                mock_pricing_service):
        """Test fallback to Cost Explorer when Storage Lens is unavailable."""
        # No storage lens service provided
        analyzer = GeneralSpendAnalyzer(
            s3_service=mock_s3_service,
            pricing_service=mock_pricing_service,
            storage_lens_service=None
        )
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-Storage-Standard"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "15.75", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "1500", "Unit": "GB-Mo"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30
            )
        
        assert result["status"] == "success"
        assert "cost_explorer" in result["data_sources"]
        assert "storage_lens" not in result["data_sources"]
    
    @pytest.mark.asyncio
    async def test_analyze_storage_costs_success(self, mock_storage_lens_service):
        """Test storage cost analysis with Storage Lens."""
        mock_storage_lens_service.get_storage_metrics = AsyncMock(return_value={
            "status": "success",
            "data": {
                "ConfigurationId": "test-config",
                "StorageMetricsEnabled": True,
                "CostOptimizationMetrics": True
            }
        })
        
        analyzer = GeneralSpendAnalyzer(storage_lens_service=mock_storage_lens_service)
        context = {"region": "us-east-1", "lookback_days": 30}
        
        result = await analyzer._analyze_storage_costs(context)
        
        assert result["status"] == "success"
        assert "storage_lens" in result["data_sources"]
        assert "by_storage_class" in result["data"]
        assert "by_object_size" in result["data"]
        assert "total_storage_cost" in result["data"]
    
    @pytest.mark.asyncio
    async def test_analyze_storage_costs_cost_explorer_fallback(self):
        """Test storage cost analysis fallback to Cost Explorer."""
        analyzer = GeneralSpendAnalyzer()  # No services
        context = {"region": "us-east-1", "lookback_days": 30}
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-Storage-StandardIA"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "8.25", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "800", "Unit": "GB-Mo"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer._analyze_storage_costs(context)
        
        assert result["status"] == "success"
        assert "cost_explorer" in result["data_sources"]
    
    @pytest.mark.asyncio
    async def test_analyze_data_transfer_costs(self):
        """Test data transfer cost analysis."""
        analyzer = GeneralSpendAnalyzer()
        context = {"region": "us-east-1", "lookback_days": 30}
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-DataTransfer-Out-Bytes"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "5.50", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "100", "Unit": "GB"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer._analyze_data_transfer_costs(context)
        
        assert result["status"] == "success"
        assert "cost_explorer" in result["data_sources"]
        assert "cross_region_transfer" in result["data"]
        assert "internet_egress" in result["data"]
        assert "total_transfer_cost" in result["data"]
    
    @pytest.mark.asyncio
    async def test_analyze_api_costs(self, mock_s3_service):
        """Test API cost analysis."""
        analyzer = GeneralSpendAnalyzer(s3_service=mock_s3_service)
        context = {"region": "us-east-1", "lookback_days": 30, "bucket_names": ["test-bucket"]}
        
        # Mock request metrics
        async def mock_get_request_metrics(bucket_names, days):
            return {
                "GET": 10000,
                "PUT": 1000,
                "LIST": 100,
                "DELETE": 50
            }
        
        analyzer._get_request_metrics_for_buckets = mock_get_request_metrics
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-API-Tier1"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "2.50", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "5000", "Unit": "Requests"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer._analyze_api_costs(context)
        
        assert result["status"] == "success"
        assert "cost_explorer" in result["data_sources"]
        assert "cloudwatch" in result["data_sources"]
        assert "request_costs_by_type" in result["data"]
        assert "total_api_cost" in result["data"]
    
    @pytest.mark.asyncio
    async def test_analyze_retrieval_costs(self):
        """Test retrieval cost analysis."""
        analyzer = GeneralSpendAnalyzer()
        context = {"region": "us-east-1", "lookback_days": 30}
        
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.return_value = {
                "status": "success",
                "data": {
                    "ResultsByTime": [
                        {
                            "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                            "Groups": [
                                {
                                    "Keys": ["S3-Retrieval-Bytes"],
                                    "Metrics": {
                                        "UnblendedCost": {"Amount": "1.25", "Unit": "USD"},
                                        "UsageQuantity": {"Amount": "50", "Unit": "GB"}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = await analyzer._analyze_retrieval_costs(context)
        
        assert result["status"] == "success"
        assert "cost_explorer" in result["data_sources"]
        assert "glacier_retrievals" in result["data"]
        assert "deep_archive_retrievals" in result["data"]
        assert "total_retrieval_cost" in result["data"]
    
    def test_process_storage_lens_data(self):
        """Test processing of Storage Lens data."""
        analyzer = GeneralSpendAnalyzer()
        
        storage_lens_data = {
            "CostOptimizationMetricsEnabled": True,
            "StorageMetricsEnabled": True
        }
        
        result = analyzer._process_storage_lens_data(storage_lens_data)
        
        assert "by_storage_class" in result
        assert "storage_distribution" in result
        assert result["cost_optimization_enabled"] is True
        assert result["data_source"] == "storage_lens"
        assert result["storage_metrics_available"] is True
    
    def test_extract_storage_class_from_usage_type(self):
        """Test storage class extraction from usage type."""
        analyzer = GeneralSpendAnalyzer()
        
        test_cases = [
            ("S3-Storage-Standard", "STANDARD"),
            ("S3-Storage-StandardIA", "STANDARD_IA"),
            ("S3-Storage-OneZone-IA", "ONEZONE_IA"),
            ("S3-Storage-Glacier", "GLACIER"),
            ("S3-Storage-DeepArchive", "DEEP_ARCHIVE"),
            ("Unknown-Usage-Type", "UNKNOWN")
        ]
        
        for usage_type, expected_class in test_cases:
            result = analyzer._extract_storage_class_from_usage_type(usage_type)
            assert result == expected_class
    
    def test_calculate_total_costs(self):
        """Test total cost calculation."""
        analyzer = GeneralSpendAnalyzer()
        
        data = {
            "storage_costs": {"total_storage_cost": 100.0},
            "data_transfer_costs": {"total_transfer_cost": 25.0},
            "api_costs": {"total_api_cost": 10.0},
            "retrieval_costs": {"total_retrieval_cost": 5.0}
        }
        
        result = analyzer._calculate_total_costs(data)
        
        assert result["total_monthly_cost"] == 140.0
        assert result["storage_percentage"] == 71.43  # 100/140 * 100
        assert result["transfer_percentage"] == 17.86  # 25/140 * 100
        assert result["api_percentage"] == 7.14      # 10/140 * 100
        assert result["retrieval_percentage"] == 3.57  # 5/140 * 100
    
    def test_create_cost_breakdown(self):
        """Test cost breakdown creation."""
        analyzer = GeneralSpendAnalyzer()
        
        data = {
            "storage_costs": {
                "by_storage_class": {
                    "STANDARD": {"total_cost": 80.0},
                    "STANDARD_IA": {"total_cost": 20.0}
                }
            },
            "data_transfer_costs": {"total_transfer_cost": 25.0},
            "api_costs": {"total_api_cost": 10.0},
            "retrieval_costs": {"total_retrieval_cost": 5.0}
        }
        
        result = analyzer._create_cost_breakdown(data)
        
        assert "by_category" in result
        assert "by_storage_class" in result
        assert result["by_category"]["storage"] == 100.0
        assert result["by_category"]["transfer"] == 25.0
        assert result["by_storage_class"]["STANDARD"] == 80.0
        assert result["by_storage_class"]["STANDARD_IA"] == 20.0
    
    def test_identify_optimization_opportunities(self):
        """Test optimization opportunity identification."""
        analyzer = GeneralSpendAnalyzer()
        
        data = {
            "storage_costs": {
                "by_storage_class": {
                    "STANDARD": {"total_cost": 100.0, "total_usage_gb": 1000}
                }
            },
            "data_transfer_costs": {"total_transfer_cost": 50.0},
            "api_costs": {"total_api_cost": 20.0},
            "retrieval_costs": {"total_retrieval_cost": 2.0}
        }
        
        result = analyzer._identify_optimization_opportunities(data)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Should identify high storage costs as optimization opportunity
        storage_opportunity = next((opp for opp in result if "storage class" in opp["title"].lower()), None)
        assert storage_opportunity is not None
        assert storage_opportunity["priority"] in ["high", "medium", "low"]
    
    def test_get_recommendations(self):
        """Test recommendation generation."""
        analyzer = GeneralSpendAnalyzer()
        
        analysis_results = {
            "data": {
                "total_costs": {"total_monthly_cost": 200.0},
                "optimization_opportunities": [
                    {
                        "type": "storage_class_optimization",
                        "potential_savings": 50.0,
                        "affected_buckets": ["bucket1", "bucket2"]
                    }
                ]
            }
        }
        
        recommendations = analyzer.get_recommendations(analysis_results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        rec = recommendations[0]
        assert "type" in rec
        assert "priority" in rec
        assert "title" in rec
        assert "description" in rec
        assert rec["analyzer"] == "general_spend"
    
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self):
        """Test error handling in analysis."""
        analyzer = GeneralSpendAnalyzer()
        
        # Mock Cost Explorer to raise an exception
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.side_effect = Exception("Cost Explorer API error")
            
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30
            )
        
        assert result["status"] == "error"
        assert "Cost Explorer API error" in result["message"]
        assert result["analysis_type"] == "general_spend"
    
    @pytest.mark.asyncio
    async def test_analyze_partial_failure(self, mock_storage_lens_service):
        """Test analysis with partial component failures."""
        # Mock storage lens to succeed
        mock_storage_lens_service.get_storage_metrics = AsyncMock(return_value={
            "status": "success",
            "data": {"ConfigurationId": "test-config"}
        })
        
        analyzer = GeneralSpendAnalyzer(storage_lens_service=mock_storage_lens_service)
        
        # Mock Cost Explorer to fail for some components
        with patch('services.cost_explorer.get_cost_and_usage') as mock_cost_explorer:
            mock_cost_explorer.side_effect = [
                {"status": "success", "data": {"ResultsByTime": []}},  # Storage costs succeed
                Exception("Transfer cost API error"),  # Transfer costs fail
                {"status": "success", "data": {"ResultsByTime": []}},  # API costs succeed
                {"status": "success", "data": {"ResultsByTime": []}}   # Retrieval costs succeed
            ]
            
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30
            )
        
        assert result["status"] == "success"  # Overall success despite partial failure
        assert "storage_costs" in result["data"]
        assert "error" in result["data"]["data_transfer_costs"]  # This component failed
        assert "api_costs" in result["data"]
        assert "retrieval_costs" in result["data"]


@pytest.mark.unit
class TestGeneralSpendAnalyzerPerformance:
    """Performance-related tests for GeneralSpendAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, performance_tracker):
        """Test that analysis components execute in parallel."""
        analyzer = GeneralSpendAnalyzer()
        
        # Mock all Cost Explorer calls to take some time
        async def slow_cost_explorer_call(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API delay
            return {"status": "success", "data": {"ResultsByTime": []}}
        
        with patch('services.cost_explorer.get_cost_and_usage', side_effect=slow_cost_explorer_call):
            performance_tracker.start_timer("analysis")
            
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30
            )
            
            execution_time = performance_tracker.end_timer("analysis")
        
        assert result["status"] == "success"
        # With 4 parallel components each taking 0.1s, total should be closer to 0.1s than 0.4s
        assert execution_time < 0.3  # Allow some overhead
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of long-running operations."""
        analyzer = GeneralSpendAnalyzer()
        
        # Mock a very slow operation
        async def very_slow_operation(*args, **kwargs):
            await asyncio.sleep(10)  # Very long delay
            return {"status": "success", "data": {"ResultsByTime": []}}
        
        with patch('services.cost_explorer.get_cost_and_usage', side_effect=very_slow_operation):
            # This should complete quickly due to timeout handling in the orchestrator
            result = await analyzer.analyze(
                region="us-east-1",
                lookback_days=30,
                timeout_seconds=1.0
            )
        
        # The analysis should still return a result (possibly with errors)
        assert "status" in result
        assert "analysis_type" in result