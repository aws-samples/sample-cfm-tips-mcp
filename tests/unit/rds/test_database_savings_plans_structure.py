"""
Unit tests for Database Savings Plans module structure and data models.

Tests the basic structure, data model instantiation, and core function signatures.
"""

import pytest
from datetime import datetime
from playbooks.rds.database_savings_plans import (
    ServiceUsage,
    DatabaseUsageData,
    SavingsPlanRecommendation,
    PurchaseAnalyzerResult,
    CommitmentComparison,
    ExistingSavingsPlan,
    analyze_database_usage,
    generate_savings_plans_recommendations,
    analyze_custom_commitment,
    compare_with_reserved_instances,
    analyze_existing_commitments
)


class TestDataModels:
    """Test data model instantiation and structure."""
    
    def test_service_usage_creation(self):
        """Test ServiceUsage data model can be instantiated."""
        service = ServiceUsage(
            service_name="rds",
            total_spend=1000.0,
            average_hourly_spend=1.39,
            instance_types={"db.t3.micro": 500.0},
            regions={"us-east-1": 1000.0}
        )
        
        assert service.service_name == "rds"
        assert service.total_spend == 1000.0
        assert service.average_hourly_spend == 1.39
        assert "db.t3.micro" in service.instance_types
        assert "us-east-1" in service.regions
    
    def test_database_usage_data_creation(self):
        """Test DatabaseUsageData data model can be instantiated."""
        service = ServiceUsage(
            service_name="rds",
            total_spend=1000.0,
            average_hourly_spend=1.39
        )
        
        usage_data = DatabaseUsageData(
            total_on_demand_spend=1000.0,
            average_hourly_spend=1.39,
            lookback_period_days=30,
            service_breakdown={"rds": service},
            region_breakdown={"us-east-1": 1000.0},
            instance_family_breakdown={"r5": 500.0},
            analysis_timestamp=datetime.now()
        )
        
        assert usage_data.total_on_demand_spend == 1000.0
        assert usage_data.lookback_period_days == 30
        assert "rds" in usage_data.service_breakdown
        assert isinstance(usage_data.analysis_timestamp, datetime)
    
    def test_savings_plan_recommendation_creation(self):
        """Test SavingsPlanRecommendation data model can be instantiated."""
        recommendation = SavingsPlanRecommendation(
            commitment_term="1_YEAR",
            payment_option="PARTIAL_UPFRONT",
            hourly_commitment=10.0,
            estimated_annual_savings=5000.0,
            estimated_monthly_savings=416.67,
            savings_percentage=25.0,
            projected_coverage=85.0,
            projected_utilization=95.0,
            break_even_months=8,
            confidence_level="high",
            upfront_cost=5000.0,
            total_commitment_cost=87600.0,
            rationale="Optimal commitment based on stable usage"
        )
        
        assert recommendation.commitment_term == "1_YEAR"
        assert recommendation.hourly_commitment == 10.0
        assert recommendation.confidence_level == "high"
        assert recommendation.projected_coverage == 85.0
    
    def test_purchase_analyzer_result_creation(self):
        """Test PurchaseAnalyzerResult data model can be instantiated."""
        result = PurchaseAnalyzerResult(
            hourly_commitment=10.0,
            commitment_term="1_YEAR",
            payment_option="PARTIAL_UPFRONT",
            projected_annual_cost=87600.0,
            projected_coverage=80.0,
            projected_utilization=92.5,
            estimated_annual_savings=12000.0,
            uncovered_on_demand_cost=20000.0,
            unused_commitment_cost=500.0
        )
        
        assert result.hourly_commitment == 10.0
        assert result.commitment_term == "1_YEAR"
        assert result.projected_coverage == 80.0
    
    def test_commitment_comparison_creation(self):
        """Test CommitmentComparison data model can be instantiated."""
        comparison = CommitmentComparison(
            service="rds",
            instance_type="db.r5.xlarge",
            savings_plan_cost=5000.0,
            ri_standard_cost=4800.0,
            ri_convertible_cost=5200.0,
            on_demand_cost=7000.0,
            recommended_option="Reserved Instance Standard",
            savings_plan_savings=2000.0,
            ri_standard_savings=2200.0,
            ri_convertible_savings=1800.0,
            flexibility_score=75.0,
            rationale="Lower cost for stable workload"
        )
        
        assert comparison.service == "rds"
        assert comparison.recommended_option == "Reserved Instance Standard"
        assert comparison.flexibility_score == 75.0
    
    def test_existing_savings_plan_creation(self):
        """Test ExistingSavingsPlan data model can be instantiated."""
        plan = ExistingSavingsPlan(
            savings_plan_id="sp-12345",
            savings_plan_arn="arn:aws:savingsplans::123456789012:savingsplan/sp-12345",
            hourly_commitment=15.0,
            commitment_term="1_YEAR",
            payment_option="PARTIAL_UPFRONT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 1, 1),
            utilization_percentage=88.5,
            coverage_percentage=75.0,
            unused_commitment_hourly=1.73,
            status="active"
        )
        
        assert plan.savings_plan_id == "sp-12345"
        assert plan.hourly_commitment == 15.0
        assert plan.utilization_percentage == 88.5
        assert plan.status == "active"


class TestCoreFunctions:
    """Test core analysis function signatures and basic behavior."""
    
    def test_analyze_database_usage_returns_dict(self):
        """Test analyze_database_usage returns expected structure."""
        result = analyze_database_usage(region="us-east-1", lookback_period_days=30)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert "message" in result
        assert result["status"] == "success"
    
    def test_analyze_database_usage_with_services(self):
        """Test analyze_database_usage accepts services parameter."""
        result = analyze_database_usage(
            region="us-east-1",
            lookback_period_days=30,
            services=["rds", "aurora"]
        )
        
        assert result["status"] == "success"
        assert "data" in result
    
    def test_generate_savings_plans_recommendations_returns_dict(self):
        """Test generate_savings_plans_recommendations returns expected structure."""
        usage_data = {"total_on_demand_spend": 1000.0}
        result = generate_savings_plans_recommendations(usage_data)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert "recommendations" in result["data"]
        assert isinstance(result["data"]["recommendations"], list)
    
    def test_generate_savings_plans_recommendations_with_custom_params(self):
        """Test generate_savings_plans_recommendations accepts custom parameters."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {}
        }
        result = generate_savings_plans_recommendations(
            usage_data,
            commitment_terms=["1_YEAR"],
            payment_options=["NO_UPFRONT"]
        )
        
        assert result["status"] == "success"
    
    def test_analyze_custom_commitment_returns_dict(self):
        """Test analyze_custom_commitment returns expected structure."""
        usage_data = {"total_on_demand_spend": 1000.0}
        result = analyze_custom_commitment(10.0, usage_data)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert result["data"]["hourly_commitment"] == 10.0
    
    def test_analyze_custom_commitment_with_all_params(self):
        """Test analyze_custom_commitment accepts all parameters."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {}
        }
        result = analyze_custom_commitment(
            hourly_commitment=15.0,
            usage_data=usage_data,
            commitment_term="1_YEAR",
            payment_option="NO_UPFRONT"
        )
        
        assert result["status"] == "success"
        assert result["data"]["commitment_term"] == "1_YEAR"
        assert result["data"]["payment_option"] == "NO_UPFRONT"
    
    def test_compare_with_reserved_instances_returns_dict(self):
        """Test compare_with_reserved_instances returns expected structure."""
        usage_data = {"total_on_demand_spend": 1000.0}
        result = compare_with_reserved_instances(usage_data)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert "latest_generation" in result["data"]
        assert "older_generation" in result["data"]
        assert isinstance(result["data"]["latest_generation"], list)
        assert isinstance(result["data"]["older_generation"], list)
    
    def test_compare_with_reserved_instances_with_services(self):
        """Test compare_with_reserved_instances accepts services parameter."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {"db.r7": 500.0, "db.r5": 500.0}
        }
        result = compare_with_reserved_instances(
            usage_data,
            services=["rds", "aurora", "dynamodb"]
        )
        
        assert result["status"] == "success"
    
    def test_analyze_existing_commitments_returns_dict(self):
        """Test analyze_existing_commitments returns expected structure."""
        result = analyze_existing_commitments(region="us-east-1")
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "data" in result
        assert "existing_plans" in result["data"]
        assert "gaps" in result["data"]
        assert isinstance(result["data"]["existing_plans"], list)
    
    def test_analyze_existing_commitments_with_lookback(self):
        """Test analyze_existing_commitments accepts lookback_period_days."""
        result = analyze_existing_commitments(
            region="us-east-1",
            lookback_period_days=60
        )
        
        assert result["status"] in ["success", "info"]  # "info" when no existing plans found


class TestDataModelDefaults:
    """Test data model default values."""
    
    def test_service_usage_default_dicts(self):
        """Test ServiceUsage has default empty dicts for optional fields."""
        service = ServiceUsage(
            service_name="rds",
            total_spend=1000.0,
            average_hourly_spend=1.39
        )
        
        assert isinstance(service.instance_types, dict)
        assert isinstance(service.regions, dict)
        assert len(service.instance_types) == 0
        assert len(service.regions) == 0
    
    def test_generate_recommendations_default_params(self):
        """Test generate_savings_plans_recommendations uses correct defaults."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {}
        }
        result = generate_savings_plans_recommendations(usage_data)
        
        # Function should handle None defaults internally
        assert result["status"] == "success"
    
    def test_analyze_custom_commitment_default_params(self):
        """Test analyze_custom_commitment uses correct defaults."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {}
        }
        result = analyze_custom_commitment(10.0, usage_data)
        
        assert result["data"]["commitment_term"] == "1_YEAR"
        assert result["data"]["payment_option"] == "NO_UPFRONT"
    
    def test_compare_with_ri_default_services(self):
        """Test compare_with_reserved_instances uses correct default services."""
        usage_data = {
            "total_on_demand_spend": 1000.0,
            "average_hourly_spend": 1.39,
            "lookback_period_days": 30,
            "service_breakdown": {"Amazon Relational Database Service": 1000.0},
            "region_breakdown": {"us-east-1": 1000.0},
            "instance_family_breakdown": {"db.r7": 500.0, "db.r5": 500.0}
        }
        result = compare_with_reserved_instances(usage_data)
        
        # Function should handle None default internally
        assert result["status"] == "success"
