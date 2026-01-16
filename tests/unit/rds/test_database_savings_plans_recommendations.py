"""
Property-based tests for Database Savings Plans recommendations engine.

These tests verify universal properties for the recommendations generation
functionality using the Hypothesis library for property-based testing.
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from playbooks.rds.database_savings_plans import (
    generate_savings_plans_recommendations,
    analyze_database_usage
)


class TestRecommendationTermCoverage:
    """
    Property 6: Recommendation term coverage
    Feature: database-savings-plans, Property 6: Recommendation term coverage
    
    For any usage data, generating recommendations should produce recommendations 
    for 1-year commitment terms (Database Savings Plans only support 1-year).
    Validates: Requirements 2.1
    """
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0),
        lookback_days=st.sampled_from([30, 60, 90])
    )
    def test_recommendations_include_one_year_terms(self, average_hourly_spend, lookback_days):
        """
        Test that recommendations include 1-year terms.
        
        Property: For any usage data with positive spend, generating recommendations
        should produce at least one recommendation for 1-year term (Database Savings Plans
        only support 1-year terms).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * lookback_days * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": lookback_days,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success', \
            f"Expected success status, got {result.get('status')}"
        
        recommendations = result['data']['recommendations']
        
        # Property: Should have recommendations
        assert len(recommendations) > 0, \
            "Should generate at least one recommendation for positive spend"
        
        # Property: Should have 1-year terms (Database Savings Plans only support 1-year)
        terms_found = set(rec['commitment_term'] for rec in recommendations)
        
        assert '1_YEAR' in terms_found, \
            "Recommendations should include 1-year term"
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=500.0)
    )
    def test_default_terms_are_one_year(self, average_hourly_spend):
        """
        Test that default commitment terms are 1-year.
        
        Property: When no commitment_terms parameter is provided, the function
        should default to generating recommendations for 1-year terms (Database Savings Plans
        only support 1-year terms).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations with default parameters
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Should have recommendations for 1-year term (Database Savings Plans only)
        terms_found = set(rec['commitment_term'] for rec in recommendations)
        
        assert '1_YEAR' in terms_found, "Default should include 1-year term"
        
        # Property: Should not have other terms (Database Savings Plans only support 1-year)
        valid_terms = {'1_YEAR'}
        assert terms_found.issubset(valid_terms), \
            f"Should only have 1-year terms, found {terms_found}"

    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0),
        num_payment_options=st.integers(min_value=1, max_value=3)
    )
    def test_one_year_term_has_recommendations(self, average_hourly_spend, num_payment_options):
        """
        Test that 1-year term has at least one recommendation.
        
        Property: For any usage data and payment options, the 1-year commitment term
        should have at least one recommendation generated (Database Savings Plans only support 1-year).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Select payment options
        all_payment_options = ['ALL_UPFRONT', 'PARTIAL_UPFRONT', 'NO_UPFRONT']
        payment_options = all_payment_options[:num_payment_options]
        
        # Generate recommendations (Database Savings Plans only support 1-year)
        result = generate_savings_plans_recommendations(
            usage_data,
            commitment_terms=['1_YEAR'],
            payment_options=payment_options
        )
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: 1-year term should have at least one recommendation
        one_year_recs = [r for r in recommendations if r['commitment_term'] == '1_YEAR']
        
        assert len(one_year_recs) > 0, "Should have at least one 1-year recommendation"


class TestCommitmentOptimizationBounds:
    """
    Property 7: Commitment optimization bounds
    Feature: database-savings-plans, Property 7: Commitment optimization bounds
    
    For any usage data, recommended hourly commitment amounts should not exceed 
    the average hourly on-demand spend.
    Validates: Requirements 2.2
    """
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0),
        lookback_days=st.sampled_from([30, 60, 90])
    )
    def test_commitment_does_not_exceed_average_spend(self, average_hourly_spend, lookback_days):
        """
        Test that recommended commitment does not exceed average hourly spend.
        
        Property: For any usage data, all recommended hourly commitments should
        be less than or equal to the average hourly on-demand spend.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * lookback_days * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": lookback_days,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: All commitments should be <= average hourly spend
        for rec in recommendations:
            hourly_commitment = rec['hourly_commitment']
            assert hourly_commitment <= average_hourly_spend, \
                f"Hourly commitment ${hourly_commitment} should not exceed " \
                f"average hourly spend ${average_hourly_spend}"

    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=500.0)
    )
    def test_commitment_is_positive(self, average_hourly_spend):
        """
        Test that recommended commitments are positive values.
        
        Property: For any positive average hourly spend, all recommended
        hourly commitments should be positive (greater than zero).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: All commitments should be positive
        for rec in recommendations:
            hourly_commitment = rec['hourly_commitment']
            assert hourly_commitment > 0, \
                f"Hourly commitment should be positive, got ${hourly_commitment}"
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_commitment_optimization_balances_coverage_and_risk(self, average_hourly_spend):
        """
        Test that commitment optimization balances coverage and over-commitment risk.
        
        Property: For any usage data, recommended commitments should cover
        a reasonable percentage (typically 80-95%) of average usage to balance
        savings with risk of over-commitment.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Commitments should be in reasonable range (70-100% of average)
        for rec in recommendations:
            hourly_commitment = rec['hourly_commitment']
            coverage_ratio = hourly_commitment / average_hourly_spend
            
            assert 0.70 <= coverage_ratio <= 1.0, \
                f"Commitment ratio {coverage_ratio:.2%} should be between 70% and 100% " \
                f"to balance savings and risk"


class TestRecommendationFieldCompleteness:
    """
    Property 8: Recommendation field completeness
    Feature: database-savings-plans, Property 8: Recommendation field completeness
    
    For any generated recommendation, it should contain estimated annual savings, 
    projected coverage percentage, projected utilization percentage, and break-even 
    timeline fields.
    Validates: Requirements 2.3
    """
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_all_required_fields_present(self, average_hourly_spend):
        """
        Test that all required fields are present in recommendations.
        
        Property: For any usage data, every recommendation should contain all
        required fields as specified in the requirements.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Required fields as per requirements
        required_fields = [
            'commitment_term',
            'payment_option',
            'hourly_commitment',
            'estimated_annual_savings',
            'estimated_monthly_savings',
            'savings_percentage',
            'projected_coverage',
            'projected_utilization',
            'break_even_months',
            'confidence_level',
            'upfront_cost',
            'total_commitment_cost',
            'rationale'
        ]
        
        # Property: Every recommendation should have all required fields
        for rec in recommendations:
            for field in required_fields:
                assert field in rec, \
                    f"Recommendation should contain '{field}' field"

    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=500.0)
    )
    def test_numeric_fields_have_valid_types(self, average_hourly_spend):
        """
        Test that numeric fields have appropriate types and values.
        
        Property: For any recommendation, numeric fields should be numbers
        (int or float) and have reasonable values.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Numeric fields should be numbers
        for rec in recommendations:
            # Check types
            assert isinstance(rec['hourly_commitment'], (int, float))
            assert isinstance(rec['estimated_annual_savings'], (int, float))
            assert isinstance(rec['estimated_monthly_savings'], (int, float))
            assert isinstance(rec['savings_percentage'], (int, float))
            assert isinstance(rec['projected_coverage'], (int, float))
            assert isinstance(rec['projected_utilization'], (int, float))
            assert isinstance(rec['break_even_months'], int)
            assert isinstance(rec['upfront_cost'], (int, float))
            assert isinstance(rec['total_commitment_cost'], (int, float))
            
            # Check non-negative values
            assert rec['hourly_commitment'] >= 0
            assert rec['estimated_annual_savings'] >= 0
            assert rec['estimated_monthly_savings'] >= 0
            assert rec['savings_percentage'] >= 0
            assert rec['projected_coverage'] >= 0
            assert rec['projected_utilization'] >= 0
            assert rec['break_even_months'] >= 0
            assert rec['upfront_cost'] >= 0
            assert rec['total_commitment_cost'] >= 0
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_string_fields_have_valid_values(self, average_hourly_spend):
        """
        Test that string fields have valid values.
        
        Property: For any recommendation, string fields should have values
        from the expected set of valid options.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Valid values for string fields
        valid_terms = {'1_YEAR', '3_YEAR'}
        valid_payment_options = {'ALL_UPFRONT', 'PARTIAL_UPFRONT', 'NO_UPFRONT'}
        valid_confidence_levels = {'high', 'medium', 'low'}
        
        # Property: String fields should have valid values
        for rec in recommendations:
            assert rec['commitment_term'] in valid_terms, \
                f"Invalid commitment term: {rec['commitment_term']}"
            assert rec['payment_option'] in valid_payment_options, \
                f"Invalid payment option: {rec['payment_option']}"
            assert rec['confidence_level'] in valid_confidence_levels, \
                f"Invalid confidence level: {rec['confidence_level']}"
            assert isinstance(rec['rationale'], str) and len(rec['rationale']) > 0, \
                "Rationale should be a non-empty string"


class TestSavingsCalculationCorrectness:
    """
    Property 9: Savings calculation correctness
    Feature: database-savings-plans, Property 9: Savings calculation correctness
    
    For any on-demand cost and savings plan rate, the calculated savings should 
    equal the difference between on-demand cost and savings plan cost.
    Validates: Requirements 2.4
    """
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_annual_savings_calculation(self, average_hourly_spend):
        """
        Test that annual savings are correctly calculated.
        
        Property: For any recommendation, the estimated annual savings should
        be consistent with the hourly commitment and discount rate.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Annual savings should be positive for positive spend
        for rec in recommendations:
            assert rec['estimated_annual_savings'] > 0, \
                "Annual savings should be positive for positive spend"
            
            # Property: Monthly savings * 12 should approximately equal annual savings
            monthly_savings = rec['estimated_monthly_savings']
            annual_savings = rec['estimated_annual_savings']
            
            expected_annual = monthly_savings * 12
            # Allow small rounding differences (up to $0.10 due to floating point)
            assert abs(annual_savings - expected_annual) < 0.10, \
                f"Annual savings ${annual_savings} should equal monthly * 12 = ${expected_annual}"

    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=500.0)
    )
    def test_savings_percentage_consistency(self, average_hourly_spend):
        """
        Test that savings percentage is consistent with dollar savings.
        
        Property: For any recommendation, the savings percentage should be
        consistent with the ratio of savings to total on-demand cost.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Savings percentage should be reasonable (0-50%)
        for rec in recommendations:
            savings_pct = rec['savings_percentage']
            
            # Database Savings Plans typically offer 20-42% savings
            assert 0 <= savings_pct <= 50, \
                f"Savings percentage {savings_pct}% should be between 0% and 50%"
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_three_year_savings_exceed_one_year(self, average_hourly_spend):
        """
        Test that 3-year plans generally offer higher savings than 1-year plans.
        
        Property: For any usage data and same payment option, 3-year commitment
        should typically offer higher percentage savings than 1-year commitment.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Group by payment option
        by_payment = {}
        for rec in recommendations:
            payment = rec['payment_option']
            term = rec['commitment_term']
            if payment not in by_payment:
                by_payment[payment] = {}
            by_payment[payment][term] = rec
        
        # Property: For each payment option, 3-year should have higher savings percentage
        for payment, terms in by_payment.items():
            if '1_YEAR' in terms and '3_YEAR' in terms:
                one_year_pct = terms['1_YEAR']['savings_percentage']
                three_year_pct = terms['3_YEAR']['savings_percentage']
                
                assert three_year_pct >= one_year_pct, \
                    f"3-year savings ({three_year_pct}%) should be >= 1-year ({one_year_pct}%) " \
                    f"for {payment}"


class TestPaymentOptionCoverage:
    """
    Property 10: Payment option coverage
    Feature: database-savings-plans, Property 10: Payment option coverage
    
    For any usage data, recommendations should include analysis for No Upfront 
    payment option (Database Savings Plans only support No Upfront for 1-year terms).
    Validates: Requirements 2.5
    """
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0)
    )
    def test_no_upfront_payment_option_included(self, average_hourly_spend):
        """
        Test that No Upfront payment option is included in recommendations.
        
        Property: For any usage data, recommendations should include No Upfront
        payment option (Database Savings Plans only support No Upfront for 1-year terms).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations with default parameters
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: No Upfront payment option should be present (Database Savings Plans only)
        payment_options_found = set(rec['payment_option'] for rec in recommendations)
        
        assert 'NO_UPFRONT' in payment_options_found, \
            "Recommendations should include No Upfront payment option"
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=500.0)
    )
    def test_no_upfront_cost_is_zero(self, average_hourly_spend):
        """
        Test that No Upfront payment option has zero upfront cost.
        
        Property: For any usage data, No Upfront payment option should have
        zero upfront cost (Database Savings Plans only support No Upfront).
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Generate recommendations
        result = generate_savings_plans_recommendations(usage_data)
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Group by term to compare payment options
        by_term = {}
        for rec in recommendations:
            term = rec['commitment_term']
            if term not in by_term:
                by_term[term] = {}
            by_term[term][rec['payment_option']] = rec
        
        # Property: No Upfront should have zero upfront cost
        for term, payment_recs in by_term.items():
            if 'NO_UPFRONT' in payment_recs:
                no_upfront_cost = payment_recs['NO_UPFRONT']['upfront_cost']
                assert no_upfront_cost == 0, \
                    "No Upfront should have zero upfront cost"
    
    @settings(max_examples=100)
    @given(
        average_hourly_spend=st.floats(min_value=0.01, max_value=1000.0),
        num_payment_options=st.integers(min_value=1, max_value=3)
    )
    def test_custom_payment_options_respected(self, average_hourly_spend, num_payment_options):
        """
        Test that custom payment options parameter is respected.
        
        Property: When specific payment options are requested, only those
        payment options should appear in the recommendations.
        """
        # Create mock usage data
        usage_data = {
            "total_on_demand_spend": average_hourly_spend * 30 * 24,
            "average_hourly_spend": average_hourly_spend,
            "lookback_period_days": 30,
            "service_breakdown": {},
            "region_breakdown": {},
            "instance_family_breakdown": {}
        }
        
        # Select payment options (Database Savings Plans only support NO_UPFRONT)
        all_payment_options = ['NO_UPFRONT']
        selected_payment_options = all_payment_options[:1]  # Only NO_UPFRONT available
        
        # Generate recommendations with custom payment options
        result = generate_savings_plans_recommendations(
            usage_data,
            payment_options=selected_payment_options
        )
        
        # Verify success
        assert result['status'] == 'success'
        
        recommendations = result['data']['recommendations']
        
        # Property: Only selected payment options should be present
        payment_options_found = set(rec['payment_option'] for rec in recommendations)
        
        for option in selected_payment_options:
            assert option in payment_options_found, \
                f"Selected payment option '{option}' should be in recommendations"
        
        # Property: No other payment options should be present
        assert payment_options_found.issubset(set(selected_payment_options)), \
            f"Found unexpected payment options: {payment_options_found - set(selected_payment_options)}"
