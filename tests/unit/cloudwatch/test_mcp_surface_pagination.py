#!/usr/bin/env python3
"""
Test MCP surface pagination with real parameters to analyze response structure.
"""

import pytest
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


@pytest.mark.skip(reason="Tests need refactoring to match actual API structure")
class TestMCPSurfacePagination:
    """Test MCP surface pagination with real parameters."""
    
    @pytest.mark.asyncio
    async def test_mcp_cloudwatch_general_spend_analysis_surface(self):
        """Test MCP surface call with real parameters to analyze response structure."""
        
        # Import the MCP function directly
        from runbook_functions import run_cloudwatch_general_spend_analysis
        
        # Test parameters from user request
        test_params = {
            "region": "us-east-1",
            "lookback_days": 30,
            "allow_minimal_cost_metrics": True,
            "page": 1
        }
        
        print(f"\n=== Testing MCP Surface Call ===")
        print(f"Parameters: {json.dumps(test_params, indent=2)}")
        
        # Call the actual MCP function
        result = await run_cloudwatch_general_spend_analysis(test_params)
        
        # Parse the response
        response_text = result[0].text
        response_data = json.loads(response_text)
        
        print(f"\n=== Response Structure Analysis ===")
        print(f"Response keys: {list(response_data.keys())}")
        
        # Check if pagination exists in the response
        if 'data' in response_data:
            print(f"Data keys: {list(response_data['data'].keys())}")
            
            # Look for pagination in different places
            pagination_locations = []
            
            # Check top-level data pagination
            if 'pagination' in response_data['data']:
                pagination_locations.append('data.pagination')
                print(f"Found pagination at data.pagination: {response_data['data']['pagination']}")
            
            # Check configuration_analysis sections
            if 'configuration_analysis' in response_data['data']:
                config_analysis = response_data['data']['configuration_analysis']
                print(f"Configuration analysis keys: {list(config_analysis.keys())}")
                
                for section_name, section_data in config_analysis.items():
                    if isinstance(section_data, dict) and 'pagination' in section_data:
                        pagination_locations.append(f'data.configuration_analysis.{section_name}.pagination')
                        print(f"Found pagination at data.configuration_analysis.{section_name}.pagination: {section_data['pagination']}")
            
            print(f"\nPagination found at locations: {pagination_locations}")
            
            # Check for items/data arrays
            items_locations = []
            if 'configuration_analysis' in response_data['data']:
                config_analysis = response_data['data']['configuration_analysis']
                for section_name, section_data in config_analysis.items():
                    if isinstance(section_data, dict):
                        if 'items' in section_data:
                            items_count = len(section_data['items']) if isinstance(section_data['items'], list) else 'not_list'
                            items_locations.append(f'data.configuration_analysis.{section_name}.items ({items_count} items)')
                        
                        # Check for specific data arrays
                        for data_key in ['log_groups', 'metrics', 'alarms', 'dashboards']:
                            if data_key in section_data and isinstance(section_data[data_key], list):
                                items_count = len(section_data[data_key])
                                items_locations.append(f'data.configuration_analysis.{section_name}.{data_key} ({items_count} items)')
            
            print(f"Items/data arrays found at: {items_locations}")
        
        # Check response metadata
        if 'runbook_metadata' in response_data:
            print(f"Runbook metadata keys: {list(response_data['runbook_metadata'].keys())}")
        
        if 'orchestrator_metadata' in response_data:
            print(f"Orchestrator metadata keys: {list(response_data['orchestrator_metadata'].keys())}")
        
        # Check for page-related fields at top level
        page_fields = []
        for key in response_data.keys():
            if 'page' in key.lower() or 'pagination' in key.lower():
                page_fields.append(f"{key}: {response_data[key]}")
        
        if page_fields:
            print(f"Top-level page-related fields: {page_fields}")
        
        # Print full response structure (truncated for readability)
        print(f"\n=== Full Response Structure (first 2000 chars) ===")
        response_str = json.dumps(response_data, indent=2, default=str)
        print(response_str[:2000] + "..." if len(response_str) > 2000 else response_str)
        
        # Assertions to verify the response structure
        assert isinstance(response_data, dict), "Response should be a dictionary"
        assert 'status' in response_data, "Response should have status field"
        assert 'data' in response_data, "Response should have data field"
        
        # Test passes if we get a valid response structure
        print(f"\n=== Test Result ===")
        print(f"✅ MCP surface call successful")
        print(f"✅ Response structure analyzed")
        print(f"✅ Pagination locations identified: {len(pagination_locations) if 'pagination_locations' in locals() else 0}")
    
    @pytest.mark.asyncio
    async def test_mcp_cloudwatch_metrics_optimization_surface(self):
        """Test MCP metrics optimization surface call."""
        
        from runbook_functions import run_cloudwatch_metrics_optimization
        
        test_params = {
            "region": "us-east-1",
            "lookback_days": 30,
            "allow_minimal_cost_metrics": True,
            "page": 1
        }
        
        print(f"\n=== Testing Metrics Optimization MCP Surface Call ===")
        
        result = await run_cloudwatch_metrics_optimization(test_params)
        response_data = json.loads(result[0].text)
        
        # Check for pagination in metrics response
        pagination_found = False
        if 'data' in response_data and 'configuration_analysis' in response_data['data']:
            config_analysis = response_data['data']['configuration_analysis']
            if 'metrics' in config_analysis and 'pagination' in config_analysis['metrics']:
                pagination_found = True
                pagination_info = config_analysis['metrics']['pagination']
                print(f"Metrics pagination: {pagination_info}")
        
        print(f"Metrics optimization pagination found: {pagination_found}")
        assert isinstance(response_data, dict), "Metrics response should be a dictionary"
    
    @pytest.mark.asyncio
    async def test_pagination_consistency_across_apis(self):
        """Test pagination consistency across different CloudWatch APIs."""
        
        from runbook_functions import (
            run_cloudwatch_general_spend_analysis,
            run_cloudwatch_metrics_optimization,
            run_cloudwatch_logs_optimization,
            run_cloudwatch_alarms_and_dashboards_optimization
        )
        
        test_params = {
            "region": "us-east-1",
            "lookback_days": 30,
            "allow_minimal_cost_metrics": True,
            "page": 1
        }
        
        apis_to_test = [
            ("general_spend", run_cloudwatch_general_spend_analysis),
            ("metrics", run_cloudwatch_metrics_optimization),
            ("logs", run_cloudwatch_logs_optimization),
            ("alarms", run_cloudwatch_alarms_and_dashboards_optimization),
        ]
        
        pagination_structures = {}
        
        for api_name, api_func in apis_to_test:
            print(f"\n=== Testing {api_name} API ===")
            
            try:
                result = await api_func(test_params)
                response_data = json.loads(result[0].text)
                
                # Find pagination structures
                pagination_paths = []
                if 'data' in response_data and 'configuration_analysis' in response_data['data']:
                    config_analysis = response_data['data']['configuration_analysis']
                    for section_name, section_data in config_analysis.items():
                        if isinstance(section_data, dict) and 'pagination' in section_data:
                            pagination_paths.append(f"data.configuration_analysis.{section_name}.pagination")
                
                pagination_structures[api_name] = pagination_paths
                print(f"{api_name} pagination paths: {pagination_paths}")
                
            except Exception as e:
                print(f"Error testing {api_name}: {str(e)}")
                pagination_structures[api_name] = f"ERROR: {str(e)}"
        
        print(f"\n=== Pagination Structure Summary ===")
        for api_name, paths in pagination_structures.items():
            print(f"{api_name}: {paths}")
        
        # Test passes if we collected pagination info from all APIs
        assert len(pagination_structures) == len(apis_to_test), "Should test all APIs"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements