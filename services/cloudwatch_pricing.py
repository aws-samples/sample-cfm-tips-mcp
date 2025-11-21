"""
CloudWatch Pricing - Backward compatibility wrapper.

This module provides backward compatibility for code that expects CloudWatchPricing.
The actual pricing logic is now internal to cloudwatch_service.py via AWSPricingDAO.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CloudWatchPricing:
    """
    Backward compatibility wrapper for CloudWatch pricing.
    
    This class provides the same interface as before but delegates to the internal
    AWSPricingDAO class in cloudwatch_service.py.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        """Initialize pricing service."""
        self.region = region
        
        # Import here to avoid circular dependency
        from services.cloudwatch_service import AWSPricingDAO
        self._pricing_dao = AWSPricingDAO(region=region)
        
        logger.debug(f"CloudWatchPricing initialized for region: {region}")
    
    def get_pricing_data(self, component: str) -> Dict[str, Any]:
        """Get pricing data for CloudWatch components."""
        return self._pricing_dao.get_pricing_data(component)
    
    def get_free_tier_limits(self) -> Dict[str, Any]:
        """Get free tier limits for CloudWatch services."""
        return self._pricing_dao.get_free_tier_limits()
    
    def calculate_cost(self, component: str, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate costs for CloudWatch components."""
        return self._pricing_dao.calculate_cost(component, usage)
    
    def calculate_logs_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Logs costs."""
        pricing = self.get_pricing_data('logs')
        return self._pricing_dao._calculate_logs_cost(usage, pricing)
    
    def calculate_metrics_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Metrics costs."""
        pricing = self.get_pricing_data('metrics')
        return self._pricing_dao._calculate_metrics_cost(usage, pricing)
    
    def calculate_alarms_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Alarms costs."""
        pricing = self.get_pricing_data('alarms')
        return self._pricing_dao._calculate_alarms_cost(usage, pricing)
    
    def calculate_dashboards_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CloudWatch Dashboards costs."""
        pricing = self.get_pricing_data('dashboards')
        return self._pricing_dao._calculate_dashboards_cost(usage, pricing)
