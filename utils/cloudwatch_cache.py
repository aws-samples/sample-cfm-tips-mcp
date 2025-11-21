"""
CloudWatch-specific intelligent cache implementation.

Provides specialized caching for CloudWatch metadata, pricing data, and analysis results
with CloudWatch-specific optimization patterns and cache warming strategies.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from utils.intelligent_cache import IntelligentCache

logger = logging.getLogger(__name__)


class CloudWatchMetadataCache(IntelligentCache):
    """Specialized cache for CloudWatch metadata with optimized TTL and warming strategies."""
    
    def __init__(self):
        # CloudWatch metadata changes less frequently than other data
        super().__init__(
            max_size_mb=30,
            max_entries=3000,
            default_ttl_seconds=1800,  # 30 minutes
            cleanup_interval_minutes=10
        )
        
        # Register CloudWatch-specific warming functions
        self.register_warming_function("alarms_metadata", self._warm_alarms_metadata)
        self.register_warming_function("dashboards_metadata", self._warm_dashboards_metadata)
        self.register_warming_function("log_groups_metadata", self._warm_log_groups_metadata)
        self.register_warming_function("metrics_metadata", self._warm_metrics_metadata)
        
        logger.info("CloudWatchMetadataCache initialized")
    
    def _warm_alarms_metadata(self, cache, region: str = "us-east-1"):
        """Warm cache with common alarm metadata patterns."""
        try:
            logger.info(f"Warming CloudWatch alarms metadata cache for region: {region}")
            
            # Common alarm metadata patterns
            patterns = [
                f"alarms_list_{region}",
                f"alarm_states_{region}",
                f"alarm_actions_{region}",
                f"alarm_history_{region}",
                f"composite_alarms_{region}"
            ]
            
            for pattern in patterns:
                cache_key = ["cloudwatch_metadata", "alarms", pattern]
                self.put(cache_key, {
                    "metadata_type": "alarms",
                    "pattern": pattern,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=2400)  # 40 minutes for metadata
            
            logger.info(f"Warmed {len(patterns)} alarm metadata patterns")
            
        except Exception as e:
            logger.error(f"Error warming alarms metadata cache: {str(e)}")
    
    def _warm_dashboards_metadata(self, cache, region: str = "us-east-1"):
        """Warm cache with common dashboard metadata patterns."""
        try:
            logger.info(f"Warming CloudWatch dashboards metadata cache for region: {region}")
            
            patterns = [
                f"dashboards_list_{region}",
                f"dashboard_widgets_{region}",
                f"dashboard_metrics_{region}",
                f"dashboard_permissions_{region}"
            ]
            
            for pattern in patterns:
                cache_key = ["cloudwatch_metadata", "dashboards", pattern]
                self.put(cache_key, {
                    "metadata_type": "dashboards",
                    "pattern": pattern,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=2400)
            
            logger.info(f"Warmed {len(patterns)} dashboard metadata patterns")
            
        except Exception as e:
            logger.error(f"Error warming dashboards metadata cache: {str(e)}")
    
    def _warm_log_groups_metadata(self, cache, region: str = "us-east-1"):
        """Warm cache with common log groups metadata patterns."""
        try:
            logger.info(f"Warming CloudWatch log groups metadata cache for region: {region}")
            
            patterns = [
                f"log_groups_list_{region}",
                f"log_group_retention_{region}",
                f"log_group_sizes_{region}",
                f"log_streams_{region}",
                f"log_group_metrics_{region}"
            ]
            
            for pattern in patterns:
                cache_key = ["cloudwatch_metadata", "log_groups", pattern]
                self.put(cache_key, {
                    "metadata_type": "log_groups",
                    "pattern": pattern,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=1800)  # 30 minutes for log groups (change more frequently)
            
            logger.info(f"Warmed {len(patterns)} log groups metadata patterns")
            
        except Exception as e:
            logger.error(f"Error warming log groups metadata cache: {str(e)}")
    
    def _warm_metrics_metadata(self, cache, region: str = "us-east-1"):
        """Warm cache with common metrics metadata patterns."""
        try:
            logger.info(f"Warming CloudWatch metrics metadata cache for region: {region}")
            
            patterns = [
                f"metrics_list_{region}",
                f"custom_metrics_{region}",
                f"metrics_namespaces_{region}",
                f"metrics_dimensions_{region}",
                f"metrics_statistics_{region}"
            ]
            
            for pattern in patterns:
                cache_key = ["cloudwatch_metadata", "metrics", pattern]
                self.put(cache_key, {
                    "metadata_type": "metrics",
                    "pattern": pattern,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=3600)  # 1 hour for metrics metadata
            
            logger.info(f"Warmed {len(patterns)} metrics metadata patterns")
            
        except Exception as e:
            logger.error(f"Error warming metrics metadata cache: {str(e)}")
    
    def get_alarm_metadata(self, region: str, alarm_name: str = None) -> Optional[Dict[str, Any]]:
        """Get cached alarm metadata with intelligent key generation."""
        if alarm_name:
            cache_key = ["cloudwatch_metadata", "alarms", f"alarm_{alarm_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "alarms", f"alarms_list_{region}"]
        
        return self.get(cache_key)
    
    def put_alarm_metadata(self, region: str, data: Dict[str, Any], alarm_name: str = None, ttl_seconds: int = None):
        """Cache alarm metadata with intelligent key generation."""
        if alarm_name:
            cache_key = ["cloudwatch_metadata", "alarms", f"alarm_{alarm_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "alarms", f"alarms_list_{region}"]
        
        self.put(cache_key, data, ttl_seconds=ttl_seconds or 2400, 
                tags={"type": "alarm_metadata", "region": region})
    
    def get_dashboard_metadata(self, region: str, dashboard_name: str = None) -> Optional[Dict[str, Any]]:
        """Get cached dashboard metadata with intelligent key generation."""
        if dashboard_name:
            cache_key = ["cloudwatch_metadata", "dashboards", f"dashboard_{dashboard_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "dashboards", f"dashboards_list_{region}"]
        
        return self.get(cache_key)
    
    def put_dashboard_metadata(self, region: str, data: Dict[str, Any], dashboard_name: str = None, ttl_seconds: int = None):
        """Cache dashboard metadata with intelligent key generation."""
        if dashboard_name:
            cache_key = ["cloudwatch_metadata", "dashboards", f"dashboard_{dashboard_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "dashboards", f"dashboards_list_{region}"]
        
        self.put(cache_key, data, ttl_seconds=ttl_seconds or 2400,
                tags={"type": "dashboard_metadata", "region": region})
    
    def get_log_group_metadata(self, region: str, log_group_name: str = None) -> Optional[Dict[str, Any]]:
        """Get cached log group metadata with intelligent key generation."""
        if log_group_name:
            cache_key = ["cloudwatch_metadata", "log_groups", f"log_group_{log_group_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "log_groups", f"log_groups_list_{region}"]
        
        return self.get(cache_key)
    
    def put_log_group_metadata(self, region: str, data: Dict[str, Any], log_group_name: str = None, ttl_seconds: int = None):
        """Cache log group metadata with intelligent key generation."""
        if log_group_name:
            cache_key = ["cloudwatch_metadata", "log_groups", f"log_group_{log_group_name}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "log_groups", f"log_groups_list_{region}"]
        
        self.put(cache_key, data, ttl_seconds=ttl_seconds or 1800,
                tags={"type": "log_group_metadata", "region": region})
    
    def get_metrics_metadata(self, region: str, namespace: str = None) -> Optional[Dict[str, Any]]:
        """Get cached metrics metadata with intelligent key generation."""
        if namespace:
            cache_key = ["cloudwatch_metadata", "metrics", f"metrics_{namespace}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "metrics", f"metrics_list_{region}"]
        
        return self.get(cache_key)
    
    def put_metrics_metadata(self, region: str, data: Dict[str, Any], namespace: str = None, ttl_seconds: int = None):
        """Cache metrics metadata with intelligent key generation."""
        if namespace:
            cache_key = ["cloudwatch_metadata", "metrics", f"metrics_{namespace}_{region}"]
        else:
            cache_key = ["cloudwatch_metadata", "metrics", f"metrics_list_{region}"]
        
        self.put(cache_key, data, ttl_seconds=ttl_seconds or 3600,
                tags={"type": "metrics_metadata", "region": region})
    
    def invalidate_region_metadata(self, region: str) -> int:
        """Invalidate all metadata for a specific region."""
        return self.invalidate_by_tags({"region": region})
    
    def invalidate_metadata_type(self, metadata_type: str) -> int:
        """Invalidate all metadata of a specific type."""
        return self.invalidate_by_tags({"type": f"{metadata_type}_metadata"})


class CloudWatchAnalysisCache(IntelligentCache):
    """Specialized cache for CloudWatch analysis results with analysis-specific optimizations."""
    
    def __init__(self):
        # Analysis results can be large but are valuable to cache longer
        super().__init__(
            max_size_mb=150,
            max_entries=1500,
            default_ttl_seconds=3600,  # 1 hour
            cleanup_interval_minutes=15
        )
        
        # Register analysis-specific warming functions
        self.register_warming_function("common_analysis_patterns", self._warm_common_patterns)
        self.register_warming_function("cost_analysis_templates", self._warm_cost_templates)
        
        logger.info("CloudWatchAnalysisCache initialized")
    
    def _warm_common_patterns(self, cache, region: str = "us-east-1"):
        """Warm cache with common analysis patterns."""
        try:
            logger.info(f"Warming common CloudWatch analysis patterns for region: {region}")
            
            patterns = [
                f"general_spend_pattern_{region}",
                f"logs_optimization_pattern_{region}",
                f"metrics_optimization_pattern_{region}",
                f"alarms_dashboards_pattern_{region}",
                f"comprehensive_pattern_{region}"
            ]
            
            for pattern in patterns:
                cache_key = ["cloudwatch_analysis", "patterns", pattern]
                self.put(cache_key, {
                    "pattern_type": "analysis",
                    "pattern": pattern,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=7200)  # 2 hours for patterns
            
            logger.info(f"Warmed {len(patterns)} analysis patterns")
            
        except Exception as e:
            logger.error(f"Error warming analysis patterns cache: {str(e)}")
    
    def _warm_cost_templates(self, cache, region: str = "us-east-1"):
        """Warm cache with cost analysis templates."""
        try:
            logger.info(f"Warming cost analysis templates for region: {region}")
            
            templates = [
                f"cost_breakdown_template_{region}",
                f"savings_opportunities_template_{region}",
                f"cost_trends_template_{region}",
                f"optimization_recommendations_template_{region}"
            ]
            
            for template in templates:
                cache_key = ["cloudwatch_analysis", "cost_templates", template]
                self.put(cache_key, {
                    "template_type": "cost_analysis",
                    "template": template,
                    "region": region,
                    "warmed_at": datetime.now().isoformat(),
                    "cache_version": "1.0"
                }, ttl_seconds=5400)  # 1.5 hours for cost templates
            
            logger.info(f"Warmed {len(templates)} cost analysis templates")
            
        except Exception as e:
            logger.error(f"Error warming cost templates cache: {str(e)}")
    
    def get_analysis_result(self, analysis_type: str, region: str, parameters_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result with intelligent key generation."""
        cache_key = ["cloudwatch_analysis", analysis_type, f"{region}_{parameters_hash}"]
        return self.get(cache_key)
    
    def put_analysis_result(self, analysis_type: str, region: str, parameters_hash: str, 
                          result: Dict[str, Any], ttl_seconds: int = None):
        """Cache analysis result with intelligent key generation."""
        cache_key = ["cloudwatch_analysis", analysis_type, f"{region}_{parameters_hash}"]
        
        # Different TTL based on analysis type
        if not ttl_seconds:
            ttl_mapping = {
                'general_spend': 3600,  # 1 hour
                'logs_optimization': 1800,  # 30 minutes
                'metrics_optimization': 1800,  # 30 minutes
                'alarms_and_dashboards': 2400,  # 40 minutes
                'comprehensive': 3600  # 1 hour
            }
            ttl_seconds = ttl_mapping.get(analysis_type, 1800)
        
        self.put(cache_key, result, ttl_seconds=ttl_seconds,
                tags={"type": "analysis_result", "analysis_type": analysis_type, "region": region})
    
    def invalidate_analysis_type(self, analysis_type: str) -> int:
        """Invalidate all cached results for a specific analysis type."""
        return self.invalidate_by_tags({"analysis_type": analysis_type})
    
    def invalidate_region_analyses(self, region: str) -> int:
        """Invalidate all cached analyses for a specific region."""
        return self.invalidate_by_tags({"region": region})


# Global cache instances
_cloudwatch_metadata_cache = None
_cloudwatch_analysis_cache = None

def get_cloudwatch_metadata_cache() -> CloudWatchMetadataCache:
    """Get the global CloudWatch metadata cache instance."""
    global _cloudwatch_metadata_cache
    if _cloudwatch_metadata_cache is None:
        _cloudwatch_metadata_cache = CloudWatchMetadataCache()
    return _cloudwatch_metadata_cache

def get_cloudwatch_analysis_cache() -> CloudWatchAnalysisCache:
    """Get the global CloudWatch analysis cache instance."""
    global _cloudwatch_analysis_cache
    if _cloudwatch_analysis_cache is None:
        _cloudwatch_analysis_cache = CloudWatchAnalysisCache()
    return _cloudwatch_analysis_cache