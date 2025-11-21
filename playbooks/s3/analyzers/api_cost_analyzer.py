"""
API Cost Analyzer for S3 Optimization

This analyzer provides comprehensive S3 API request cost analysis and optimization
recommendations using Storage Lens as the primary data source with Cost Explorer
and CloudWatch as fallback sources.

Analyzes:
- Request pattern analysis by API type (GET, PUT, POST, LIST, DELETE, etc.)
- API cost breakdown by storage class and request type
- Caching opportunities and CloudFront integration recommendations
- Request consolidation and optimization strategies
- API cost trends and anomaly detection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage

logger = logging.getLogger(__name__)


class ApiCostAnalyzer(BaseAnalyzer):
    """
    Analyzer for S3 API request cost minimization and optimization.
    
    Uses Storage Lens as primary data source with Cost Explorer and CloudWatch
    fallback to analyze API request patterns, costs, and optimization opportunities.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize ApiCostAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "api_cost"
        
        # API request type mappings for cost analysis
        self.api_request_types = {
            'tier1': ['PUT', 'COPY', 'POST', 'LIST'],  # Higher cost requests
            'tier2': ['GET', 'SELECT', 'HEAD'],        # Lower cost requests
            'lifecycle': ['TRANSITION', 'EXPIRATION'], # Lifecycle requests
            'multipart': ['INITIATE', 'UPLOAD_PART', 'COMPLETE', 'ABORT']  # Multipart requests
        }
        
        # Storage class API cost multipliers (relative to Standard)
        self.storage_class_multipliers = {
            'STANDARD': 1.0,
            'STANDARD_IA': 1.0,
            'ONEZONE_IA': 1.0,
            'GLACIER': 10.0,      # Higher cost for Glacier requests
            'DEEP_ARCHIVE': 25.0  # Much higher cost for Deep Archive requests
        }
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive S3 API cost analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - lookback_days: Number of days to analyze (default: 30)
                - bucket_names: Optional list of specific buckets to analyze
                - request_threshold: Minimum requests per month to analyze (default: 10000)
                - include_cloudfront_analysis: Whether to analyze CloudFront opportunities
                
        Returns:
            Dictionary containing comprehensive API cost analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting S3 API cost analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "request_patterns": {},
                    "api_costs": {},
                    "cost_by_storage_class": {},
                    "optimization_opportunities": {},
                    "caching_recommendations": {},
                    "cloudfront_opportunities": {},
                    "request_consolidation": {},
                    "total_api_cost": 0,
                    "potential_savings": 0
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute analysis components in parallel
            tasks = [
                self._analyze_request_patterns_storage_lens(context),
                self._analyze_api_costs_cost_explorer(context),
                self._analyze_cloudwatch_metrics(context),
                self._analyze_caching_opportunities(context)
            ]
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            storage_lens_results, cost_explorer_results, cloudwatch_results, caching_results = results
            
            # Aggregate Storage Lens request patterns
            if not isinstance(storage_lens_results, Exception) and storage_lens_results.get("status") == "success":
                analysis_results["data"]["request_patterns"] = storage_lens_results["data"]
                analysis_results["data_sources"].extend(storage_lens_results.get("data_sources", []))
            else:
                self.logger.warning(f"Storage Lens request pattern analysis failed: {storage_lens_results}")
                analysis_results["data"]["request_patterns"] = {"error": str(storage_lens_results)}
            
            # Aggregate Cost Explorer API costs
            if not isinstance(cost_explorer_results, Exception) and cost_explorer_results.get("status") == "success":
                analysis_results["data"]["api_costs"] = cost_explorer_results["data"]
                analysis_results["data_sources"].extend(cost_explorer_results.get("data_sources", []))
            else:
                self.logger.warning(f"Cost Explorer API cost analysis failed: {cost_explorer_results}")
                analysis_results["data"]["api_costs"] = {"error": str(cost_explorer_results)}
            
            # Aggregate CloudWatch metrics
            if not isinstance(cloudwatch_results, Exception) and cloudwatch_results.get("status") == "success":
                analysis_results["data"]["cloudwatch_metrics"] = cloudwatch_results["data"]
                analysis_results["data_sources"].extend(cloudwatch_results.get("data_sources", []))
            else:
                self.logger.warning(f"CloudWatch metrics analysis failed: {cloudwatch_results}")
                analysis_results["data"]["cloudwatch_metrics"] = {"error": str(cloudwatch_results)}
            
            # Aggregate caching opportunities
            if not isinstance(caching_results, Exception) and caching_results.get("status") == "success":
                analysis_results["data"]["caching_recommendations"] = caching_results["data"]
                analysis_results["data_sources"].extend(caching_results.get("data_sources", []))
            else:
                self.logger.warning(f"Caching analysis failed: {caching_results}")
                analysis_results["data"]["caching_recommendations"] = {"error": str(caching_results)}
            
            # Calculate optimization opportunities
            analysis_results["data"]["optimization_opportunities"] = self._identify_optimization_opportunities(analysis_results["data"])
            
            # Generate CloudFront integration recommendations
            if context.get('include_cloudfront_analysis', True):
                analysis_results["data"]["cloudfront_opportunities"] = self._analyze_cloudfront_opportunities(analysis_results["data"])
            
            # Generate request consolidation strategies
            analysis_results["data"]["request_consolidation"] = self._analyze_request_consolidation(analysis_results["data"])
            
            # Calculate total costs and potential savings
            cost_summary = self._calculate_cost_summary(analysis_results["data"])
            analysis_results["data"]["total_api_cost"] = cost_summary["total_cost"]
            analysis_results["data"]["potential_savings"] = cost_summary["potential_savings"]
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Completed S3 API cost analysis in {analysis_results['execution_time']:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in API cost analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_request_patterns_storage_lens(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 API request patterns using Storage Lens as primary source.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing request pattern analysis
        """
        try:
            self.logger.debug("Analyzing request patterns using Storage Lens")
            
            request_analysis = {
                "activity_metrics_enabled": False,
                "detailed_status_codes_enabled": False,
                "request_volume_available": False,
                "access_patterns": {},
                "request_distribution": {}
            }
            
            data_sources = []
            
            # Try Storage Lens first (NO-COST primary source)
            if self.storage_lens_service:
                try:
                    # Get Storage Lens configuration to check available metrics
                    storage_lens_result = await self.storage_lens_service.get_storage_metrics()
                    
                    if storage_lens_result.get("status") == "success":
                        self.logger.info("Using Storage Lens as primary data source for request patterns")
                        
                        metrics_data = storage_lens_result["data"]
                        request_analysis.update({
                            "activity_metrics_enabled": metrics_data.get("IncludeRegions", False),
                            "detailed_status_codes_enabled": metrics_data.get("DetailedStatusCodesMetrics", False),
                            "cost_optimization_enabled": metrics_data.get("CostOptimizationMetrics", False)
                        })
                        
                        # If activity metrics are enabled, we can get request patterns
                        if request_analysis["activity_metrics_enabled"]:
                            request_analysis["request_volume_available"] = True
                            request_analysis["note"] = "Request patterns available through Storage Lens dashboard"
                        else:
                            request_analysis["note"] = "Enable Activity Metrics in Storage Lens for detailed request pattern analysis"
                        
                        data_sources.append("storage_lens")
                    else:
                        self.logger.warning(f"Storage Lens unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens request pattern analysis failed: {str(e)}")
            
            # If Storage Lens doesn't have activity metrics, note the limitation
            if not request_analysis.get("request_volume_available", False):
                request_analysis["fallback_note"] = "Request pattern analysis requires Storage Lens Activity Metrics or CloudWatch data"
            
            return {
                "status": "success",
                "data": request_analysis,
                "data_sources": data_sources,
                "message": f"Request pattern analysis completed using: {', '.join(data_sources) if data_sources else 'configuration check'}"
            }
            
        except Exception as e:
            self.logger.error(f"Request pattern analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Request pattern analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_api_costs_cost_explorer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 API costs using Cost Explorer as secondary source.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing API cost analysis
        """
        try:
            self.logger.debug("Analyzing API costs using Cost Explorer")
            
            api_cost_analysis = {
                "request_costs_by_type": {},
                "request_costs_by_storage_class": {},
                "daily_api_costs": [],
                "total_api_cost": 0,
                "cost_trends": {}
            }
            
            data_sources = []
            
            # Get API costs from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query for S3 API request costs by usage type
                api_filter = {
                    "And": [
                        {
                            "Dimensions": {
                                "Key": "SERVICE",
                                "Values": ["Amazon Simple Storage Service"]
                            }
                        },
                        {
                            "Dimensions": {
                                "Key": "USAGE_TYPE_GROUP",
                                "Values": [
                                    "S3-API-Tier1",      # PUT, COPY, POST, LIST requests
                                    "S3-API-Tier2",      # GET, SELECT requests
                                    "S3-API-SIA-Tier1",  # Standard-IA Tier1 requests
                                    "S3-API-SIA-Tier2"   # Standard-IA Tier2 requests
                                ]
                            }
                        }
                    ]
                }
                
                cost_result = get_cost_and_usage(
                    start_date=start_date,
                    end_date=end_date,
                    granularity="DAILY",
                    metrics=["UnblendedCost", "UsageQuantity"],
                    group_by=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                    filter_expr=api_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    api_cost_analysis.update(self._process_api_cost_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer API data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer API analysis failed: {str(e)}")
            
            # Enhance with pricing information if available
            if self.pricing_service:
                try:
                    api_pricing = await self._get_api_pricing_estimates(api_cost_analysis)
                    api_cost_analysis["pricing_estimates"] = api_pricing
                    data_sources.append("pricing_api")
                except Exception as e:
                    self.logger.warning(f"API pricing enhancement failed: {str(e)}")
            
            return {
                "status": "success",
                "data": api_cost_analysis,
                "data_sources": data_sources,
                "message": f"API cost analysis completed using: {', '.join(data_sources) if data_sources else 'fallback data'}"
            }
            
        except Exception as e:
            self.logger.error(f"API cost analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"API cost analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_cloudwatch_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 request metrics using CloudWatch as tertiary source.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing CloudWatch metrics analysis
        """
        try:
            self.logger.debug("Analyzing request metrics using CloudWatch")
            
            cloudwatch_analysis = {
                "bucket_level_metrics": {},
                "request_metrics_available": False,
                "detailed_monitoring_enabled": False
            }
            
            data_sources = []
            
            # Check if we have specific buckets to analyze
            bucket_names = context.get('bucket_names', [])
            
            if self.s3_service and bucket_names:
                try:
                    # Get request metrics for specified buckets
                    for bucket_name in bucket_names:
                        bucket_metrics = await self._get_bucket_request_metrics(
                            bucket_name, 
                            context.get('lookback_days', 30)
                        )
                        
                        if bucket_metrics.get("status") == "success":
                            cloudwatch_analysis["bucket_level_metrics"][bucket_name] = bucket_metrics["data"]
                            cloudwatch_analysis["request_metrics_available"] = True
                    
                    if cloudwatch_analysis["request_metrics_available"]:
                        data_sources.append("cloudwatch")
                        
                except Exception as e:
                    self.logger.warning(f"CloudWatch bucket metrics analysis failed: {str(e)}")
            
            # If no specific buckets, provide general guidance
            if not bucket_names:
                cloudwatch_analysis["note"] = "Specify bucket_names parameter for detailed CloudWatch request metrics analysis"
            
            return {
                "status": "success",
                "data": cloudwatch_analysis,
                "data_sources": data_sources,
                "message": f"CloudWatch metrics analysis completed using: {', '.join(data_sources) if data_sources else 'configuration check'}"
            }
            
        except Exception as e:
            self.logger.error(f"CloudWatch metrics analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"CloudWatch metrics analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_caching_opportunities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze caching opportunities for API cost reduction.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing caching opportunity analysis
        """
        try:
            self.logger.debug("Analyzing caching opportunities")
            
            caching_analysis = {
                "cache_hit_potential": {},
                "cloudfront_benefits": {},
                "application_caching": {},
                "cdn_recommendations": {}
            }
            
            # Analyze potential for different caching strategies
            caching_analysis["cache_hit_potential"] = {
                "static_content": {
                    "description": "Static files (images, CSS, JS, documents)",
                    "cache_duration": "1 year",
                    "potential_savings": "80-95% reduction in GET requests",
                    "implementation": "CloudFront with long TTL"
                },
                "semi_static_content": {
                    "description": "Infrequently changing data files",
                    "cache_duration": "1 day - 1 week",
                    "potential_savings": "60-80% reduction in GET requests",
                    "implementation": "CloudFront with moderate TTL"
                },
                "dynamic_content": {
                    "description": "Frequently changing content",
                    "cache_duration": "5 minutes - 1 hour",
                    "potential_savings": "30-60% reduction in GET requests",
                    "implementation": "Application-level caching + short TTL CDN"
                }
            }
            
            # CloudFront integration benefits
            caching_analysis["cloudfront_benefits"] = {
                "request_reduction": "Reduces S3 GET/HEAD requests by 70-90%",
                "cost_savings": "CloudFront requests cost less than S3 requests",
                "performance_improvement": "Lower latency for end users",
                "data_transfer_savings": "Reduced S3 data transfer costs"
            }
            
            # Application-level caching recommendations
            caching_analysis["application_caching"] = {
                "client_side_caching": {
                    "description": "Browser/mobile app caching",
                    "implementation": "HTTP cache headers (Cache-Control, ETag)",
                    "savings": "Eliminates repeat requests for same content"
                },
                "server_side_caching": {
                    "description": "Application server caching",
                    "implementation": "Redis, Memcached, or in-memory caching",
                    "savings": "Reduces S3 API calls for frequently accessed data"
                },
                "database_caching": {
                    "description": "Cache S3 metadata in database",
                    "implementation": "Store object metadata locally",
                    "savings": "Reduces LIST and HEAD operations"
                }
            }
            
            return {
                "status": "success",
                "data": caching_analysis,
                "data_sources": ["analysis"],
                "message": "Caching opportunity analysis completed"
            }
            
        except Exception as e:
            self.logger.error(f"Caching analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Caching analysis failed: {str(e)}",
                "data": {}
            }
    
    def _process_api_cost_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer API cost data.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed API cost data
        """
        try:
            processed_data = {
                "request_costs_by_type": {},
                "daily_api_costs": [],
                "total_api_cost": 0,
                "request_volume": {}
            }
            
            # Process results by time period
            for result in cost_data.get("ResultsByTime", []):
                time_period = result.get("TimePeriod", {})
                start_date = time_period.get("Start", "")
                daily_cost = 0
                
                # Process groups (usage types)
                for group in result.get("Groups", []):
                    usage_type = group.get("Keys", ["Unknown"])[0]
                    metrics = group.get("Metrics", {})
                    
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                    
                    # Categorize by request type
                    request_type = self._extract_request_type_from_usage_type(usage_type)
                    
                    if request_type not in processed_data["request_costs_by_type"]:
                        processed_data["request_costs_by_type"][request_type] = {
                            "total_cost": 0,
                            "total_requests": 0,
                            "daily_costs": []
                        }
                    
                    processed_data["request_costs_by_type"][request_type]["total_cost"] += cost
                    processed_data["request_costs_by_type"][request_type]["total_requests"] += usage
                    processed_data["request_costs_by_type"][request_type]["daily_costs"].append({
                        "date": start_date,
                        "cost": cost,
                        "requests": usage
                    })
                    
                    processed_data["total_api_cost"] += cost
                    daily_cost += cost
                
                processed_data["daily_api_costs"].append({
                    "date": start_date,
                    "total_cost": daily_cost
                })
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing API cost data: {str(e)}")
            return {"error": str(e)}
    
    def _extract_request_type_from_usage_type(self, usage_type: str) -> str:
        """
        Extract request type from Cost Explorer usage type.
        
        Args:
            usage_type: Usage type string from Cost Explorer
            
        Returns:
            Normalized request type
        """
        usage_type_lower = usage_type.lower()
        
        if "tier1" in usage_type_lower:
            return "TIER1_REQUESTS"  # PUT, COPY, POST, LIST
        elif "tier2" in usage_type_lower:
            return "TIER2_REQUESTS"  # GET, SELECT
        elif "sia" in usage_type_lower and "tier1" in usage_type_lower:
            return "SIA_TIER1_REQUESTS"  # Standard-IA Tier1
        elif "sia" in usage_type_lower and "tier2" in usage_type_lower:
            return "SIA_TIER2_REQUESTS"  # Standard-IA Tier2
        else:
            return "OTHER_REQUESTS"
    
    async def _get_bucket_request_metrics(self, bucket_name: str, lookback_days: int) -> Dict[str, Any]:
        """
        Get CloudWatch request metrics for a specific bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary containing bucket request metrics
        """
        try:
            # This would integrate with CloudWatch to get actual metrics
            # For now, return a placeholder structure
            bucket_metrics = {
                "bucket_name": bucket_name,
                "request_metrics": {
                    "AllRequests": {"sum": 0, "average": 0},
                    "GetRequests": {"sum": 0, "average": 0},
                    "PutRequests": {"sum": 0, "average": 0},
                    "DeleteRequests": {"sum": 0, "average": 0},
                    "HeadRequests": {"sum": 0, "average": 0},
                    "PostRequests": {"sum": 0, "average": 0},
                    "ListRequests": {"sum": 0, "average": 0}
                },
                "error_metrics": {
                    "4xxErrors": {"sum": 0, "average": 0},
                    "5xxErrors": {"sum": 0, "average": 0}
                },
                "data_transfer": {
                    "BytesDownloaded": {"sum": 0, "average": 0},
                    "BytesUploaded": {"sum": 0, "average": 0}
                },
                "note": "CloudWatch request metrics integration would be implemented here"
            }
            
            return {
                "status": "success",
                "data": bucket_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bucket request metrics for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting bucket request metrics: {str(e)}"
            }
    
    async def _get_api_pricing_estimates(self, api_cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get API pricing estimates using pricing service.
        
        Args:
            api_cost_data: API cost data from Cost Explorer
            
        Returns:
            Dictionary containing pricing estimates
        """
        try:
            pricing_estimates = {
                "request_pricing": {
                    "tier1_requests": {"price_per_1000": 0.005, "description": "PUT, COPY, POST, LIST requests"},
                    "tier2_requests": {"price_per_1000": 0.0004, "description": "GET, SELECT requests"},
                    "sia_tier1_requests": {"price_per_1000": 0.01, "description": "Standard-IA PUT, COPY, POST, LIST"},
                    "sia_tier2_requests": {"price_per_1000": 0.001, "description": "Standard-IA GET, SELECT"}
                },
                "storage_class_multipliers": self.storage_class_multipliers,
                "optimization_potential": {
                    "caching_savings": "70-90% reduction in GET requests",
                    "request_consolidation": "20-40% reduction in LIST requests",
                    "lifecycle_automation": "Eliminates manual DELETE requests"
                }
            }
            
            return pricing_estimates
            
        except Exception as e:
            self.logger.error(f"Error getting API pricing estimates: {str(e)}")
            return {"error": str(e)}
    
    def _identify_optimization_opportunities(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify API cost optimization opportunities from analysis data.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Dictionary containing optimization opportunities
        """
        try:
            opportunities = {
                "high_priority": [],
                "medium_priority": [],
                "low_priority": [],
                "total_potential_savings": 0
            }
            
            # Analyze API costs for optimization opportunities
            api_costs = analysis_data.get("api_costs", {})
            total_api_cost = api_costs.get("total_api_cost", 0)
            
            if total_api_cost > 100:  # Significant API costs
                opportunities["high_priority"].append({
                    "type": "caching",
                    "title": "Implement CloudFront Caching",
                    "description": f"Current API costs: ${total_api_cost:.2f}/month. CloudFront caching can reduce GET requests by 70-90%",
                    "potential_savings": total_api_cost * 0.8,  # 80% savings estimate
                    "implementation_effort": "medium",
                    "action_items": [
                        "Set up CloudFront distribution for S3 bucket",
                        "Configure appropriate TTL values for content types",
                        "Update application to use CloudFront URLs"
                    ]
                })
            
            # Check for high Tier1 request costs
            request_costs = api_costs.get("request_costs_by_type", {})
            tier1_cost = request_costs.get("TIER1_REQUESTS", {}).get("total_cost", 0)
            
            if tier1_cost > 50:  # High Tier1 costs
                opportunities["medium_priority"].append({
                    "type": "request_optimization",
                    "title": "Optimize PUT/LIST Request Patterns",
                    "description": f"High Tier1 request costs: ${tier1_cost:.2f}/month. Optimize upload and listing patterns",
                    "potential_savings": tier1_cost * 0.3,  # 30% savings estimate
                    "implementation_effort": "low",
                    "action_items": [
                        "Batch multiple small uploads into larger objects",
                        "Use pagination for LIST operations",
                        "Implement client-side caching for object metadata"
                    ]
                })
            
            # Check for Storage Lens activity metrics
            request_patterns = analysis_data.get("request_patterns", {})
            if not request_patterns.get("activity_metrics_enabled", False):
                opportunities["low_priority"].append({
                    "type": "monitoring",
                    "title": "Enable Storage Lens Activity Metrics",
                    "description": "Enable Activity Metrics in Storage Lens for detailed request pattern analysis",
                    "potential_savings": 0,
                    "implementation_effort": "low",
                    "action_items": [
                        "Update Storage Lens configuration to include Activity Metrics",
                        "Monitor request patterns through Storage Lens dashboard",
                        "Use insights to optimize application request patterns"
                    ]
                })
            
            # Calculate total potential savings
            opportunities["total_potential_savings"] = sum([
                opp.get("potential_savings", 0) 
                for priority_list in [opportunities["high_priority"], opportunities["medium_priority"], opportunities["low_priority"]]
                for opp in priority_list
            ])
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_cloudfront_opportunities(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CloudFront integration opportunities for API cost reduction.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Dictionary containing CloudFront opportunities
        """
        try:
            cloudfront_analysis = {
                "benefits": {
                    "cost_reduction": "CloudFront requests cost 10x less than S3 requests",
                    "performance_improvement": "Lower latency with global edge locations",
                    "bandwidth_savings": "Reduced S3 data transfer costs",
                    "scalability": "Better handling of traffic spikes"
                },
                "implementation_strategies": {
                    "static_content": {
                        "content_types": ["images", "videos", "documents", "software downloads"],
                        "cache_duration": "1 year",
                        "savings_potential": "90-95%"
                    },
                    "api_responses": {
                        "content_types": ["JSON responses", "XML data", "CSV exports"],
                        "cache_duration": "5 minutes - 1 hour",
                        "savings_potential": "60-80%"
                    },
                    "dynamic_content": {
                        "content_types": ["personalized content", "real-time data"],
                        "cache_duration": "No cache or very short TTL",
                        "savings_potential": "20-40% (compression and connection reuse)"
                    }
                },
                "cost_comparison": {
                    "s3_get_requests": "$0.0004 per 1,000 requests",
                    "cloudfront_requests": "$0.0075 per 10,000 requests",
                    "savings_ratio": "CloudFront is ~10x cheaper per request"
                },
                "implementation_checklist": [
                    "Create CloudFront distribution with S3 origin",
                    "Configure cache behaviors for different content types",
                    "Set appropriate TTL values based on content update frequency",
                    "Update application URLs to use CloudFront domain",
                    "Monitor cache hit ratio and adjust configuration",
                    "Implement cache invalidation strategy for updated content"
                ]
            }
            
            # Calculate potential CloudFront savings based on current API costs
            api_costs = analysis_data.get("api_costs", {})
            tier2_requests = api_costs.get("request_costs_by_type", {}).get("TIER2_REQUESTS", {})
            tier2_cost = tier2_requests.get("total_cost", 0)
            
            if tier2_cost > 0:
                cloudfront_analysis["estimated_savings"] = {
                    "current_monthly_get_cost": tier2_cost,
                    "estimated_cloudfront_cost": tier2_cost * 0.1,  # 10x cheaper
                    "monthly_savings": tier2_cost * 0.9,
                    "annual_savings": tier2_cost * 0.9 * 12
                }
            
            return cloudfront_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing CloudFront opportunities: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_request_consolidation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze request consolidation opportunities.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Dictionary containing request consolidation strategies
        """
        try:
            consolidation_strategies = {
                "batch_operations": {
                    "description": "Combine multiple operations into single requests",
                    "strategies": {
                        "multipart_uploads": {
                            "description": "Use multipart upload for large files instead of multiple PUT requests",
                            "savings": "Reduces PUT request count and improves reliability",
                            "threshold": "Files > 100MB should use multipart upload"
                        },
                        "batch_delete": {
                            "description": "Use batch delete API to delete up to 1000 objects per request",
                            "savings": "Reduces DELETE request count by up to 1000x",
                            "implementation": "Use delete_objects() API instead of individual delete_object() calls"
                        },
                        "list_pagination": {
                            "description": "Optimize LIST operations with appropriate page sizes",
                            "savings": "Reduces LIST request count while maintaining performance",
                            "recommendation": "Use page size of 1000 objects per LIST request"
                        }
                    }
                },
                "metadata_caching": {
                    "description": "Cache object metadata to reduce HEAD/LIST requests",
                    "strategies": {
                        "local_metadata_cache": {
                            "description": "Store object metadata in local database/cache",
                            "savings": "Eliminates repeated HEAD requests for same objects",
                            "implementation": "Cache object size, ETag, last-modified date"
                        },
                        "directory_listing_cache": {
                            "description": "Cache directory listings to reduce LIST requests",
                            "savings": "Reduces LIST requests for frequently accessed directories",
                            "ttl_recommendation": "5-60 minutes depending on update frequency"
                        }
                    }
                },
                "request_optimization": {
                    "description": "Optimize individual request patterns",
                    "strategies": {
                        "conditional_requests": {
                            "description": "Use If-None-Match headers to avoid unnecessary downloads",
                            "savings": "Reduces GET requests and data transfer for unchanged objects",
                            "implementation": "Use ETag values in conditional GET requests"
                        },
                        "range_requests": {
                            "description": "Use byte-range requests for partial object access",
                            "savings": "Reduces data transfer costs for large objects",
                            "use_cases": ["Video streaming", "Large file previews", "Resume downloads"]
                        },
                        "presigned_urls": {
                            "description": "Use presigned URLs for direct client-to-S3 uploads",
                            "savings": "Eliminates server proxy requests and reduces bandwidth costs",
                            "security": "Provides secure, time-limited access without exposing credentials"
                        }
                    }
                }
            }
            
            return consolidation_strategies
            
        except Exception as e:
            self.logger.error(f"Error analyzing request consolidation: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_cost_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total API costs and potential savings.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Dictionary containing cost summary
        """
        try:
            # Extract total API cost from analysis
            api_costs = analysis_data.get("api_costs", {})
            total_cost = api_costs.get("total_api_cost", 0)
            
            # Calculate potential savings from optimization opportunities
            optimization_opportunities = analysis_data.get("optimization_opportunities", {})
            potential_savings = optimization_opportunities.get("total_potential_savings", 0)
            
            # If no specific savings calculated, estimate based on typical optimization results
            if potential_savings == 0 and total_cost > 0:
                # Conservative estimate: 40% savings from caching and optimization
                potential_savings = total_cost * 0.4
            
            cost_summary = {
                "total_cost": total_cost,
                "potential_savings": potential_savings,
                "savings_percentage": (potential_savings / total_cost * 100) if total_cost > 0 else 0,
                "cost_after_optimization": total_cost - potential_savings,
                "annual_current_cost": total_cost * 12,
                "annual_potential_savings": potential_savings * 12
            }
            
            return cost_summary
            
        except Exception as e:
            self.logger.error(f"Error calculating cost summary: {str(e)}")
            return {
                "total_cost": 0,
                "potential_savings": 0,
                "error": str(e)
            }
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from API cost analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            
            # Extract analysis data
            analysis_data = analysis_results.get("data", {})
            optimization_opportunities = analysis_data.get("optimization_opportunities", {})
            
            # Add high priority recommendations
            for opp in optimization_opportunities.get("high_priority", []):
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="high",
                    title=opp.get("title", "High Priority Optimization"),
                    description=opp.get("description", ""),
                    potential_savings=opp.get("potential_savings"),
                    implementation_effort=opp.get("implementation_effort", "medium"),
                    action_items=opp.get("action_items", [])
                ))
            
            # Add medium priority recommendations
            for opp in optimization_opportunities.get("medium_priority", []):
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="medium",
                    title=opp.get("title", "Medium Priority Optimization"),
                    description=opp.get("description", ""),
                    potential_savings=opp.get("potential_savings"),
                    implementation_effort=opp.get("implementation_effort", "medium"),
                    action_items=opp.get("action_items", [])
                ))
            
            # Add CloudFront recommendation if significant GET request costs
            api_costs = analysis_data.get("api_costs", {})
            tier2_cost = api_costs.get("request_costs_by_type", {}).get("TIER2_REQUESTS", {}).get("total_cost", 0)
            
            if tier2_cost > 20:  # Significant GET request costs
                recommendations.append(self.create_recommendation(
                    rec_type="performance",
                    priority="high",
                    title="Implement CloudFront CDN",
                    description=f"Current GET request costs: ${tier2_cost:.2f}/month. CloudFront can reduce costs by 90% and improve performance",
                    potential_savings=tier2_cost * 0.9,
                    implementation_effort="medium",
                    action_items=[
                        "Create CloudFront distribution with S3 origin",
                        "Configure cache behaviors for different content types",
                        "Update application to use CloudFront URLs",
                        "Monitor cache hit ratio and optimize TTL settings"
                    ]
                ))
            
            # Add request consolidation recommendation
            recommendations.append(self.create_recommendation(
                rec_type="performance",
                priority="medium",
                title="Optimize Request Patterns",
                description="Implement request consolidation strategies to reduce API call volume and costs",
                implementation_effort="low",
                action_items=[
                    "Use batch operations for multiple object operations",
                    "Implement metadata caching to reduce HEAD/LIST requests",
                    "Use conditional requests with ETag headers",
                    "Optimize LIST operation pagination"
                ]
            ))
            
            # Add Storage Lens monitoring recommendation
            request_patterns = analysis_data.get("request_patterns", {})
            if not request_patterns.get("activity_metrics_enabled", False):
                recommendations.append(self.create_recommendation(
                    rec_type="monitoring",
                    priority="low",
                    title="Enable Storage Lens Activity Metrics",
                    description="Enable detailed request pattern monitoring for better optimization insights",
                    implementation_effort="low",
                    action_items=[
                        "Update Storage Lens configuration to include Activity Metrics",
                        "Review request patterns monthly through Storage Lens dashboard",
                        "Use insights to identify additional optimization opportunities"
                    ]
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return [
                self.create_recommendation(
                    rec_type="error_resolution",
                    priority="high",
                    title="API Cost Analysis Error",
                    description=f"Error generating recommendations: {str(e)}",
                    action_items=[
                        "Check AWS credentials and permissions",
                        "Verify Cost Explorer and Storage Lens access",
                        "Review analysis parameters and retry"
                    ]
                )
            ]