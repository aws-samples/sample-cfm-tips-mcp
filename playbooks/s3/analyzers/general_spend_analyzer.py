"""
General S3 Spend Analyzer

This analyzer provides comprehensive S3 spending pattern analysis using Storage Lens
as the primary data source and Cost Explorer as fallback for historical spend data.

Analyzes:
- Storage costs by storage class and object size
- Data transfer costs (cross-region, internet egress, Direct Connect)
- API request costs by type and storage class
- Data retrieval costs for archived objects
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage

logger = logging.getLogger(__name__)


class GeneralSpendAnalyzer(BaseAnalyzer):
    """
    Analyzer for comprehensive S3 spending pattern analysis.
    
    Uses Storage Lens as primary data source with Cost Explorer fallback
    to analyze storage costs, data transfer, API charges, and retrieval costs.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize GeneralSpendAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "general_spend"
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive S3 spend analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - lookback_days: Number of days to analyze (default: 30)
                - bucket_names: Optional list of specific buckets to analyze
                - include_cost_analysis: Whether to include detailed cost breakdown
                
        Returns:
            Dictionary containing comprehensive spend analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting general S3 spend analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "storage_costs": {},
                    "data_transfer_costs": {},
                    "api_costs": {},
                    "retrieval_costs": {},
                    "total_costs": {},
                    "cost_breakdown": {},
                    "optimization_opportunities": []
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute analysis components in parallel
            tasks = [
                self._analyze_storage_costs(context),
                self._analyze_data_transfer_costs(context),
                self._analyze_api_costs(context),
                self._analyze_retrieval_costs(context),
                self._analyze_bucket_level_costs(context)
            ]
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            storage_results, transfer_results, api_results, retrieval_results, bucket_results = results
            
            # Aggregate storage costs
            if not isinstance(storage_results, Exception) and storage_results.get("status") == "success":
                analysis_results["data"]["storage_costs"] = storage_results["data"]
                analysis_results["data_sources"].extend(storage_results.get("data_sources", []))
            else:
                self.logger.warning(f"Storage cost analysis failed: {storage_results}")
                analysis_results["data"]["storage_costs"] = {"error": str(storage_results)}
            
            # Aggregate data transfer costs
            if not isinstance(transfer_results, Exception) and transfer_results.get("status") == "success":
                analysis_results["data"]["data_transfer_costs"] = transfer_results["data"]
                analysis_results["data_sources"].extend(transfer_results.get("data_sources", []))
            else:
                self.logger.warning(f"Data transfer cost analysis failed: {transfer_results}")
                analysis_results["data"]["data_transfer_costs"] = {"error": str(transfer_results)}
            
            # Aggregate API costs
            if not isinstance(api_results, Exception) and api_results.get("status") == "success":
                analysis_results["data"]["api_costs"] = api_results["data"]
                analysis_results["data_sources"].extend(api_results.get("data_sources", []))
            else:
                self.logger.warning(f"API cost analysis failed: {api_results}")
                analysis_results["data"]["api_costs"] = {"error": str(api_results)}
            
            # Aggregate retrieval costs
            if not isinstance(retrieval_results, Exception) and retrieval_results.get("status") == "success":
                analysis_results["data"]["retrieval_costs"] = retrieval_results["data"]
                analysis_results["data_sources"].extend(retrieval_results.get("data_sources", []))
            else:
                self.logger.warning(f"Retrieval cost analysis failed: {retrieval_results}")
                analysis_results["data"]["retrieval_costs"] = {"error": str(retrieval_results)}
            
            # Aggregate bucket-level costs
            if not isinstance(bucket_results, Exception) and bucket_results.get("status") == "success":
                analysis_results["data"]["bucket_costs"] = bucket_results["data"]
                analysis_results["data_sources"].extend(bucket_results.get("data_sources", []))
            else:
                self.logger.warning(f"Bucket cost analysis failed: {bucket_results}")
                analysis_results["data"]["bucket_costs"] = {"error": str(bucket_results)}
            
            # Calculate total costs and create breakdown
            analysis_results["data"]["total_costs"] = self._calculate_total_costs(analysis_results["data"])
            analysis_results["data"]["cost_breakdown"] = self._create_cost_breakdown(analysis_results["data"])
            
            # Identify optimization opportunities
            analysis_results["data"]["optimization_opportunities"] = self._identify_optimization_opportunities(analysis_results["data"])
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Completed general S3 spend analysis in {analysis_results['execution_time']:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in general spend analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_storage_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 storage costs by storage class and object size.
        
        Uses Storage Lens as primary source, Cost Explorer as fallback.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing storage cost analysis
        """
        try:
            self.logger.debug("Analyzing storage costs using Storage Lens and Cost Explorer")
            
            storage_analysis = {
                "by_storage_class": {},
                "by_object_size": {},
                "monthly_trends": {},
                "total_storage_cost": 0,
                "storage_distribution": {}
            }
            
            data_sources = []
            
            # Try Storage Lens first (NO-COST primary source)
            storage_lens_has_cost_data = False
            if self.storage_lens_service:
                try:
                    storage_lens_result = await self.storage_lens_service.get_storage_metrics()
                    
                    if storage_lens_result.get("status") == "success":
                        processed_storage_lens = self._process_storage_lens_data(storage_lens_result["data"])
                        # Check if Storage Lens actually has cost data
                        if processed_storage_lens.get("storage_metrics_available") and processed_storage_lens.get("by_storage_class"):
                            self.logger.info("Using Storage Lens as primary data source for storage costs")
                            storage_analysis.update(processed_storage_lens)
                            data_sources.append("storage_lens")
                            storage_lens_has_cost_data = True
                        else:
                            self.logger.info("Storage Lens available but no cost data - will use Cost Explorer")
                            storage_analysis.update(processed_storage_lens)  # Keep the metadata
                    else:
                        self.logger.warning(f"Storage Lens unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens analysis failed: {str(e)}")
            
            # Fallback to Cost Explorer for historical data (always try if Storage Lens has no cost data)
            if not storage_lens_has_cost_data:
                try:
                    self.logger.info("Using Cost Explorer as fallback for storage cost data")
                    cost_explorer_result = await self._get_storage_costs_from_cost_explorer(context)
                    
                    if cost_explorer_result.get("status") == "success":
                        storage_analysis.update(cost_explorer_result["data"])
                        data_sources.append("cost_explorer")
                    else:
                        self.logger.warning(f"Cost Explorer fallback failed: {cost_explorer_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Cost Explorer fallback failed: {str(e)}")
            
            # If we have pricing service, enhance with detailed cost calculations
            if self.pricing_service and storage_analysis.get("storage_distribution"):
                try:
                    enhanced_costs = await self._enhance_storage_costs_with_pricing(
                        storage_analysis["storage_distribution"]
                    )
                    storage_analysis["detailed_pricing"] = enhanced_costs
                    data_sources.append("pricing_api")
                except Exception as e:
                    self.logger.warning(f"Pricing enhancement failed: {str(e)}")
            
            return {
                "status": "success",
                "data": storage_analysis,
                "data_sources": data_sources,
                "message": f"Storage cost analysis completed using: {', '.join(data_sources)}"
            }
            
        except Exception as e:
            self.logger.error(f"Storage cost analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Storage cost analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_data_transfer_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 data transfer costs including cross-region, internet egress, and Direct Connect.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing data transfer cost analysis
        """
        try:
            self.logger.debug("Analyzing data transfer costs")
            
            transfer_analysis = {
                "cross_region_transfer": {"cost": 0, "volume_gb": 0},
                "internet_egress": {"cost": 0, "volume_gb": 0},
                "direct_connect": {"cost": 0, "volume_gb": 0},
                "cloudfront_transfer": {"cost": 0, "volume_gb": 0},
                "total_transfer_cost": 0,
                "transfer_breakdown": {}
            }
            
            data_sources = []
            
            # Get data transfer costs from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query Cost Explorer for S3 data transfer costs
                transfer_filter = {
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
                                "Values": ["S3-DataTransfer-Out-Bytes", "S3-DataTransfer-Regional-Bytes"]
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
                    filter_expr=transfer_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    transfer_analysis.update(self._process_transfer_cost_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer transfer data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer transfer analysis failed: {str(e)}")
            
            # Enhance with pricing information if available
            if self.pricing_service:
                try:
                    transfer_pricing = self.pricing_service.get_data_transfer_pricing()
                    if transfer_pricing.get("status") == "success":
                        transfer_analysis["pricing_details"] = transfer_pricing["transfer_pricing"]
                        data_sources.append("pricing_api")
                except Exception as e:
                    self.logger.warning(f"Transfer pricing enhancement failed: {str(e)}")
            
            return {
                "status": "success",
                "data": transfer_analysis,
                "data_sources": data_sources,
                "message": f"Data transfer cost analysis completed using: {', '.join(data_sources) if data_sources else 'fallback data'}"
            }
            
        except Exception as e:
            self.logger.error(f"Data transfer cost analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Data transfer cost analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_api_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 API request costs by type and storage class.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing API cost analysis
        """
        try:
            self.logger.debug("Analyzing API request costs")
            
            api_analysis = {
                "request_costs_by_type": {},
                "request_costs_by_storage_class": {},
                "total_api_cost": 0,
                "request_volume": {},
                "cost_per_request_type": {}
            }
            
            data_sources = []
            
            # Get API costs from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query for S3 API request costs
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
                                "Values": ["S3-API-Tier1", "S3-API-Tier2", "S3-API-SIA-Tier1", "S3-API-SIA-Tier2"]
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
                    api_analysis.update(self._process_api_cost_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer API data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer API analysis failed: {str(e)}")
            
            # Enhance with request volume from S3 service if available
            if self.s3_service and context.get('bucket_names'):
                try:
                    request_metrics = await self._get_request_metrics_for_buckets(
                        context['bucket_names'], 
                        context.get('lookback_days', 30)
                    )
                    api_analysis["detailed_request_metrics"] = request_metrics
                    data_sources.append("cloudwatch")
                except Exception as e:
                    self.logger.warning(f"Request metrics enhancement failed: {str(e)}")
            
            # Enhance with pricing information
            if self.pricing_service:
                try:
                    # Estimate costs based on typical request patterns
                    sample_requests = {
                        'GET': api_analysis.get("request_volume", {}).get("GET", 10000),
                        'PUT': api_analysis.get("request_volume", {}).get("PUT", 1000),
                        'LIST': api_analysis.get("request_volume", {}).get("LIST", 100),
                        'DELETE': api_analysis.get("request_volume", {}).get("DELETE", 50)
                    }
                    
                    request_cost_estimate = self.pricing_service.estimate_request_costs(sample_requests)
                    if request_cost_estimate.get("status") == "success":
                        api_analysis["cost_estimates"] = request_cost_estimate
                        data_sources.append("pricing_api")
                except Exception as e:
                    self.logger.warning(f"API cost estimation failed: {str(e)}")
            
            return {
                "status": "success",
                "data": api_analysis,
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
    
    async def _analyze_retrieval_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze S3 data retrieval costs for archived objects.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing retrieval cost analysis
        """
        try:
            self.logger.debug("Analyzing data retrieval costs")
            
            retrieval_analysis = {
                "glacier_retrievals": {"cost": 0, "volume_gb": 0},
                "deep_archive_retrievals": {"cost": 0, "volume_gb": 0},
                "intelligent_tiering_retrievals": {"cost": 0, "volume_gb": 0},
                "total_retrieval_cost": 0,
                "retrieval_patterns": {}
            }
            
            data_sources = []
            
            # Get retrieval costs from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query for S3 retrieval costs
                retrieval_filter = {
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
                                "Values": ["S3-Retrieval-Bytes", "S3-GlacierByteHrs", "S3-DeepArchive-Retrieval"]
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
                    filter_expr=retrieval_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    retrieval_analysis.update(self._process_retrieval_cost_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer retrieval data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer retrieval analysis failed: {str(e)}")
            
            return {
                "status": "success",
                "data": retrieval_analysis,
                "data_sources": data_sources,
                "message": f"Retrieval cost analysis completed using: {', '.join(data_sources) if data_sources else 'fallback data'}"
            }
            
        except Exception as e:
            self.logger.error(f"Retrieval cost analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Retrieval cost analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_bucket_level_costs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze costs at the bucket level to identify top spending buckets.
        
        Uses bucket size and storage class data to estimate costs per bucket.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing bucket-level cost analysis
        """
        try:
            self.logger.debug("Analyzing bucket-level costs")
            
            bucket_analysis = {
                "by_bucket": {},
                "top_10_buckets": [],
                "total_buckets_analyzed": 0,
                "cost_estimation_method": "size_based"
            }
            
            data_sources = []
            
            # Get list of all buckets
            if not self.s3_service:
                self.logger.warning("S3 service not available for bucket-level analysis")
                return {
                    "status": "error",
                    "message": "S3 service not available",
                    "data": bucket_analysis
                }
            
            try:
                # List all buckets
                buckets_result = await self.s3_service.list_buckets()
                
                if buckets_result.get("status") != "success":
                    self.logger.warning(f"Failed to list buckets: {buckets_result.get('message')}")
                    return {
                        "status": "error",
                        "message": "Failed to list buckets",
                        "data": bucket_analysis
                    }
                
                buckets = buckets_result.get("data", {}).get("Buckets", [])
                bucket_analysis["total_buckets_analyzed"] = len(buckets)
                data_sources.append("s3_api")
                
                # Filter by region if specified
                region_filter = context.get('region')
                if region_filter:
                    buckets = [b for b in buckets if b.get('Region') == region_filter]
                    self.logger.info(f"Filtered to {len(buckets)} buckets in region {region_filter}")
                
                # Filter by specific bucket names if provided
                bucket_names_filter = context.get('bucket_names')
                if bucket_names_filter:
                    buckets = [b for b in buckets if b.get('Name') in bucket_names_filter]
                    self.logger.info(f"Filtered to {len(buckets)} specified buckets")
                
                # Get size and storage class info for each bucket
                bucket_tasks = []
                for bucket in buckets[:100]:  # Limit to 100 buckets to avoid timeout
                    bucket_name = bucket.get('Name')
                    task = self._get_bucket_cost_estimate(bucket_name, context)
                    bucket_tasks.append((bucket_name, task))
                
                # Execute bucket analysis in parallel
                bucket_results = await asyncio.gather(*[task for _, task in bucket_tasks], return_exceptions=True)
                
                # Process bucket results
                for i, (bucket_name, _) in enumerate(bucket_tasks):
                    result = bucket_results[i]
                    if not isinstance(result, Exception) and result.get("status") == "success":
                        bucket_data = result["data"]
                        bucket_analysis["by_bucket"][bucket_name] = bucket_data
                    else:
                        self.logger.warning(f"Failed to analyze bucket {bucket_name}: {result}")
                
                # Sort buckets by estimated monthly cost and get top 10
                sorted_buckets = sorted(
                    bucket_analysis["by_bucket"].items(),
                    key=lambda x: x[1].get("estimated_monthly_cost", 0),
                    reverse=True
                )
                
                bucket_analysis["top_10_buckets"] = [
                    {
                        "bucket_name": name,
                        "estimated_monthly_cost": data.get("estimated_monthly_cost", 0),
                        "size_gb": data.get("size_gb", 0),
                        "object_count": data.get("object_count", 0),
                        "primary_storage_class": data.get("primary_storage_class", "STANDARD")
                    }
                    for name, data in sorted_buckets[:10]
                ]
                
                self.logger.info(f"Analyzed {len(bucket_analysis['by_bucket'])} buckets for cost estimation")
                
            except Exception as e:
                self.logger.error(f"Error analyzing bucket-level costs: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Bucket analysis failed: {str(e)}",
                    "data": bucket_analysis
                }
            
            return {
                "status": "success",
                "data": bucket_analysis,
                "data_sources": data_sources,
                "message": f"Bucket-level cost analysis completed for {bucket_analysis['total_buckets_analyzed']} buckets"
            }
            
        except Exception as e:
            self.logger.error(f"Bucket-level cost analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Bucket-level cost analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _get_bucket_cost_estimate(self, bucket_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate monthly cost for a specific bucket based on size and storage class.
        
        Args:
            bucket_name: Name of the bucket
            context: Analysis context
            
        Returns:
            Dictionary containing bucket cost estimate
        """
        try:
            # Get bucket metrics (size, object count, storage class distribution)
            metrics_result = await self.s3_service.get_bucket_metrics(bucket_name)
            
            if metrics_result.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Failed to get metrics for bucket {bucket_name}"
                }
            
            metrics = metrics_result.get("data", {})
            size_bytes = metrics.get("size_bytes", 0)
            size_gb = size_bytes / (1024 ** 3)
            object_count = metrics.get("object_count", 0)
            storage_class_distribution = metrics.get("storage_class_distribution", {})
            
            # Estimate cost based on storage class distribution
            estimated_cost = 0
            primary_storage_class = "STANDARD"
            max_storage = 0
            
            # Default pricing per GB-month (approximate US East)
            storage_class_pricing = {
                "STANDARD": 0.023,
                "STANDARD_IA": 0.0125,
                "ONEZONE_IA": 0.01,
                "INTELLIGENT_TIERING": 0.023,  # Varies, using Standard as baseline
                "GLACIER": 0.004,
                "GLACIER_IR": 0.004,
                "DEEP_ARCHIVE": 0.00099
            }
            
            if storage_class_distribution:
                # Calculate cost based on actual distribution
                for storage_class, class_size_bytes in storage_class_distribution.items():
                    class_size_gb = class_size_bytes / (1024 ** 3)
                    price_per_gb = storage_class_pricing.get(storage_class, 0.023)
                    estimated_cost += class_size_gb * price_per_gb
                    
                    # Track primary storage class (largest)
                    if class_size_bytes > max_storage:
                        max_storage = class_size_bytes
                        primary_storage_class = storage_class
            else:
                # Fallback: assume STANDARD storage class
                estimated_cost = size_gb * storage_class_pricing["STANDARD"]
                primary_storage_class = "STANDARD"
            
            return {
                "status": "success",
                "data": {
                    "bucket_name": bucket_name,
                    "size_gb": round(size_gb, 2),
                    "object_count": object_count,
                    "estimated_monthly_cost": round(estimated_cost, 2),
                    "primary_storage_class": primary_storage_class,
                    "storage_class_distribution": storage_class_distribution
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating cost for bucket {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Cost estimation failed for {bucket_name}: {str(e)}"
            }
    
    def _process_storage_lens_data(self, storage_lens_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Storage Lens data to extract storage cost information.
        
        Args:
            storage_lens_data: Raw Storage Lens data
            
        Returns:
            Processed storage cost data
        """
        try:
            processed_data = {
                "by_storage_class": {},
                "storage_distribution": {},
                "cost_optimization_enabled": storage_lens_data.get("CostOptimizationMetricsEnabled", False),
                "data_source": "storage_lens"
            }
            
            # Extract storage class information if available
            if storage_lens_data.get("StorageMetricsEnabled"):
                processed_data["storage_metrics_available"] = True
                processed_data["note"] = "Detailed storage metrics available through Storage Lens dashboard"
            else:
                processed_data["storage_metrics_available"] = False
                processed_data["note"] = "Enable Storage Lens storage metrics for detailed analysis"
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing Storage Lens data: {str(e)}")
            return {"error": str(e), "data_source": "storage_lens"}
    
    async def _get_storage_costs_from_cost_explorer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get storage costs from Cost Explorer as fallback.
        
        Args:
            context: Analysis context
            
        Returns:
            Storage cost data from Cost Explorer
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
            
            # Query for all S3 costs by usage type - use broader filter to capture all S3 costs
            storage_filter = {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": ["Amazon Simple Storage Service"]
                }
            }
            
            cost_result = get_cost_and_usage(
                start_date=start_date,
                end_date=end_date,
                granularity="MONTHLY",  # Use monthly for better aggregation
                metrics=["UnblendedCost", "UsageQuantity"],
                group_by=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                filter_expr=storage_filter,
                region=context.get('region')
            )
            
            if cost_result.get("status") == "success":
                return {
                    "status": "success",
                    "data": self._process_cost_explorer_storage_data(cost_result["data"])
                }
            else:
                return {
                    "status": "error",
                    "message": cost_result.get("message", "Cost Explorer query failed")
                }
                
        except Exception as e:
            self.logger.error(f"Cost Explorer storage query error: {str(e)}")
            return {
                "status": "error",
                "message": f"Cost Explorer storage query failed: {str(e)}"
            }
    
    def _process_cost_explorer_storage_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer storage data.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed storage cost data
        """
        try:
            processed_data = {
                "by_storage_class": {},
                "monthly_trends": {},
                "total_storage_cost": 0,
                "data_source": "cost_explorer"
            }
            
            # Process results by time period
            for result in cost_data.get("ResultsByTime", []):
                time_period = result.get("TimePeriod", {})
                start_date = time_period.get("Start", "")
                
                # Process groups (usage types)
                for group in result.get("Groups", []):
                    usage_type = group.get("Keys", ["Unknown"])[0]
                    metrics = group.get("Metrics", {})
                    
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                    
                    # Categorize by storage class
                    storage_class = self._extract_storage_class_from_usage_type(usage_type)
                    
                    if storage_class not in processed_data["by_storage_class"]:
                        processed_data["by_storage_class"][storage_class] = {
                            "total_cost": 0,
                            "total_usage_gb": 0,
                            "daily_costs": []
                        }
                    
                    processed_data["by_storage_class"][storage_class]["total_cost"] += cost
                    processed_data["by_storage_class"][storage_class]["total_usage_gb"] += usage
                    processed_data["by_storage_class"][storage_class]["daily_costs"].append({
                        "date": start_date,
                        "cost": cost,
                        "usage_gb": usage
                    })
                    
                    processed_data["total_storage_cost"] += cost
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing Cost Explorer storage data: {str(e)}")
            return {"error": str(e), "data_source": "cost_explorer"}
    
    def _extract_storage_class_from_usage_type(self, usage_type: str) -> str:
        """
        Extract storage class from Cost Explorer usage type.
        
        Args:
            usage_type: Usage type string from Cost Explorer
            
        Returns:
            Normalized storage class name
        """
        usage_type_lower = usage_type.lower()
        
        # Handle actual usage types seen in Cost Explorer data
        if "timedstorage" in usage_type_lower or "storage-bytehrs" in usage_type_lower:
            # This is storage cost - try to determine class from region prefix or default to STANDARD
            if "standard" in usage_type_lower and "ia" not in usage_type_lower:
                return "STANDARD"
            elif "standard-ia" in usage_type_lower or "standardia" in usage_type_lower:
                return "STANDARD_IA"
            elif "onezone" in usage_type_lower or "onezone-ia" in usage_type_lower:
                return "ONEZONE_IA"
            elif "glacier" in usage_type_lower and "deep" not in usage_type_lower:
                return "GLACIER"
            elif "deep" in usage_type_lower and "archive" in usage_type_lower:
                return "DEEP_ARCHIVE"
            elif "intelligent" in usage_type_lower or "int" in usage_type_lower:
                return "INTELLIGENT_TIERING"
            elif "reduced" in usage_type_lower or "rrs" in usage_type_lower:
                return "REDUCED_REDUNDANCY"
            else:
                return "STANDARD"  # Default for TimedStorage-ByteHrs
        elif "requests-tier1" in usage_type_lower or "requests-tier2" in usage_type_lower:
            return "API_REQUESTS"
        elif "datatransfer" in usage_type_lower or "aws-out-bytes" in usage_type_lower or "aws-in-bytes" in usage_type_lower:
            return "DATA_TRANSFER"
        else:
            return "OTHER"
    
    async def _enhance_storage_costs_with_pricing(self, storage_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance storage cost analysis with detailed pricing information.
        
        Args:
            storage_distribution: Storage distribution data
            
        Returns:
            Enhanced cost data with pricing details
        """
        try:
            enhanced_costs = {}
            
            # Get pricing for all storage classes
            all_pricing = self.pricing_service.get_all_storage_class_pricing()
            
            if all_pricing.get("status") == "success":
                enhanced_costs["pricing_by_class"] = all_pricing["storage_class_pricing"]
                enhanced_costs["pricing_comparison"] = all_pricing["comparison"]
            
            return enhanced_costs
            
        except Exception as e:
            self.logger.error(f"Error enhancing storage costs with pricing: {str(e)}")
            return {"error": str(e)}
    
    def _process_transfer_cost_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer data transfer cost data.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed transfer cost data
        """
        try:
            processed_data = {
                "cross_region_transfer": {"cost": 0, "volume_gb": 0},
                "internet_egress": {"cost": 0, "volume_gb": 0},
                "total_transfer_cost": 0
            }
            
            # Process transfer cost results
            for result in cost_data.get("ResultsByTime", []):
                for group in result.get("Groups", []):
                    usage_type = group.get("Keys", ["Unknown"])[0]
                    metrics = group.get("Metrics", {})
                    
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                    
                    # Categorize transfer type
                    if "regional" in usage_type.lower():
                        processed_data["cross_region_transfer"]["cost"] += cost
                        processed_data["cross_region_transfer"]["volume_gb"] += usage
                    elif "out" in usage_type.lower():
                        processed_data["internet_egress"]["cost"] += cost
                        processed_data["internet_egress"]["volume_gb"] += usage
                    
                    processed_data["total_transfer_cost"] += cost
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing transfer cost data: {str(e)}")
            return {"error": str(e)}
    
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
                "total_api_cost": 0,
                "request_volume": {}
            }
            
            # Process API cost results
            for result in cost_data.get("ResultsByTime", []):
                for group in result.get("Groups", []):
                    usage_type = group.get("Keys", ["Unknown"])[0]
                    metrics = group.get("Metrics", {})
                    
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                    
                    # Categorize request type
                    request_type = self._extract_request_type_from_usage_type(usage_type)
                    
                    if request_type not in processed_data["request_costs_by_type"]:
                        processed_data["request_costs_by_type"][request_type] = 0
                        processed_data["request_volume"][request_type] = 0
                    
                    processed_data["request_costs_by_type"][request_type] += cost
                    processed_data["request_volume"][request_type] += usage
                    processed_data["total_api_cost"] += cost
            
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
        
        if "tier1" in usage_type_lower or "put" in usage_type_lower:
            return "PUT_COPY_POST_LIST"
        elif "tier2" in usage_type_lower or "get" in usage_type_lower:
            return "GET_SELECT_OTHER"
        elif "lifecycle" in usage_type_lower:
            return "LIFECYCLE_TRANSITION"
        else:
            return "OTHER"
    
    def _process_retrieval_cost_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer retrieval cost data.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed retrieval cost data
        """
        try:
            processed_data = {
                "glacier_retrievals": {"cost": 0, "volume_gb": 0},
                "deep_archive_retrievals": {"cost": 0, "volume_gb": 0},
                "total_retrieval_cost": 0
            }
            
            # Process retrieval cost results
            for result in cost_data.get("ResultsByTime", []):
                for group in result.get("Groups", []):
                    usage_type = group.get("Keys", ["Unknown"])[0]
                    metrics = group.get("Metrics", {})
                    
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                    
                    # Categorize retrieval type
                    if "glacier" in usage_type.lower() and "deep" not in usage_type.lower():
                        processed_data["glacier_retrievals"]["cost"] += cost
                        processed_data["glacier_retrievals"]["volume_gb"] += usage
                    elif "deep" in usage_type.lower():
                        processed_data["deep_archive_retrievals"]["cost"] += cost
                        processed_data["deep_archive_retrievals"]["volume_gb"] += usage
                    
                    processed_data["total_retrieval_cost"] += cost
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing retrieval cost data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_request_metrics_for_buckets(self, bucket_names: List[str], lookback_days: int) -> Dict[str, Any]:
        """
        Get request metrics for specific buckets from CloudWatch.
        
        Args:
            bucket_names: List of bucket names to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Request metrics data
        """
        try:
            all_metrics = {}
            
            # Get request metrics for each bucket in parallel
            tasks = []
            for bucket_name in bucket_names:
                task = self.s3_service.get_request_metrics(bucket_name, lookback_days)
                tasks.append((bucket_name, task))
            
            # Execute all tasks
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            for i, (bucket_name, _) in enumerate(tasks):
                result = results[i]
                if not isinstance(result, Exception) and result.get("status") == "success":
                    all_metrics[bucket_name] = result["data"]
                else:
                    self.logger.warning(f"Failed to get request metrics for bucket {bucket_name}: {result}")
                    all_metrics[bucket_name] = {"error": str(result)}
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting request metrics for buckets: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_total_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total costs across all categories.
        
        Args:
            data: Analysis data containing all cost categories
            
        Returns:
            Total cost summary
        """
        try:
            total_costs = {
                "storage_cost": 0,
                "transfer_cost": 0,
                "api_cost": 0,
                "retrieval_cost": 0,
                "total_monthly_cost": 0,
                "total_annual_cost": 0
            }
            
            # Sum storage costs
            storage_costs = data.get("storage_costs", {})
            if isinstance(storage_costs, dict) and "total_storage_cost" in storage_costs:
                total_costs["storage_cost"] = storage_costs["total_storage_cost"]
            
            # Sum transfer costs
            transfer_costs = data.get("data_transfer_costs", {})
            if isinstance(transfer_costs, dict) and "total_transfer_cost" in transfer_costs:
                total_costs["transfer_cost"] = transfer_costs["total_transfer_cost"]
            
            # Sum API costs
            api_costs = data.get("api_costs", {})
            if isinstance(api_costs, dict) and "total_api_cost" in api_costs:
                total_costs["api_cost"] = api_costs["total_api_cost"]
            
            # Sum retrieval costs
            retrieval_costs = data.get("retrieval_costs", {})
            if isinstance(retrieval_costs, dict) and "total_retrieval_cost" in retrieval_costs:
                total_costs["retrieval_cost"] = retrieval_costs["total_retrieval_cost"]
            
            # Calculate totals
            total_costs["total_monthly_cost"] = (
                total_costs["storage_cost"] + 
                total_costs["transfer_cost"] + 
                total_costs["api_cost"] + 
                total_costs["retrieval_cost"]
            )
            total_costs["total_annual_cost"] = total_costs["total_monthly_cost"] * 12
            
            return total_costs
            
        except Exception as e:
            self.logger.error(f"Error calculating total costs: {str(e)}")
            return {"error": str(e)}
    
    def _create_cost_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed cost breakdown with percentages.
        
        Args:
            data: Analysis data
            
        Returns:
            Cost breakdown with percentages
        """
        try:
            total_costs = data.get("total_costs", {})
            total_monthly = total_costs.get("total_monthly_cost", 0)
            
            if total_monthly == 0:
                return {"message": "No cost data available for breakdown"}
            
            breakdown = {
                "storage": {
                    "cost": total_costs.get("storage_cost", 0),
                    "percentage": (total_costs.get("storage_cost", 0) / total_monthly * 100) if total_monthly > 0 else 0
                },
                "data_transfer": {
                    "cost": total_costs.get("transfer_cost", 0),
                    "percentage": (total_costs.get("transfer_cost", 0) / total_monthly * 100) if total_monthly > 0 else 0
                },
                "api_requests": {
                    "cost": total_costs.get("api_cost", 0),
                    "percentage": (total_costs.get("api_cost", 0) / total_monthly * 100) if total_monthly > 0 else 0
                },
                "data_retrieval": {
                    "cost": total_costs.get("retrieval_cost", 0),
                    "percentage": (total_costs.get("retrieval_cost", 0) / total_monthly * 100) if total_monthly > 0 else 0
                }
            }
            
            # Sort by cost (highest first)
            sorted_breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True))
            
            return {
                "breakdown": sorted_breakdown,
                "largest_cost_category": list(sorted_breakdown.keys())[0] if sorted_breakdown else "unknown",
                "total_monthly_cost": total_monthly
            }
            
        except Exception as e:
            self.logger.error(f"Error creating cost breakdown: {str(e)}")
            return {"error": str(e)}
    
    def _identify_optimization_opportunities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify cost optimization opportunities based on spend analysis.
        
        Args:
            data: Analysis data
            
        Returns:
            List of optimization opportunities
        """
        try:
            opportunities = []
            
            # Analyze storage class optimization opportunities
            storage_costs = data.get("storage_costs", {})
            if isinstance(storage_costs, dict) and "by_storage_class" in storage_costs:
                storage_opportunities = self._analyze_storage_class_opportunities(storage_costs["by_storage_class"])
                opportunities.extend(storage_opportunities)
            
            # Analyze data transfer optimization opportunities
            transfer_costs = data.get("data_transfer_costs", {})
            if isinstance(transfer_costs, dict):
                transfer_opportunities = self._analyze_transfer_opportunities(transfer_costs)
                opportunities.extend(transfer_opportunities)
            
            # Analyze API cost optimization opportunities
            api_costs = data.get("api_costs", {})
            if isinstance(api_costs, dict):
                api_opportunities = self._analyze_api_opportunities(api_costs)
                opportunities.extend(api_opportunities)
            
            # Sort opportunities by potential savings (highest first)
            opportunities.sort(key=lambda x: x.get("potential_savings", 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
    
    def _analyze_storage_class_opportunities(self, storage_by_class: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze storage class optimization opportunities.
        
        Args:
            storage_by_class: Storage costs by storage class
            
        Returns:
            List of storage class optimization opportunities
        """
        opportunities = []
        
        try:
            # Check for high Standard storage costs
            standard_cost = storage_by_class.get("STANDARD", {}).get("total_cost", 0)
            if standard_cost > 100:  # Threshold for significant Standard storage costs
                opportunities.append(self.create_recommendation(
                    rec_type="storage_optimization",
                    priority="high",
                    title="Optimize Standard Storage Class Usage",
                    description=f"High Standard storage costs (${standard_cost:.2f}/month) detected. Consider transitioning infrequently accessed data to Standard-IA or other lower-cost storage classes.",
                    potential_savings=standard_cost * 0.4,  # Estimate 40% savings
                    implementation_effort="medium",
                    action_items=[
                        "Analyze access patterns for Standard storage objects",
                        "Implement lifecycle policies to transition to Standard-IA after 30 days",
                        "Consider Intelligent Tiering for unpredictable access patterns"
                    ]
                ))
            
            # Check for Reduced Redundancy usage (deprecated)
            rr_cost = storage_by_class.get("REDUCED_REDUNDANCY", {}).get("total_cost", 0)
            if rr_cost > 0:
                opportunities.append(self.create_recommendation(
                    rec_type="storage_optimization",
                    priority="high",
                    title="Migrate from Reduced Redundancy Storage",
                    description=f"Reduced Redundancy storage (${rr_cost:.2f}/month) is deprecated and more expensive than Standard storage. Migrate to Standard storage class.",
                    potential_savings=rr_cost * 0.2,  # Estimate 20% savings
                    implementation_effort="low",
                    action_items=[
                        "Identify objects using Reduced Redundancy storage",
                        "Copy objects to Standard storage class",
                        "Update applications to use Standard storage for new objects"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error analyzing storage class opportunities: {str(e)}")
        
        return opportunities
    
    def _analyze_transfer_opportunities(self, transfer_costs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze data transfer optimization opportunities.
        
        Args:
            transfer_costs: Data transfer cost data
            
        Returns:
            List of transfer optimization opportunities
        """
        opportunities = []
        
        try:
            # Check for high internet egress costs
            egress_cost = transfer_costs.get("internet_egress", {}).get("cost", 0)
            if egress_cost > 50:  # Threshold for significant egress costs
                opportunities.append(self.create_recommendation(
                    rec_type="transfer_optimization",
                    priority="high",
                    title="Optimize Internet Data Transfer Costs",
                    description=f"High internet egress costs (${egress_cost:.2f}/month) detected. Consider using CloudFront CDN to reduce data transfer charges.",
                    potential_savings=egress_cost * 0.6,  # Estimate 60% savings with CloudFront
                    implementation_effort="medium",
                    action_items=[
                        "Analyze data transfer patterns and frequently accessed content",
                        "Implement CloudFront distribution for static content",
                        "Configure appropriate caching policies",
                        "Monitor CloudFront usage and costs"
                    ]
                ))
            
            # Check for cross-region transfer costs
            cross_region_cost = transfer_costs.get("cross_region_transfer", {}).get("cost", 0)
            if cross_region_cost > 20:  # Threshold for significant cross-region costs
                opportunities.append(self.create_recommendation(
                    rec_type="transfer_optimization",
                    priority="medium",
                    title="Optimize Cross-Region Data Transfer",
                    description=f"Cross-region transfer costs (${cross_region_cost:.2f}/month) detected. Consider data locality optimization and regional data placement strategies.",
                    potential_savings=cross_region_cost * 0.5,  # Estimate 50% savings
                    implementation_effort="high",
                    action_items=[
                        "Analyze cross-region data access patterns",
                        "Implement regional data placement strategies",
                        "Consider S3 Cross-Region Replication optimization",
                        "Evaluate application architecture for data locality"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error analyzing transfer opportunities: {str(e)}")
        
        return opportunities
    
    def _analyze_api_opportunities(self, api_costs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze API cost optimization opportunities.
        
        Args:
            api_costs: API cost data
            
        Returns:
            List of API optimization opportunities
        """
        opportunities = []
        
        try:
            # Check for high API request costs
            total_api_cost = api_costs.get("total_api_cost", 0)
            if total_api_cost > 25:  # Threshold for significant API costs
                opportunities.append(self.create_recommendation(
                    rec_type="api_optimization",
                    priority="medium",
                    title="Optimize S3 API Request Patterns",
                    description=f"High S3 API request costs (${total_api_cost:.2f}/month) detected. Consider request optimization strategies and caching.",
                    potential_savings=total_api_cost * 0.3,  # Estimate 30% savings
                    implementation_effort="medium",
                    action_items=[
                        "Analyze application request patterns",
                        "Implement client-side caching for frequently accessed objects",
                        "Optimize list operations with appropriate prefixes",
                        "Consider batch operations where possible",
                        "Implement request rate limiting and retry logic"
                    ]
                ))
            
            # Check for high PUT/LIST request costs
            put_costs = api_costs.get("request_costs_by_type", {}).get("PUT_COPY_POST_LIST", 0)
            if put_costs > 15:  # Threshold for high PUT/LIST costs
                opportunities.append(self.create_recommendation(
                    rec_type="api_optimization",
                    priority="medium",
                    title="Optimize PUT and LIST Request Usage",
                    description=f"High PUT/LIST request costs (${put_costs:.2f}/month) detected. These requests are more expensive than GET requests.",
                    potential_savings=put_costs * 0.25,  # Estimate 25% savings
                    implementation_effort="low",
                    action_items=[
                        "Review application upload patterns",
                        "Implement multipart upload for large objects",
                        "Optimize LIST operations with pagination",
                        "Consider reducing unnecessary COPY operations"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error analyzing API opportunities: {str(e)}")
        
        return opportunities
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on general spend analysis.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of actionable recommendations
        """
        try:
            recommendations = []
            
            # Get optimization opportunities from analysis
            optimization_opportunities = analysis_results.get("data", {}).get("optimization_opportunities", [])
            recommendations.extend(optimization_opportunities)
            
            # Add general recommendations based on data sources used
            data_sources = analysis_results.get("data_sources", [])
            
            if "storage_lens" not in data_sources:
                recommendations.append(self.create_recommendation(
                    rec_type="configuration",
                    priority="high",
                    title="Enable S3 Storage Lens for Better Cost Visibility",
                    description="Storage Lens provides comprehensive S3 metrics without additional costs. Enable it for better cost optimization insights.",
                    implementation_effort="low",
                    action_items=[
                        "Enable S3 Storage Lens in the AWS Console",
                        "Configure cost optimization metrics",
                        "Set up data export for detailed analysis",
                        "Review Storage Lens dashboard regularly"
                    ]
                ))
            
            # Add cost monitoring recommendation
            total_monthly_cost = analysis_results.get("data", {}).get("total_costs", {}).get("total_monthly_cost", 0)
            if total_monthly_cost > 100:
                recommendations.append(self.create_recommendation(
                    rec_type="monitoring",
                    priority="medium",
                    title="Implement S3 Cost Monitoring and Alerting",
                    description=f"With monthly S3 costs of ${total_monthly_cost:.2f}, implement proactive cost monitoring to prevent unexpected charges.",
                    implementation_effort="medium",
                    action_items=[
                        "Set up AWS Budgets for S3 spending alerts",
                        "Configure Cost Anomaly Detection for S3",
                        "Implement regular cost review processes",
                        "Create cost allocation tags for better tracking"
                    ]
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return [self.create_recommendation(
                rec_type="error",
                priority="high",
                title="Analysis Error",
                description=f"Error generating recommendations: {str(e)}",
                implementation_effort="low",
                action_items=["Review analysis logs", "Check AWS permissions", "Retry analysis"]
            )]