"""
Multipart Cleanup Analyzer

This analyzer identifies and analyzes incomplete multipart uploads using Storage Lens
as the primary data source. It calculates cost waste from incomplete uploads and
generates cleanup recommendations and lifecycle policy suggestions.

Requirements covered:
- 7.1: Use S3 Storage Lens as primary data source for multipart upload analysis
- 7.2: Identify incomplete uploads showing age, size, and associated costs
- 7.3: Calculate cost savings from cleanup
- 7.4: Provide actionable cleanup steps
- 7.5: Recommend lifecycle policies for automatic cleanup prevention
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage

logger = logging.getLogger(__name__)


class MultipartCleanupAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying and cleaning up incomplete multipart uploads.
    
    Uses Storage Lens as primary data source to identify incomplete multipart uploads,
    calculate storage waste costs, and generate cleanup recommendations with
    prevention strategies through lifecycle policies.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize MultipartCleanupAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations (NO-COST operations only)
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "multipart_cleanup"
        
        # Multipart upload thresholds (configurable)
        self.cleanup_thresholds = {
            "min_age_days": 7,  # Minimum age for cleanup consideration
            "max_results_per_bucket": 1000,  # Limit to prevent timeout
            "cost_threshold_dollars": 1.0,  # Minimum cost to flag for cleanup
            "size_threshold_mb": 100  # Minimum size to flag for cleanup
        }
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute multipart cleanup analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - bucket_names: Optional list of specific buckets to analyze
                - min_age_days: Minimum age for cleanup consideration (default: 7)
                - max_results_per_bucket: Maximum uploads to analyze per bucket (default: 1000)
                - include_cost_analysis: Whether to calculate waste costs (default: True)
                
        Returns:
            Dictionary containing multipart cleanup analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting multipart cleanup analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "incomplete_uploads": {},
                    "cost_waste_analysis": {},
                    "cleanup_recommendations": {},
                    "lifecycle_policy_suggestions": {},
                    "prevention_strategies": {},
                    "optimization_summary": {}
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute analysis components in parallel
            tasks = [
                self._identify_incomplete_multipart_uploads(context),
                self._analyze_multipart_storage_waste(context),
                self._analyze_existing_lifecycle_policies(context)
            ]
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            uploads_result, waste_result, lifecycle_result = results
            
            # Process incomplete uploads identification
            if not isinstance(uploads_result, Exception) and uploads_result.get("status") == "success":
                analysis_results["data"]["incomplete_uploads"] = uploads_result["data"]
                analysis_results["data_sources"].extend(uploads_result.get("data_sources", []))
            else:
                self.logger.warning(f"Incomplete uploads identification failed: {uploads_result}")
                analysis_results["data"]["incomplete_uploads"] = {"error": str(uploads_result)}
            
            # Process storage waste analysis
            if not isinstance(waste_result, Exception) and waste_result.get("status") == "success":
                analysis_results["data"]["cost_waste_analysis"] = waste_result["data"]
                analysis_results["data_sources"].extend(waste_result.get("data_sources", []))
            else:
                self.logger.warning(f"Storage waste analysis failed: {waste_result}")
                analysis_results["data"]["cost_waste_analysis"] = {"error": str(waste_result)}
            
            # Process lifecycle policy analysis
            if not isinstance(lifecycle_result, Exception) and lifecycle_result.get("status") == "success":
                analysis_results["data"]["existing_lifecycle_policies"] = lifecycle_result["data"]
                analysis_results["data_sources"].extend(lifecycle_result.get("data_sources", []))
            else:
                self.logger.warning(f"Lifecycle policy analysis failed: {lifecycle_result}")
                analysis_results["data"]["existing_lifecycle_policies"] = {"error": str(lifecycle_result)}
            
            # Generate cleanup recommendations
            analysis_results["data"]["cleanup_recommendations"] = await self._generate_cleanup_recommendations(
                analysis_results["data"], context
            )
            
            # Generate lifecycle policy suggestions
            analysis_results["data"]["lifecycle_policy_suggestions"] = await self._generate_lifecycle_policy_suggestions(
                analysis_results["data"], context
            )
            
            # Generate prevention strategies
            analysis_results["data"]["prevention_strategies"] = self._generate_prevention_strategies(
                analysis_results["data"], context
            )
            
            # Create optimization summary
            analysis_results["data"]["optimization_summary"] = self._create_optimization_summary(analysis_results["data"])
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Completed multipart cleanup analysis in {analysis_results['execution_time']:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in multipart cleanup analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _identify_incomplete_multipart_uploads(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify incomplete multipart uploads using Storage Lens and S3 service.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing incomplete multipart uploads analysis
        """
        try:
            self.logger.debug("Identifying incomplete multipart uploads")
            
            uploads_analysis = {
                "by_bucket": {},
                "by_age_group": {},
                "total_incomplete_uploads": 0,
                "total_waste_size_gb": 0,
                "storage_lens_metrics": {},
                "detailed_uploads": []
            }
            
            data_sources = []
            
            # Try Storage Lens first (NO-COST primary source)
            if self.storage_lens_service:
                try:
                    # Get multipart upload metrics from Storage Lens
                    storage_lens_result = await self.storage_lens_service.get_incomplete_multipart_uploads_metrics()
                    
                    if storage_lens_result.get("status") == "success":
                        self.logger.info("Using Storage Lens as primary data source for multipart uploads")
                        uploads_analysis["storage_lens_metrics"] = storage_lens_result["data"]
                        data_sources.append("storage_lens")
                        
                        # Check if multipart tracking is available
                        if storage_lens_result["data"].get("MultipartUploadTrackingAvailable", False):
                            uploads_analysis["storage_lens_available"] = True
                            uploads_analysis["note"] = "Storage Lens multipart tracking is enabled"
                        else:
                            uploads_analysis["storage_lens_available"] = False
                            uploads_analysis["note"] = "Enable Cost Optimization Metrics in Storage Lens for detailed multipart tracking"
                    else:
                        self.logger.warning(f"Storage Lens multipart metrics unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens multipart analysis failed: {str(e)}")
            
            # Use S3 service for direct multipart upload identification (NO-COST operations only)
            if self.s3_service:
                try:
                    self.logger.info("Using S3 service for direct multipart upload identification")
                    bucket_analysis_result = await self._analyze_buckets_for_multipart_uploads(context)
                    
                    if bucket_analysis_result.get("status") == "success":
                        uploads_analysis.update(bucket_analysis_result["data"])
                        data_sources.append("s3_service")
                    else:
                        self.logger.warning(f"S3 service multipart analysis failed: {bucket_analysis_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"S3 service multipart analysis failed: {str(e)}")
            
            return {
                "status": "success",
                "data": uploads_analysis,
                "data_sources": data_sources,
                "message": f"Incomplete multipart uploads identification completed using: {', '.join(data_sources) if data_sources else 'fallback analysis'}"
            }
            
        except Exception as e:
            self.logger.error(f"Incomplete multipart uploads identification error: {str(e)}")
            return {
                "status": "error",
                "message": f"Incomplete multipart uploads identification failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_multipart_storage_waste(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze storage waste and costs from incomplete multipart uploads.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing storage waste analysis
        """
        try:
            self.logger.debug("Analyzing multipart storage waste")
            
            waste_analysis = {
                "total_waste_cost_monthly": 0,
                "waste_by_bucket": {},
                "waste_by_age_group": {},
                "cost_breakdown": {
                    "storage_cost": 0,
                    "request_cost": 0,
                    "total_cost": 0
                },
                "potential_savings": 0
            }
            
            data_sources = []
            
            # Get storage waste costs from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query for S3 storage costs to estimate multipart waste
                s3_filter = {
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": ["Amazon Simple Storage Service"]
                    }
                }
                
                cost_result = get_cost_and_usage(
                    start_date=start_date,
                    end_date=end_date,
                    granularity="DAILY",
                    metrics=["UnblendedCost", "UsageQuantity"],
                    group_by=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                    filter_expr=s3_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    waste_analysis.update(self._process_storage_cost_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer storage data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer waste analysis failed: {str(e)}")
            
            # Enhance with pricing calculations if pricing service available
            if self.pricing_service:
                try:
                    pricing_result = await self.pricing_service.get_storage_class_pricing()
                    if pricing_result.get("status") == "success":
                        waste_analysis["pricing_data"] = pricing_result["pricing"]
                        data_sources.append("pricing_service")
                except Exception as e:
                    self.logger.warning(f"Pricing service calculation failed: {str(e)}")
            
            # Use fallback cost estimation if no data sources available
            if not data_sources:
                waste_analysis.update(self._estimate_multipart_waste_fallback(context))
                data_sources.append("fallback_estimation")
            
            return {
                "status": "success",
                "data": waste_analysis,
                "data_sources": data_sources,
                "message": f"Multipart storage waste analysis completed using: {', '.join(data_sources)}"
            }
            
        except Exception as e:
            self.logger.error(f"Multipart storage waste analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Multipart storage waste analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_existing_lifecycle_policies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze existing lifecycle policies for multipart upload cleanup rules.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing lifecycle policy analysis
        """
        try:
            self.logger.debug("Analyzing existing lifecycle policies")
            
            lifecycle_analysis = {
                "buckets_with_multipart_rules": [],
                "buckets_without_multipart_rules": [],
                "policy_effectiveness": {},
                "recommended_improvements": []
            }
            
            data_sources = []
            
            # Analyze bucket lifecycle policies (NO-COST operations)
            if self.s3_service:
                try:
                    bucket_names = context.get('bucket_names')
                    if not bucket_names:
                        # Get all buckets if none specified (NO-COST operation)
                        buckets_result = await self.s3_service.list_buckets()
                        if buckets_result.get("status") == "success":
                            # Fix: S3Service returns buckets under data.Buckets, not buckets
                            buckets_data = buckets_result.get("data", {}).get("Buckets", [])
                            bucket_names = [bucket["Name"] for bucket in buckets_data]
                    
                    if bucket_names:
                        # Limit to prevent timeout
                        bucket_names = bucket_names[:20]
                        
                        lifecycle_tasks = [
                            self._analyze_bucket_lifecycle_policy(bucket_name) for bucket_name in bucket_names
                        ]
                        
                        lifecycle_results = await asyncio.gather(*lifecycle_tasks, return_exceptions=True)
                        
                        for i, result in enumerate(lifecycle_results):
                            if not isinstance(result, Exception) and result.get("status") == "success":
                                bucket_name = bucket_names[i]
                                bucket_data = result["data"]
                                
                                if bucket_data.get("has_multipart_rule", False):
                                    lifecycle_analysis["buckets_with_multipart_rules"].append({
                                        "bucket_name": bucket_name,
                                        "rule_details": bucket_data.get("multipart_rule_details", {})
                                    })
                                else:
                                    lifecycle_analysis["buckets_without_multipart_rules"].append({
                                        "bucket_name": bucket_name,
                                        "has_lifecycle_policy": bucket_data.get("has_lifecycle_policy", False)
                                    })
                        
                        data_sources.append("s3_service")
                        
                except Exception as e:
                    self.logger.warning(f"Bucket lifecycle policy analysis failed: {str(e)}")
            
            # Generate policy effectiveness analysis
            lifecycle_analysis["policy_effectiveness"] = self._analyze_policy_effectiveness(lifecycle_analysis)
            
            return {
                "status": "success",
                "data": lifecycle_analysis,
                "data_sources": data_sources,
                "message": f"Lifecycle policy analysis completed using: {', '.join(data_sources) if data_sources else 'fallback analysis'}"
            }
            
        except Exception as e:
            self.logger.error(f"Lifecycle policy analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Lifecycle policy analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_buckets_for_multipart_uploads(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze buckets for incomplete multipart uploads using S3 service (NO-COST operations only).
        
        Args:
            context: Analysis context
            
        Returns:
            Bucket-level multipart uploads analysis
        """
        try:
            bucket_analysis = {
                "by_bucket": {},
                "total_incomplete_uploads": 0,
                "total_waste_size_gb": 0
            }
            
            bucket_names = context.get('bucket_names')
            if not bucket_names:
                # Get all buckets (NO-COST operation)
                buckets_result = await self.s3_service.list_buckets()
                if buckets_result.get("status") == "success":
                    # Fix: S3Service returns buckets under data.Buckets, not buckets
                    buckets_data = buckets_result.get("data", {}).get("Buckets", [])
                    bucket_names = [bucket["Name"] for bucket in buckets_data][:15]  # Limit to prevent timeout
            
            if bucket_names:
                # Analyze each bucket for multipart uploads (NO-COST operations only)
                bucket_tasks = [
                    self._analyze_single_bucket_multipart_uploads(bucket_name, context) for bucket_name in bucket_names
                ]
                
                bucket_results = await asyncio.gather(*bucket_tasks, return_exceptions=True)
                
                for i, result in enumerate(bucket_results):
                    if not isinstance(result, Exception) and result.get("status") == "success":
                        bucket_name = bucket_names[i]
                        bucket_data = result["data"]
                        bucket_analysis["by_bucket"][bucket_name] = bucket_data
                        bucket_analysis["total_incomplete_uploads"] += bucket_data.get("upload_count", 0)
                        bucket_analysis["total_waste_size_gb"] += bucket_data.get("total_size_gb", 0)
            
            return {
                "status": "success",
                "data": bucket_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Bucket multipart analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Bucket multipart analysis failed: {str(e)}"
            }
    
    async def _analyze_single_bucket_multipart_uploads(self, bucket_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single bucket for incomplete multipart uploads (NO-COST operations only).
        
        Args:
            bucket_name: Name of the bucket to analyze
            context: Analysis context
            
        Returns:
            Single bucket multipart uploads analysis
        """
        try:
            bucket_data = {
                "bucket_name": bucket_name,
                "upload_count": 0,
                "total_size_gb": 0,
                "uploads_by_age": {
                    "1_7_days": 0,
                    "8_30_days": 0,
                    "31_90_days": 0,
                    "over_90_days": 0
                },
                "oldest_upload_days": 0,
                "estimated_monthly_cost": 0,
                "uploads": []
            }
            
            min_age_days = context.get('min_age_days', self.cleanup_thresholds['min_age_days'])
            max_results = context.get('max_results_per_bucket', self.cleanup_thresholds['max_results_per_bucket'])
            
            # List incomplete multipart uploads (NO-COST operation - metadata only)
            try:
                multipart_result = await self.s3_service.get_multipart_uploads(
                    bucket_name, 
                    max_results=max_results
                )
                
                if multipart_result.get("status") == "success":
                    uploads = multipart_result.get("uploads", [])
                    current_time = datetime.now()
                    
                    for upload in uploads:
                        initiated = upload.get("Initiated")
                        if initiated:
                            # Calculate age in days
                            if isinstance(initiated, str):
                                initiated = datetime.fromisoformat(initiated.replace('Z', '+00:00'))
                            
                            age_days = (current_time - initiated.replace(tzinfo=None)).days
                            
                            # Only consider uploads older than minimum age
                            if age_days >= min_age_days:
                                bucket_data["upload_count"] += 1
                                
                                # Estimate size (this is approximate since we can't get exact size without cost)
                                estimated_size_mb = 100  # Conservative estimate for incomplete uploads
                                bucket_data["total_size_gb"] += estimated_size_mb / 1024
                                
                                # Categorize by age
                                if age_days <= 7:
                                    bucket_data["uploads_by_age"]["1_7_days"] += 1
                                elif age_days <= 30:
                                    bucket_data["uploads_by_age"]["8_30_days"] += 1
                                elif age_days <= 90:
                                    bucket_data["uploads_by_age"]["31_90_days"] += 1
                                else:
                                    bucket_data["uploads_by_age"]["over_90_days"] += 1
                                
                                # Track oldest upload
                                if age_days > bucket_data["oldest_upload_days"]:
                                    bucket_data["oldest_upload_days"] = age_days
                                
                                # Store upload details (limited to prevent large responses)
                                if len(bucket_data["uploads"]) < 10:
                                    bucket_data["uploads"].append({
                                        "upload_id": upload.get("UploadId", ""),
                                        "key": upload.get("Key", ""),
                                        "initiated": initiated.isoformat(),
                                        "age_days": age_days,
                                        "estimated_size_mb": estimated_size_mb
                                    })
                else:
                    self.logger.debug(f"No multipart uploads found for bucket {bucket_name}")
                    
            except Exception as e:
                self.logger.debug(f"Multipart uploads check failed for {bucket_name}: {str(e)}")
                bucket_data["error"] = str(e)
            
            # Estimate monthly cost (approximate)
            if bucket_data["total_size_gb"] > 0:
                # Standard storage cost estimate: ~$0.023 per GB per month
                bucket_data["estimated_monthly_cost"] = bucket_data["total_size_gb"] * 0.023
            
            return {
                "status": "success",
                "data": bucket_data
            }
            
        except Exception as e:
            self.logger.error(f"Single bucket multipart analysis error for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Single bucket multipart analysis failed: {str(e)}"
            }
    
    async def _analyze_bucket_lifecycle_policy(self, bucket_name: str) -> Dict[str, Any]:
        """
        Analyze a bucket's lifecycle policy for multipart upload cleanup rules.
        
        Args:
            bucket_name: Name of the bucket to analyze
            
        Returns:
            Bucket lifecycle policy analysis
        """
        try:
            policy_data = {
                "bucket_name": bucket_name,
                "has_lifecycle_policy": False,
                "has_multipart_rule": False,
                "multipart_rule_details": {},
                "recommendations": []
            }
            
            # Get lifecycle configuration (NO-COST operation)
            try:
                lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
                
                if lifecycle_result.get("status") == "success":
                    policy_data["has_lifecycle_policy"] = True
                    lifecycle_config = lifecycle_result.get("lifecycle_configuration", {})
                    
                    # Check for multipart upload cleanup rules
                    rules = lifecycle_config.get("Rules", [])
                    for rule in rules:
                        abort_rule = rule.get("AbortIncompleteMultipartUpload")
                        if abort_rule:
                            policy_data["has_multipart_rule"] = True
                            policy_data["multipart_rule_details"] = {
                                "rule_id": rule.get("ID", ""),
                                "status": rule.get("Status", ""),
                                "days_after_initiation": abort_rule.get("DaysAfterInitiation", 0),
                                "filter": rule.get("Filter", {})
                            }
                            break
                    
                    if not policy_data["has_multipart_rule"]:
                        policy_data["recommendations"].append({
                            "type": "add_multipart_rule",
                            "description": "Add AbortIncompleteMultipartUpload rule to existing lifecycle policy"
                        })
                else:
                    policy_data["has_lifecycle_policy"] = False
                    policy_data["recommendations"].append({
                        "type": "create_lifecycle_policy",
                        "description": "Create lifecycle policy with AbortIncompleteMultipartUpload rule"
                    })
                    
            except Exception as e:
                self.logger.debug(f"Lifecycle policy check failed for {bucket_name}: {str(e)}")
                policy_data["error"] = str(e)
            
            return {
                "status": "success",
                "data": policy_data
            }
            
        except Exception as e:
            self.logger.error(f"Bucket lifecycle policy analysis error for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Bucket lifecycle policy analysis failed: {str(e)}"
            }
    
    async def _generate_cleanup_recommendations(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate cleanup recommendations based on analysis results.
        
        Args:
            analysis_data: Combined analysis data
            context: Analysis context
            
        Returns:
            Dictionary containing cleanup recommendations
        """
        try:
            self.logger.debug("Generating cleanup recommendations")
            
            recommendations = {
                "immediate_cleanup": [],
                "scheduled_cleanup": [],
                "automated_cleanup": [],
                "priority_buckets": [],
                "cost_impact": {}
            }
            
            incomplete_uploads = analysis_data.get("incomplete_uploads", {})
            waste_analysis = analysis_data.get("cost_waste_analysis", {})
            
            # Generate immediate cleanup recommendations
            for bucket_name, bucket_data in incomplete_uploads.get("by_bucket", {}).items():
                upload_count = bucket_data.get("upload_count", 0)
                estimated_cost = bucket_data.get("estimated_monthly_cost", 0)
                oldest_upload_days = bucket_data.get("oldest_upload_days", 0)
                
                if upload_count > 0:
                    priority = self._calculate_cleanup_priority(upload_count, estimated_cost, oldest_upload_days)
                    
                    recommendation = {
                        "bucket_name": bucket_name,
                        "priority": priority,
                        "upload_count": upload_count,
                        "estimated_monthly_savings": estimated_cost,
                        "oldest_upload_days": oldest_upload_days,
                        "action_items": self._generate_bucket_cleanup_actions(bucket_data),
                        "implementation_effort": "low" if upload_count < 100 else "medium"
                    }
                    
                    if priority == "high":
                        recommendations["immediate_cleanup"].append(recommendation)
                        recommendations["priority_buckets"].append(bucket_name)
                    elif priority == "medium":
                        recommendations["scheduled_cleanup"].append(recommendation)
                    else:
                        recommendations["automated_cleanup"].append(recommendation)
            
            # Calculate total cost impact
            total_potential_savings = sum(
                rec.get("estimated_monthly_savings", 0) 
                for rec in recommendations["immediate_cleanup"] + recommendations["scheduled_cleanup"]
            )
            
            recommendations["cost_impact"] = {
                "total_monthly_savings": total_potential_savings,
                "annual_savings": total_potential_savings * 12,
                "high_priority_savings": sum(
                    rec.get("estimated_monthly_savings", 0) 
                    for rec in recommendations["immediate_cleanup"]
                )
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Cleanup recommendations generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_lifecycle_policy_suggestions(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate lifecycle policy suggestions for preventing future incomplete uploads.
        
        Args:
            analysis_data: Combined analysis data
            context: Analysis context
            
        Returns:
            Dictionary containing lifecycle policy suggestions
        """
        try:
            self.logger.debug("Generating lifecycle policy suggestions")
            
            suggestions = {
                "new_policies": [],
                "policy_updates": [],
                "best_practices": [],
                "implementation_templates": {}
            }
            
            lifecycle_data = analysis_data.get("existing_lifecycle_policies", {})
            
            # Generate suggestions for buckets without multipart rules
            buckets_without_rules = lifecycle_data.get("buckets_without_multipart_rules", [])
            
            for bucket_info in buckets_without_rules:
                bucket_name = bucket_info.get("bucket_name")
                has_lifecycle_policy = bucket_info.get("has_lifecycle_policy", False)
                
                if has_lifecycle_policy:
                    # Update existing policy
                    suggestions["policy_updates"].append({
                        "bucket_name": bucket_name,
                        "action": "add_multipart_rule",
                        "description": f"Add AbortIncompleteMultipartUpload rule to existing lifecycle policy for {bucket_name}",
                        "recommended_days": 7,
                        "implementation_effort": "low"
                    })
                else:
                    # Create new policy
                    suggestions["new_policies"].append({
                        "bucket_name": bucket_name,
                        "action": "create_lifecycle_policy",
                        "description": f"Create new lifecycle policy with multipart cleanup rule for {bucket_name}",
                        "recommended_days": 7,
                        "implementation_effort": "low"
                    })
            
            # Generate best practices
            suggestions["best_practices"] = [
                {
                    "title": "Standard Multipart Cleanup Rule",
                    "description": "Set AbortIncompleteMultipartUpload to 7 days for all buckets",
                    "rationale": "Prevents accumulation of incomplete uploads while allowing reasonable time for completion"
                },
                {
                    "title": "Shorter Cleanup for High-Volume Buckets",
                    "description": "Use 1-3 days for buckets with frequent multipart uploads",
                    "rationale": "Reduces storage costs in high-activity environments"
                },
                {
                    "title": "Monitor Lifecycle Policy Effectiveness",
                    "description": "Regularly review and adjust cleanup periods based on usage patterns",
                    "rationale": "Ensures policies remain effective as usage patterns change"
                }
            ]
            
            # Generate implementation templates
            suggestions["implementation_templates"] = {
                "basic_multipart_cleanup": {
                    "Rules": [
                        {
                            "ID": "MultipartUploadCleanup",
                            "Status": "Enabled",
                            "Filter": {"Prefix": ""},
                            "AbortIncompleteMultipartUpload": {
                                "DaysAfterInitiation": 7
                            }
                        }
                    ]
                },
                "comprehensive_lifecycle": {
                    "Rules": [
                        {
                            "ID": "MultipartUploadCleanup",
                            "Status": "Enabled",
                            "Filter": {"Prefix": ""},
                            "AbortIncompleteMultipartUpload": {
                                "DaysAfterInitiation": 7
                            }
                        },
                        {
                            "ID": "NonCurrentVersionCleanup",
                            "Status": "Enabled",
                            "Filter": {"Prefix": ""},
                            "NoncurrentVersionExpiration": {
                                "NoncurrentDays": 30
                            }
                        }
                    ]
                }
            }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Lifecycle policy suggestions generation error: {str(e)}")
            return {"error": str(e)}
    
    def _generate_prevention_strategies(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prevention strategies for future incomplete uploads.
        
        Args:
            analysis_data: Combined analysis data
            context: Analysis context
            
        Returns:
            Dictionary containing prevention strategies
        """
        try:
            strategies = {
                "application_level": [],
                "infrastructure_level": [],
                "monitoring_strategies": [],
                "governance_policies": []
            }
            
            # Application-level strategies
            strategies["application_level"] = [
                {
                    "strategy": "Implement Proper Error Handling",
                    "description": "Ensure applications properly handle multipart upload failures and clean up incomplete uploads",
                    "implementation": "Add try-catch blocks around multipart upload operations with cleanup in finally blocks",
                    "effort": "medium"
                },
                {
                    "strategy": "Use AWS SDK Retry Logic",
                    "description": "Configure AWS SDK with appropriate retry policies for multipart uploads",
                    "implementation": "Set exponential backoff and maximum retry attempts in SDK configuration",
                    "effort": "low"
                },
                {
                    "strategy": "Implement Upload Timeout Handling",
                    "description": "Set reasonable timeouts for multipart upload operations",
                    "implementation": "Configure connection and read timeouts in application code",
                    "effort": "low"
                }
            ]
            
            # Infrastructure-level strategies
            strategies["infrastructure_level"] = [
                {
                    "strategy": "Automated Lifecycle Policies",
                    "description": "Implement lifecycle policies on all S3 buckets to automatically clean up incomplete uploads",
                    "implementation": "Deploy lifecycle policies via CloudFormation or Terraform",
                    "effort": "low"
                },
                {
                    "strategy": "Bucket Policy Templates",
                    "description": "Create standardized bucket templates that include multipart cleanup rules",
                    "implementation": "Use AWS Config rules to enforce lifecycle policy presence",
                    "effort": "medium"
                },
                {
                    "strategy": "Cross-Region Replication Considerations",
                    "description": "Ensure multipart cleanup policies are consistent across replicated buckets",
                    "implementation": "Include lifecycle policies in replication configuration",
                    "effort": "low"
                }
            ]
            
            # Monitoring strategies
            strategies["monitoring_strategies"] = [
                {
                    "strategy": "CloudWatch Metrics Monitoring",
                    "description": "Monitor S3 metrics for incomplete multipart uploads",
                    "implementation": "Set up CloudWatch alarms for multipart upload metrics",
                    "effort": "medium"
                },
                {
                    "strategy": "Storage Lens Dashboard Monitoring",
                    "description": "Use Storage Lens to track multipart upload waste across accounts",
                    "implementation": "Enable Cost Optimization Metrics in Storage Lens configuration",
                    "effort": "low"
                },
                {
                    "strategy": "Regular Cleanup Audits",
                    "description": "Schedule regular audits to identify and clean up incomplete uploads",
                    "implementation": "Create Lambda function for automated cleanup reporting",
                    "effort": "medium"
                }
            ]
            
            # Governance policies
            strategies["governance_policies"] = [
                {
                    "strategy": "Mandatory Lifecycle Policies",
                    "description": "Require all S3 buckets to have lifecycle policies with multipart cleanup rules",
                    "implementation": "Use AWS Config rules and Service Control Policies (SCPs)",
                    "effort": "medium"
                },
                {
                    "strategy": "Cost Allocation Tags",
                    "description": "Tag buckets and track multipart upload costs by team or project",
                    "implementation": "Implement tagging strategy and cost allocation reports",
                    "effort": "low"
                },
                {
                    "strategy": "Developer Training",
                    "description": "Train development teams on proper multipart upload handling",
                    "implementation": "Create documentation and training materials",
                    "effort": "medium"
                }
            ]
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Prevention strategies generation error: {str(e)}")
            return {"error": str(e)}
    
    def _create_optimization_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create optimization summary from analysis results.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Dictionary containing optimization summary
        """
        try:
            incomplete_uploads = analysis_data.get("incomplete_uploads", {})
            cleanup_recommendations = analysis_data.get("cleanup_recommendations", {})
            lifecycle_suggestions = analysis_data.get("lifecycle_policy_suggestions", {})
            
            summary = {
                "total_incomplete_uploads": incomplete_uploads.get("total_incomplete_uploads", 0),
                "total_waste_size_gb": incomplete_uploads.get("total_waste_size_gb", 0),
                "buckets_analyzed": len(incomplete_uploads.get("by_bucket", {})),
                "buckets_with_issues": len([
                    bucket for bucket, data in incomplete_uploads.get("by_bucket", {}).items()
                    if data.get("upload_count", 0) > 0
                ]),
                "potential_monthly_savings": cleanup_recommendations.get("cost_impact", {}).get("total_monthly_savings", 0),
                "high_priority_buckets": len(cleanup_recommendations.get("priority_buckets", [])),
                "lifecycle_policy_gaps": len(lifecycle_suggestions.get("new_policies", [])) + len(lifecycle_suggestions.get("policy_updates", [])),
                "storage_lens_available": incomplete_uploads.get("storage_lens_metrics", {}).get("MultipartUploadTrackingAvailable", False)
            }
            
            # Generate key insights
            insights = []
            
            if summary["total_incomplete_uploads"] > 0:
                insights.append(f"Found {summary['total_incomplete_uploads']} incomplete multipart uploads across {summary['buckets_with_issues']} buckets")
            
            if summary["potential_monthly_savings"] > 0:
                insights.append(f"Potential monthly savings of ${summary['potential_monthly_savings']:.2f} from cleanup")
            
            if summary["lifecycle_policy_gaps"] > 0:
                insights.append(f"{summary['lifecycle_policy_gaps']} buckets need lifecycle policy updates for automatic cleanup")
            
            if not summary["storage_lens_available"]:
                insights.append("Enable Storage Lens Cost Optimization Metrics for better multipart upload tracking")
            
            summary["key_insights"] = insights
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Optimization summary creation error: {str(e)}")
            return {"error": str(e)}
    
    def _process_storage_cost_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer data to estimate multipart upload waste.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed cost waste data
        """
        try:
            processed_data = {
                "total_s3_cost": 0,
                "estimated_multipart_waste": 0,
                "cost_breakdown": {}
            }
            
            # Extract S3 storage costs from Cost Explorer data
            results_by_time = cost_data.get("ResultsByTime", [])
            
            for time_period in results_by_time:
                groups = time_period.get("Groups", [])
                for group in groups:
                    usage_type = group.get("Keys", [""])[0]
                    metrics = group.get("Metrics", {})
                    cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                    
                    processed_data["total_s3_cost"] += cost
                    
                    # Estimate multipart waste (conservative 1-5% of storage costs)
                    if "Storage" in usage_type:
                        processed_data["estimated_multipart_waste"] += cost * 0.02  # 2% estimate
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing storage cost data: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_multipart_waste_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide fallback cost estimation when other data sources are unavailable.
        
        Args:
            context: Analysis context
            
        Returns:
            Fallback cost estimation
        """
        return {
            "estimated_monthly_waste": 10.0,  # Conservative estimate
            "note": "Fallback estimation used - enable Storage Lens or Cost Explorer for accurate data",
            "recommendation": "Enable Storage Lens Cost Optimization Metrics for precise multipart upload tracking"
        }
    
    def _calculate_cleanup_priority(self, upload_count: int, estimated_cost: float, oldest_upload_days: int) -> str:
        """
        Calculate cleanup priority based on upload metrics.
        
        Args:
            upload_count: Number of incomplete uploads
            estimated_cost: Estimated monthly cost
            oldest_upload_days: Age of oldest upload in days
            
        Returns:
            Priority level (high, medium, low)
        """
        score = 0
        
        # Upload count scoring
        if upload_count > 100:
            score += 3
        elif upload_count > 10:
            score += 2
        elif upload_count > 0:
            score += 1
        
        # Cost scoring
        if estimated_cost > 10:
            score += 3
        elif estimated_cost > 1:
            score += 2
        elif estimated_cost > 0:
            score += 1
        
        # Age scoring
        if oldest_upload_days > 90:
            score += 3
        elif oldest_upload_days > 30:
            score += 2
        elif oldest_upload_days > 7:
            score += 1
        
        # Determine priority
        if score >= 7:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"
    
    def _generate_bucket_cleanup_actions(self, bucket_data: Dict[str, Any]) -> List[str]:
        """
        Generate specific cleanup actions for a bucket.
        
        Args:
            bucket_data: Bucket analysis data
            
        Returns:
            List of cleanup action items
        """
        actions = []
        
        upload_count = bucket_data.get("upload_count", 0)
        bucket_name = bucket_data.get("bucket_name", "")
        
        if upload_count > 0:
            actions.append(f"Review and abort {upload_count} incomplete multipart uploads in {bucket_name}")
            
            if upload_count > 50:
                actions.append("Consider scripted cleanup for large number of incomplete uploads")
            
            actions.append("Implement lifecycle policy with AbortIncompleteMultipartUpload rule")
            actions.append("Monitor for new incomplete uploads after cleanup")
        
        return actions
    
    def _analyze_policy_effectiveness(self, lifecycle_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of existing lifecycle policies.
        
        Args:
            lifecycle_analysis: Lifecycle policy analysis data
            
        Returns:
            Policy effectiveness analysis
        """
        try:
            buckets_with_rules = lifecycle_analysis.get("buckets_with_multipart_rules", [])
            buckets_without_rules = lifecycle_analysis.get("buckets_without_multipart_rules", [])
            
            total_buckets = len(buckets_with_rules) + len(buckets_without_rules)
            coverage_percentage = (len(buckets_with_rules) / total_buckets * 100) if total_buckets > 0 else 0
            
            effectiveness = {
                "policy_coverage_percentage": coverage_percentage,
                "buckets_with_policies": len(buckets_with_rules),
                "buckets_without_policies": len(buckets_without_rules),
                "total_buckets_analyzed": total_buckets,
                "effectiveness_rating": "high" if coverage_percentage >= 80 else "medium" if coverage_percentage >= 50 else "low"
            }
            
            # Analyze rule configurations
            rule_analysis = {}
            for bucket_info in buckets_with_rules:
                rule_details = bucket_info.get("rule_details", {})
                days_after_initiation = rule_details.get("days_after_initiation", 0)
                
                if days_after_initiation <= 7:
                    rule_analysis["optimal_rules"] = rule_analysis.get("optimal_rules", 0) + 1
                elif days_after_initiation <= 30:
                    rule_analysis["acceptable_rules"] = rule_analysis.get("acceptable_rules", 0) + 1
                else:
                    rule_analysis["suboptimal_rules"] = rule_analysis.get("suboptimal_rules", 0) + 1
            
            effectiveness["rule_configuration_analysis"] = rule_analysis
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Policy effectiveness analysis error: {str(e)}")
            return {"error": str(e)}
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            
            data = analysis_results.get("data", {})
            cleanup_recs = data.get("cleanup_recommendations", {})
            lifecycle_suggestions = data.get("lifecycle_policy_suggestions", {})
            optimization_summary = data.get("optimization_summary", {})
            
            # High priority cleanup recommendations
            for cleanup_rec in cleanup_recs.get("immediate_cleanup", []):
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="high",
                    title=f"Clean up incomplete multipart uploads in {cleanup_rec['bucket_name']}",
                    description=f"Found {cleanup_rec['upload_count']} incomplete multipart uploads. "
                               f"Oldest upload is {cleanup_rec['oldest_upload_days']} days old.",
                    potential_savings=cleanup_rec.get("estimated_monthly_savings", 0),
                    implementation_effort=cleanup_rec.get("implementation_effort", "low"),
                    affected_resources=[cleanup_rec["bucket_name"]],
                    action_items=cleanup_rec.get("action_items", [])
                ))
            
            # Lifecycle policy recommendations
            for policy_update in lifecycle_suggestions.get("policy_updates", []):
                recommendations.append(self.create_recommendation(
                    rec_type="governance",
                    priority="medium",
                    title=f"Add multipart cleanup rule to {policy_update['bucket_name']}",
                    description=policy_update.get("description", ""),
                    implementation_effort=policy_update.get("implementation_effort", "low"),
                    affected_resources=[policy_update["bucket_name"]],
                    action_items=[
                        "Update existing lifecycle policy",
                        f"Add AbortIncompleteMultipartUpload rule with {policy_update.get('recommended_days', 7)} days"
                    ]
                ))
            
            for new_policy in lifecycle_suggestions.get("new_policies", []):
                recommendations.append(self.create_recommendation(
                    rec_type="governance",
                    priority="medium",
                    title=f"Create lifecycle policy for {new_policy['bucket_name']}",
                    description=new_policy.get("description", ""),
                    implementation_effort=new_policy.get("implementation_effort", "low"),
                    affected_resources=[new_policy["bucket_name"]],
                    action_items=[
                        "Create new lifecycle policy",
                        f"Include AbortIncompleteMultipartUpload rule with {new_policy.get('recommended_days', 7)} days"
                    ]
                ))
            
            # Storage Lens configuration recommendation
            if not optimization_summary.get("storage_lens_available", False):
                recommendations.append(self.create_recommendation(
                    rec_type="configuration",
                    priority="medium",
                    title="Enable Storage Lens Cost Optimization Metrics",
                    description="Enable Cost Optimization Metrics in Storage Lens configuration to track incomplete multipart uploads accurately.",
                    implementation_effort="low",
                    action_items=[
                        "Access S3 Storage Lens console",
                        "Edit default configuration",
                        "Enable Cost Optimization Metrics",
                        "Enable Advanced Cost Optimization Metrics for detailed tracking"
                    ]
                ))
            
            # Overall optimization recommendation
            total_savings = cleanup_recs.get("cost_impact", {}).get("total_monthly_savings", 0)
            if total_savings > 0:
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="high",
                    title="Implement comprehensive multipart upload cleanup strategy",
                    description=f"Total potential monthly savings of ${total_savings:.2f} from cleaning up incomplete multipart uploads across {optimization_summary.get('buckets_with_issues', 0)} buckets.",
                    potential_savings=total_savings,
                    implementation_effort="medium",
                    action_items=[
                        "Prioritize high-impact buckets for immediate cleanup",
                        "Implement automated lifecycle policies",
                        "Set up monitoring for future incomplete uploads",
                        "Train development teams on proper multipart upload handling"
                    ]
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return [self.create_recommendation(
                rec_type="error_resolution",
                priority="high",
                title="Multipart Cleanup Analysis Error",
                description=f"Failed to generate recommendations: {str(e)}",
                action_items=["Review analyzer logs", "Check AWS permissions", "Verify service availability"]
            )]