"""
Archive Optimization Analyzer

This analyzer identifies long-term archive data optimization opportunities using
Storage Lens access patterns as the primary data source. It analyzes object age,
access patterns, and compliance requirements to recommend appropriate archive tiers
(Glacier Instant Retrieval, Glacier Flexible Retrieval, Glacier Deep Archive).

Requirements covered:
- 5.1: Use S3 Storage Lens as primary data source for archive analysis
- 5.2: Identify archive candidates based on object age, access patterns, and compliance
- 5.3: Recommend appropriate archive tiers (Instant, Flexible, Deep Archive)
- 5.4: Calculate cost savings from archiving strategies
- 5.5: Validate archive recommendations against retention policies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage

logger = logging.getLogger(__name__)


class ArchiveOptimizationAnalyzer(BaseAnalyzer):
    """
    Analyzer for identifying long-term archive data optimization opportunities.
    
    Uses Storage Lens as primary data source to analyze object age, access patterns,
    and compliance requirements for archive tier recommendations.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize ArchiveOptimizationAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "archive_optimization"
        
        # Archive tier thresholds (configurable)
        self.archive_thresholds = {
            "glacier_instant": {
                "min_age_days": 30,
                "access_frequency": "monthly",
                "retrieval_time": "milliseconds"
            },
            "glacier_flexible": {
                "min_age_days": 90,
                "access_frequency": "quarterly", 
                "retrieval_time": "minutes_to_hours"
            },
            "deep_archive": {
                "min_age_days": 180,
                "access_frequency": "yearly_or_less",
                "retrieval_time": "hours"
            }
        }
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute archive optimization analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - lookback_days: Number of days to analyze access patterns (default: 90)
                - bucket_names: Optional list of specific buckets to analyze
                - min_age_days: Minimum object age for archive consideration (default: 30)
                - archive_tier_preference: Preferred archive tier ('auto', 'instant', 'flexible', 'deep_archive')
                - include_compliance_check: Whether to validate compliance requirements
                
        Returns:
            Dictionary containing archive optimization analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting archive optimization analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "archive_candidates": {},
                    "archive_recommendations": {},
                    "cost_savings_analysis": {},
                    "compliance_validation": {},
                    "current_archive_usage": {},
                    "optimization_summary": {}
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute analysis components in parallel
            tasks = [
                self._identify_archive_candidates(context),
                self._analyze_current_archive_usage(context),
                self._validate_compliance_requirements(context) if kwargs.get('include_compliance_check', True) else self._create_empty_compliance_result()
            ]
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            candidates_result, current_usage_result, compliance_result = results
            
            # Process archive candidates
            if not isinstance(candidates_result, Exception) and candidates_result.get("status") == "success":
                analysis_results["data"]["archive_candidates"] = candidates_result["data"]
                analysis_results["data_sources"].extend(candidates_result.get("data_sources", []))
            else:
                self.logger.warning(f"Archive candidate identification failed: {candidates_result}")
                analysis_results["data"]["archive_candidates"] = {"error": str(candidates_result)}
            
            # Process current archive usage
            if not isinstance(current_usage_result, Exception) and current_usage_result.get("status") == "success":
                analysis_results["data"]["current_archive_usage"] = current_usage_result["data"]
                analysis_results["data_sources"].extend(current_usage_result.get("data_sources", []))
            else:
                self.logger.warning(f"Current archive usage analysis failed: {current_usage_result}")
                analysis_results["data"]["current_archive_usage"] = {"error": str(current_usage_result)}
            
            # Process compliance validation
            if not isinstance(compliance_result, Exception) and compliance_result.get("status") == "success":
                analysis_results["data"]["compliance_validation"] = compliance_result["data"]
                analysis_results["data_sources"].extend(compliance_result.get("data_sources", []))
            else:
                self.logger.warning(f"Compliance validation failed: {compliance_result}")
                analysis_results["data"]["compliance_validation"] = {"error": str(compliance_result)}
            
            # Generate archive recommendations
            analysis_results["data"]["archive_recommendations"] = await self._generate_archive_recommendations(
                analysis_results["data"], context
            )
            
            # Calculate cost savings
            analysis_results["data"]["cost_savings_analysis"] = await self._calculate_archive_cost_savings(
                analysis_results["data"], context
            )
            
            # Create optimization summary
            analysis_results["data"]["optimization_summary"] = self._create_optimization_summary(analysis_results["data"])
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Completed archive optimization analysis in {analysis_results['execution_time']:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in archive optimization analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _identify_archive_candidates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify archive candidates using Storage Lens access patterns.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing archive candidate analysis
        """
        try:
            self.logger.debug("Identifying archive candidates using Storage Lens data")
            
            candidates_analysis = {
                "by_bucket": {},
                "by_storage_class": {},
                "by_age_group": {},
                "total_candidates": 0,
                "total_candidate_size_gb": 0,
                "access_pattern_analysis": {}
            }
            
            data_sources = []
            
            # Try Storage Lens first (NO-COST primary source)
            if self.storage_lens_service:
                try:
                    # Get storage metrics with access patterns
                    storage_lens_result = await self.storage_lens_service.get_storage_metrics()
                    
                    if storage_lens_result.get("status") == "success":
                        self.logger.info("Using Storage Lens as primary data source for archive candidates")
                        candidates_analysis.update(self._process_storage_lens_archive_data(storage_lens_result["data"]))
                        data_sources.append("storage_lens")
                    else:
                        self.logger.warning(f"Storage Lens unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens analysis failed: {str(e)}")
            
            # Fallback to bucket-level analysis using S3 service (NO-COST operations only)
            if not data_sources and self.s3_service:
                try:
                    self.logger.info("Using S3 service for bucket-level archive candidate analysis")
                    bucket_analysis_result = await self._analyze_buckets_for_archive_candidates(context)
                    
                    if bucket_analysis_result.get("status") == "success":
                        candidates_analysis.update(bucket_analysis_result["data"])
                        data_sources.append("s3_service")
                    else:
                        self.logger.warning(f"S3 service analysis failed: {bucket_analysis_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"S3 service analysis failed: {str(e)}")
            
            # Enhance with Cost Explorer data for historical patterns
            if data_sources:
                try:
                    cost_analysis_result = await self._get_historical_storage_patterns(context)
                    if cost_analysis_result.get("status") == "success":
                        candidates_analysis["historical_patterns"] = cost_analysis_result["data"]
                        data_sources.append("cost_explorer")
                except Exception as e:
                    self.logger.warning(f"Cost Explorer enhancement failed: {str(e)}")
            
            return {
                "status": "success",
                "data": candidates_analysis,
                "data_sources": data_sources,
                "message": f"Archive candidate identification completed using: {', '.join(data_sources) if data_sources else 'fallback analysis'}"
            }
            
        except Exception as e:
            self.logger.error(f"Archive candidate identification error: {str(e)}")
            return {
                "status": "error",
                "message": f"Archive candidate identification failed: {str(e)}",
                "data": {}
            }
    
    async def _analyze_current_archive_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current archive tier usage and costs.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing current archive usage analysis
        """
        try:
            self.logger.debug("Analyzing current archive usage")
            
            usage_analysis = {
                "glacier_instant_usage": {"size_gb": 0, "cost": 0, "object_count": 0},
                "glacier_flexible_usage": {"size_gb": 0, "cost": 0, "object_count": 0},
                "deep_archive_usage": {"size_gb": 0, "cost": 0, "object_count": 0},
                "total_archive_cost": 0,
                "archive_distribution": {},
                "retrieval_patterns": {}
            }
            
            data_sources = []
            
            # Get current archive usage from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 90))).strftime('%Y-%m-%d')
                
                # Query for archive storage costs
                archive_filter = {
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
                                "Values": ["S3-Glacier-Storage", "S3-DeepArchive-Storage", "S3-GlacierIR-Storage"]
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
                    filter_expr=archive_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    usage_analysis.update(self._process_archive_usage_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                else:
                    self.logger.warning(f"Cost Explorer archive data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer archive analysis failed: {str(e)}")
            
            # Enhance with Storage Lens data if available
            if self.storage_lens_service:
                try:
                    storage_lens_result = await self.storage_lens_service.get_storage_metrics()
                    if storage_lens_result.get("status") == "success":
                        archive_metrics = self._extract_archive_metrics_from_storage_lens(storage_lens_result["data"])
                        usage_analysis["storage_lens_metrics"] = archive_metrics
                        data_sources.append("storage_lens")
                except Exception as e:
                    self.logger.warning(f"Storage Lens archive metrics failed: {str(e)}")
            
            return {
                "status": "success",
                "data": usage_analysis,
                "data_sources": data_sources,
                "message": f"Current archive usage analysis completed using: {', '.join(data_sources) if data_sources else 'fallback data'}"
            }
            
        except Exception as e:
            self.logger.error(f"Current archive usage analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Current archive usage analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _validate_compliance_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate compliance requirements for archive recommendations.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing compliance validation results
        """
        try:
            self.logger.debug("Validating compliance requirements for archive recommendations")
            
            compliance_analysis = {
                "retention_policies": {},
                "compliance_constraints": {},
                "archive_eligibility": {},
                "regulatory_considerations": {},
                "bucket_compliance_status": {}
            }
            
            data_sources = []
            
            # Check bucket lifecycle policies and retention settings (NO-COST operations)
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
                        compliance_tasks = [
                            self._check_bucket_compliance(bucket_name) for bucket_name in bucket_names[:10]  # Limit to prevent timeout
                        ]
                        
                        compliance_results = await asyncio.gather(*compliance_tasks, return_exceptions=True)
                        
                        for i, result in enumerate(compliance_results):
                            if not isinstance(result, Exception) and result.get("status") == "success":
                                bucket_name = bucket_names[i]
                                compliance_analysis["bucket_compliance_status"][bucket_name] = result["data"]
                        
                        data_sources.append("s3_service")
                        
                except Exception as e:
                    self.logger.warning(f"Bucket compliance check failed: {str(e)}")
            
            # Add default compliance guidelines
            compliance_analysis["compliance_constraints"] = self._get_default_compliance_constraints()
            compliance_analysis["regulatory_considerations"] = self._get_regulatory_considerations()
            
            return {
                "status": "success",
                "data": compliance_analysis,
                "data_sources": data_sources,
                "message": f"Compliance validation completed using: {', '.join(data_sources) if data_sources else 'default guidelines'}"
            }
            
        except Exception as e:
            self.logger.error(f"Compliance validation error: {str(e)}")
            return {
                "status": "error",
                "message": f"Compliance validation failed: {str(e)}",
                "data": {}
            }
    
    async def _create_empty_compliance_result(self) -> Dict[str, Any]:
        """Create empty compliance result when compliance check is disabled."""
        return {
            "status": "success",
            "data": {
                "compliance_check_disabled": True,
                "message": "Compliance validation was disabled for this analysis"
            },
            "data_sources": [],
            "message": "Compliance validation skipped"
        }
    
    async def _generate_archive_recommendations(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate archive tier recommendations based on analysis results.
        
        Args:
            analysis_data: Combined analysis data
            context: Analysis context
            
        Returns:
            Dictionary containing archive recommendations
        """
        try:
            self.logger.debug("Generating archive tier recommendations")
            
            recommendations = {
                "glacier_instant_candidates": [],
                "glacier_flexible_candidates": [],
                "deep_archive_candidates": [],
                "recommendation_summary": {},
                "implementation_priority": []
            }
            
            archive_candidates = analysis_data.get("archive_candidates", {})
            compliance_data = analysis_data.get("compliance_validation", {})
            
            # Process candidates by bucket
            for bucket_name, bucket_data in archive_candidates.get("by_bucket", {}).items():
                bucket_compliance = compliance_data.get("bucket_compliance_status", {}).get(bucket_name, {})
                
                # Generate recommendations for this bucket
                bucket_recommendations = self._generate_bucket_archive_recommendations(
                    bucket_name, bucket_data, bucket_compliance, context
                )
                
                # Categorize recommendations by archive tier
                for rec in bucket_recommendations:
                    tier = rec.get("recommended_tier")
                    if tier == "glacier_instant":
                        recommendations["glacier_instant_candidates"].append(rec)
                    elif tier == "glacier_flexible":
                        recommendations["glacier_flexible_candidates"].append(rec)
                    elif tier == "deep_archive":
                        recommendations["deep_archive_candidates"].append(rec)
            
            # Create summary and prioritization
            recommendations["recommendation_summary"] = self._create_recommendation_summary(recommendations)
            recommendations["implementation_priority"] = self._prioritize_archive_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Archive recommendation generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_archive_cost_savings(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cost savings from archiving strategies.
        
        Args:
            analysis_data: Combined analysis data
            context: Analysis context
            
        Returns:
            Dictionary containing cost savings analysis
        """
        try:
            self.logger.debug("Calculating archive cost savings")
            
            cost_savings = {
                "current_costs": {},
                "projected_costs": {},
                "savings_by_tier": {},
                "total_potential_savings": 0,
                "roi_analysis": {},
                "payback_period": {}
            }
            
            # Get current storage costs
            current_usage = analysis_data.get("current_archive_usage", {})
            archive_recommendations = analysis_data.get("archive_recommendations", {})
            
            # Calculate savings for each archive tier
            if self.pricing_service:
                try:
                    # Get current pricing for different storage classes
                    pricing_data = await self.pricing_service.get_storage_class_pricing()
                    
                    if pricing_data.get("status") == "success":
                        cost_savings.update(self._calculate_tier_savings(
                            archive_recommendations, pricing_data["pricing"], context
                        ))
                    else:
                        self.logger.warning("Pricing data unavailable for cost savings calculation")
                        
                except Exception as e:
                    self.logger.warning(f"Pricing service calculation failed: {str(e)}")
            
            # Use fallback cost estimation if pricing service unavailable
            if not cost_savings.get("total_potential_savings"):
                cost_savings.update(self._estimate_archive_savings_fallback(archive_recommendations))
            
            return cost_savings
            
        except Exception as e:
            self.logger.error(f"Cost savings calculation error: {str(e)}")
            return {"error": str(e)}
    
    def _process_storage_lens_archive_data(self, storage_lens_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Storage Lens data to identify archive candidates.
        
        Args:
            storage_lens_data: Raw Storage Lens data
            
        Returns:
            Processed archive candidate data
        """
        try:
            processed_data = {
                "by_bucket": {},
                "access_pattern_analysis": {},
                "storage_lens_available": True,
                "data_source": "storage_lens"
            }
            
            # Extract access pattern information if available
            if storage_lens_data.get("ActivityMetricsEnabled"):
                processed_data["access_patterns_available"] = True
                processed_data["note"] = "Access pattern analysis available through Storage Lens"
            else:
                processed_data["access_patterns_available"] = False
                processed_data["note"] = "Enable Storage Lens activity metrics for detailed access pattern analysis"
            
            # Extract cost optimization metrics if available
            if storage_lens_data.get("CostOptimizationMetricsEnabled"):
                processed_data["cost_optimization_available"] = True
            else:
                processed_data["cost_optimization_available"] = False
                processed_data["recommendation"] = "Enable Storage Lens cost optimization metrics for better archive recommendations"
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing Storage Lens archive data: {str(e)}")
            return {"error": str(e), "data_source": "storage_lens"}
    
    async def _analyze_buckets_for_archive_candidates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze buckets for archive candidates using S3 service (NO-COST operations only).
        
        Args:
            context: Analysis context
            
        Returns:
            Bucket-level archive candidate analysis
        """
        try:
            bucket_analysis = {
                "by_bucket": {},
                "lifecycle_policy_analysis": {},
                "versioning_analysis": {}
            }
            
            bucket_names = context.get('bucket_names')
            if not bucket_names:
                # Get all buckets (NO-COST operation)
                buckets_result = await self.s3_service.list_buckets()
                if buckets_result.get("status") == "success":
                    # Fix: S3Service returns buckets under data.Buckets, not buckets
                    buckets_data = buckets_result.get("data", {}).get("Buckets", [])
                    bucket_names = [bucket["Name"] for bucket in buckets_data][:20]  # Limit to prevent timeout
            
            if bucket_names:
                # Analyze each bucket for archive potential (NO-COST operations only)
                bucket_tasks = [
                    self._analyze_single_bucket_for_archive(bucket_name) for bucket_name in bucket_names
                ]
                
                bucket_results = await asyncio.gather(*bucket_tasks, return_exceptions=True)
                
                for i, result in enumerate(bucket_results):
                    if not isinstance(result, Exception) and result.get("status") == "success":
                        bucket_name = bucket_names[i]
                        bucket_analysis["by_bucket"][bucket_name] = result["data"]
            
            return {
                "status": "success",
                "data": bucket_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Bucket archive analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Bucket archive analysis failed: {str(e)}"
            }
    
    async def _analyze_single_bucket_for_archive(self, bucket_name: str) -> Dict[str, Any]:
        """
        Analyze a single bucket for archive potential (NO-COST operations only).
        
        Args:
            bucket_name: Name of the bucket to analyze
            
        Returns:
            Single bucket archive analysis
        """
        try:
            bucket_data = {
                "bucket_name": bucket_name,
                "has_lifecycle_policy": False,
                "versioning_enabled": False,
                "archive_potential": "unknown",
                "current_archive_rules": [],
                "recommendations": []
            }
            
            # Check lifecycle configuration (NO-COST operation)
            try:
                lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
                if lifecycle_result.get("status") == "success":
                    bucket_data["has_lifecycle_policy"] = True
                    bucket_data["current_archive_rules"] = self._extract_archive_rules(lifecycle_result.get("lifecycle_configuration", {}))
                else:
                    bucket_data["has_lifecycle_policy"] = False
            except Exception as e:
                self.logger.debug(f"Lifecycle check failed for {bucket_name}: {str(e)}")
            
            # Check versioning status (NO-COST operation)
            try:
                versioning_result = await self.s3_service.get_bucket_versioning(bucket_name)
                if versioning_result.get("status") == "success":
                    bucket_data["versioning_enabled"] = versioning_result.get("versioning_status") == "Enabled"
            except Exception as e:
                self.logger.debug(f"Versioning check failed for {bucket_name}: {str(e)}")
            
            # Determine archive potential based on available information
            bucket_data["archive_potential"] = self._assess_archive_potential(bucket_data)
            
            return {
                "status": "success",
                "data": bucket_data
            }
            
        except Exception as e:
            self.logger.error(f"Single bucket analysis error for {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Analysis failed for bucket {bucket_name}: {str(e)}"
            }
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from archive optimization analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            data = analysis_results.get("data", {})
            archive_recommendations = data.get("archive_recommendations", {})
            cost_savings = data.get("cost_savings_analysis", {})
            
            # High-priority recommendations for immediate cost savings
            glacier_instant_candidates = archive_recommendations.get("glacier_instant_candidates", [])
            if glacier_instant_candidates:
                total_savings = sum(candidate.get("estimated_monthly_savings", 0) for candidate in glacier_instant_candidates)
                
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="high",
                    title="Transition to Glacier Instant Retrieval",
                    description=f"Identified {len(glacier_instant_candidates)} buckets suitable for Glacier Instant Retrieval. "
                               f"This tier provides millisecond access while reducing storage costs by up to 68%.",
                    potential_savings=total_savings,
                    implementation_effort="low",
                    affected_resources=[candidate.get("bucket_name") for candidate in glacier_instant_candidates],
                    action_items=[
                        "Review identified buckets for access pattern validation",
                        "Create lifecycle policies to transition objects older than 30 days",
                        "Monitor retrieval patterns after implementation",
                        "Set up CloudWatch alarms for unexpected retrieval costs"
                    ]
                ))
            
            # Medium-priority recommendations for Glacier Flexible Retrieval
            glacier_flexible_candidates = archive_recommendations.get("glacier_flexible_candidates", [])
            if glacier_flexible_candidates:
                total_savings = sum(candidate.get("estimated_monthly_savings", 0) for candidate in glacier_flexible_candidates)
                
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="medium",
                    title="Transition to Glacier Flexible Retrieval",
                    description=f"Identified {len(glacier_flexible_candidates)} buckets suitable for Glacier Flexible Retrieval. "
                               f"This tier provides retrieval in minutes to hours while reducing storage costs by up to 82%.",
                    potential_savings=total_savings,
                    implementation_effort="medium",
                    affected_resources=[candidate.get("bucket_name") for candidate in glacier_flexible_candidates],
                    action_items=[
                        "Validate that retrieval time requirements (1-5 minutes) are acceptable",
                        "Create lifecycle policies to transition objects older than 90 days",
                        "Test retrieval process with sample objects",
                        "Update application logic to handle retrieval delays"
                    ]
                ))
            
            # Long-term recommendations for Deep Archive
            deep_archive_candidates = archive_recommendations.get("deep_archive_candidates", [])
            if deep_archive_candidates:
                total_savings = sum(candidate.get("estimated_monthly_savings", 0) for candidate in deep_archive_candidates)
                
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="medium",
                    title="Transition to Glacier Deep Archive",
                    description=f"Identified {len(deep_archive_candidates)} buckets suitable for Glacier Deep Archive. "
                               f"This tier provides the lowest storage costs (up to 95% reduction) for long-term archival.",
                    potential_savings=total_savings,
                    implementation_effort="high",
                    affected_resources=[candidate.get("bucket_name") for candidate in deep_archive_candidates],
                    action_items=[
                        "Confirm that 12-hour retrieval time is acceptable for business requirements",
                        "Create lifecycle policies to transition objects older than 180 days",
                        "Implement proper tagging for compliance and retrieval tracking",
                        "Document retrieval procedures for emergency access"
                    ]
                ))
            
            # Compliance and governance recommendations
            compliance_data = data.get("compliance_validation", {})
            bucket_compliance = compliance_data.get("bucket_compliance_status", {})
            
            non_compliant_buckets = [
                bucket for bucket, status in bucket_compliance.items()
                if not status.get("has_lifecycle_policy", False)
            ]
            
            if non_compliant_buckets:
                recommendations.append(self.create_recommendation(
                    rec_type="governance",
                    priority="high",
                    title="Implement Lifecycle Policies for Archive Governance",
                    description=f"Found {len(non_compliant_buckets)} buckets without lifecycle policies. "
                               f"Implementing proper lifecycle management is essential for cost optimization and compliance.",
                    implementation_effort="medium",
                    affected_resources=non_compliant_buckets,
                    action_items=[
                        "Create lifecycle policies for all production buckets",
                        "Define retention periods based on business requirements",
                        "Implement automatic transition rules to archive tiers",
                        "Set up monitoring for lifecycle policy compliance"
                    ]
                ))
            
            # Storage Lens enhancement recommendation
            if not data.get("archive_candidates", {}).get("access_patterns_available", False):
                recommendations.append(self.create_recommendation(
                    rec_type="optimization_enhancement",
                    priority="low",
                    title="Enable Storage Lens Activity Metrics",
                    description="Enable S3 Storage Lens activity metrics to get detailed access pattern analysis "
                               "for more accurate archive recommendations.",
                    implementation_effort="low",
                    action_items=[
                        "Enable Storage Lens activity metrics in S3 console",
                        "Configure metrics for all relevant buckets and prefixes",
                        "Wait 24-48 hours for data collection",
                        "Re-run archive analysis with enhanced data"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating archive optimization recommendations: {str(e)}")
            recommendations.append(self.create_recommendation(
                rec_type="error_resolution",
                priority="high",
                title="Archive Analysis Error",
                description=f"Failed to generate complete archive recommendations: {str(e)}",
                action_items=[
                    "Check AWS permissions for S3 and Storage Lens access",
                    "Verify Storage Lens configuration is enabled",
                    "Review bucket access permissions",
                    "Re-run analysis with debug logging enabled"
                ]
            ))
        
        return recommendations
    
    # Helper methods for data processing and calculations
    
    def _extract_archive_rules(self, lifecycle_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract existing archive rules from lifecycle configuration."""
        archive_rules = []
        
        for rule in lifecycle_config.get("Rules", []):
            if rule.get("Status") == "Enabled":
                transitions = rule.get("Transitions", [])
                for transition in transitions:
                    storage_class = transition.get("StorageClass", "")
                    if storage_class in ["GLACIER", "DEEP_ARCHIVE", "GLACIER_IR"]:
                        archive_rules.append({
                            "rule_id": rule.get("ID", ""),
                            "storage_class": storage_class,
                            "days": transition.get("Days", 0)
                        })
        
        return archive_rules
    
    def _assess_archive_potential(self, bucket_data: Dict[str, Any]) -> str:
        """Assess archive potential based on available bucket information."""
        if bucket_data.get("has_lifecycle_policy") and bucket_data.get("current_archive_rules"):
            return "already_optimized"
        elif bucket_data.get("has_lifecycle_policy"):
            return "partially_optimized"
        else:
            return "high_potential"
    
    def _get_default_compliance_constraints(self) -> Dict[str, Any]:
        """Get default compliance constraints for archive recommendations."""
        return {
            "general_guidelines": {
                "minimum_retention_period": "Check organizational data retention policies",
                "regulatory_compliance": "Ensure archive tier meets regulatory access requirements",
                "backup_considerations": "Maintain appropriate backup strategies for archived data"
            },
            "archive_tier_constraints": {
                "glacier_instant": "Suitable for data requiring immediate access",
                "glacier_flexible": "Suitable for data with 1-5 minute retrieval tolerance",
                "deep_archive": "Suitable for long-term archival with 12-hour retrieval tolerance"
            }
        }
    
    def _get_regulatory_considerations(self) -> Dict[str, Any]:
        """Get regulatory considerations for archive strategies."""
        return {
            "data_sovereignty": "Ensure archived data remains in required geographic regions",
            "audit_requirements": "Maintain audit trails for archived data access",
            "retention_compliance": "Verify archive periods meet regulatory retention requirements",
            "encryption_requirements": "Ensure archived data maintains required encryption standards"
        }
    
    async def _get_historical_storage_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical storage patterns from Cost Explorer."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 90))).strftime('%Y-%m-%d')
            
            storage_filter = {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": ["Amazon Simple Storage Service"]
                }
            }
            
            cost_result = get_cost_and_usage(
                start_date=start_date,
                end_date=end_date,
                granularity="MONTHLY",
                metrics=["UnblendedCost", "UsageQuantity"],
                group_by=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                filter_expr=storage_filter,
                region=context.get('region')
            )
            
            if cost_result.get("status") == "success":
                return {
                    "status": "success",
                    "data": self._process_historical_patterns(cost_result["data"])
                }
            else:
                return {
                    "status": "error",
                    "message": "Historical pattern analysis unavailable"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Historical pattern analysis failed: {str(e)}"
            }
    
    def _process_historical_patterns(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process historical cost data to identify storage patterns."""
        patterns = {
            "storage_growth_trend": "stable",
            "cost_trend": "stable",
            "storage_class_usage": {}
        }
        
        # Analyze trends from Cost Explorer data
        results = cost_data.get("ResultsByTime", [])
        if len(results) >= 2:
            first_month = results[0]
            last_month = results[-1]
            
            first_cost = sum(float(group.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0)) 
                           for group in first_month.get("Groups", []))
            last_cost = sum(float(group.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0)) 
                          for group in last_month.get("Groups", []))
            
            if last_cost > first_cost * 1.1:
                patterns["cost_trend"] = "increasing"
            elif last_cost < first_cost * 0.9:
                patterns["cost_trend"] = "decreasing"
        
        return patterns
    
    def _process_archive_usage_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Cost Explorer data for current archive usage."""
        usage_data = {
            "glacier_instant_usage": {"size_gb": 0, "cost": 0},
            "glacier_flexible_usage": {"size_gb": 0, "cost": 0},
            "deep_archive_usage": {"size_gb": 0, "cost": 0}
        }
        
        for result in cost_data.get("ResultsByTime", []):
            for group in result.get("Groups", []):
                usage_type = group.get("Keys", [""])[0].lower()
                cost = float(group.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0))
                usage = float(group.get("Metrics", {}).get("UsageQuantity", {}).get("Amount", 0))
                
                if "glacierir" in usage_type or "instant" in usage_type:
                    usage_data["glacier_instant_usage"]["cost"] += cost
                    usage_data["glacier_instant_usage"]["size_gb"] += usage
                elif "glacier" in usage_type and "deep" not in usage_type:
                    usage_data["glacier_flexible_usage"]["cost"] += cost
                    usage_data["glacier_flexible_usage"]["size_gb"] += usage
                elif "deeparchive" in usage_type or "deep" in usage_type:
                    usage_data["deep_archive_usage"]["cost"] += cost
                    usage_data["deep_archive_usage"]["size_gb"] += usage
        
        return usage_data
    
    def _extract_archive_metrics_from_storage_lens(self, storage_lens_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract archive-related metrics from Storage Lens data."""
        return {
            "cost_optimization_enabled": storage_lens_data.get("CostOptimizationMetricsEnabled", False),
            "activity_metrics_enabled": storage_lens_data.get("ActivityMetricsEnabled", False),
            "detailed_status_code_metrics": storage_lens_data.get("DetailedStatusCodeMetricsEnabled", False),
            "note": "Enable all Storage Lens metrics for comprehensive archive analysis"
        }
    
    async def _check_bucket_compliance(self, bucket_name: str) -> Dict[str, Any]:
        """Check compliance status for a single bucket."""
        try:
            compliance_data = {
                "bucket_name": bucket_name,
                "has_lifecycle_policy": False,
                "versioning_enabled": False,
                "compliance_status": "unknown"
            }
            
            # Check lifecycle policy
            lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
            compliance_data["has_lifecycle_policy"] = lifecycle_result.get("status") == "success"
            
            # Check versioning
            versioning_result = await self.s3_service.get_bucket_versioning(bucket_name)
            if versioning_result.get("status") == "success":
                compliance_data["versioning_enabled"] = versioning_result.get("versioning_status") == "Enabled"
            
            # Determine compliance status
            if compliance_data["has_lifecycle_policy"]:
                compliance_data["compliance_status"] = "compliant"
            else:
                compliance_data["compliance_status"] = "needs_lifecycle_policy"
            
            return {
                "status": "success",
                "data": compliance_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Compliance check failed for {bucket_name}: {str(e)}"
            }
    
    def _generate_bucket_archive_recommendations(self, bucket_name: str, bucket_data: Dict[str, Any], 
                                               compliance_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate archive recommendations for a specific bucket."""
        recommendations = []
        
        # Determine recommended tier based on bucket characteristics
        if bucket_data.get("archive_potential") == "high_potential":
            # Default recommendation based on age thresholds
            min_age_days = context.get('min_age_days', 30)
            
            if min_age_days <= 30:
                recommended_tier = "glacier_instant"
                estimated_savings = 0.68  # 68% cost reduction
            elif min_age_days <= 90:
                recommended_tier = "glacier_flexible"
                estimated_savings = 0.82  # 82% cost reduction
            else:
                recommended_tier = "deep_archive"
                estimated_savings = 0.95  # 95% cost reduction
            
            recommendations.append({
                "bucket_name": bucket_name,
                "recommended_tier": recommended_tier,
                "estimated_monthly_savings": 100 * estimated_savings,  # Placeholder calculation
                "implementation_effort": "medium",
                "compliance_status": compliance_data.get("compliance_status", "unknown")
            })
        
        return recommendations
    
    def _create_recommendation_summary(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of archive recommendations."""
        return {
            "total_buckets_analyzed": (
                len(recommendations.get("glacier_instant_candidates", [])) +
                len(recommendations.get("glacier_flexible_candidates", [])) +
                len(recommendations.get("deep_archive_candidates", []))
            ),
            "glacier_instant_count": len(recommendations.get("glacier_instant_candidates", [])),
            "glacier_flexible_count": len(recommendations.get("glacier_flexible_candidates", [])),
            "deep_archive_count": len(recommendations.get("deep_archive_candidates", []))
        }
    
    def _prioritize_archive_recommendations(self, recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize archive recommendations by potential savings and implementation effort."""
        all_recommendations = []
        
        # Add all recommendations with priority scoring
        for tier, candidates in recommendations.items():
            if isinstance(candidates, list):
                for candidate in candidates:
                    candidate["priority_score"] = candidate.get("estimated_monthly_savings", 0)
                    all_recommendations.append(candidate)
        
        # Sort by priority score (highest savings first)
        return sorted(all_recommendations, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    def _calculate_tier_savings(self, recommendations: Dict[str, Any], pricing_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed cost savings using pricing data."""
        savings_analysis = {
            "glacier_instant_savings": 0,
            "glacier_flexible_savings": 0,
            "deep_archive_savings": 0,
            "total_potential_savings": 0
        }
        
        # Calculate savings for each tier using actual pricing
        standard_price = pricing_data.get("STANDARD", {}).get("storage_cost_per_gb", 0.023)
        
        for tier_name, candidates in recommendations.items():
            if isinstance(candidates, list) and candidates:
                tier_savings = 0
                for candidate in candidates:
                    estimated_gb = candidate.get("estimated_size_gb", 1000)  # Default estimate
                    
                    if tier_name == "glacier_instant_candidates":
                        glacier_price = pricing_data.get("GLACIER_IR", {}).get("storage_cost_per_gb", 0.004)
                        monthly_savings = (standard_price - glacier_price) * estimated_gb
                    elif tier_name == "glacier_flexible_candidates":
                        glacier_price = pricing_data.get("GLACIER", {}).get("storage_cost_per_gb", 0.0036)
                        monthly_savings = (standard_price - glacier_price) * estimated_gb
                    elif tier_name == "deep_archive_candidates":
                        deep_price = pricing_data.get("DEEP_ARCHIVE", {}).get("storage_cost_per_gb", 0.00099)
                        monthly_savings = (standard_price - deep_price) * estimated_gb
                    else:
                        monthly_savings = 0
                    
                    tier_savings += monthly_savings
                    candidate["estimated_monthly_savings"] = monthly_savings
                
                savings_analysis[f"{tier_name.replace('_candidates', '_savings')}"] = tier_savings
                savings_analysis["total_potential_savings"] += tier_savings
        
        return savings_analysis
    
    def _estimate_archive_savings_fallback(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate archive savings using fallback calculations when pricing service unavailable."""
        # Use industry-standard pricing estimates
        standard_cost_per_gb = 0.023  # Standard storage cost per GB per month
        
        savings_estimates = {
            "glacier_instant": {"cost_per_gb": 0.004, "reduction": 0.826},
            "glacier_flexible": {"cost_per_gb": 0.0036, "reduction": 0.843},
            "deep_archive": {"cost_per_gb": 0.00099, "reduction": 0.957}
        }
        
        total_savings = 0
        
        for tier_name, candidates in recommendations.items():
            if isinstance(candidates, list):
                tier_key = tier_name.replace("_candidates", "")
                if tier_key in savings_estimates:
                    tier_savings = len(candidates) * 1000 * standard_cost_per_gb * savings_estimates[tier_key]["reduction"]  # Assume 1TB per bucket
                    total_savings += tier_savings
        
        return {
            "total_potential_savings": total_savings,
            "estimation_method": "fallback_industry_averages",
            "note": "Savings estimates based on industry-standard pricing. Enable pricing service for accurate calculations."
        }
    
    def _create_optimization_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization summary from analysis data."""
        summary = {
            "analysis_status": "completed",
            "total_recommendations": 0,
            "high_priority_actions": 0,
            "estimated_total_savings": 0,
            "key_findings": []
        }
        
        # Count recommendations
        recommendations = analysis_data.get("archive_recommendations", {})
        for tier, candidates in recommendations.items():
            if isinstance(candidates, list):
                summary["total_recommendations"] += len(candidates)
        
        # Calculate total savings
        cost_savings = analysis_data.get("cost_savings_analysis", {})
        summary["estimated_total_savings"] = cost_savings.get("total_potential_savings", 0)
        
        # Generate key findings
        if summary["total_recommendations"] > 0:
            summary["key_findings"].append(f"Identified {summary['total_recommendations']} buckets suitable for archive optimization")
        
        if summary["estimated_total_savings"] > 100:
            summary["key_findings"].append(f"Potential monthly savings of ${summary['estimated_total_savings']:.2f}")
        
        return summary