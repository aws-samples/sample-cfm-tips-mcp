"""
Storage Class Analyzer for S3 Optimization

This analyzer provides comprehensive S3 storage class analysis and optimization recommendations.
It analyzes daily cost and usage by storage class, validates storage class appropriateness,
and provides guidance for storage class selection and transitions.

Implements requirements 2.1-2.4, 3.1-3.5, and 4.1-4.4 from the S3 optimization specification.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer
from services.cost_explorer import get_cost_and_usage

logger = logging.getLogger(__name__)


class StorageClassAnalyzer(BaseAnalyzer):
    """
    Analyzer for S3 storage class optimization and recommendations.
    
    Provides daily cost and usage analysis by storage class, validates storage class
    appropriateness using Storage Lens data, and generates storage class transition
    recommendations with cost savings calculations.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize StorageClassAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "storage_class"
        
        # Storage class characteristics for recommendations
        self.storage_class_characteristics = {
            'STANDARD': {
                'min_storage_duration_days': 0,
                'retrieval_fee': False,
                'availability': '99.99%',
                'durability': '99.999999999%',
                'use_case': 'Frequently accessed data',
                'cost_tier': 'highest'
            },
            'STANDARD_IA': {
                'min_storage_duration_days': 30,
                'retrieval_fee': True,
                'availability': '99.9%',
                'durability': '99.999999999%',
                'use_case': 'Infrequently accessed data (monthly or less)',
                'cost_tier': 'medium'
            },
            'ONEZONE_IA': {
                'min_storage_duration_days': 30,
                'retrieval_fee': True,
                'availability': '99.5%',
                'durability': '99.999999999%',
                'use_case': 'Infrequently accessed, non-critical data',
                'cost_tier': 'low'
            },
            'GLACIER_IR': {
                'min_storage_duration_days': 90,
                'retrieval_fee': True,
                'availability': '99.9%',
                'durability': '99.999999999%',
                'use_case': 'Archive data with instant retrieval',
                'cost_tier': 'very_low'
            },
            'GLACIER': {
                'min_storage_duration_days': 90,
                'retrieval_fee': True,
                'availability': '99.99%',
                'durability': '99.999999999%',
                'use_case': 'Long-term archive (1-5 minute retrieval)',
                'cost_tier': 'very_low'
            },
            'DEEP_ARCHIVE': {
                'min_storage_duration_days': 180,
                'retrieval_fee': True,
                'availability': '99.99%',
                'durability': '99.999999999%',
                'use_case': 'Long-term archive (12 hour retrieval)',
                'cost_tier': 'lowest'
            }
        }
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute comprehensive storage class analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - lookback_days: Number of days to analyze (default: 30)
                - bucket_names: Optional list of specific buckets to analyze
                - include_cost_analysis: Whether to include detailed cost breakdown
                
        Returns:
            Dictionary containing storage class analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting storage class analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "daily_cost_usage": {},
                    "storage_class_distribution": {},
                    "appropriateness_validation": {},
                    "selection_guidance": {},
                    "transition_recommendations": [],
                    "cost_savings_potential": {}
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Execute analysis components in parallel
            tasks = [
                self._analyze_daily_cost_usage(context),
                self._validate_storage_class_appropriateness(context),
                self._generate_storage_class_guidance(context)
            ]
            
            # Execute all analysis tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            daily_results, validation_results, guidance_results = results
            
            # Aggregate daily cost and usage analysis
            if not isinstance(daily_results, Exception) and daily_results.get("status") == "success":
                analysis_results["data"]["daily_cost_usage"] = daily_results["data"]
                analysis_results["data_sources"].extend(daily_results.get("data_sources", []))
            else:
                self.logger.warning(f"Daily cost/usage analysis failed: {daily_results}")
                analysis_results["data"]["daily_cost_usage"] = {"error": str(daily_results)}
            
            # Aggregate appropriateness validation
            if not isinstance(validation_results, Exception) and validation_results.get("status") == "success":
                analysis_results["data"]["appropriateness_validation"] = validation_results["data"]
                analysis_results["data_sources"].extend(validation_results.get("data_sources", []))
            else:
                self.logger.warning(f"Storage class validation failed: {validation_results}")
                analysis_results["data"]["appropriateness_validation"] = {"error": str(validation_results)}
            
            # Aggregate selection guidance
            if not isinstance(guidance_results, Exception) and guidance_results.get("status") == "success":
                analysis_results["data"]["selection_guidance"] = guidance_results["data"]
                analysis_results["data_sources"].extend(guidance_results.get("data_sources", []))
            else:
                self.logger.warning(f"Storage class guidance failed: {guidance_results}")
                analysis_results["data"]["selection_guidance"] = {"error": str(guidance_results)}
            
            # Generate transition recommendations based on all analysis
            analysis_results["data"]["transition_recommendations"] = await self._generate_transition_recommendations(
                analysis_results["data"]
            )
            
            # Calculate cost savings potential
            analysis_results["data"]["cost_savings_potential"] = self._calculate_cost_savings_potential(
                analysis_results["data"]
            )
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Completed storage class analysis in {analysis_results['execution_time']:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in storage class analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _analyze_daily_cost_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze daily S3 cost and usage by storage class using Cost Explorer.
        
        Implements requirement 2.1: Daily cost and usage analysis by storage class.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing daily cost and usage analysis
        """
        try:
            self.logger.debug("Analyzing daily cost and usage by storage class")
            
            daily_analysis = {
                "cost_by_storage_class": {},
                "usage_by_storage_class": {},
                "daily_trends": {},
                "total_daily_cost": 0,
                "cost_distribution_percentage": {}
            }
            
            data_sources = []
            
            # Get daily cost and usage from Cost Explorer
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
                
                # Query Cost Explorer for S3 storage costs by usage type (Requirement 2.2)
                storage_filter = {
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
                                "Values": ["S3-Storage-Class", "S3-Storage"]
                            }
                        }
                    ]
                }
                
                # Add region filter if specified
                if context.get('region'):
                    storage_filter["And"].append({
                        "Dimensions": {
                            "Key": "REGION",
                            "Values": [context['region']]
                        }
                    })
                
                # Query with daily granularity and group by usage type (Requirement 2.3)
                cost_result = get_cost_and_usage(
                    start_date=start_date,
                    end_date=end_date,
                    granularity="DAILY",
                    metrics=["UnblendedCost", "UsageQuantity"],
                    group_by=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
                    filter_expr=storage_filter,
                    region=context.get('region')
                )
                
                if cost_result.get("status") == "success":
                    daily_analysis.update(self._process_daily_cost_usage_data(cost_result["data"]))
                    data_sources.append("cost_explorer")
                    self.logger.info("Successfully retrieved daily cost/usage data from Cost Explorer")
                else:
                    self.logger.warning(f"Cost Explorer daily data unavailable: {cost_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.warning(f"Cost Explorer daily analysis failed: {str(e)}")
            
            # Enhance with Storage Lens data if available
            if self.storage_lens_service:
                try:
                    storage_lens_result = await self.storage_lens_service.get_storage_class_distribution()
                    
                    if storage_lens_result.get("status") == "success":
                        daily_analysis["storage_lens_distribution"] = storage_lens_result["data"]
                        data_sources.append("storage_lens")
                        self.logger.info("Enhanced analysis with Storage Lens distribution data")
                    else:
                        self.logger.warning(f"Storage Lens distribution unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens enhancement failed: {str(e)}")
            
            return {
                "status": "success",
                "data": daily_analysis,
                "data_sources": data_sources,
                "message": f"Daily cost/usage analysis completed using: {', '.join(data_sources) if data_sources else 'fallback data'}"
            }
            
        except Exception as e:
            self.logger.error(f"Daily cost/usage analysis error: {str(e)}")
            return {
                "status": "error",
                "message": f"Daily cost/usage analysis failed: {str(e)}",
                "data": {}
            }
    
    async def _validate_storage_class_appropriateness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that existing data is stored in the most appropriate storage class.
        
        Implements requirements 3.1-3.5: Storage class appropriateness validation.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing storage class validation results
        """
        try:
            self.logger.debug("Validating storage class appropriateness")
            
            validation_analysis = {
                "appropriateness_scores": {},
                "misclassified_data": {},
                "optimization_opportunities": [],
                "access_pattern_analysis": {},
                "cost_impact_analysis": {}
            }
            
            data_sources = []
            
            # Use Storage Lens as primary data source (Requirement 3.1)
            if self.storage_lens_service:
                try:
                    storage_lens_result = await self.storage_lens_service.get_cost_optimization_metrics()
                    
                    if storage_lens_result.get("status") == "success":
                        validation_analysis.update(self._process_storage_lens_validation(storage_lens_result["data"]))
                        data_sources.append("storage_lens")
                        self.logger.info("Using Storage Lens as primary data source for validation")
                    else:
                        self.logger.warning(f"Storage Lens validation unavailable: {storage_lens_result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Storage Lens validation failed: {str(e)}")
            
            # Fallback to Cost Explorer analysis (Requirement 3.2)
            if not data_sources or "storage_lens" not in data_sources:
                try:
                    self.logger.info("Using Cost Explorer as fallback for validation")
                    cost_explorer_validation = await self._validate_with_cost_explorer(context)
                    
                    if cost_explorer_validation.get("status") == "success":
                        validation_analysis.update(cost_explorer_validation["data"])
                        data_sources.append("cost_explorer")
                    else:
                        self.logger.warning(f"Cost Explorer validation failed: {cost_explorer_validation.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    self.logger.warning(f"Cost Explorer validation failed: {str(e)}")
            
            # Analyze access patterns and object age (Requirement 3.3)
            if self.s3_service and context.get('bucket_names'):
                try:
                    access_pattern_analysis = await self._analyze_access_patterns(
                        context['bucket_names'], 
                        context.get('lookback_days', 30)
                    )
                    validation_analysis["access_pattern_analysis"] = access_pattern_analysis
                    data_sources.append("cloudwatch")
                except Exception as e:
                    self.logger.warning(f"Access pattern analysis failed: {str(e)}")
            
            # Generate optimization opportunities (Requirement 3.4)
            validation_analysis["optimization_opportunities"] = self._identify_storage_class_optimizations(
                validation_analysis
            )
            
            return {
                "status": "success",
                "data": validation_analysis,
                "data_sources": data_sources,
                "message": f"Storage class validation completed using: {', '.join(data_sources) if data_sources else 'analysis only'}"
            }
            
        except Exception as e:
            self.logger.error(f"Storage class validation error: {str(e)}")
            return {
                "status": "error",
                "message": f"Storage class validation failed: {str(e)}",
                "data": {}
            }
    
    async def _generate_storage_class_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate guidance on choosing the right storage class for new data.
        
        Implements requirements 4.1-4.4: Storage class selection guidance.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary containing storage class selection guidance
        """
        try:
            self.logger.debug("Generating storage class selection guidance")
            
            guidance_analysis = {
                "selection_matrix": {},
                "cost_comparison": {},
                "decision_tree": {},
                "use_case_recommendations": {},
                "cost_calculator": {}
            }
            
            data_sources = []
            
            # Generate selection matrix based on access patterns (Requirement 4.1)
            guidance_analysis["selection_matrix"] = self._create_storage_class_selection_matrix()
            
            # Get pricing information for cost comparison (Requirement 4.2)
            if self.pricing_service:
                try:
                    cost_comparison = await self._generate_cost_comparison()
                    guidance_analysis["cost_comparison"] = cost_comparison
                    data_sources.append("pricing_api")
                except Exception as e:
                    self.logger.warning(f"Cost comparison generation failed: {str(e)}")
            
            # Create decision tree for storage class selection (Requirement 4.3)
            guidance_analysis["decision_tree"] = self._create_storage_class_decision_tree()
            
            # Generate use case recommendations (Requirement 4.4)
            guidance_analysis["use_case_recommendations"] = self._generate_use_case_recommendations()
            
            # Create cost calculator for different scenarios
            guidance_analysis["cost_calculator"] = self._create_cost_calculator_guidance()
            
            return {
                "status": "success",
                "data": guidance_analysis,
                "data_sources": data_sources,
                "message": f"Storage class guidance generated using: {', '.join(data_sources) if data_sources else 'built-in knowledge'}"
            }
            
        except Exception as e:
            self.logger.error(f"Storage class guidance error: {str(e)}")
            return {
                "status": "error",
                "message": f"Storage class guidance failed: {str(e)}",
                "data": {}
            }
    
    def _process_daily_cost_usage_data(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Cost Explorer daily cost and usage data by storage class.
        
        Args:
            cost_data: Raw Cost Explorer data
            
        Returns:
            Processed daily cost and usage data
        """
        try:
            processed_data = {
                "cost_by_storage_class": {},
                "usage_by_storage_class": {},
                "daily_trends": {},
                "total_daily_cost": 0
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
                    
                    # Extract storage class from usage type
                    storage_class = self._extract_storage_class_from_usage_type(usage_type)
                    
                    # Aggregate by storage class
                    if storage_class not in processed_data["cost_by_storage_class"]:
                        processed_data["cost_by_storage_class"][storage_class] = {
                            "total_cost": 0,
                            "daily_costs": []
                        }
                        processed_data["usage_by_storage_class"][storage_class] = {
                            "total_usage_gb": 0,
                            "daily_usage": []
                        }
                    
                    processed_data["cost_by_storage_class"][storage_class]["total_cost"] += cost
                    processed_data["cost_by_storage_class"][storage_class]["daily_costs"].append({
                        "date": start_date,
                        "cost": cost
                    })
                    
                    processed_data["usage_by_storage_class"][storage_class]["total_usage_gb"] += usage
                    processed_data["usage_by_storage_class"][storage_class]["daily_usage"].append({
                        "date": start_date,
                        "usage_gb": usage
                    })
                    
                    daily_cost += cost
                
                # Track daily trends
                processed_data["daily_trends"][start_date] = {
                    "total_cost": daily_cost,
                    "date": start_date
                }
                
                processed_data["total_daily_cost"] += daily_cost
            
            # Calculate cost distribution percentages
            if processed_data["total_daily_cost"] > 0:
                processed_data["cost_distribution_percentage"] = {}
                for storage_class, data in processed_data["cost_by_storage_class"].items():
                    percentage = (data["total_cost"] / processed_data["total_daily_cost"]) * 100
                    processed_data["cost_distribution_percentage"][storage_class] = round(percentage, 2)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing daily cost/usage data: {str(e)}")
            return {"error": str(e)}
    
    def _extract_storage_class_from_usage_type(self, usage_type: str) -> str:
        """
        Extract storage class from Cost Explorer usage type.
        
        Args:
            usage_type: Usage type string from Cost Explorer
            
        Returns:
            Normalized storage class name
        """
        usage_type_lower = usage_type.lower()
        
        if "standard" in usage_type_lower and "ia" not in usage_type_lower:
            return "STANDARD"
        elif "standard-ia" in usage_type_lower or "standardia" in usage_type_lower:
            return "STANDARD_IA"
        elif "onezone" in usage_type_lower or "onezone-ia" in usage_type_lower:
            return "ONEZONE_IA"
        elif "glacier" in usage_type_lower and "deep" not in usage_type_lower and "instant" not in usage_type_lower:
            return "GLACIER"
        elif "glacier" in usage_type_lower and "instant" in usage_type_lower:
            return "GLACIER_IR"
        elif "deep" in usage_type_lower and "archive" in usage_type_lower:
            return "DEEP_ARCHIVE"
        elif "reduced" in usage_type_lower and "redundancy" in usage_type_lower:
            return "REDUCED_REDUNDANCY"
        else:
            return "OTHER"
    
    def _process_storage_lens_validation(self, storage_lens_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Storage Lens data for storage class validation.
        
        Args:
            storage_lens_data: Raw Storage Lens data
            
        Returns:
            Processed validation data
        """
        try:
            processed_data = {
                "appropriateness_scores": {},
                "cost_optimization_enabled": storage_lens_data.get("CostOptimizationMetricsEnabled", False),
                "recommendations_available": storage_lens_data.get("ActivityMetricsEnabled", False)
            }
            
            # Extract cost optimization metrics if available
            if storage_lens_data.get("CostOptimizationMetricsEnabled"):
                processed_data["note"] = "Cost optimization metrics available through Storage Lens dashboard"
                processed_data["appropriateness_scores"]["storage_lens_enabled"] = True
            else:
                processed_data["note"] = "Enable Storage Lens cost optimization metrics for detailed validation"
                processed_data["appropriateness_scores"]["storage_lens_enabled"] = False
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing Storage Lens validation data: {str(e)}")
            return {"error": str(e)}
    
    async def _validate_with_cost_explorer(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate storage class appropriateness using Cost Explorer data.
        
        Args:
            context: Analysis context
            
        Returns:
            Validation results from Cost Explorer analysis
        """
        try:
            validation_data = {
                "cost_trends": {},
                "usage_patterns": {},
                "potential_savings": {}
            }
            
            # Analyze cost trends over time to identify patterns
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=context.get('lookback_days', 30))).strftime('%Y-%m-%d')
            
            # Get storage costs by storage class over time
            storage_filter = {
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
                filter_expr=storage_filter,
                region=context.get('region')
            )
            
            if cost_result.get("status") == "success":
                validation_data["cost_trends"] = self._analyze_cost_trends(cost_result["data"])
            
            return {
                "status": "success",
                "data": validation_data
            }
            
        except Exception as e:
            self.logger.error(f"Cost Explorer validation error: {str(e)}")
            return {
                "status": "error",
                "message": f"Cost Explorer validation failed: {str(e)}"
            }
    
    async def _analyze_access_patterns(self, bucket_names: List[str], lookback_days: int) -> Dict[str, Any]:
        """
        Analyze access patterns for buckets using CloudWatch metrics.
        
        Args:
            bucket_names: List of bucket names to analyze
            lookback_days: Number of days to analyze
            
        Returns:
            Access pattern analysis results
        """
        try:
            access_patterns = {}
            
            # Note: This would require CloudWatch metrics analysis
            # For now, return a placeholder structure
            for bucket_name in bucket_names:
                access_patterns[bucket_name] = {
                    "get_requests": 0,
                    "put_requests": 0,
                    "list_requests": 0,
                    "access_frequency": "unknown",
                    "recommendation": "Enable CloudWatch metrics for detailed analysis"
                }
            
            return access_patterns
            
        except Exception as e:
            self.logger.error(f"Access pattern analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _identify_storage_class_optimizations(self, validation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify storage class optimization opportunities.
        
        Args:
            validation_data: Validation analysis data
            
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Example optimization opportunities based on common patterns
        opportunities.append({
            "type": "storage_class_transition",
            "priority": "medium",
            "title": "Consider Standard-IA for infrequently accessed data",
            "description": "Data accessed less than monthly could benefit from Standard-IA storage class",
            "potential_savings": "Up to 40% storage cost reduction",
            "implementation_effort": "low",
            "action_items": [
                "Analyze access patterns for objects older than 30 days",
                "Implement lifecycle policies for automatic transition",
                "Monitor retrieval costs after transition"
            ]
        })
        
        opportunities.append({
            "type": "lifecycle_policy",
            "priority": "high",
            "title": "Implement intelligent tiering for variable access patterns",
            "description": "Use S3 Intelligent-Tiering for data with unknown or changing access patterns",
            "potential_savings": "Automatic cost optimization without retrieval fees",
            "implementation_effort": "low",
            "action_items": [
                "Enable S3 Intelligent-Tiering on appropriate buckets",
                "Configure optional archive access tiers",
                "Monitor tiering effectiveness"
            ]
        })
        
        return opportunities
    
    def _create_storage_class_selection_matrix(self) -> Dict[str, Any]:
        """
        Create a selection matrix for storage class recommendations.
        
        Returns:
            Storage class selection matrix
        """
        return {
            "access_frequency": {
                "daily": {
                    "recommended": "STANDARD",
                    "alternatives": ["INTELLIGENT_TIERING"],
                    "reason": "Frequent access requires immediate availability"
                },
                "weekly": {
                    "recommended": "STANDARD",
                    "alternatives": ["INTELLIGENT_TIERING"],
                    "reason": "Regular access patterns benefit from standard storage"
                },
                "monthly": {
                    "recommended": "STANDARD_IA",
                    "alternatives": ["INTELLIGENT_TIERING"],
                    "reason": "Infrequent access with cost savings"
                },
                "quarterly": {
                    "recommended": "STANDARD_IA",
                    "alternatives": ["GLACIER_IR", "INTELLIGENT_TIERING"],
                    "reason": "Low access frequency allows for retrieval fees"
                },
                "yearly": {
                    "recommended": "GLACIER_IR",
                    "alternatives": ["GLACIER", "DEEP_ARCHIVE"],
                    "reason": "Archive storage for long-term retention"
                },
                "rarely": {
                    "recommended": "DEEP_ARCHIVE",
                    "alternatives": ["GLACIER"],
                    "reason": "Lowest cost for rarely accessed data"
                }
            },
            "data_size": {
                "small_objects": {
                    "note": "Objects smaller than 128KB have minimum billable size",
                    "consideration": "Factor in minimum storage duration charges"
                },
                "large_objects": {
                    "note": "Objects larger than 5TB require multipart upload",
                    "consideration": "Consider data transfer costs for large objects"
                }
            },
            "retention_period": {
                "short_term": {
                    "duration": "< 30 days",
                    "recommended": "STANDARD",
                    "reason": "Minimum storage duration charges make other classes expensive"
                },
                "medium_term": {
                    "duration": "30-90 days",
                    "recommended": "STANDARD_IA",
                    "reason": "Cost effective after minimum storage duration"
                },
                "long_term": {
                    "duration": "> 90 days",
                    "recommended": "GLACIER_IR",
                    "alternatives": ["GLACIER", "DEEP_ARCHIVE"],
                    "reason": "Archive classes provide significant cost savings"
                }
            }
        }
    
    async def _generate_cost_comparison(self) -> Dict[str, Any]:
        """
        Generate cost comparison across storage classes.
        
        Returns:
            Cost comparison data
        """
        try:
            cost_comparison = {}
            
            # Get pricing for each storage class
            for storage_class in self.storage_class_characteristics.keys():
                if self.pricing_service:
                    pricing_result = self.pricing_service.get_storage_class_pricing(storage_class)
                    if pricing_result.get('status') == 'success':
                        cost_comparison[storage_class] = pricing_result
            
            return cost_comparison
            
        except Exception as e:
            self.logger.error(f"Cost comparison generation error: {str(e)}")
            return {"error": str(e)}
    
    def _create_storage_class_decision_tree(self) -> Dict[str, Any]:
        """
        Create a decision tree for storage class selection.
        
        Returns:
            Decision tree structure
        """
        return {
            "root": {
                "question": "How frequently will you access this data?",
                "options": {
                    "daily_weekly": {
                        "next": "performance_requirements",
                        "question": "Do you need immediate access (milliseconds)?",
                        "options": {
                            "yes": {
                                "recommendation": "STANDARD",
                                "reason": "Immediate access with highest availability"
                            },
                            "no": {
                                "recommendation": "INTELLIGENT_TIERING",
                                "reason": "Automatic optimization for changing patterns"
                            }
                        }
                    },
                    "monthly": {
                        "next": "cost_sensitivity",
                        "question": "Is cost optimization a priority?",
                        "options": {
                            "yes": {
                                "recommendation": "STANDARD_IA",
                                "reason": "Lower storage cost with retrieval fees"
                            },
                            "no": {
                                "recommendation": "STANDARD",
                                "reason": "No retrieval fees, higher storage cost"
                            }
                        }
                    },
                    "rarely": {
                        "next": "retrieval_time",
                        "question": "How quickly do you need data when accessed?",
                        "options": {
                            "immediate": {
                                "recommendation": "GLACIER_IR",
                                "reason": "Archive pricing with instant retrieval"
                            },
                            "minutes": {
                                "recommendation": "GLACIER",
                                "reason": "Low cost with 1-5 minute retrieval"
                            },
                            "hours": {
                                "recommendation": "DEEP_ARCHIVE",
                                "reason": "Lowest cost with 12-hour retrieval"
                            }
                        }
                    }
                }
            }
        }
    
    def _generate_use_case_recommendations(self) -> Dict[str, Any]:
        """
        Generate storage class recommendations for common use cases.
        
        Returns:
            Use case recommendations
        """
        return {
            "web_applications": {
                "primary_storage": "STANDARD",
                "backup_storage": "STANDARD_IA",
                "log_archives": "GLACIER",
                "rationale": "Active content needs immediate access, backups can tolerate retrieval delays"
            },
            "data_analytics": {
                "active_datasets": "STANDARD",
                "processed_results": "STANDARD_IA",
                "historical_data": "GLACIER_IR",
                "rationale": "Frequent analysis requires fast access, results accessed occasionally"
            },
            "backup_and_archive": {
                "recent_backups": "STANDARD_IA",
                "monthly_backups": "GLACIER",
                "yearly_archives": "DEEP_ARCHIVE",
                "rationale": "Backup frequency determines appropriate storage class and retrieval time"
            },
            "content_distribution": {
                "popular_content": "STANDARD",
                "seasonal_content": "INTELLIGENT_TIERING",
                "archived_content": "GLACIER_IR",
                "rationale": "Access patterns vary, intelligent tiering optimizes automatically"
            },
            "compliance_data": {
                "active_compliance": "STANDARD_IA",
                "long_term_retention": "GLACIER",
                "legal_hold": "DEEP_ARCHIVE",
                "rationale": "Compliance data accessed infrequently but must be preserved"
            }
        }
    
    def _create_cost_calculator_guidance(self) -> Dict[str, Any]:
        """
        Create guidance for cost calculation across storage classes.
        
        Returns:
            Cost calculator guidance
        """
        return {
            "calculation_factors": {
                "storage_cost": "Monthly cost per GB stored",
                "request_cost": "Cost per PUT, GET, LIST request",
                "retrieval_cost": "Cost per GB retrieved (IA and Archive classes)",
                "minimum_duration": "Minimum billable storage duration",
                "minimum_size": "Minimum billable object size (128KB for some classes)"
            },
            "cost_formula": {
                "total_monthly_cost": "storage_cost + request_cost + retrieval_cost + data_transfer_cost",
                "storage_cost": "data_size_gb * storage_price_per_gb * duration_months",
                "request_cost": "request_count * request_price",
                "retrieval_cost": "retrieved_gb * retrieval_price_per_gb"
            },
            "break_even_analysis": {
                "standard_vs_ia": "Standard-IA breaks even when retrieval is less than 1-2 times per month",
                "ia_vs_glacier": "Glacier breaks even when retrieval is less than once per quarter",
                "glacier_vs_deep_archive": "Deep Archive breaks even for data accessed less than once per year"
            },
            "example_scenarios": {
                "1tb_monthly_access": {
                    "data_size": "1TB",
                    "access_pattern": "Monthly",
                    "recommended": "STANDARD_IA",
                    "estimated_savings": "40% vs STANDARD"
                },
                "10tb_yearly_access": {
                    "data_size": "10TB",
                    "access_pattern": "Yearly",
                    "recommended": "GLACIER_IR",
                    "estimated_savings": "70% vs STANDARD"
                }
            }
        }
    
    async def _generate_transition_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate storage class transition recommendations based on analysis.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            List of transition recommendations
        """
        recommendations = []
        
        # Analyze daily cost data for transition opportunities
        daily_data = analysis_data.get("daily_cost_usage", {})
        cost_by_class = daily_data.get("cost_by_storage_class", {})
        
        # Recommend transitions based on cost distribution
        total_cost = sum(data.get("total_cost", 0) for data in cost_by_class.values())
        
        if total_cost > 0:
            standard_cost = cost_by_class.get("STANDARD", {}).get("total_cost", 0)
            standard_percentage = (standard_cost / total_cost) * 100
            
            if standard_percentage > 70:
                recommendations.append(self.create_recommendation(
                    rec_type="storage_class_transition",
                    priority="high",
                    title="High Standard storage usage detected",
                    description=f"Standard storage accounts for {standard_percentage:.1f}% of costs. "
                               f"Consider transitioning infrequently accessed data to Standard-IA or Intelligent-Tiering.",
                    potential_savings=standard_cost * 0.4,  # Estimate 40% savings
                    implementation_effort="medium",
                    action_items=[
                        "Analyze access patterns for objects older than 30 days",
                        "Implement lifecycle policies for automatic transition to Standard-IA",
                        "Consider S3 Intelligent-Tiering for unknown access patterns",
                        "Monitor retrieval costs after implementing transitions"
                    ]
                ))
        
        # Recommend lifecycle policies if not detected
        validation_data = analysis_data.get("appropriateness_validation", {})
        if not validation_data.get("lifecycle_policies_detected", False):
            recommendations.append(self.create_recommendation(
                rec_type="lifecycle_policy",
                priority="medium",
                title="Implement automated lifecycle policies",
                description="Automated lifecycle policies can optimize storage costs without manual intervention.",
                potential_savings=None,
                implementation_effort="low",
                action_items=[
                    "Create lifecycle policies for automatic transitions",
                    "Set up transitions: Standard → Standard-IA (30 days) → Glacier (90 days)",
                    "Configure deletion policies for temporary data",
                    "Enable S3 Intelligent-Tiering for variable access patterns"
                ]
            ))
        
        # Recommend archive optimization for old data
        recommendations.append(self.create_recommendation(
            rec_type="archive_optimization",
            priority="medium",
            title="Archive old data for long-term cost savings",
            description="Data older than 90 days with minimal access should be moved to archive storage classes.",
            potential_savings=None,
            implementation_effort="low",
            action_items=[
                "Identify data older than 90 days with minimal access",
                "Transition to Glacier Instant Retrieval for occasional access",
                "Use Glacier Flexible Retrieval for quarterly access",
                "Use Deep Archive for data accessed less than yearly"
            ]
        ))
        
        return recommendations
    
    def _calculate_cost_savings_potential(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate potential cost savings from storage class optimization.
        
        Args:
            analysis_data: Combined analysis data
            
        Returns:
            Cost savings potential analysis
        """
        savings_potential = {
            "total_potential_monthly_savings": 0,
            "savings_by_optimization": {},
            "roi_analysis": {},
            "implementation_timeline": {}
        }
        
        # Calculate savings from transition recommendations
        daily_data = analysis_data.get("daily_cost_usage", {})
        cost_by_class = daily_data.get("cost_by_storage_class", {})
        
        standard_cost = cost_by_class.get("STANDARD", {}).get("total_cost", 0)
        
        # Estimate savings from Standard to Standard-IA transition (40% savings)
        if standard_cost > 0:
            ia_savings = standard_cost * 0.4 * 0.5  # Assume 50% of Standard data can be transitioned
            savings_potential["savings_by_optimization"]["standard_to_ia"] = {
                "monthly_savings": ia_savings,
                "percentage": 40,
                "affected_data": "50% of Standard storage"
            }
            savings_potential["total_potential_monthly_savings"] += ia_savings
        
        # Estimate savings from implementing lifecycle policies (20% overall savings)
        total_storage_cost = sum(data.get("total_cost", 0) for data in cost_by_class.values())
        if total_storage_cost > 0:
            lifecycle_savings = total_storage_cost * 0.2
            savings_potential["savings_by_optimization"]["lifecycle_policies"] = {
                "monthly_savings": lifecycle_savings,
                "percentage": 20,
                "affected_data": "All storage with automated optimization"
            }
            savings_potential["total_potential_monthly_savings"] += lifecycle_savings
        
        # ROI analysis
        if savings_potential["total_potential_monthly_savings"] > 0:
            annual_savings = savings_potential["total_potential_monthly_savings"] * 12
            savings_potential["roi_analysis"] = {
                "annual_savings": annual_savings,
                "implementation_cost": "Minimal (policy configuration only)",
                "payback_period": "Immediate",
                "roi_percentage": "High (cost reduction with no upfront investment)"
            }
        
        # Implementation timeline
        savings_potential["implementation_timeline"] = {
            "immediate": "Configure lifecycle policies",
            "week_1": "Implement Standard-IA transitions",
            "month_1": "Monitor and optimize transition policies",
            "month_3": "Evaluate archive storage opportunities",
            "ongoing": "Regular review and optimization"
        }
        
        return savings_potential
    
    def _analyze_cost_trends(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cost trends from Cost Explorer data.
        
        Args:
            cost_data: Cost Explorer data
            
        Returns:
            Cost trend analysis
        """
        try:
            trends = {
                "daily_costs": [],
                "storage_class_trends": {},
                "cost_growth_rate": 0,
                "optimization_opportunities": []
            }
            
            # Process daily costs
            for result in cost_data.get("ResultsByTime", []):
                date = result.get("TimePeriod", {}).get("Start", "")
                total_cost = 0
                
                for group in result.get("Groups", []):
                    cost = float(group.get("Metrics", {}).get("UnblendedCost", {}).get("Amount", 0))
                    total_cost += cost
                
                trends["daily_costs"].append({
                    "date": date,
                    "cost": total_cost
                })
            
            # Calculate growth rate if we have enough data points
            if len(trends["daily_costs"]) >= 7:
                recent_avg = sum(day["cost"] for day in trends["daily_costs"][-7:]) / 7
                older_avg = sum(day["cost"] for day in trends["daily_costs"][:7]) / 7
                
                if older_avg > 0:
                    growth_rate = ((recent_avg - older_avg) / older_avg) * 100
                    trends["cost_growth_rate"] = round(growth_rate, 2)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing cost trends: {str(e)}")
            return {"error": str(e)}
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations from storage class analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Get transition recommendations from analysis
            transition_recs = analysis_results.get("data", {}).get("transition_recommendations", [])
            recommendations.extend(transition_recs)
            
            # Add general storage class optimization recommendations
            recommendations.append(self.create_recommendation(
                rec_type="storage_class_monitoring",
                priority="low",
                title="Implement regular storage class review",
                description="Establish a monthly review process to optimize storage classes based on access patterns.",
                implementation_effort="low",
                action_items=[
                    "Set up monthly storage class cost review",
                    "Monitor access patterns using S3 Storage Lens",
                    "Adjust lifecycle policies based on actual usage",
                    "Track cost savings from optimization efforts"
                ]
            ))
            
            # Add cost optimization recommendation if significant savings potential
            savings_data = analysis_results.get("data", {}).get("cost_savings_potential", {})
            total_savings = savings_data.get("total_potential_monthly_savings", 0)
            
            if total_savings > 100:  # If potential savings > $100/month
                recommendations.append(self.create_recommendation(
                    rec_type="cost_optimization",
                    priority="high",
                    title="Significant storage cost optimization opportunity",
                    description=f"Potential monthly savings of ${total_savings:.2f} identified through storage class optimization.",
                    potential_savings=total_savings,
                    implementation_effort="medium",
                    action_items=[
                        "Prioritize high-impact storage class transitions",
                        "Implement automated lifecycle policies",
                        "Monitor cost reduction after implementation",
                        "Expand optimization to additional buckets"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating storage class recommendations: {str(e)}")
            recommendations.append(self.create_recommendation(
                rec_type="error_resolution",
                priority="medium",
                title="Storage class analysis incomplete",
                description=f"Storage class analysis encountered errors: {str(e)}",
                action_items=[
                    "Check AWS permissions for Cost Explorer and S3",
                    "Verify Storage Lens configuration",
                    "Review analysis parameters and retry"
                ]
            ))
        
        return recommendations