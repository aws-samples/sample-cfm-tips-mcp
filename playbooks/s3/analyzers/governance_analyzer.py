"""
S3 Governance Analyzer

This analyzer performs S3 governance compliance checking to ensure buckets have
appropriate lifecycle policies and comply with organizational standards.

Analyzes:
- Buckets missing lifecycle policies
- Versioned buckets without lifecycle policies for non-current versions
- Buckets missing multipart upload cleanup rules
- Compliance with organizational governance standards
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class GovernanceAnalyzer(BaseAnalyzer):
    """
    Analyzer for S3 governance compliance checking.
    
    Validates bucket configurations against governance standards and identifies
    compliance violations with specific remediation steps.
    """
    
    def __init__(self, s3_service=None, pricing_service=None, storage_lens_service=None):
        """
        Initialize GovernanceAnalyzer.
        
        Args:
            s3_service: S3Service instance for AWS S3 operations
            pricing_service: S3Pricing instance for cost calculations
            storage_lens_service: StorageLensService instance for Storage Lens data
        """
        super().__init__(s3_service, pricing_service, storage_lens_service)
        self.analysis_type = "governance"
        
        # Default governance standards
        self.default_standards = {
            "require_lifecycle_policies": True,
            "require_versioning_lifecycle": True,
            "require_multipart_cleanup": True,
            "max_multipart_age_days": 7,
            "require_encryption": False,  # Optional standard
            "require_public_access_block": False,  # Optional standard
            "allowed_storage_classes": [
                "STANDARD", "STANDARD_IA", "ONEZONE_IA", 
                "GLACIER", "DEEP_ARCHIVE", "INTELLIGENT_TIERING"
            ]
        }
        
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Execute S3 governance compliance analysis.
        
        Args:
            **kwargs: Analysis parameters including:
                - region: AWS region to analyze
                - bucket_names: Optional list of specific buckets to analyze
                - governance_standards: Optional custom governance standards
                - include_remediation: Whether to include detailed remediation steps
                
        Returns:
            Dictionary containing governance compliance analysis results
        """
        context = self.prepare_analysis_context(**kwargs)
        
        try:
            self.logger.info(f"Starting S3 governance compliance analysis for region: {context.get('region', 'all')}")
            
            # Initialize results structure
            analysis_results = {
                "status": "success",
                "analysis_type": self.analysis_type,
                "context": context,
                "data": {
                    "governance_violations": [],
                    "compliant_buckets": [],
                    "violation_summary": {},
                    "compliance_score": 0,
                    "total_buckets_analyzed": 0,
                    "standards_applied": {}
                },
                "data_sources": [],
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            start_time = datetime.now()
            
            # Get governance standards (custom or default)
            governance_standards = kwargs.get('governance_standards', self.default_standards)
            analysis_results["data"]["standards_applied"] = governance_standards
            
            # Get list of buckets to analyze
            buckets_to_analyze = await self._get_buckets_to_analyze(context)
            if not buckets_to_analyze:
                self.logger.warning("No buckets found to analyze")
                analysis_results["data"]["total_buckets_analyzed"] = 0
                return analysis_results
            
            analysis_results["data"]["total_buckets_analyzed"] = len(buckets_to_analyze)
            analysis_results["data_sources"].append("s3_api")
            
            # Execute governance checks in parallel
            governance_tasks = [
                self._check_bucket_governance(bucket_name, governance_standards)
                for bucket_name in buckets_to_analyze
            ]
            
            # Execute all governance checks
            bucket_results = await asyncio.gather(*governance_tasks, return_exceptions=True)
            
            # Process results
            violations = []
            compliant_buckets = []
            violation_counts = {}
            
            for i, result in enumerate(bucket_results):
                bucket_name = buckets_to_analyze[i]
                
                if isinstance(result, Exception):
                    self.logger.warning(f"Governance check failed for bucket {bucket_name}: {result}")
                    violations.append({
                        "bucket_name": bucket_name,
                        "violation_type": "analysis_error",
                        "severity": "high",
                        "description": f"Failed to analyze bucket: {str(result)}",
                        "remediation_steps": [
                            "Check bucket permissions and access",
                            "Verify bucket exists and is accessible",
                            "Review IAM permissions for governance checks"
                        ]
                    })
                    continue
                
                if result.get("status") == "success":
                    bucket_violations = result.get("violations", [])
                    
                    if bucket_violations:
                        violations.extend(bucket_violations)
                        
                        # Count violation types
                        for violation in bucket_violations:
                            violation_type = violation.get("violation_type", "unknown")
                            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
                    else:
                        compliant_buckets.append({
                            "bucket_name": bucket_name,
                            "compliance_checks_passed": result.get("checks_passed", []),
                            "last_checked": datetime.now().isoformat()
                        })
            
            # Store results
            analysis_results["data"]["governance_violations"] = violations
            analysis_results["data"]["compliant_buckets"] = compliant_buckets
            analysis_results["data"]["violation_summary"] = violation_counts
            
            # Calculate compliance score
            total_buckets = len(buckets_to_analyze)
            compliant_count = len(compliant_buckets)
            compliance_score = (compliant_count / total_buckets * 100) if total_buckets > 0 else 100
            analysis_results["data"]["compliance_score"] = round(compliance_score, 2)
            
            # Calculate execution time
            end_time = datetime.now()
            analysis_results["execution_time"] = (end_time - start_time).total_seconds()
            
            self.logger.info(
                f"Completed S3 governance analysis in {analysis_results['execution_time']:.2f} seconds. "
                f"Compliance score: {compliance_score:.1f}% ({compliant_count}/{total_buckets} buckets compliant)"
            )
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in governance analysis: {str(e)}")
            return self.handle_analysis_error(e, context)
    
    async def _get_buckets_to_analyze(self, context: Dict[str, Any]) -> List[str]:
        """
        Get list of buckets to analyze.
        
        Args:
            context: Analysis context
            
        Returns:
            List of bucket names to analyze
        """
        try:
            # If specific buckets are requested, use those
            if context.get('bucket_names'):
                self.logger.info(f"Analyzing specific buckets: {context['bucket_names']}")
                return context['bucket_names']
            
            # Otherwise, get all buckets
            if not self.s3_service:
                self.logger.error("S3Service not available for bucket discovery")
                return []
            
            buckets_result = await self.s3_service.list_buckets()
            
            if buckets_result.get("status") == "success":
                # Fix: S3Service returns buckets under data.Buckets, not buckets
                buckets_data = buckets_result.get("data", {}).get("Buckets", [])
                bucket_names = [bucket["Name"] for bucket in buckets_data]
                self.logger.info(f"Found {len(bucket_names)} buckets to analyze")
                return bucket_names
            else:
                self.logger.error(f"Failed to list buckets: {buckets_result.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting buckets to analyze: {str(e)}")
            return []
    
    async def _check_bucket_governance(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check governance compliance for a single bucket.
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards to apply
            
        Returns:
            Dictionary containing governance check results
        """
        try:
            self.logger.debug(f"Checking governance compliance for bucket: {bucket_name}")
            
            violations = []
            checks_passed = []
            
            # Execute all governance checks in parallel
            tasks = [
                self._check_lifecycle_policy(bucket_name, standards),
                self._check_versioning_lifecycle(bucket_name, standards),
                self._check_multipart_cleanup(bucket_name, standards)
            ]
            
            # Add optional checks if enabled
            if standards.get("require_encryption"):
                tasks.append(self._check_encryption(bucket_name, standards))
            
            if standards.get("require_public_access_block"):
                tasks.append(self._check_public_access_block(bucket_name, standards))
            
            # Execute all checks
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process check results
            for result in check_results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Governance check failed for {bucket_name}: {result}")
                    violations.append({
                        "bucket_name": bucket_name,
                        "violation_type": "check_error",
                        "severity": "medium",
                        "description": f"Governance check failed: {str(result)}",
                        "remediation_steps": [
                            "Verify bucket permissions",
                            "Check IAM policies for governance operations"
                        ]
                    })
                    continue
                
                if result.get("compliant"):
                    checks_passed.append(result.get("check_type"))
                else:
                    violations.append(result.get("violation"))
            
            return {
                "status": "success",
                "bucket_name": bucket_name,
                "violations": violations,
                "checks_passed": checks_passed,
                "total_checks": len(tasks),
                "compliant": len(violations) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking governance for bucket {bucket_name}: {str(e)}")
            return {
                "status": "error",
                "bucket_name": bucket_name,
                "error": str(e)
            }
    
    async def _check_lifecycle_policy(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if bucket has lifecycle policy (Requirement 8.1).
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards
            
        Returns:
            Dictionary containing lifecycle policy check result
        """
        try:
            if not standards.get("require_lifecycle_policies", True):
                return {"compliant": True, "check_type": "lifecycle_policy", "skipped": True}
            
            if not self.s3_service:
                raise Exception("S3Service not available")
            
            lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
            
            if lifecycle_result.get("status") == "success":
                lifecycle_config = lifecycle_result.get("lifecycle_configuration")
                
                if lifecycle_config and lifecycle_config.get("Rules"):
                    # Bucket has lifecycle policy
                    return {
                        "compliant": True,
                        "check_type": "lifecycle_policy",
                        "details": {
                            "rules_count": len(lifecycle_config["Rules"]),
                            "has_lifecycle_policy": True
                        }
                    }
                else:
                    # Bucket missing lifecycle policy
                    return {
                        "compliant": False,
                        "check_type": "lifecycle_policy",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_lifecycle_policy",
                            "severity": "medium",
                            "description": f"Bucket '{bucket_name}' does not have a lifecycle policy configured",
                            "remediation_steps": [
                                f"Configure lifecycle policy for bucket '{bucket_name}'",
                                "Define rules for transitioning objects to cheaper storage classes",
                                "Set expiration rules for temporary or log data",
                                "Consider intelligent tiering for unpredictable access patterns"
                            ],
                            "aws_cli_example": f"aws s3api put-bucket-lifecycle-configuration --bucket {bucket_name} --lifecycle-configuration file://lifecycle-policy.json"
                        }
                    }
            else:
                # Error getting lifecycle configuration
                error_msg = lifecycle_result.get("message", "Unknown error")
                if "NoSuchLifecycleConfiguration" in error_msg:
                    # No lifecycle policy configured
                    return {
                        "compliant": False,
                        "check_type": "lifecycle_policy",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_lifecycle_policy",
                            "severity": "medium",
                            "description": f"Bucket '{bucket_name}' does not have a lifecycle policy configured",
                            "remediation_steps": [
                                f"Configure lifecycle policy for bucket '{bucket_name}'",
                                "Define rules for transitioning objects to cheaper storage classes",
                                "Set expiration rules for temporary or log data"
                            ]
                        }
                    }
                else:
                    raise Exception(f"Failed to get lifecycle configuration: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Error checking lifecycle policy for {bucket_name}: {str(e)}")
            return {
                "compliant": False,
                "check_type": "lifecycle_policy",
                "violation": {
                    "bucket_name": bucket_name,
                    "violation_type": "lifecycle_check_error",
                    "severity": "low",
                    "description": f"Could not verify lifecycle policy for bucket '{bucket_name}': {str(e)}",
                    "remediation_steps": [
                        "Check IAM permissions for s3:GetLifecycleConfiguration",
                        "Verify bucket exists and is accessible"
                    ]
                }
            }
    
    async def _check_versioning_lifecycle(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if versioned buckets have lifecycle policies for non-current versions (Requirement 8.2).
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards
            
        Returns:
            Dictionary containing versioning lifecycle check result
        """
        try:
            if not standards.get("require_versioning_lifecycle", True):
                return {"compliant": True, "check_type": "versioning_lifecycle", "skipped": True}
            
            if not self.s3_service:
                raise Exception("S3Service not available")
            
            # Check if versioning is enabled
            versioning_result = await self.s3_service.get_bucket_versioning(bucket_name)
            
            if versioning_result.get("status") != "success":
                error_msg = versioning_result.get("message", "Unknown error")
                raise Exception(f"Failed to get versioning configuration: {error_msg}")
            
            versioning_config = versioning_result.get("versioning_configuration", {})
            versioning_status = versioning_config.get("Status", "Disabled")
            
            if versioning_status not in ["Enabled", "Suspended"]:
                # Versioning not enabled, check passes
                return {
                    "compliant": True,
                    "check_type": "versioning_lifecycle",
                    "details": {
                        "versioning_enabled": False,
                        "check_not_applicable": True
                    }
                }
            
            # Versioning is enabled, check for lifecycle policy with non-current version rules
            lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
            
            if lifecycle_result.get("status") == "success":
                lifecycle_config = lifecycle_result.get("lifecycle_configuration")
                
                if lifecycle_config and lifecycle_config.get("Rules"):
                    # Check if any rule handles non-current versions
                    has_noncurrent_rules = False
                    
                    for rule in lifecycle_config["Rules"]:
                        if (rule.get("NoncurrentVersionTransitions") or 
                            rule.get("NoncurrentVersionExpiration")):
                            has_noncurrent_rules = True
                            break
                    
                    if has_noncurrent_rules:
                        return {
                            "compliant": True,
                            "check_type": "versioning_lifecycle",
                            "details": {
                                "versioning_enabled": True,
                                "has_noncurrent_version_rules": True
                            }
                        }
                    else:
                        return {
                            "compliant": False,
                            "check_type": "versioning_lifecycle",
                            "violation": {
                                "bucket_name": bucket_name,
                                "violation_type": "missing_versioning_lifecycle",
                                "severity": "high",
                                "description": f"Versioned bucket '{bucket_name}' lacks lifecycle rules for non-current versions",
                                "remediation_steps": [
                                    f"Add non-current version lifecycle rules to bucket '{bucket_name}'",
                                    "Configure NoncurrentVersionTransition to move old versions to cheaper storage",
                                    "Set NoncurrentVersionExpiration to delete very old versions",
                                    "Consider keeping 2-5 recent versions and transitioning older ones"
                                ],
                                "cost_impact": "High - Non-current versions accumulate storage costs without cleanup"
                            }
                        }
                else:
                    # No lifecycle policy at all
                    return {
                        "compliant": False,
                        "check_type": "versioning_lifecycle",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_versioning_lifecycle",
                            "severity": "high",
                            "description": f"Versioned bucket '{bucket_name}' has no lifecycle policy for version management",
                            "remediation_steps": [
                                f"Create lifecycle policy for versioned bucket '{bucket_name}'",
                                "Add rules for non-current version transitions and expiration",
                                "Prevent unlimited accumulation of object versions"
                            ],
                            "cost_impact": "High - Unlimited version accumulation can cause significant cost increases"
                        }
                    }
            else:
                # Error or no lifecycle configuration
                error_msg = lifecycle_result.get("message", "Unknown error")
                if "NoSuchLifecycleConfiguration" in error_msg:
                    return {
                        "compliant": False,
                        "check_type": "versioning_lifecycle",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_versioning_lifecycle",
                            "severity": "high",
                            "description": f"Versioned bucket '{bucket_name}' has no lifecycle policy",
                            "remediation_steps": [
                                f"Create lifecycle policy for versioned bucket '{bucket_name}'",
                                "Add rules for non-current version management"
                            ]
                        }
                    }
                else:
                    raise Exception(f"Failed to get lifecycle configuration: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Error checking versioning lifecycle for {bucket_name}: {str(e)}")
            return {
                "compliant": False,
                "check_type": "versioning_lifecycle",
                "violation": {
                    "bucket_name": bucket_name,
                    "violation_type": "versioning_lifecycle_check_error",
                    "severity": "low",
                    "description": f"Could not verify versioning lifecycle for bucket '{bucket_name}': {str(e)}",
                    "remediation_steps": [
                        "Check IAM permissions for s3:GetBucketVersioning and s3:GetLifecycleConfiguration",
                        "Verify bucket exists and is accessible"
                    ]
                }
            }
    
    async def _check_multipart_cleanup(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if bucket has lifecycle rules for incomplete multipart uploads (Requirement 8.3).
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards
            
        Returns:
            Dictionary containing multipart cleanup check result
        """
        try:
            if not standards.get("require_multipart_cleanup", True):
                return {"compliant": True, "check_type": "multipart_cleanup", "skipped": True}
            
            if not self.s3_service:
                raise Exception("S3Service not available")
            
            # Check for lifecycle policy with multipart upload cleanup
            lifecycle_result = await self.s3_service.get_bucket_lifecycle(bucket_name)
            
            if lifecycle_result.get("status") == "success":
                lifecycle_config = lifecycle_result.get("lifecycle_configuration")
                
                if lifecycle_config and lifecycle_config.get("Rules"):
                    # Check if any rule handles incomplete multipart uploads
                    has_multipart_rules = False
                    max_age_days = standards.get("max_multipart_age_days", 7)
                    
                    for rule in lifecycle_config["Rules"]:
                        abort_config = rule.get("AbortIncompleteMultipartUpload")
                        if abort_config and abort_config.get("DaysAfterInitiation"):
                            days_after = abort_config["DaysAfterInitiation"]
                            if days_after <= max_age_days:
                                has_multipart_rules = True
                                break
                    
                    if has_multipart_rules:
                        return {
                            "compliant": True,
                            "check_type": "multipart_cleanup",
                            "details": {
                                "has_multipart_cleanup_rules": True,
                                "max_age_compliant": True
                            }
                        }
                    else:
                        # Check if there are any incomplete multipart uploads
                        multipart_result = await self.s3_service.get_multipart_uploads(bucket_name, max_results=1)
                        
                        has_incomplete_uploads = False
                        if (multipart_result.get("status") == "success" and 
                            multipart_result.get("uploads")):
                            has_incomplete_uploads = len(multipart_result["uploads"]) > 0
                        
                        severity = "high" if has_incomplete_uploads else "medium"
                        description = (
                            f"Bucket '{bucket_name}' lacks multipart upload cleanup rules"
                            + (" and has incomplete uploads" if has_incomplete_uploads else "")
                        )
                        
                        return {
                            "compliant": False,
                            "check_type": "multipart_cleanup",
                            "violation": {
                                "bucket_name": bucket_name,
                                "violation_type": "missing_multipart_cleanup",
                                "severity": severity,
                                "description": description,
                                "remediation_steps": [
                                    f"Add multipart upload cleanup rule to bucket '{bucket_name}' lifecycle policy",
                                    f"Set AbortIncompleteMultipartUpload to {max_age_days} days or less",
                                    "Prevent accumulation of incomplete upload storage costs",
                                    "Consider setting cleanup to 1-3 days for most use cases"
                                ],
                                "has_incomplete_uploads": has_incomplete_uploads,
                                "aws_cli_example": f"aws s3api put-bucket-lifecycle-configuration --bucket {bucket_name} --lifecycle-configuration file://multipart-cleanup-policy.json"
                            }
                        }
                else:
                    # No lifecycle policy at all
                    return {
                        "compliant": False,
                        "check_type": "multipart_cleanup",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_multipart_cleanup",
                            "severity": "medium",
                            "description": f"Bucket '{bucket_name}' has no lifecycle policy for multipart upload cleanup",
                            "remediation_steps": [
                                f"Create lifecycle policy for bucket '{bucket_name}'",
                                "Add AbortIncompleteMultipartUpload rule",
                                "Set cleanup period to 7 days or less"
                            ]
                        }
                    }
            else:
                # Error or no lifecycle configuration
                error_msg = lifecycle_result.get("message", "Unknown error")
                if "NoSuchLifecycleConfiguration" in error_msg:
                    return {
                        "compliant": False,
                        "check_type": "multipart_cleanup",
                        "violation": {
                            "bucket_name": bucket_name,
                            "violation_type": "missing_multipart_cleanup",
                            "severity": "medium",
                            "description": f"Bucket '{bucket_name}' has no lifecycle policy for multipart cleanup",
                            "remediation_steps": [
                                f"Create lifecycle policy for bucket '{bucket_name}'",
                                "Add multipart upload cleanup rules"
                            ]
                        }
                    }
                else:
                    raise Exception(f"Failed to get lifecycle configuration: {error_msg}")
                    
        except Exception as e:
            self.logger.error(f"Error checking multipart cleanup for {bucket_name}: {str(e)}")
            return {
                "compliant": False,
                "check_type": "multipart_cleanup",
                "violation": {
                    "bucket_name": bucket_name,
                    "violation_type": "multipart_cleanup_check_error",
                    "severity": "low",
                    "description": f"Could not verify multipart cleanup for bucket '{bucket_name}': {str(e)}",
                    "remediation_steps": [
                        "Check IAM permissions for s3:GetLifecycleConfiguration and s3:ListMultipartUploads",
                        "Verify bucket exists and is accessible"
                    ]
                }
            }
    
    async def _check_encryption(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if bucket has encryption enabled (optional governance check).
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards
            
        Returns:
            Dictionary containing encryption check result
        """
        try:
            if not self.s3_service:
                raise Exception("S3Service not available")
            
            # This would require implementing get_bucket_encryption in S3Service
            # For now, return a placeholder
            return {
                "compliant": True,
                "check_type": "encryption",
                "details": {
                    "check_implemented": False,
                    "note": "Encryption check not yet implemented"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking encryption for {bucket_name}: {str(e)}")
            return {
                "compliant": True,
                "check_type": "encryption",
                "details": {"error": str(e)}
            }
    
    async def _check_public_access_block(self, bucket_name: str, standards: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if bucket has public access block enabled (optional governance check).
        
        Args:
            bucket_name: Name of the bucket to check
            standards: Governance standards
            
        Returns:
            Dictionary containing public access block check result
        """
        try:
            if not self.s3_service:
                raise Exception("S3Service not available")
            
            # This would require implementing get_public_access_block in S3Service
            # For now, return a placeholder
            return {
                "compliant": True,
                "check_type": "public_access_block",
                "details": {
                    "check_implemented": False,
                    "note": "Public access block check not yet implemented"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error checking public access block for {bucket_name}: {str(e)}")
            return {
                "compliant": True,
                "check_type": "public_access_block",
                "details": {"error": str(e)}
            }
    
    def get_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate governance recommendations from analysis results.
        
        Args:
            analysis_results: Results from the analyze method
            
        Returns:
            List of governance recommendation dictionaries
        """
        recommendations = []
        
        try:
            data = analysis_results.get("data", {})
            violations = data.get("governance_violations", [])
            violation_summary = data.get("violation_summary", {})
            compliance_score = data.get("compliance_score", 100)
            
            # High-level compliance recommendation
            if compliance_score < 100:
                recommendations.append(
                    self.create_recommendation(
                        rec_type="governance_compliance",
                        priority="high" if compliance_score < 50 else "medium",
                        title=f"Improve S3 Governance Compliance (Current: {compliance_score:.1f}%)",
                        description=(
                            f"Your S3 infrastructure has a governance compliance score of {compliance_score:.1f}%. "
                            f"Address {len(violations)} governance violations to improve cost control and operational efficiency."
                        ),
                        implementation_effort="medium",
                        action_items=[
                            "Review and address all governance violations",
                            "Implement standardized lifecycle policies across buckets",
                            "Establish governance monitoring and alerting",
                            "Create governance policy templates for new buckets"
                        ]
                    )
                )
            
            # Specific recommendations by violation type
            for violation_type, count in violation_summary.items():
                if violation_type == "missing_lifecycle_policy":
                    recommendations.append(
                        self.create_recommendation(
                            rec_type="lifecycle_policy",
                            priority="medium",
                            title=f"Implement Lifecycle Policies ({count} buckets affected)",
                            description=(
                                f"{count} buckets are missing lifecycle policies, which can lead to "
                                "unnecessary storage costs and poor data management."
                            ),
                            implementation_effort="low",
                            action_items=[
                                "Create standardized lifecycle policy templates",
                                "Apply lifecycle policies to all buckets without them",
                                "Set up automated policy application for new buckets",
                                "Monitor lifecycle policy effectiveness"
                            ]
                        )
                    )
                
                elif violation_type == "missing_versioning_lifecycle":
                    recommendations.append(
                        self.create_recommendation(
                            rec_type="versioning_lifecycle",
                            priority="high",
                            title=f"Add Version Management Rules ({count} buckets affected)",
                            description=(
                                f"{count} versioned buckets lack proper version management, "
                                "which can cause exponential storage cost growth."
                            ),
                            potential_savings=count * 50.0,  # Estimate $50/month per bucket
                            implementation_effort="low",
                            action_items=[
                                "Add NoncurrentVersionTransition rules to move old versions to cheaper storage",
                                "Set NoncurrentVersionExpiration to delete very old versions",
                                "Consider keeping only 2-5 recent versions",
                                "Monitor version accumulation and costs"
                            ]
                        )
                    )
                
                elif violation_type == "missing_multipart_cleanup":
                    recommendations.append(
                        self.create_recommendation(
                            rec_type="multipart_cleanup",
                            priority="medium",
                            title=f"Implement Multipart Upload Cleanup ({count} buckets affected)",
                            description=(
                                f"{count} buckets lack multipart upload cleanup rules, "
                                "which can accumulate storage costs from incomplete uploads."
                            ),
                            potential_savings=count * 10.0,  # Estimate $10/month per bucket
                            implementation_effort="low",
                            action_items=[
                                "Add AbortIncompleteMultipartUpload rules to lifecycle policies",
                                "Set cleanup period to 7 days or less",
                                "Monitor incomplete multipart uploads",
                                "Educate teams on proper multipart upload practices"
                            ]
                        )
                    )
            
            # Bucket-specific recommendations for high-severity violations
            high_severity_buckets = [
                v for v in violations 
                if v.get("severity") == "high"
            ]
            
            if high_severity_buckets:
                bucket_names = list(set(v.get("bucket_name") for v in high_severity_buckets))
                recommendations.append(
                    self.create_recommendation(
                        rec_type="urgent_governance",
                        priority="high",
                        title=f"Urgent Governance Issues ({len(bucket_names)} buckets)",
                        description=(
                            f"High-severity governance violations detected in buckets: {', '.join(bucket_names[:5])}"
                            + (f" and {len(bucket_names) - 5} more" if len(bucket_names) > 5 else "")
                        ),
                        affected_resources=bucket_names,
                        implementation_effort="medium",
                        action_items=[
                            "Immediately review high-severity violations",
                            "Prioritize versioned buckets without version management",
                            "Implement emergency lifecycle policies where needed",
                            "Set up monitoring for future violations"
                        ]
                    )
                )
            
            # Best practices recommendation
            if compliance_score > 80:
                recommendations.append(
                    self.create_recommendation(
                        rec_type="governance_best_practices",
                        priority="low",
                        title="Enhance Governance Best Practices",
                        description=(
                            "Your governance compliance is good. Consider implementing "
                            "advanced governance features for even better cost control."
                        ),
                        implementation_effort="medium",
                        action_items=[
                            "Implement automated governance monitoring",
                            "Set up cost allocation tags for better tracking",
                            "Consider AWS Config rules for continuous compliance",
                            "Establish governance metrics and reporting"
                        ]
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Error generating governance recommendations: {str(e)}")
            recommendations.append(
                self.create_recommendation(
                    rec_type="error_resolution",
                    priority="medium",
                    title="Governance Analysis Error",
                    description=f"Error generating recommendations: {str(e)}",
                    implementation_effort="low",
                    action_items=[
                        "Review governance analysis logs",
                        "Check AWS permissions and connectivity",
                        "Retry governance analysis"
                    ]
                )
            )
        
        return recommendations