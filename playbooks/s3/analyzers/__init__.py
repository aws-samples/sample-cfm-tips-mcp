"""
S3 Optimization Analyzers Package

This package contains all S3 optimization analyzers that extend BaseAnalyzer.
Each analyzer focuses on a specific aspect of S3 cost optimization.
"""

from .general_spend_analyzer import GeneralSpendAnalyzer
from .storage_class_analyzer import StorageClassAnalyzer
from .archive_optimization_analyzer import ArchiveOptimizationAnalyzer
from .api_cost_analyzer import ApiCostAnalyzer
from .multipart_cleanup_analyzer import MultipartCleanupAnalyzer
from .governance_analyzer import GovernanceAnalyzer

__all__ = [
    'GeneralSpendAnalyzer',
    'StorageClassAnalyzer', 
    'ArchiveOptimizationAnalyzer',
    'ApiCostAnalyzer',
    'MultipartCleanupAnalyzer',
    'GovernanceAnalyzer'
]