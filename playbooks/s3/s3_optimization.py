"""
S3 Cost Optimization Playbook - Consolidated Module

This module has been consolidated and cleaned up. The main S3 optimization functionality
has been moved to the new architecture:
- core/s3_optimization_orchestrator.py (main orchestrator)
- core/s3_analysis_engine.py (analysis engine)
- core/analyzers/ (individual analyzer implementations)

This file now contains only essential imports and references for backward compatibility.
All new development should use the S3OptimizationOrchestrator from playbooks.s3.s3_optimization_orchestrator.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import the new orchestrator for any legacy compatibility needs
try:
    from .s3_optimization_orchestrator import S3OptimizationOrchestrator
    
    # Provide a compatibility alias for any remaining legacy code
    S3Optimization = S3OptimizationOrchestrator
    
    logger.info("S3 optimization functionality available via S3OptimizationOrchestrator")
    
except ImportError as e:
    logger.error(f"Failed to import S3OptimizationOrchestrator: {e}")
    
    # Fallback class for error handling
    class S3Optimization:
        """
        Fallback S3Optimization class when the new orchestrator is not available.
        
        This class provides basic error handling and guidance to use the new architecture.
        """
        
        def __init__(self, region: Optional[str] = None, timeout_seconds: int = 45):
            """
            Initialize fallback S3 optimization.
            
            Args:
                region: AWS region (optional)
                timeout_seconds: Maximum execution time per analysis (default: 45)
            """
            self.region = region
            self.timeout_seconds = timeout_seconds
            logger.warning("Using fallback S3Optimization class. Please use S3OptimizationOrchestrator instead.")
        
        def __getattr__(self, name: str) -> Any:
            """
            Handle any method calls by providing guidance to use the new architecture.
            
            Args:
                name: Method name being called
                
            Returns:
                Error response with guidance
            """
            logger.error(f"Method '{name}' called on fallback S3Optimization class")
            return lambda *args, **kwargs: {
                "status": "error",
                "message": f"S3Optimization.{name}() is deprecated. Use S3OptimizationOrchestrator instead.",
                "guidance": {
                    "new_class": "S3OptimizationOrchestrator",
                    "import_path": "from playbooks.s3.s3_optimization_orchestrator import S3OptimizationOrchestrator",
                    "migration_note": "The new orchestrator provides all S3 optimization functionality with improved performance and session integration."
                },
                "data": {}
            }


# Utility functions for backward compatibility
def get_s3_optimization_instance(region: Optional[str] = None, timeout_seconds: int = 45) -> S3Optimization:
    """
    Get an S3 optimization instance (preferably the new orchestrator).
    
    Args:
        region: AWS region (optional)
        timeout_seconds: Maximum execution time per analysis (default: 45)
        
    Returns:
        S3Optimization instance (either orchestrator or fallback)
    """
    try:
        return S3OptimizationOrchestrator(region=region)
    except Exception as e:
        logger.warning(f"Could not create S3OptimizationOrchestrator, using fallback: {e}")
        return S3Optimization(region=region, timeout_seconds=timeout_seconds)


# Export the main class for backward compatibility
__all__ = ['S3Optimization', 'get_s3_optimization_instance']