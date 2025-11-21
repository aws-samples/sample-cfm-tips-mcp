"""
Cleanup Manager for CFM Tips MCP Server

Handles automatic cleanup of sessions, results, and temporary data.
"""

import logging
import threading
import time
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages automatic cleanup of sessions and temporary data."""
    
    def __init__(self, 
                 session_timeout_minutes: int = 60,
                 result_retention_minutes: int = 120,
                 cleanup_interval_minutes: int = 15):
        self.session_timeout_minutes = session_timeout_minutes
        self.result_retention_minutes = result_retention_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self._shutdown = False
        self._cleanup_thread = None
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="CleanupManager"
            )
            self._cleanup_thread.start()
            logger.info(f"Cleanup manager started (session timeout: {self.session_timeout_minutes}min)")
    
    def _cleanup_worker(self):
        """Background worker for periodic cleanup."""
        while not self._shutdown:
            try:
                self._perform_cleanup()
                time.sleep(self.cleanup_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_cleanup(self):
        """Perform all cleanup operations."""
        logger.debug("Starting periodic cleanup")
        
        # Clean up session files
        self._cleanup_session_files()
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        # Clean up old log files
        self._cleanup_log_files()
        
        logger.debug("Periodic cleanup completed")
    
    def _cleanup_session_files(self):
        """Clean up old session database files."""
        try:
            sessions_dir = "sessions"
            if not os.path.exists(sessions_dir):
                return
            
            cutoff_time = datetime.now() - timedelta(minutes=self.session_timeout_minutes)
            cleaned_count = 0
            
            # Find all session database files
            session_files = glob.glob(os.path.join(sessions_dir, "session_*.db"))
            
            for session_file in session_files:
                try:
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(session_file))
                    
                    if file_mtime < cutoff_time:
                        # Remove old session file
                        os.remove(session_file)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old session file: {session_file}")
                        
                        # Also remove any associated WAL and SHM files
                        for ext in ['-wal', '-shm']:
                            wal_file = session_file + ext
                            if os.path.exists(wal_file):
                                os.remove(wal_file)
                
                except Exception as e:
                    logger.warning(f"Error cleaning session file {session_file}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old session files")
                
        except Exception as e:
            logger.error(f"Error in session file cleanup: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        try:
            temp_patterns = [
                "*.tmp",
                "*.temp",
                "__pycache__/*.pyc",
                "*.log.old"
            ]
            
            cleaned_count = 0
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern, recursive=True)
                
                for temp_file in temp_files:
                    try:
                        # Check if file is older than retention period
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(temp_file))
                        cutoff_time = datetime.now() - timedelta(minutes=self.result_retention_minutes)
                        
                        if file_mtime < cutoff_time:
                            if os.path.isfile(temp_file):
                                os.remove(temp_file)
                                cleaned_count += 1
                            elif os.path.isdir(temp_file):
                                import shutil
                                shutil.rmtree(temp_file)
                                cleaned_count += 1
                    
                    except Exception as e:
                        logger.warning(f"Error cleaning temp file {temp_file}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")
                
        except Exception as e:
            logger.error(f"Error in temp file cleanup: {e}")
    
    def _cleanup_log_files(self):
        """Clean up old log files."""
        try:
            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                return
            
            # Keep logs for 7 days
            cutoff_time = datetime.now() - timedelta(days=7)
            cleaned_count = 0
            
            log_files = glob.glob(os.path.join(logs_dir, "*.log.*"))
            
            for log_file in log_files:
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                    
                    if file_mtime < cutoff_time:
                        os.remove(log_file)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old log file: {log_file}")
                
                except Exception as e:
                    logger.warning(f"Error cleaning log file {log_file}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old log files")
                
        except Exception as e:
            logger.error(f"Error in log file cleanup: {e}")
    
    def force_cleanup(self):
        """Force immediate cleanup of all resources."""
        logger.info("Forcing immediate cleanup")
        self._perform_cleanup()
        
        # Also clean up from session manager and parallel executor
        try:
            from . import get_session_manager, get_parallel_executor
            
            # Clean up session manager
            session_manager = get_session_manager()
            session_manager._cleanup_expired_sessions()
            
            # Clean up parallel executor results
            executor = get_parallel_executor()
            executor.clear_results(older_than_minutes=self.result_retention_minutes)
            
            logger.info("Force cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get statistics about cleanup operations."""
        try:
            stats = {
                'session_timeout_minutes': self.session_timeout_minutes,
                'result_retention_minutes': self.result_retention_minutes,
                'cleanup_interval_minutes': self.cleanup_interval_minutes,
                'cleanup_thread_alive': self._cleanup_thread.is_alive() if self._cleanup_thread else False,
                'sessions_directory_exists': os.path.exists('sessions'),
                'logs_directory_exists': os.path.exists('logs')
            }
            
            # Count current files
            if os.path.exists('sessions'):
                session_files = glob.glob('sessions/session_*.db')
                stats['active_session_files'] = len(session_files)
            else:
                stats['active_session_files'] = 0
            
            if os.path.exists('logs'):
                log_files = glob.glob('logs/*.log*')
                stats['log_files'] = len(log_files)
            else:
                stats['log_files'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cleanup stats: {e}")
            return {'error': str(e)}
    
    def update_settings(self, 
                       session_timeout_minutes: Optional[int] = None,
                       result_retention_minutes: Optional[int] = None,
                       cleanup_interval_minutes: Optional[int] = None):
        """Update cleanup settings."""
        if session_timeout_minutes is not None:
            self.session_timeout_minutes = session_timeout_minutes
            logger.info(f"Updated session timeout to {session_timeout_minutes} minutes")
        
        if result_retention_minutes is not None:
            self.result_retention_minutes = result_retention_minutes
            logger.info(f"Updated result retention to {result_retention_minutes} minutes")
        
        if cleanup_interval_minutes is not None:
            self.cleanup_interval_minutes = cleanup_interval_minutes
            logger.info(f"Updated cleanup interval to {cleanup_interval_minutes} minutes")
    
    def shutdown(self):
        """Shutdown the cleanup manager."""
        logger.info("Shutting down cleanup manager")
        self._shutdown = True
        
        # Perform final cleanup
        try:
            self._perform_cleanup()
        except Exception as e:
            logger.error(f"Error in final cleanup: {e}")
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=10)

# Global cleanup manager instance
_cleanup_manager = None

def get_cleanup_manager() -> CleanupManager:
    """Get the global cleanup manager instance."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = CleanupManager()
    return _cleanup_manager