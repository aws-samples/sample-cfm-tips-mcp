"""
Session SQL Manager for CFM Tips MCP Server

Provides centralized session management with SQL storage and automatic cleanup.
"""

import sqlite3
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import os
import tempfile

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages SQL sessions with automatic cleanup and thread safety."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self.session_timeout_minutes = session_timeout_minutes
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = False
        self._use_memory_only = False
        
        # Determine storage mode - try persistent first, then temp, then memory-only
        self.sessions_dir = None
        
        # Try persistent sessions directory first
        try:
            sessions_dir = "sessions"
            if os.path.exists(sessions_dir):
                # Test write permissions on existing directory
                test_file = os.path.join(sessions_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.sessions_dir = sessions_dir
                logger.info(f"Using existing sessions directory: {self.sessions_dir}")
            else:
                # Try to create the directory
                os.makedirs(sessions_dir, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(sessions_dir, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.sessions_dir = sessions_dir
                logger.info(f"Created sessions directory: {self.sessions_dir}")
        except Exception:
            # Try temporary directory
            try:
                self.sessions_dir = tempfile.mkdtemp(prefix="cfm_sessions_")
                logger.info(f"Using temporary sessions directory: {self.sessions_dir}")
            except Exception:
                # Fall back to memory-only mode
                logger.info("Using memory-only session storage")
                self._use_memory_only = True
                self.sessions_dir = None
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="SessionCleanup"
            )
            self._cleanup_thread.start()
            logger.info("Session cleanup thread started")
    
    def _cleanup_worker(self):
        """Background worker for session cleanup."""
        while not self._shutdown:
            try:
                self._cleanup_expired_sessions()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        cutoff_time = datetime.now() - timedelta(minutes=self.session_timeout_minutes)
        
        with self._lock:
            expired_sessions = []
            
            for session_id, session_info in self.active_sessions.items():
                if session_info['last_accessed'] < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                try:
                    self._close_session(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session with SQL database or memory-only storage."""
        if session_id is None:
            session_id = f"session_{int(time.time())}_{threading.current_thread().ident}"
        
        with self._lock:
            if session_id in self.active_sessions:
                # Update last accessed time
                self.active_sessions[session_id]['last_accessed'] = datetime.now()
                return session_id
            
            try:
                if self._use_memory_only:
                    # Use in-memory SQLite database
                    conn = sqlite3.connect(":memory:", check_same_thread=False)
                    db_path = ":memory:"
                    logger.info(f"Created in-memory session: {session_id}")
                else:
                    # Create persistent session database
                    db_path = os.path.join(self.sessions_dir, f"{session_id}.db")
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                    conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
                    logger.info(f"Created persistent session: {session_id}")
                
                session_info = {
                    'session_id': session_id,
                    'db_path': db_path,
                    'connection': conn,
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now(),
                    'tables': set(),
                    'memory_only': self._use_memory_only
                }
                
                self.active_sessions[session_id] = session_info
                return session_id
                
            except Exception as e:
                logger.error(f"Error creating session {session_id}: {e}")
                raise
    
    @contextmanager
    def get_connection(self, session_id: str):
        """Get database connection for a session."""
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session_info = self.active_sessions[session_id]
            session_info['last_accessed'] = datetime.now()
            
            try:
                yield session_info['connection']
            except Exception as e:
                logger.error(f"Database error in session {session_id}: {e}")
                raise
    
    def execute_query(self, session_id: str, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        with self.get_connection(session_id) as conn:
            cursor = conn.cursor()
            
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                conn.commit()
                return results
            except sqlite3.Error as e:
                logger.error(f"SQLite error in session {session_id}: {e}")
                logger.error(f"Query: {query}")
                if params:
                    logger.error(f"Params: {params}")
                raise
    
    def store_data(self, session_id: str, table_name: str, data: List[Dict[str, Any]], 
                   replace: bool = False) -> bool:
        """Store data in session database."""
        if not data:
            return True
        
        with self.get_connection(session_id) as conn:
            try:
                # Create table if it doesn't exist
                sample_row = data[0]
                columns = list(sample_row.keys())
                
                # Create table schema with proper escaping
                column_defs = []
                for col in columns:
                    # Escape column names to handle special characters
                    escaped_col = f'"{col}"'
                    value = sample_row[col]
                    if isinstance(value, (int, float)):
                        column_defs.append(f"{escaped_col} REAL")
                    elif isinstance(value, bool):
                        column_defs.append(f"{escaped_col} INTEGER")
                    else:
                        column_defs.append(f"{escaped_col} TEXT")
                
                # Escape table name as well
                escaped_table_name = f'"{table_name}"'
                create_sql = f"CREATE TABLE IF NOT EXISTS {escaped_table_name} ({', '.join(column_defs)})"
                conn.execute(create_sql)
                
                # Clear table if replace is True
                if replace:
                    conn.execute(f"DELETE FROM {escaped_table_name}")
                
                # Insert data with escaped column names
                escaped_columns = [f'"{col}"' for col in columns]
                placeholders = ', '.join(['?' for _ in columns])
                insert_sql = f"INSERT INTO {escaped_table_name} ({', '.join(escaped_columns)}) VALUES ({placeholders})"
                
                for row in data:
                    values = []
                    for col in columns:
                        value = row.get(col)
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        values.append(value)
                    conn.execute(insert_sql, values)
                
                conn.commit()
                
                # Track table
                with self._lock:
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]['tables'].add(table_name)
                
                logger.info(f"Stored {len(data)} rows in {table_name} for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error storing data in session {session_id}: {e}")
                conn.rollback()
                return False
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        with self._lock:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session_info = self.active_sessions[session_id].copy()
            # Remove connection object for serialization
            session_info.pop('connection', None)
            session_info['tables'] = list(session_info['tables'])
            
            # Convert datetime objects to strings
            for key in ['created_at', 'last_accessed']:
                if key in session_info and isinstance(session_info[key], datetime):
                    session_info[key] = session_info[key].isoformat()
            
            return session_info
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        with self._lock:
            sessions = []
            for session_id in self.active_sessions:
                sessions.append(self.get_session_info(session_id))
            return sessions
    
    def _close_session(self, session_id: str):
        """Close a session and clean up resources."""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            
            try:
                # Close database connection
                if 'connection' in session_info:
                    session_info['connection'].close()
                
                # Remove database file (only for persistent sessions)
                if (not session_info.get('memory_only', False) and 
                    'db_path' in session_info and 
                    session_info['db_path'] != ":memory:" and
                    os.path.exists(session_info['db_path'])):
                    os.remove(session_info['db_path'])
                    
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
    
    def close_session(self, session_id: str) -> bool:
        """Manually close a session."""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            try:
                self._close_session(session_id)
                logger.info(f"Closed session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
                return False
    
    def shutdown(self):
        """Shutdown the session manager and clean up all resources."""
        logger.info("Shutting down session manager")
        self._shutdown = True
        
        with self._lock:
            # Close all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                try:
                    self._close_session(session_id)
                except Exception as e:
                    logger.error(f"Error closing session {session_id} during shutdown: {e}")
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        try:
            _session_manager = SessionManager()
        except Exception as e:
            logger.error(f"Failed to create SessionManager: {e}")
            # Create a minimal session manager that only uses memory
            _session_manager = SessionManager()
            _session_manager._use_memory_only = True
            _session_manager.sessions_dir = None
    return _session_manager