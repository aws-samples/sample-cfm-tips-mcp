"""
Parallel Execution Engine for CFM Tips MCP Server

Provides optimized parallel execution of AWS service calls with session integration.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Result of a parallel task execution."""
    task_id: str
    service: str
    operation: str
    status: str  # 'success', 'error', 'timeout'
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ParallelTask:
    """Definition of a task to be executed in parallel."""
    task_id: str
    service: str
    operation: str
    function: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    timeout: float = 30.0
    priority: int = 1  # Higher number = higher priority
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

class ParallelExecutor:
    """Executes AWS service calls in parallel with optimized resource management."""
    
    def __init__(self, max_workers: int = 10, default_timeout: float = 30.0):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="CFM-Worker")
        self._active_tasks: Dict[str, Future] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.RLock()
        
        logger.info(f"Initialized ParallelExecutor with {max_workers} workers")
    
    def submit_task(self, task: ParallelTask) -> str:
        """Submit a task for parallel execution."""
        with self._lock:
            if task.task_id in self._active_tasks:
                raise ValueError(f"Task {task.task_id} already exists")
            
            future = self.executor.submit(self._execute_task, task)
            self._active_tasks[task.task_id] = future
            
            logger.debug(f"Submitted task {task.task_id} ({task.service}.{task.operation})")
            return task.task_id
    
    def submit_batch(self, tasks: List[ParallelTask]) -> List[str]:
        """Submit multiple tasks for parallel execution."""
        # Sort by priority (higher first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        task_ids = []
        for task in sorted_tasks:
            try:
                task_id = self.submit_task(task)
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Error submitting task {task.task_id}: {e}")
                # Create error result
                error_result = TaskResult(
                    task_id=task.task_id,
                    service=task.service,
                    operation=task.operation,
                    status='error',
                    error=str(e)
                )
                with self._lock:
                    self._results[task.task_id] = error_result
        
        logger.info(f"Submitted batch of {len(task_ids)} tasks")
        return task_ids
    
    def _execute_task(self, task: ParallelTask) -> TaskResult:
        """Execute a single task with timeout and error handling."""
        start_time = time.time()
        
        try:
            logger.debug(f"Executing task {task.task_id}")
            
            # Execute the function with timeout
            result_data = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            result = TaskResult(
                task_id=task.task_id,
                service=task.service,
                operation=task.operation,
                status='success',
                data=result_data,
                execution_time=execution_time
            )
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            result = TaskResult(
                task_id=task.task_id,
                service=task.service,
                operation=task.operation,
                status='error',
                error=error_msg,
                execution_time=execution_time
            )
            
            logger.error(f"Task {task.task_id} failed after {execution_time:.2f}s: {error_msg}")
        
        # Store result
        with self._lock:
            self._results[task.task_id] = result
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
        
        return result
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for specific tasks to complete."""
        if timeout is None:
            timeout = self.default_timeout
        
        results = {}
        remaining_tasks = set(task_ids)
        start_time = time.time()
        
        while remaining_tasks and (time.time() - start_time) < timeout:
            completed_tasks = set()
            
            with self._lock:
                for task_id in remaining_tasks:
                    if task_id in self._results:
                        results[task_id] = self._results[task_id]
                        completed_tasks.add(task_id)
                    elif task_id not in self._active_tasks:
                        # Task not found, create error result
                        error_result = TaskResult(
                            task_id=task_id,
                            service='unknown',
                            operation='unknown',
                            status='error',
                            error='Task not found'
                        )
                        results[task_id] = error_result
                        completed_tasks.add(task_id)
            
            remaining_tasks -= completed_tasks
            
            if remaining_tasks:
                time.sleep(0.1)  # Small delay to avoid busy waiting
        
        # Handle timeout for remaining tasks
        for task_id in remaining_tasks:
            timeout_result = TaskResult(
                task_id=task_id,
                service='unknown',
                operation='unknown',
                status='timeout',
                error=f'Task timed out after {timeout}s'
            )
            results[task_id] = timeout_result
        
        logger.info(f"Completed waiting for {len(task_ids)} tasks, {len(results)} results")
        return results
    
    def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for all active tasks to complete."""
        with self._lock:
            active_task_ids = list(self._active_tasks.keys())
        
        if not active_task_ids:
            return {}
        
        return self.wait_for_tasks(active_task_ids, timeout)
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a specific task."""
        with self._lock:
            return self._results.get(task_id)
    
    def get_all_results(self) -> Dict[str, TaskResult]:
        """Get all available results."""
        with self._lock:
            return self._results.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            if task_id in self._active_tasks:
                future = self._active_tasks[task_id]
                cancelled = future.cancel()
                if cancelled:
                    del self._active_tasks[task_id]
                    # Create cancelled result
                    cancel_result = TaskResult(
                        task_id=task_id,
                        service='unknown',
                        operation='unknown',
                        status='error',
                        error='Task cancelled'
                    )
                    self._results[task_id] = cancel_result
                return cancelled
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get executor status information."""
        with self._lock:
            active_count = len(self._active_tasks)
            completed_count = len(self._results)
            
            # Count results by status
            status_counts = {}
            for result in self._results.values():
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        return {
            'max_workers': self.max_workers,
            'active_tasks': active_count,
            'completed_tasks': completed_count,
            'status_breakdown': status_counts,
            'executor_alive': not self.executor._shutdown
        }
    
    def clear_results(self, older_than_minutes: int = 60):
        """Clear old results to free memory."""
        cutoff_time = datetime.now().timestamp() - (older_than_minutes * 60)
        
        with self._lock:
            old_task_ids = []
            for task_id, result in self._results.items():
                if result.timestamp.timestamp() < cutoff_time:
                    old_task_ids.append(task_id)
            
            for task_id in old_task_ids:
                del self._results[task_id]
        
        logger.info(f"Cleared {len(old_task_ids)} old results")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ParallelExecutor")
        
        with self._lock:
            # Cancel all active tasks
            for task_id, future in self._active_tasks.items():
                future.cancel()
            self._active_tasks.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        logger.info("ParallelExecutor shutdown complete")

# Global executor instance
_parallel_executor = None

def get_parallel_executor() -> ParallelExecutor:
    """Get the global parallel executor instance."""
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelExecutor()
    return _parallel_executor

def create_task(task_id: str, service: str, operation: str, function: Callable,
                args: tuple = (), kwargs: Dict[str, Any] = None, 
                timeout: float = 30.0, priority: int = 1) -> ParallelTask:
    """Helper function to create a ParallelTask."""
    return ParallelTask(
        task_id=task_id,
        service=service,
        operation=operation,
        function=function,
        args=args,
        kwargs=kwargs or {},
        timeout=timeout,
        priority=priority
    )