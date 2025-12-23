"""
Parallel Computing Module for SmartAgri

This module provides high-performance parallel computing capabilities
for agricultural data processing, including:
- Multi-threaded execution for I/O-bound tasks
- Multi-process execution for CPU-bound tasks
- Asynchronous task execution
- Work distribution and load balancing
- MapReduce-style operations

Features:
    - Automatic workload distribution across CPU cores
    - Thread pool management with dynamic sizing
    - Process pool with shared memory support
    - Futures-based async execution
    - Progress tracking and cancellation support

Example:
    >>> from smartagri.core.parallel import ParallelExecutor, parallel_map
    >>> # Process satellite images in parallel
    >>> executor = ParallelExecutor(n_workers=8)
    >>> results = executor.map(process_image, image_files)
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Callable,
    Any,
    Dict,
    TypeVar,
    Generic,
    Iterable,
    Iterator,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
    wait,
    FIRST_COMPLETED,
    ALL_COMPLETED,
)
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import asyncio
import queue
import time
import logging
from functools import partial
from abc import ABC, abstractmethod
import os

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")
ArrayLike = Union[np.ndarray, List, Tuple]


class ExecutorType(Enum):
    """Types of parallel executors available."""

    THREAD = auto()  # Thread-based parallelism
    PROCESS = auto()  # Process-based parallelism
    ASYNC = auto()  # Asyncio-based parallelism


@dataclass
class TaskResult(Generic[T]):
    """
    Container for parallel task results.

    Attributes:
        task_id: Unique identifier for the task
        result: Task result value
        success: Whether task completed successfully
        error: Exception if task failed
        execution_time: Time taken to execute in seconds
        worker_id: ID of worker that executed the task
    """

    task_id: int
    result: Optional[T]
    success: bool
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: int = -1


@dataclass
class WorkloadPartition:
    """
    Represents a partition of work for parallel execution.

    Attributes:
        partition_id: Unique identifier for the partition
        start_index: Starting index in the data
        end_index: Ending index in the data
        data_slice: Slice of data for this partition
        weight: Relative workload weight
    """

    partition_id: int
    start_index: int
    end_index: int
    data_slice: Any
    weight: float = 1.0

    @property
    def size(self) -> int:
        """Number of elements in this partition."""
        return self.end_index - self.start_index


class WorkloadBalancer:
    """
    Distributes workloads across workers with load balancing.

    Supports various partitioning strategies:
    - Equal partitioning: Divide work equally
    - Weighted partitioning: Based on estimated complexity
    - Dynamic partitioning: Adaptive based on worker performance

    Example:
        >>> balancer = WorkloadBalancer(n_workers=4)
        >>> partitions = balancer.partition(data, strategy='equal')
    """

    def __init__(self, n_workers: int):
        """
        Initialize workload balancer.

        Args:
            n_workers: Number of workers to distribute work to
        """
        self.n_workers = n_workers
        self._worker_speeds: List[float] = [1.0] * n_workers
        self._lock = threading.Lock()

    def partition(
        self,
        data: ArrayLike,
        strategy: str = "equal",
        weights: Optional[List[float]] = None,
    ) -> List[WorkloadPartition]:
        """
        Partition data across workers.

        Args:
            data: Data to partition
            strategy: Partitioning strategy ('equal', 'weighted', 'adaptive')
            weights: Custom weights for each partition (weighted strategy)

        Returns:
            List of WorkloadPartition objects
        """
        n = len(data)

        if strategy == "equal":
            return self._equal_partition(data, n)
        elif strategy == "weighted" and weights is not None:
            return self._weighted_partition(data, n, weights)
        elif strategy == "adaptive":
            return self._adaptive_partition(data, n)
        else:
            return self._equal_partition(data, n)

    def _equal_partition(
        self, data: ArrayLike, n: int
    ) -> List[WorkloadPartition]:
        """Create equal-sized partitions."""
        partitions = []
        chunk_size = (n + self.n_workers - 1) // self.n_workers

        for i in range(self.n_workers):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)

            if start >= n:
                break

            partitions.append(
                WorkloadPartition(
                    partition_id=i,
                    start_index=start,
                    end_index=end,
                    data_slice=data[start:end] if hasattr(data, "__getitem__") else None,
                )
            )

        return partitions

    def _weighted_partition(
        self, data: ArrayLike, n: int, weights: List[float]
    ) -> List[WorkloadPartition]:
        """Create partitions based on complexity weights."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        partitions = []
        current_pos = 0

        for i, weight in enumerate(normalized_weights):
            size = int(n * weight)
            end = min(current_pos + size, n)

            if i == len(normalized_weights) - 1:
                end = n  # Ensure we cover all elements

            if current_pos >= n:
                break

            partitions.append(
                WorkloadPartition(
                    partition_id=i,
                    start_index=current_pos,
                    end_index=end,
                    data_slice=data[current_pos:end] if hasattr(data, "__getitem__") else None,
                    weight=weight,
                )
            )
            current_pos = end

        return partitions

    def _adaptive_partition(
        self, data: ArrayLike, n: int
    ) -> List[WorkloadPartition]:
        """Create partitions based on worker performance history."""
        with self._lock:
            total_speed = sum(self._worker_speeds)
            normalized_speeds = [s / total_speed for s in self._worker_speeds]

        return self._weighted_partition(data, n, normalized_speeds)

    def update_worker_speed(self, worker_id: int, execution_time: float, work_size: int) -> None:
        """
        Update worker speed estimation based on execution metrics.

        Args:
            worker_id: ID of the worker
            execution_time: Time taken for execution
            work_size: Amount of work processed
        """
        if execution_time > 0 and work_size > 0:
            speed = work_size / execution_time
            with self._lock:
                # Exponential moving average
                alpha = 0.3
                self._worker_speeds[worker_id] = (
                    alpha * speed + (1 - alpha) * self._worker_speeds[worker_id]
                )


class ThreadPoolManager:
    """
    Manages a pool of threads for parallel I/O-bound operations.

    Ideal for tasks like:
    - Reading multiple sensor data files
    - Fetching weather data from APIs
    - Database queries
    - Network operations

    Example:
        >>> manager = ThreadPoolManager(max_workers=16)
        >>> futures = [manager.submit(fetch_data, url) for url in urls]
        >>> results = manager.gather(futures)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "smartagri_thread",
    ):
        """
        Initialize thread pool manager.

        Args:
            max_workers: Maximum number of threads (default: CPU count * 5)
            thread_name_prefix: Prefix for thread names
        """
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 5)

        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._active_futures: Dict[int, Future] = {}
        self._future_counter = 0
        self._lock = threading.Lock()

        logger.info(f"ThreadPoolManager initialized with {max_workers} workers")

    def submit(
        self,
        fn: Callable[..., T],
        *args,
        **kwargs,
    ) -> Future:
        """
        Submit a task for execution.

        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Future representing the pending result
        """
        future = self._executor.submit(fn, *args, **kwargs)

        with self._lock:
            self._future_counter += 1
            self._active_futures[self._future_counter] = future

        return future

    def map(
        self,
        fn: Callable[[T], R],
        iterable: Iterable[T],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> Iterator[R]:
        """
        Map function over iterable in parallel.

        Args:
            fn: Function to apply to each element
            iterable: Input iterable
            timeout: Maximum time to wait for results
            chunksize: Number of items per thread

        Returns:
            Iterator of results in order
        """
        return self._executor.map(fn, iterable, timeout=timeout, chunksize=chunksize)

    def gather(
        self,
        futures: List[Future],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        Gather results from multiple futures.

        Args:
            futures: List of futures to gather
            timeout: Maximum time to wait
            return_exceptions: Include exceptions in results

        Returns:
            List of results in order
        """
        results = []

        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.

        Args:
            wait: Wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)
        logger.info("ThreadPoolManager shutdown complete")

    def __enter__(self) -> "ThreadPoolManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


class ProcessPoolManager:
    """
    Manages a pool of processes for parallel CPU-bound operations.

    Ideal for computationally intensive tasks like:
    - Image processing and analysis
    - Machine learning model training
    - Numerical simulations
    - Large-scale data transformations

    Supports shared memory for efficient data transfer between processes.

    Example:
        >>> manager = ProcessPoolManager(max_workers=8)
        >>> with manager:
        ...     results = manager.map(process_image, images)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_shared_memory: bool = True,
    ):
        """
        Initialize process pool manager.

        Args:
            max_workers: Maximum number of processes (default: CPU count)
            use_shared_memory: Enable shared memory for large arrays
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 1

        self.max_workers = max_workers
        self.use_shared_memory = use_shared_memory
        self._executor: Optional[ProcessPoolExecutor] = None
        self._shared_arrays: Dict[str, shared_memory.SharedMemory] = {}
        self._lock = threading.Lock()

        logger.info(f"ProcessPoolManager initialized with {max_workers} workers")

    def _ensure_executor(self) -> None:
        """Ensure process pool executor is initialized."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def submit(
        self,
        fn: Callable[..., T],
        *args,
        **kwargs,
    ) -> Future:
        """
        Submit a task for execution.

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future representing the pending result
        """
        self._ensure_executor()
        return self._executor.submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[[T], R],
        iterable: Iterable[T],
        chunksize: int = 1,
    ) -> Iterator[R]:
        """
        Map function over iterable using process pool.

        Args:
            fn: Function to apply
            iterable: Input iterable
            chunksize: Number of items per process

        Returns:
            Iterator of results
        """
        self._ensure_executor()
        return self._executor.map(fn, iterable, chunksize=chunksize)

    def create_shared_array(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """
        Create a shared memory array accessible by all processes.

        Args:
            name: Unique name for the shared array
            shape: Shape of the array
            dtype: Data type

        Returns:
            NumPy array backed by shared memory
        """
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize

        shm = shared_memory.SharedMemory(create=True, size=size)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        with self._lock:
            self._shared_arrays[name] = shm

        logger.debug(f"Created shared array '{name}' with shape {shape}")
        return arr

    def get_shared_array(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """
        Access an existing shared memory array.

        Args:
            name: Name of the shared array
            shape: Shape of the array
            dtype: Data type

        Returns:
            NumPy array view of shared memory
        """
        shm = shared_memory.SharedMemory(name=name)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def cleanup_shared_memory(self) -> None:
        """Clean up all shared memory allocations."""
        with self._lock:
            for name, shm in self._shared_arrays.items():
                try:
                    shm.close()
                    shm.unlink()
                    logger.debug(f"Cleaned up shared array '{name}'")
                except Exception as e:
                    logger.warning(f"Error cleaning up shared array '{name}': {e}")
            self._shared_arrays.clear()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the process pool.

        Args:
            wait: Wait for pending tasks
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
        self.cleanup_shared_memory()
        logger.info("ProcessPoolManager shutdown complete")

    def __enter__(self) -> "ProcessPoolManager":
        self._ensure_executor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


class ParallelExecutor:
    """
    High-level interface for parallel execution of agricultural computations.

    Automatically selects the best execution strategy based on task
    characteristics and provides a unified interface for both thread
    and process-based parallelism.

    Features:
        - Automatic strategy selection
        - Load balancing across workers
        - Progress tracking
        - Error handling and retry logic
        - Resource management

    Example:
        >>> executor = ParallelExecutor(n_workers=8)
        >>> # Process crop images in parallel
        >>> results = executor.map(analyze_crop, images)
        >>> # Execute with progress tracking
        >>> for result in executor.imap(process_field, fields, progress=True):
        ...     print(f"Processed: {result}")
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        executor_type: ExecutorType = ExecutorType.PROCESS,
        max_retries: int = 3,
    ):
        """
        Initialize parallel executor.

        Args:
            n_workers: Number of workers (default: CPU count)
            executor_type: Type of executor (THREAD, PROCESS, or ASYNC)
            max_retries: Maximum retry attempts for failed tasks
        """
        if n_workers is None:
            n_workers = os.cpu_count() or 1

        self.n_workers = n_workers
        self.executor_type = executor_type
        self.max_retries = max_retries

        self._balancer = WorkloadBalancer(n_workers)
        self._thread_pool: Optional[ThreadPoolManager] = None
        self._process_pool: Optional[ProcessPoolManager] = None
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        }
        self._lock = threading.Lock()

        logger.info(
            f"ParallelExecutor initialized with {n_workers} workers "
            f"using {executor_type.name} strategy"
        )

    def _get_executor(self) -> Union[ThreadPoolManager, ProcessPoolManager]:
        """Get or create the appropriate executor."""
        if self.executor_type == ExecutorType.THREAD:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolManager(max_workers=self.n_workers)
            return self._thread_pool
        else:
            if self._process_pool is None:
                self._process_pool = ProcessPoolManager(max_workers=self.n_workers)
            return self._process_pool

    def map(
        self,
        fn: Callable[[T], R],
        iterable: Iterable[T],
        chunksize: int = 1,
    ) -> List[R]:
        """
        Apply function to each element in parallel.

        Args:
            fn: Function to apply
            iterable: Input iterable
            chunksize: Number of items per worker

        Returns:
            List of results in order
        """
        items = list(iterable)

        with self._lock:
            self._stats["tasks_submitted"] += len(items)

        start_time = time.time()
        executor = self._get_executor()
        results = list(executor.map(fn, items, chunksize=chunksize))

        with self._lock:
            self._stats["tasks_completed"] += len(results)
            self._stats["total_execution_time"] += time.time() - start_time

        return results

    def imap(
        self,
        fn: Callable[[T], R],
        iterable: Iterable[T],
        progress: bool = False,
    ) -> Iterator[TaskResult[R]]:
        """
        Lazy parallel map with optional progress tracking.

        Args:
            fn: Function to apply
            iterable: Input iterable
            progress: Show progress information

        Yields:
            TaskResult objects containing results
        """
        items = list(iterable)
        n_items = len(items)
        executor = self._get_executor()

        # Submit all tasks
        futures = {}
        for i, item in enumerate(items):
            future = executor.submit(fn, item)
            futures[future] = i

        # Yield results as they complete
        completed = 0
        for future in as_completed(futures.keys()):
            task_id = futures[future]
            start_time = time.time()

            try:
                result = future.result()
                yield TaskResult(
                    task_id=task_id,
                    result=result,
                    success=True,
                    execution_time=time.time() - start_time,
                )
            except Exception as e:
                yield TaskResult(
                    task_id=task_id,
                    result=None,
                    success=False,
                    error=e,
                    execution_time=time.time() - start_time,
                )

            completed += 1
            if progress:
                pct = completed / n_items * 100
                logger.info(f"Progress: {completed}/{n_items} ({pct:.1f}%)")

    def starmap(
        self,
        fn: Callable[..., R],
        iterable: Iterable[Tuple],
    ) -> List[R]:
        """
        Parallel starmap - apply function to argument tuples.

        Args:
            fn: Function to apply
            iterable: Iterable of argument tuples

        Returns:
            List of results
        """
        items = list(iterable)

        def wrapper(args):
            return fn(*args)

        return self.map(wrapper, items)

    def reduce(
        self,
        fn: Callable[[R, R], R],
        data: ArrayLike,
        initial: Optional[R] = None,
    ) -> R:
        """
        Parallel reduction operation.

        Performs tree-based parallel reduction for associative operations.

        Args:
            fn: Binary reduction function (must be associative)
            data: Input data array
            initial: Initial value for reduction

        Returns:
            Reduced result
        """
        if len(data) == 0:
            if initial is not None:
                return initial
            raise ValueError("Cannot reduce empty data without initial value")

        # Partition data for parallel reduction
        partitions = self._balancer.partition(data, strategy="equal")

        # First pass: reduce within partitions
        def reduce_partition(partition: WorkloadPartition) -> Any:
            from functools import reduce as functools_reduce

            return functools_reduce(fn, partition.data_slice)

        partial_results = self.map(reduce_partition, partitions)

        # Final reduction of partial results
        from functools import reduce as functools_reduce

        if initial is not None:
            return functools_reduce(fn, partial_results, initial)
        return functools_reduce(fn, partial_results)

    def filter(
        self,
        fn: Callable[[T], bool],
        iterable: Iterable[T],
    ) -> List[T]:
        """
        Parallel filter operation.

        Args:
            fn: Predicate function
            iterable: Input iterable

        Returns:
            Filtered list
        """
        items = list(iterable)

        # Apply predicate in parallel
        masks = self.map(fn, items)

        # Collect matching items
        return [item for item, keep in zip(items, masks) if keep]

    def batch_process(
        self,
        fn: Callable[[List[T]], List[R]],
        data: List[T],
        batch_size: int = 100,
    ) -> List[R]:
        """
        Process data in batches across workers.

        Args:
            fn: Function that processes a batch
            data: Input data
            batch_size: Size of each batch

        Returns:
            Combined results from all batches
        """
        # Create batches
        batches = [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]

        # Process batches in parallel
        batch_results = self.map(fn, batches)

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            return {
                **self._stats,
                "avg_time_per_task": (
                    self._stats["total_execution_time"] / self._stats["tasks_completed"]
                    if self._stats["tasks_completed"] > 0
                    else 0
                ),
                "success_rate": (
                    self._stats["tasks_completed"]
                    / (self._stats["tasks_completed"] + self._stats["tasks_failed"])
                    if self._stats["tasks_submitted"] > 0
                    else 0
                ),
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and release resources."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=wait)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=wait)
        logger.info("ParallelExecutor shutdown complete")

    def __enter__(self) -> "ParallelExecutor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


# Convenience functions


def distribute_workload(
    data: ArrayLike,
    n_workers: int,
    strategy: str = "equal",
) -> List[WorkloadPartition]:
    """
    Distribute data across workers.

    Args:
        data: Data to distribute
        n_workers: Number of workers
        strategy: Distribution strategy

    Returns:
        List of WorkloadPartition objects
    """
    balancer = WorkloadBalancer(n_workers)
    return balancer.partition(data, strategy=strategy)


def parallel_map(
    fn: Callable[[T], R],
    iterable: Iterable[T],
    n_workers: Optional[int] = None,
    use_threads: bool = False,
) -> List[R]:
    """
    Apply function in parallel across workers.

    Convenience function for simple parallel mapping operations.

    Args:
        fn: Function to apply
        iterable: Input iterable
        n_workers: Number of workers (default: CPU count)
        use_threads: Use threads instead of processes

    Returns:
        List of results

    Example:
        >>> def process_field(field):
        ...     return calculate_ndvi(field)
        >>> ndvi_values = parallel_map(process_field, fields)
    """
    executor_type = ExecutorType.THREAD if use_threads else ExecutorType.PROCESS

    with ParallelExecutor(n_workers=n_workers, executor_type=executor_type) as executor:
        return executor.map(fn, iterable)


async def async_executor(
    fn: Callable[[T], R],
    items: List[T],
    max_concurrent: int = 10,
) -> List[R]:
    """
    Async executor for I/O-bound operations.

    Args:
        fn: Async function to apply
        items: Input items
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(fn):
                return await fn(item)
            return fn(item)

    tasks = [bounded_task(item) for item in items]
    return await asyncio.gather(*tasks)
