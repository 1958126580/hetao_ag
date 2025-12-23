"""
Core Computational Module for SmartAgri

This module provides the fundamental computational infrastructure including:
- GPU acceleration using CUDA/OpenCL
- Parallel computing with multiprocessing and threading
- Optimized matrix algebra operations
- Numerical optimization algorithms
- Statistical computing primitives

The core module is designed for high-performance computing in agricultural
and animal husbandry applications, supporting both CPU and GPU execution.
"""

from smartagri.core.gpu import (
    GPUAccelerator,
    GPUContext,
    GPUMemoryPool,
    is_gpu_available,
    get_gpu_info,
    synchronize_gpu,
)

from smartagri.core.parallel import (
    ParallelExecutor,
    ThreadPoolManager,
    ProcessPoolManager,
    distribute_workload,
    parallel_map,
    async_executor,
)

from smartagri.core.matrix import (
    MatrixOps,
    SparseMatrix,
    TensorOps,
    eigendecomposition,
    svd_decomposition,
    qr_decomposition,
    cholesky_decomposition,
)

from smartagri.core.optimization import (
    GradientDescent,
    AdamOptimizer,
    LBFGSOptimizer,
    GeneticAlgorithm,
    ParticleSwarmOptimizer,
    SimulatedAnnealing,
)

from smartagri.core.statistics import (
    DescriptiveStats,
    BayesianInference,
    HypothesisTesting,
    TimeSeriesAnalysis,
    SpatialStatistics,
)

__all__ = [
    # GPU acceleration
    "GPUAccelerator",
    "GPUContext",
    "GPUMemoryPool",
    "is_gpu_available",
    "get_gpu_info",
    "synchronize_gpu",
    # Parallel computing
    "ParallelExecutor",
    "ThreadPoolManager",
    "ProcessPoolManager",
    "distribute_workload",
    "parallel_map",
    "async_executor",
    # Matrix operations
    "MatrixOps",
    "SparseMatrix",
    "TensorOps",
    "eigendecomposition",
    "svd_decomposition",
    "qr_decomposition",
    "cholesky_decomposition",
    # Optimization
    "GradientDescent",
    "AdamOptimizer",
    "LBFGSOptimizer",
    "GeneticAlgorithm",
    "ParticleSwarmOptimizer",
    "SimulatedAnnealing",
    # Statistics
    "DescriptiveStats",
    "BayesianInference",
    "HypothesisTesting",
    "TimeSeriesAnalysis",
    "SpatialStatistics",
]
