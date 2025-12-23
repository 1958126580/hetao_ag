"""
Matrix Algebra Module for SmartAgri

This module provides high-performance matrix operations optimized for
agricultural and animal husbandry computations, including:
- Dense and sparse matrix operations
- Matrix decompositions (SVD, QR, Cholesky, Eigendecomposition)
- Tensor operations for multi-dimensional data
- GPU-accelerated linear algebra

Applications in Smart Agriculture:
    - Yield prediction models (linear regression, PCA)
    - Spatial interpolation (kriging matrices)
    - Image processing (convolutions, transformations)
    - Time series analysis (covariance matrices)
    - Machine learning (kernel methods, neural networks)

Example:
    >>> from smartagri.core.matrix import MatrixOps, svd_decomposition
    >>> ops = MatrixOps(use_gpu=True)
    >>> # Solve linear system for crop yield model
    >>> coefficients = ops.solve(X, y)
    >>> # Perform PCA on sensor data
    >>> U, S, Vt = svd_decomposition(sensor_data)
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
)
from dataclasses import dataclass, field
from enum import Enum, auto
import scipy.sparse as sp
from scipy import linalg
import warnings
import logging
from abc import ABC, abstractmethod

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, List, Tuple]
SparseMatrix = Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]


class MatrixFormat(Enum):
    """Sparse matrix storage formats."""

    CSR = auto()  # Compressed Sparse Row
    CSC = auto()  # Compressed Sparse Column
    COO = auto()  # Coordinate format
    DENSE = auto()  # Dense matrix


@dataclass
class DecompositionResult:
    """
    Container for matrix decomposition results.

    Attributes:
        matrices: Dict of resulting matrices
        info: Additional information about the decomposition
        condition_number: Condition number of the matrix
        rank: Numerical rank of the matrix
    """

    matrices: Dict[str, np.ndarray]
    info: Dict[str, Any] = field(default_factory=dict)
    condition_number: Optional[float] = None
    rank: Optional[int] = None


class MatrixOps:
    """
    High-performance matrix operations for agricultural computations.

    Provides optimized implementations of common linear algebra operations
    with automatic GPU acceleration when available.

    Features:
        - BLAS-optimized matrix multiplication
        - Batch matrix operations
        - Automatic precision handling
        - GPU acceleration support
        - Memory-efficient sparse operations

    Example:
        >>> ops = MatrixOps(use_gpu=True, precision='float64')
        >>> # Matrix multiplication for yield model
        >>> predictions = ops.matmul(features, weights)
        >>> # Solve least squares problem
        >>> coefficients = ops.lstsq(X, y)
    """

    def __init__(
        self,
        use_gpu: bool = False,
        precision: str = "float64",
        num_threads: Optional[int] = None,
    ):
        """
        Initialize matrix operations.

        Args:
            use_gpu: Enable GPU acceleration
            precision: Floating point precision ('float32' or 'float64')
            num_threads: Number of threads for BLAS operations
        """
        self.use_gpu = use_gpu
        self.precision = np.dtype(precision)
        self.num_threads = num_threads

        self._gpu_accelerator = None
        if use_gpu:
            try:
                from smartagri.core.gpu import GPUAccelerator

                self._gpu_accelerator = GPUAccelerator()
                logger.info("GPU acceleration enabled for matrix operations")
            except ImportError:
                logger.warning("GPU acceleration not available")
                self.use_gpu = False

    def _ensure_array(self, x: ArrayLike) -> np.ndarray:
        """Convert input to numpy array with correct precision."""
        arr = np.asarray(x, dtype=self.precision)
        return np.ascontiguousarray(arr)

    def matmul(
        self,
        a: ArrayLike,
        b: ArrayLike,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Matrix multiplication C = A @ B.

        Optimized matrix multiplication using BLAS or GPU.

        Args:
            a: First matrix (M x K)
            b: Second matrix (K x N)
            out: Optional output array

        Returns:
            Product matrix (M x N)

        Example:
            >>> ops = MatrixOps()
            >>> X = np.random.randn(1000, 50)  # Feature matrix
            >>> W = np.random.randn(50, 10)   # Weights
            >>> predictions = ops.matmul(X, W)
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)

        if self.use_gpu and self._gpu_accelerator:
            return self._gpu_accelerator.matmul(a, b)

        if out is not None:
            return np.matmul(a, b, out=out)
        return np.matmul(a, b)

    def batch_matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Batched matrix multiplication.

        Performs matrix multiplication on batches of matrices.
        Useful for processing multiple fields or time steps.

        Args:
            a: Batch of matrices (batch_size, M, K)
            b: Batch of matrices (batch_size, K, N)

        Returns:
            Batch of products (batch_size, M, N)

        Example:
            >>> # Process multiple field matrices
            >>> X_batch = np.random.randn(100, 50, 20)  # 100 fields
            >>> W = np.random.randn(100, 20, 5)
            >>> results = ops.batch_matmul(X_batch, W)
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)

        return np.matmul(a, b)

    def dot(self, a: ArrayLike, b: ArrayLike) -> Union[np.ndarray, float]:
        """
        Dot product of two arrays.

        Args:
            a: First array
            b: Second array

        Returns:
            Dot product result
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)
        return np.dot(a, b)

    def outer(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """
        Outer product of two vectors.

        Args:
            a: First vector (M,)
            b: Second vector (N,)

        Returns:
            Outer product matrix (M x N)
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)
        return np.outer(a, b)

    def transpose(
        self,
        x: ArrayLike,
        axes: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """
        Transpose matrix or permute tensor axes.

        Args:
            x: Input array
            axes: Optional permutation of axes

        Returns:
            Transposed array
        """
        x = self._ensure_array(x)
        return np.transpose(x, axes=axes)

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """
        Compute matrix inverse.

        Args:
            x: Square matrix

        Returns:
            Inverse matrix

        Raises:
            LinAlgError: If matrix is singular
        """
        x = self._ensure_array(x)

        if x.shape[0] != x.shape[1]:
            raise ValueError("Matrix must be square for inversion")

        return linalg.inv(x)

    def pinv(
        self,
        x: ArrayLike,
        rcond: float = 1e-15,
    ) -> np.ndarray:
        """
        Moore-Penrose pseudoinverse.

        Computes the pseudoinverse using SVD, useful for
        solving overdetermined or underdetermined systems.

        Args:
            x: Input matrix
            rcond: Cutoff for small singular values

        Returns:
            Pseudoinverse matrix
        """
        x = self._ensure_array(x)
        return linalg.pinv(x, rcond=rcond)

    def solve(
        self,
        a: ArrayLike,
        b: ArrayLike,
        assume_a: str = "gen",
    ) -> np.ndarray:
        """
        Solve linear system Ax = b.

        Args:
            a: Coefficient matrix (N x N)
            b: Right-hand side (N,) or (N x M)
            assume_a: Matrix type ('gen', 'sym', 'pos', 'her')

        Returns:
            Solution x

        Example:
            >>> # Solve normal equations for linear regression
            >>> X = np.random.randn(100, 5)
            >>> y = np.random.randn(100)
            >>> XtX = ops.matmul(X.T, X)
            >>> Xty = ops.matmul(X.T, y)
            >>> beta = ops.solve(XtX, Xty)
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)

        return linalg.solve(a, b, assume_a=assume_a)

    def lstsq(
        self,
        a: ArrayLike,
        b: ArrayLike,
        rcond: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """
        Least squares solution to overdetermined system.

        Solves min ||Ax - b||_2 using SVD decomposition.
        Commonly used for linear regression in yield modeling.

        Args:
            a: Design matrix (M x N)
            b: Observation vector (M,)
            rcond: Cutoff for singular values

        Returns:
            Tuple of (solution, residuals, rank, singular_values)

        Example:
            >>> # Fit yield model: yield = b0 + b1*temp + b2*rain + b3*soil
            >>> X = np.column_stack([np.ones(n), temp, rain, soil])
            >>> beta, residuals, rank, s = ops.lstsq(X, yields)
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)

        result = linalg.lstsq(a, b, cond=rcond)
        return result

    def qr(
        self,
        x: ArrayLike,
        mode: str = "reduced",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition.

        Factorizes A = QR where Q is orthogonal and R is upper triangular.

        Args:
            x: Input matrix (M x N)
            mode: 'reduced' (default), 'complete', 'r', or 'economic'

        Returns:
            Tuple of (Q, R) matrices
        """
        x = self._ensure_array(x)
        return linalg.qr(x, mode=mode)

    def cholesky(
        self,
        x: ArrayLike,
        lower: bool = True,
    ) -> np.ndarray:
        """
        Cholesky decomposition for positive definite matrices.

        Factorizes A = LL^T (or A = U^TU). Useful for solving
        systems involving covariance matrices.

        Args:
            x: Symmetric positive definite matrix
            lower: Return lower triangular factor

        Returns:
            Cholesky factor

        Example:
            >>> # Solve system with covariance matrix
            >>> L = ops.cholesky(covariance_matrix)
            >>> x = ops.solve_triangular(L, b, lower=True)
        """
        x = self._ensure_array(x)
        return linalg.cholesky(x, lower=lower)

    def eig(
        self,
        x: ArrayLike,
        compute_left: bool = False,
        compute_right: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Eigenvalue decomposition.

        Args:
            x: Square matrix
            compute_left: Compute left eigenvectors
            compute_right: Compute right eigenvectors

        Returns:
            Tuple of (eigenvalues, eigenvectors)

        Example:
            >>> # PCA via eigendecomposition of covariance
            >>> cov = np.cov(data.T)
            >>> eigenvalues, eigenvectors = ops.eig(cov)
        """
        x = self._ensure_array(x)

        left = "N" if not compute_left else "V"
        right = "V" if compute_right else "N"

        if compute_left or compute_right:
            return linalg.eig(x, left=compute_left, right=compute_right)
        return linalg.eigvals(x), None

    def eigh(
        self,
        x: ArrayLike,
        subset_by_index: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition for symmetric/Hermitian matrices.

        More efficient and numerically stable than general eig
        for symmetric matrices like covariance matrices.

        Args:
            x: Symmetric matrix
            subset_by_index: Only compute eigenvalues in range [lo, hi]

        Returns:
            Tuple of (eigenvalues, eigenvectors) in ascending order
        """
        x = self._ensure_array(x)

        if subset_by_index is not None:
            return linalg.eigh(
                x,
                subset_by_index=subset_by_index,
            )
        return linalg.eigh(x)

    def svd(
        self,
        x: ArrayLike,
        full_matrices: bool = False,
        compute_uv: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Singular Value Decomposition.

        Factorizes A = U @ diag(S) @ V^T. Fundamental for PCA,
        dimensionality reduction, and matrix approximation.

        Args:
            x: Input matrix (M x N)
            full_matrices: Return full U and Vt matrices
            compute_uv: Compute U and V^T (if False, only singular values)

        Returns:
            If compute_uv: (U, S, Vt) where A = U @ diag(S) @ Vt
            Otherwise: S (singular values only)

        Example:
            >>> # Low-rank approximation of satellite image
            >>> U, S, Vt = ops.svd(image)
            >>> k = 50  # Keep top 50 components
            >>> compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        """
        x = self._ensure_array(x)

        if compute_uv:
            return linalg.svd(x, full_matrices=full_matrices)
        return linalg.svdvals(x)

    def norm(
        self,
        x: ArrayLike,
        ord: Optional[Union[int, float, str]] = None,
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Matrix or vector norm.

        Args:
            x: Input array
            ord: Order of norm (1, 2, inf, 'fro', 'nuc', etc.)
            axis: Axis along which to compute norm

        Returns:
            Norm value(s)
        """
        x = self._ensure_array(x)
        return linalg.norm(x, ord=ord, axis=axis)

    def cond(self, x: ArrayLike, p: int = 2) -> float:
        """
        Condition number of a matrix.

        Measures sensitivity to numerical errors. High condition
        number indicates ill-conditioned matrix.

        Args:
            x: Input matrix
            p: Norm type (1, 2, inf, or 'fro')

        Returns:
            Condition number
        """
        x = self._ensure_array(x)
        return np.linalg.cond(x, p)

    def rank(
        self,
        x: ArrayLike,
        tol: Optional[float] = None,
    ) -> int:
        """
        Numerical rank of a matrix.

        Args:
            x: Input matrix
            tol: Tolerance for singular values

        Returns:
            Numerical rank
        """
        x = self._ensure_array(x)
        return np.linalg.matrix_rank(x, tol=tol)

    def det(self, x: ArrayLike) -> float:
        """
        Matrix determinant.

        Args:
            x: Square matrix

        Returns:
            Determinant value
        """
        x = self._ensure_array(x)
        return linalg.det(x)

    def trace(self, x: ArrayLike) -> float:
        """
        Sum of diagonal elements.

        Args:
            x: Input matrix

        Returns:
            Trace value
        """
        x = self._ensure_array(x)
        return np.trace(x)

    def diag(
        self,
        x: ArrayLike,
        k: int = 0,
    ) -> np.ndarray:
        """
        Extract diagonal or create diagonal matrix.

        Args:
            x: Input array (1D creates matrix, 2D extracts diagonal)
            k: Diagonal offset (0 = main diagonal)

        Returns:
            Diagonal elements or diagonal matrix
        """
        x = self._ensure_array(x)
        return np.diag(x, k=k)

    def eye(self, n: int, m: Optional[int] = None) -> np.ndarray:
        """
        Identity matrix.

        Args:
            n: Number of rows
            m: Number of columns (default: n)

        Returns:
            Identity matrix
        """
        return np.eye(n, m, dtype=self.precision)

    def kron(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """
        Kronecker product.

        Args:
            a: First matrix
            b: Second matrix

        Returns:
            Kronecker product
        """
        a = self._ensure_array(a)
        b = self._ensure_array(b)
        return np.kron(a, b)

    def block_diag(self, *matrices: ArrayLike) -> np.ndarray:
        """
        Create block diagonal matrix.

        Args:
            *matrices: Matrices to place on diagonal

        Returns:
            Block diagonal matrix
        """
        arrays = [self._ensure_array(m) for m in matrices]
        return linalg.block_diag(*arrays)

    def expm(self, x: ArrayLike) -> np.ndarray:
        """
        Matrix exponential.

        Args:
            x: Square matrix

        Returns:
            Matrix exponential exp(X)
        """
        x = self._ensure_array(x)
        return linalg.expm(x)

    def logm(self, x: ArrayLike) -> np.ndarray:
        """
        Matrix logarithm.

        Args:
            x: Square matrix with no negative eigenvalues

        Returns:
            Matrix logarithm log(X)
        """
        x = self._ensure_array(x)
        return linalg.logm(x)


class SparseMatrixOps:
    """
    Operations on sparse matrices for large-scale agricultural data.

    Optimized for matrices with mostly zero elements, common in:
    - Graph-based field connectivity
    - Sensor network adjacency matrices
    - Spatial weight matrices for kriging
    - Feature extraction from high-dimensional data

    Example:
        >>> sparse_ops = SparseMatrixOps()
        >>> # Create sparse adjacency matrix
        >>> adj = sparse_ops.from_coordinates(row, col, data, shape)
        >>> # Sparse matrix-vector product
        >>> result = sparse_ops.matvec(adj, vector)
    """

    def __init__(self, format: MatrixFormat = MatrixFormat.CSR):
        """
        Initialize sparse matrix operations.

        Args:
            format: Default sparse matrix format
        """
        self.default_format = format

    def from_dense(
        self,
        x: np.ndarray,
        threshold: float = 0.0,
    ) -> sp.csr_matrix:
        """
        Convert dense matrix to sparse format.

        Args:
            x: Dense matrix
            threshold: Values below threshold become zero

        Returns:
            Sparse matrix in CSR format
        """
        if threshold > 0:
            x = x.copy()
            x[np.abs(x) < threshold] = 0
        return sp.csr_matrix(x)

    def from_coordinates(
        self,
        row: np.ndarray,
        col: np.ndarray,
        data: np.ndarray,
        shape: Tuple[int, int],
    ) -> sp.csr_matrix:
        """
        Create sparse matrix from coordinate lists.

        Args:
            row: Row indices
            col: Column indices
            data: Values at (row, col)
            shape: Matrix shape

        Returns:
            Sparse matrix
        """
        coo = sp.coo_matrix((data, (row, col)), shape=shape)
        return coo.tocsr()

    def from_diagonals(
        self,
        diagonals: List[np.ndarray],
        offsets: List[int],
        shape: Tuple[int, int],
    ) -> sp.dia_matrix:
        """
        Create sparse matrix from diagonals.

        Useful for banded matrices in differential equations.

        Args:
            diagonals: List of diagonal arrays
            offsets: Offset of each diagonal from main
            shape: Matrix shape

        Returns:
            Sparse diagonal matrix
        """
        return sp.diags(diagonals, offsets, shape=shape)

    def to_dense(self, x: SparseMatrix) -> np.ndarray:
        """
        Convert sparse matrix to dense.

        Args:
            x: Sparse matrix

        Returns:
            Dense numpy array
        """
        return x.toarray()

    def matvec(
        self,
        a: SparseMatrix,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Sparse matrix-vector multiplication.

        Args:
            a: Sparse matrix (M x N)
            x: Dense vector (N,)

        Returns:
            Result vector (M,)
        """
        return a @ x

    def matmat(
        self,
        a: SparseMatrix,
        b: Union[SparseMatrix, np.ndarray],
    ) -> Union[SparseMatrix, np.ndarray]:
        """
        Sparse matrix-matrix multiplication.

        Args:
            a: Sparse matrix
            b: Sparse or dense matrix

        Returns:
            Product matrix
        """
        return a @ b

    def add(
        self,
        a: SparseMatrix,
        b: SparseMatrix,
    ) -> SparseMatrix:
        """
        Add two sparse matrices.

        Args:
            a: First sparse matrix
            b: Second sparse matrix

        Returns:
            Sum matrix
        """
        return a + b

    def scale(
        self,
        a: SparseMatrix,
        scalar: float,
    ) -> SparseMatrix:
        """
        Scale sparse matrix by scalar.

        Args:
            a: Sparse matrix
            scalar: Scaling factor

        Returns:
            Scaled matrix
        """
        return scalar * a

    def solve(
        self,
        a: SparseMatrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve sparse linear system Ax = b.

        Uses direct or iterative solver based on matrix properties.

        Args:
            a: Sparse coefficient matrix
            b: Right-hand side vector

        Returns:
            Solution vector
        """
        from scipy.sparse.linalg import spsolve

        return spsolve(a, b)

    def eigsh(
        self,
        a: SparseMatrix,
        k: int = 6,
        which: str = "LM",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalues/vectors of sparse symmetric matrix.

        Uses Arnoldi iteration for efficient computation.

        Args:
            a: Sparse symmetric matrix
            k: Number of eigenvalues to compute
            which: Which eigenvalues ('LM', 'SM', 'LA', 'SA')

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        from scipy.sparse.linalg import eigsh

        return eigsh(a, k=k, which=which)

    def svds(
        self,
        a: SparseMatrix,
        k: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Truncated SVD of sparse matrix.

        Efficient for low-rank approximations of large matrices.

        Args:
            a: Sparse matrix
            k: Number of singular values

        Returns:
            Tuple of (U, S, Vt)
        """
        from scipy.sparse.linalg import svds

        return svds(a, k=k)

    def nnz(self, a: SparseMatrix) -> int:
        """
        Number of non-zero elements.

        Args:
            a: Sparse matrix

        Returns:
            Count of non-zeros
        """
        return a.nnz

    def sparsity(self, a: SparseMatrix) -> float:
        """
        Sparsity ratio (fraction of zeros).

        Args:
            a: Sparse matrix

        Returns:
            Sparsity ratio (0 to 1)
        """
        total = a.shape[0] * a.shape[1]
        return 1.0 - a.nnz / total


class TensorOps:
    """
    Tensor operations for multi-dimensional agricultural data.

    Handles 3D+ arrays common in:
    - Multi-spectral satellite imagery (bands x height x width)
    - Time series of field data (time x fields x features)
    - Video streams from drones (frames x height x width x channels)

    Example:
        >>> tensor_ops = TensorOps()
        >>> # Contract over spectral bands
        >>> result = tensor_ops.contract(spectral_cube, weights, axis=0)
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize tensor operations.

        Args:
            use_gpu: Enable GPU acceleration
        """
        self.use_gpu = use_gpu

    def einsum(
        self,
        subscripts: str,
        *operands: np.ndarray,
        optimize: bool = True,
    ) -> np.ndarray:
        """
        Einstein summation for tensor contractions.

        Provides flexible tensor operations using Einstein notation.

        Args:
            subscripts: Einstein summation subscripts
            *operands: Input tensors
            optimize: Optimize contraction order

        Returns:
            Result tensor

        Example:
            >>> # Batch matrix multiply: (batch, i, j) x (batch, j, k) -> (batch, i, k)
            >>> result = tensor_ops.einsum('bij,bjk->bik', A, B)
        """
        return np.einsum(subscripts, *operands, optimize=optimize)

    def tensordot(
        self,
        a: np.ndarray,
        b: np.ndarray,
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
    ) -> np.ndarray:
        """
        Tensor contraction over specified axes.

        Args:
            a: First tensor
            b: Second tensor
            axes: Axes to contract over

        Returns:
            Contracted tensor
        """
        return np.tensordot(a, b, axes=axes)

    def mode_product(
        self,
        tensor: np.ndarray,
        matrix: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        """
        Mode-n product of tensor with matrix.

        Multiplies tensor along specified mode with matrix.

        Args:
            tensor: Input tensor
            matrix: Matrix to multiply
            mode: Mode (axis) for multiplication

        Returns:
            Transformed tensor
        """
        # Move mode to first axis
        tensor_moved = np.moveaxis(tensor, mode, 0)

        # Reshape for matrix multiply
        shape = tensor_moved.shape
        tensor_2d = tensor_moved.reshape(shape[0], -1)

        # Multiply
        result_2d = matrix @ tensor_2d

        # Reshape back
        new_shape = (matrix.shape[0],) + shape[1:]
        result = result_2d.reshape(new_shape)

        # Move axis back
        return np.moveaxis(result, 0, mode)

    def unfold(
        self,
        tensor: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        """
        Unfold tensor along specified mode.

        Converts n-dimensional tensor to 2D matrix.

        Args:
            tensor: Input tensor
            mode: Mode to unfold along

        Returns:
            Unfolded matrix
        """
        n_dims = tensor.ndim
        dims = list(range(n_dims))
        dims.remove(mode)
        dims.insert(0, mode)

        tensor_transposed = np.transpose(tensor, dims)
        return tensor_transposed.reshape(tensor.shape[mode], -1)

    def fold(
        self,
        matrix: np.ndarray,
        mode: int,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Fold matrix back to tensor.

        Inverse of unfold operation.

        Args:
            matrix: Unfolded matrix
            mode: Original mode
            shape: Target tensor shape

        Returns:
            Folded tensor
        """
        # Create permuted shape
        n_dims = len(shape)
        dims = list(range(n_dims))
        dims.remove(mode)
        dims.insert(0, mode)

        # Create intermediate shape
        inter_shape = [shape[d] for d in dims]

        # Reshape and transpose back
        tensor = matrix.reshape(inter_shape)

        # Inverse permutation
        inverse_dims = [0] * n_dims
        for i, d in enumerate(dims):
            inverse_dims[d] = i

        return np.transpose(tensor, inverse_dims)

    def tucker_decomposition(
        self,
        tensor: np.ndarray,
        ranks: List[int],
        n_iter_max: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Tucker decomposition for tensor dimensionality reduction.

        Decomposes tensor into core tensor and factor matrices.
        Useful for multi-way data analysis.

        Args:
            tensor: Input tensor
            ranks: Ranks for each mode
            n_iter_max: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Tuple of (core_tensor, [factor_matrices])
        """
        n_dims = tensor.ndim
        factors = []

        # Initialize factors using HOSVD
        for mode in range(n_dims):
            unfolded = self.unfold(tensor, mode)
            U, _, _ = linalg.svd(unfolded, full_matrices=False)
            factors.append(U[:, :ranks[mode]])

        # Alternating least squares refinement
        for iteration in range(n_iter_max):
            factors_old = [f.copy() for f in factors]

            for mode in range(n_dims):
                # Compute core using all other factors
                core = tensor.copy()
                for m in range(n_dims):
                    if m != mode:
                        core = self.mode_product(core, factors[m].T, m)

                # Update factor
                unfolded = self.unfold(core, mode)
                U, _, _ = linalg.svd(unfolded, full_matrices=False)
                factors[mode] = U[:, :ranks[mode]]

            # Check convergence
            diff = sum(
                np.linalg.norm(factors[m] - factors_old[m])
                for m in range(n_dims)
            )
            if diff < tol:
                break

        # Compute final core
        core = tensor.copy()
        for mode in range(n_dims):
            core = self.mode_product(core, factors[mode].T, mode)

        return core, factors


# Convenience functions


def eigendecomposition(
    x: ArrayLike,
    symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalue decomposition of a matrix.

    Args:
        x: Input matrix
        symmetric: Whether matrix is symmetric

    Returns:
        Tuple of (eigenvalues, eigenvectors)

    Example:
        >>> # PCA via eigendecomposition
        >>> cov = np.cov(data.T)
        >>> eigenvalues, eigenvectors = eigendecomposition(cov)
        >>> principal_components = eigenvectors[:, :n_components]
    """
    ops = MatrixOps()
    if symmetric:
        return ops.eigh(x)
    return ops.eig(x)


def svd_decomposition(
    x: ArrayLike,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular Value Decomposition.

    Args:
        x: Input matrix
        n_components: Number of components to keep

    Returns:
        Tuple of (U, S, Vt)

    Example:
        >>> # Dimensionality reduction of sensor data
        >>> U, S, Vt = svd_decomposition(sensor_matrix, n_components=10)
        >>> reduced_data = U[:, :10] @ np.diag(S[:10])
    """
    ops = MatrixOps()
    U, S, Vt = ops.svd(x)

    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]

    return U, S, Vt


def qr_decomposition(x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition.

    Args:
        x: Input matrix

    Returns:
        Tuple of (Q, R)
    """
    ops = MatrixOps()
    return ops.qr(x)


def cholesky_decomposition(
    x: ArrayLike,
    lower: bool = True,
) -> np.ndarray:
    """
    Cholesky decomposition for positive definite matrices.

    Args:
        x: Positive definite matrix
        lower: Return lower triangular factor

    Returns:
        Cholesky factor

    Example:
        >>> # Sample from multivariate normal
        >>> L = cholesky_decomposition(covariance)
        >>> samples = mean + L @ np.random.randn(n_features, n_samples)
    """
    ops = MatrixOps()
    return ops.cholesky(x, lower=lower)
