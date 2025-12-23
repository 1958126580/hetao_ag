"""
Data Preprocessing Module

Data preprocessing utilities for ML pipelines:
- Feature scaling and normalization
- Feature engineering
- Missing value handling
- Time series transformation

Example:
    >>> preprocessor = DataPreprocessor()
    >>> X_scaled = preprocessor.fit_transform(X)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing.

    Handles scaling, normalization, and missing values
    for ML model input preparation.

    Example:
        >>> prep = DataPreprocessor(method="standardize")
        >>> X_train_scaled = prep.fit_transform(X_train)
        >>> X_test_scaled = prep.transform(X_test)
    """

    def __init__(
        self,
        method: str = "standardize",
        handle_missing: str = "mean",
    ):
        """
        Initialize preprocessor.

        Args:
            method: Scaling method ("standardize", "minmax", "robust")
            handle_missing: Missing value strategy ("mean", "median", "zero", "drop")
        """
        self.method = method
        self.handle_missing = handle_missing

        self._fitted = False
        self._params: Dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """
        Fit preprocessor to training data.

        Args:
            X: Training features

        Returns:
            Fitted preprocessor
        """
        # Handle missing values first
        X_clean = self._handle_missing_fit(X)

        if self.method == "standardize":
            self._params["mean"] = np.nanmean(X_clean, axis=0)
            self._params["std"] = np.nanstd(X_clean, axis=0) + 1e-8

        elif self.method == "minmax":
            self._params["min"] = np.nanmin(X_clean, axis=0)
            self._params["max"] = np.nanmax(X_clean, axis=0)
            self._params["range"] = self._params["max"] - self._params["min"] + 1e-8

        elif self.method == "robust":
            self._params["median"] = np.nanmedian(X_clean, axis=0)
            q75 = np.nanpercentile(X_clean, 75, axis=0)
            q25 = np.nanpercentile(X_clean, 25, axis=0)
            self._params["iqr"] = q75 - q25 + 1e-8

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted")

        X_clean = self._handle_missing_transform(X)

        if self.method == "standardize":
            return (X_clean - self._params["mean"]) / self._params["std"]

        elif self.method == "minmax":
            return (X_clean - self._params["min"]) / self._params["range"]

        elif self.method == "robust":
            return (X_clean - self._params["median"]) / self._params["iqr"]

        return X_clean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform to original scale.

        Args:
            X: Transformed features

        Returns:
            Original scale features
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted")

        if self.method == "standardize":
            return X * self._params["std"] + self._params["mean"]

        elif self.method == "minmax":
            return X * self._params["range"] + self._params["min"]

        elif self.method == "robust":
            return X * self._params["iqr"] + self._params["median"]

        return X

    def _handle_missing_fit(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values during fitting."""
        X_clean = X.copy()

        if self.handle_missing == "mean":
            self._params["fill_values"] = np.nanmean(X, axis=0)
        elif self.handle_missing == "median":
            self._params["fill_values"] = np.nanmedian(X, axis=0)
        elif self.handle_missing == "zero":
            self._params["fill_values"] = np.zeros(X.shape[1])

        if self.handle_missing != "drop":
            for i in range(X.shape[1]):
                mask = np.isnan(X_clean[:, i])
                X_clean[mask, i] = self._params["fill_values"][i]

        return X_clean

    def _handle_missing_transform(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values during transform."""
        X_clean = X.copy()

        if self.handle_missing != "drop" and "fill_values" in self._params:
            for i in range(X.shape[1]):
                mask = np.isnan(X_clean[:, i])
                X_clean[mask, i] = self._params["fill_values"][i]

        return X_clean


class FeatureEngineering:
    """
    Feature engineering utilities.

    Creates new features from existing data
    for improved model performance.

    Example:
        >>> eng = FeatureEngineering()
        >>> X_enhanced = eng.add_polynomial_features(X, degree=2)
    """

    def __init__(self):
        """Initialize feature engineering."""
        pass

    def add_polynomial_features(
        self,
        X: np.ndarray,
        degree: int = 2,
        include_bias: bool = False,
    ) -> np.ndarray:
        """
        Add polynomial features.

        Args:
            X: Input features
            degree: Polynomial degree
            include_bias: Include bias term

        Returns:
            Features with polynomial terms
        """
        n_samples, n_features = X.shape
        features = [X]

        if include_bias:
            features.insert(0, np.ones((n_samples, 1)))

        for d in range(2, degree + 1):
            for i in range(n_features):
                features.append(X[:, i:i+1] ** d)

        # Interaction terms (degree 2 only for efficiency)
        if degree >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(features)

    def add_lag_features(
        self,
        X: np.ndarray,
        lags: List[int],
    ) -> np.ndarray:
        """
        Add lagged features for time series.

        Args:
            X: Input features
            lags: List of lag values

        Returns:
            Features with lags
        """
        n_samples, n_features = X.shape
        max_lag = max(lags)

        # Create lagged features
        lag_features = []
        for lag in lags:
            lagged = np.zeros((n_samples, n_features))
            lagged[lag:] = X[:-lag] if lag > 0 else X
            lag_features.append(lagged)

        # Combine with original (trimming to valid rows)
        all_features = np.hstack([X] + lag_features)
        return all_features[max_lag:]

    def add_rolling_features(
        self,
        X: np.ndarray,
        window_sizes: List[int],
        functions: List[str] = None,
    ) -> np.ndarray:
        """
        Add rolling window features.

        Args:
            X: Input features
            window_sizes: List of window sizes
            functions: Aggregation functions ("mean", "std", "min", "max")

        Returns:
            Features with rolling statistics
        """
        if functions is None:
            functions = ["mean", "std"]

        n_samples, n_features = X.shape
        all_features = [X]

        for window in window_sizes:
            for func in functions:
                rolled = np.zeros((n_samples, n_features))

                for i in range(window - 1, n_samples):
                    window_data = X[i - window + 1:i + 1]

                    if func == "mean":
                        rolled[i] = np.mean(window_data, axis=0)
                    elif func == "std":
                        rolled[i] = np.std(window_data, axis=0)
                    elif func == "min":
                        rolled[i] = np.min(window_data, axis=0)
                    elif func == "max":
                        rolled[i] = np.max(window_data, axis=0)

                all_features.append(rolled)

        return np.hstack(all_features)

    def interaction_features(
        self,
        X: np.ndarray,
        feature_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Create interaction features.

        Args:
            X: Input features
            feature_pairs: Specific pairs to interact (all if None)

        Returns:
            Features with interactions
        """
        n_features = X.shape[1]

        if feature_pairs is None:
            feature_pairs = [
                (i, j) for i in range(n_features)
                for j in range(i + 1, n_features)
            ]

        interactions = []
        for i, j in feature_pairs:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack([X] + interactions)


class DataAugmentation:
    """
    Data augmentation for training data expansion.

    Generates synthetic samples to increase
    training data size and diversity.

    Example:
        >>> augmentor = DataAugmentation()
        >>> X_aug, y_aug = augmentor.noise_augmentation(X, y)
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize augmentation.

        Args:
            random_state: Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)

    def noise_augmentation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_level: float = 0.1,
        n_copies: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise augmentation.

        Args:
            X: Features
            y: Target values
            noise_level: Noise standard deviation (fraction of feature std)
            n_copies: Number of augmented copies

        Returns:
            Augmented X and y
        """
        X_list = [X]
        y_list = [y]

        feature_std = np.std(X, axis=0)

        for _ in range(n_copies):
            noise = np.random.randn(*X.shape) * feature_std * noise_level
            X_list.append(X + noise)
            y_list.append(y)

        return np.vstack(X_list), np.hstack(y_list)

    def smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE-like oversampling (simplified).

        Args:
            X: Features
            y: Target values
            k_neighbors: Number of neighbors

        Returns:
            Oversampled X and y
        """
        n_samples = X.shape[0]

        # Generate synthetic samples
        synthetic_X = []
        synthetic_y = []

        for i in range(n_samples):
            # Find k nearest neighbors
            distances = np.sum((X - X[i]) ** 2, axis=1)
            neighbor_indices = np.argsort(distances)[1:k_neighbors + 1]

            # Interpolate with random neighbor
            neighbor = X[neighbor_indices[np.random.randint(k_neighbors)]]
            alpha = np.random.random()

            synthetic = X[i] + alpha * (neighbor - X[i])
            synthetic_X.append(synthetic)
            synthetic_y.append(y[i])

        return (
            np.vstack([X, np.array(synthetic_X)]),
            np.hstack([y, np.array(synthetic_y)])
        )


class TimeSeriesTransformer:
    """
    Time series data transformation.

    Prepares time series data for ML models
    with proper train/test splitting.

    Example:
        >>> transformer = TimeSeriesTransformer(lookback=10)
        >>> X, y = transformer.create_sequences(data)
    """

    def __init__(
        self,
        lookback: int = 10,
        horizon: int = 1,
    ):
        """
        Initialize transformer.

        Args:
            lookback: Number of past time steps
            horizon: Forecast horizon
        """
        self.lookback = lookback
        self.horizon = horizon

    def create_sequences(
        self,
        data: np.ndarray,
        target_col: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning.

        Args:
            data: Time series data (n_samples, n_features)
            target_col: Target column index

        Returns:
            X (sequences), y (targets)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_samples = len(data) - self.lookback - self.horizon + 1

        X = np.zeros((n_samples, self.lookback, data.shape[1]))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i] = data[i:i + self.lookback]
            y[i] = data[i + self.lookback + self.horizon - 1, target_col]

        return X.reshape(n_samples, -1), y

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time series aware train/test split.

        Args:
            X: Features
            y: Targets
            test_size: Fraction for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * (1 - test_size))

        return (
            X[:split_idx],
            X[split_idx:],
            y[:split_idx],
            y[split_idx:],
        )

    def difference(
        self,
        data: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        """
        Apply differencing for stationarity.

        Args:
            data: Time series data
            order: Differencing order

        Returns:
            Differenced data
        """
        result = data.copy()
        for _ in range(order):
            result = np.diff(result, axis=0)
        return result

    def inverse_difference(
        self,
        diff_data: np.ndarray,
        initial_values: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse differencing transform.

        Args:
            diff_data: Differenced data
            initial_values: Initial values for integration

        Returns:
            Original scale data
        """
        return np.cumsum(np.vstack([initial_values, diff_data]), axis=0)
