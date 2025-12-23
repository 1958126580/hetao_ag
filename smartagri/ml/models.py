"""
Machine Learning Models

Core ML model implementations for agricultural applications:
- Random Forest
- Gradient Boosting
- Neural Networks
- Ensemble methods

Example:
    >>> model = RandomForestModel(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for ML models.

    Provides common interface for all model types.
    """

    def __init__(self, **params):
        """Initialize model with parameters."""
        self.params = params
        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X, y)
        return self.predict(X)

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.params.update(params)
        return self


class DecisionTree:
    """
    Decision tree implementation.

    Used as base learner for ensemble methods.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        """
        Initialize decision tree.

        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Fit tree to data."""
        self._tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.array([self._predict_sample(x, self._tree) for x in X])

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> Dict[str, Any]:
        """Recursively build tree."""
        n_samples, n_features = X.shape

        # Check stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {"leaf": True, "value": np.mean(y)}

        # Find best split
        best_gain = 0
        best_feature = 0
        best_threshold = 0

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain == 0:
            return {"leaf": True, "value": np.mean(y)}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if (np.sum(left_mask) < self.min_samples_leaf or
            np.sum(right_mask) < self.min_samples_leaf):
            return {"leaf": True, "value": np.mean(y)}

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _information_gain(
        self,
        feature: np.ndarray,
        y: np.ndarray,
        threshold: float,
    ) -> float:
        """Calculate information gain (variance reduction)."""
        parent_var = np.var(y)

        left_mask = feature <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)

        left_var = np.var(y[left_mask])
        right_var = np.var(y[right_mask])

        child_var = (n_left / n) * left_var + (n_right / n) * right_var

        return parent_var - child_var

    def _predict_sample(self, x: np.ndarray, node: Dict) -> float:
        """Predict single sample."""
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])


class RandomForestModel(BaseModel):
    """
    Random Forest regression/classification.

    Ensemble of decision trees with bagging
    and random feature selection.

    Example:
        >>> rf = RandomForestModel(n_estimators=100, max_depth=10)
        >>> rf.fit(X_train, y_train)
        >>> predictions = rf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        max_features: str = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Random Forest.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            max_features: Features to consider at each split
            min_samples_split: Minimum samples to split node
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=random_state,
        )

        self._trees: List[DecisionTree] = []
        self._feature_indices: List[np.ndarray] = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """
        Fit Random Forest to training data.

        Args:
            X: Features (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            Fitted model
        """
        n_samples, n_features = X.shape

        # Determine features per tree
        if self.params["max_features"] == "sqrt":
            n_features_tree = int(np.sqrt(n_features))
        elif self.params["max_features"] == "log2":
            n_features_tree = int(np.log2(n_features))
        else:
            n_features_tree = n_features

        self._trees = []
        self._feature_indices = []

        for _ in range(self.params["n_estimators"]):
            # Bootstrap sample
            if self.params["bootstrap"]:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            # Random feature subset
            feature_idx = np.random.choice(
                n_features, n_features_tree, replace=False
            )
            self._feature_indices.append(feature_idx)

            # Train tree
            tree = DecisionTree(
                max_depth=self.params["max_depth"],
                min_samples_split=self.params["min_samples_split"],
            )
            tree.fit(X_sample[:, feature_idx], y_sample)
            self._trees.append(tree)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = np.zeros((len(X), len(self._trees)))

        for i, (tree, feat_idx) in enumerate(zip(self._trees, self._feature_indices)):
            predictions[:, i] = tree.predict(X[:, feat_idx])

        return np.mean(predictions, axis=1)

    def feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance using permutation.

        Args:
            X: Features
            y: Target values

        Returns:
            Feature importance scores
        """
        baseline_score = self._score(X, y)
        n_features = X.shape[1]
        importance = np.zeros(n_features)

        for i in range(n_features):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self._score(X_permuted, y)
            importance[i] = baseline_score - permuted_score

        return importance / np.sum(importance) if np.sum(importance) > 0 else importance

    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting regression.

    Sequential ensemble that fits residuals
    of previous predictions.

    Example:
        >>> gb = GradientBoostingModel(n_estimators=100, learning_rate=0.1)
        >>> gb.fit(X_train, y_train)
        >>> predictions = gb.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Gradient Boosting.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate (shrinkage)
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            subsample: Fraction of samples for each tree
            random_state: Random seed
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            random_state=random_state,
        )

        self._trees: List[DecisionTree] = []
        self._initial_prediction = 0.0

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingModel':
        """
        Fit Gradient Boosting model.

        Args:
            X: Features
            y: Target values

        Returns:
            Fitted model
        """
        n_samples = X.shape[0]
        self._initial_prediction = np.mean(y)

        # Initialize predictions
        predictions = np.full(n_samples, self._initial_prediction)

        self._trees = []

        for _ in range(self.params["n_estimators"]):
            # Calculate residuals (negative gradient)
            residuals = y - predictions

            # Subsample
            if self.params["subsample"] < 1.0:
                n_subsample = int(n_samples * self.params["subsample"])
                indices = np.random.choice(n_samples, n_subsample, replace=False)
                X_sample, residuals_sample = X[indices], residuals[indices]
            else:
                X_sample, residuals_sample = X, residuals

            # Fit tree to residuals
            tree = DecisionTree(
                max_depth=self.params["max_depth"],
                min_samples_split=self.params["min_samples_split"],
            )
            tree.fit(X_sample, residuals_sample)
            self._trees.append(tree)

            # Update predictions
            predictions += self.params["learning_rate"] * tree.predict(X)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = np.full(X.shape[0], self._initial_prediction)

        for tree in self._trees:
            predictions += self.params["learning_rate"] * tree.predict(X)

        return predictions


class NeuralNetworkModel(BaseModel):
    """
    Simple feedforward neural network.

    Multi-layer perceptron for regression
    and classification tasks.

    Example:
        >>> nn = NeuralNetworkModel(hidden_layers=[64, 32])
        >>> nn.fit(X_train, y_train)
        >>> predictions = nn.predict(X_test)
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Neural Network.

        Args:
            hidden_layers: Hidden layer sizes
            activation: Activation function
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Mini-batch size
            random_state: Random seed
        """
        super().__init__(
            hidden_layers=hidden_layers or [64, 32],
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []

        if random_state is not None:
            np.random.seed(random_state)

    def _init_weights(self, input_size: int, output_size: int = 1):
        """Initialize network weights."""
        layer_sizes = [input_size] + self.params["hidden_layers"] + [output_size]

        self._weights = []
        self._biases = []

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])

            self._weights.append(w)
            self._biases.append(b)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.params["activation"] == "relu":
            return np.maximum(0, x)
        elif self.params["activation"] == "tanh":
            return np.tanh(x)
        elif self.params["activation"] == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Activation function derivative."""
        if self.params["activation"] == "relu":
            return (x > 0).astype(float)
        elif self.params["activation"] == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.params["activation"] == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            return np.ones_like(x)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through network."""
        activations = [X]
        zs = []

        a = X
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = a @ w + b
            zs.append(z)

            # Apply activation (except last layer)
            if i < len(self._weights) - 1:
                a = self._activation(z)
            else:
                a = z  # Linear output for regression

            activations.append(a)

        return activations, zs

    def _backward(
        self,
        activations: List[np.ndarray],
        zs: List[np.ndarray],
        y: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass (backpropagation)."""
        m = y.shape[0]
        grad_w = [np.zeros_like(w) for w in self._weights]
        grad_b = [np.zeros_like(b) for b in self._biases]

        # Output layer error
        delta = activations[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self._weights))):
            grad_w[i] = activations[i].T @ delta / m
            grad_b[i] = np.mean(delta, axis=0)

            if i > 0:
                delta = (delta @ self._weights[i].T) * self._activation_derivative(zs[i - 1])

        return grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkModel':
        """
        Fit neural network.

        Args:
            X: Features
            y: Target values

        Returns:
            Fitted model
        """
        n_samples = X.shape[0]
        self._init_weights(X.shape[1], 1)

        # Normalize inputs
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self._X_mean) / self._X_std

        self._y_mean = np.mean(y)
        self._y_std = np.std(y) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        batch_size = self.params["batch_size"]
        lr = self.params["learning_rate"]

        for epoch in range(self.params["epochs"]):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = X_norm[batch_idx]
                y_batch = y_norm[batch_idx]

                # Forward and backward pass
                activations, zs = self._forward(X_batch)
                grad_w, grad_b = self._backward(activations, zs, y_batch)

                # Update weights
                for i in range(len(self._weights)):
                    self._weights[i] -= lr * grad_w[i]
                    self._biases[i] -= lr * grad_b[i]

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        X_norm = (X - self._X_mean) / self._X_std
        activations, _ = self._forward(X_norm)

        # Denormalize output
        return activations[-1].flatten() * self._y_std + self._y_mean


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models.

    Combines predictions from different model types
    using averaging or stacking.

    Example:
        >>> ensemble = EnsembleModel([
        ...     RandomForestModel(n_estimators=50),
        ...     GradientBoostingModel(n_estimators=50),
        ... ])
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
        method: str = "average",
    ):
        """
        Initialize ensemble.

        Args:
            models: List of base models
            weights: Model weights for averaging
            method: Combination method ("average" or "stack")
        """
        super().__init__(method=method)

        self.models = models
        self.weights = weights
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit all models in ensemble.

        Args:
            X: Features
            y: Target values

        Returns:
            Fitted ensemble
        """
        for model in self.models:
            model.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features

        Returns:
            Combined predictions
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        # Weighted average
        return np.average(predictions, axis=1, weights=self.weights)
