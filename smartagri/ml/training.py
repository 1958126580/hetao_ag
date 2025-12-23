"""
Model Training Module

Training utilities and optimization:
- Cross-validation
- Hyperparameter tuning
- Early stopping
- Training callbacks

Example:
    >>> trainer = ModelTrainer(model)
    >>> trainer.fit(X_train, y_train, validation_data=(X_val, y_val))
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging

from .models import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Training history record."""
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool


class EarlyStopping:
    """
    Early stopping callback.

    Stops training when validation loss
    stops improving.

    Example:
        >>> early_stop = EarlyStopping(patience=10)
        >>> if early_stop(val_loss):
        ...     break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs without improvement
            min_delta: Minimum change threshold
            restore_best: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self._best_loss = float('inf')
        self._counter = 0
        self._best_weights = None

    def __call__(self, val_loss: float, weights: Any = None) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            weights: Current model weights

        Returns:
            True if should stop
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
            if self.restore_best and weights is not None:
                self._best_weights = weights
            return False
        else:
            self._counter += 1
            return self._counter >= self.patience

    def get_best_weights(self) -> Any:
        """Get best weights."""
        return self._best_weights

    def reset(self) -> None:
        """Reset state."""
        self._best_loss = float('inf')
        self._counter = 0
        self._best_weights = None


class CrossValidator:
    """
    Cross-validation for model evaluation.

    Provides k-fold and time series cross-validation.

    Example:
        >>> cv = CrossValidator(n_folds=5)
        >>> scores = cv.validate(model, X, y)
    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def split(
        self,
        X: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for folds.

        Args:
            X: Features

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        fold_size = n_samples // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n_samples

            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            folds.append((train_indices, test_indices))

        return folds

    def validate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "mse",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric

        Returns:
            Dict with CV results
        """
        folds = self.split(X)
        scores = []

        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone model and fit
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate score
            if scoring == "mse":
                score = np.mean((y_test - predictions) ** 2)
            elif scoring == "rmse":
                score = np.sqrt(np.mean((y_test - predictions) ** 2))
            elif scoring == "mae":
                score = np.mean(np.abs(y_test - predictions))
            elif scoring == "r2":
                ss_res = np.sum((y_test - predictions) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                score = np.mean((y_test - predictions) ** 2)

            scores.append(score)

        return {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "n_folds": self.n_folds,
            "scoring": scoring,
        }

    def time_series_split(
        self,
        X: np.ndarray,
        gap: int = 0,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Time series cross-validation (expanding window).

        Args:
            X: Features
            gap: Gap between train and test

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_folds + 1)
        folds = []

        for i in range(self.n_folds):
            train_end = (i + 1) * test_size
            test_start = train_end + gap
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)

            folds.append((train_indices, test_indices))

        return folds


class HyperparameterTuner:
    """
    Hyperparameter optimization.

    Grid search and random search for
    hyperparameter tuning.

    Example:
        >>> tuner = HyperparameterTuner(model_class=RandomForestModel)
        >>> best_params = tuner.grid_search(param_grid, X, y)
    """

    def __init__(
        self,
        model_class: type,
        cv_folds: int = 5,
        scoring: str = "mse",
        random_state: Optional[int] = None,
    ):
        """
        Initialize tuner.

        Args:
            model_class: Model class to tune
            cv_folds: Cross-validation folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.model_class = model_class
        self.cv = CrossValidator(n_folds=cv_folds, random_state=random_state)
        self.scoring = scoring
        self.random_state = random_state

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Grid search over parameter combinations.

        Args:
            param_grid: Parameter grid
            X: Features
            y: Targets

        Returns:
            Dict with best parameters and scores
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = [[]]
        for values in param_values:
            combinations = [
                combo + [val]
                for combo in combinations
                for val in values
            ]

        best_score = float('inf') if self.scoring == "mse" else float('-inf')
        best_params = None
        all_results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            if self.random_state is not None:
                params['random_state'] = self.random_state

            model = self.model_class(**params)
            cv_result = self.cv.validate(model, X, y, self.scoring)

            all_results.append({
                "params": params,
                "mean_score": cv_result["mean"],
                "std_score": cv_result["std"],
            })

            # Check if better (lower is better for mse/rmse/mae)
            is_better = cv_result["mean"] < best_score if self.scoring in ["mse", "rmse", "mae"] \
                else cv_result["mean"] > best_score

            if is_better:
                best_score = cv_result["mean"]
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }

    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        X: np.ndarray = None,
        y: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Random search over parameter distributions.

        Args:
            param_distributions: Parameter distributions
            n_iter: Number of iterations
            X: Features
            y: Targets

        Returns:
            Dict with best parameters
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_score = float('inf') if self.scoring == "mse" else float('-inf')
        best_params = None
        all_results = []

        for _ in range(n_iter):
            params = {}
            for name, dist in param_distributions.items():
                if isinstance(dist, list):
                    params[name] = np.random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    if isinstance(dist[0], int) and isinstance(dist[1], int):
                        params[name] = np.random.randint(dist[0], dist[1] + 1)
                    else:
                        params[name] = np.random.uniform(dist[0], dist[1])
                else:
                    params[name] = dist

            if self.random_state is not None:
                params['random_state'] = self.random_state

            model = self.model_class(**params)
            cv_result = self.cv.validate(model, X, y, self.scoring)

            all_results.append({
                "params": params,
                "mean_score": cv_result["mean"],
            })

            is_better = cv_result["mean"] < best_score if self.scoring in ["mse", "rmse", "mae"] \
                else cv_result["mean"] > best_score

            if is_better:
                best_score = cv_result["mean"]
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }


class ModelTrainer:
    """
    Model training manager.

    Handles training loop with validation,
    early stopping, and callbacks.

    Example:
        >>> trainer = ModelTrainer(model)
        >>> history = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        model: BaseModel,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            early_stopping: Early stopping callback
        """
        self.model = model
        self.early_stopping = early_stopping or EarlyStopping(patience=10)
        self._history: Optional[TrainingHistory] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train model with validation.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            verbose: Print progress

        Returns:
            TrainingHistory
        """
        train_losses = []
        val_losses = []
        stopped_early = False

        # For simple models, just fit once
        self.model.fit(X_train, y_train)

        # Calculate losses
        train_pred = self.model.predict(X_train)
        train_loss = np.mean((y_train - train_pred) ** 2)
        train_losses.append(train_loss)

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_loss = np.mean((y_val - val_pred) ** 2)
            val_losses.append(val_loss)
        else:
            val_loss = train_loss
            val_losses.append(val_loss)

        if verbose:
            logger.info(f"Training complete - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        self._history = TrainingHistory(
            train_loss=train_losses,
            val_loss=val_losses,
            best_epoch=0,
            best_val_loss=val_loss,
            stopped_early=stopped_early,
        )

        return self._history

    def get_history(self) -> Optional[TrainingHistory]:
        """Get training history."""
        return self._history

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on data.

        Args:
            X: Features
            y: Targets

        Returns:
            Dict with metrics
        """
        predictions = self.model.predict(X)

        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
