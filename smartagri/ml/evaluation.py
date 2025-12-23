"""
Model Evaluation Module

Comprehensive model evaluation and metrics:
- Regression metrics
- Classification metrics
- Confusion matrix analysis
- Performance visualization

Example:
    >>> evaluator = ModelEvaluator(model)
    >>> metrics = evaluator.evaluate(X_test, y_test)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    adjusted_r2: float
    n_samples: int
    n_features: int


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    n_samples: int
    n_classes: int


@dataclass
class ConfusionMatrix:
    """Confusion matrix results."""
    matrix: np.ndarray
    classes: List[Any]
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class ModelEvaluator:
    """
    Comprehensive model evaluation.

    Calculates metrics for regression and
    classification models.

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.regression_metrics(y_true, y_pred)
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int = 1,
    ) -> RegressionMetrics:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features

        Returns:
            RegressionMetrics object
        """
        n = len(y_true)

        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))

        # Mean Absolute Percentage Error
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 0.0

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Adjusted R-squared
        if n > n_features + 1:
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        else:
            adjusted_r2 = r2

        return RegressionMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2=r2,
            adjusted_r2=adjusted_r2,
            n_samples=n,
            n_features=n_features,
        )

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            ClassificationMetrics object
        """
        n = len(y_true)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        # For binary classification
        if n_classes == 2:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            accuracy = (tp + tn) / n if n > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Multi-class: macro average
            accuracy = np.mean(y_true == y_pred)

            precisions = []
            recalls = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                precisions.append(p)
                recalls.append(r)

            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = 0  # Not applicable for multi-class

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            n_samples=n,
            n_classes=n_classes,
        )

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ConfusionMatrix:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            ConfusionMatrix object
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i, cls_true in enumerate(classes):
            for j, cls_pred in enumerate(classes):
                matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))

        # Binary classification metrics
        if n_classes == 2:
            tp = matrix[1, 1]
            tn = matrix[0, 0]
            fp = matrix[0, 1]
            fn = matrix[1, 0]
        else:
            tp = tn = fp = fn = 0

        return ConfusionMatrix(
            matrix=matrix,
            classes=list(classes),
            true_positive=tp,
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
        )

    def residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze prediction residuals.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dict with residual analysis
        """
        residuals = y_true - y_pred

        return {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "min": np.min(residuals),
            "max": np.max(residuals),
            "skewness": self._skewness(residuals),
            "kurtosis": self._kurtosis(residuals),
            "normality_test": self._normality_test(residuals),
        }

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum(((x - mean) / std) ** 3) / n

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum(((x - mean) / std) ** 4) / n - 3

    def _normality_test(self, x: np.ndarray) -> Dict[str, Any]:
        """Simplified normality test."""
        skew = abs(self._skewness(x))
        kurt = abs(self._kurtosis(x))

        # Heuristic test
        is_normal = skew < 2 and kurt < 7

        return {
            "is_normal": is_normal,
            "skewness": self._skewness(x),
            "kurtosis": self._kurtosis(x),
        }

    def feature_importance_analysis(
        self,
        importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze feature importance.

        Args:
            importance: Feature importance scores
            feature_names: Feature names

        Returns:
            List of feature importance info
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        # Normalize importance
        total = np.sum(np.abs(importance))
        normalized = importance / total if total > 0 else importance

        results = []
        for name, imp, norm_imp in zip(feature_names, importance, normalized):
            results.append({
                "feature": name,
                "importance": imp,
                "normalized_importance": norm_imp,
                "rank": 0,  # Will be filled
            })

        # Sort and rank
        results.sort(key=lambda x: x["importance"], reverse=True)
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results
