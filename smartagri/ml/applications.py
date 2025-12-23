"""
Agricultural ML Applications

Domain-specific ML models for agriculture:
- Yield prediction
- Disease classification
- Anomaly detection
- Time series forecasting

Example:
    >>> predictor = YieldPredictor()
    >>> predictor.fit(X_train, y_train)
    >>> yield_forecast = predictor.predict(X_test)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .models import RandomForestModel, GradientBoostingModel, EnsembleModel
from .preprocessing import DataPreprocessor, FeatureEngineering
from .training import CrossValidator
from .evaluation import ModelEvaluator, RegressionMetrics

logger = logging.getLogger(__name__)


class YieldPredictor:
    """
    Crop yield prediction model.

    Predicts crop yields based on weather,
    soil, and management factors.

    Example:
        >>> predictor = YieldPredictor()
        >>> predictor.fit(features, yields)
        >>> predicted = predictor.predict(new_features)
    """

    def __init__(
        self,
        model_type: str = "ensemble",
        n_estimators: int = 100,
    ):
        """
        Initialize yield predictor.

        Args:
            model_type: Model type ("rf", "gb", "ensemble")
            n_estimators: Number of estimators
        """
        self.model_type = model_type

        if model_type == "rf":
            self.model = RandomForestModel(n_estimators=n_estimators)
        elif model_type == "gb":
            self.model = GradientBoostingModel(n_estimators=n_estimators)
        else:
            self.model = EnsembleModel([
                RandomForestModel(n_estimators=n_estimators // 2),
                GradientBoostingModel(n_estimators=n_estimators // 2),
            ])

        self.preprocessor = DataPreprocessor(method="standardize")
        self.evaluator = ModelEvaluator()
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'YieldPredictor':
        """
        Fit yield prediction model.

        Args:
            X: Features (weather, soil, management)
            y: Yield values
            feature_names: Feature names

        Returns:
            Fitted predictor
        """
        self._feature_names = feature_names

        # Preprocess
        X_processed = self.preprocessor.fit_transform(X)

        # Fit model
        self.model.fit(X_processed, y)
        self._is_fitted = True

        logger.info("Yield predictor fitted successfully")
        return self

    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False,
    ) -> np.ndarray:
        """
        Predict crop yields.

        Args:
            X: Features
            return_confidence: Return confidence intervals

        Returns:
            Predicted yields
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)

        if return_confidence:
            # Estimate uncertainty
            std_estimate = np.std(predictions) * 0.1
            return predictions, predictions - 1.96 * std_estimate, predictions + 1.96 * std_estimate

        return predictions

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> RegressionMetrics:
        """
        Evaluate model performance.

        Args:
            X: Test features
            y: True yields

        Returns:
            RegressionMetrics
        """
        predictions = self.predict(X)
        return self.evaluator.regression_metrics(y, predictions, X.shape[1])

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Cross-validate model.

        Args:
            X: Features
            y: Yields
            n_folds: Number of folds

        Returns:
            CV results
        """
        cv = CrossValidator(n_folds=n_folds)
        X_processed = self.preprocessor.fit_transform(X)
        return cv.validate(self.model, X_processed, y, scoring="r2")


class DiseaseClassifier:
    """
    Plant disease classification model.

    Classifies diseases based on symptoms
    or image features.

    Example:
        >>> classifier = DiseaseClassifier()
        >>> classifier.fit(features, disease_labels)
        >>> predicted = classifier.predict(new_features)
    """

    def __init__(
        self,
        n_estimators: int = 100,
    ):
        """
        Initialize disease classifier.

        Args:
            n_estimators: Number of estimators
        """
        self.model = RandomForestModel(n_estimators=n_estimators)
        self.preprocessor = DataPreprocessor(method="standardize")
        self._is_fitted = False
        self._classes = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> 'DiseaseClassifier':
        """
        Fit disease classifier.

        Args:
            X: Features
            y: Disease labels

        Returns:
            Fitted classifier
        """
        self._classes = np.unique(y)

        # Preprocess
        X_processed = self.preprocessor.fit_transform(X)

        # Fit model
        self.model.fit(X_processed, y)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict disease classes.

        Args:
            X: Features

        Returns:
            Predicted classes
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (simplified).

        Args:
            X: Features

        Returns:
            Class probabilities
        """
        predictions = self.predict(X)
        n_classes = len(self._classes)

        # Simplified: return one-hot encoding
        proba = np.zeros((len(X), n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self._classes == pred)[0]
            if len(class_idx) > 0:
                proba[i, class_idx[0]] = 1.0

        return proba


class AnomalyDetector:
    """
    Anomaly detection for agricultural data.

    Detects unusual patterns in sensor data,
    weather, or crop conditions.

    Example:
        >>> detector = AnomalyDetector()
        >>> detector.fit(normal_data)
        >>> anomalies = detector.detect(new_data)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        method: str = "isolation_forest",
    ):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected anomaly fraction
            method: Detection method
        """
        self.contamination = contamination
        self.method = method
        self._is_fitted = False
        self._threshold = None
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """
        Fit anomaly detector.

        Args:
            X: Normal training data

        Returns:
            Fitted detector
        """
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-8

        # Calculate anomaly scores
        scores = self._calculate_scores(X)

        # Set threshold based on contamination
        self._threshold = np.percentile(scores, (1 - self.contamination) * 100)

        self._is_fitted = True
        return self

    def detect(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies.

        Args:
            X: Data to check

        Returns:
            Boolean array (True = anomaly)
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted")

        scores = self._calculate_scores(X)
        return scores > self._threshold

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted")

        return self._calculate_scores(X)

    def _calculate_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores (Mahalanobis-like distance)."""
        normalized = (X - self._mean) / self._std
        return np.sqrt(np.sum(normalized ** 2, axis=1))


class TimeSeriesForecaster:
    """
    Time series forecasting for agriculture.

    Forecasts weather, prices, or crop growth
    time series data.

    Example:
        >>> forecaster = TimeSeriesForecaster(lookback=30)
        >>> forecaster.fit(historical_data)
        >>> forecast = forecaster.predict(steps=7)
    """

    def __init__(
        self,
        lookback: int = 30,
        horizon: int = 7,
    ):
        """
        Initialize forecaster.

        Args:
            lookback: Past time steps to use
            horizon: Forecast horizon
        """
        self.lookback = lookback
        self.horizon = horizon
        self.model = GradientBoostingModel(n_estimators=50, max_depth=5)
        self._is_fitted = False
        self._last_values = None

    def fit(
        self,
        data: np.ndarray,
    ) -> 'TimeSeriesForecaster':
        """
        Fit forecaster.

        Args:
            data: Time series data

        Returns:
            Fitted forecaster
        """
        # Create sequences
        X, y = self._create_sequences(data)

        # Fit model
        self.model.fit(X, y)

        # Store last values for prediction
        self._last_values = data[-self.lookback:]
        self._is_fitted = True

        return self

    def predict(
        self,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Make forecast.

        Args:
            steps: Number of steps to forecast

        Returns:
            Forecasted values
        """
        if not self._is_fitted:
            raise ValueError("Forecaster not fitted")

        if steps is None:
            steps = self.horizon

        predictions = []
        current_input = self._last_values.copy()

        for _ in range(steps):
            # Predict next value
            X = current_input.reshape(1, -1)
            pred = self.model.predict(X)[0]
            predictions.append(pred)

            # Update input window
            current_input = np.roll(current_input, -1)
            current_input[-1] = pred

        return np.array(predictions)

    def _create_sequences(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for training."""
        n_samples = len(data) - self.lookback

        X = np.zeros((n_samples, self.lookback))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i] = data[i:i + self.lookback]
            y[i] = data[i + self.lookback]

        return X, y

    def evaluate(
        self,
        data: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.

        Args:
            data: Full time series
            test_size: Test set fraction

        Returns:
            Evaluation metrics
        """
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Fit on training data
        self.fit(train_data)

        # Forecast test period
        n_test = len(test_data)
        forecast = self.predict(steps=n_test)

        # Calculate metrics
        mse = np.mean((test_data - forecast) ** 2)
        mae = np.mean(np.abs(test_data - forecast))
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "mape": mape,
        }
