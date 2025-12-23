"""
Machine Learning Module for SmartAgri

Agricultural machine learning models and utilities:
- Crop yield prediction
- Disease classification
- Anomaly detection
- Time series forecasting
- Computer vision for agriculture

Example:
    >>> from smartagri.ml import YieldPredictor, DiseaseClassifier
    >>> predictor = YieldPredictor()
    >>> predictor.fit(X_train, y_train)
    >>> predictions = predictor.predict(X_test)
"""

from .models import (
    BaseModel,
    RandomForestModel,
    GradientBoostingModel,
    NeuralNetworkModel,
    EnsembleModel,
)
from .preprocessing import (
    DataPreprocessor,
    FeatureEngineering,
    DataAugmentation,
    TimeSeriesTransformer,
)
from .training import (
    ModelTrainer,
    CrossValidator,
    HyperparameterTuner,
    EarlyStopping,
)
from .evaluation import (
    ModelEvaluator,
    RegressionMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
)
from .applications import (
    YieldPredictor,
    DiseaseClassifier,
    AnomalyDetector,
    TimeSeriesForecaster,
)

__all__ = [
    # Models
    "BaseModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "NeuralNetworkModel",
    "EnsembleModel",
    # Preprocessing
    "DataPreprocessor",
    "FeatureEngineering",
    "DataAugmentation",
    "TimeSeriesTransformer",
    # Training
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterTuner",
    "EarlyStopping",
    # Evaluation
    "ModelEvaluator",
    "RegressionMetrics",
    "ClassificationMetrics",
    "ConfusionMatrix",
    # Applications
    "YieldPredictor",
    "DiseaseClassifier",
    "AnomalyDetector",
    "TimeSeriesForecaster",
]
