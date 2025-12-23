"""
SmartAgri: A Comprehensive Smart Agriculture and Animal Husbandry Library

This library provides world-class computational tools for modern agriculture and
animal husbandry, featuring GPU acceleration, parallel computing, and advanced
machine learning capabilities.

Core Modules:
    - core: Mathematical foundations with GPU/parallel computing support
    - crops: Crop management, yield prediction, and growth modeling
    - soil: Soil analysis, nutrient management, and fertility assessment
    - weather: Weather data processing, forecasting, and climate analysis
    - irrigation: Smart irrigation scheduling and water management
    - pests: Pest and disease detection using computer vision
    - livestock: Livestock management and tracking systems
    - health: Animal health monitoring and diagnostics
    - feed: Feed optimization and nutrition planning
    - spatial: GIS and spatial analysis for precision agriculture
    - imagery: Satellite and drone imagery processing
    - sensors: IoT sensor data processing and fusion
    - ml: Machine learning models for agricultural predictions

Example Usage:
    >>> import smartagri as sa
    >>> # GPU-accelerated yield prediction
    >>> predictor = sa.crops.YieldPredictor(use_gpu=True)
    >>> yield_forecast = predictor.predict(field_data, weather_data)
    >>>
    >>> # Real-time livestock health monitoring
    >>> monitor = sa.health.HealthMonitor()
    >>> health_status = monitor.analyze(sensor_readings)

Author: SmartAgri Development Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SmartAgri Development Team"
__license__ = "MIT"

# Core computational modules
from smartagri import core
from smartagri import utils

# Agricultural modules
from smartagri import crops
from smartagri import soil
from smartagri import weather
from smartagri import irrigation
from smartagri import pests

# Animal husbandry modules
from smartagri import livestock
from smartagri import health
from smartagri import feed

# Advanced analysis modules
from smartagri import spatial
from smartagri import imagery
from smartagri import sensors
from smartagri import ml

# Convenience imports for common operations
from smartagri.core.gpu import GPUAccelerator, is_gpu_available
from smartagri.core.parallel import ParallelExecutor
from smartagri.core.matrix import MatrixOps

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core modules
    "core",
    "utils",
    # Agricultural modules
    "crops",
    "soil",
    "weather",
    "irrigation",
    "pests",
    # Animal husbandry modules
    "livestock",
    "health",
    "feed",
    # Advanced modules
    "spatial",
    "imagery",
    "sensors",
    "ml",
    # Convenience classes
    "GPUAccelerator",
    "is_gpu_available",
    "ParallelExecutor",
    "MatrixOps",
]
