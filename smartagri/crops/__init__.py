"""
Crop Management Module for SmartAgri

Comprehensive crop management system providing:
- Crop growth modeling and simulation
- Yield prediction with machine learning
- Phenological stage tracking
- Nutrient requirement calculations
- Harvest timing optimization
- Crop rotation planning

This module integrates with weather, soil, and imagery data for
precision agriculture applications.

Example:
    >>> from smartagri.crops import YieldPredictor, GrowthModel, CropManager
    >>> # Predict crop yield
    >>> predictor = YieldPredictor(crop="corn", use_gpu=True)
    >>> yield_forecast = predictor.predict(weather_data, soil_data)
    >>> # Simulate crop growth
    >>> model = GrowthModel(crop="wheat")
    >>> growth = model.simulate(start_date, end_date, weather)
"""

from smartagri.crops.growth import (
    GrowthModel,
    GrowthStage,
    PhenologyModel,
    BiomassAccumulation,
    LeafAreaIndex,
    RootGrowthModel,
)

from smartagri.crops.yield_prediction import (
    YieldPredictor,
    YieldModel,
    StatisticalYieldModel,
    MLYieldPredictor,
    EnsembleYieldPredictor,
)

from smartagri.crops.management import (
    CropManager,
    PlantingOptimizer,
    HarvestOptimizer,
    CropRotation,
    FertilizerScheduler,
)

from smartagri.crops.parameters import (
    CropParameters,
    get_crop_parameters,
    list_available_crops,
    CROP_DATABASE,
)

__all__ = [
    # Growth modeling
    "GrowthModel",
    "GrowthStage",
    "PhenologyModel",
    "BiomassAccumulation",
    "LeafAreaIndex",
    "RootGrowthModel",
    # Yield prediction
    "YieldPredictor",
    "YieldModel",
    "StatisticalYieldModel",
    "MLYieldPredictor",
    "EnsembleYieldPredictor",
    # Management
    "CropManager",
    "PlantingOptimizer",
    "HarvestOptimizer",
    "CropRotation",
    "FertilizerScheduler",
    # Parameters
    "CropParameters",
    "get_crop_parameters",
    "list_available_crops",
    "CROP_DATABASE",
]
