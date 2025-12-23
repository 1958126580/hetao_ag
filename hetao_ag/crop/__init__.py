# -*- coding: utf-8 -*-
"""
hetao_ag.crop - 作物模块

作物生长模拟、胁迫响应和物候期管理。

作者: Hetao College
版本: 1.0.0
"""

from .stress import (
    yield_reduction_salinity,
    yield_reduction_salinity_crop,
    water_stress_factor,
    water_stress_from_moisture,
    combined_stress_factor,
    yield_with_stress,
    classify_salt_tolerance,
    CropSaltTolerance,
    CROP_SALT_TOLERANCE,
)

from .phenology import (
    PhenologyTracker,
    PhenologyConfig,
    GrowthStage,
    growing_degree_days,
    CROP_PHENOLOGY,
)

from .growth import (
    CropModel,
    CropConfig,
    simulate_growing_season,
    CROP_CONFIGS,
)

__all__ = [
    # stress
    "yield_reduction_salinity",
    "yield_reduction_salinity_crop",
    "water_stress_factor",
    "water_stress_from_moisture",
    "combined_stress_factor",
    "yield_with_stress",
    "classify_salt_tolerance",
    "CropSaltTolerance",
    "CROP_SALT_TOLERANCE",
    # phenology
    "PhenologyTracker",
    "PhenologyConfig",
    "GrowthStage",
    "growing_degree_days",
    "CROP_PHENOLOGY",
    # growth
    "CropModel",
    "CropConfig",
    "simulate_growing_season",
    "CROP_CONFIGS",
]
