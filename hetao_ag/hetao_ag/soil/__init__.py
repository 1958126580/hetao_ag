# -*- coding: utf-8 -*-
"""
hetao_ag.soil - 土壤模块

土壤水分和盐分建模、IoT传感器校准。

作者: Hetao College
版本: 1.0.0
"""

from .moisture import (
    SoilMoistureModel,
    SoilLayer,
    SoilType,
    SOIL_PARAMETERS,
    van_genuchten_theta,
)

from .salinity import (
    SalinityModel,
    SalinityState,
    classify_soil_salinity,
    classify_water_salinity,
)

from .sensors import (
    SensorCalibrator,
    CalibrationResult,
    CalibrationMethod,
    MoistureSensor,
    capacitive_sensor_formula,
)

__all__ = [
    # moisture
    "SoilMoistureModel",
    "SoilLayer",
    "SoilType",
    "SOIL_PARAMETERS",
    "van_genuchten_theta",
    # salinity
    "SalinityModel",
    "SalinityState",
    "classify_soil_salinity",
    "classify_water_salinity",
    # sensors
    "SensorCalibrator",
    "CalibrationResult",
    "CalibrationMethod",
    "MoistureSensor",
    "capacitive_sensor_formula",
]
