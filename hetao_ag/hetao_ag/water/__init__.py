# -*- coding: utf-8 -*-
"""
hetao_ag.water - 水循环模块

蒸散发计算、水量平衡和灌溉调度。

作者: Hetao College
版本: 1.0.0
"""

from .evapotranspiration import (
    eto_penman_monteith,
    eto_hargreaves,
    extraterrestrial_radiation,
    crop_coefficient,
    etc_crop,
    WeatherData,
    ETMethod,
)

from .balance import (
    WaterBalance,
    WaterBalanceRecord,
)

from .irrigation import (
    IrrigationScheduler,
    IrrigationEvent,
    IrrigationRecommendation,
    IrrigationMethod,
    ScheduleType,
    calculate_net_irrigation_requirement,
    gross_irrigation_requirement,
)

__all__ = [
    # evapotranspiration
    "eto_penman_monteith",
    "eto_hargreaves",
    "extraterrestrial_radiation",
    "crop_coefficient",
    "etc_crop",
    "WeatherData",
    "ETMethod",
    # balance
    "WaterBalance",
    "WaterBalanceRecord",
    # irrigation
    "IrrigationScheduler",
    "IrrigationEvent",
    "IrrigationRecommendation",
    "IrrigationMethod",
    "ScheduleType",
    "calculate_net_irrigation_requirement",
    "gross_irrigation_requirement",
]
