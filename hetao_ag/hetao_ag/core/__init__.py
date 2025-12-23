# -*- coding: utf-8 -*-
"""
hetao_ag.core - 核心功能模块
=============================

提供智慧农牧业库的基础组件：单位系统、配置管理、日志工具和通用函数。

模块:
    - units: SI单位系统和物理量类
    - config: 配置管理器
    - logger: 日志工具
    - utils: 通用工具函数

作者: Hetao College
版本: 1.0.0
"""

from .units import (
    Unit,
    Quantity,
    Dimension,
    DimensionError,
    meters,
    kilometers,
    hectares,
    celsius,
    kilopascals,
    ds_per_m,
    megajoules_per_m2,
    mm_per_day,
)

from .config import (
    ConfigManager,
    ConfigError,
    create_default_config,
)

from .logger import (
    Logger,
    get_logger,
    ColoredFormatter,
)

from .utils import (
    safe_divide,
    clamp,
    linear_interpolate,
    array_interpolate,
    day_of_year,
    degrees_to_radians,
    radians_to_degrees,
    moving_average,
    normalize,
    rmse,
    mae,
    r_squared,
    ensure_path,
    ensure_directory,
    validate_model,
    ValidationResult,
    Timer,
)

__version__ = "1.0.0"

__all__ = [
    # units
    "Unit",
    "Quantity",
    "Dimension",
    "DimensionError",
    "meters",
    "kilometers",
    "hectares",
    "celsius",
    "kilopascals",
    "ds_per_m",
    "megajoules_per_m2",
    "mm_per_day",
    # config
    "ConfigManager",
    "ConfigError",
    "create_default_config",
    # logger
    "Logger",
    "get_logger",
    "ColoredFormatter",
    # utils
    "safe_divide",
    "clamp",
    "linear_interpolate",
    "array_interpolate",
    "day_of_year",
    "degrees_to_radians",
    "radians_to_degrees",
    "moving_average",
    "normalize",
    "rmse",
    "mae",
    "r_squared",
    "ensure_path",
    "ensure_directory",
    "validate_model",
    "ValidationResult",
    "Timer",
]
