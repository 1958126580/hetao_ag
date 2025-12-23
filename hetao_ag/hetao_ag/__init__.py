# -*- coding: utf-8 -*-
"""
河套智慧农牧业库 (hetao_ag)
===========================

一个面向智慧农业和畜牧业的综合Python库，涵盖土壤建模、水循环管理、
作物生长模拟、畜牧监测、遥感分析和农场优化。

模块:
    - core: 核心功能（单位系统、配置、日志、工具）
    - soil: 土壤水分和盐分建模
    - water: 蒸散发计算和灌溉调度
    - crop: 作物生长和胁迫响应
    - livestock: 动物检测和健康监测
    - space: 遥感植被指数和分类
    - opt: 农业优化和决策支持

快速开始:
    >>> import hetao_ag as hag
    >>> 
    >>> # 计算参考蒸散发
    >>> weather = hag.water.WeatherData(t_mean=25, t_max=32, t_min=18)
    >>> et0 = hag.water.eto_penman_monteith(weather)
    >>> print(f"ET0: {et0:.2f} mm/day")
    >>>
    >>> # 土壤水分模拟
    >>> soil = hag.soil.SoilMoistureModel(field_capacity=0.32)
    >>> soil.step_day(rain_mm=15, et_mm=5)
    >>> print(f"含水量: {soil.moisture:.3f}")

作者: Hetao College
版本: 1.0.0
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "Hetao College"
__license__ = "MIT"

# 导入子模块
from . import core
from . import soil
from . import water
from . import crop
from . import livestock
from . import space
from . import opt

# 常用功能快捷导入
from .core import (
    Unit, Quantity, ConfigManager, get_logger,
    meters, hectares, celsius, ds_per_m,
)

__all__ = [
    "core",
    "soil", 
    "water",
    "crop",
    "livestock",
    "space",
    "opt",
    # 常用
    "Unit",
    "Quantity",
    "ConfigManager",
    "get_logger",
    "meters",
    "hectares",
    "celsius",
    "ds_per_m",
]
