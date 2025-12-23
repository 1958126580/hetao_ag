# -*- coding: utf-8 -*-
"""
hetao_ag.crop.stress - 作物胁迫响应

水分和盐分胁迫对作物的影响模型。

作者: Hetao College
版本: 1.0.0
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CropSaltTolerance:
    """作物盐分耐受性参数
    
    基于Maas-Hoffman模型。
    
    属性:
        threshold: ECe阈值(dS/m),低于此值无减产
        slope: 超过阈值后每dS/m的减产率(0-1)
    """
    threshold: float  # dS/m
    slope: float      # 减产率/dS/m


# 常见作物盐分耐受性参数(Maas & Hoffman, 1977)
CROP_SALT_TOLERANCE = {
    "wheat": CropSaltTolerance(6.0, 0.071),
    "maize": CropSaltTolerance(1.7, 0.120),
    "rice": CropSaltTolerance(3.0, 0.120),
    "cotton": CropSaltTolerance(7.7, 0.052),
    "barley": CropSaltTolerance(8.0, 0.050),
    "sorghum": CropSaltTolerance(6.8, 0.160),
    "soybean": CropSaltTolerance(5.0, 0.200),
    "sunflower": CropSaltTolerance(4.8, 0.050),
    "alfalfa": CropSaltTolerance(2.0, 0.073),
    "potato": CropSaltTolerance(1.7, 0.120),
    "tomato": CropSaltTolerance(2.5, 0.095),
    "pepper": CropSaltTolerance(1.5, 0.140),
}


def yield_reduction_salinity(ECe: float, threshold: float, slope: float) -> float:
    """盐分胁迫产量系数
    
    Maas-Hoffman线性阈值-斜率模型。
    
    参数:
        ECe: 土壤电导率(dS/m)
        threshold: 阈值(dS/m)
        slope: 斜率(减产比例/dS/m)
        
    返回:
        相对产量(0-1)
        
    参考:
        Maas & Hoffman (1977)
    """
    if ECe <= threshold:
        return 1.0
    
    reduction = slope * (ECe - threshold)
    relative_yield = 1.0 - reduction
    
    return max(0.0, relative_yield)


def yield_reduction_salinity_crop(ECe: float, crop: str) -> float:
    """根据作物类型计算盐分胁迫
    
    参数:
        ECe: 土壤电导率(dS/m)
        crop: 作物名称
        
    返回:
        相对产量(0-1)
    """
    tolerance = CROP_SALT_TOLERANCE.get(crop.lower())
    if tolerance is None:
        # 默认使用中等敏感作物参数
        tolerance = CropSaltTolerance(4.0, 0.10)
    
    return yield_reduction_salinity(ECe, tolerance.threshold, tolerance.slope)


def water_stress_factor(
    actual_et: float,
    potential_et: float,
    p: float = 0.5
) -> float:
    """水分胁迫因子
    
    参数:
        actual_et: 实际蒸散发
        potential_et: 潜在蒸散发
        p: 可耗水量系数(默认0.5)
        
    返回:
        水分胁迫因子Ks (0-1)
    """
    if potential_et <= 0:
        return 1.0
    
    ratio = actual_et / potential_et
    return min(1.0, max(0.0, ratio))


def water_stress_from_moisture(
    soil_moisture: float,
    field_capacity: float,
    wilting_point: float,
    p: float = 0.5
) -> float:
    """基于土壤水分的胁迫因子
    
    当土壤水分低于临界点时产生胁迫。
    
    参数:
        soil_moisture: 当前含水量
        field_capacity: 田间持水量
        wilting_point: 凋萎点
        p: 可耗水量系数
        
    返回:
        Ks (0-1)
    """
    taw = field_capacity - wilting_point  # 总可用水
    raw = p * taw  # 易可用水
    
    if soil_moisture >= field_capacity - raw:
        return 1.0
    elif soil_moisture <= wilting_point:
        return 0.0
    else:
        # 在raw和凋萎点之间线性下降
        return (soil_moisture - wilting_point) / (field_capacity - raw - wilting_point + taw * (1 - p))


def combined_stress_factor(
    water_stress: float,
    salinity_stress: float,
    method: str = "multiplicative"
) -> float:
    """组合胁迫因子
    
    参数:
        water_stress: 水分胁迫因子
        salinity_stress: 盐分胁迫因子
        method: 组合方法(multiplicative, minimum, additive)
        
    返回:
        组合胁迫因子(0-1)
    """
    if method == "multiplicative":
        return water_stress * salinity_stress
    elif method == "minimum":
        return min(water_stress, salinity_stress)
    elif method == "additive":
        return max(0, 1 - (1 - water_stress) - (1 - salinity_stress))
    else:
        return water_stress * salinity_stress


def yield_with_stress(
    potential_yield: float,
    water_stress: float = 1.0,
    salinity_stress: float = 1.0,
    other_stress: float = 1.0
) -> float:
    """计算胁迫条件下的产量
    
    参数:
        potential_yield: 潜在产量(kg/ha)
        water_stress: 水分胁迫因子
        salinity_stress: 盐分胁迫因子
        other_stress: 其他胁迫因子
        
    返回:
        实际产量(kg/ha)
    """
    stress = water_stress * salinity_stress * other_stress
    return potential_yield * stress


def classify_salt_tolerance(crop: str) -> str:
    """作物盐分耐受性分级
    
    参数:
        crop: 作物名称
        
    返回:
        耐盐等级描述
    """
    tolerance = CROP_SALT_TOLERANCE.get(crop.lower())
    if tolerance is None:
        return "未知"
    
    threshold = tolerance.threshold
    
    if threshold >= 8.0:
        return "高度耐盐"
    elif threshold >= 6.0:
        return "中度耐盐"
    elif threshold >= 4.0:
        return "轻度耐盐"
    elif threshold >= 2.0:
        return "敏感"
    else:
        return "非常敏感"


if __name__ == "__main__":
    print("=" * 50)
    print("作物胁迫响应演示")
    print("=" * 50)
    
    # 盐分胁迫
    print("\n【盐分胁迫】")
    crops = ["wheat", "maize", "cotton", "barley"]
    ECe = 8.0
    
    for crop in crops:
        rel_yield = yield_reduction_salinity_crop(ECe, crop)
        tolerance = classify_salt_tolerance(crop)
        print(f"  {crop}: ECe={ECe} dS/m时产量={rel_yield*100:.1f}% ({tolerance})")
    
    # 水分胁迫
    print("\n【水分胁迫】")
    ks = water_stress_from_moisture(0.18, 0.32, 0.12, p=0.5)
    print(f"  土壤含水量0.18时Ks={ks:.3f}")
    
    # 组合胁迫
    print("\n【组合胁迫】")
    combined = combined_stress_factor(ks, 0.85)
    yield_actual = yield_with_stress(5000, ks, 0.85)
    print(f"  组合因子={combined:.3f}")
    print(f"  潜在产量5000kg/ha -> 实际{yield_actual:.0f}kg/ha")
