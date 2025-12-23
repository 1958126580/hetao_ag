# -*- coding: utf-8 -*-
"""
hetao_ag.soil.salinity - 土壤盐分模型

土壤盐分动态模拟，用于盐碱地管理和灌溉决策。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SalinityState:
    """盐分状态
    
    属性:
        ECe: 饱和泥浆提取液电导率(dS/m)
        salt_mass_kg_ha: 盐分总量(kg/ha)
        leaching_fraction: 淋洗系数
    """
    ECe: float
    salt_mass_kg_ha: float = 0.0
    leaching_fraction: float = 0.0


class SalinityModel:
    """土壤盐分模型
    
    基于质量平衡原理模拟土壤盐分累积和淋洗过程。
    符合FAO灌溉排水指南标准。
    
    示例:
        >>> model = SalinityModel(initial_ECe=2.0)
        >>> model.irrigate(100, ec_water=1.5)  # 灌溉100mm, 水EC=1.5 dS/m
        >>> print(f"土壤EC: {model.ECe:.2f} dS/m")
    """
    
    # EC到盐浓度的近似转换: TDS (mg/L) ≈ EC (dS/m) × 640
    EC_TO_TDS_FACTOR = 640
    
    def __init__(
        self,
        initial_ECe: float = 2.0,
        root_depth_m: float = 0.3,
        soil_water_content: float = 0.30,
        bulk_density: float = 1.4
    ):
        """初始化盐分模型
        
        参数:
            initial_ECe: 初始土壤EC (dS/m)
            root_depth_m: 根区深度(m)
            soil_water_content: 土壤含水量
            bulk_density: 土壤容重(g/cm³)
        """
        self.ECe = initial_ECe
        self.root_depth_m = root_depth_m
        self.soil_water_content = soil_water_content
        self.bulk_density = bulk_density
        
        # 计算初始盐量
        self.salt_mass = self._ec_to_salt_mass(initial_ECe)
        
        self.history: List[float] = [initial_ECe]
    
    def _ec_to_salt_mass(self, ec: float) -> float:
        """将EC转换为盐质量(kg/ha)"""
        # 土壤水体积 = 深度 × 含水量 × 面积(1ha = 10000m²)
        water_volume_L = self.root_depth_m * self.soil_water_content * 10000 * 1000
        # 盐浓度(mg/L)
        tds_mg_L = ec * self.EC_TO_TDS_FACTOR
        # 盐质量(kg)
        salt_kg = tds_mg_L * water_volume_L / 1e6
        return salt_kg
    
    def _salt_mass_to_ec(self, salt_kg: float) -> float:
        """将盐质量转换为EC"""
        water_volume_L = self.root_depth_m * self.soil_water_content * 10000 * 1000
        if water_volume_L <= 0:
            return 0.0
        tds_mg_L = salt_kg * 1e6 / water_volume_L
        return tds_mg_L / self.EC_TO_TDS_FACTOR
    
    def irrigate(self, amount_mm: float, ec_water: float) -> dict:
        """灌溉过程
        
        参数:
            amount_mm: 灌溉量(mm)
            ec_water: 灌溉水EC (dS/m)
            
        返回:
            盐分收支详情
        """
        # 灌溉水带入的盐量
        water_volume_L = amount_mm * 10000  # L/ha
        salt_input_kg = ec_water * self.EC_TO_TDS_FACTOR * water_volume_L / 1e6
        
        self.salt_mass += salt_input_kg
        self.ECe = self._salt_mass_to_ec(self.salt_mass)
        self.history.append(self.ECe)
        
        return {
            "salt_input_kg_ha": salt_input_kg,
            "ECe": self.ECe
        }
    
    def leach(self, drainage_mm: float) -> dict:
        """淋洗过程
        
        参数:
            drainage_mm: 排水量(mm)
            
        返回:
            淋洗详情
        """
        # 淋洗系数 = 排水量 / 根区水量
        root_water_mm = self.root_depth_m * self.soil_water_content * 1000
        
        if root_water_mm > 0:
            leaching_fraction = min(1.0, drainage_mm / root_water_mm)
        else:
            leaching_fraction = 0.0
        
        # 盐分随水淋洗
        salt_removed = self.salt_mass * leaching_fraction * 0.8  # 假设80%效率
        self.salt_mass -= salt_removed
        self.ECe = self._salt_mass_to_ec(self.salt_mass)
        self.history.append(self.ECe)
        
        return {
            "salt_removed_kg_ha": salt_removed,
            "leaching_fraction": leaching_fraction,
            "ECe": self.ECe
        }
    
    def leaching_requirement(self, ec_irrigation: float, ec_threshold: float) -> float:
        """计算淋洗需水量
        
        根据FAO方法计算维持土壤盐分在阈值以下所需的淋洗系数。
        
        参数:
            ec_irrigation: 灌溉水EC (dS/m)
            ec_threshold: 目标土壤EC阈值 (dS/m)
            
        返回:
            淋洗需求系数(0-1)
        """
        if ec_threshold <= 0:
            return 1.0
        
        # FAO简化公式: LR = ECiw / (5 * ECe - ECiw)
        denominator = 5 * ec_threshold - ec_irrigation
        if denominator <= 0:
            return 1.0
        
        lr = ec_irrigation / denominator
        return min(1.0, max(0.0, lr))
    
    def step_day(self, irrigation_mm: float = 0, ec_irrigation: float = 1.0,
                 drainage_mm: float = 0) -> dict:
        """模拟一天的盐分变化
        
        参数:
            irrigation_mm: 灌溉量
            ec_irrigation: 灌溉水EC
            drainage_mm: 排水量
            
        返回:
            当日盐分收支
        """
        result = {"ECe_start": self.ECe}
        
        if irrigation_mm > 0:
            irr_result = self.irrigate(irrigation_mm, ec_irrigation)
            result["salt_input_kg_ha"] = irr_result["salt_input_kg_ha"]
        
        if drainage_mm > 0:
            leach_result = self.leach(drainage_mm)
            result["salt_removed_kg_ha"] = leach_result["salt_removed_kg_ha"]
        
        result["ECe_end"] = self.ECe
        return result
    
    def get_history_array(self) -> np.ndarray:
        """获取EC历史数组"""
        return np.array(self.history)


def classify_soil_salinity(ECe: float) -> str:
    """土壤盐分分级
    
    参数:
        ECe: 土壤电导率(dS/m)
        
    返回:
        盐分等级描述
    """
    if ECe < 2:
        return "非盐渍化(适合大多数作物)"
    elif ECe < 4:
        return "轻度盐渍化(敏感作物受影响)"
    elif ECe < 8:
        return "中度盐渍化(多数作物减产)"
    elif ECe < 16:
        return "重度盐渍化(仅耐盐作物可种植)"
    else:
        return "极重度盐渍化(不适宜种植)"


def classify_water_salinity(EC: float) -> str:
    """灌溉水盐分分级
    
    参数:
        EC: 水电导率(dS/m)
        
    返回:
        水质等级
    """
    if EC < 0.7:
        return "优良(无限制)"
    elif EC < 3.0:
        return "良好(轻度限制)"
    else:
        return "较差(重度限制)"


if __name__ == "__main__":
    print("=" * 50)
    print("土壤盐分模型演示")
    print("=" * 50)
    
    model = SalinityModel(initial_ECe=3.0, root_depth_m=0.3)
    
    print(f"\n初始EC: {model.ECe:.2f} dS/m")
    print(f"等级: {classify_soil_salinity(model.ECe)}")
    
    # 灌溉
    result = model.irrigate(100, ec_water=2.0)
    print(f"\n灌溉后EC: {model.ECe:.2f} dS/m")
    
    # 淋洗
    result = model.leach(50)
    print(f"淋洗后EC: {model.ECe:.2f} dS/m")
    
    # 计算淋洗需求
    lr = model.leaching_requirement(ec_irrigation=1.5, ec_threshold=4.0)
    print(f"\n淋洗需求系数: {lr:.3f}")
