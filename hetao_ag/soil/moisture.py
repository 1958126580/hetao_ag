# -*- coding: utf-8 -*-
"""
hetao_ag.soil.moisture - 土壤水分模型

提供土壤水分动态模拟，包括入渗、蒸发、渗透等过程。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class SoilType(Enum):
    """土壤类型枚举"""
    SAND = "sand"
    LOAMY_SAND = "loamy_sand"
    SANDY_LOAM = "sandy_loam"
    LOAM = "loam"
    SILT_LOAM = "silt_loam"
    SILT = "silt"
    SANDY_CLAY_LOAM = "sandy_clay_loam"
    CLAY_LOAM = "clay_loam"
    SILTY_CLAY_LOAM = "silty_clay_loam"
    SANDY_CLAY = "sandy_clay"
    SILTY_CLAY = "silty_clay"
    CLAY = "clay"


# 土壤水力特性参数(van Genuchten参数)
SOIL_PARAMETERS = {
    SoilType.SAND: {"theta_r": 0.045, "theta_s": 0.43, "alpha": 0.145, "n": 2.68, "Ks": 712.8},
    SoilType.LOAMY_SAND: {"theta_r": 0.057, "theta_s": 0.41, "alpha": 0.124, "n": 2.28, "Ks": 350.2},
    SoilType.SANDY_LOAM: {"theta_r": 0.065, "theta_s": 0.41, "alpha": 0.075, "n": 1.89, "Ks": 106.1},
    SoilType.LOAM: {"theta_r": 0.078, "theta_s": 0.43, "alpha": 0.036, "n": 1.56, "Ks": 25.0},
    SoilType.SILT_LOAM: {"theta_r": 0.067, "theta_s": 0.45, "alpha": 0.020, "n": 1.41, "Ks": 10.8},
    SoilType.SILT: {"theta_r": 0.034, "theta_s": 0.46, "alpha": 0.016, "n": 1.37, "Ks": 6.0},
    SoilType.SANDY_CLAY_LOAM: {"theta_r": 0.100, "theta_s": 0.39, "alpha": 0.059, "n": 1.48, "Ks": 31.4},
    SoilType.CLAY_LOAM: {"theta_r": 0.095, "theta_s": 0.41, "alpha": 0.019, "n": 1.31, "Ks": 6.2},
    SoilType.SILTY_CLAY_LOAM: {"theta_r": 0.089, "theta_s": 0.43, "alpha": 0.010, "n": 1.23, "Ks": 1.7},
    SoilType.SANDY_CLAY: {"theta_r": 0.100, "theta_s": 0.38, "alpha": 0.027, "n": 1.23, "Ks": 2.9},
    SoilType.SILTY_CLAY: {"theta_r": 0.070, "theta_s": 0.36, "alpha": 0.005, "n": 1.09, "Ks": 0.5},
    SoilType.CLAY: {"theta_r": 0.068, "theta_s": 0.38, "alpha": 0.008, "n": 1.09, "Ks": 4.8},
}


@dataclass
class SoilLayer:
    """土壤层数据类
    
    属性:
        depth_m: 层厚度(m)
        moisture: 体积含水量(m³/m³)
        field_capacity: 田间持水量
        wilting_point: 凋萎点
        saturation: 饱和含水量
        soil_type: 土壤类型
    """
    depth_m: float
    moisture: float
    field_capacity: float = 0.30
    wilting_point: float = 0.12
    saturation: float = 0.45
    soil_type: SoilType = SoilType.LOAM
    
    def __post_init__(self):
        params = SOIL_PARAMETERS.get(self.soil_type, {})
        if params and self.saturation == 0.45:
            self.saturation = params.get("theta_s", 0.45)
            self.wilting_point = params.get("theta_r", 0.12) * 1.5
    
    @property
    def available_water(self) -> float:
        """可利用水分(相对于凋萎点)"""
        return max(0, self.moisture - self.wilting_point)
    
    @property
    def deficit_to_fc(self) -> float:
        """到田间持水量的亏缺"""
        return max(0, self.field_capacity - self.moisture)
    
    @property
    def relative_saturation(self) -> float:
        """相对饱和度(0-1)"""
        return (self.moisture - self.wilting_point) / (self.saturation - self.wilting_point)


class SoilMoistureModel:
    """土壤水分模型
    
    模拟土壤水分动态变化,包括降水入渗、蒸散发消耗、深层渗透等。
    
    示例:
        >>> model = SoilMoistureModel(field_capacity=0.32, wilting_point=0.12)
        >>> model.add_water(15.0)  # 添加15mm水
        >>> model.remove_water(5.0)  # 移除5mm(蒸散发)
        >>> print(model.moisture)
    """
    
    def __init__(
        self,
        field_capacity: float = 0.30,
        wilting_point: float = 0.12,
        initial_moisture: float = 0.25,
        root_depth_m: float = 0.3,
        soil_type: SoilType = SoilType.LOAM
    ):
        """初始化土壤水分模型
        
        参数:
            field_capacity: 田间持水量(体积含水量)
            wilting_point: 凋萎点
            initial_moisture: 初始含水量
            root_depth_m: 根区深度(m)
            soil_type: 土壤类型
        """
        self.field_capacity = field_capacity
        self.wilting_point = wilting_point
        self.moisture = initial_moisture
        self.root_depth_m = root_depth_m
        self.soil_type = soil_type
        
        params = SOIL_PARAMETERS.get(soil_type, {})
        self.saturation = params.get("theta_s", 0.45)
        self.Ks = params.get("Ks", 25.0)  # 饱和导水率 mm/day
        
        self.history: List[float] = [initial_moisture]
        self.runoff_mm: float = 0.0
        self.drainage_mm: float = 0.0
    
    def add_water(self, amount_mm: float) -> Tuple[float, float]:
        """添加水分(降水/灌溉)
        
        参数:
            amount_mm: 水量(mm)
            
        返回:
            (实际入渗量mm, 地表径流mm)
        """
        # 转换mm水深到体积含水量变化
        added_theta = amount_mm / (self.root_depth_m * 1000)
        
        new_moisture = self.moisture + added_theta
        
        # 超过饱和的部分变成径流
        if new_moisture > self.saturation:
            runoff_theta = new_moisture - self.saturation
            runoff_mm = runoff_theta * self.root_depth_m * 1000
            new_moisture = self.saturation
        else:
            runoff_mm = 0.0
        
        infiltration = amount_mm - runoff_mm
        self.moisture = new_moisture
        self.runoff_mm += runoff_mm
        
        return infiltration, runoff_mm
    
    def remove_water(self, amount_mm: float) -> float:
        """移除水分(蒸散发)
        
        参数:
            amount_mm: 移除量(mm)
            
        返回:
            实际移除量(mm)
        """
        removed_theta = amount_mm / (self.root_depth_m * 1000)
        
        new_moisture = self.moisture - removed_theta
        
        # 不能低于凋萎点
        if new_moisture < self.wilting_point:
            actual_removed_theta = self.moisture - self.wilting_point
            new_moisture = self.wilting_point
        else:
            actual_removed_theta = removed_theta
        
        self.moisture = new_moisture
        return actual_removed_theta * self.root_depth_m * 1000
    
    def deep_percolation(self) -> float:
        """计算深层渗透(超过田间持水量的水分)
        
        返回:
            渗透量(mm)
        """
        if self.moisture > self.field_capacity:
            excess_theta = self.moisture - self.field_capacity
            # 简化模型：假设一天内过量水分下渗
            drainage = excess_theta * self.root_depth_m * 1000
            self.moisture = self.field_capacity
            self.drainage_mm += drainage
            return drainage
        return 0.0
    
    def step_day(self, rain_mm: float = 0, irrigation_mm: float = 0, et_mm: float = 0) -> dict:
        """模拟一天的水分变化
        
        参数:
            rain_mm: 降水量
            irrigation_mm: 灌溉量
            et_mm: 蒸散发量
            
        返回:
            当日水分收支详情
        """
        # 1. 入渗
        total_input = rain_mm + irrigation_mm
        infiltration, runoff = self.add_water(total_input)
        
        # 2. 蒸散发
        actual_et = self.remove_water(et_mm)
        
        # 3. 深层渗透
        drainage = self.deep_percolation()
        
        # 记录历史
        self.history.append(self.moisture)
        
        return {
            "moisture": self.moisture,
            "infiltration_mm": infiltration,
            "runoff_mm": runoff,
            "et_mm": actual_et,
            "drainage_mm": drainage
        }
    
    @property
    def stress_factor(self) -> float:
        """水分胁迫因子(0-1, 1表示无胁迫)"""
        if self.moisture >= self.field_capacity:
            return 1.0
        elif self.moisture <= self.wilting_point:
            return 0.0
        else:
            return (self.moisture - self.wilting_point) / (self.field_capacity - self.wilting_point)
    
    @property
    def irrigation_need_mm(self) -> float:
        """需要灌溉量(补充到田间持水量)"""
        deficit = self.field_capacity - self.moisture
        return max(0, deficit * self.root_depth_m * 1000)
    
    def get_history_array(self) -> np.ndarray:
        """获取含水量历史数组"""
        return np.array(self.history)


def van_genuchten_theta(h: float, theta_r: float, theta_s: float, alpha: float, n: float) -> float:
    """van Genuchten土壤水分特征曲线
    
    参数:
        h: 压力水头(cm, 负值)
        theta_r: 残余含水量
        theta_s: 饱和含水量
        alpha: 形状参数
        n: 形状参数
        
    返回:
        体积含水量
    """
    if h >= 0:
        return theta_s
    
    m = 1 - 1/n
    return theta_r + (theta_s - theta_r) / (1 + (alpha * abs(h)) ** n) ** m


if __name__ == "__main__":
    print("=" * 50)
    print("土壤水分模型演示")
    print("=" * 50)
    
    model = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.25,
        root_depth_m=0.3,
        soil_type=SoilType.LOAM
    )
    
    print(f"\n初始含水量: {model.moisture:.3f}")
    print(f"胁迫因子: {model.stress_factor:.3f}")
    
    # 模拟5天
    weather = [
        {"rain": 10, "et": 4},
        {"rain": 0, "et": 5},
        {"rain": 0, "et": 5},
        {"rain": 20, "et": 3},
        {"rain": 0, "et": 6},
    ]
    
    print("\n逐日模拟:")
    for i, w in enumerate(weather, 1):
        result = model.step_day(rain_mm=w["rain"], et_mm=w["et"])
        print(f"第{i}天: 含水量={result['moisture']:.3f}, ET={result['et_mm']:.1f}mm")
    
    print(f"\n最终含水量: {model.moisture:.3f}")
    print(f"需灌溉量: {model.irrigation_need_mm:.1f} mm")
