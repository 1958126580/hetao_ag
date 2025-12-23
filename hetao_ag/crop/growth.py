# -*- coding: utf-8 -*-
"""
hetao_ag.crop.growth - 作物生长模型

作物生物量累积和产量模拟。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

from .stress import yield_reduction_salinity_crop, water_stress_from_moisture, combined_stress_factor
from .phenology import PhenologyTracker, GrowthStage


@dataclass
class CropConfig:
    """作物配置参数"""
    name: str = "wheat"
    potential_yield_kg_ha: float = 6000.0
    harvest_index: float = 0.45
    transpiration_efficiency: float = 20.0  # kg/ha per mm
    max_lai: float = 5.0
    salt_threshold: float = 6.0
    salt_slope: float = 0.071


# 预定义作物配置
CROP_CONFIGS = {
    "wheat": CropConfig("wheat", 6000, 0.45, 20, 5.0, 6.0, 0.071),
    "maize": CropConfig("maize", 10000, 0.50, 25, 6.0, 1.7, 0.12),
    "rice": CropConfig("rice", 8000, 0.50, 15, 7.0, 3.0, 0.12),
    "cotton": CropConfig("cotton", 4000, 0.35, 18, 4.0, 7.7, 0.052),
    "sunflower": CropConfig("sunflower", 3000, 0.35, 15, 4.5, 4.8, 0.05),
}


class CropModel:
    """作物生长模型
    
    模拟作物生物量累积,考虑水分和盐分胁迫。
    
    示例:
        >>> model = CropModel("wheat")
        >>> for day in range(120):
        ...     model.update_daily(et=5, soil_moisture=0.25, ECe=3.0)
        >>> print(f"产量: {model.estimate_yield():.0f} kg/ha")
    """
    
    def __init__(self, crop: str = "wheat", config: Optional[CropConfig] = None):
        """初始化作物模型
        
        参数:
            crop: 作物类型
            config: 自定义配置
        """
        self.crop = crop
        self.config = config or CROP_CONFIGS.get(crop, CropConfig())
        
        self.phenology = PhenologyTracker(crop)
        
        self.accumulated_biomass = 0.0  # kg/ha
        self.lai = 0.0
        self.days_after_planting = 0
        
        self.daily_biomass: List[float] = []
        self.stress_history: List[float] = []
    
    def update_daily(
        self,
        t_max: float = 25.0,
        t_min: float = 15.0,
        et: float = 5.0,
        soil_moisture: float = 0.25,
        field_capacity: float = 0.32,
        wilting_point: float = 0.12,
        ECe: float = 2.0
    ) -> Dict:
        """更新一天的生长
        
        参数:
            t_max, t_min: 日最高/最低温度(°C)
            et: 参考蒸散发(mm/day)
            soil_moisture: 土壤含水量
            field_capacity: 田间持水量
            wilting_point: 凋萎点
            ECe: 土壤电导率(dS/m)
            
        返回:
            当日生长详情
        """
        self.days_after_planting += 1
        
        # 更新物候期
        self.phenology.accumulate_gdd(t_max, t_min)
        
        # 获取作物系数
        kc = self.phenology.get_kc_for_stage()
        
        # 计算作物蒸散发
        etc = et * kc
        
        # 水分胁迫
        ks_water = water_stress_from_moisture(
            soil_moisture, field_capacity, wilting_point
        )
        
        # 盐分胁迫
        ks_salt = yield_reduction_salinity_crop(ECe, self.crop)
        
        # 组合胁迫
        ks_combined = combined_stress_factor(ks_water, ks_salt)
        self.stress_history.append(ks_combined)
        
        # 实际蒸散发
        actual_et = etc * ks_water
        
        # 生物量累积(基于蒸腾效率)
        potential_biomass = self.config.transpiration_efficiency * actual_et
        actual_biomass = potential_biomass * ks_salt  # 盐分降低生物量转化
        
        self.accumulated_biomass += actual_biomass
        self.daily_biomass.append(self.accumulated_biomass)
        
        # 更新LAI(简化模型)
        progress = self.phenology.progress_to_maturity()
        if progress < 0.5:
            self.lai = self.config.max_lai * (progress / 0.5)
        else:
            self.lai = self.config.max_lai * (1 - (progress - 0.5) / 0.5)
        self.lai = max(0, self.lai)
        
        return {
            "day": self.days_after_planting,
            "stage": self.phenology.current_stage.value,
            "biomass_kg_ha": self.accumulated_biomass,
            "lai": self.lai,
            "stress_factor": ks_combined,
            "actual_et_mm": actual_et
        }
    
    def estimate_yield(self) -> float:
        """估算最终产量
        
        返回:
            产量(kg/ha)
        """
        # 使用收获指数从生物量估算产量
        grain_yield = self.accumulated_biomass * self.config.harvest_index
        
        # 考虑季节平均胁迫
        if self.stress_history:
            avg_stress = np.mean(self.stress_history)
        else:
            avg_stress = 1.0
        
        # 产量不超过潜在产量
        return min(grain_yield, self.config.potential_yield_kg_ha * avg_stress)
    
    def water_use_efficiency(self, total_et_mm: float) -> float:
        """计算水分利用效率
        
        参数:
            total_et_mm: 总蒸散发量
            
        返回:
            WUE (kg/m³)
        """
        if total_et_mm <= 0:
            return 0.0
        
        yield_kg = self.estimate_yield()
        water_m3_ha = total_et_mm * 10  # mm to m³/ha
        
        return yield_kg / water_m3_ha
    
    def reset(self):
        """重置模型"""
        self.phenology.reset()
        self.accumulated_biomass = 0.0
        self.lai = 0.0
        self.days_after_planting = 0
        self.daily_biomass.clear()
        self.stress_history.clear()


def simulate_growing_season(
    crop: str,
    weather: List[Dict],
    soil_moisture: float = 0.25,
    ECe: float = 2.0
) -> Dict:
    """模拟一个生长季
    
    参数:
        crop: 作物类型
        weather: 气象数据列表 [{t_max, t_min, et}, ...]
        soil_moisture: 平均土壤含水量
        ECe: 土壤电导率
        
    返回:
        模拟结果
    """
    model = CropModel(crop)
    
    for day_weather in weather:
        model.update_daily(
            t_max=day_weather.get("t_max", 25),
            t_min=day_weather.get("t_min", 15),
            et=day_weather.get("et", 5),
            soil_moisture=soil_moisture,
            ECe=ECe
        )
    
    return {
        "crop": crop,
        "days": model.days_after_planting,
        "final_stage": model.phenology.current_stage.value,
        "biomass_kg_ha": model.accumulated_biomass,
        "yield_kg_ha": model.estimate_yield(),
        "avg_stress": np.mean(model.stress_history) if model.stress_history else 1.0
    }


if __name__ == "__main__":
    print("=" * 50)
    print("作物生长模型演示")
    print("=" * 50)
    
    model = CropModel("wheat")
    
    # 模拟120天生长季
    np.random.seed(42)
    
    print("\n逐日模拟:")
    for day in range(120):
        t_max = 22 + 10 * np.sin(day / 120 * np.pi) + np.random.randn()
        t_min = t_max - 10 + np.random.randn()
        et = 4 + 2 * np.sin(day / 120 * np.pi)
        
        result = model.update_daily(
            t_max=t_max, t_min=t_min, et=et,
            soil_moisture=0.25, ECe=3.0
        )
        
        if day % 30 == 0:
            print(f"第{result['day']}天: 阶段={result['stage']}, "
                  f"生物量={result['biomass_kg_ha']:.0f}kg/ha")
    
    final_yield = model.estimate_yield()
    print(f"\n最终产量: {final_yield:.0f} kg/ha")
    print(f"平均胁迫因子: {np.mean(model.stress_history):.3f}")
