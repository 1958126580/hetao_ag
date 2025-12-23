# -*- coding: utf-8 -*-
"""
hetao_ag.opt.planning - 农场规划

综合农场规划和场景分析工具。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class FarmScenario:
    """农场情景"""
    name: str
    crop_areas: Dict[str, float]
    irrigation_mm: float
    expected_yield: Dict[str, float]
    total_profit: float
    water_use_efficiency: float


class ScenarioEvaluator:
    """场景评估器
    
    评估和比较不同农场管理方案。
    """
    
    def __init__(
        self,
        crop_params: Dict[str, Dict],
        total_land: float,
        total_water: float
    ):
        self.crop_params = crop_params
        self.total_land = total_land
        self.total_water = total_water
        self.scenarios: List[FarmScenario] = []
    
    def evaluate_scenario(
        self,
        name: str,
        crop_areas: Dict[str, float],
        irrigation_mm: float
    ) -> FarmScenario:
        """评估单个方案"""
        yields = {}
        profit = 0.0
        total_water_used = 0.0
        
        for crop, area in crop_areas.items():
            params = self.crop_params.get(crop, {})
            
            # 基础产量
            base_yield = params.get("yield_kg_ha", 5000)
            
            # 水分响应
            water_need = params.get("water_need_mm", 400)
            water_factor = min(1.0, irrigation_mm / water_need)
            
            actual_yield = base_yield * water_factor * 0.9
            yields[crop] = actual_yield
            
            # 利润
            price = params.get("price_per_kg", 1.0)
            cost = params.get("cost_per_ha", 1000)
            profit += actual_yield * area * price - cost * area
            
            total_water_used += irrigation_mm * area * 10  # m³
        
        wue = sum(y * a for y, a in zip(yields.values(), crop_areas.values())) / total_water_used if total_water_used > 0 else 0
        
        scenario = FarmScenario(
            name=name,
            crop_areas=crop_areas,
            irrigation_mm=irrigation_mm,
            expected_yield=yields,
            total_profit=profit,
            water_use_efficiency=wue
        )
        
        self.scenarios.append(scenario)
        return scenario
    
    def compare_scenarios(self) -> Dict[str, FarmScenario]:
        """比较所有方案"""
        if not self.scenarios:
            return {}
        
        best_profit = max(self.scenarios, key=lambda s: s.total_profit)
        best_wue = max(self.scenarios, key=lambda s: s.water_use_efficiency)
        
        return {
            "best_profit": best_profit,
            "best_water_efficiency": best_wue
        }
    
    def sensitivity_analysis(
        self,
        base_scenario: FarmScenario,
        parameter: str,
        variations: List[float]
    ) -> List[FarmScenario]:
        """敏感性分析"""
        results = []
        
        for var in variations:
            if parameter == "irrigation":
                new_irrig = base_scenario.irrigation_mm * (1 + var)
                scenario = self.evaluate_scenario(
                    f"irrigation_{var:+.0%}",
                    base_scenario.crop_areas,
                    new_irrig
                )
            results.append(scenario)
        
        return results


def multi_objective_score(
    profit: float,
    water_use: float,
    sustainability: float,
    weights: Dict[str, float] = None
) -> float:
    """多目标评分
    
    参数:
        profit: 利润指标(0-1归一化)
        water_use: 水效指标(0-1)
        sustainability: 可持续性指标(0-1)
        weights: 权重
        
    返回:
        综合得分
    """
    w = weights or {"profit": 0.4, "water": 0.35, "sustainability": 0.25}
    
    score = (profit * w["profit"] + 
             water_use * w["water"] + 
             sustainability * w["sustainability"])
    
    return score


if __name__ == "__main__":
    print("=" * 50)
    print("农场规划演示")
    print("=" * 50)
    
    crop_params = {
        "wheat": {"yield_kg_ha": 6000, "water_need_mm": 400, "price_per_kg": 0.8, "cost_per_ha": 1200},
        "maize": {"yield_kg_ha": 10000, "water_need_mm": 600, "price_per_kg": 0.6, "cost_per_ha": 1500},
    }
    
    evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=500000)
    
    # 评估方案
    s1 = evaluator.evaluate_scenario("全小麦", {"wheat": 100}, 450)
    s2 = evaluator.evaluate_scenario("全玉米", {"maize": 100}, 650)
    s3 = evaluator.evaluate_scenario("混种", {"wheat": 50, "maize": 50}, 500)
    
    print("\n方案评估:")
    for s in [s1, s2, s3]:
        print(f"  {s.name}: 利润=¥{s.total_profit:,.0f}, WUE={s.water_use_efficiency:.2f}")
    
    best = evaluator.compare_scenarios()
    print(f"\n最佳利润方案: {best['best_profit'].name}")
    print(f"最佳水效方案: {best['best_water_efficiency'].name}")
