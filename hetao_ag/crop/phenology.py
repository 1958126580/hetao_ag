# -*- coding: utf-8 -*-
"""
hetao_ag.crop.phenology - 物候期管理

作物生长期跟踪和积温计算。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class GrowthStage(Enum):
    """生长阶段"""
    DORMANT = "dormant"
    EMERGENCE = "emergence"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    GRAIN_FILL = "grain_fill"
    MATURITY = "maturity"
    HARVEST = "harvest"


@dataclass
class PhenologyConfig:
    """物候配置
    
    属性:
        base_temperature: 基础温度(°C)
        stage_gdd: 各阶段所需积温(GDD)
    """
    base_temperature: float = 10.0
    stage_gdd: Dict[str, float] = field(default_factory=lambda: {
        "emergence": 100,
        "vegetative": 400,
        "flowering": 700,
        "grain_fill": 1100,
        "maturity": 1500
    })


# 预定义作物物候参数
CROP_PHENOLOGY = {
    "wheat": PhenologyConfig(
        base_temperature=0.0,
        stage_gdd={"emergence": 120, "vegetative": 450, "flowering": 800, "grain_fill": 1200, "maturity": 1600}
    ),
    "maize": PhenologyConfig(
        base_temperature=10.0,
        stage_gdd={"emergence": 60, "vegetative": 350, "flowering": 700, "grain_fill": 1100, "maturity": 1400}
    ),
    "rice": PhenologyConfig(
        base_temperature=10.0,
        stage_gdd={"emergence": 80, "vegetative": 400, "flowering": 650, "grain_fill": 950, "maturity": 1200}
    ),
    "cotton": PhenologyConfig(
        base_temperature=15.5,
        stage_gdd={"emergence": 80, "vegetative": 500, "flowering": 900, "grain_fill": 1400, "maturity": 1800}
    ),
}


class PhenologyTracker:
    """物候期跟踪器
    
    基于积温(GDD)跟踪作物生长阶段。
    
    示例:
        >>> tracker = PhenologyTracker("wheat")
        >>> tracker.accumulate_gdd(t_max=28, t_min=15)
        >>> print(tracker.current_stage)
    """
    
    def __init__(self, crop: str = "wheat", config: Optional[PhenologyConfig] = None):
        """初始化物候跟踪器
        
        参数:
            crop: 作物类型
            config: 自定义物候配置
        """
        self.crop = crop
        self.config = config or CROP_PHENOLOGY.get(crop, PhenologyConfig())
        
        self.accumulated_gdd = 0.0
        self.days_after_planting = 0
        self.current_stage = GrowthStage.DORMANT
        
        self.gdd_history: List[float] = []
        self.stage_history: List[GrowthStage] = []
    
    def calculate_daily_gdd(self, t_max: float, t_min: float) -> float:
        """计算日积温(GDD)
        
        使用单正弦方法。
        
        参数:
            t_max: 日最高温度(°C)
            t_min: 日最低温度(°C)
            
        返回:
            日积温
        """
        t_base = self.config.base_temperature
        
        # 简化方法: (Tmax + Tmin)/2 - Tbase
        t_mean = (t_max + t_min) / 2
        gdd = max(0, t_mean - t_base)
        
        return gdd
    
    def accumulate_gdd(self, t_max: float, t_min: float) -> float:
        """累积一天的积温并更新阶段
        
        参数:
            t_max: 日最高温
            t_min: 日最低温
            
        返回:
            当日GDD
        """
        daily_gdd = self.calculate_daily_gdd(t_max, t_min)
        self.accumulated_gdd += daily_gdd
        self.days_after_planting += 1
        
        self.gdd_history.append(self.accumulated_gdd)
        
        # 更新生长阶段
        self._update_stage()
        self.stage_history.append(self.current_stage)
        
        return daily_gdd
    
    def _update_stage(self):
        """更新当前生长阶段"""
        stage_gdd = self.config.stage_gdd
        
        if self.accumulated_gdd >= stage_gdd.get("maturity", float('inf')):
            self.current_stage = GrowthStage.MATURITY
        elif self.accumulated_gdd >= stage_gdd.get("grain_fill", float('inf')):
            self.current_stage = GrowthStage.GRAIN_FILL
        elif self.accumulated_gdd >= stage_gdd.get("flowering", float('inf')):
            self.current_stage = GrowthStage.FLOWERING
        elif self.accumulated_gdd >= stage_gdd.get("vegetative", float('inf')):
            self.current_stage = GrowthStage.VEGETATIVE
        elif self.accumulated_gdd >= stage_gdd.get("emergence", float('inf')):
            self.current_stage = GrowthStage.EMERGENCE
    
    def progress_to_maturity(self) -> float:
        """到成熟期的进度(0-1)"""
        maturity_gdd = self.config.stage_gdd.get("maturity", 1500)
        return min(1.0, self.accumulated_gdd / maturity_gdd)
    
    def days_to_maturity(self, avg_daily_gdd: float = 15.0) -> int:
        """估计到成熟期的天数"""
        maturity_gdd = self.config.stage_gdd.get("maturity", 1500)
        remaining_gdd = maturity_gdd - self.accumulated_gdd
        
        if avg_daily_gdd <= 0:
            return 999
        
        return max(0, int(remaining_gdd / avg_daily_gdd))
    
    def get_kc_for_stage(self) -> float:
        """获取当前阶段的作物系数Kc"""
        KC_VALUES = {
            GrowthStage.DORMANT: 0.3,
            GrowthStage.EMERGENCE: 0.3,
            GrowthStage.VEGETATIVE: 0.7,
            GrowthStage.FLOWERING: 1.15,
            GrowthStage.GRAIN_FILL: 1.1,
            GrowthStage.MATURITY: 0.4,
            GrowthStage.HARVEST: 0.3,
        }
        return KC_VALUES.get(self.current_stage, 1.0)
    
    def reset(self):
        """重置跟踪器"""
        self.accumulated_gdd = 0.0
        self.days_after_planting = 0
        self.current_stage = GrowthStage.DORMANT
        self.gdd_history.clear()
        self.stage_history.clear()


def growing_degree_days(
    t_max: np.ndarray,
    t_min: np.ndarray,
    t_base: float = 10.0,
    t_upper: float = 30.0
) -> np.ndarray:
    """批量计算积温序列
    
    参数:
        t_max: 最高温度数组
        t_min: 最低温度数组
        t_base: 基础温度
        t_upper: 上限温度
        
    返回:
        累积GDD数组
    """
    t_max_adj = np.clip(t_max, t_base, t_upper)
    t_min_adj = np.clip(t_min, t_base, t_upper)
    
    daily_gdd = (t_max_adj + t_min_adj) / 2 - t_base
    daily_gdd = np.maximum(0, daily_gdd)
    
    return np.cumsum(daily_gdd)


if __name__ == "__main__":
    print("=" * 50)
    print("物候期跟踪演示")
    print("=" * 50)
    
    tracker = PhenologyTracker("wheat")
    
    # 模拟30天温度
    np.random.seed(42)
    temps = [(25 + np.random.randn()*3, 12 + np.random.randn()*2) for _ in range(30)]
    
    print("\n逐日跟踪:")
    for i, (t_max, t_min) in enumerate(temps, 1):
        gdd = tracker.accumulate_gdd(t_max, t_min)
        if i % 10 == 0:
            print(f"第{i}天: 积温={tracker.accumulated_gdd:.0f}, 阶段={tracker.current_stage.value}")
    
    print(f"\n累积积温: {tracker.accumulated_gdd:.0f}")
    print(f"当前阶段: {tracker.current_stage.value}")
    print(f"成熟进度: {tracker.progress_to_maturity()*100:.1f}%")
    print(f"当前Kc: {tracker.get_kc_for_stage()}")
