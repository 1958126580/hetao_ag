# -*- coding: utf-8 -*-
"""
hetao_ag.water.irrigation - 灌溉调度

灌溉决策和调度工具。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class IrrigationMethod(Enum):
    """灌溉方法"""
    FLOOD = "flood"
    SPRINKLER = "sprinkler"
    DRIP = "drip"
    CENTER_PIVOT = "center_pivot"


class ScheduleType(Enum):
    """调度类型"""
    FIXED_INTERVAL = "fixed_interval"
    SOIL_MOISTURE = "soil_moisture"
    ET_BASED = "et_based"
    DEFICIT = "deficit"


@dataclass
class IrrigationEvent:
    """灌溉事件"""
    day: int
    amount_mm: float
    method: IrrigationMethod = IrrigationMethod.FLOOD
    duration_hours: float = 0.0
    ec_water: float = 1.0


@dataclass
class IrrigationRecommendation:
    """灌溉建议"""
    should_irrigate: bool
    amount_mm: float
    reason: str
    urgency: str = "normal"  # low, normal, high, critical


class IrrigationScheduler:
    """灌溉调度器
    
    基于土壤水分、ET或固定间隔的灌溉调度。
    
    示例:
        >>> scheduler = IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE)
        >>> rec = scheduler.recommend(soil_moisture=0.18, field_capacity=0.32)
        >>> if rec.should_irrigate:
        ...     print(f"建议灌溉 {rec.amount_mm:.1f} mm")
    """
    
    def __init__(
        self,
        method: ScheduleType = ScheduleType.SOIL_MOISTURE,
        trigger_threshold: float = 0.5,
        max_application_mm: float = 50.0,
        irrigation_efficiency: float = 0.85
    ):
        """初始化调度器
        
        参数:
            method: 调度方法
            trigger_threshold: 触发阈值(相对可用水分)
            max_application_mm: 单次最大灌溉量
            irrigation_efficiency: 灌溉效率
        """
        self.method = method
        self.trigger_threshold = trigger_threshold
        self.max_application_mm = max_application_mm
        self.efficiency = irrigation_efficiency
        
        self.schedule: List[IrrigationEvent] = []
    
    def recommend_by_moisture(
        self,
        current_moisture: float,
        field_capacity: float,
        wilting_point: float,
        root_depth_m: float = 0.3
    ) -> IrrigationRecommendation:
        """基于土壤水分的灌溉建议
        
        参数:
            current_moisture: 当前含水量
            field_capacity: 田间持水量
            wilting_point: 凋萎点
            root_depth_m: 根区深度
            
        返回:
            IrrigationRecommendation
        """
        available_range = field_capacity - wilting_point
        current_available = current_moisture - wilting_point
        
        if available_range <= 0:
            return IrrigationRecommendation(False, 0, "参数错误")
        
        depletion_fraction = 1 - (current_available / available_range)
        
        # 判断是否需要灌溉
        if depletion_fraction < self.trigger_threshold:
            return IrrigationRecommendation(
                should_irrigate=False,
                amount_mm=0,
                reason=f"土壤水分充足(亏缺{depletion_fraction*100:.0f}%)",
                urgency="low"
            )
        
        # 计算灌溉量(补充到田间持水量)
        deficit = field_capacity - current_moisture
        deficit_mm = deficit * root_depth_m * 1000
        
        # 考虑效率
        gross_amount = deficit_mm / self.efficiency
        amount = min(gross_amount, self.max_application_mm)
        
        # 紧急程度
        if depletion_fraction > 0.8:
            urgency = "critical"
        elif depletion_fraction > 0.6:
            urgency = "high"
        else:
            urgency = "normal"
        
        return IrrigationRecommendation(
            should_irrigate=True,
            amount_mm=amount,
            reason=f"土壤水分亏缺{depletion_fraction*100:.0f}%",
            urgency=urgency
        )
    
    def recommend_by_et(
        self,
        days_since_irrigation: int,
        cumulative_et_mm: float,
        cumulative_rain_mm: float = 0
    ) -> IrrigationRecommendation:
        """基于ET的灌溉建议
        
        参数:
            days_since_irrigation: 距上次灌溉天数
            cumulative_et_mm: 累计ET
            cumulative_rain_mm: 累计降水
            
        返回:
            IrrigationRecommendation
        """
        net_deficit = cumulative_et_mm - cumulative_rain_mm
        
        if net_deficit <= 0:
            return IrrigationRecommendation(
                should_irrigate=False,
                amount_mm=0,
                reason="降水已补充水分亏缺"
            )
        
        threshold_mm = self.trigger_threshold * self.max_application_mm
        
        if net_deficit < threshold_mm:
            return IrrigationRecommendation(
                should_irrigate=False,
                amount_mm=0,
                reason=f"净亏缺{net_deficit:.1f}mm低于阈值"
            )
        
        amount = min(net_deficit / self.efficiency, self.max_application_mm)
        
        return IrrigationRecommendation(
            should_irrigate=True,
            amount_mm=amount,
            reason=f"补充{days_since_irrigation}天ET亏缺"
        )
    
    def fixed_schedule(
        self,
        interval_days: int,
        amount_mm: float,
        total_days: int,
        start_day: int = 0
    ) -> List[IrrigationEvent]:
        """生成固定间隔灌溉计划
        
        参数:
            interval_days: 灌溉间隔
            amount_mm: 每次灌溉量
            total_days: 总天数
            start_day: 起始日
            
        返回:
            灌溉事件列表
        """
        events = []
        day = start_day
        
        while day < total_days:
            events.append(IrrigationEvent(
                day=day,
                amount_mm=amount_mm
            ))
            day += interval_days
        
        self.schedule = events
        return events
    
    def deficit_irrigation_schedule(
        self,
        full_et_mm: np.ndarray,
        deficit_fraction: float = 0.7,
        min_interval: int = 3
    ) -> List[IrrigationEvent]:
        """亏缺灌溉计划
        
        参数:
            full_et_mm: 逐日完全ET需求
            deficit_fraction: 亏缺系数(0.7=70%ET)
            min_interval: 最小灌溉间隔
            
        返回:
            灌溉事件列表
        """
        target_et = full_et_mm * deficit_fraction
        events = []
        accumulated = 0.0
        last_irrigation_day = -min_interval
        
        for day, et in enumerate(target_et):
            accumulated += et
            
            if (accumulated >= self.max_application_mm * 0.8 and 
                day - last_irrigation_day >= min_interval):
                
                amount = min(accumulated, self.max_application_mm)
                events.append(IrrigationEvent(day=day, amount_mm=amount))
                accumulated = 0.0
                last_irrigation_day = day
        
        self.schedule = events
        return events
    
    def total_irrigation(self) -> float:
        """计算总灌溉量"""
        return sum(e.amount_mm for e in self.schedule)


def calculate_net_irrigation_requirement(
    et_mm: float,
    effective_rain_mm: float,
    soil_contribution_mm: float = 0
) -> float:
    """计算净灌溉需水量
    
    参数:
        et_mm: 蒸散发量
        effective_rain_mm: 有效降水
        soil_contribution_mm: 土壤贡献
        
    返回:
        净灌溉需水量(mm)
    """
    nir = et_mm - effective_rain_mm - soil_contribution_mm
    return max(0, nir)


def gross_irrigation_requirement(nir_mm: float, efficiency: float = 0.85) -> float:
    """计算毛灌溉需水量
    
    参数:
        nir_mm: 净灌溉需水量
        efficiency: 灌溉效率
        
    返回:
        毛灌溉需水量(mm)
    """
    if efficiency <= 0:
        return nir_mm
    return nir_mm / efficiency


if __name__ == "__main__":
    print("=" * 50)
    print("灌溉调度演示")
    print("=" * 50)
    
    scheduler = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5
    )
    
    # 基于土壤水分的建议
    rec = scheduler.recommend_by_moisture(
        current_moisture=0.18,
        field_capacity=0.32,
        wilting_point=0.12,
        root_depth_m=0.3
    )
    
    print(f"\n是否灌溉: {rec.should_irrigate}")
    print(f"建议量: {rec.amount_mm:.1f} mm")
    print(f"原因: {rec.reason}")
    print(f"紧急程度: {rec.urgency}")
    
    # 固定计划
    print("\n固定间隔灌溉计划:")
    events = scheduler.fixed_schedule(interval_days=7, amount_mm=40, total_days=60)
    for e in events[:5]:
        print(f"  第{e.day}天: {e.amount_mm}mm")
    print(f"  总灌溉量: {scheduler.total_irrigation():.0f}mm")
