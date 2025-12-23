# -*- coding: utf-8 -*-
"""
hetao_ag.water.balance - 水量平衡模型

农田水量平衡核算和跟踪。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date


@dataclass
class WaterBalanceRecord:
    """水量平衡日记录"""
    date: Optional[date] = None
    precipitation_mm: float = 0.0
    irrigation_mm: float = 0.0
    et_mm: float = 0.0
    runoff_mm: float = 0.0
    drainage_mm: float = 0.0
    soil_moisture: float = 0.0
    
    @property
    def net_change(self) -> float:
        """净水量变化"""
        return (self.precipitation_mm + self.irrigation_mm - 
                self.et_mm - self.runoff_mm - self.drainage_mm)


class WaterBalance:
    """农田水量平衡模型
    
    跟踪农田水分收支,包括降水、灌溉、蒸散发、径流和渗透。
    
    示例:
        >>> wb = WaterBalance(initial_storage_mm=100)
        >>> wb.add_precipitation(25)
        >>> wb.remove_et(5)
        >>> print(wb.storage_mm)
    """
    
    def __init__(
        self,
        initial_storage_mm: float = 100.0,
        max_storage_mm: float = 150.0,
        min_storage_mm: float = 40.0
    ):
        """初始化水量平衡模型
        
        参数:
            initial_storage_mm: 初始储水量(mm)
            max_storage_mm: 最大储水量(田间持水量)
            min_storage_mm: 最小储水量(凋萎点)
        """
        self.storage_mm = initial_storage_mm
        self.max_storage_mm = max_storage_mm
        self.min_storage_mm = min_storage_mm
        
        # 累计量
        self.total_precipitation = 0.0
        self.total_irrigation = 0.0
        self.total_et = 0.0
        self.total_runoff = 0.0
        self.total_drainage = 0.0
        
        self.history: List[WaterBalanceRecord] = []
    
    def add_precipitation(self, amount_mm: float) -> tuple:
        """添加降水
        
        返回:
            (入渗量, 径流量)
        """
        new_storage = self.storage_mm + amount_mm
        
        if new_storage > self.max_storage_mm:
            runoff = new_storage - self.max_storage_mm
            self.storage_mm = self.max_storage_mm
        else:
            runoff = 0.0
            self.storage_mm = new_storage
        
        infiltration = amount_mm - runoff
        self.total_precipitation += infiltration
        self.total_runoff += runoff
        
        return infiltration, runoff
    
    def add_irrigation(self, amount_mm: float) -> float:
        """添加灌溉
        
        返回:
            实际入渗量
        """
        new_storage = self.storage_mm + amount_mm
        
        if new_storage > self.max_storage_mm:
            overflow = new_storage - self.max_storage_mm
            self.storage_mm = self.max_storage_mm
            actual = amount_mm - overflow
        else:
            self.storage_mm = new_storage
            actual = amount_mm
        
        self.total_irrigation += actual
        return actual
    
    def remove_et(self, amount_mm: float) -> float:
        """移除蒸散发
        
        返回:
            实际移除量
        """
        new_storage = self.storage_mm - amount_mm
        
        if new_storage < self.min_storage_mm:
            actual = self.storage_mm - self.min_storage_mm
            self.storage_mm = self.min_storage_mm
        else:
            actual = amount_mm
            self.storage_mm = new_storage
        
        self.total_et += actual
        return actual
    
    def deep_drainage(self) -> float:
        """计算深层渗漏"""
        if self.storage_mm > self.max_storage_mm:
            drainage = self.storage_mm - self.max_storage_mm
            self.storage_mm = self.max_storage_mm
            self.total_drainage += drainage
            return drainage
        return 0.0
    
    def step_day(self, precip_mm: float = 0, irrig_mm: float = 0, 
                 et_mm: float = 0, record_date: Optional[date] = None) -> WaterBalanceRecord:
        """模拟一天
        
        参数:
            precip_mm: 降水量
            irrig_mm: 灌溉量
            et_mm: 蒸散发量
            record_date: 日期
            
        返回:
            当日记录
        """
        # 输入
        _, runoff = self.add_precipitation(precip_mm)
        self.add_irrigation(irrig_mm)
        
        # 输出
        actual_et = self.remove_et(et_mm)
        drainage = self.deep_drainage()
        
        record = WaterBalanceRecord(
            date=record_date,
            precipitation_mm=precip_mm,
            irrigation_mm=irrig_mm,
            et_mm=actual_et,
            runoff_mm=runoff,
            drainage_mm=drainage,
            soil_moisture=self.storage_mm
        )
        
        self.history.append(record)
        return record
    
    @property
    def available_water(self) -> float:
        """可用水量(mm)"""
        return max(0, self.storage_mm - self.min_storage_mm)
    
    @property
    def deficit_mm(self) -> float:
        """水分亏缺(mm)"""
        return max(0, self.max_storage_mm - self.storage_mm)
    
    @property
    def relative_storage(self) -> float:
        """相对储水量(0-1)"""
        total_range = self.max_storage_mm - self.min_storage_mm
        if total_range <= 0:
            return 1.0
        return (self.storage_mm - self.min_storage_mm) / total_range
    
    def water_use_efficiency(self, yield_kg_ha: float) -> float:
        """水分利用效率(kg/m³)
        
        参数:
            yield_kg_ha: 产量(kg/ha)
            
        返回:
            WUE (kg/m³)
        """
        total_water_mm = self.total_precipitation + self.total_irrigation
        if total_water_mm <= 0:
            return 0.0
        # 1mm = 10 m³/ha
        total_water_m3_ha = total_water_mm * 10
        return yield_kg_ha / total_water_m3_ha
    
    def get_summary(self) -> dict:
        """获取水量平衡汇总"""
        return {
            "current_storage_mm": self.storage_mm,
            "total_precipitation_mm": self.total_precipitation,
            "total_irrigation_mm": self.total_irrigation,
            "total_et_mm": self.total_et,
            "total_runoff_mm": self.total_runoff,
            "total_drainage_mm": self.total_drainage,
            "available_water_mm": self.available_water,
            "deficit_mm": self.deficit_mm
        }


if __name__ == "__main__":
    print("=" * 50)
    print("水量平衡模型演示")
    print("=" * 50)
    
    wb = WaterBalance(initial_storage_mm=80, max_storage_mm=120, min_storage_mm=40)
    
    # 模拟10天
    weather = [
        {"precip": 0, "et": 5},
        {"precip": 15, "et": 4},
        {"precip": 0, "et": 6},
        {"precip": 0, "et": 5},
        {"precip": 0, "et": 5},
        {"precip": 30, "et": 3},
        {"precip": 0, "et": 4},
        {"precip": 0, "et": 6},
        {"precip": 0, "et": 5},
        {"precip": 0, "et": 5},
    ]
    
    print("\n逐日模拟:")
    for i, w in enumerate(weather, 1):
        record = wb.step_day(precip_mm=w["precip"], et_mm=w["et"])
        print(f"第{i}天: 储水={record.soil_moisture:.1f}mm, ET={record.et_mm:.1f}mm")
    
    print("\n水量平衡汇总:")
    summary = wb.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.1f}")
