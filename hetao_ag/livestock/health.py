# -*- coding: utf-8 -*-
"""
hetao_ag.livestock.health - 健康监测

动物健康和福利监测预警。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime
import numpy as np


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    ATTENTION = "attention"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """预警类型"""
    REDUCED_ACTIVITY = "reduced_activity"
    REDUCED_FEEDING = "reduced_feeding"
    ABNORMAL_BEHAVIOR = "abnormal_behavior"
    TEMPERATURE = "temperature"
    LAMENESS = "lameness"
    HEAT_DETECTION = "heat_detection"


@dataclass
class HealthAlert:
    """健康预警"""
    animal_id: str
    alert_type: AlertType
    severity: HealthStatus
    message: str
    timestamp: datetime
    value: Optional[float] = None


class HealthMonitor:
    """健康监测器
    
    综合多源数据监测动物健康状态。
    
    示例:
        >>> monitor = HealthMonitor("cow_001")
        >>> monitor.update_activity(85)
        >>> monitor.update_feeding_time(180)
        >>> alerts = monitor.check_health()
    """
    
    def __init__(self, animal_id: str):
        self.animal_id = animal_id
        
        # 历史数据
        self.activity_history: List[float] = []
        self.feeding_history: List[float] = []
        self.temperature_history: List[float] = []
        
        # 基线值
        self.activity_baseline: Optional[float] = None
        self.feeding_baseline: Optional[float] = None
        
        # 预警阈值
        self.thresholds = {
            "activity_drop": 0.30,  # 活动量下降30%
            "feeding_drop": 0.25,   # 采食时间下降25%
            "temp_high": 39.5,      # 高温阈值(°C)
            "temp_low": 37.5,       # 低温阈值(°C)
        }
        
        self.alerts: List[HealthAlert] = []
    
    def update_activity(self, value: float):
        """更新活动量"""
        self.activity_history.append(value)
        if len(self.activity_history) >= 7:
            self.activity_baseline = np.mean(self.activity_history[-7:-1])
    
    def update_feeding_time(self, minutes: float):
        """更新采食时间(分钟)"""
        self.feeding_history.append(minutes)
        if len(self.feeding_history) >= 7:
            self.feeding_baseline = np.mean(self.feeding_history[-7:-1])
    
    def update_temperature(self, temp_celsius: float):
        """更新体温"""
        self.temperature_history.append(temp_celsius)
    
    def check_health(self) -> List[HealthAlert]:
        """检查健康状态并生成预警"""
        alerts = []
        now = datetime.now()
        
        # 检查活动量
        if self.activity_baseline and len(self.activity_history) > 0:
            current = self.activity_history[-1]
            if current < self.activity_baseline * (1 - self.thresholds["activity_drop"]):
                drop_percent = (1 - current / self.activity_baseline) * 100
                alerts.append(HealthAlert(
                    animal_id=self.animal_id,
                    alert_type=AlertType.REDUCED_ACTIVITY,
                    severity=HealthStatus.WARNING if drop_percent > 40 else HealthStatus.ATTENTION,
                    message=f"活动量下降{drop_percent:.0f}%",
                    timestamp=now,
                    value=current
                ))
        
        # 检查采食
        if self.feeding_baseline and len(self.feeding_history) > 0:
            current = self.feeding_history[-1]
            if current < self.feeding_baseline * (1 - self.thresholds["feeding_drop"]):
                drop_percent = (1 - current / self.feeding_baseline) * 100
                alerts.append(HealthAlert(
                    animal_id=self.animal_id,
                    alert_type=AlertType.REDUCED_FEEDING,
                    severity=HealthStatus.WARNING,
                    message=f"采食时间下降{drop_percent:.0f}%",
                    timestamp=now,
                    value=current
                ))
        
        # 检查体温
        if len(self.temperature_history) > 0:
            temp = self.temperature_history[-1]
            if temp > self.thresholds["temp_high"]:
                alerts.append(HealthAlert(
                    animal_id=self.animal_id,
                    alert_type=AlertType.TEMPERATURE,
                    severity=HealthStatus.CRITICAL if temp > 40.5 else HealthStatus.WARNING,
                    message=f"体温异常: {temp:.1f}°C",
                    timestamp=now,
                    value=temp
                ))
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_status(self) -> HealthStatus:
        """获取当前健康状态"""
        if not self.alerts:
            return HealthStatus.HEALTHY
        
        recent_alerts = [a for a in self.alerts[-10:]]
        severities = [a.severity for a in recent_alerts]
        
        if HealthStatus.CRITICAL in severities:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in severities:
            return HealthStatus.WARNING
        elif HealthStatus.ATTENTION in severities:
            return HealthStatus.ATTENTION
        
        return HealthStatus.HEALTHY


class HerdHealthMonitor:
    """群体健康监测"""
    
    def __init__(self):
        self.animals: Dict[str, HealthMonitor] = {}
    
    def add_animal(self, animal_id: str):
        """添加监测动物"""
        self.animals[animal_id] = HealthMonitor(animal_id)
    
    def get_monitor(self, animal_id: str) -> Optional[HealthMonitor]:
        """获取个体监测器"""
        return self.animals.get(animal_id)
    
    def check_all(self) -> List[HealthAlert]:
        """检查所有动物"""
        all_alerts = []
        for monitor in self.animals.values():
            all_alerts.extend(monitor.check_health())
        return all_alerts
    
    def get_summary(self) -> Dict[str, int]:
        """获取健康状态汇总"""
        summary = {status.value: 0 for status in HealthStatus}
        for monitor in self.animals.values():
            status = monitor.get_status()
            summary[status.value] += 1
        return summary


if __name__ == "__main__":
    print("=" * 50)
    print("健康监测演示")
    print("=" * 50)
    
    monitor = HealthMonitor("cow_001")
    
    # 模拟正常数据
    for _ in range(7):
        monitor.update_activity(100 + np.random.randn() * 5)
        monitor.update_feeding_time(240 + np.random.randn() * 10)
    
    # 添加异常数据
    monitor.update_activity(60)  # 活动量下降
    monitor.update_temperature(40.2)  # 发烧
    
    alerts = monitor.check_health()
    
    print(f"\n当前状态: {monitor.get_status().value}")
    print(f"\n预警数量: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
