# -*- coding: utf-8 -*-
"""
hetao_ag.livestock.behavior - 行为分类

动物行为识别和分类。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class AnimalBehavior(Enum):
    """动物行为类型"""
    STANDING = "standing"
    LYING = "lying"
    WALKING = "walking"
    GRAZING = "grazing"
    DRINKING = "drinking"
    RUMINATING = "ruminating"
    RUNNING = "running"
    ABNORMAL = "abnormal"


@dataclass
class BehaviorRecord:
    """行为记录"""
    timestamp: float
    animal_id: Optional[str]
    behavior: AnimalBehavior
    confidence: float
    duration_seconds: float = 0.0


class BehaviorClassifier:
    """行为分类器
    
    从视频或传感器数据分类动物行为。
    
    示例:
        >>> classifier = BehaviorClassifier()
        >>> behavior = classifier.classify_from_motion(motion_data)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """初始化分类器"""
        self.model_path = model_path
        self.model = None
        
        # 行为识别阈值
        self.thresholds = {
            "motion_low": 0.1,
            "motion_high": 0.5,
            "head_down": 0.3,
        }
    
    def classify_from_motion(
        self,
        motion_magnitude: float,
        head_position: Optional[str] = None
    ) -> AnimalBehavior:
        """基于运动特征分类行为
        
        参数:
            motion_magnitude: 运动幅度(0-1)
            head_position: 头部位置(up, down, level)
            
        返回:
            行为类别
        """
        if motion_magnitude < self.thresholds["motion_low"]:
            if head_position == "down":
                return AnimalBehavior.RUMINATING
            return AnimalBehavior.LYING
        
        elif motion_magnitude < self.thresholds["motion_high"]:
            if head_position == "down":
                return AnimalBehavior.GRAZING
            return AnimalBehavior.STANDING
        
        else:
            return AnimalBehavior.WALKING
    
    def classify_sequence(self, frames: List) -> AnimalBehavior:
        """从帧序列分类行为
        
        参数:
            frames: 视频帧列表
            
        返回:
            行为类别
        """
        # 简化实现:分析帧间运动
        if len(frames) < 2:
            return AnimalBehavior.STANDING
        
        # 计算帧间差异(模拟)
        motion = np.random.random()  # 实际应计算光流
        
        return self.classify_from_motion(motion)
    
    def analyze_daily_pattern(
        self,
        records: List[BehaviorRecord]
    ) -> Dict[str, float]:
        """分析日行为模式
        
        参数:
            records: 行为记录列表
            
        返回:
            各行为时间占比
        """
        total_time = sum(r.duration_seconds for r in records)
        if total_time == 0:
            return {}
        
        pattern = {}
        for behavior in AnimalBehavior:
            time = sum(r.duration_seconds for r in records if r.behavior == behavior)
            pattern[behavior.value] = time / total_time
        
        return pattern


class ActivityMonitor:
    """活动量监测器
    
    基于加速度计或计步器数据监测动物活动量。
    """
    
    def __init__(self, animal_id: str):
        self.animal_id = animal_id
        self.activity_history: List[float] = []
        self.baseline: Optional[float] = None
    
    def add_reading(self, activity_level: float):
        """添加活动量读数"""
        self.activity_history.append(activity_level)
        
        # 更新基线(使用滑动平均)
        if len(self.activity_history) >= 7:
            self.baseline = np.mean(self.activity_history[-7:])
    
    def get_daily_activity(self) -> float:
        """获取日活动量"""
        if not self.activity_history:
            return 0.0
        return self.activity_history[-1]
    
    def detect_anomaly(self, threshold: float = 0.3) -> bool:
        """检测活动异常
        
        参数:
            threshold: 异常阈值(相对偏差)
            
        返回:
            是否异常
        """
        if self.baseline is None or len(self.activity_history) < 2:
            return False
        
        current = self.activity_history[-1]
        deviation = abs(current - self.baseline) / self.baseline
        
        return deviation > threshold


if __name__ == "__main__":
    print("=" * 50)
    print("行为分类演示")
    print("=" * 50)
    
    classifier = BehaviorClassifier()
    
    # 测试分类
    test_cases = [
        (0.05, "down"),
        (0.2, "level"),
        (0.3, "down"),
        (0.7, None),
    ]
    
    print("\n行为分类测试:")
    for motion, head in test_cases:
        behavior = classifier.classify_from_motion(motion, head)
        print(f"  运动={motion:.2f}, 头位置={head} -> {behavior.value}")
    
    # 活动监测
    monitor = ActivityMonitor("cow_001")
    for day in range(10):
        activity = 100 + np.random.randn() * 10
        monitor.add_reading(activity)
    
    print(f"\n日活动量: {monitor.get_daily_activity():.1f}")
    print(f"基线: {monitor.baseline:.1f}")
