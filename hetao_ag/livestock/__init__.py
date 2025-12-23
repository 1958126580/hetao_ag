# -*- coding: utf-8 -*-
"""
hetao_ag.livestock - 畜牧模块

动物检测、行为分类和健康监测。

作者: Hetao College
版本: 1.0.0
"""

from .vision import (
    AnimalDetector,
    Detection,
    calculate_iou,
)

from .behavior import (
    BehaviorClassifier,
    AnimalBehavior,
    BehaviorRecord,
    ActivityMonitor,
)

from .health import (
    HealthMonitor,
    HerdHealthMonitor,
    HealthAlert,
    HealthStatus,
    AlertType,
)

__all__ = [
    # vision
    "AnimalDetector",
    "Detection",
    "calculate_iou",
    # behavior
    "BehaviorClassifier",
    "AnimalBehavior",
    "BehaviorRecord",
    "ActivityMonitor",
    # health
    "HealthMonitor",
    "HerdHealthMonitor",
    "HealthAlert",
    "HealthStatus",
    "AlertType",
]
