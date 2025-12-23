# -*- coding: utf-8 -*-
"""
Livestock模块使用示例
=====================

演示hetao_ag.livestock模块的动物检测、行为分类和健康监测功能。

作者: Hetao College
"""

import numpy as np
from datetime import datetime
from hetao_ag.livestock import (
    # 视觉检测
    AnimalDetector, Detection, calculate_iou,
    # 行为分类
    BehaviorClassifier, AnimalBehavior, BehaviorRecord, ActivityMonitor,
    # 健康监测
    HealthMonitor, HerdHealthMonitor, HealthAlert, HealthStatus, AlertType
)


def example_animal_detection():
    """
    示例1: 动物检测
    ===============
    
    基于YOLO的动物检测和计数。
    """
    print("\n" + "=" * 60)
    print("示例1: 动物检测 (YOLO)")
    print("=" * 60)
    
    # 1.1 创建检测器
    detector = AnimalDetector(
        model_name="yolov5s",      # 模型名称
        confidence_threshold=0.5,  # 置信度阈值
        use_gpu=True               # 使用GPU
    )
    
    print(f"\n【检测器配置】")
    print(f"  模型: {detector.model_name}")
    print(f"  置信度阈值: {detector.confidence_threshold}")
    print(f"  支持的动物: {detector.SUPPORTED_ANIMALS}")
    
    # 1.2 检测图像
    # 注意: 这里使用模拟模式，实际使用需要安装PyTorch
    detections = detector.detect("farm_image.jpg")
    
    print(f"\n【检测结果】")
    print(f"  检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections, 1):
        print(f"    {i}. {det.label} (置信度={det.confidence:.2f})")
        print(f"       位置: x1={det.bbox[0]:.0f}, y1={det.bbox[1]:.0f}, "
              f"x2={det.bbox[2]:.0f}, y2={det.bbox[3]:.0f}")
    
    # 1.3 动物计数
    counts = detector.count_animals("farm_image.jpg")
    
    print(f"\n【动物计数】")
    for species, count in counts.items():
        print(f"  {species}: {count}头")
    
    # 1.4 IoU计算
    print(f"\n【IoU计算示例】")
    box1 = (100, 100, 200, 200)
    box2 = (150, 150, 250, 250)
    iou = calculate_iou(box1, box2)
    print(f"  Box1: {box1}")
    print(f"  Box2: {box2}")
    print(f"  IoU: {iou:.3f}")


def example_behavior_classification():
    """
    示例2: 行为分类
    ===============
    
    从运动特征识别动物行为。
    """
    print("\n" + "=" * 60)
    print("示例2: 行为分类")
    print("=" * 60)
    
    # 2.1 创建行为分类器
    classifier = BehaviorClassifier()
    
    print(f"\n【行为类型】")
    for behavior in AnimalBehavior:
        print(f"  - {behavior.value}")
    
    # 2.2 基于运动特征分类
    print(f"\n【运动特征分类】")
    test_cases = [
        (0.05, "down", "静卧反刍"),
        (0.08, "level", "静止站立"),
        (0.25, "down", "采食"),
        (0.30, "level", "站立"),
        (0.60, None, "行走"),
    ]
    
    print(f"{'运动幅度':>10} {'头部位置':>10} {'识别行为':>12} {'说明':>10}")
    print("-" * 50)
    
    for motion, head_pos, desc in test_cases:
        behavior = classifier.classify_from_motion(motion, head_pos)
        print(f"{motion:>10.2f} {str(head_pos):>10} {behavior.value:>12} {desc:>10}")
    
    # 2.3 日行为模式分析
    print(f"\n【日行为模式分析】")
    
    # 模拟24小时行为记录
    records = [
        BehaviorRecord(0, "cow_001", AnimalBehavior.LYING, 0.9, 6*3600),      # 0-6点 躺卧
        BehaviorRecord(6*3600, "cow_001", AnimalBehavior.GRAZING, 0.85, 3*3600),  # 6-9点 采食
        BehaviorRecord(9*3600, "cow_001", AnimalBehavior.RUMINATING, 0.8, 2*3600), # 9-11点 反刍
        BehaviorRecord(11*3600, "cow_001", AnimalBehavior.GRAZING, 0.85, 2*3600),  # 11-13点 采食
        BehaviorRecord(13*3600, "cow_001", AnimalBehavior.LYING, 0.9, 3*3600),     # 13-16点 休息
        BehaviorRecord(16*3600, "cow_001", AnimalBehavior.GRAZING, 0.85, 3*3600),  # 16-19点 采食
        BehaviorRecord(19*3600, "cow_001", AnimalBehavior.RUMINATING, 0.8, 2*3600), # 19-21点 反刍
        BehaviorRecord(21*3600, "cow_001", AnimalBehavior.LYING, 0.9, 3*3600),     # 21-24点 躺卧
    ]
    
    pattern = classifier.analyze_daily_pattern(records)
    
    print(f"  24小时行为时间分配:")
    for behavior, ratio in sorted(pattern.items(), key=lambda x: x[1], reverse=True):
        hours = ratio * 24
        print(f"    {behavior}: {hours:.1f}小时 ({ratio*100:.1f}%)")
    
    # 2.4 活动量监测
    print(f"\n【活动量监测】")
    monitor = ActivityMonitor("cow_001")
    
    # 模拟7天正常活动
    np.random.seed(42)
    normal_activity = 100 + np.random.randn(7) * 5
    for activity in normal_activity:
        monitor.add_reading(activity)
    
    print(f"  个体ID: {monitor.animal_id}")
    print(f"  基线活动量: {monitor.baseline:.1f}")
    print(f"  今日活动量: {monitor.get_daily_activity():.1f}")
    
    # 模拟异常活动
    monitor.add_reading(65)  # 显著下降
    is_anomaly = monitor.detect_anomaly(threshold=0.3)
    print(f"\n  添加异常读数后:")
    print(f"  当前活动量: {monitor.get_daily_activity():.1f}")
    print(f"  检测到异常: {'是' if is_anomaly else '否'}")


def example_health_monitoring():
    """
    示例3: 健康监测
    ===============
    
    综合监测动物健康状态并生成预警。
    """
    print("\n" + "=" * 60)
    print("示例3: 健康监测")
    print("=" * 60)
    
    # 3.1 预警类型说明
    print(f"\n【健康预警类型】")
    for alert_type in AlertType:
        print(f"  - {alert_type.value}")
    
    print(f"\n【健康状态等级】")
    for status in HealthStatus:
        print(f"  - {status.value}")
    
    # 3.2 创建个体监测器
    monitor = HealthMonitor("cow_001")
    
    print(f"\n【配置阈值】")
    for key, value in monitor.thresholds.items():
        print(f"  {key}: {value}")
    
    # 3.3 录入正常数据(建立基线)
    print(f"\n【建立健康基线 (7天正常数据)】")
    np.random.seed(42)
    
    for day in range(7):
        monitor.update_activity(100 + np.random.randn() * 5)
        monitor.update_feeding_time(240 + np.random.randn() * 10)
        monitor.update_temperature(38.5 + np.random.randn() * 0.2)
    
    print(f"  活动量基线: {monitor.activity_baseline:.1f}")
    print(f"  采食时间基线: {monitor.feeding_baseline:.1f} 分钟")
    
    # 3.4 录入异常数据
    print(f"\n【录入异常数据】")
    
    # 场景1: 活动量下降
    monitor.update_activity(60)  # 明显下降
    alerts1 = monitor.check_health()
    
    print(f"  活动量: 60 (下降40%)")
    for alert in alerts1:
        print(f"    [{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
    
    # 场景2: 发烧
    monitor.update_temperature(40.5)
    alerts2 = monitor.check_health()
    
    print(f"\n  体温: 40.5°C (发烧)")
    for alert in alerts2:
        if alert.alert_type == AlertType.TEMPERATURE:
            print(f"    [{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
    
    # 3.5 查看健康状态
    status = monitor.get_status()
    print(f"\n【当前健康状态】")
    print(f"  状态: {status.value}")
    print(f"  累计预警: {len(monitor.alerts)}")
    
    # 3.6 群体健康监测
    print(f"\n【群体健康监测】")
    
    herd = HerdHealthMonitor()
    
    # 添加10头牛
    for i in range(10):
        herd.add_animal(f"cow_{i+1:03d}")
    
    # 模拟数据
    np.random.seed(123)
    for animal_id, monitor in herd.animals.items():
        # 建立基线
        for _ in range(7):
            monitor.update_activity(100 + np.random.randn() * 8)
            monitor.update_feeding_time(240 + np.random.randn() * 15)
        
        # 模拟当日数据(部分个体异常)
        if np.random.random() < 0.2:  # 20%概率异常
            monitor.update_activity(50 + np.random.randn() * 10)
        else:
            monitor.update_activity(100 + np.random.randn() * 8)
    
    # 检查全部
    all_alerts = herd.check_all()
    
    print(f"  群体规模: {len(herd.animals)}头")
    print(f"  检测到预警: {len(all_alerts)}条")
    
    # 健康状态汇总
    summary = herd.get_summary()
    print(f"\n  健康状态汇总:")
    for status, count in summary.items():
        if count > 0:
            print(f"    {status}: {count}头")
    
    # 显示需要关注的个体
    print(f"\n  需要关注的个体:")
    for animal_id, monitor in herd.animals.items():
        status = monitor.get_status()
        if status != HealthStatus.HEALTHY:
            print(f"    {animal_id}: {status.value}")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.livestock 模块使用示例")
    print("=" * 60)
    
    example_animal_detection()
    example_behavior_classification()
    example_health_monitoring()
    
    print("\n" + "=" * 60)
    print("Livestock模块示例完成")
    print("=" * 60)
