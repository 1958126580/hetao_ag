# -*- coding: utf-8 -*-
"""
Soil模块使用示例
================

演示hetao_ag.soil模块的土壤水分、盐分建模和传感器校准功能。

作者: Hetao College
"""

import numpy as np
from hetao_ag.soil import (
    # 水分模型
    SoilMoistureModel, SoilLayer, SoilType, SOIL_PARAMETERS,
    van_genuchten_theta,
    # 盐分模型
    SalinityModel, classify_soil_salinity, classify_water_salinity,
    # 传感器校准
    SensorCalibrator, CalibrationResult, MoistureSensor,
    capacitive_sensor_formula
)


def example_soil_moisture():
    """
    示例1: 土壤水分模型
    ===================
    
    模拟土壤水分动态变化，包括降水入渗、蒸散发消耗和深层渗透。
    这是灌溉决策和作物水分管理的基础。
    """
    print("\n" + "=" * 60)
    print("示例1: 土壤水分模型")
    print("=" * 60)
    
    # 1.1 查看土壤类型参数
    print("\n【土壤水力参数(van Genuchten)】")
    for soil_type in [SoilType.SAND, SoilType.LOAM, SoilType.CLAY]:
        params = SOIL_PARAMETERS[soil_type]
        print(f"  {soil_type.value}:")
        print(f"    θr={params['theta_r']:.3f}, θs={params['theta_s']:.3f}")
        print(f"    Ks={params['Ks']:.1f} mm/day")
    
    # 1.2 创建土壤水分模型
    model = SoilMoistureModel(
        field_capacity=0.32,      # 田间持水量
        wilting_point=0.12,        # 凋萎点
        initial_moisture=0.25,     # 初始含水量
        root_depth_m=0.30,         # 根区深度30cm
        soil_type=SoilType.LOAM    # 壤土
    )
    
    print(f"\n【模型初始状态】")
    print(f"  含水量: {model.moisture:.3f} m³/m³")
    print(f"  饱和含水量: {model.saturation:.3f}")
    print(f"  水分胁迫因子: {model.stress_factor:.3f}")
    print(f"  需灌溉量: {model.irrigation_need_mm:.1f} mm")
    
    # 1.3 模拟降水入渗
    infiltration, runoff = model.add_water(25.0)  # 25mm降水
    print(f"\n【添加25mm降水】")
    print(f"  入渗量: {infiltration:.1f} mm")
    print(f"  地表径流: {runoff:.1f} mm")
    print(f"  当前含水量: {model.moisture:.3f}")
    
    # 1.4 模拟蒸散发消耗
    actual_et = model.remove_water(8.0)  # 8mm蒸散发
    print(f"\n【移除8mm蒸散发】")
    print(f"  实际移除: {actual_et:.1f} mm")
    print(f"  当前含水量: {model.moisture:.3f}")
    
    # 1.5 深层渗透
    drainage = model.deep_percolation()
    print(f"\n【深层渗透】")
    print(f"  渗透量: {drainage:.1f} mm")
    
    # 1.6 逐日模拟
    model2 = SoilMoistureModel(
        field_capacity=0.32, wilting_point=0.12,
        initial_moisture=0.28, soil_type=SoilType.LOAM
    )
    
    # 10天气象数据
    weather_data = [
        {"rain": 0, "et": 5},
        {"rain": 0, "et": 6},
        {"rain": 15, "et": 4},
        {"rain": 0, "et": 5},
        {"rain": 0, "et": 6},
        {"rain": 0, "et": 5},
        {"rain": 25, "et": 3},
        {"rain": 5, "et": 4},
        {"rain": 0, "et": 5},
        {"rain": 0, "et": 6},
    ]
    
    print(f"\n【10天逐日模拟】")
    print(f"{'日期':>4} {'降水':>6} {'ET':>6} {'含水量':>8} {'胁迫':>6}")
    print("-" * 36)
    
    for day, w in enumerate(weather_data, 1):
        result = model2.step_day(
            rain_mm=w["rain"],
            et_mm=w["et"]
        )
        print(f"{day:>4} {w['rain']:>6.0f} {result['et_mm']:>6.1f} "
              f"{result['moisture']:>8.3f} {model2.stress_factor:>6.2f}")
    
    print(f"\n最终状态: 含水量={model2.moisture:.3f}, 胁迫因子={model2.stress_factor:.2f}")


def example_soil_salinity():
    """
    示例2: 土壤盐分模型
    ===================
    
    模拟土壤盐分累积和淋洗过程，用于盐碱地管理。
    这对河套灌区等盐渍化地区尤为重要。
    """
    print("\n" + "=" * 60)
    print("示例2: 土壤盐分模型")
    print("=" * 60)
    
    # 2.1 盐分分级标准
    print("\n【土壤盐分分级标准】")
    ec_levels = [1.0, 3.0, 5.0, 10.0, 20.0]
    for ec in ec_levels:
        grade = classify_soil_salinity(ec)
        print(f"  ECe={ec:>5.1f} dS/m: {grade}")
    
    # 2.2 灌溉水质分级
    print("\n【灌溉水质分级】")
    water_ec = [0.5, 1.5, 4.0]
    for ec in water_ec:
        grade = classify_water_salinity(ec)
        print(f"  EC={ec:.1f} dS/m: {grade}")
    
    # 2.3 创建盐分模型
    model = SalinityModel(
        initial_ECe=4.0,           # 初始EC 4 dS/m (轻度盐渍化)
        root_depth_m=0.30,         # 根区30cm
        soil_water_content=0.28    # 土壤含水量
    )
    
    print(f"\n【盐分模型初始状态】")
    print(f"  土壤EC: {model.ECe:.2f} dS/m")
    print(f"  等级: {classify_soil_salinity(model.ECe)}")
    
    # 2.4 灌溉带入盐分
    result = model.irrigate(amount_mm=60, ec_water=1.5)
    print(f"\n【灌溉60mm (EC=1.5 dS/m)】")
    print(f"  盐分输入: {result['salt_input_kg_ha']:.0f} kg/ha")
    print(f"  灌后EC: {model.ECe:.2f} dS/m")
    
    # 2.5 淋洗降低盐分
    result = model.leach(drainage_mm=40)
    print(f"\n【淋洗40mm】")
    print(f"  盐分移除: {result['salt_removed_kg_ha']:.0f} kg/ha")
    print(f"  淋洗后EC: {model.ECe:.2f} dS/m")
    
    # 2.6 计算淋洗需求
    lr = model.leaching_requirement(
        ec_irrigation=1.5,   # 灌溉水EC
        ec_threshold=4.0     # 目标土壤EC
    )
    print(f"\n【淋洗需求计算】")
    print(f"  目标EC ≤ 4.0 dS/m")
    print(f"  淋洗系数: {lr:.3f} ({lr*100:.1f}%)")
    
    # 2.7 长期模拟
    print(f"\n【30天盐分动态模拟】")
    model2 = SalinityModel(initial_ECe=5.0)
    
    for day in range(30):
        if day % 7 == 0:  # 每周灌溉
            model2.step_day(irrigation_mm=50, ec_irrigation=1.2, drainage_mm=15)
        else:
            model2.step_day(drainage_mm=2)  # 自然渗漏
    
    print(f"  初始EC: 5.00 dS/m")
    print(f"  30天后EC: {model2.ECe:.2f} dS/m")


def example_sensor_calibration():
    """
    示例3: 传感器校准
    =================
    
    校准低成本IoT土壤传感器，提高测量精度。
    """
    print("\n" + "=" * 60)
    print("示例3: 传感器校准")
    print("=" * 60)
    
    # 3.1 准备校准数据
    # 原始ADC读数
    raw_readings = np.array([280, 350, 420, 500, 580, 650, 720])
    # 烘干法测定的真实体积含水量
    ground_truth = np.array([0.08, 0.14, 0.20, 0.26, 0.32, 0.38, 0.44])
    
    print("\n【校准数据】")
    print(f"  原始读数: {raw_readings}")
    print(f"  真实值:   {ground_truth}")
    
    # 3.2 创建校准器
    calibrator = SensorCalibrator()
    
    # 3.3 线性校准
    linear_result = calibrator.linear_calibration(raw_readings, ground_truth)
    print(f"\n【线性校准结果】")
    print(f"  方程: y = {linear_result.coefficients[0]:.6f}x + {linear_result.coefficients[1]:.4f}")
    print(f"  R² = {linear_result.r_squared:.4f}")
    print(f"  RMSE = {linear_result.rmse:.4f}")
    
    # 3.4 多项式校准
    poly_result = calibrator.polynomial_calibration(raw_readings, ground_truth, degree=2)
    print(f"\n【二次多项式校准】")
    print(f"  R² = {poly_result.r_squared:.4f}")
    print(f"  RMSE = {poly_result.rmse:.4f}")
    
    # 3.5 自动选择最佳方法
    best_result = calibrator.auto_calibrate(raw_readings, ground_truth)
    print(f"\n【自动选择最佳方法】")
    print(f"  方法: {best_result.method.value}")
    print(f"  R² = {best_result.r_squared:.4f}")
    
    # 3.6 应用校准
    test_raw = 450
    calibrated = best_result.apply(test_raw)
    print(f"\n【应用校准】")
    print(f"  原始读数: {test_raw}")
    print(f"  校准后含水量: {calibrated:.3f} m³/m³")
    
    # 3.7 创建传感器实例
    sensor = MoistureSensor(
        sensor_id="SM-001",
        calibration=best_result
    )
    
    # 模拟采集数据
    import time
    print(f"\n【传感器数据采集】")
    for i, raw in enumerate([400, 420, 450, 430, 440]):
        sensor.add_reading(time.time() + i*60, raw)
    
    avg_moisture = sensor.get_average()
    print(f"  最近5次平均含水量: {avg_moisture:.3f}")
    
    # 3.8 电容式传感器通用公式
    print(f"\n【电容式传感器公式】")
    for raw in [400, 350, 300]:
        moisture_pct = capacitive_sensor_formula(raw, dry_value=520, wet_value=260)
        print(f"  ADC={raw} -> 湿度={moisture_pct:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.soil 模块使用示例")
    print("=" * 60)
    
    example_soil_moisture()
    example_soil_salinity()
    example_sensor_calibration()
    
    print("\n" + "=" * 60)
    print("Soil模块示例完成")
    print("=" * 60)
