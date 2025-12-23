# -*- coding: utf-8 -*-
"""
hetao_ag 综合使用示例

演示智慧农牧业库的主要功能。

作者: Hetao College
"""

import numpy as np
from datetime import date


def demo_core_module():
    """核心模块演示"""
    print("\n" + "=" * 60)
    print("【1. 核心模块 - 单位系统和配置】")
    print("=" * 60)
    
    from hetao_ag.core import (
        Quantity, Unit, meters, hectares, celsius, ds_per_m,
        get_logger, ConfigManager, create_default_config
    )
    
    # 单位转换
    distance = Quantity(5.5, Unit.KILOMETER)
    print(f"距离: {distance} = {distance.to(Unit.METER)}")
    
    # 面积计算
    farm_area = hectares(150)
    print(f"农场面积: {farm_area}")
    
    # 温度
    temp = celsius(28.5)
    print(f"温度: {temp} = {temp.to(Unit.KELVIN)}")
    
    # 土壤盐分
    soil_ec = ds_per_m(4.5)
    print(f"土壤电导率: {soil_ec}")
    
    # 日志
    logger = get_logger("demo")
    logger.info("系统初始化完成")


def demo_soil_module():
    """土壤模块演示"""
    print("\n" + "=" * 60)
    print("【2. 土壤模块 - 水分和盐分建模】")
    print("=" * 60)
    
    from hetao_ag.soil import (
        SoilMoistureModel, SoilType, SalinityModel,
        SensorCalibrator, classify_soil_salinity
    )
    
    # 土壤水分模型
    moisture_model = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.25,
        soil_type=SoilType.LOAM
    )
    
    print(f"初始含水量: {moisture_model.moisture:.3f}")
    print(f"需灌溉量: {moisture_model.irrigation_need_mm:.1f} mm")
    
    # 模拟降水
    result = moisture_model.step_day(rain_mm=20, et_mm=5)
    print(f"降水后含水量: {result['moisture']:.3f}")
    
    # 盐分模型
    salinity_model = SalinityModel(initial_ECe=4.0)
    print(f"\n土壤EC: {salinity_model.ECe:.1f} dS/m")
    print(f"等级: {classify_soil_salinity(salinity_model.ECe)}")
    
    # 传感器校准
    calibrator = SensorCalibrator()
    raw_vals = np.array([300, 450, 600, 750])
    true_vals = np.array([0.10, 0.20, 0.30, 0.40])
    result = calibrator.linear_calibration(raw_vals, true_vals)
    print(f"\n传感器校准: {result}")


def demo_water_module():
    """水循环模块演示"""
    print("\n" + "=" * 60)
    print("【3. 水循环模块 - 蒸散发和灌溉】")
    print("=" * 60)
    
    from hetao_ag.water import (
        eto_penman_monteith, eto_hargreaves, WeatherData,
        extraterrestrial_radiation, crop_coefficient, etc_crop,
        WaterBalance, IrrigationScheduler, ScheduleType
    )
    
    # FAO-56 Penman-Monteith
    weather = WeatherData(
        t_mean=25.0, t_max=32.0, t_min=18.0,
        rh=55.0, u2=2.0, rs=22.0,
        elevation=1050, latitude=40.8, doy=180
    )
    
    et0 = eto_penman_monteith(weather)
    print(f"Penman-Monteith ET₀: {et0:.2f} mm/day")
    
    # Hargreaves方法
    Ra = extraterrestrial_radiation(40.8, 180)
    et0_hg = eto_hargreaves(25.0, 32.0, 18.0, Ra)
    print(f"Hargreaves ET₀: {et0_hg:.2f} mm/day")
    
    # 作物蒸散发
    kc = crop_coefficient("mid", "wheat")
    etc = etc_crop(et0, kc)
    print(f"小麦ETc (Kc={kc}): {etc:.2f} mm/day")
    
    # 水量平衡
    wb = WaterBalance(initial_storage_mm=80, max_storage_mm=120)
    wb.step_day(precip_mm=15, et_mm=5)
    print(f"\n土壤储水: {wb.storage_mm:.1f} mm")
    
    # 灌溉调度
    scheduler = IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE)
    rec = scheduler.recommend_by_moisture(0.18, 0.32, 0.12)
    print(f"灌溉建议: {'是' if rec.should_irrigate else '否'}, {rec.amount_mm:.1f} mm")


def demo_crop_module():
    """作物模块演示"""
    print("\n" + "=" * 60)
    print("【4. 作物模块 - 生长和胁迫】")
    print("=" * 60)
    
    from hetao_ag.crop import (
        CropModel, PhenologyTracker, GrowthStage,
        yield_reduction_salinity_crop, water_stress_from_moisture,
        classify_salt_tolerance
    )
    
    # 盐分胁迫
    crops = ["wheat", "maize", "cotton"]
    ECe = 6.0
    print(f"土壤EC={ECe} dS/m时的相对产量:")
    for crop in crops:
        rel_yield = yield_reduction_salinity_crop(ECe, crop)
        tolerance = classify_salt_tolerance(crop)
        print(f"  {crop}: {rel_yield*100:.1f}% ({tolerance})")
    
    # 水分胁迫
    ks = water_stress_from_moisture(0.20, 0.32, 0.12)
    print(f"\n水分胁迫因子Ks: {ks:.3f}")
    
    # 物候期跟踪
    tracker = PhenologyTracker("wheat")
    for _ in range(30):
        tracker.accumulate_gdd(25 + np.random.randn()*3, 15 + np.random.randn()*2)
    print(f"\n积温: {tracker.accumulated_gdd:.0f}")
    print(f"生长阶段: {tracker.current_stage.value}")
    
    # 作物生长模型
    model = CropModel("wheat")
    for _ in range(60):
        model.update_daily(t_max=26, t_min=14, et=5, soil_moisture=0.25, ECe=3.0)
    print(f"\n生物量: {model.accumulated_biomass:.0f} kg/ha")
    print(f"预估产量: {model.estimate_yield():.0f} kg/ha")


def demo_livestock_module():
    """畜牧模块演示"""
    print("\n" + "=" * 60)
    print("【5. 畜牧模块 - 检测和健康监测】")
    print("=" * 60)
    
    from hetao_ag.livestock import (
        AnimalDetector, BehaviorClassifier, AnimalBehavior,
        HealthMonitor, HealthStatus
    )
    
    # 动物检测(模拟)
    detector = AnimalDetector(confidence_threshold=0.5)
    detections = detector.detect("farm_image.jpg")
    print(f"检测到 {len(detections)} 个目标")
    for det in detections:
        print(f"  {det.label}: 置信度={det.confidence:.2f}")
    
    # 行为分类
    classifier = BehaviorClassifier()
    behavior = classifier.classify_from_motion(0.3, "down")
    print(f"\n行为分类: {behavior.value}")
    
    # 健康监测
    monitor = HealthMonitor("cow_001")
    for _ in range(7):
        monitor.update_activity(100 + np.random.randn() * 5)
        monitor.update_feeding_time(240 + np.random.randn() * 10)
    
    # 模拟异常
    monitor.update_activity(60)
    alerts = monitor.check_health()
    
    print(f"\n健康状态: {monitor.get_status().value}")
    print(f"预警数量: {len(alerts)}")


def demo_space_module():
    """遥感模块演示"""
    print("\n" + "=" * 60)
    print("【6. 遥感模块 - 光谱指数】")
    print("=" * 60)
    
    from hetao_ag.space import (
        compute_ndvi, compute_savi, compute_lswi,
        classify_vegetation_health, RasterImage,
        PhenologyClassifier
    )
    
    # 计算NDVI
    red = np.array([[120, 130], [110, 90]], dtype=np.uint16)
    nir = np.array([[200, 210], [180, 160]], dtype=np.uint16)
    
    ndvi = compute_ndvi(red, nir)
    print(f"NDVI:\n{ndvi}")
    print(f"平均NDVI: {ndvi.mean():.3f}")
    
    # SAVI(土壤调节)
    savi = compute_savi(red, nir, L=0.5)
    print(f"\nSAVI (L=0.5):\n{savi}")
    
    # 植被分类
    print(f"\nNDVI=0.65: {classify_vegetation_health(0.65)}")
    print(f"NDVI=0.25: {classify_vegetation_health(0.25)}")


def demo_opt_module():
    """优化模块演示"""
    print("\n" + "=" * 60)
    print("【7. 优化模块 - 农场规划】")
    print("=" * 60)
    
    from hetao_ag.opt import (
        LinearOptimizer, optimize_crop_mix,
        GeneticOptimizer, GAConfig,
        ScenarioEvaluator
    )
    
    # 作物组合优化
    crops = [
        {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
        {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
        {"name": "alfalfa", "profit_per_ha": 400, "water_per_ha": 2000},
    ]
    
    solution = optimize_crop_mix(crops, total_land=100, total_water=300000)
    print("优化种植方案:")
    for crop, area in solution.items():
        if area and area > 0:
            print(f"  {crop}: {area:.1f} ha")
    
    # 遗传算法示例
    def sphere(x):
        return -sum(xi**2 for xi in x)
    
    optimizer = GeneticOptimizer(
        sphere, n_vars=3,
        bounds=[(-5, 5)] * 3,
        config=GAConfig(generations=30)
    )
    result = optimizer.optimize()
    print(f"\nGA优化: 最优解={[f'{x:.3f}' for x in result.best_solution]}")


def main():
    """主演示函数"""
    print("=" * 60)
    print("   河套智慧农牧业库 (hetao_ag) - 综合演示")
    print("=" * 60)
    print("\n本演示展示hetao_ag库的7个核心模块功能。")
    
    try:
        demo_core_module()
        demo_soil_module()
        demo_water_module()
        demo_crop_module()
        demo_livestock_module()
        demo_space_module()
        demo_opt_module()
        
        print("\n" + "=" * 60)
        print("演示完成！所有模块运行正常。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
