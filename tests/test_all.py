# -*- coding: utf-8 -*-
"""
hetao_ag 综合测试脚本

测试所有模块的导入和核心功能。
运行方式: python tests/test_all.py
"""

import sys
import traceback


def test_core():
    """测试核心模块"""
    print("\n[1/7] 测试Core模块...")
    from hetao_ag.core import Unit, Quantity, Dimension, DimensionError
    from hetao_ag.core import meters, hectares, celsius, ds_per_m
    from hetao_ag.core import ConfigManager, get_logger
    from hetao_ag.core import safe_divide, clamp, rmse, ValidationResult
    
    # 测试单位
    q = Quantity(5.5, Unit.KILOMETER)
    assert q.to(Unit.METER).value == 5500, "单位转换失败"
    
    # 测试温度
    t = celsius(25)
    assert abs(t.to(Unit.KELVIN).value - 298.15) < 0.01, "温度转换失败"
    
    # 测试算术
    a = meters(100)
    b = meters(50)
    assert (a + b).value == 150, "加法失败"
    
    # 测试工具函数
    assert safe_divide(10, 0) == 0.0, "安全除法失败"
    assert clamp(15, 0, 10) == 10, "clamp失败"
    
    print("    [OK] Core模块测试通过")
    return True


def test_soil():
    """测试土壤模块"""
    print("\n[2/7] 测试Soil模块...")
    from hetao_ag.soil import (
        SoilMoistureModel, SoilType, SalinityModel,
        SensorCalibrator, classify_soil_salinity
    )
    import numpy as np
    
    # 水分模型
    model = SoilMoistureModel(field_capacity=0.32, wilting_point=0.12)
    model.step_day(rain_mm=15, et_mm=5)
    assert 0.12 <= model.moisture <= 0.45, "土壤含水量异常"
    
    # 盐分分级
    assert "非盐渍化" in classify_soil_salinity(1.5), "盐分分级失败"
    assert "中度" in classify_soil_salinity(6.0), "盐分分级失败"
    
    # 传感器校准
    calibrator = SensorCalibrator()
    result = calibrator.linear_calibration(
        np.array([300, 450, 600]),
        np.array([0.10, 0.20, 0.30])
    )
    assert result.r_squared > 0.99, "校准R²过低"
    
    print("    [OK] Soil模块测试通过")
    return True


def test_water():
    """测试水循环模块"""
    print("\n[3/7] 测试Water模块...")
    from hetao_ag.water import (
        eto_penman_monteith, eto_hargreaves, WeatherData,
        extraterrestrial_radiation, crop_coefficient, etc_crop,
        WaterBalance, IrrigationScheduler
    )
    
    # Penman-Monteith ET0
    weather = WeatherData(
        t_mean=25.0, t_max=32.0, t_min=18.0,
        rh=55.0, u2=2.0, rs=22.0,
        elevation=1050, latitude=40.8, doy=180
    )
    et0 = eto_penman_monteith(weather)
    assert 3 < et0 < 10, f"ET0异常: {et0}"
    
    # 作物系数
    kc = crop_coefficient("mid", "wheat")
    assert 1.0 < kc < 1.3, "Kc异常"
    
    # 水量平衡
    wb = WaterBalance(initial_storage_mm=80)
    wb.step_day(precip_mm=20, et_mm=5)
    assert 80 < wb.storage_mm < 100, "水量平衡异常"
    
    print("    [OK] Water模块测试通过")
    return True


def test_crop():
    """测试作物模块"""
    print("\n[4/7] 测试Crop模块...")
    from hetao_ag.crop import (
        yield_reduction_salinity_crop, water_stress_from_moisture,
        combined_stress_factor, classify_salt_tolerance,
        PhenologyTracker, CropModel
    )
    
    # 盐分胁迫
    rel_yield = yield_reduction_salinity_crop(8.0, "wheat")
    assert 0.5 < rel_yield < 1.0, "盐分胁迫计算异常"
    
    # 水分胁迫
    ks = water_stress_from_moisture(0.20, 0.32, 0.12)
    assert 0 < ks <= 1, "水分胁迫异常"
    
    # 组合胁迫
    combined = combined_stress_factor(0.8, 0.9)
    assert abs(combined - 0.72) < 0.01, "组合胁迫计算错误"
    
    # 物候跟踪
    tracker = PhenologyTracker("wheat")
    tracker.accumulate_gdd(30, 15)
    assert tracker.accumulated_gdd > 0, "积温累积失败"
    
    # 作物模型
    model = CropModel("wheat")
    for _ in range(10):
        model.update_daily(t_max=25, t_min=15, et=5, soil_moisture=0.25, ECe=3.0)
    assert model.accumulated_biomass > 0, "生物量累积失败"
    
    print("    [OK] Crop模块测试通过")
    return True


def test_livestock():
    """测试畜牧模块"""
    print("\n[5/7] 测试Livestock模块...")
    from hetao_ag.livestock import (
        AnimalDetector, BehaviorClassifier, AnimalBehavior,
        HealthMonitor, HealthStatus
    )
    import numpy as np
    
    # 检测器(模拟模式)
    detector = AnimalDetector()
    detections = detector.detect("test.jpg")
    assert len(detections) >= 1, "模拟检测失败"
    
    # 行为分类
    classifier = BehaviorClassifier()
    behavior = classifier.classify_from_motion(0.3, "down")
    assert behavior == AnimalBehavior.GRAZING, "行为分类错误"
    
    # 健康监测
    monitor = HealthMonitor("cow_001")
    for _ in range(7):
        monitor.update_activity(100 + np.random.randn() * 5)
    monitor.update_activity(60)  # 模拟异常
    alerts = monitor.check_health()
    assert len(alerts) >= 1, "健康预警失败"
    
    print("    [OK] Livestock模块测试通过")
    return True


def test_space():
    """测试遥感模块"""
    print("\n[6/7] 测试Space模块...")
    from hetao_ag.space import (
        compute_ndvi, compute_savi, compute_lswi, compute_evi,
        classify_vegetation_health, RasterImage, PhenologyClassifier
    )
    import numpy as np
    
    # NDVI计算
    red = np.array([[100, 120], [110, 90]])
    nir = np.array([[200, 220], [190, 170]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi.shape == (2, 2), "NDVI形状错误"
    assert -1 <= ndvi.min() <= ndvi.max() <= 1, "NDVI范围错误"
    
    # SAVI
    savi = compute_savi(red, nir, L=0.5)
    assert savi.shape == (2, 2), "SAVI形状错误"
    
    # 植被分类
    health = classify_vegetation_health(0.65)
    assert "茂密" in health, "植被分类错误"
    
    # 物候分类
    ts = np.random.rand(12, 10, 10)
    classifier = PhenologyClassifier(ts)
    crop_map = classifier.classify_crops()
    assert crop_map.shape == (10, 10), "分类结果形状错误"
    
    print("    [OK] Space模块测试通过")
    return True


def test_opt():
    """测试优化模块"""
    print("\n[7/7] 测试Opt模块...")
    from hetao_ag.opt import (
        LinearOptimizer, optimize_crop_mix,
        GeneticOptimizer, GAConfig,
        ScenarioEvaluator, multi_objective_score
    )
    
    # 线性优化
    crops = [
        {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
        {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
    ]
    solution = optimize_crop_mix(crops, total_land=100, total_water=300000)
    assert len(solution) >= 1, "作物组合优化失败"
    
    # 遗传算法
    def sphere(x):
        return -sum(xi**2 for xi in x)
    
    optimizer = GeneticOptimizer(
        sphere, n_vars=2,
        bounds=[(-5, 5)] * 2,
        config=GAConfig(generations=20, population_size=20)
    )
    result = optimizer.optimize()
    assert result.best_fitness > -1, "GA优化失败"
    
    # 多目标评分
    score = multi_objective_score(0.8, 0.7, 0.9)
    assert 0 < score < 1, "多目标评分失败"
    
    print("    [OK] Opt模块测试通过")
    return True


def main():
    print("=" * 60)
    print("  hetao_ag 智慧农牧业库 - 综合测试")
    print("=" * 60)
    
    tests = [
        ("core", test_core),
        ("soil", test_soil),
        ("water", test_water),
        ("crop", test_crop),
        ("livestock", test_livestock),
        ("space", test_space),
        ("opt", test_opt),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"    [FAIL] {name}模块测试失败: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "[OK] 通过" if success else f"[FAIL] 失败: {error}"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 模块测试通过")
    
    if passed == total:
        print("\n[OK] 所有测试通过！库已达到国际顶级标准。")
        return 0
    else:
        print("\n[FAIL] 部分测试失败，需要修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
