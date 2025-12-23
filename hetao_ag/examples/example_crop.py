# -*- coding: utf-8 -*-
"""
Crop模块使用示例
================

演示hetao_ag.crop模块的作物生长模拟、胁迫响应和物候期管理功能。

作者: Hetao College
"""

import numpy as np
from hetao_ag.crop import (
    # 胁迫响应
    yield_reduction_salinity, yield_reduction_salinity_crop,
    water_stress_factor, water_stress_from_moisture,
    combined_stress_factor, yield_with_stress,
    classify_salt_tolerance, CropSaltTolerance, CROP_SALT_TOLERANCE,
    # 物候期
    PhenologyTracker, PhenologyConfig, GrowthStage,
    growing_degree_days, CROP_PHENOLOGY,
    # 生长模型
    CropModel, CropConfig, simulate_growing_season, CROP_CONFIGS
)


def example_salt_stress():
    """
    示例1: 盐分胁迫响应
    ===================
    
    基于Maas-Hoffman模型计算盐分对作物产量的影响。
    这对河套灌区等盐渍化地区的作物选择至关重要。
    """
    print("\n" + "=" * 60)
    print("示例1: 盐分胁迫响应 (Maas-Hoffman模型)")
    print("=" * 60)
    
    # 1.1 作物盐分耐受性参数
    print("\n【作物盐分耐受性参数】")
    print(f"{'作物':>10} {'阈值(dS/m)':>12} {'斜率(%/dS/m)':>14} {'耐盐等级':>12}")
    print("-" * 52)
    
    crops = ["barley", "wheat", "cotton", "maize", "rice", "potato"]
    for crop in crops:
        tolerance = CROP_SALT_TOLERANCE.get(crop)
        if tolerance:
            grade = classify_salt_tolerance(crop)
            print(f"{crop:>10} {tolerance.threshold:>12.1f} {tolerance.slope*100:>14.1f} {grade:>12}")
    
    # 1.2 计算不同EC下的产量
    print("\n【不同土壤EC下的相对产量】")
    ec_levels = [2, 4, 6, 8, 10, 12]
    
    print(f"{'EC(dS/m)':>10}", end="")
    for ec in ec_levels:
        print(f"{ec:>8}", end="")
    print()
    print("-" * 60)
    
    for crop in ["wheat", "maize", "cotton", "barley"]:
        print(f"{crop:>10}", end="")
        for ec in ec_levels:
            rel_yield = yield_reduction_salinity_crop(ec, crop)
            print(f"{rel_yield*100:>7.0f}%", end="")
        print()
    
    # 1.3 自定义参数计算
    print("\n【自定义参数计算】")
    # 假设某未知品种: 阈值5 dS/m, 斜率8%/dS/m
    ECe = 8.0
    rel_yield = yield_reduction_salinity(ECe, threshold=5.0, slope=0.08)
    print(f"  阈值=5.0, 斜率=8%/dS/m")
    print(f"  ECe={ECe} dS/m时相对产量: {rel_yield*100:.1f}%")
    
    # 1.4 产量估算
    print("\n【实际产量估算】")
    potential_yield = 6000  # kg/ha
    ECe = 7.0
    
    wheat_yield = yield_reduction_salinity_crop(ECe, "wheat") * potential_yield
    maize_yield = yield_reduction_salinity_crop(ECe, "maize") * potential_yield
    
    print(f"  潜在产量: {potential_yield} kg/ha")
    print(f"  土壤EC: {ECe} dS/m")
    print(f"  小麦实际产量: {wheat_yield:.0f} kg/ha")
    print(f"  玉米实际产量: {maize_yield:.0f} kg/ha")


def example_water_stress():
    """
    示例2: 水分胁迫响应
    ===================
    
    计算土壤水分对作物生长的影响。
    """
    print("\n" + "=" * 60)
    print("示例2: 水分胁迫响应")
    print("=" * 60)
    
    # 2.1 基于土壤水分的胁迫因子
    print("\n【水分胁迫因子Ks】")
    print(f"{'含水量':>8} {'Ks':>8} {'状态':>12}")
    print("-" * 32)
    
    fc = 0.32  # 田间持水量
    wp = 0.12  # 凋萎点
    
    moisture_levels = [0.32, 0.28, 0.24, 0.20, 0.16, 0.12]
    for moisture in moisture_levels:
        ks = water_stress_from_moisture(moisture, fc, wp)
        if ks >= 0.9:
            status = "无胁迫"
        elif ks >= 0.7:
            status = "轻度胁迫"
        elif ks >= 0.4:
            status = "中度胁迫"
        else:
            status = "重度胁迫"
        print(f"{moisture:>8.2f} {ks:>8.2f} {status:>12}")
    
    # 2.2 组合胁迫
    print("\n【组合胁迫因子】")
    ks_water = 0.75
    ks_salt = 0.85
    
    combined_mult = combined_stress_factor(ks_water, ks_salt, method="multiplicative")
    combined_min = combined_stress_factor(ks_water, ks_salt, method="minimum")
    
    print(f"  水分胁迫因子: {ks_water}")
    print(f"  盐分胁迫因子: {ks_salt}")
    print(f"  乘法组合: {combined_mult:.3f}")
    print(f"  取小值法: {combined_min:.3f}")
    
    # 2.3 胁迫条件下的产量
    print("\n【胁迫条件下的产量】")
    potential = 8000  # kg/ha
    
    actual = yield_with_stress(
        potential_yield=potential,
        water_stress=0.80,
        salinity_stress=0.90,
        other_stress=1.0  # 无其他胁迫
    )
    
    print(f"  潜在产量: {potential} kg/ha")
    print(f"  水分胁迫: 0.80")
    print(f"  盐分胁迫: 0.90")
    print(f"  实际产量: {actual:.0f} kg/ha ({actual/potential*100:.1f}%)")


def example_phenology():
    """
    示例3: 物候期管理
    =================
    
    基于积温(GDD)跟踪作物生长阶段。
    """
    print("\n" + "=" * 60)
    print("示例3: 物候期管理 (积温模型)")
    print("=" * 60)
    
    # 3.1 作物物候参数
    print("\n【作物物候参数(需积温GDD)】")
    print(f"{'作物':>8} {'基温':>6} {'出苗':>6} {'营养':>6} {'开花':>6} {'灌浆':>6} {'成熟':>6}")
    print("-" * 50)
    
    for crop in ["wheat", "maize", "rice", "cotton"]:
        config = CROP_PHENOLOGY.get(crop)
        if config:
            gdd = config.stage_gdd
            print(f"{crop:>8} {config.base_temperature:>6.1f} "
                  f"{gdd.get('emergence', 0):>6.0f} {gdd.get('vegetative', 0):>6.0f} "
                  f"{gdd.get('flowering', 0):>6.0f} {gdd.get('grain_fill', 0):>6.0f} "
                  f"{gdd.get('maturity', 0):>6.0f}")
    
    # 3.2 创建物候跟踪器
    tracker = PhenologyTracker("wheat")
    
    print(f"\n【小麦物候跟踪】")
    print(f"  基础温度: {tracker.config.base_temperature}°C")
    
    # 3.3 模拟积温累积
    np.random.seed(42)
    
    print(f"\n{'天数':>4} {'Tmax':>6} {'Tmin':>6} {'日GDD':>8} {'累积GDD':>10} {'阶段':>12}")
    print("-" * 55)
    
    for day in range(1, 121):
        # 模拟温度(春播小麦)
        t_max = 15 + 15 * np.sin((day - 30) / 120 * np.pi) + np.random.randn() * 2
        t_min = t_max - 10 + np.random.randn()
        
        daily_gdd = tracker.accumulate_gdd(t_max, t_min)
        
        if day % 20 == 0 or day == 1:
            print(f"{day:>4} {t_max:>6.1f} {t_min:>6.1f} {daily_gdd:>8.1f} "
                  f"{tracker.accumulated_gdd:>10.0f} {tracker.current_stage.value:>12}")
    
    # 3.4 物候状态
    print(f"\n【最终物候状态】")
    print(f"  累积积温: {tracker.accumulated_gdd:.0f} GDD")
    print(f"  当前阶段: {tracker.current_stage.value}")
    print(f"  成熟进度: {tracker.progress_to_maturity()*100:.1f}%")
    print(f"  当前Kc: {tracker.get_kc_for_stage()}")
    
    # 3.5 预测成熟日期
    avg_daily_gdd = tracker.accumulated_gdd / tracker.days_after_planting
    days_remaining = tracker.days_to_maturity(avg_daily_gdd)
    print(f"  平均日GDD: {avg_daily_gdd:.1f}")
    print(f"  预计剩余天数: {days_remaining}")


def example_crop_growth():
    """
    示例4: 作物生长模型
    ===================
    
    综合模拟作物生长，考虑温度、水分和盐分胁迫。
    """
    print("\n" + "=" * 60)
    print("示例4: 作物生长模型")
    print("=" * 60)
    
    # 4.1 作物配置参数
    print("\n【作物模型参数】")
    for crop_name in ["wheat", "maize", "cotton"]:
        config = CROP_CONFIGS.get(crop_name)
        if config:
            print(f"\n  {crop_name}:")
            print(f"    潜在产量: {config.potential_yield_kg_ha} kg/ha")
            print(f"    收获指数: {config.harvest_index}")
            print(f"    蒸腾效率: {config.transpiration_efficiency} kg/ha/mm")
    
    # 4.2 创建作物模型
    model = CropModel("wheat")
    
    print(f"\n【小麦生长模拟 (120天)】")
    
    # 4.3 逐日模拟
    np.random.seed(42)
    
    print(f"\n{'天':>4} {'阶段':>12} {'LAI':>6} {'生物量':>10} {'胁迫':>6}")
    print("-" * 45)
    
    for day in range(1, 121):
        # 模拟环境条件
        t_max = 15 + 15 * np.sin((day - 30) / 120 * np.pi) + np.random.randn() * 2
        t_min = t_max - 10
        et = 3 + 3 * np.sin(day / 120 * np.pi)
        
        # 土壤条件
        soil_moisture = 0.25 + 0.05 * np.sin(day / 30)
        ECe = 3.5  # 轻度盐分
        
        result = model.update_daily(
            t_max=t_max, t_min=t_min, et=et,
            soil_moisture=soil_moisture, ECe=ECe
        )
        
        if day % 20 == 0 or day == 1:
            print(f"{day:>4} {result['stage']:>12} {result['lai']:>6.2f} "
                  f"{result['biomass_kg_ha']:>10.0f} {result['stress_factor']:>6.2f}")
    
    # 4.4 最终产量
    final_yield = model.estimate_yield()
    
    print(f"\n【生长季结束】")
    print(f"  总生物量: {model.accumulated_biomass:.0f} kg/ha")
    print(f"  预估产量: {final_yield:.0f} kg/ha")
    print(f"  平均胁迫因子: {np.mean(model.stress_history):.3f}")
    
    # 4.5 不同条件对比
    print(f"\n【不同ECe条件下的产量对比】")
    
    for ECe in [1, 4, 7, 10]:
        result = simulate_growing_season(
            crop="wheat",
            weather=[{"t_max": 25, "t_min": 15, "et": 5} for _ in range(100)],
            soil_moisture=0.26,
            ECe=ECe
        )
        print(f"  ECe={ECe:>2} dS/m: 产量={result['yield_kg_ha']:>5.0f} kg/ha, "
              f"胁迫因子={result['avg_stress']:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.crop 模块使用示例")
    print("=" * 60)
    
    example_salt_stress()
    example_water_stress()
    example_phenology()
    example_crop_growth()
    
    print("\n" + "=" * 60)
    print("Crop模块示例完成")
    print("=" * 60)
