# -*- coding: utf-8 -*-
"""
Water模块使用示例
=================

演示hetao_ag.water模块的蒸散发计算、水量平衡和灌溉调度功能。

作者: Hetao College
"""

import numpy as np
from datetime import date, timedelta
from hetao_ag.water import (
    # 蒸散发
    eto_penman_monteith, eto_hargreaves, WeatherData,
    extraterrestrial_radiation, crop_coefficient, etc_crop,
    # 水量平衡
    WaterBalance, WaterBalanceRecord,
    # 灌溉
    IrrigationScheduler, IrrigationEvent, IrrigationRecommendation,
    IrrigationMethod, ScheduleType,
    calculate_net_irrigation_requirement, gross_irrigation_requirement
)


def example_evapotranspiration():
    """
    示例1: 蒸散发计算
    =================
    
    使用FAO-56 Penman-Monteith方法计算参考作物蒸散发(ET0)。
    这是灌溉需水量计算的国际标准方法。
    """
    print("\n" + "=" * 60)
    print("示例1: 蒸散发计算 (FAO-56)")
    print("=" * 60)
    
    # 1.1 准备气象数据
    # 河套灌区夏季典型气象条件
    weather = WeatherData(
        t_mean=26.0,      # 日平均气温 (°C)
        t_max=33.0,       # 日最高气温 (°C)
        t_min=19.0,       # 日最低气温 (°C)
        rh=50.0,          # 相对湿度 (%)
        u2=2.5,           # 2m高度风速 (m/s)
        rs=24.0,          # 太阳辐射 (MJ/m²/day)
        elevation=1050,   # 海拔 (m)
        latitude=40.8,    # 纬度 (度)
        doy=195           # 年积日 (7月14日)
    )
    
    print(f"\n【气象数据】")
    print(f"  位置: 北纬{weather.latitude}°, 海拔{weather.elevation}m")
    print(f"  日期: 第{weather.doy}天 (约7月中旬)")
    print(f"  温度: {weather.t_min}°C ~ {weather.t_max}°C (平均{weather.t_mean}°C)")
    print(f"  相对湿度: {weather.rh}%")
    print(f"  风速: {weather.u2} m/s")
    print(f"  太阳辐射: {weather.rs} MJ/m²/day")
    
    # 1.2 FAO-56 Penman-Monteith计算
    et0_pm = eto_penman_monteith(weather)
    print(f"\n【Penman-Monteith ET₀】")
    print(f"  ET₀ = {et0_pm:.2f} mm/day")
    
    # 1.3 天文辐射
    Ra = extraterrestrial_radiation(weather.latitude, weather.doy)
    print(f"\n【天文辐射】")
    print(f"  Ra = {Ra:.2f} MJ/m²/day")
    
    # 1.4 Hargreaves简化方法(仅需温度)
    et0_hg = eto_hargreaves(weather.t_mean, weather.t_max, weather.t_min, Ra)
    print(f"\n【Hargreaves ET₀ (简化方法)】")
    print(f"  ET₀ = {et0_hg:.2f} mm/day")
    print(f"  与PM方法差异: {abs(et0_pm - et0_hg):.2f} mm/day ({abs(et0_pm-et0_hg)/et0_pm*100:.1f}%)")
    
    # 1.5 作物系数
    print(f"\n【作物系数 Kc】")
    crops = ["wheat", "maize", "cotton", "alfalfa"]
    stages = ["initial", "mid", "late"]
    
    print(f"{'作物':>8} {'初期':>6} {'中期':>6} {'末期':>6}")
    print("-" * 30)
    for crop in crops:
        kc_vals = [crop_coefficient(stage, crop) for stage in stages]
        print(f"{crop:>8} {kc_vals[0]:>6.2f} {kc_vals[1]:>6.2f} {kc_vals[2]:>6.2f}")
    
    # 1.6 作物蒸散发ETc
    kc_wheat_mid = crop_coefficient("mid", "wheat")
    etc_wheat = etc_crop(et0_pm, kc_wheat_mid)
    
    print(f"\n【小麦中期蒸散发】")
    print(f"  ET₀ = {et0_pm:.2f} mm/day")
    print(f"  Kc = {kc_wheat_mid}")
    print(f"  ETc = ET₀ × Kc = {etc_wheat:.2f} mm/day")
    
    # 1.7 水分胁迫调整
    ks = 0.85  # 假设轻度水分胁迫
    etc_actual = etc_crop(et0_pm, kc_wheat_mid, ks)
    print(f"\n【考虑水分胁迫】")
    print(f"  Ks = {ks}")
    print(f"  ETc_adj = ET₀ × Kc × Ks = {etc_actual:.2f} mm/day")


def example_water_balance():
    """
    示例2: 水量平衡模型
    ===================
    
    跟踪农田水分收支，用于灌溉决策和水资源管理。
    """
    print("\n" + "=" * 60)
    print("示例2: 水量平衡模型")
    print("=" * 60)
    
    # 2.1 创建水量平衡模型
    wb = WaterBalance(
        initial_storage_mm=90,    # 初始储水量
        max_storage_mm=120,       # 田间持水量对应储水
        min_storage_mm=40         # 凋萎点对应储水
    )
    
    print(f"\n【模型参数】")
    print(f"  初始储水: {wb.storage_mm} mm")
    print(f"  最大储水(FC): {wb.max_storage_mm} mm")
    print(f"  最小储水(WP): {wb.min_storage_mm} mm")
    print(f"  可用水分范围: {wb.max_storage_mm - wb.min_storage_mm} mm")
    
    # 2.2 添加降水
    infiltration, runoff = wb.add_precipitation(35)
    print(f"\n【添加35mm降水】")
    print(f"  入渗: {infiltration:.1f} mm")
    print(f"  径流: {runoff:.1f} mm")
    print(f"  当前储水: {wb.storage_mm:.1f} mm")
    
    # 2.3 添加灌溉
    actual_irrig = wb.add_irrigation(30)
    print(f"\n【添加30mm灌溉】")
    print(f"  实际入渗: {actual_irrig:.1f} mm")
    print(f"  当前储水: {wb.storage_mm:.1f} mm")
    
    # 2.4 蒸散发消耗
    actual_et = wb.remove_et(8)
    print(f"\n【8mm蒸散发】")
    print(f"  实际ET: {actual_et:.1f} mm")
    print(f"  当前储水: {wb.storage_mm:.1f} mm")
    
    # 2.5 深层渗漏
    drainage = wb.deep_drainage()
    print(f"\n【深层渗漏】")
    print(f"  渗漏量: {drainage:.1f} mm")
    
    # 2.6 20天逐日模拟
    print(f"\n【20天逐日模拟】")
    
    wb2 = WaterBalance(initial_storage_mm=100, max_storage_mm=120, min_storage_mm=40)
    
    # 模拟天气和灌溉
    np.random.seed(42)
    
    print(f"{'日':>3} {'降水':>6} {'灌溉':>6} {'ET':>6} {'储水':>8} {'状态':>8}")
    print("-" * 45)
    
    for day in range(1, 21):
        # 随机降水
        rain = 20 if np.random.random() < 0.15 else 0
        # ET随时间增加
        et = 4 + day * 0.1 + np.random.random()
        # 灌溉决策
        irrig = 40 if wb2.relative_storage < 0.4 else 0
        
        record = wb2.step_day(precip_mm=rain, irrig_mm=irrig, et_mm=et)
        
        status = "正常" if wb2.relative_storage > 0.5 else ("注意" if wb2.relative_storage > 0.3 else "缺水")
        
        if day % 5 == 0 or day == 1:
            print(f"{day:>3} {rain:>6.0f} {irrig:>6.0f} {record.et_mm:>6.1f} "
                  f"{record.soil_moisture:>8.1f} {status:>8}")
    
    # 2.7 水量平衡汇总
    summary = wb2.get_summary()
    print(f"\n【水量平衡汇总】")
    for key, value in summary.items():
        print(f"  {key}: {value:.1f} mm")
    
    # 2.8 水分利用效率
    yield_kg = 5500  # 假设产量5500 kg/ha
    wue = wb2.water_use_efficiency(yield_kg)
    print(f"\n【水分利用效率】")
    print(f"  产量: {yield_kg} kg/ha")
    print(f"  WUE: {wue:.2f} kg/m³")


def example_irrigation_scheduling():
    """
    示例3: 灌溉调度
    ===============
    
    智能灌溉决策，支持多种调度策略。
    """
    print("\n" + "=" * 60)
    print("示例3: 灌溉调度")
    print("=" * 60)
    
    # 3.1 创建调度器
    scheduler = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5,      # 可用水消耗50%时触发
        max_application_mm=50,      # 单次最大灌溉量
        irrigation_efficiency=0.85  # 灌溉效率85%
    )
    
    print(f"\n【调度器配置】")
    print(f"  调度方法: {scheduler.method.value}")
    print(f"  触发阈值: {scheduler.trigger_threshold} (可用水消耗比例)")
    print(f"  最大灌溉量: {scheduler.max_application_mm} mm")
    print(f"  灌溉效率: {scheduler.efficiency}")
    
    # 3.2 基于土壤水分的灌溉建议
    print(f"\n【土壤水分灌溉建议】")
    
    # 场景1: 水分充足
    rec1 = scheduler.recommend_by_moisture(
        current_moisture=0.28,
        field_capacity=0.32,
        wilting_point=0.12,
        root_depth_m=0.30
    )
    print(f"\n  场景1: 含水量=0.28")
    print(f"    需灌溉: {'是' if rec1.should_irrigate else '否'}")
    print(f"    原因: {rec1.reason}")
    
    # 场景2: 水分不足
    rec2 = scheduler.recommend_by_moisture(
        current_moisture=0.18,
        field_capacity=0.32,
        wilting_point=0.12,
        root_depth_m=0.30
    )
    print(f"\n  场景2: 含水量=0.18")
    print(f"    需灌溉: {'是' if rec2.should_irrigate else '否'}")
    print(f"    建议量: {rec2.amount_mm:.1f} mm")
    print(f"    紧急程度: {rec2.urgency}")
    print(f"    原因: {rec2.reason}")
    
    # 场景3: 严重缺水
    rec3 = scheduler.recommend_by_moisture(
        current_moisture=0.14,
        field_capacity=0.32,
        wilting_point=0.12,
        root_depth_m=0.30
    )
    print(f"\n  场景3: 含水量=0.14 (接近凋萎点)")
    print(f"    需灌溉: {'是' if rec3.should_irrigate else '否'}")
    print(f"    建议量: {rec3.amount_mm:.1f} mm")
    print(f"    紧急程度: {rec3.urgency}")
    
    # 3.3 基于ET的灌溉建议
    scheduler_et = IrrigationScheduler(method=ScheduleType.ET_BASED)
    
    rec_et = scheduler_et.recommend_by_et(
        days_since_irrigation=5,
        cumulative_et_mm=32,
        cumulative_rain_mm=10
    )
    
    print(f"\n【基于ET的灌溉建议】")
    print(f"  距上次灌溉: 5天")
    print(f"  累计ET: 32 mm")
    print(f"  累计降水: 10 mm")
    print(f"  净亏缺: 22 mm")
    print(f"  需灌溉: {'是' if rec_et.should_irrigate else '否'}")
    if rec_et.should_irrigate:
        print(f"  建议量: {rec_et.amount_mm:.1f} mm")
    
    # 3.4 固定间隔灌溉计划
    print(f"\n【固定间隔灌溉计划】")
    events = scheduler.fixed_schedule(
        interval_days=7,
        amount_mm=40,
        total_days=60,
        start_day=0
    )
    
    print(f"  灌溉间隔: 7天")
    print(f"  每次灌溉: 40mm")
    print(f"  计划周期: 60天")
    print(f"  灌溉次数: {len(events)}")
    print(f"  总灌溉量: {scheduler.total_irrigation():.0f} mm")
    
    # 3.5 亏缺灌溉
    print(f"\n【亏缺灌溉计划】")
    daily_et = np.array([5.0 + 0.1*i for i in range(60)])  # 逐日ET
    
    deficit_events = scheduler.deficit_irrigation_schedule(
        full_et_mm=daily_et,
        deficit_fraction=0.75,  # 仅补充75%的ET
        min_interval=4
    )
    
    print(f"  亏缺系数: 0.75 (节水25%)")
    print(f"  灌溉次数: {len(deficit_events)}")
    print(f"  总灌溉量: {sum(e.amount_mm for e in deficit_events):.0f} mm")
    
    # 3.6 计算灌溉需水量
    print(f"\n【灌溉需水量计算】")
    et_mm = 45
    effective_rain = 15
    
    nir = calculate_net_irrigation_requirement(et_mm, effective_rain)
    gir = gross_irrigation_requirement(nir, efficiency=0.85)
    
    print(f"  作物需水(ETc): {et_mm} mm")
    print(f"  有效降水: {effective_rain} mm")
    print(f"  净灌溉需求: {nir:.1f} mm")
    print(f"  毛灌溉需求(效率85%): {gir:.1f} mm")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.water 模块使用示例")
    print("=" * 60)
    
    example_evapotranspiration()
    example_water_balance()
    example_irrigation_scheduling()
    
    print("\n" + "=" * 60)
    print("Water模块示例完成")
    print("=" * 60)
