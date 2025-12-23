# -*- coding: utf-8 -*-
"""
Core模块使用示例
================

演示hetao_ag.core模块的核心功能：单位系统、配置管理、日志和工具函数。

作者: Hetao College
"""

from hetao_ag.core import (
    # 单位系统
    Unit, Quantity, Dimension, DimensionError,
    meters, kilometers, hectares, celsius, kilopascals, ds_per_m,
    # 配置管理  
    ConfigManager, ConfigError, create_default_config,
    # 日志
    Logger, get_logger,
    # 工具函数
    safe_divide, clamp, linear_interpolate, day_of_year,
    rmse, mae, r_squared, validate_model, ValidationResult, Timer
)
import numpy as np


def example_unit_system():
    """
    示例1: 单位系统
    ===============
    
    hetao_ag使用SI国际单位制，提供物理量的自动转换和算术运算。
    这确保了科学计算的准确性，避免单位混淆导致的错误。
    """
    print("\n" + "=" * 60)
    print("示例1: 单位系统")
    print("=" * 60)
    
    # 1.1 创建物理量
    # 使用Quantity类或便捷函数
    distance1 = Quantity(5.5, Unit.KILOMETER)
    distance2 = kilometers(5.5)  # 等效写法
    
    print(f"\n距离: {distance1}")
    print(f"等效: {distance2}")
    
    # 1.2 单位转换
    # 自动转换为目标单位
    distance_m = distance1.to(Unit.METER)
    print(f"转换为米: {distance_m}")
    
    # 1.3 温度转换（特殊处理摄氏度-开尔文偏移）
    temp_c = celsius(25.0)
    temp_k = temp_c.to(Unit.KELVIN)
    print(f"\n温度: {temp_c} = {temp_k}")
    
    # 1.4 农业常用单位
    farm_area = hectares(150)  # 150公顷农场
    soil_ec = ds_per_m(4.5)    # 土壤电导率
    pressure = kilopascals(101.3)  # 大气压
    
    print(f"\n农场面积: {farm_area}")
    print(f"土壤电导率: {soil_ec}")
    print(f"大气压: {pressure}")
    
    # 1.5 算术运算
    # 相同维度的物理量可以进行加减运算
    a = meters(100)
    b = meters(50)
    c = Quantity(0.2, Unit.KILOMETER)  # 200米
    
    print(f"\n算术运算:")
    print(f"  {a} + {b} = {a + b}")
    print(f"  {a} - {b} = {a - b}")
    print(f"  {a} + {c.to(Unit.METER)} = {a + c}")  # 自动转换后相加
    
    # 1.6 数值乘除
    doubled = a * 2
    half = a / 2
    print(f"  {a} × 2 = {doubled}")
    print(f"  {a} ÷ 2 = {half}")
    
    # 1.7 比较运算
    print(f"\n比较运算:")
    print(f"  {a} > {b}: {a > b}")
    print(f"  {a} == {meters(100)}: {a == meters(100)}")
    
    # 1.8 维度检查
    # 不同维度的物理量相加会抛出异常
    try:
        result = meters(100) + celsius(25)  # 错误：长度不能与温度相加
    except DimensionError as e:
        print(f"\n维度错误(预期): {e}")


def example_config_manager():
    """
    示例2: 配置管理
    ===============
    
    ConfigManager支持YAML/JSON配置文件、环境变量覆盖、嵌套键访问等功能。
    这对于可复现的科学研究和灵活的系统部署至关重要。
    """
    print("\n" + "=" * 60)
    print("示例2: 配置管理")
    print("=" * 60)
    
    # 2.1 创建默认配置
    default_config = create_default_config()
    print(f"\n默认配置项数: {len(default_config)}")
    
    # 2.2 从默认值初始化ConfigManager
    config = ConfigManager(defaults=default_config)
    
    # 2.3 获取配置项（支持嵌套键）
    soil_fc = config.get("soil.field_capacity")
    print(f"土壤田间持水量: {soil_fc}")
    
    base_temp = config.get("crop.thermal_time_base_celsius")
    print(f"作物基础温度: {base_temp}°C")
    
    # 2.4 带默认值的获取
    custom_param = config.get("custom.param", default=42)
    print(f"自定义参数(使用默认值): {custom_param}")
    
    # 2.5 设置配置项
    config.set("irrigation.efficiency", 0.90)
    print(f"\n设置灌溉效率: {config.get('irrigation.efficiency')}")
    
    # 2.6 检查配置项是否存在
    print(f"\n'soil.field_capacity'存在: {config.has('soil.field_capacity')}")
    print(f"'unknown.key'存在: {config.has('unknown.key')}")
    
    # 2.7 验证必需配置
    required_keys = ["soil.field_capacity", "soil.wilting_point"]
    is_valid = config.validate(required_keys)
    print(f"\n配置验证通过: {is_valid}")


def example_logger():
    """
    示例3: 日志系统
    ===============
    
    Logger提供统一的日志接口，支持彩色控制台输出、文件轮转和实验追踪。
    """
    print("\n" + "=" * 60)
    print("示例3: 日志系统")
    print("=" * 60)
    
    # 3.1 获取日志器
    logger = get_logger("hetao_demo")
    
    # 3.2 不同级别的日志
    print("\n日志输出演示:")
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    
    # 3.3 带额外信息的结构化日志
    logger.info("灌溉事件", amount_mm=35, field_id="F001")
    
    # 3.4 实验追踪
    logger.log_experiment_start(
        experiment_name="灌溉优化实验",
        parameters={"method": "deficit", "target_etc": 0.8},
        random_seed=42
    )
    
    # ... 实验代码 ...
    
    logger.log_experiment_end(
        success=True,
        results={"water_saved": 15.2, "yield_ratio": 0.95}
    )


def example_utility_functions():
    """
    示例4: 工具函数
    ===============
    
    提供常用的数学计算、插值、模型验证等实用函数。
    """
    print("\n" + "=" * 60)
    print("示例4: 工具函数")
    print("=" * 60)
    
    # 4.1 安全除法
    result = safe_divide(10, 0)  # 除零返回默认值0
    print(f"\n安全除法 10/0 = {result}")
    
    result = safe_divide(10, 0, default=-1)  # 自定义默认值
    print(f"安全除法 10/0 (默认-1) = {result}")
    
    # 4.2 值限制
    value = clamp(15, min_val=0, max_val=10)
    print(f"\nclamp(15, 0, 10) = {value}")
    
    value = clamp(-5, min_val=0, max_val=10)
    print(f"clamp(-5, 0, 10) = {value}")
    
    # 4.3 线性插值
    y = linear_interpolate(x=1.5, x1=1, y1=10, x2=2, y2=20)
    print(f"\n线性插值 f(1.5) = {y}")
    
    # 4.4 年积日计算
    from datetime import date
    doy = day_of_year(date(2024, 7, 15))
    print(f"\n2024年7月15日是第{doy}天")
    
    # 4.5 模型验证指标
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    print(f"\n模型验证指标:")
    print(f"  RMSE: {rmse(observed, predicted):.4f}")
    print(f"  MAE:  {mae(observed, predicted):.4f}")
    print(f"  R^2:  {r_squared(observed, predicted):.4f}")
    
    # 4.6 综合验证
    result = validate_model(observed, predicted)
    print(f"\n{result}")
    
    # 4.7 计时器
    print("\n计时器演示:")
    with Timer("模拟计算"):
        # 模拟耗时计算
        total = sum(range(500000))


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.core 模块使用示例")
    print("=" * 60)
    
    example_unit_system()
    example_config_manager()
    example_logger()
    example_utility_functions()
    
    print("\n" + "=" * 60)
    print("Core模块示例完成")
    print("=" * 60)
