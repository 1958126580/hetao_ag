# -*- coding: utf-8 -*-
"""
hetao_ag.core.units - 单位系统模块
=====================================

提供国际标准(SI)单位系统，支持自动单位转换和物理量运算。

主要功能:
    - Unit: 单位枚举类，定义常用单位及其SI转换因子
    - Quantity: 物理量类，支持带单位的数值运算
    - DimensionError: 量纲不匹配异常

设计原则:
    - 内部统一使用SI单位，避免单位混淆导致的严重错误
    - 参考NASA火星气候轨道器事故教训（公制与英制混淆）
    - 符合国际计量标准

作者: Hetao College
版本: 1.0.0
"""

from enum import Enum
from typing import Union, Optional
from dataclasses import dataclass
import math


class DimensionError(Exception):
    """量纲不匹配异常
    
    当尝试对不同量纲的物理量进行不兼容运算时抛出。
    
    示例:
        >>> length = Quantity(1.0, Unit.METER)
        >>> mass = Quantity(1.0, Unit.KILOGRAM)
        >>> length + mass  # 引发 DimensionError
    """
    pass


class Dimension(Enum):
    """物理量纲枚举
    
    定义基本物理量纲，用于确保运算的量纲一致性。
    
    属性:
        LENGTH: 长度量纲
        AREA: 面积量纲
        VOLUME: 体积量纲
        MASS: 质量量纲
        TIME: 时间量纲
        TEMPERATURE: 温度量纲
        PRESSURE: 压强量纲
        VELOCITY: 速度量纲
        CONDUCTIVITY: 电导率量纲
        DIMENSIONLESS: 无量纲
    """
    LENGTH = "length"
    AREA = "area"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VELOCITY = "velocity"
    CONDUCTIVITY = "conductivity"
    ENERGY = "energy"
    POWER = "power"
    DIMENSIONLESS = "dimensionless"


class Unit(Enum):
    """单位枚举类
    
    定义常用物理单位及其到SI基本单位的转换因子。
    每个单位包含: (符号, SI转换因子, 量纲)
    
    长度单位:
        METER, KILOMETER, CENTIMETER, MILLIMETER
        
    面积单位:
        SQUARE_METER, HECTARE, SQUARE_KILOMETER
        
    体积单位:
        CUBIC_METER, LITER, MILLILITER
        
    质量单位:
        KILOGRAM, GRAM, TON
        
    时间单位:
        SECOND, MINUTE, HOUR, DAY
        
    温度单位:
        KELVIN, CELSIUS (注意: 摄氏度需要偏移转换)
        
    压强单位:
        PASCAL, KILOPASCAL, BAR
        
    速度单位:
        METER_PER_SECOND, KILOMETER_PER_HOUR
        
    电导率单位:
        SIEMENS_PER_METER, DECISIEMENS_PER_METER
        
    能量和功率单位:
        JOULE, MEGAJOULE, WATT, KILOWATT
    
    示例:
        >>> Unit.METER.symbol
        'm'
        >>> Unit.KILOMETER.to_si_factor
        1000.0
    """
    
    # 长度单位 (基本单位: 米)
    METER = ("m", 1.0, Dimension.LENGTH)
    KILOMETER = ("km", 1000.0, Dimension.LENGTH)
    CENTIMETER = ("cm", 0.01, Dimension.LENGTH)
    MILLIMETER = ("mm", 0.001, Dimension.LENGTH)
    
    # 面积单位 (基本单位: 平方米)
    SQUARE_METER = ("m²", 1.0, Dimension.AREA)
    HECTARE = ("ha", 10000.0, Dimension.AREA)
    SQUARE_KILOMETER = ("km²", 1000000.0, Dimension.AREA)
    
    # 体积单位 (基本单位: 立方米)
    CUBIC_METER = ("m³", 1.0, Dimension.VOLUME)
    LITER = ("L", 0.001, Dimension.VOLUME)
    MILLILITER = ("mL", 0.000001, Dimension.VOLUME)
    
    # 质量单位 (基本单位: 千克)
    KILOGRAM = ("kg", 1.0, Dimension.MASS)
    GRAM = ("g", 0.001, Dimension.MASS)
    TON = ("t", 1000.0, Dimension.MASS)
    
    # 时间单位 (基本单位: 秒)
    SECOND = ("s", 1.0, Dimension.TIME)
    MINUTE = ("min", 60.0, Dimension.TIME)
    HOUR = ("h", 3600.0, Dimension.TIME)
    DAY = ("d", 86400.0, Dimension.TIME)
    
    # 温度单位 (基本单位: 开尔文, 注意摄氏度需要偏移)
    KELVIN = ("K", 1.0, Dimension.TEMPERATURE)
    CELSIUS = ("°C", 1.0, Dimension.TEMPERATURE)  # 需要特殊处理偏移
    
    # 压强单位 (基本单位: 帕斯卡)
    PASCAL = ("Pa", 1.0, Dimension.PRESSURE)
    KILOPASCAL = ("kPa", 1000.0, Dimension.PRESSURE)
    BAR = ("bar", 100000.0, Dimension.PRESSURE)
    
    # 速度单位 (基本单位: 米/秒)
    METER_PER_SECOND = ("m/s", 1.0, Dimension.VELOCITY)
    KILOMETER_PER_HOUR = ("km/h", 1.0/3.6, Dimension.VELOCITY)
    
    # 电导率单位 (基本单位: S/m, 土壤盐分常用dS/m)
    SIEMENS_PER_METER = ("S/m", 1.0, Dimension.CONDUCTIVITY)
    DECISIEMENS_PER_METER = ("dS/m", 0.1, Dimension.CONDUCTIVITY)
    
    # 能量单位 (基本单位: 焦耳)
    JOULE = ("J", 1.0, Dimension.ENERGY)
    MEGAJOULE = ("MJ", 1000000.0, Dimension.ENERGY)
    KILOJOULE = ("kJ", 1000.0, Dimension.ENERGY)
    
    # 功率单位 (基本单位: 瓦特)
    WATT = ("W", 1.0, Dimension.POWER)
    KILOWATT = ("kW", 1000.0, Dimension.POWER)
    
    # 无量纲
    PERCENT = ("%", 0.01, Dimension.DIMENSIONLESS)
    FRACTION = ("", 1.0, Dimension.DIMENSIONLESS)
    
    def __init__(self, symbol: str, to_si: float, dimension: Dimension):
        """初始化单位
        
        参数:
            symbol: 单位符号
            to_si: 到SI基本单位的转换因子
            dimension: 物理量纲
        """
        self._symbol = symbol
        self._to_si = to_si
        self._dimension = dimension
    
    @property
    def symbol(self) -> str:
        """获取单位符号"""
        return self._symbol
    
    @property
    def to_si_factor(self) -> float:
        """获取到SI单位的转换因子"""
        return self._to_si
    
    @property
    def dimension(self) -> Dimension:
        """获取物理量纲"""
        return self._dimension


@dataclass
class Quantity:
    """物理量类
    
    表示带有单位的数值，支持自动单位转换和算术运算。
    所有运算自动确保量纲一致性。
    
    属性:
        value: 数值
        unit: 单位
        
    方法:
        to(unit): 转换到指定单位
        to_si(): 转换到SI基本单位
        
    运算符支持:
        +, -, *, /, ==, <, >, <=, >=
        
    示例:
        >>> length1 = Quantity(100, Unit.CENTIMETER)
        >>> length2 = Quantity(2, Unit.METER)
        >>> total = length1 + length2
        >>> print(total)  # 3.0 m (以第一个操作数的单位表示)
        
        >>> # 单位转换
        >>> distance = Quantity(5, Unit.KILOMETER)
        >>> print(distance.to(Unit.METER))  # 5000.0 m
    """
    
    value: float
    unit: Unit
    
    def __post_init__(self):
        """验证初始化参数"""
        if not isinstance(self.unit, Unit):
            raise TypeError(f"unit必须是Unit枚举类型，得到: {type(self.unit)}")
        self.value = float(self.value)
    
    def to_si(self) -> 'Quantity':
        """转换到SI基本单位
        
        返回:
            新的Quantity对象，使用SI基本单位
            
        示例:
            >>> q = Quantity(5, Unit.KILOMETER)
            >>> q.to_si()
            Quantity(value=5000.0, unit=<Unit.METER>)
        """
        # 找到同量纲的SI基本单位
        si_units = {
            Dimension.LENGTH: Unit.METER,
            Dimension.AREA: Unit.SQUARE_METER,
            Dimension.VOLUME: Unit.CUBIC_METER,
            Dimension.MASS: Unit.KILOGRAM,
            Dimension.TIME: Unit.SECOND,
            Dimension.TEMPERATURE: Unit.KELVIN,
            Dimension.PRESSURE: Unit.PASCAL,
            Dimension.VELOCITY: Unit.METER_PER_SECOND,
            Dimension.CONDUCTIVITY: Unit.SIEMENS_PER_METER,
            Dimension.ENERGY: Unit.JOULE,
            Dimension.POWER: Unit.WATT,
            Dimension.DIMENSIONLESS: Unit.FRACTION,
        }
        target_unit = si_units.get(self.unit.dimension, self.unit)
        return self.to(target_unit)
    
    def to(self, target_unit: Unit) -> 'Quantity':
        """转换到指定单位
        
        参数:
            target_unit: 目标单位
            
        返回:
            新的Quantity对象
            
        异常:
            DimensionError: 量纲不匹配
            
        示例:
            >>> q = Quantity(1, Unit.KILOMETER)
            >>> q.to(Unit.METER)
            Quantity(value=1000.0, unit=<Unit.METER>)
        """
        if self.unit.dimension != target_unit.dimension:
            raise DimensionError(
                f"无法将{self.unit.dimension.value}转换为{target_unit.dimension.value}"
            )
        
        # 温度需要特殊处理（摄氏度与开尔文之间有偏移）
        if self.unit.dimension == Dimension.TEMPERATURE:
            return self._convert_temperature(target_unit)
        
        # 标准转换: 当前值 -> SI值 -> 目标值
        si_value = self.value * self.unit.to_si_factor
        new_value = si_value / target_unit.to_si_factor
        return Quantity(new_value, target_unit)
    
    def _convert_temperature(self, target_unit: Unit) -> 'Quantity':
        """温度转换（处理摄氏度偏移）
        
        参数:
            target_unit: 目标温度单位
            
        返回:
            转换后的温度Quantity
        """
        # 先转换到开尔文
        if self.unit == Unit.CELSIUS:
            kelvin_value = self.value + 273.15
        else:
            kelvin_value = self.value
        
        # 再转换到目标单位
        if target_unit == Unit.CELSIUS:
            new_value = kelvin_value - 273.15
        else:
            new_value = kelvin_value
        
        return Quantity(new_value, target_unit)
    
    def __add__(self, other: 'Quantity') -> 'Quantity':
        """加法运算
        
        结果使用第一个操作数的单位。
        
        参数:
            other: 另一个Quantity
            
        返回:
            相加结果
            
        异常:
            TypeError: other不是Quantity
            DimensionError: 量纲不匹配
        """
        if not isinstance(other, Quantity):
            raise TypeError("只能与Quantity相加")
        if self.unit.dimension != other.unit.dimension:
            raise DimensionError(
                f"无法相加: {self.unit.dimension.value} 与 {other.unit.dimension.value}"
            )
        
        # 将other转换到self的单位
        other_converted = other.to(self.unit)
        return Quantity(self.value + other_converted.value, self.unit)
    
    def __sub__(self, other: 'Quantity') -> 'Quantity':
        """减法运算"""
        if not isinstance(other, Quantity):
            raise TypeError("只能与Quantity相减")
        if self.unit.dimension != other.unit.dimension:
            raise DimensionError(
                f"无法相减: {self.unit.dimension.value} 与 {other.unit.dimension.value}"
            )
        
        other_converted = other.to(self.unit)
        return Quantity(self.value - other_converted.value, self.unit)
    
    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        """乘法运算
        
        支持与标量或其他Quantity相乘。
        注意: 两个Quantity相乘会产生新的量纲。
        
        参数:
            other: 标量或Quantity
            
        返回:
            乘积结果
        """
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        elif isinstance(other, Quantity):
            # 两个Quantity相乘（简化处理，返回SI值的乘积）
            si_product = (self.value * self.unit.to_si_factor * 
                         other.value * other.unit.to_si_factor)
            return Quantity(si_product, Unit.FRACTION)
        else:
            raise TypeError(f"不支持与{type(other)}相乘")
    
    def __rmul__(self, other: Union[float, int]) -> 'Quantity':
        """右乘运算"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        """除法运算"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("除数不能为零")
            return Quantity(self.value / other, self.unit)
        elif isinstance(other, Quantity):
            if other.value == 0:
                raise ZeroDivisionError("除数不能为零")
            # 同量纲相除得到无量纲数
            if self.unit.dimension == other.unit.dimension:
                other_converted = other.to(self.unit)
                return Quantity(self.value / other_converted.value, Unit.FRACTION)
            else:
                # 不同量纲相除
                si_quotient = (self.value * self.unit.to_si_factor / 
                              (other.value * other.unit.to_si_factor))
                return Quantity(si_quotient, Unit.FRACTION)
        else:
            raise TypeError(f"不支持与{type(other)}相除")
    
    def __eq__(self, other: 'Quantity') -> bool:
        """相等比较"""
        if not isinstance(other, Quantity):
            return False
        if self.unit.dimension != other.unit.dimension:
            return False
        other_converted = other.to(self.unit)
        return math.isclose(self.value, other_converted.value, rel_tol=1e-9)
    
    def __lt__(self, other: 'Quantity') -> bool:
        """小于比较"""
        if not isinstance(other, Quantity):
            raise TypeError("只能与Quantity比较")
        if self.unit.dimension != other.unit.dimension:
            raise DimensionError("无法比较不同量纲的物理量")
        other_converted = other.to(self.unit)
        return self.value < other_converted.value
    
    def __le__(self, other: 'Quantity') -> bool:
        """小于等于比较"""
        return self == other or self < other
    
    def __gt__(self, other: 'Quantity') -> bool:
        """大于比较"""
        return not self <= other
    
    def __ge__(self, other: 'Quantity') -> bool:
        """大于等于比较"""
        return not self < other
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Quantity({self.value}, {self.unit.name})"
    
    def __str__(self) -> str:
        """人类可读的字符串"""
        return f"{self.value:.4g} {self.unit.symbol}"
    
    def __float__(self) -> float:
        """转换为浮点数（返回SI值）"""
        return self.value * self.unit.to_si_factor
    
    def __neg__(self) -> 'Quantity':
        """取负"""
        return Quantity(-self.value, self.unit)
    
    def __abs__(self) -> 'Quantity':
        """取绝对值"""
        return Quantity(abs(self.value), self.unit)


# ============================================================================
# 便捷函数和常量
# ============================================================================

def meters(value: float) -> Quantity:
    """创建米为单位的长度
    
    示例:
        >>> distance = meters(100)
        >>> print(distance)
        100.0 m
    """
    return Quantity(value, Unit.METER)


def kilometers(value: float) -> Quantity:
    """创建千米为单位的长度"""
    return Quantity(value, Unit.KILOMETER)


def hectares(value: float) -> Quantity:
    """创建公顷为单位的面积"""
    return Quantity(value, Unit.HECTARE)


def celsius(value: float) -> Quantity:
    """创建摄氏度为单位的温度"""
    return Quantity(value, Unit.CELSIUS)


def kilopascals(value: float) -> Quantity:
    """创建千帕为单位的压强"""
    return Quantity(value, Unit.KILOPASCAL)


def ds_per_m(value: float) -> Quantity:
    """创建dS/m为单位的电导率（土壤盐分常用单位）
    
    示例:
        >>> soil_ec = ds_per_m(4.5)  # 土壤盐分4.5 dS/m
    """
    return Quantity(value, Unit.DECISIEMENS_PER_METER)


def megajoules_per_m2(value: float) -> Quantity:
    """创建MJ/m²为单位的能量（太阳辐射常用单位）"""
    return Quantity(value, Unit.MEGAJOULE)


def mm_per_day(value: float) -> Quantity:
    """创建mm/day为单位的蒸散发速率
    
    注: 1 mm/day 水深变化相当于 1 L/m²/day
    """
    return Quantity(value, Unit.MILLIMETER)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("河套智慧农牧业库 - 单位系统演示")
    print("=" * 60)
    
    # 演示1: 长度单位转换
    print("\n【长度单位转换】")
    distance_km = Quantity(5.5, Unit.KILOMETER)
    distance_m = distance_km.to(Unit.METER)
    print(f"  {distance_km} = {distance_m}")
    
    # 演示2: 单位加法（自动转换）
    print("\n【单位加法】")
    length1 = Quantity(100, Unit.CENTIMETER)
    length2 = Quantity(2, Unit.METER)
    total = length1 + length2
    print(f"  {length1} + {length2} = {total}")
    
    # 演示3: 面积单位
    print("\n【面积单位】")
    field_area = Quantity(150, Unit.HECTARE)
    field_sqm = field_area.to(Unit.SQUARE_METER)
    print(f"  农田面积: {field_area} = {field_sqm}")
    
    # 演示4: 温度转换
    print("\n【温度转换】")
    temp_c = Quantity(25, Unit.CELSIUS)
    temp_k = temp_c.to(Unit.KELVIN)
    print(f"  {temp_c} = {temp_k}")
    
    # 演示5: 土壤盐分（电导率）
    print("\n【土壤盐分】")
    soil_ec = ds_per_m(6.5)
    print(f"  土壤电导率: {soil_ec}")
    
    # 演示6: 便捷函数
    print("\n【便捷函数使用】")
    farm_size = hectares(500)
    daily_temp = celsius(28.5)
    print(f"  农场规模: {farm_size}")
    print(f"  日均温度: {daily_temp}")
    
    print("\n" + "=" * 60)
    print("单位系统演示完成！")
    print("=" * 60)
