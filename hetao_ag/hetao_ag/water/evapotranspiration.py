# -*- coding: utf-8 -*-
"""
hetao_ag.water.evapotranspiration - 蒸散发计算

实现FAO-56 Penman-Monteith等蒸散发计算方法。

作者: Hetao College
版本: 1.0.0
"""

import math
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ETMethod(Enum):
    """蒸散发计算方法"""
    PENMAN_MONTEITH = "penman_monteith"
    HARGREAVES = "hargreaves"
    PRIESTLEY_TAYLOR = "priestley_taylor"


@dataclass
class WeatherData:
    """气象数据
    
    属性:
        t_mean: 平均气温(°C)
        t_max: 最高气温(°C)
        t_min: 最低气温(°C)
        rh: 相对湿度(%)
        u2: 2m高度风速(m/s)
        rs: 太阳辐射(MJ/m²/day)
        elevation: 海拔(m)
        latitude: 纬度(度)
        doy: 年积日
    """
    t_mean: float
    t_max: float
    t_min: float
    rh: float = 60.0
    u2: float = 2.0
    rs: Optional[float] = None
    elevation: float = 1000.0
    latitude: float = 40.0
    doy: int = 180


def eto_penman_monteith(weather: WeatherData) -> float:
    """FAO-56 Penman-Monteith参考作物蒸散发
    
    国际标准方法,适用于草地参考作物。
    
    参数:
        weather: WeatherData对象
        
    返回:
        日参考蒸散发ET₀ (mm/day)
        
    参考:
        Allen et al. (1998) FAO Irrigation and Drainage Paper 56
    """
    T = weather.t_mean
    Tmax = weather.t_max
    Tmin = weather.t_min
    RH = weather.rh
    u2 = weather.u2
    elev = weather.elevation
    lat = weather.latitude
    doy = weather.doy
    
    # 太阳常数
    Gsc = 0.0820  # MJ/m²/min
    
    # 饱和水汽压(kPa) - Tetens公式
    es_tmax = 0.6108 * math.exp(17.27 * Tmax / (Tmax + 237.3))
    es_tmin = 0.6108 * math.exp(17.27 * Tmin / (Tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0  # 平均饱和水汽压
    
    # 实际水汽压
    ea = es * RH / 100.0
    
    # 饱和水汽压曲线斜率(kPa/°C)
    delta = 4098 * (0.6108 * math.exp(17.27 * T / (T + 237.3))) / ((T + 237.3) ** 2)
    
    # 大气压(kPa)
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    
    # 干湿表常数
    gamma = 0.000665 * P
    
    # 日地距离校正因子
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    
    # 太阳赤纬角(rad)
    delta_sol = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    
    # 纬度(rad)
    lat_rad = math.radians(lat)
    
    # 日落时角
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta_sol))
    
    # 天文辐射(MJ/m²/day)
    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(lat_rad) * math.sin(delta_sol) +
        math.cos(lat_rad) * math.cos(delta_sol) * math.sin(ws)
    )
    
    # 太阳辐射(如未提供,用Hargreaves公式估算)
    if weather.rs is not None:
        Rs = weather.rs
    else:
        krs = 0.16  # 内陆地区系数
        Rs = krs * math.sqrt(Tmax - Tmin) * Ra
    
    # 晴空辐射
    Rso = (0.75 + 2e-5 * elev) * Ra
    
    # 净短波辐射(反射率0.23)
    Rns = 0.77 * Rs
    
    # 净长波辐射
    Rs_Rso = Rs / Rso if Rso > 0 else 0.5
    Rs_Rso = min(1.0, Rs_Rso)
    
    Rnl = (4.903e-9 * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4) / 2 *
           (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * Rs_Rso - 0.35))
    
    # 净辐射
    Rn = Rns - Rnl
    
    # 土壤热通量(日尺度假设为0)
    G = 0
    
    # FAO-56 Penman-Monteith公式
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    
    ET0 = numerator / denominator
    
    return max(0, ET0)


def eto_hargreaves(t_mean: float, t_max: float, t_min: float, Ra: float) -> float:
    """Hargreaves蒸散发估算
    
    简化方法,仅需温度数据。
    
    参数:
        t_mean: 平均气温(°C)
        t_max: 最高气温(°C)
        t_min: 最低气温(°C)
        Ra: 天文辐射(MJ/m²/day)
        
    返回:
        日参考蒸散发(mm/day)
    """
    ET0 = 0.0023 * (t_mean + 17.8) * math.sqrt(t_max - t_min) * Ra * 0.408
    return max(0, ET0)


def extraterrestrial_radiation(latitude: float, doy: int) -> float:
    """计算天文辐射
    
    参数:
        latitude: 纬度(度)
        doy: 年积日
        
    返回:
        天文辐射Ra (MJ/m²/day)
    """
    Gsc = 0.0820
    lat_rad = math.radians(latitude)
    
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    delta = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    
    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(lat_rad) * math.sin(delta) +
        math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    
    return Ra


def crop_coefficient(growth_stage: str, crop: str = "wheat") -> float:
    """作物系数Kc
    
    参数:
        growth_stage: 生长阶段(initial, mid, late)
        crop: 作物类型
        
    返回:
        Kc值
    """
    KC_VALUES = {
        "wheat": {"initial": 0.3, "mid": 1.15, "late": 0.4},
        "maize": {"initial": 0.3, "mid": 1.20, "late": 0.6},
        "rice": {"initial": 1.05, "mid": 1.20, "late": 0.9},
        "cotton": {"initial": 0.35, "mid": 1.20, "late": 0.7},
        "alfalfa": {"initial": 0.4, "mid": 0.95, "late": 0.9},
        "sunflower": {"initial": 0.35, "mid": 1.15, "late": 0.35},
    }
    
    crop_kc = KC_VALUES.get(crop.lower(), KC_VALUES["wheat"])
    return crop_kc.get(growth_stage.lower(), 1.0)


def etc_crop(et0: float, kc: float, ks: float = 1.0) -> float:
    """计算作物蒸散发ETc
    
    参数:
        et0: 参考蒸散发(mm/day)
        kc: 作物系数
        ks: 水分胁迫系数(0-1)
        
    返回:
        作物蒸散发(mm/day)
    """
    return et0 * kc * ks


if __name__ == "__main__":
    print("=" * 50)
    print("蒸散发计算演示")
    print("=" * 50)
    
    # 创建气象数据
    weather = WeatherData(
        t_mean=25.0,
        t_max=32.0,
        t_min=18.0,
        rh=55.0,
        u2=2.0,
        rs=22.0,
        elevation=1050,
        latitude=40.8,
        doy=180
    )
    
    # 计算ET0
    et0_pm = eto_penman_monteith(weather)
    print(f"\nPenman-Monteith ET₀: {et0_pm:.2f} mm/day")
    
    # Hargreaves方法
    Ra = extraterrestrial_radiation(40.8, 180)
    et0_hg = eto_hargreaves(25.0, 32.0, 18.0, Ra)
    print(f"Hargreaves ET₀: {et0_hg:.2f} mm/day")
    
    # 作物蒸散发
    kc = crop_coefficient("mid", "wheat")
    etc = etc_crop(et0_pm, kc)
    print(f"\n小麦ETc (生育中期): {etc:.2f} mm/day")
