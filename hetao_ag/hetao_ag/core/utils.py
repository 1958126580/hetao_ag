# -*- coding: utf-8 -*-
"""
hetao_ag.core.utils - 通用工具模块

提供共享的辅助函数和基类。

作者: Hetao College
版本: 1.0.0
"""

import math
import numpy as np
from typing import Union, Optional, Callable, Any, List
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误
    
    参数:
        numerator: 分子
        denominator: 分母
        default: 除零时返回的默认值
        
    返回:
        除法结果或默认值
        
    示例:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
    """
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在指定范围内
    
    参数:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
        
    返回:
        限制后的值
    """
    return max(min_val, min(value, max_val))


def linear_interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """线性插值
    
    参数:
        x: 插值点
        x1, y1: 第一个已知点
        x2, y2: 第二个已知点
        
    返回:
        插值结果
    """
    if x1 == x2:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def array_interpolate(x: float, x_array: np.ndarray, y_array: np.ndarray) -> float:
    """数组线性插值
    
    参数:
        x: 插值点
        x_array: x值数组（已排序）
        y_array: y值数组
        
    返回:
        插值结果
    """
    return float(np.interp(x, x_array, y_array))


def day_of_year(dt: Union[datetime, date]) -> int:
    """获取年内天数（1-365/366）
    
    参数:
        dt: 日期对象
        
    返回:
        年内天数
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt.timetuple().tm_yday


def degrees_to_radians(degrees: float) -> float:
    """角度转弧度"""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """弧度转角度"""
    return math.degrees(radians)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """计算移动平均
    
    参数:
        data: 输入数据数组
        window: 窗口大小
        
    返回:
        移动平均数组
    """
    if window <= 0:
        raise ValueError("窗口大小必须为正整数")
    if len(data) < window:
        return data.copy()
    
    return np.convolve(data, np.ones(window) / window, mode='valid')


def normalize(data: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """将数据归一化到指定范围
    
    参数:
        data: 输入数据
        min_val: 目标最小值
        max_val: 目标最大值
        
    返回:
        归一化后的数组
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.full_like(data, (min_val + max_val) / 2, dtype=float)
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """计算均方根误差
    
    参数:
        observed: 观测值
        predicted: 预测值
        
    返回:
        RMSE值
    """
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """计算平均绝对误差
    
    参数:
        observed: 观测值
        predicted: 预测值
        
    返回:
        MAE值
    """
    return float(np.mean(np.abs(observed - predicted)))


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """计算决定系数R²
    
    参数:
        observed: 观测值
        predicted: 预测值
        
    返回:
        R²值
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1.0 - ss_res / ss_tot


def ensure_path(path: Union[str, Path]) -> Path:
    """确保路径对象
    
    参数:
        path: 字符串或Path对象
        
    返回:
        Path对象
    """
    return Path(path) if isinstance(path, str) else path


def ensure_directory(path: Union[str, Path]) -> Path:
    """确保目录存在
    
    参数:
        path: 目录路径
        
    返回:
        Path对象
    """
    p = ensure_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class ValidationResult:
    """验证结果数据类
    
    属性:
        rmse: 均方根误差
        mae: 平均绝对误差
        r_squared: 决定系数
        n_samples: 样本数量
    """
    rmse: float
    mae: float
    r_squared: float
    n_samples: int
    
    def __str__(self) -> str:
        return (f"验证结果: RMSE={self.rmse:.4f}, MAE={self.mae:.4f}, "
                f"R^2={self.r_squared:.4f}, N={self.n_samples}")


def validate_model(observed: np.ndarray, predicted: np.ndarray) -> ValidationResult:
    """模型验证
    
    参数:
        observed: 观测值数组
        predicted: 预测值数组
        
    返回:
        ValidationResult对象
    """
    obs = np.asarray(observed).flatten()
    pred = np.asarray(predicted).flatten()
    
    if len(obs) != len(pred):
        raise ValueError("观测值和预测值数组长度必须相同")
    
    return ValidationResult(
        rmse=rmse(obs, pred),
        mae=mae(obs, pred),
        r_squared=r_squared(obs, pred),
        n_samples=len(obs)
    )


class Timer:
    """计时器上下文管理器
    
    示例:
        >>> with Timer("模型训练"):
        ...     train_model()
        模型训练 耗时: 2.34 秒
    """
    
    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} 耗时: {elapsed:.2f} 秒")
    
    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


if __name__ == "__main__":
    print("=" * 50)
    print("河套智慧农牧业库 - 工具函数演示")
    print("=" * 50)
    
    # 安全除法
    print(f"\n安全除法: 10/0 = {safe_divide(10, 0)}")
    
    # 插值
    print(f"线性插值: f(1.5) = {linear_interpolate(1.5, 1, 10, 2, 20)}")
    
    # 模型验证
    obs = np.array([1, 2, 3, 4, 5])
    pred = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
    result = validate_model(obs, pred)
    print(f"\n{result}")
    
    # 计时器
    with Timer("示例计算"):
        _ = sum(range(1000000))
