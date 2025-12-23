# -*- coding: utf-8 -*-
"""
hetao_ag.soil.sensors - 土壤传感器校准

IoT土壤传感器的校准和数据处理工具。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from enum import Enum


class CalibrationMethod(Enum):
    """校准方法枚举"""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    POWER = "power"


@dataclass
class CalibrationResult:
    """校准结果
    
    属性:
        method: 校准方法
        coefficients: 系数
        r_squared: R²值
        rmse: 均方根误差
    """
    method: CalibrationMethod
    coefficients: Tuple
    r_squared: float
    rmse: float
    
    def apply(self, raw_value: float) -> float:
        """应用校准"""
        if self.method == CalibrationMethod.LINEAR:
            slope, intercept = self.coefficients
            return slope * raw_value + intercept
        elif self.method == CalibrationMethod.POLYNOMIAL:
            return np.polyval(self.coefficients, raw_value)
        elif self.method == CalibrationMethod.POWER:
            a, b = self.coefficients
            return a * (raw_value ** b)
        return raw_value
    
    def __str__(self) -> str:
        return f"{self.method.value}校准: R²={self.r_squared:.4f}, RMSE={self.rmse:.4f}"


class SensorCalibrator:
    """传感器校准器
    
    支持多种校准方法，用于校准低成本土壤传感器。
    
    示例:
        >>> calibrator = SensorCalibrator()
        >>> raw = [300, 450, 600, 750]
        >>> true = [0.10, 0.20, 0.30, 0.40]
        >>> result = calibrator.linear_calibration(raw, true)
        >>> corrected = result.apply(500)
    """
    
    def __init__(self):
        self.calibrations: dict = {}
    
    def linear_calibration(
        self,
        raw_readings: np.ndarray,
        ground_truth: np.ndarray
    ) -> CalibrationResult:
        """线性校准
        
        参数:
            raw_readings: 原始传感器读数
            ground_truth: 真实值
            
        返回:
            CalibrationResult对象
        """
        raw = np.array(raw_readings, dtype=float)
        true = np.array(ground_truth, dtype=float)
        
        # 最小二乘线性拟合
        A = np.vstack([raw, np.ones(len(raw))]).T
        slope, intercept = np.linalg.lstsq(A, true, rcond=None)[0]
        
        # 计算拟合优度
        predicted = slope * raw + intercept
        r_squared = self._r_squared(true, predicted)
        rmse = self._rmse(true, predicted)
        
        return CalibrationResult(
            method=CalibrationMethod.LINEAR,
            coefficients=(slope, intercept),
            r_squared=r_squared,
            rmse=rmse
        )
    
    def polynomial_calibration(
        self,
        raw_readings: np.ndarray,
        ground_truth: np.ndarray,
        degree: int = 2
    ) -> CalibrationResult:
        """多项式校准
        
        参数:
            raw_readings: 原始读数
            ground_truth: 真实值
            degree: 多项式次数
            
        返回:
            CalibrationResult对象
        """
        raw = np.array(raw_readings, dtype=float)
        true = np.array(ground_truth, dtype=float)
        
        coeffs = np.polyfit(raw, true, degree)
        predicted = np.polyval(coeffs, raw)
        
        return CalibrationResult(
            method=CalibrationMethod.POLYNOMIAL,
            coefficients=tuple(coeffs),
            r_squared=self._r_squared(true, predicted),
            rmse=self._rmse(true, predicted)
        )
    
    def auto_calibrate(
        self,
        raw_readings: np.ndarray,
        ground_truth: np.ndarray
    ) -> CalibrationResult:
        """自动选择最佳校准方法
        
        参数:
            raw_readings: 原始读数
            ground_truth: 真实值
            
        返回:
            最佳校准结果
        """
        results = [
            self.linear_calibration(raw_readings, ground_truth),
            self.polynomial_calibration(raw_readings, ground_truth, degree=2),
        ]
        
        # 选择R²最高的
        best = max(results, key=lambda r: r.r_squared)
        return best
    
    @staticmethod
    def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算R²"""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
    
    @staticmethod
    def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
        """计算RMSE"""
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))


class MoistureSensor:
    """土壤水分传感器接口
    
    示例:
        >>> sensor = MoistureSensor(sensor_id="SM01", calibration=result)
        >>> moisture = sensor.read_calibrated(raw_value=450)
    """
    
    def __init__(
        self,
        sensor_id: str,
        calibration: Optional[CalibrationResult] = None,
        min_raw: float = 0,
        max_raw: float = 1023
    ):
        self.sensor_id = sensor_id
        self.calibration = calibration
        self.min_raw = min_raw
        self.max_raw = max_raw
        self.readings: List[Tuple[float, float]] = []  # (timestamp, value)
    
    def read_calibrated(self, raw_value: float) -> float:
        """读取校准后的值
        
        参数:
            raw_value: 原始读数
            
        返回:
            校准后的体积含水量
        """
        if self.calibration:
            return self.calibration.apply(raw_value)
        else:
            # 默认线性映射到0-1范围
            return (raw_value - self.min_raw) / (self.max_raw - self.min_raw)
    
    def add_reading(self, timestamp: float, raw_value: float):
        """添加读数记录"""
        calibrated = self.read_calibrated(raw_value)
        self.readings.append((timestamp, calibrated))
    
    def get_average(self, n_last: int = 10) -> float:
        """获取最近n次读数的平均值"""
        if not self.readings:
            return 0.0
        recent = self.readings[-n_last:]
        return np.mean([r[1] for r in recent])


def capacitive_sensor_formula(raw: float, dry_value: float = 520, wet_value: float = 260) -> float:
    """电容式土壤水分传感器通用公式
    
    适用于常见低成本电容式传感器(如DFRobot SEN0193)。
    
    参数:
        raw: 原始ADC读数
        dry_value: 干燥空气中的读数
        wet_value: 纯水中的读数
        
    返回:
        相对湿度(0-100%)
    """
    if dry_value == wet_value:
        return 50.0
    
    moisture_percent = 100 * (dry_value - raw) / (dry_value - wet_value)
    return max(0, min(100, moisture_percent))


if __name__ == "__main__":
    print("=" * 50)
    print("传感器校准演示")
    print("=" * 50)
    
    # 模拟校准数据
    raw_vals = np.array([300, 400, 500, 600, 700])
    true_vwc = np.array([0.10, 0.18, 0.28, 0.35, 0.42])
    
    calibrator = SensorCalibrator()
    
    # 线性校准
    linear_result = calibrator.linear_calibration(raw_vals, true_vwc)
    print(f"\n{linear_result}")
    
    # 多项式校准
    poly_result = calibrator.polynomial_calibration(raw_vals, true_vwc, degree=2)
    print(f"{poly_result}")
    
    # 自动校准
    best = calibrator.auto_calibrate(raw_vals, true_vwc)
    print(f"\n最佳方法: {best.method.value}")
    
    # 应用校准
    test_raw = 450
    calibrated = best.apply(test_raw)
    print(f"\n原始读数 {test_raw} -> 校准值 {calibrated:.3f}")
