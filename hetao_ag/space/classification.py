# -*- coding: utf-8 -*-
"""
hetao_ag.space.classification - 物候分类

遥感时序物候分类和作物识别。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class PhenologyFeatures:
    """物候特征"""
    peak_value: float
    peak_time: int
    start_of_season: int
    end_of_season: int
    amplitude: float


class PhenologyClassifier:
    """物候分类器
    
    基于NDVI时序数据进行作物分类。
    
    示例:
        >>> classifier = PhenologyClassifier(ndvi_series, dates)
        >>> crop_map = classifier.classify_crops()
    """
    
    def __init__(self, time_series: np.ndarray, dates: Optional[List] = None):
        """初始化分类器
        
        参数:
            time_series: NDVI时序数据 (time, height, width)
            dates: 日期列表
        """
        self.data = time_series
        self.dates = dates or list(range(time_series.shape[0]))
        
        self.peak = np.max(time_series, axis=0)
        self.peak_time = np.argmax(time_series, axis=0)
    
    def extract_features(self, row: int, col: int) -> PhenologyFeatures:
        """提取像素物候特征
        
        参数:
            row, col: 像素坐标
            
        返回:
            物候特征
        """
        series = self.data[:, row, col]
        peak_val = np.max(series)
        peak_idx = np.argmax(series)
        
        # 简化的季节起止检测
        threshold = peak_val * 0.3
        above_threshold = series > threshold
        start = np.argmax(above_threshold) if np.any(above_threshold) else 0
        end = len(series) - np.argmax(above_threshold[::-1]) if np.any(above_threshold) else len(series)
        
        return PhenologyFeatures(
            peak_value=float(peak_val),
            peak_time=int(peak_idx),
            start_of_season=int(start),
            end_of_season=int(end),
            amplitude=float(peak_val - np.min(series))
        )
    
    def classify_crops(self, n_classes: int = 3) -> np.ndarray:
        """作物分类
        
        基于峰值时间和峰值大小进行简单分类。
        
        参数:
            n_classes: 类别数
            
        返回:
            分类结果
        """
        classes = np.zeros_like(self.peak_time)
        n_times = self.data.shape[0]
        early_threshold = n_times // 3
        late_threshold = 2 * n_times // 3
        
        # 规则分类
        # 类别1: 早峰(如冬小麦)
        classes[(self.peak_time < early_threshold) & (self.peak > 0.5)] = 1
        
        # 类别2: 晚峰(如玉米)
        classes[(self.peak_time >= late_threshold) & (self.peak > 0.5)] = 2
        
        # 类别3: 中峰
        classes[(self.peak_time >= early_threshold) & (self.peak_time < late_threshold) & (self.peak > 0.5)] = 3
        
        # 低NDVI区域为非作物(0)
        return classes
    
    def get_phenology_map(self) -> Dict[str, np.ndarray]:
        """生成物候参数图"""
        return {
            "peak_ndvi": self.peak,
            "peak_time": self.peak_time,
            "season_length": self._compute_season_length(),
        }
    
    def _compute_season_length(self, threshold_ratio: float = 0.3) -> np.ndarray:
        """计算生长季长度"""
        threshold = self.peak * threshold_ratio
        
        season_length = np.zeros(self.peak.shape, dtype=int)
        
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[2]):
                series = self.data[:, i, j]
                thresh = threshold[i, j]
                above = series > thresh
                if np.any(above):
                    start = np.argmax(above)
                    end = len(series) - np.argmax(above[::-1])
                    season_length[i, j] = end - start
        
        return season_length


def temporal_smoothing(time_series: np.ndarray, window: int = 3) -> np.ndarray:
    """时序平滑
    
    参数:
        time_series: 输入时序 (time, ...)
        window: 平滑窗口
        
    返回:
        平滑后的时序
    """
    if window <= 1:
        return time_series
    
    kernel = np.ones(window) / window
    shape = time_series.shape
    
    if len(shape) == 1:
        return np.convolve(time_series, kernel, mode='same')
    
    result = np.zeros_like(time_series, dtype=float)
    for i in range(shape[1]):
        for j in range(shape[2]):
            result[:, i, j] = np.convolve(time_series[:, i, j], kernel, mode='same')
    
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("物候分类演示")
    print("=" * 50)
    
    # 模拟NDVI时序数据
    np.random.seed(42)
    times, h, w = 12, 50, 50
    
    # 创建模拟的季节性NDVI
    t = np.linspace(0, 2*np.pi, times)
    base_curve = 0.3 + 0.4 * np.sin(t)
    
    ndvi_series = np.zeros((times, h, w))
    for i in range(h):
        for j in range(w):
            shift = np.random.randint(-2, 3)
            noise = np.random.randn(times) * 0.05
            ndvi_series[:, i, j] = np.roll(base_curve, shift) + noise
    
    ndvi_series = np.clip(ndvi_series, 0, 1)
    
    classifier = PhenologyClassifier(ndvi_series)
    
    # 分类
    crop_map = classifier.classify_crops()
    
    print(f"\n分类结果统计:")
    for c in range(4):
        count = np.sum(crop_map == c)
        print(f"  类别{c}: {count}像素 ({count/crop_map.size*100:.1f}%)")
    
    # 物候特征
    features = classifier.extract_features(25, 25)
    print(f"\n像素(25,25)物候特征:")
    print(f"  峰值NDVI: {features.peak_value:.3f}")
    print(f"  峰值时间: 第{features.peak_time}期")
