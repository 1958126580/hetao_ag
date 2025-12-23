# -*- coding: utf-8 -*-
"""
hetao_ag.space.indices - 光谱指数计算

NDVI、SAVI、LSWI等植被指数计算。

作者: Hetao College
版本: 1.0.0
"""

import numpy as np
from typing import Optional


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算NDVI(归一化差值植被指数)
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    参数:
        red: 红光波段
        nir: 近红外波段
        
    返回:
        NDVI数组(-1到1)
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    
    numerator = nir - red
    denominator = nir + red
    
    ndvi = np.where(denominator == 0, 0, numerator / denominator)
    return np.clip(ndvi, -1, 1)


def compute_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    """计算SAVI(土壤调节植被指数)
    
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    
    参数:
        red: 红光波段
        nir: 近红外波段
        L: 土壤调节因子(0-1, 0.5适用于中等植被覆盖)
        
    返回:
        SAVI数组
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    
    numerator = nir - red
    denominator = nir + red + L
    
    savi = np.where(denominator == 0, 0, numerator / denominator * (1 + L))
    return savi


def compute_lswi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """计算LSWI(地表水指数)
    
    LSWI = (NIR - SWIR) / (NIR + SWIR)
    
    参数:
        nir: 近红外波段
        swir: 短波红外波段
        
    返回:
        LSWI数组(-1到1)
    """
    nir = nir.astype(np.float32)
    swir = swir.astype(np.float32)
    
    numerator = nir - swir
    denominator = nir + swir
    
    lswi = np.where(denominator == 0, 0, numerator / denominator)
    return np.clip(lswi, -1, 1)


def compute_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray,
                G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """计算EVI(增强型植被指数)
    
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    
    参数:
        blue, red, nir: 波段数据
        G, C1, C2, L: EVI参数
        
    返回:
        EVI数组
    """
    blue = blue.astype(np.float32)
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L
    
    evi = np.where(denominator == 0, 0, numerator / denominator)
    return np.clip(evi, -1, 1)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """计算NDWI(归一化差值水体指数)
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    参数:
        green: 绿光波段
        nir: 近红外波段
        
    返回:
        NDWI数组
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    
    numerator = green - nir
    denominator = green + nir
    
    ndwi = np.where(denominator == 0, 0, numerator / denominator)
    return np.clip(ndwi, -1, 1)


def classify_vegetation_health(ndvi: float) -> str:
    """根据NDVI分类植被健康状态
    
    参数:
        ndvi: NDVI值
        
    返回:
        健康状态描述
    """
    if ndvi < 0:
        return "水体/裸土"
    elif ndvi < 0.2:
        return "稀疏或无植被"
    elif ndvi < 0.4:
        return "轻度植被"
    elif ndvi < 0.6:
        return "中等植被"
    elif ndvi < 0.8:
        return "茂密植被"
    else:
        return "非常茂密植被"


if __name__ == "__main__":
    print("=" * 50)
    print("光谱指数计算演示")
    print("=" * 50)
    
    # 模拟波段数据
    red = np.array([[120, 130], [110, 90]], dtype=np.uint16)
    nir = np.array([[200, 210], [180, 160]], dtype=np.uint16)
    
    ndvi = compute_ndvi(red, nir)
    print(f"\nNDVI:\n{ndvi}")
    
    savi = compute_savi(red, nir, L=0.5)
    print(f"\nSAVI (L=0.5):\n{savi}")
    
    print(f"\n植被状态: {classify_vegetation_health(0.65)}")
