# -*- coding: utf-8 -*-
"""
hetao_ag.space.imagery - 影像处理

遥感影像加载和处理。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
from pathlib import Path


@dataclass
class GeoMetadata:
    """地理空间元数据"""
    crs: str = "EPSG:4326"
    transform: Optional[Tuple] = None
    bounds: Optional[Tuple] = None
    resolution: Optional[float] = None


class RasterImage:
    """栅格影像类
    
    示例:
        >>> img = RasterImage.from_file("sentinel2.tif")
        >>> red = img.get_band("red")
        >>> nir = img.get_band("nir")
    """
    
    # Sentinel-2波段映射
    SENTINEL2_BANDS = {
        "blue": 1, "green": 2, "red": 3, "nir": 7, "swir1": 11, "swir2": 12
    }
    
    def __init__(
        self,
        data: np.ndarray,
        band_names: Optional[Dict[str, int]] = None,
        metadata: Optional[GeoMetadata] = None
    ):
        """初始化栅格影像
        
        参数:
            data: 影像数据 (bands, height, width)
            band_names: 波段名称映射
            metadata: 地理空间元数据
        """
        self.data = data
        self.band_names = band_names or {}
        self.metadata = metadata or GeoMetadata()
    
    @classmethod
    def from_file(cls, path: str) -> 'RasterImage':
        """从文件加载影像"""
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()
                metadata = GeoMetadata(
                    crs=str(src.crs),
                    transform=src.transform,
                    bounds=src.bounds,
                    resolution=src.res[0]
                )
            return cls(data, metadata=metadata)
        except ImportError:
            # 返回模拟数据
            print("警告: rasterio未安装,使用模拟数据")
            data = np.random.randint(0, 255, (4, 100, 100), dtype=np.uint8)
            return cls(data)
    
    def get_band(self, name_or_index) -> np.ndarray:
        """获取指定波段
        
        参数:
            name_or_index: 波段名称或索引
            
        返回:
            波段数据
        """
        if isinstance(name_or_index, str):
            idx = self.band_names.get(name_or_index)
            if idx is None:
                idx = self.SENTINEL2_BANDS.get(name_or_index.lower(), 0)
            if idx >= 1:
                idx -= 1  # 转为0索引
        else:
            idx = name_or_index
        
        if idx < self.data.shape[0]:
            return self.data[idx]
        return self.data[0]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """影像形状 (bands, height, width)"""
        return self.data.shape
    
    @property
    def n_bands(self) -> int:
        """波段数量"""
        return self.data.shape[0]
    
    def subset(self, row_slice: slice, col_slice: slice) -> 'RasterImage':
        """裁剪影像"""
        new_data = self.data[:, row_slice, col_slice]
        return RasterImage(new_data, self.band_names, self.metadata)
    
    def apply_mask(self, mask: np.ndarray) -> 'RasterImage':
        """应用掩膜"""
        masked_data = self.data.copy()
        for i in range(self.n_bands):
            masked_data[i] = np.where(mask, self.data[i], np.nan)
        return RasterImage(masked_data, self.band_names, self.metadata)


class CloudMask:
    """云掩膜生成器"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def from_qa_band(self, qa_band: np.ndarray) -> np.ndarray:
        """从QA波段生成云掩膜"""
        # 简化实现:假设高值为云
        cloud_mask = qa_band > (np.max(qa_band) * self.threshold)
        return ~cloud_mask  # 返回有效像素掩膜
    
    def simple_detection(self, blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """简单云检测"""
        # 云通常在蓝光和近红外都很亮
        bright = (blue > np.percentile(blue, 90)) & (nir > np.percentile(nir, 90))
        return ~bright


if __name__ == "__main__":
    print("=" * 50)
    print("影像处理演示")
    print("=" * 50)
    
    # 创建模拟影像
    data = np.random.randint(0, 10000, (4, 100, 100), dtype=np.uint16)
    img = RasterImage(data, {"blue": 0, "green": 1, "red": 2, "nir": 3})
    
    print(f"\n影像形状: {img.shape}")
    print(f"波段数: {img.n_bands}")
    
    red = img.get_band("red")
    nir = img.get_band("nir")
    print(f"红光波段均值: {red.mean():.1f}")
    print(f"近红外波段均值: {nir.mean():.1f}")
