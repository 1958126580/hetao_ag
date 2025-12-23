# -*- coding: utf-8 -*-
"""
Space模块使用示例
=================

演示hetao_ag.space模块的遥感影像处理、光谱指数计算和物候分类功能。

作者: Hetao College
"""

import numpy as np
from hetao_ag.space import (
    # 光谱指数
    compute_ndvi, compute_savi, compute_lswi, compute_evi, compute_ndwi,
    classify_vegetation_health,
    # 影像处理
    RasterImage, GeoMetadata, CloudMask,
    # 物候分类
    PhenologyClassifier, PhenologyFeatures, temporal_smoothing
)


def example_spectral_indices():
    """
    示例1: 光谱指数计算
    ===================
    
    计算常用的遥感植被指数：NDVI、SAVI、LSWI、EVI等。
    """
    print("\n" + "=" * 60)
    print("示例1: 光谱指数计算")
    print("=" * 60)
    
    # 1.1 模拟多光谱数据
    # 假设4个像素的Sentinel-2数据
    blue = np.array([[800, 850], [780, 900]], dtype=np.uint16)
    green = np.array([[1000, 1100], [980, 1150]], dtype=np.uint16)
    red = np.array([[1200, 1300], [1150, 1400]], dtype=np.uint16)
    nir = np.array([[4500, 4800], [4300, 5000]], dtype=np.uint16)
    swir = np.array([[2000, 2200], [1900, 2400]], dtype=np.uint16)
    
    print("\n【模拟波段数据】")
    print(f"  Blue: {blue.flatten()}")
    print(f"  Green: {green.flatten()}")
    print(f"  Red: {red.flatten()}")
    print(f"  NIR: {nir.flatten()}")
    print(f"  SWIR: {swir.flatten()}")
    
    # 1.2 NDVI (归一化差值植被指数)
    ndvi = compute_ndvi(red, nir)
    
    print(f"\n【NDVI】")
    print(f"  公式: NDVI = (NIR - Red) / (NIR + Red)")
    print(f"  值范围: -1 到 1")
    print(f"  结果:\n{ndvi}")
    print(f"  平均NDVI: {ndvi.mean():.3f}")
    
    # 1.3 SAVI (土壤调节植被指数)
    savi = compute_savi(red, nir, L=0.5)
    
    print(f"\n【SAVI (L=0.5)】")
    print(f"  公式: SAVI = ((NIR - Red) / (NIR + Red + L)) × (1 + L)")
    print(f"  L=0.5适用于中等植被覆盖")
    print(f"  结果:\n{savi}")
    
    # 1.4 LSWI (地表水指数)
    lswi = compute_lswi(nir, swir)
    
    print(f"\n【LSWI】")
    print(f"  公式: LSWI = (NIR - SWIR) / (NIR + SWIR)")
    print(f"  用于监测植被水分含量")
    print(f"  结果:\n{lswi}")
    
    # 1.5 EVI (增强型植被指数)
    evi = compute_evi(blue, red, nir)
    
    print(f"\n【EVI】")
    print(f"  优点: 对大气和土壤背景校正")
    print(f"  结果:\n{evi}")
    
    # 1.6 NDWI (归一化差值水体指数)
    ndwi = compute_ndwi(green, nir)
    
    print(f"\n【NDWI】")
    print(f"  用于水体提取")
    print(f"  结果:\n{ndwi}")
    
    # 1.7 植被健康分级
    print(f"\n【植被健康分级】")
    ndvi_values = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.85]
    for ndvi_val in ndvi_values:
        health = classify_vegetation_health(ndvi_val)
        print(f"  NDVI={ndvi_val:>5.2f}: {health}")


def example_raster_image():
    """
    示例2: 栅格影像处理
    ===================
    
    加载和处理遥感影像数据。
    """
    print("\n" + "=" * 60)
    print("示例2: 栅格影像处理")
    print("=" * 60)
    
    # 2.1 创建模拟影像
    np.random.seed(42)
    
    # 模拟4波段影像 (Blue, Green, Red, NIR)
    height, width = 100, 100
    data = np.zeros((4, height, width), dtype=np.uint16)
    
    # 创建植被斑块图案
    for i in range(height):
        for j in range(width):
            # 模拟不同地物
            if (i - 50)**2 + (j - 50)**2 < 900:  # 中心植被区
                data[0, i, j] = 800 + np.random.randint(0, 100)   # Blue
                data[1, i, j] = 1000 + np.random.randint(0, 100)  # Green
                data[2, i, j] = 1100 + np.random.randint(0, 100)  # Red
                data[3, i, j] = 4500 + np.random.randint(0, 500)  # NIR
            else:  # 裸土区
                data[0, i, j] = 1500 + np.random.randint(0, 100)
                data[1, i, j] = 1600 + np.random.randint(0, 100)
                data[2, i, j] = 1700 + np.random.randint(0, 100)
                data[3, i, j] = 2000 + np.random.randint(0, 200)
    
    # 创建RasterImage对象
    band_names = {"blue": 0, "green": 1, "red": 2, "nir": 3}
    metadata = GeoMetadata(crs="EPSG:32649", resolution=10.0)
    
    img = RasterImage(data, band_names, metadata)
    
    print(f"\n【影像信息】")
    print(f"  形状: {img.shape} (波段, 高度, 宽度)")
    print(f"  波段数: {img.n_bands}")
    print(f"  坐标系: {img.metadata.crs}")
    print(f"  分辨率: {img.metadata.resolution}m")
    
    # 2.2 获取波段
    red = img.get_band("red")
    nir = img.get_band("nir")
    
    print(f"\n【波段统计】")
    print(f"  Red: min={red.min()}, max={red.max()}, mean={red.mean():.0f}")
    print(f"  NIR: min={nir.min()}, max={nir.max()}, mean={nir.mean():.0f}")
    
    # 2.3 计算NDVI
    ndvi = compute_ndvi(red, nir)
    
    print(f"\n【NDVI统计】")
    print(f"  最小值: {ndvi.min():.3f}")
    print(f"  最大值: {ndvi.max():.3f}")
    print(f"  平均值: {ndvi.mean():.3f}")
    print(f"  高植被区(NDVI>0.5)像素数: {(ndvi > 0.5).sum()}")
    
    # 2.4 影像裁剪
    subset = img.subset(slice(40, 60), slice(40, 60))
    print(f"\n【影像裁剪】")
    print(f"  原始形状: {img.shape}")
    print(f"  裁剪后形状: {subset.shape}")
    
    # 2.5 云掩膜
    print(f"\n【云掩膜】")
    cloud_mask = CloudMask(threshold=0.3)
    
    # 模拟QA波段
    qa_band = np.random.randint(0, 100, (height, width))
    valid_mask = cloud_mask.from_qa_band(qa_band)
    
    print(f"  有效像素比例: {valid_mask.sum() / valid_mask.size * 100:.1f}%")


def example_phenology_classification():
    """
    示例3: 物候分类
    ===============
    
    基于NDVI时序数据进行作物分类和物候提取。
    """
    print("\n" + "=" * 60)
    print("示例3: 物候分类")
    print("=" * 60)
    
    # 3.1 创建模拟NDVI时序
    np.random.seed(42)
    
    n_times = 12  # 12期(约每月一期)
    height, width = 50, 50
    
    # 时间轴(月份)
    months = list(range(1, 13))
    print(f"\n【模拟NDVI时序数据】")
    print(f"  时间序列: {n_times}期 (月度)")
    print(f"  影像尺寸: {height}×{width}像素")
    
    # 创建不同作物的时序特征
    ndvi_series = np.zeros((n_times, height, width))
    
    for i in range(height):
        for j in range(width):
            # 分区模拟不同作物
            if i < height // 3:
                # 冬小麦: 春季峰值(4月)
                peak_month = 4
                base = 0.3
            elif i < 2 * height // 3:
                # 夏玉米: 夏季峰值(8月)
                peak_month = 8
                base = 0.25
            else:
                # 棉花: 晚夏峰值(9月)
                peak_month = 9
                base = 0.2
            
            # 生成季节性曲线
            for t in range(n_times):
                month = t + 1
                # 高斯型生长曲线
                ndvi = base + 0.5 * np.exp(-((month - peak_month) ** 2) / 8)
                ndvi += np.random.randn() * 0.03
                ndvi_series[t, i, j] = np.clip(ndvi, 0, 1)
    
    # 3.2 时序平滑
    print(f"\n【时序平滑】")
    smoothed = temporal_smoothing(ndvi_series, window=3)
    print(f"  原始噪声标准差: {ndvi_series.std():.3f}")
    print(f"  平滑后标准差: {smoothed.std():.3f}")
    
    # 3.3 物候分类
    classifier = PhenologyClassifier(ndvi_series)
    
    print(f"\n【物候特征提取】")
    
    # 获取不同位置的物候特征
    positions = [(10, 25), (25, 25), (40, 25)]
    crop_names = ["冬小麦", "夏玉米", "棉花"]
    
    for (row, col), crop_name in zip(positions, crop_names):
        features = classifier.extract_features(row, col)
        print(f"\n  {crop_name} (像素{row},{col}):")
        print(f"    峰值NDVI: {features.peak_value:.3f}")
        print(f"    峰值时间: 第{features.peak_time + 1}期 (~{features.peak_time + 1}月)")
        print(f"    生长季: 第{features.start_of_season + 1}期 ~ 第{features.end_of_season}期")
        print(f"    振幅: {features.amplitude:.3f}")
    
    # 3.4 作物分类
    print(f"\n【作物分类】")
    crop_map = classifier.classify_crops(n_classes=3)
    
    print(f"  分类结果统计:")
    for cls in range(4):
        count = (crop_map == cls).sum()
        pct = count / crop_map.size * 100
        if count > 0:
            label = ["非作物", "早峰作物", "晚峰作物", "中峰作物"][cls]
            print(f"    类别{cls} ({label}): {count}像素 ({pct:.1f}%)")
    
    # 3.5 物候参数图
    print(f"\n【物候参数图】")
    pheno_maps = classifier.get_phenology_map()
    
    for name, data in pheno_maps.items():
        print(f"  {name}: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}")


def example_application():
    """
    示例4: 实际应用场景
    ===================
    
    综合应用：农田长势监测。
    """
    print("\n" + "=" * 60)
    print("示例4: 农田长势监测应用")
    print("=" * 60)
    
    # 4.1 模拟采集的卫星数据
    np.random.seed(123)
    
    # 模拟100×100像素农田
    h, w = 100, 100
    
    # 生成带有空间变异的NDVI
    x = np.linspace(0, 4*np.pi, w)
    y = np.linspace(0, 4*np.pi, h)
    xx, yy = np.meshgrid(x, y)
    
    base_ndvi = 0.5 + 0.2 * np.sin(xx) + 0.2 * np.sin(yy)
    noise = np.random.randn(h, w) * 0.05
    ndvi = np.clip(base_ndvi + noise, 0, 1)
    
    print(f"\n【农田NDVI分析】")
    print(f"  农田面积: {h*w}像素 (假设10m分辨率 = 100公顷)")
    print(f"  NDVI范围: {ndvi.min():.3f} ~ {ndvi.max():.3f}")
    print(f"  平均NDVI: {ndvi.mean():.3f}")
    
    # 4.2 长势分级
    print(f"\n【长势分级统计】")
    grades = {
        "优良 (NDVI>0.7)": (ndvi > 0.7).sum(),
        "良好 (0.5-0.7)": ((ndvi >= 0.5) & (ndvi <= 0.7)).sum(),
        "一般 (0.3-0.5)": ((ndvi >= 0.3) & (ndvi < 0.5)).sum(),
        "较差 (<0.3)": (ndvi < 0.3).sum(),
    }
    
    for grade, count in grades.items():
        pct = count / ndvi.size * 100
        print(f"  {grade}: {count}像素 ({pct:.1f}%)")
    
    # 4.3 异常区域识别
    print(f"\n【异常区域识别】")
    threshold = ndvi.mean() - ndvi.std()
    abnormal_pixels = ndvi < threshold
    abnormal_count = abnormal_pixels.sum()
    
    print(f"  低于阈值({threshold:.3f})的像素: {abnormal_count} ({abnormal_count/ndvi.size*100:.1f}%)")
    
    # 4.4 分区统计
    print(f"\n【分区统计】")
    n_zones = 4
    zone_size = h // 2
    
    for zi in range(2):
        for zj in range(2):
            zone_ndvi = ndvi[zi*zone_size:(zi+1)*zone_size, 
                            zj*zone_size:(zj+1)*zone_size]
            zone_name = f"Zone-{zi*2+zj+1}"
            print(f"  {zone_name}: 平均NDVI={zone_ndvi.mean():.3f}, "
                  f"标准差={zone_ndvi.std():.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.space 模块使用示例")
    print("=" * 60)
    
    example_spectral_indices()
    example_raster_image()
    example_phenology_classification()
    example_application()
    
    print("\n" + "=" * 60)
    print("Space模块示例完成")
    print("=" * 60)
