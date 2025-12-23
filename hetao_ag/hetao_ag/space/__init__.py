# -*- coding: utf-8 -*-
"""
hetao_ag.space - 遥感模块

遥感影像处理、光谱指数计算和物候分类。

作者: Hetao College
版本: 1.0.0
"""

from .indices import (
    compute_ndvi,
    compute_savi,
    compute_lswi,
    compute_evi,
    compute_ndwi,
    classify_vegetation_health,
)

from .imagery import (
    RasterImage,
    GeoMetadata,
    CloudMask,
)

from .classification import (
    PhenologyClassifier,
    PhenologyFeatures,
    temporal_smoothing,
)

__all__ = [
    # indices
    "compute_ndvi",
    "compute_savi",
    "compute_lswi",
    "compute_evi",
    "compute_ndwi",
    "classify_vegetation_health",
    # imagery
    "RasterImage",
    "GeoMetadata",
    "CloudMask",
    # classification
    "PhenologyClassifier",
    "PhenologyFeatures",
    "temporal_smoothing",
]
