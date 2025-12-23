"""
Soil Analysis Module for SmartAgri

Comprehensive soil analysis and management tools including:
- Soil classification and property estimation
- Nutrient analysis and recommendations
- Water holding capacity calculations
- Soil health assessment
- Spatial soil mapping

Example:
    >>> from smartagri.soil import SoilAnalyzer, NutrientManager
    >>> analyzer = SoilAnalyzer()
    >>> properties = analyzer.estimate_properties(sand=40, silt=35, clay=25)
    >>> nutrients = NutrientManager()
    >>> recommendations = nutrients.fertilizer_recommendation(soil_test, crop="corn")
"""

from smartagri.soil.analysis import (
    SoilAnalyzer,
    SoilClassifier,
    TextureClassifier,
    SoilProperties,
    WaterRetention,
)

from smartagri.soil.nutrients import (
    NutrientManager,
    NutrientAnalysis,
    FertilizerRecommendation,
    calculate_cec,
    estimate_nitrogen_mineralization,
)

from smartagri.soil.health import (
    SoilHealthAssessment,
    OrganicMatterDynamics,
    BiologicalActivity,
    SoilQualityIndex,
)

from smartagri.soil.mapping import (
    SoilMapper,
    SpatialInterpolation,
    VariabilityAnalysis,
    ManagementZones,
)

__all__ = [
    # Analysis
    "SoilAnalyzer",
    "SoilClassifier",
    "TextureClassifier",
    "SoilProperties",
    "WaterRetention",
    # Nutrients
    "NutrientManager",
    "NutrientAnalysis",
    "FertilizerRecommendation",
    "calculate_cec",
    "estimate_nitrogen_mineralization",
    # Health
    "SoilHealthAssessment",
    "OrganicMatterDynamics",
    "BiologicalActivity",
    "SoilQualityIndex",
    # Mapping
    "SoilMapper",
    "SpatialInterpolation",
    "VariabilityAnalysis",
    "ManagementZones",
]
