"""
Livestock Management Module for SmartAgri

Comprehensive livestock management tools including:
- Animal tracking and identification
- Growth modeling and weight prediction
- Reproduction management
- Herd analytics
- Performance monitoring

Example:
    >>> from smartagri.livestock import AnimalTracker, GrowthModel, HerdManager
    >>> tracker = AnimalTracker()
    >>> animal = tracker.register_animal("cattle", birth_date=date(2023, 3, 15))
    >>> growth = GrowthModel(species="cattle")
    >>> predicted_weight = growth.predict_weight(age_days=365)
"""

from .tracking import (
    AnimalTracker,
    Animal,
    AnimalGroup,
    LocationTracker,
    MovementHistory,
    LocationRecord,
    MovementRecord,
    Sex,
    AnimalStatus,
)

from .growth import (
    GrowthModel,
    GrowthCurve,
    GrowthCurveType,
    GrowthParameters,
    BodyConditionScore,
    WeightMonitor,
    FeedConversionAnalyzer,
    GROWTH_PARAMETERS,
)

from .reproduction import (
    ReproductionManager,
    EstrusDetector,
    BreedingManager,
    PregnancyManager,
    CalvingManager,
    GeneticEvaluator,
    HeatRecord,
    BreedingRecord,
    PregnancyRecord,
    BirthRecord,
    ReproductiveStatus,
    BreedingMethod,
)

from .herd import (
    HerdManager,
    HerdInventory,
    HerdComposition,
    PerformanceBenchmark,
    CullingDecision,
    EconomicAnalyzer,
    AnimalRecord,
    HerdStats,
    CullingCandidate,
    ProductionSystem,
    CullingReason,
)

__all__ = [
    # Tracking
    "AnimalTracker",
    "Animal",
    "AnimalGroup",
    "LocationTracker",
    "MovementHistory",
    "LocationRecord",
    "MovementRecord",
    "Sex",
    "AnimalStatus",
    # Growth
    "GrowthModel",
    "GrowthCurve",
    "GrowthCurveType",
    "GrowthParameters",
    "BodyConditionScore",
    "WeightMonitor",
    "FeedConversionAnalyzer",
    "GROWTH_PARAMETERS",
    # Reproduction
    "ReproductionManager",
    "EstrusDetector",
    "BreedingManager",
    "PregnancyManager",
    "CalvingManager",
    "GeneticEvaluator",
    "HeatRecord",
    "BreedingRecord",
    "PregnancyRecord",
    "BirthRecord",
    "ReproductiveStatus",
    "BreedingMethod",
    # Herd management
    "HerdManager",
    "HerdInventory",
    "HerdComposition",
    "PerformanceBenchmark",
    "CullingDecision",
    "EconomicAnalyzer",
    "AnimalRecord",
    "HerdStats",
    "CullingCandidate",
    "ProductionSystem",
    "CullingReason",
]
