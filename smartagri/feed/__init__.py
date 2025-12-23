"""
Feed and Nutrition Module

Feed formulation and nutrition optimization:
- Nutrient requirement calculations
- Ration formulation and optimization
- Feed inventory management
- Feed cost analysis

Example:
    >>> from smartagri.feed import RationOptimizer, NutrientCalculator
    >>> optimizer = RationOptimizer(species="cattle")
    >>> ration = optimizer.optimize(target_weight=500, production_level="high")
"""

from .nutrition import (
    NutrientCalculator,
    NutrientRequirements,
    FeedAnalysis,
    NutrientBalance,
)
from .formulation import (
    RationOptimizer,
    FeedIngredient,
    Ration,
    FormulationConstraint,
)
from .inventory import (
    FeedInventory,
    FeedPurchase,
    FeedUsage,
    CostAnalyzer,
)

__all__ = [
    # Nutrition
    "NutrientCalculator",
    "NutrientRequirements",
    "FeedAnalysis",
    "NutrientBalance",
    # Formulation
    "RationOptimizer",
    "FeedIngredient",
    "Ration",
    "FormulationConstraint",
    # Inventory
    "FeedInventory",
    "FeedPurchase",
    "FeedUsage",
    "CostAnalyzer",
]
