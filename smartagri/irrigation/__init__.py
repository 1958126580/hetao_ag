"""
Smart Irrigation Module

Precision irrigation management and optimization:
- Soil moisture monitoring
- Irrigation scheduling
- Water balance calculations
- System efficiency analysis

Example:
    >>> from smartagri.irrigation import IrrigationScheduler, WaterBalance
    >>> scheduler = IrrigationScheduler()
    >>> schedule = scheduler.create_schedule(field_id, crop="corn")
"""

from .scheduling import (
    IrrigationScheduler,
    IrrigationEvent,
    IrrigationZone,
    ScheduleOptimizer,
)
from .water_balance import (
    WaterBalance,
    SoilMoistureTracker,
    WaterBudget,
    DeficitCalculator,
)
from .systems import (
    IrrigationSystem,
    DripSystem,
    SprinklerSystem,
    PivotSystem,
    SystemEfficiency,
)

__all__ = [
    # Scheduling
    "IrrigationScheduler",
    "IrrigationEvent",
    "IrrigationZone",
    "ScheduleOptimizer",
    # Water balance
    "WaterBalance",
    "SoilMoistureTracker",
    "WaterBudget",
    "DeficitCalculator",
    # Systems
    "IrrigationSystem",
    "DripSystem",
    "SprinklerSystem",
    "PivotSystem",
    "SystemEfficiency",
]
