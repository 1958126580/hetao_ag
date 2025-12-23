"""
Water Balance Module

Soil-water-plant balance calculations:
- Soil moisture tracking
- Water budget accounting
- Deficit calculations
- Root zone management

Example:
    >>> balance = WaterBalance(soil_type="loam", root_depth=0.6)
    >>> balance.add_irrigation(25.0)
    >>> balance.add_et(5.0)
    >>> status = balance.get_status()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SoilWaterParameters:
    """
    Soil water holding parameters.

    Attributes:
        field_capacity: Field capacity (mm/m)
        wilting_point: Permanent wilting point (mm/m)
        saturation: Saturation point (mm/m)
        infiltration_rate: Infiltration rate (mm/hr)
    """
    field_capacity: float
    wilting_point: float
    saturation: float
    infiltration_rate: float


# Soil water parameters by texture
SOIL_WATER_PARAMS = {
    "sand": SoilWaterParameters(
        field_capacity=100, wilting_point=40,
        saturation=380, infiltration_rate=200
    ),
    "loamy_sand": SoilWaterParameters(
        field_capacity=140, wilting_point=60,
        saturation=400, infiltration_rate=100
    ),
    "sandy_loam": SoilWaterParameters(
        field_capacity=180, wilting_point=80,
        saturation=430, infiltration_rate=50
    ),
    "loam": SoilWaterParameters(
        field_capacity=250, wilting_point=110,
        saturation=450, infiltration_rate=25
    ),
    "silt_loam": SoilWaterParameters(
        field_capacity=300, wilting_point=130,
        saturation=470, infiltration_rate=15
    ),
    "clay_loam": SoilWaterParameters(
        field_capacity=340, wilting_point=160,
        saturation=490, infiltration_rate=8
    ),
    "clay": SoilWaterParameters(
        field_capacity=380, wilting_point=200,
        saturation=510, infiltration_rate=3
    ),
}


@dataclass
class WaterBudget:
    """
    Water budget record.

    Attributes:
        date: Budget date
        initial_storage: Initial soil water (mm)
        irrigation: Irrigation applied (mm)
        rainfall: Rainfall received (mm)
        et: Evapotranspiration (mm)
        drainage: Deep percolation (mm)
        runoff: Surface runoff (mm)
        final_storage: Final soil water (mm)
    """
    date: date
    initial_storage: float
    irrigation: float = 0.0
    rainfall: float = 0.0
    et: float = 0.0
    drainage: float = 0.0
    runoff: float = 0.0
    final_storage: float = 0.0


class SoilMoistureTracker:
    """
    Soil moisture monitoring and tracking.

    Tracks soil moisture at multiple depths
    and provides status assessments.

    Example:
        >>> tracker = SoilMoistureTracker(depths=[0.1, 0.3, 0.6])
        >>> tracker.record_reading(datetime.now(), [25, 28, 32])
        >>> status = tracker.get_status()
    """

    def __init__(
        self,
        depths: List[float] = None,
        soil_type: str = "loam",
    ):
        """
        Initialize moisture tracker.

        Args:
            depths: Sensor depths (m)
            soil_type: Soil texture class
        """
        self.depths = depths or [0.1, 0.3, 0.6]
        self.soil_type = soil_type.lower().replace(" ", "_")
        self.params = SOIL_WATER_PARAMS.get(
            self.soil_type,
            SOIL_WATER_PARAMS["loam"]
        )

        self._readings: List[Dict[str, Any]] = []

    def record_reading(
        self,
        timestamp: datetime,
        values: List[float],
        sensor_type: str = "tdr",
    ) -> None:
        """
        Record moisture reading.

        Args:
            timestamp: Reading timestamp
            values: Moisture values by depth (% vol)
            sensor_type: Sensor type
        """
        if len(values) != len(self.depths):
            raise ValueError("Values must match number of depths")

        self._readings.append({
            "timestamp": timestamp,
            "values": values,
            "sensor_type": sensor_type,
        })

    def get_current_moisture(self) -> Dict[str, Any]:
        """Get most recent moisture readings."""
        if not self._readings:
            return {"status": "no_data"}

        latest = self._readings[-1]
        return {
            "timestamp": latest["timestamp"],
            "depths": self.depths,
            "values": latest["values"],
            "avg_moisture": np.mean(latest["values"]),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get soil moisture status assessment.

        Returns:
            Dict with status information
        """
        if not self._readings:
            return {"status": "no_data"}

        latest = self._readings[-1]
        values = latest["values"]

        # Calculate weighted average (more weight to shallow layers)
        weights = [1 / (d + 0.1) for d in self.depths]
        weights = [w / sum(weights) for w in weights]
        weighted_avg = sum(v * w for v, w in zip(values, weights))

        # Convert to mm in root zone
        root_depth = max(self.depths)
        fc = self.params.field_capacity * root_depth
        wp = self.params.wilting_point * root_depth

        # Available water depletion
        current_mm = weighted_avg / 100 * (fc - wp) + wp
        depletion = (fc - current_mm) / (fc - wp)

        # Status classification
        if depletion < 0.3:
            status = "adequate"
        elif depletion < 0.5:
            status = "optimal_for_irrigation"
        elif depletion < 0.7:
            status = "stress_approaching"
        else:
            status = "stress"

        return {
            "timestamp": latest["timestamp"],
            "weighted_avg_pct": weighted_avg,
            "current_mm": current_mm,
            "field_capacity_mm": fc,
            "wilting_point_mm": wp,
            "depletion_fraction": depletion,
            "status": status,
            "irrigation_needed": depletion > 0.5,
        }

    def get_trend(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Analyze moisture trend.

        Args:
            hours: Hours to analyze

        Returns:
            Dict with trend analysis
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self._readings if r["timestamp"] >= cutoff]

        if len(recent) < 2:
            return {"status": "insufficient_data"}

        # Calculate average for each reading
        averages = [np.mean(r["values"]) for r in recent]
        times = [(r["timestamp"] - recent[0]["timestamp"]).total_seconds() / 3600
                 for r in recent]

        # Linear regression for trend
        if len(times) >= 2:
            slope = np.polyfit(times, averages, 1)[0]
        else:
            slope = 0

        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "period_hours": hours,
            "n_readings": len(recent),
            "start_avg": averages[0] if averages else 0,
            "end_avg": averages[-1] if averages else 0,
            "slope_pct_per_hour": slope,
            "trend": trend,
        }


class DeficitCalculator:
    """
    Soil water deficit calculations.

    Calculates irrigation requirements based
    on soil water deficit.

    Example:
        >>> calc = DeficitCalculator(soil_type="loam", root_depth=0.6)
        >>> deficit = calc.calculate_deficit(current_moisture=180)
        >>> irrigation = calc.irrigation_requirement(deficit, efficiency=0.85)
    """

    def __init__(
        self,
        soil_type: str = "loam",
        root_depth: float = 0.6,
        mad: float = 0.50,
    ):
        """
        Initialize deficit calculator.

        Args:
            soil_type: Soil texture class
            root_depth: Effective root depth (m)
            mad: Management allowable depletion (0-1)
        """
        self.soil_type = soil_type.lower().replace(" ", "_")
        self.root_depth = root_depth
        self.mad = mad

        self.params = SOIL_WATER_PARAMS.get(
            self.soil_type,
            SOIL_WATER_PARAMS["loam"]
        )

        # Calculate thresholds
        self.fc = self.params.field_capacity * root_depth  # mm
        self.wp = self.params.wilting_point * root_depth  # mm
        self.taw = self.fc - self.wp  # Total available water
        self.raw = self.taw * self.mad  # Readily available water

    def calculate_deficit(
        self,
        current_moisture: float,
    ) -> Dict[str, float]:
        """
        Calculate soil water deficit.

        Args:
            current_moisture: Current soil water (mm)

        Returns:
            Dict with deficit calculations
        """
        # Deficit from field capacity
        deficit_from_fc = self.fc - current_moisture

        # Deficit from refill point (FC - RAW)
        refill_point = self.fc - self.raw
        deficit_from_refill = max(0, refill_point - current_moisture)

        # Available water remaining
        available = max(0, current_moisture - self.wp)

        # Depletion fraction
        depletion = (self.fc - current_moisture) / self.taw if self.taw > 0 else 0

        return {
            "current_moisture_mm": current_moisture,
            "field_capacity_mm": self.fc,
            "wilting_point_mm": self.wp,
            "total_available_mm": self.taw,
            "readily_available_mm": self.raw,
            "deficit_from_fc_mm": max(0, deficit_from_fc),
            "deficit_from_refill_mm": deficit_from_refill,
            "available_water_mm": available,
            "depletion_fraction": min(1, max(0, depletion)),
            "irrigation_needed": depletion > self.mad,
        }

    def irrigation_requirement(
        self,
        deficit_mm: float,
        efficiency: float = 0.85,
        leaching_fraction: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate irrigation requirement.

        Args:
            deficit_mm: Soil water deficit (mm)
            efficiency: Application efficiency (0-1)
            leaching_fraction: Leaching requirement (0-1)

        Returns:
            Dict with irrigation requirements
        """
        # Net irrigation (to refill deficit)
        net_irrigation = deficit_mm

        # Gross irrigation (account for efficiency)
        gross_irrigation = net_irrigation / efficiency if efficiency > 0 else net_irrigation

        # Add leaching requirement
        if leaching_fraction > 0:
            gross_irrigation = gross_irrigation / (1 - leaching_fraction)

        return {
            "net_irrigation_mm": net_irrigation,
            "gross_irrigation_mm": gross_irrigation,
            "efficiency": efficiency,
            "leaching_fraction": leaching_fraction,
            "application_losses_mm": gross_irrigation - net_irrigation,
        }


class WaterBalance:
    """
    Soil water balance model.

    Tracks water inputs and outputs to maintain
    soil water budget over time.

    Example:
        >>> balance = WaterBalance(soil_type="loam", root_depth=0.6)
        >>> balance.add_irrigation(25.0)
        >>> balance.add_et(5.0)
        >>> status = balance.get_status()
    """

    def __init__(
        self,
        soil_type: str = "loam",
        root_depth: float = 0.6,
        initial_moisture: Optional[float] = None,
    ):
        """
        Initialize water balance.

        Args:
            soil_type: Soil texture class
            root_depth: Root zone depth (m)
            initial_moisture: Initial soil water (mm)
        """
        self.soil_type = soil_type.lower().replace(" ", "_")
        self.root_depth = root_depth

        self.params = SOIL_WATER_PARAMS.get(
            self.soil_type,
            SOIL_WATER_PARAMS["loam"]
        )

        # Calculate capacity
        self.fc = self.params.field_capacity * root_depth
        self.wp = self.params.wilting_point * root_depth
        self.sat = self.params.saturation * root_depth

        # Initial moisture (default to 80% of FC)
        if initial_moisture is None:
            self.current_moisture = self.fc * 0.8
        else:
            self.current_moisture = initial_moisture

        self._history: List[WaterBudget] = []
        self._today_budget = WaterBudget(
            date=date.today(),
            initial_storage=self.current_moisture,
        )

    def add_irrigation(
        self,
        amount_mm: float,
        efficiency: float = 1.0,
    ) -> float:
        """
        Add irrigation water.

        Args:
            amount_mm: Irrigation amount (mm)
            efficiency: Application efficiency

        Returns:
            Actual water added (mm)
        """
        effective_amount = amount_mm * efficiency
        self._today_budget.irrigation += effective_amount

        # Check for excess (drainage)
        new_moisture = self.current_moisture + effective_amount
        if new_moisture > self.fc:
            drainage = new_moisture - self.fc
            self._today_budget.drainage += drainage
            self.current_moisture = self.fc
        else:
            self.current_moisture = new_moisture

        return effective_amount

    def add_rainfall(
        self,
        amount_mm: float,
    ) -> Tuple[float, float]:
        """
        Add rainfall.

        Args:
            amount_mm: Rainfall amount (mm)

        Returns:
            Tuple of (effective_rain, runoff)
        """
        self._today_budget.rainfall += amount_mm

        # Calculate infiltration capacity
        available_capacity = self.sat - self.current_moisture
        max_infiltration = self.params.infiltration_rate  # mm/hr (simplified)

        # Partition to infiltration and runoff
        infiltration = min(amount_mm, available_capacity, max_infiltration)
        runoff = amount_mm - infiltration

        self._today_budget.runoff += runoff

        # Update moisture and drainage
        new_moisture = self.current_moisture + infiltration
        if new_moisture > self.fc:
            drainage = new_moisture - self.fc
            self._today_budget.drainage += drainage
            self.current_moisture = self.fc
        else:
            self.current_moisture = new_moisture

        return infiltration, runoff

    def add_et(
        self,
        amount_mm: float,
    ) -> float:
        """
        Remove water through evapotranspiration.

        Args:
            amount_mm: ET amount (mm)

        Returns:
            Actual ET (limited by available water)
        """
        self._today_budget.et += amount_mm

        # ET limited by available water
        available = self.current_moisture - self.wp
        actual_et = min(amount_mm, available)

        self.current_moisture -= actual_et

        return actual_et

    def get_status(self) -> Dict[str, Any]:
        """
        Get current water balance status.

        Returns:
            Dict with status information
        """
        taw = self.fc - self.wp
        available = self.current_moisture - self.wp
        depletion = (self.fc - self.current_moisture) / taw if taw > 0 else 0

        # Classify status
        if depletion < 0.3:
            status = "adequate"
        elif depletion < 0.5:
            status = "optimal"
        elif depletion < 0.7:
            status = "approaching_stress"
        else:
            status = "stress"

        return {
            "date": date.today(),
            "current_moisture_mm": self.current_moisture,
            "field_capacity_mm": self.fc,
            "wilting_point_mm": self.wp,
            "available_water_mm": max(0, available),
            "depletion_fraction": min(1, max(0, depletion)),
            "status": status,
            "irrigation_recommended": depletion > 0.5,
            "deficit_mm": max(0, self.fc - self.current_moisture),
        }

    def close_day(self) -> WaterBudget:
        """
        Close daily water budget.

        Returns:
            Completed WaterBudget for the day
        """
        self._today_budget.final_storage = self.current_moisture
        completed = self._today_budget

        self._history.append(completed)

        # Start new day
        self._today_budget = WaterBudget(
            date=date.today() + timedelta(days=1),
            initial_storage=self.current_moisture,
        )

        return completed

    def get_history(
        self,
        days: int = 30,
    ) -> List[WaterBudget]:
        """Get water budget history."""
        return self._history[-days:]

    def project_deficit(
        self,
        days: int,
        daily_et: float,
        expected_rain: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Project future water deficit.

        Args:
            days: Days to project
            daily_et: Expected daily ET (mm)
            expected_rain: Expected rainfall (mm total)

        Returns:
            Dict with projection
        """
        projected_moisture = self.current_moisture
        daily_rain = expected_rain / days

        for _ in range(days):
            # Add rain
            projected_moisture += daily_rain
            projected_moisture = min(projected_moisture, self.fc)

            # Remove ET
            projected_moisture -= daily_et
            projected_moisture = max(projected_moisture, self.wp)

        taw = self.fc - self.wp
        projected_depletion = (self.fc - projected_moisture) / taw if taw > 0 else 0

        return {
            "projection_days": days,
            "daily_et_mm": daily_et,
            "total_rain_mm": expected_rain,
            "projected_moisture_mm": projected_moisture,
            "projected_depletion": projected_depletion,
            "irrigation_needed": projected_depletion > 0.5,
            "required_irrigation_mm": max(0, self.fc - projected_moisture),
        }
