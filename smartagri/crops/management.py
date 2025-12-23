"""
Crop Management Module

Provides tools for optimizing crop management decisions including:
- Planting date optimization
- Harvest timing
- Crop rotation planning
- Fertilizer scheduling

Example:
    >>> manager = CropManager(location=(40.0, -90.0))
    >>> planting = manager.optimize_planting_date("corn", weather_forecast)
    >>> rotation = manager.plan_rotation(fields, years=5)
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Dict,
    Any,
)
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class Field:
    """
    Represents an agricultural field.

    Attributes:
        id: Unique identifier
        name: Field name
        area: Field area (hectares)
        soil_type: Soil classification
        previous_crops: History of crops grown
        constraints: Management constraints
    """
    id: str
    name: str
    area: float
    soil_type: str = "loam"
    previous_crops: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlantingRecommendation:
    """
    Planting date recommendation.

    Attributes:
        optimal_date: Recommended planting date
        earliest_date: Earliest safe planting date
        latest_date: Latest recommended date
        confidence: Confidence score (0-1)
        factors: Factors affecting recommendation
    """
    optimal_date: date
    earliest_date: date
    latest_date: date
    confidence: float
    factors: Dict[str, str]


@dataclass
class HarvestRecommendation:
    """
    Harvest timing recommendation.

    Attributes:
        optimal_date: Recommended harvest date
        harvest_window: (start, end) of harvest window
        expected_yield: Expected yield (kg/ha)
        moisture_content: Expected grain moisture (%)
        drying_cost: Estimated drying cost if harvested on optimal date
    """
    optimal_date: date
    harvest_window: Tuple[date, date]
    expected_yield: float
    moisture_content: float
    drying_cost: float = 0.0


class CropManager:
    """
    Comprehensive crop management system.

    Provides decision support for key crop management activities
    using weather data, soil information, and economic factors.

    Example:
        >>> manager = CropManager(location=(40.0, -90.0))
        >>> rec = manager.optimize_planting_date("corn", weather_data)
        >>> print(f"Plant on {rec.optimal_date}")
    """

    def __init__(
        self,
        location: Tuple[float, float],
        elevation: float = 0.0,
    ):
        """
        Initialize crop manager.

        Args:
            location: (latitude, longitude) tuple
            elevation: Elevation in meters
        """
        self.latitude, self.longitude = location
        self.elevation = elevation

        # Crop-specific parameters
        self._crop_params = {
            "corn": {
                "min_soil_temp": 10,
                "optimal_soil_temp": 15,
                "gdd_maturity": 1500,
                "optimal_moisture_harvest": 20,
                "price_per_kg": 0.20,
            },
            "soybean": {
                "min_soil_temp": 12,
                "optimal_soil_temp": 18,
                "gdd_maturity": 1400,
                "optimal_moisture_harvest": 13,
                "price_per_kg": 0.35,
            },
            "wheat": {
                "min_soil_temp": 5,
                "optimal_soil_temp": 10,
                "gdd_maturity": 1800,
                "optimal_moisture_harvest": 14,
                "price_per_kg": 0.25,
            },
        }

    def optimize_planting_date(
        self,
        crop: str,
        weather_forecast: Dict[str, np.ndarray],
        soil_conditions: Optional[Dict[str, float]] = None,
    ) -> PlantingRecommendation:
        """
        Optimize planting date based on weather and soil.

        Args:
            crop: Crop type
            weather_forecast: Weather forecast data
            soil_conditions: Current soil conditions

        Returns:
            PlantingRecommendation with optimal timing
        """
        params = self._crop_params.get(crop, self._crop_params["corn"])

        t_min = weather_forecast.get("t_min", np.full(90, 10.0))
        t_max = weather_forecast.get("t_max", np.full(90, 25.0))
        precip = weather_forecast.get("precipitation", np.zeros(90))

        # Estimate soil temperature from air temperature
        soil_temp = (t_min + t_max) / 2 - 2  # Simplified

        # Find first suitable planting date
        min_temp = params["min_soil_temp"]
        optimal_temp = params["optimal_soil_temp"]

        earliest_idx = None
        optimal_idx = None

        for i in range(len(soil_temp) - 7):
            # Check 7-day average
            avg_temp = np.mean(soil_temp[i:i + 7])

            if earliest_idx is None and avg_temp >= min_temp:
                earliest_idx = i

            if optimal_idx is None and avg_temp >= optimal_temp:
                optimal_idx = i
                break

        if earliest_idx is None:
            earliest_idx = 30  # Default fallback

        if optimal_idx is None:
            optimal_idx = earliest_idx + 7

        today = date.today()

        # Check for wet conditions
        factors = {}
        if np.mean(precip[optimal_idx:optimal_idx + 7]) > 10:
            factors["precipitation"] = "High rainfall expected, may delay planting"
            optimal_idx += 7

        earliest_date = today + timedelta(days=earliest_idx)
        optimal_date = today + timedelta(days=optimal_idx)
        latest_date = optimal_date + timedelta(days=21)

        # Calculate confidence based on forecast reliability
        confidence = 0.8 - 0.01 * optimal_idx  # Lower confidence for later dates

        return PlantingRecommendation(
            optimal_date=optimal_date,
            earliest_date=earliest_date,
            latest_date=latest_date,
            confidence=max(0.5, confidence),
            factors=factors,
        )

    def optimize_harvest_date(
        self,
        crop: str,
        planting_date: date,
        weather_data: Dict[str, np.ndarray],
        current_gdd: float = 0.0,
    ) -> HarvestRecommendation:
        """
        Optimize harvest timing.

        Args:
            crop: Crop type
            planting_date: Planting date
            weather_data: Historical/forecast weather
            current_gdd: Accumulated GDD

        Returns:
            HarvestRecommendation with optimal timing
        """
        params = self._crop_params.get(crop, self._crop_params["corn"])
        target_gdd = params["gdd_maturity"]

        t_min = weather_data.get("t_min", np.full(180, 15.0))
        t_max = weather_data.get("t_max", np.full(180, 28.0))

        # Calculate GDD accumulation
        gdd_daily = np.maximum((t_min + t_max) / 2 - 10, 0)
        gdd_cumulative = np.cumsum(gdd_daily) + current_gdd

        # Find maturity date
        maturity_idx = np.searchsorted(gdd_cumulative, target_gdd)

        if maturity_idx >= len(gdd_daily):
            maturity_idx = len(gdd_daily) - 1

        maturity_date = planting_date + timedelta(days=int(maturity_idx))

        # Harvest window (7-14 days after maturity)
        harvest_start = maturity_date + timedelta(days=7)
        harvest_end = maturity_date + timedelta(days=21)
        optimal_date = maturity_date + timedelta(days=10)

        # Estimate moisture content
        days_past_maturity = 10
        moisture_content = params["optimal_moisture_harvest"] + 5 - 0.3 * days_past_maturity

        # Estimate yield (simplified)
        expected_yield = 10000  # kg/ha baseline

        return HarvestRecommendation(
            optimal_date=optimal_date,
            harvest_window=(harvest_start, harvest_end),
            expected_yield=expected_yield,
            moisture_content=max(moisture_content, params["optimal_moisture_harvest"]),
        )


class PlantingOptimizer:
    """
    Advanced planting date optimization.

    Uses multi-objective optimization to balance yield potential,
    risk, and resource constraints.
    """

    def __init__(self, risk_tolerance: float = 0.5):
        """
        Initialize planting optimizer.

        Args:
            risk_tolerance: Risk tolerance (0=risk-averse, 1=risk-seeking)
        """
        self.risk_tolerance = risk_tolerance

    def optimize(
        self,
        crop: str,
        weather_scenarios: List[Dict[str, np.ndarray]],
        field: Field,
    ) -> PlantingRecommendation:
        """
        Optimize planting date across weather scenarios.

        Args:
            crop: Crop type
            weather_scenarios: Multiple weather scenarios
            field: Field information

        Returns:
            Robust planting recommendation
        """
        # Evaluate multiple planting dates across scenarios
        candidate_dates = range(0, 60, 7)  # Weekly candidates
        scores = np.zeros((len(candidate_dates), len(weather_scenarios)))

        for i, offset in enumerate(candidate_dates):
            for j, weather in enumerate(weather_scenarios):
                score = self._evaluate_planting(crop, offset, weather)
                scores[i, j] = score

        # Robust optimization: weighted sum of mean and worst-case
        mean_scores = np.mean(scores, axis=1)
        min_scores = np.min(scores, axis=1)

        robust_scores = (
            self.risk_tolerance * mean_scores +
            (1 - self.risk_tolerance) * min_scores
        )

        best_idx = np.argmax(robust_scores)
        best_offset = candidate_dates[best_idx]

        today = date.today()

        return PlantingRecommendation(
            optimal_date=today + timedelta(days=best_offset),
            earliest_date=today + timedelta(days=max(0, best_offset - 14)),
            latest_date=today + timedelta(days=best_offset + 14),
            confidence=min(1.0, robust_scores[best_idx] / 100),
            factors={"method": "robust_optimization"},
        )

    def _evaluate_planting(
        self,
        crop: str,
        planting_offset: int,
        weather: Dict[str, np.ndarray],
    ) -> float:
        """Evaluate yield potential for a planting date."""
        # Simplified yield scoring
        t_avg = (weather.get("t_min", [20])[0] + weather.get("t_max", [30])[0]) / 2
        score = 100 - abs(t_avg - 22) * 5
        return max(0, score)


class HarvestOptimizer:
    """
    Harvest timing optimization.

    Balances yield, quality, and drying costs to determine
    optimal harvest timing.
    """

    def __init__(
        self,
        drying_cost_per_point: float = 0.05,
        yield_loss_per_day: float = 0.002,
    ):
        """
        Initialize harvest optimizer.

        Args:
            drying_cost_per_point: Cost per moisture point to dry ($/kg)
            yield_loss_per_day: Daily yield loss from delayed harvest (fraction)
        """
        self.drying_cost_per_point = drying_cost_per_point
        self.yield_loss_per_day = yield_loss_per_day

    def optimize(
        self,
        crop: str,
        maturity_date: date,
        weather_forecast: Dict[str, np.ndarray],
        grain_price: float,
    ) -> HarvestRecommendation:
        """
        Optimize harvest date considering costs and weather.

        Args:
            crop: Crop type
            maturity_date: Estimated maturity date
            weather_forecast: Weather forecast
            grain_price: Current grain price ($/kg)

        Returns:
            HarvestRecommendation maximizing net returns
        """
        # Evaluate harvest dates from maturity to maturity+21
        candidates = []
        base_yield = 10000  # kg/ha

        for days in range(0, 22):
            harvest_date = maturity_date + timedelta(days=days)

            # Estimate moisture content (dries ~0.5% per day)
            moisture = 25 - 0.5 * days
            moisture = max(14, moisture)  # Minimum moisture

            # Drying cost
            target_moisture = 14
            if moisture > target_moisture:
                drying_cost = (moisture - target_moisture) * self.drying_cost_per_point * base_yield
            else:
                drying_cost = 0

            # Yield loss from delay
            yield_loss = base_yield * self.yield_loss_per_day * days

            # Weather risk (simplified)
            precip_risk = 0
            if days < len(weather_forecast.get("precipitation", [])):
                precip_risk = weather_forecast["precipitation"][days] * 10

            # Net return
            actual_yield = base_yield - yield_loss
            revenue = actual_yield * grain_price
            net_return = revenue - drying_cost - precip_risk

            candidates.append({
                "date": harvest_date,
                "days": days,
                "moisture": moisture,
                "yield": actual_yield,
                "drying_cost": drying_cost,
                "net_return": net_return,
            })

        # Find optimal
        best = max(candidates, key=lambda x: x["net_return"])

        return HarvestRecommendation(
            optimal_date=best["date"],
            harvest_window=(maturity_date, maturity_date + timedelta(days=21)),
            expected_yield=best["yield"],
            moisture_content=best["moisture"],
            drying_cost=best["drying_cost"],
        )


class CropRotation:
    """
    Crop rotation planning and optimization.

    Plans multi-year crop sequences considering soil health,
    pest pressure, and economic returns.

    Example:
        >>> rotation = CropRotation()
        >>> plan = rotation.generate_plan(
        ...     fields=[field1, field2],
        ...     years=5,
        ...     available_crops=["corn", "soybean", "wheat"]
        ... )
    """

    def __init__(self):
        """Initialize rotation planner."""
        # Rotation rules: crop -> (good_precursors, bad_precursors, min_years_between)
        self._rotation_rules = {
            "corn": {
                "good_precursors": ["soybean", "alfalfa", "wheat"],
                "bad_precursors": ["corn"],
                "min_years_between": 1,
            },
            "soybean": {
                "good_precursors": ["corn", "wheat"],
                "bad_precursors": ["soybean"],
                "min_years_between": 2,
            },
            "wheat": {
                "good_precursors": ["corn", "soybean"],
                "bad_precursors": ["wheat", "barley"],
                "min_years_between": 2,
            },
        }

    def generate_plan(
        self,
        fields: List[Field],
        years: int,
        available_crops: List[str],
        optimize_for: str = "yield",
    ) -> Dict[str, List[str]]:
        """
        Generate rotation plan for multiple fields and years.

        Args:
            fields: List of fields
            years: Number of years to plan
            available_crops: Crops available to grow
            optimize_for: Optimization objective ('yield', 'profit', 'soil_health')

        Returns:
            Dict mapping field IDs to crop sequence
        """
        plan = {}

        for field in fields:
            sequence = self._optimize_sequence(
                field, years, available_crops, optimize_for
            )
            plan[field.id] = sequence

        return plan

    def _optimize_sequence(
        self,
        field: Field,
        years: int,
        crops: List[str],
        objective: str,
    ) -> List[str]:
        """Optimize crop sequence for a single field."""
        sequence = []
        previous = field.previous_crops[-1] if field.previous_crops else None

        for year in range(years):
            best_crop = None
            best_score = -float("inf")

            for crop in crops:
                score = self._score_crop(crop, previous, sequence, objective)
                if score > best_score:
                    best_score = score
                    best_crop = crop

            sequence.append(best_crop)
            previous = best_crop

        return sequence

    def _score_crop(
        self,
        crop: str,
        previous: Optional[str],
        history: List[str],
        objective: str,
    ) -> float:
        """Score a crop choice based on rotation rules and objectives."""
        rules = self._rotation_rules.get(crop, {})
        score = 50  # Base score

        # Precursor bonus/penalty
        if previous in rules.get("good_precursors", []):
            score += 30
        if previous in rules.get("bad_precursors", []):
            score -= 40

        # Check minimum years between same crop
        min_years = rules.get("min_years_between", 1)
        if crop in history[-min_years:]:
            score -= 50

        # Objective-specific adjustments
        if objective == "yield":
            if crop == "corn":
                score += 10  # Corn typically higher yielding
        elif objective == "profit":
            if crop == "soybean":
                score += 15  # Often more profitable per ha
        elif objective == "soil_health":
            if crop in ["soybean", "alfalfa"]:
                score += 20  # Legumes for N fixation

        return score

    def evaluate_plan(
        self,
        plan: Dict[str, List[str]],
        fields: List[Field],
    ) -> Dict[str, Any]:
        """
        Evaluate a rotation plan.

        Args:
            plan: Rotation plan
            fields: Field list

        Returns:
            Evaluation metrics
        """
        metrics = {
            "total_score": 0,
            "diversity_score": 0,
            "soil_health_score": 0,
            "warnings": [],
        }

        all_crops = []
        for field_id, sequence in plan.items():
            all_crops.extend(sequence)

            # Check for violations
            for i, crop in enumerate(sequence[1:], 1):
                prev = sequence[i - 1]
                rules = self._rotation_rules.get(crop, {})

                if prev in rules.get("bad_precursors", []):
                    metrics["warnings"].append(
                        f"Field {field_id}: {crop} after {prev} in year {i + 1}"
                    )

        # Diversity score
        unique_crops = len(set(all_crops))
        metrics["diversity_score"] = unique_crops / max(len(all_crops), 1) * 100

        return metrics


class FertilizerScheduler:
    """
    Fertilizer application scheduling.

    Plans fertilizer applications based on crop needs, soil tests,
    and environmental conditions.
    """

    def __init__(self):
        """Initialize fertilizer scheduler."""
        # Nutrient requirements (kg/ha per ton of yield)
        self._crop_requirements = {
            "corn": {"N": 25, "P2O5": 8, "K2O": 6},
            "soybean": {"N": 0, "P2O5": 12, "K2O": 20},  # N from fixation
            "wheat": {"N": 25, "P2O5": 10, "K2O": 5},
        }

    def create_schedule(
        self,
        crop: str,
        target_yield: float,
        soil_test: Dict[str, float],
        planting_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Create fertilizer application schedule.

        Args:
            crop: Crop type
            target_yield: Target yield (t/ha)
            soil_test: Soil test results (nutrient levels)
            planting_date: Planting date

        Returns:
            List of scheduled applications
        """
        requirements = self._crop_requirements.get(crop, self._crop_requirements["corn"])
        schedule = []

        # Calculate total nutrient needs
        total_n = requirements["N"] * target_yield
        total_p = requirements["P2O5"] * target_yield
        total_k = requirements["K2O"] * target_yield

        # Adjust for soil nutrients
        soil_n = soil_test.get("N", 0)
        soil_p = soil_test.get("P", 0)
        soil_k = soil_test.get("K", 0)

        needed_n = max(0, total_n - soil_n)
        needed_p = max(0, total_p - soil_p)
        needed_k = max(0, total_k - soil_k)

        # Schedule P and K at planting
        if needed_p > 0 or needed_k > 0:
            schedule.append({
                "date": planting_date,
                "timing": "at_planting",
                "nutrients": {"P2O5": needed_p, "K2O": needed_k},
                "method": "broadcast_incorporated",
            })

        # Split N applications
        if needed_n > 0:
            # Starter N
            starter_n = min(30, needed_n * 0.2)
            schedule.append({
                "date": planting_date,
                "timing": "at_planting",
                "nutrients": {"N": starter_n},
                "method": "starter_band",
            })

            # Side-dress N
            sidedress_n = needed_n - starter_n
            if sidedress_n > 0:
                sidedress_date = planting_date + timedelta(days=45)
                schedule.append({
                    "date": sidedress_date,
                    "timing": "V6_stage",
                    "nutrients": {"N": sidedress_n},
                    "method": "sidedress",
                })

        return schedule

    def optimize_timing(
        self,
        schedule: List[Dict[str, Any]],
        weather_forecast: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Adjust application timing based on weather.

        Args:
            schedule: Initial schedule
            weather_forecast: Weather forecast

        Returns:
            Adjusted schedule
        """
        precip = weather_forecast.get("precipitation", np.zeros(90))
        optimized = []

        for app in schedule:
            app_date = app["date"]
            today = date.today()
            days_ahead = (app_date - today).days

            if days_ahead >= 0 and days_ahead < len(precip):
                # Check for heavy rain within 3 days
                rain_window = precip[days_ahead:min(days_ahead + 3, len(precip))]
                if np.any(rain_window > 20):
                    # Delay by 3 days
                    app = dict(app)
                    app["date"] = app_date + timedelta(days=3)
                    app["notes"] = "Delayed due to rainfall forecast"

            optimized.append(app)

        return optimized
