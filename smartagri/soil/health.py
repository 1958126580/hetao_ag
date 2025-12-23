"""
Soil Health Assessment Module

Provides comprehensive soil health evaluation tools:
- Physical, chemical, and biological indicators
- Soil quality index calculation
- Organic matter dynamics modeling
- Management recommendations

Example:
    >>> health = SoilHealthAssessment()
    >>> score = health.calculate_index(soil_data)
    >>> print(f"Soil Health Score: {score.overall_score:.0f}/100")
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SoilQualityIndex:
    """
    Soil quality assessment results.

    Attributes:
        overall_score: Overall soil quality score (0-100)
        physical_score: Physical properties score
        chemical_score: Chemical properties score
        biological_score: Biological properties score
        limiting_factors: Identified limiting factors
        recommendations: Management recommendations
    """
    overall_score: float
    physical_score: float
    chemical_score: float
    biological_score: float
    limiting_factors: List[str]
    recommendations: List[str]


class SoilHealthAssessment:
    """
    Comprehensive soil health assessment system.

    Evaluates soil health across physical, chemical, and biological
    dimensions to provide actionable management recommendations.

    Example:
        >>> assessment = SoilHealthAssessment()
        >>> result = assessment.calculate_index({
        ...     "organic_matter": 3.5,
        ...     "bulk_density": 1.3,
        ...     "pH": 6.5,
        ...     "aggregate_stability": 60
        ... })
    """

    def __init__(self):
        """Initialize soil health assessment."""
        # Indicator scoring functions and weights
        self._physical_indicators = {
            "bulk_density": {"weight": 0.25, "optimal": (1.1, 1.4), "unit": "g/cm3"},
            "aggregate_stability": {"weight": 0.35, "optimal": (60, 100), "unit": "%"},
            "infiltration_rate": {"weight": 0.25, "optimal": (25, 200), "unit": "mm/h"},
            "porosity": {"weight": 0.15, "optimal": (0.45, 0.55), "unit": "m3/m3"},
        }

        self._chemical_indicators = {
            "pH": {"weight": 0.20, "optimal": (6.0, 7.0), "unit": ""},
            "organic_matter": {"weight": 0.30, "optimal": (3.0, 6.0), "unit": "%"},
            "cec": {"weight": 0.20, "optimal": (15, 30), "unit": "meq/100g"},
            "nutrient_balance": {"weight": 0.30, "optimal": (0.8, 1.2), "unit": "ratio"},
        }

        self._biological_indicators = {
            "microbial_biomass": {"weight": 0.30, "optimal": (300, 600), "unit": "mg/kg"},
            "respiration_rate": {"weight": 0.25, "optimal": (50, 150), "unit": "mg CO2/kg/day"},
            "earthworm_count": {"weight": 0.20, "optimal": (10, 30), "unit": "per m2"},
            "active_carbon": {"weight": 0.25, "optimal": (400, 800), "unit": "mg/kg"},
        }

    def calculate_index(
        self,
        soil_data: Dict[str, float],
    ) -> SoilQualityIndex:
        """
        Calculate comprehensive soil quality index.

        Args:
            soil_data: Dict of measured soil properties

        Returns:
            SoilQualityIndex with scores and recommendations
        """
        physical_score = self._calculate_category_score(
            soil_data, self._physical_indicators
        )
        chemical_score = self._calculate_category_score(
            soil_data, self._chemical_indicators
        )
        biological_score = self._calculate_category_score(
            soil_data, self._biological_indicators
        )

        # Weighted overall score
        overall = (
            physical_score * 0.30 +
            chemical_score * 0.35 +
            biological_score * 0.35
        )

        # Identify limiting factors
        limiting = self._identify_limiting_factors(soil_data)

        # Generate recommendations
        recommendations = self._generate_recommendations(soil_data, limiting)

        return SoilQualityIndex(
            overall_score=overall,
            physical_score=physical_score,
            chemical_score=chemical_score,
            biological_score=biological_score,
            limiting_factors=limiting,
            recommendations=recommendations,
        )

    def _calculate_category_score(
        self,
        soil_data: Dict[str, float],
        indicators: Dict[str, Dict],
    ) -> float:
        """Calculate weighted score for an indicator category."""
        total_weight = 0.0
        weighted_sum = 0.0

        for indicator, params in indicators.items():
            if indicator in soil_data:
                value = soil_data[indicator]
                score = self._score_indicator(value, params["optimal"])
                weighted_sum += score * params["weight"]
                total_weight += params["weight"]

        if total_weight > 0:
            return weighted_sum / total_weight * 100
        return 50.0  # Default when no data

    def _score_indicator(
        self,
        value: float,
        optimal_range: tuple,
    ) -> float:
        """Score an indicator value (0-1)."""
        low, high = optimal_range

        if low <= value <= high:
            return 1.0
        elif value < low:
            return max(0, 1 - (low - value) / low)
        else:
            return max(0, 1 - (value - high) / high)

    def _identify_limiting_factors(
        self,
        soil_data: Dict[str, float],
    ) -> List[str]:
        """Identify soil health limiting factors."""
        limiting = []

        # Check key indicators
        if soil_data.get("organic_matter", 3.0) < 2.0:
            limiting.append("Low organic matter")

        if soil_data.get("bulk_density", 1.3) > 1.5:
            limiting.append("Soil compaction")

        ph = soil_data.get("pH", 6.5)
        if ph < 5.5:
            limiting.append("Soil acidity")
        elif ph > 7.5:
            limiting.append("Alkaline conditions")

        if soil_data.get("aggregate_stability", 50) < 40:
            limiting.append("Poor soil structure")

        return limiting

    def _generate_recommendations(
        self,
        soil_data: Dict[str, float],
        limiting_factors: List[str],
    ) -> List[str]:
        """Generate management recommendations."""
        recommendations = []

        if "Low organic matter" in limiting_factors:
            recommendations.append(
                "Add organic amendments (compost, cover crops, crop residues)"
            )

        if "Soil compaction" in limiting_factors:
            recommendations.append(
                "Reduce tillage, add organic matter, limit field traffic"
            )

        if "Soil acidity" in limiting_factors:
            recommendations.append("Apply agricultural lime to raise pH")

        if "Poor soil structure" in limiting_factors:
            recommendations.append(
                "Improve aggregation through organic matter and reduced tillage"
            )

        if not recommendations:
            recommendations.append("Continue current management practices")

        return recommendations


class OrganicMatterDynamics:
    """
    Model organic matter dynamics in soil.

    Simulates carbon input, decomposition, and stabilization
    for long-term soil carbon management.
    """

    def __init__(
        self,
        initial_som: float = 3.0,
        clay_content: float = 25.0,
    ):
        """
        Initialize organic matter dynamics model.

        Args:
            initial_som: Initial SOM content (%)
            clay_content: Clay content (%)
        """
        self.initial_som = initial_som
        self.clay_content = clay_content

        # Decomposition parameters
        self.k_active = 0.5  # Active pool decay (per year)
        self.k_slow = 0.05  # Slow pool decay
        self.k_passive = 0.001  # Passive pool decay

    def simulate(
        self,
        carbon_inputs: np.ndarray,
        temperature: np.ndarray,
        moisture: np.ndarray,
        years: int,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate SOM dynamics over time.

        Args:
            carbon_inputs: Annual C inputs (t/ha/year)
            temperature: Mean annual temperature (°C)
            moisture: Soil moisture factor (0-1)
            years: Number of years to simulate

        Returns:
            Dict with SOM pools over time
        """
        # Initialize pools (approximate partition of SOM)
        som_total = self.initial_som * 100 * 0.58  # t C/ha (top 30cm)
        active = som_total * 0.05
        slow = som_total * 0.60
        passive = som_total * 0.35

        # Output arrays
        active_series = np.zeros(years)
        slow_series = np.zeros(years)
        passive_series = np.zeros(years)
        total_series = np.zeros(years)

        for year in range(years):
            # Temperature effect (Q10 = 2)
            temp = temperature[year] if year < len(temperature) else 15.0
            temp_factor = 2 ** ((temp - 15) / 10)

            # Moisture effect
            moist = moisture[year] if year < len(moisture) else 0.7
            moist_factor = min(1.0, moist / 0.5)

            # Combined effect
            env_factor = temp_factor * moist_factor

            # Carbon input
            c_input = carbon_inputs[year] if year < len(carbon_inputs) else 2.0

            # Decomposition and transfers
            active_decomp = active * self.k_active * env_factor
            slow_decomp = slow * self.k_slow * env_factor
            passive_decomp = passive * self.k_passive * env_factor

            # Update pools
            active = active + c_input * 0.8 - active_decomp
            slow = slow + c_input * 0.15 + active_decomp * 0.3 - slow_decomp
            passive = passive + c_input * 0.05 + slow_decomp * 0.1 - passive_decomp

            # Ensure non-negative
            active = max(0, active)
            slow = max(0, slow)
            passive = max(0, passive)

            # Store
            active_series[year] = active
            slow_series[year] = slow
            passive_series[year] = passive
            total_series[year] = active + slow + passive

        # Convert back to SOM %
        som_percent = total_series / (100 * 0.58)

        return {
            "active_c": active_series,
            "slow_c": slow_series,
            "passive_c": passive_series,
            "total_c": total_series,
            "som_percent": som_percent,
        }


class BiologicalActivity:
    """
    Assess soil biological activity indicators.
    """

    def estimate_microbial_biomass(
        self,
        organic_matter: float,
        clay: float,
        ph: float,
    ) -> float:
        """
        Estimate microbial biomass carbon.

        Args:
            organic_matter: Organic matter (%)
            clay: Clay content (%)
            ph: Soil pH

        Returns:
            Estimated MBC (mg C/kg soil)
        """
        # Base estimate from organic matter
        mbc = organic_matter * 100  # ~1% of OM as MBC

        # Clay effect (protection)
        clay_factor = 1 + clay / 100

        # pH effect (optimal around 6-7)
        ph_factor = 1 - abs(ph - 6.5) * 0.1

        return mbc * clay_factor * ph_factor

    def estimate_respiration(
        self,
        organic_matter: float,
        temperature: float,
        moisture: float = 0.5,
    ) -> float:
        """
        Estimate soil respiration rate.

        Args:
            organic_matter: Organic matter (%)
            temperature: Soil temperature (°C)
            moisture: Volumetric water content

        Returns:
            Respiration rate (mg CO2/kg/day)
        """
        # Base respiration
        base_resp = organic_matter * 20  # mg CO2/kg/day

        # Temperature effect
        temp_factor = 2 ** ((temperature - 15) / 10)

        # Moisture effect (optimum around 0.4)
        if moisture < 0.2:
            moist_factor = moisture / 0.2
        elif moisture < 0.5:
            moist_factor = 1.0
        else:
            moist_factor = 1 - (moisture - 0.5) / 0.5

        return base_resp * temp_factor * max(0, moist_factor)
