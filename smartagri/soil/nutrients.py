"""
Soil Nutrient Management Module

Provides comprehensive nutrient analysis and recommendation tools:
- Soil test interpretation
- Fertilizer recommendations
- Nutrient balance calculations
- Environmental risk assessment

Example:
    >>> manager = NutrientManager()
    >>> rec = manager.fertilizer_recommendation(
    ...     soil_test={"N": 25, "P": 15, "K": 120, "pH": 6.5},
    ...     crop="corn",
    ...     target_yield=12000
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NutrientAnalysis:
    """
    Soil nutrient analysis results.

    Attributes:
        nitrogen: Available nitrogen (kg/ha)
        phosphorus: Available phosphorus (mg/kg Olsen or Mehlich-3)
        potassium: Available potassium (mg/kg)
        calcium: Exchangeable calcium (mg/kg)
        magnesium: Exchangeable magnesium (mg/kg)
        sulfur: Available sulfur (mg/kg)
        organic_matter: Organic matter (%)
        ph: Soil pH
        cec: Cation exchange capacity (meq/100g)
        micronutrients: Dict of micronutrient levels
    """
    nitrogen: float
    phosphorus: float
    potassium: float
    calcium: float = 1500.0
    magnesium: float = 200.0
    sulfur: float = 10.0
    organic_matter: float = 2.5
    ph: float = 6.5
    cec: float = 15.0
    micronutrients: Dict[str, float] = None

    def __post_init__(self):
        if self.micronutrients is None:
            self.micronutrients = {
                "Zn": 1.0, "Fe": 10.0, "Mn": 5.0,
                "Cu": 1.0, "B": 0.5, "Mo": 0.1
            }


@dataclass
class FertilizerRecommendation:
    """
    Fertilizer application recommendation.

    Attributes:
        nitrogen: Recommended N rate (kg/ha)
        phosphorus: Recommended P2O5 rate (kg/ha)
        potassium: Recommended K2O rate (kg/ha)
        lime: Lime recommendation (kg/ha CaCO3 equivalent)
        sulfur: Sulfur recommendation (kg/ha)
        micronutrients: Micronutrient recommendations
        application_schedule: Timing recommendations
        notes: Additional notes
    """
    nitrogen: float
    phosphorus: float
    potassium: float
    lime: float = 0.0
    sulfur: float = 0.0
    micronutrients: Dict[str, float] = None
    application_schedule: List[Dict[str, Any]] = None
    notes: List[str] = None

    def total_cost(
        self,
        prices: Dict[str, float] = None,
    ) -> float:
        """Calculate total fertilizer cost."""
        if prices is None:
            prices = {"N": 1.2, "P2O5": 1.0, "K2O": 0.8, "lime": 0.05}

        cost = (
            self.nitrogen * prices.get("N", 1.2) +
            self.phosphorus * prices.get("P2O5", 1.0) +
            self.potassium * prices.get("K2O", 0.8) +
            self.lime * prices.get("lime", 0.05)
        )
        return cost


class NutrientRecommender:
    """
    Simplified nutrient recommender for quick fertilizer estimates.

    Provides quick fertilizer recommendations based on
    crop type and soil nutrient levels.

    Example:
        >>> recommender = NutrientRecommender(crop_type='corn')
        >>> rec = recommender.recommend(soil_n=25, soil_p=15, soil_k=120)
    """

    def __init__(
        self,
        crop_type: str = "corn",
        target_yield: float = 10000,
    ):
        """
        Initialize recommender.

        Args:
            crop_type: Crop type
            target_yield: Target yield (kg/ha)
        """
        self.crop_type = crop_type.lower()
        self.target_yield = target_yield

        # Crop nutrient requirements (kg per ton yield)
        self._requirements = {
            "corn": {"N": 22, "P2O5": 9, "K2O": 6},
            "wheat": {"N": 25, "P2O5": 11, "K2O": 6},
            "soybean": {"N": 0, "P2O5": 14, "K2O": 24},
            "rice": {"N": 18, "P2O5": 8, "K2O": 4},
        }

    def recommend(
        self,
        soil_n: float = 0,
        soil_p: float = 0,
        soil_k: float = 0,
        crop_type: str = None,
    ) -> Dict[str, float]:
        """
        Generate fertilizer recommendations.

        Args:
            soil_n: Soil available N (kg/ha)
            soil_p: Soil available P (mg/kg)
            soil_k: Soil available K (mg/kg)
            crop_type: Override crop type

        Returns:
            Dict with N, P2O5, K2O recommendations (kg/ha)
        """
        crop = crop_type or self.crop_type
        reqs = self._requirements.get(crop, self._requirements["corn"])
        yield_t = self.target_yield / 1000

        # Calculate requirements
        n_need = reqs["N"] * yield_t - soil_n * 0.5
        p_need = reqs["P2O5"] * yield_t - soil_p * 0.3
        k_need = reqs["K2O"] * yield_t - soil_k * 0.2

        return {
            "N": max(0, n_need),
            "P2O5": max(0, p_need),
            "K2O": max(0, k_need),
        }

    def calculate(
        self,
        soil_test: Dict[str, float],
        crop_requirements: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate fertilizer recommendations from soil test and crop requirements.

        Args:
            soil_test: Dict with nitrogen, phosphorus, potassium, ph, organic_matter
            crop_requirements: Dict with crop, yield_target

        Returns:
            Dict with N, P2O5, K2O, lime recommendations (kg/ha)
        """
        crop = crop_requirements.get('crop', self.crop_type)
        yield_target = crop_requirements.get('yield_target', self.target_yield / 1000)

        reqs = self._requirements.get(crop.lower(), self._requirements["corn"])

        # Extract soil values
        soil_n = soil_test.get('nitrogen', 0)
        soil_p = soil_test.get('phosphorus', 0)
        soil_k = soil_test.get('potassium', 0)
        ph = soil_test.get('ph', 6.5)
        om = soil_test.get('organic_matter', 2.5)

        # Calculate requirements
        n_need = reqs["N"] * yield_target - soil_n * 0.5 - om * 10
        p_need = reqs["P2O5"] * yield_target * (1.5 if soil_p < 15 else 1.0)
        k_need = reqs["K2O"] * yield_target * (1.5 if soil_k < 120 else 1.0)

        # Lime requirement based on pH
        if ph < 6.0:
            lime = (6.5 - ph) * 2000
        else:
            lime = 0

        return {
            "nitrogen": max(0, n_need),
            "phosphorus": max(0, p_need),
            "potassium": max(0, k_need),
            "lime": lime,
            "N": max(0, n_need),
            "P2O5": max(0, p_need),
            "K2O": max(0, k_need),
        }


class NutrientManager:
    """
    Comprehensive nutrient management system.

    Provides soil test interpretation and fertilizer recommendations
    based on crop requirements and environmental conditions.

    Example:
        >>> manager = NutrientManager()
        >>> rec = manager.fertilizer_recommendation(
        ...     soil_test={"N": 25, "P": 15, "K": 120},
        ...     crop="corn",
        ...     target_yield=12000
        ... )
        >>> print(f"N recommendation: {rec.nitrogen:.0f} kg/ha")
    """

    def __init__(self, region: str = "midwest"):
        """
        Initialize nutrient manager.

        Args:
            region: Geographic region for recommendations
        """
        self.region = region

        # Crop nutrient requirements (kg per ton yield)
        self._crop_requirements = {
            "corn": {"N": 22, "P2O5": 9, "K2O": 6},
            "soybean": {"N": 0, "P2O5": 14, "K2O": 24},  # N from fixation
            "wheat": {"N": 25, "P2O5": 11, "K2O": 6},
            "cotton": {"N": 50, "P2O5": 20, "K2O": 30},
            "rice": {"N": 18, "P2O5": 8, "K2O": 4},
        }

        # Critical soil test levels (mg/kg or appropriate units)
        self._critical_levels = {
            "P": {"very_low": 8, "low": 15, "medium": 25, "high": 40},
            "K": {"very_low": 80, "low": 120, "medium": 170, "high": 250},
        }

    def interpret_soil_test(
        self,
        soil_test: Dict[str, float],
    ) -> Dict[str, str]:
        """
        Interpret soil test results.

        Args:
            soil_test: Soil test values

        Returns:
            Dict with interpretations for each nutrient
        """
        interpretations = {}

        # Phosphorus interpretation
        p = soil_test.get("P", 0)
        levels = self._critical_levels["P"]
        if p < levels["very_low"]:
            interpretations["P"] = "very_low"
        elif p < levels["low"]:
            interpretations["P"] = "low"
        elif p < levels["medium"]:
            interpretations["P"] = "medium"
        elif p < levels["high"]:
            interpretations["P"] = "optimum"
        else:
            interpretations["P"] = "high"

        # Potassium interpretation
        k = soil_test.get("K", 0)
        levels = self._critical_levels["K"]
        if k < levels["very_low"]:
            interpretations["K"] = "very_low"
        elif k < levels["low"]:
            interpretations["K"] = "low"
        elif k < levels["medium"]:
            interpretations["K"] = "medium"
        elif k < levels["high"]:
            interpretations["K"] = "optimum"
        else:
            interpretations["K"] = "high"

        # pH interpretation
        ph = soil_test.get("pH", 7.0)
        if ph < 5.5:
            interpretations["pH"] = "strongly_acid"
        elif ph < 6.0:
            interpretations["pH"] = "moderately_acid"
        elif ph < 6.5:
            interpretations["pH"] = "slightly_acid"
        elif ph < 7.5:
            interpretations["pH"] = "neutral"
        else:
            interpretations["pH"] = "alkaline"

        return interpretations

    def fertilizer_recommendation(
        self,
        soil_test: Dict[str, float],
        crop: str,
        target_yield: float,
        previous_crop: Optional[str] = None,
        organic_amendments: float = 0.0,
    ) -> FertilizerRecommendation:
        """
        Generate fertilizer recommendation.

        Args:
            soil_test: Soil test results
            crop: Target crop
            target_yield: Target yield (kg/ha)
            previous_crop: Previous crop grown
            organic_amendments: Organic matter applied (t/ha)

        Returns:
            FertilizerRecommendation with rates and timing
        """
        requirements = self._crop_requirements.get(
            crop, self._crop_requirements["corn"]
        )
        interpretations = self.interpret_soil_test(soil_test)

        # Calculate N requirement
        target_yield_t = target_yield / 1000
        n_removal = requirements["N"] * target_yield_t

        # Credits
        n_credits = 0.0
        if previous_crop == "soybean":
            n_credits += 40  # N credit from soybean
        if organic_amendments > 0:
            n_credits += organic_amendments * 20  # N from organic amendments

        # Soil N contribution
        soil_n = soil_test.get("N", 0)
        n_credits += soil_n * 0.5

        n_recommendation = max(0, n_removal - n_credits)

        # Calculate P recommendation based on soil test
        p_interp = interpretations.get("P", "medium")
        p_factors = {
            "very_low": 1.5, "low": 1.2, "medium": 1.0,
            "optimum": 0.5, "high": 0.0
        }
        p_removal = requirements["P2O5"] * target_yield_t
        p_recommendation = p_removal * p_factors.get(p_interp, 1.0)

        # Calculate K recommendation
        k_interp = interpretations.get("K", "medium")
        k_factors = {
            "very_low": 1.5, "low": 1.2, "medium": 1.0,
            "optimum": 0.5, "high": 0.0
        }
        k_removal = requirements["K2O"] * target_yield_t
        k_recommendation = k_removal * k_factors.get(k_interp, 1.0)

        # Lime recommendation
        lime_rec = self._calculate_lime_requirement(soil_test)

        # Application schedule
        schedule = self._create_application_schedule(
            crop, n_recommendation, p_recommendation, k_recommendation
        )

        notes = []
        if lime_rec > 0:
            notes.append(f"Apply lime at least 3 months before planting")
        if n_recommendation > 150:
            notes.append("Split N applications to reduce loss")

        return FertilizerRecommendation(
            nitrogen=n_recommendation,
            phosphorus=p_recommendation,
            potassium=k_recommendation,
            lime=lime_rec,
            application_schedule=schedule,
            notes=notes,
        )

    def _calculate_lime_requirement(
        self,
        soil_test: Dict[str, float],
        target_ph: float = 6.5,
    ) -> float:
        """Calculate lime requirement to reach target pH."""
        current_ph = soil_test.get("pH", 7.0)

        if current_ph >= target_ph:
            return 0.0

        # Buffer pH method (simplified)
        buffer_ph = soil_test.get("buffer_pH", current_ph + 0.5)
        cec = soil_test.get("CEC", 15.0)

        # Lime requirement (kg CaCO3/ha)
        lime_factor = 2000  # kg/ha per unit pH change for medium CEC
        lime_factor *= cec / 15  # Adjust for CEC

        lime_req = (target_ph - current_ph) * lime_factor

        return max(0, lime_req)

    def _create_application_schedule(
        self,
        crop: str,
        n_rate: float,
        p_rate: float,
        k_rate: float,
    ) -> List[Dict[str, Any]]:
        """Create fertilizer application schedule."""
        schedule = []

        # P and K at planting
        if p_rate > 0 or k_rate > 0:
            schedule.append({
                "timing": "pre-plant or at planting",
                "nutrients": {"P2O5": p_rate, "K2O": k_rate},
                "method": "broadcast and incorporate",
            })

        # N split for crops with high N demand
        if n_rate > 0:
            if n_rate > 100:
                # Split application
                schedule.append({
                    "timing": "at planting",
                    "nutrients": {"N": min(40, n_rate * 0.3)},
                    "method": "starter band",
                })
                schedule.append({
                    "timing": "V6-V8 stage",
                    "nutrients": {"N": n_rate * 0.7},
                    "method": "sidedress",
                })
            else:
                schedule.append({
                    "timing": "at planting",
                    "nutrients": {"N": n_rate},
                    "method": "broadcast or band",
                })

        return schedule

    def nutrient_balance(
        self,
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate nutrient balance.

        Args:
            inputs: Nutrient inputs (fertilizer, deposition, fixation)
            outputs: Nutrient outputs (harvest removal, losses)

        Returns:
            Dict with balance for each nutrient
        """
        nutrients = set(inputs.keys()) | set(outputs.keys())

        balance = {}
        for nutrient in nutrients:
            input_val = inputs.get(nutrient, 0)
            output_val = outputs.get(nutrient, 0)
            balance[nutrient] = input_val - output_val

        return balance


def calculate_cec(
    clay: float,
    organic_matter: float,
    clay_type: str = "mixed",
) -> float:
    """
    Calculate cation exchange capacity.

    Args:
        clay: Clay content (%)
        organic_matter: Organic matter content (%)
        clay_type: Clay mineralogy ('kaolinite', 'illite', 'smectite', 'mixed')

    Returns:
        CEC (meq/100g)
    """
    # CEC contribution from clay (varies by type)
    clay_cec_factors = {
        "kaolinite": 0.1,  # 3-15 meq/100g clay
        "illite": 0.25,    # 15-40 meq/100g clay
        "smectite": 0.8,   # 60-150 meq/100g clay
        "mixed": 0.35,     # Average
    }

    clay_factor = clay_cec_factors.get(clay_type, 0.35)
    cec_from_clay = clay * clay_factor

    # CEC from organic matter (~200 meq/100g OM)
    cec_from_om = organic_matter * 2.0

    return cec_from_clay + cec_from_om


def estimate_nitrogen_mineralization(
    organic_matter: float,
    temperature: float,
    moisture_factor: float = 1.0,
    days: int = 120,
) -> float:
    """
    Estimate nitrogen mineralization from organic matter.

    Args:
        organic_matter: Organic matter content (%)
        temperature: Average soil temperature (°C)
        moisture_factor: Soil moisture factor (0-1)
        days: Growing season length (days)

    Returns:
        Estimated mineralizable N (kg/ha)
    """
    # Potentially mineralizable N (kg/ha per % OM)
    pmn_per_om = 25.0

    # Temperature factor (Q10 relationship)
    temp_factor = 2 ** ((temperature - 15) / 10)

    # Mineralization rate constant (fraction per day)
    k = 0.01 * temp_factor * moisture_factor

    # Total mineralization over growing season
    total_pmn = organic_matter * pmn_per_om
    mineralized = total_pmn * (1 - np.exp(-k * days))

    return mineralized
