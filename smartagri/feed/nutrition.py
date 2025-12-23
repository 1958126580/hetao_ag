"""
Nutrition Calculation Module

Nutrient requirement calculations and feed analysis:
- Species-specific nutrient requirements
- Feed composition analysis
- Nutrient balance assessment
- Deficiency detection

Example:
    >>> calculator = NutrientCalculator(species="cattle")
    >>> requirements = calculator.calculate_requirements(
    ...     weight=500, production="lactating", milk_yield=30
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProductionStage(Enum):
    """Animal production stage."""
    MAINTENANCE = "maintenance"
    GROWTH = "growth"
    LACTATION = "lactation"
    GESTATION = "gestation"
    FINISHING = "finishing"
    BREEDING = "breeding"


@dataclass
class NutrientRequirements:
    """
    Daily nutrient requirements.

    Attributes:
        dry_matter: Dry matter intake (kg/day)
        energy_me: Metabolizable energy (MJ/day)
        energy_ne: Net energy (MJ/day)
        crude_protein: Crude protein (kg/day)
        rumen_degradable_protein: RDP (kg/day)
        rumen_undegradable_protein: RUP (kg/day)
        ndf: Neutral detergent fiber (kg/day)
        calcium: Calcium (g/day)
        phosphorus: Phosphorus (g/day)
        magnesium: Magnesium (g/day)
        potassium: Potassium (g/day)
        sodium: Sodium (g/day)
        sulfur: Sulfur (g/day)
        vitamin_a: Vitamin A (IU/day)
        vitamin_d: Vitamin D (IU/day)
        vitamin_e: Vitamin E (IU/day)
    """
    dry_matter: float = 0.0
    energy_me: float = 0.0
    energy_ne: float = 0.0
    crude_protein: float = 0.0
    rumen_degradable_protein: float = 0.0
    rumen_undegradable_protein: float = 0.0
    ndf: float = 0.0
    calcium: float = 0.0
    phosphorus: float = 0.0
    magnesium: float = 0.0
    potassium: float = 0.0
    sodium: float = 0.0
    sulfur: float = 0.0
    vitamin_a: float = 0.0
    vitamin_d: float = 0.0
    vitamin_e: float = 0.0


@dataclass
class FeedAnalysis:
    """
    Feed composition analysis.

    Attributes:
        name: Feed name
        dry_matter: Dry matter (%)
        crude_protein: Crude protein (% DM)
        energy_me: Metabolizable energy (MJ/kg DM)
        ndf: Neutral detergent fiber (% DM)
        adf: Acid detergent fiber (% DM)
        fat: Crude fat (% DM)
        ash: Ash (% DM)
        calcium: Calcium (% DM)
        phosphorus: Phosphorus (% DM)
        tdn: Total digestible nutrients (% DM)
        rdp: Rumen degradable protein (% CP)
        cost_per_kg: Cost per kg as fed
    """
    name: str
    dry_matter: float = 90.0
    crude_protein: float = 10.0
    energy_me: float = 10.0
    ndf: float = 40.0
    adf: float = 25.0
    fat: float = 3.0
    ash: float = 8.0
    calcium: float = 0.5
    phosphorus: float = 0.3
    tdn: float = 65.0
    rdp: float = 65.0
    cost_per_kg: float = 0.20


@dataclass
class NutrientBalance:
    """
    Nutrient balance assessment.

    Attributes:
        nutrient: Nutrient name
        required: Required amount
        supplied: Supplied amount
        balance: Surplus/deficit
        balance_pct: Balance as percentage
        status: Balance status
    """
    nutrient: str
    required: float
    supplied: float
    balance: float
    balance_pct: float
    status: str  # "adequate", "deficient", "excess"


# Standard feed composition database
FEED_DATABASE = {
    "corn_grain": FeedAnalysis(
        name="Corn Grain",
        dry_matter=88.0,
        crude_protein=9.0,
        energy_me=13.5,
        ndf=9.0,
        adf=3.5,
        fat=4.0,
        calcium=0.03,
        phosphorus=0.30,
        tdn=88.0,
        rdp=50.0,
        cost_per_kg=0.25,
    ),
    "soybean_meal": FeedAnalysis(
        name="Soybean Meal (48% CP)",
        dry_matter=90.0,
        crude_protein=48.0,
        energy_me=13.0,
        ndf=14.0,
        adf=10.0,
        fat=1.5,
        calcium=0.35,
        phosphorus=0.70,
        tdn=84.0,
        rdp=70.0,
        cost_per_kg=0.45,
    ),
    "alfalfa_hay": FeedAnalysis(
        name="Alfalfa Hay",
        dry_matter=90.0,
        crude_protein=18.0,
        energy_me=9.5,
        ndf=42.0,
        adf=32.0,
        fat=2.5,
        calcium=1.40,
        phosphorus=0.25,
        tdn=58.0,
        rdp=75.0,
        cost_per_kg=0.18,
    ),
    "grass_hay": FeedAnalysis(
        name="Grass Hay",
        dry_matter=88.0,
        crude_protein=10.0,
        energy_me=8.5,
        ndf=60.0,
        adf=38.0,
        fat=2.0,
        calcium=0.45,
        phosphorus=0.25,
        tdn=52.0,
        rdp=65.0,
        cost_per_kg=0.12,
    ),
    "corn_silage": FeedAnalysis(
        name="Corn Silage",
        dry_matter=35.0,
        crude_protein=8.0,
        energy_me=10.8,
        ndf=45.0,
        adf=28.0,
        fat=3.0,
        calcium=0.25,
        phosphorus=0.22,
        tdn=68.0,
        rdp=55.0,
        cost_per_kg=0.05,
    ),
    "wheat_middlings": FeedAnalysis(
        name="Wheat Middlings",
        dry_matter=89.0,
        crude_protein=17.0,
        energy_me=12.0,
        ndf=38.0,
        adf=12.0,
        fat=4.0,
        calcium=0.12,
        phosphorus=0.95,
        tdn=75.0,
        rdp=75.0,
        cost_per_kg=0.20,
    ),
    "distillers_grains": FeedAnalysis(
        name="Distillers Grains (Dried)",
        dry_matter=90.0,
        crude_protein=27.0,
        energy_me=13.0,
        ndf=38.0,
        adf=18.0,
        fat=10.0,
        calcium=0.08,
        phosphorus=0.75,
        tdn=85.0,
        rdp=45.0,
        cost_per_kg=0.22,
    ),
    "cottonseed_meal": FeedAnalysis(
        name="Cottonseed Meal",
        dry_matter=91.0,
        crude_protein=41.0,
        energy_me=11.5,
        ndf=29.0,
        adf=20.0,
        fat=2.0,
        calcium=0.18,
        phosphorus=1.10,
        tdn=72.0,
        rdp=55.0,
        cost_per_kg=0.35,
    ),
}


class NutrientCalculator:
    """
    Calculate nutrient requirements for livestock.

    Uses NRC (National Research Council) equations
    for species-specific nutrient requirements.

    Example:
        >>> calc = NutrientCalculator(species="cattle")
        >>> req = calc.calculate_requirements(weight=600, production="lactating")
    """

    def __init__(self, species: str = "cattle"):
        """
        Initialize nutrient calculator.

        Args:
            species: Animal species
        """
        self.species = species.lower()

    def calculate_requirements(
        self,
        weight: float,
        production: str = "maintenance",
        adg: float = 0.0,
        milk_yield: float = 0.0,
        milk_fat: float = 3.5,
        milk_protein: float = 3.2,
        days_pregnant: int = 0,
        mature_weight: float = 0.0,
    ) -> NutrientRequirements:
        """
        Calculate nutrient requirements.

        Args:
            weight: Body weight (kg)
            production: Production stage
            adg: Average daily gain (kg/day)
            milk_yield: Milk production (kg/day)
            milk_fat: Milk fat percentage
            milk_protein: Milk protein percentage
            days_pregnant: Days of pregnancy
            mature_weight: Expected mature weight

        Returns:
            NutrientRequirements object
        """
        if self.species == "cattle":
            return self._cattle_requirements(
                weight, production, adg, milk_yield,
                milk_fat, milk_protein, days_pregnant, mature_weight
            )
        elif self.species == "sheep":
            return self._sheep_requirements(weight, production, adg)
        elif self.species == "swine":
            return self._swine_requirements(weight, production, adg)
        else:
            return self._cattle_requirements(
                weight, production, adg, milk_yield,
                milk_fat, milk_protein, days_pregnant, mature_weight
            )

    def _cattle_requirements(
        self,
        weight: float,
        production: str,
        adg: float,
        milk_yield: float,
        milk_fat: float,
        milk_protein: float,
        days_pregnant: int,
        mature_weight: float,
    ) -> NutrientRequirements:
        """Calculate cattle nutrient requirements (NRC 2001)."""
        req = NutrientRequirements()

        # Metabolic body weight
        mbw = weight ** 0.75

        # Maintenance requirements
        ne_m = 0.322 * mbw  # Net energy for maintenance (MJ/day)
        mp_m = 3.8 * mbw / 1000  # Metabolizable protein (kg/day)

        # Dry matter intake prediction
        if milk_yield > 0:
            # Lactating cow DMI (NRC 2001)
            fcm = 0.4 * milk_yield + 15 * milk_fat / 100 * milk_yield
            dmi = (0.0185 * weight + 0.305 * fcm)
        elif adg > 0:
            # Growing cattle DMI
            dmi = weight * (0.016 * weight / mature_weight + 0.02) if mature_weight > 0 \
                else weight * 0.025
        else:
            # Maintenance DMI
            dmi = weight * 0.02

        req.dry_matter = dmi

        # Energy requirements
        if production == "lactating" and milk_yield > 0:
            # Milk energy (NRC 2001)
            ne_l = (0.36 + 0.0969 * milk_fat) * milk_yield
            total_ne = ne_m + ne_l
        elif production == "growth" and adg > 0:
            # Growth energy
            re = 0.0635 * weight ** 0.75 * adg ** 1.097
            ne_g = re / 0.64  # Efficiency
            total_ne = ne_m + ne_g
        elif production == "gestation" and days_pregnant > 200:
            # Pregnancy energy
            calf_bw = 40  # Expected calf weight
            ne_p = 0.00159 * days_pregnant - 0.0352 * calf_bw
            total_ne = ne_m + ne_p
        else:
            total_ne = ne_m

        req.energy_ne = total_ne
        req.energy_me = total_ne / 0.64  # Convert NE to ME

        # Protein requirements
        if production == "lactating" and milk_yield > 0:
            mp_l = milk_yield * milk_protein / 100 / 0.67
            total_mp = mp_m + mp_l
        elif production == "growth" and adg > 0:
            mp_g = adg * 0.268 / 0.49
            total_mp = mp_m + mp_g
        else:
            total_mp = mp_m

        req.crude_protein = total_mp / 0.64  # Convert MP to CP
        req.rumen_degradable_protein = req.crude_protein * 0.65
        req.rumen_undegradable_protein = req.crude_protein * 0.35

        # Fiber requirements
        req.ndf = dmi * 0.28  # Minimum 28% NDF

        # Mineral requirements (g/day)
        req.calcium = 0.031 * weight + (milk_yield * 1.22 if milk_yield > 0 else 0)
        req.phosphorus = 0.028 * weight + (milk_yield * 0.90 if milk_yield > 0 else 0)
        req.magnesium = 0.003 * weight + (milk_yield * 0.12 if milk_yield > 0 else 0)
        req.potassium = 0.038 * weight
        req.sodium = 0.006 * weight
        req.sulfur = 0.002 * weight

        # Vitamin requirements
        req.vitamin_a = 30 * weight
        req.vitamin_d = 10 * weight
        req.vitamin_e = 0.8 * weight

        return req

    def _sheep_requirements(
        self,
        weight: float,
        production: str,
        adg: float,
    ) -> NutrientRequirements:
        """Calculate sheep nutrient requirements (NRC 2007)."""
        req = NutrientRequirements()

        mbw = weight ** 0.75

        # DMI
        if adg > 0:
            dmi = weight * 0.035
        else:
            dmi = weight * 0.025

        req.dry_matter = dmi

        # Energy
        ne_m = 0.236 * mbw
        if adg > 0:
            ne_g = (0.276 * adg + 0.006) * weight
            req.energy_ne = ne_m + ne_g
        else:
            req.energy_ne = ne_m

        req.energy_me = req.energy_ne / 0.60

        # Protein
        cp_m = 0.003 * weight
        if adg > 0:
            cp_g = 0.268 * adg
            req.crude_protein = cp_m + cp_g
        else:
            req.crude_protein = cp_m

        # Minerals (g/day)
        req.calcium = 0.004 * weight
        req.phosphorus = 0.003 * weight

        return req

    def _swine_requirements(
        self,
        weight: float,
        production: str,
        adg: float,
    ) -> NutrientRequirements:
        """Calculate swine nutrient requirements (NRC 2012)."""
        req = NutrientRequirements()

        # DMI (feed intake)
        if weight < 25:
            dmi = weight * 0.05
        elif weight < 60:
            dmi = weight * 0.04
        else:
            dmi = weight * 0.03

        req.dry_matter = dmi

        # Energy (ME, MJ/day)
        me_m = 0.44 * weight ** 0.60
        if adg > 0:
            me_g = 23 * adg
            req.energy_me = me_m + me_g
        else:
            req.energy_me = me_m

        # Protein - varies by weight
        if weight < 20:
            cp_pct = 0.22
        elif weight < 50:
            cp_pct = 0.18
        else:
            cp_pct = 0.14

        req.crude_protein = dmi * cp_pct

        # Lysine requirement (primary limiting amino acid)
        lysine_pct = 0.011 * weight + 1.1 if weight < 100 else 0.7

        # Minerals
        req.calcium = dmi * 0.007
        req.phosphorus = dmi * 0.006

        return req

    def estimate_dmi(
        self,
        weight: float,
        production: str = "maintenance",
        **kwargs,
    ) -> float:
        """
        Estimate dry matter intake.

        Args:
            weight: Body weight (kg)
            production: Production stage
            **kwargs: Additional parameters

        Returns:
            Estimated DMI (kg/day)
        """
        req = self.calculate_requirements(weight, production, **kwargs)
        return req.dry_matter


class NutrientAnalyzer:
    """
    Analyze nutrient content and balance.

    Evaluates feed rations against requirements
    and identifies deficiencies or excesses.

    Example:
        >>> analyzer = NutrientAnalyzer()
        >>> balance = analyzer.evaluate_balance(requirements, ration)
    """

    def __init__(self):
        """Initialize nutrient analyzer."""
        self._tolerance = {
            "energy": 0.05,  # 5% tolerance
            "protein": 0.10,
            "minerals": 0.15,
            "vitamins": 0.20,
        }

    def get_feed_analysis(self, feed_name: str) -> Optional[FeedAnalysis]:
        """Get feed analysis from database."""
        key = feed_name.lower().replace(" ", "_")
        return FEED_DATABASE.get(key)

    def calculate_ration_nutrients(
        self,
        feeds: Dict[str, float],  # feed_name: amount_kg
    ) -> Dict[str, float]:
        """
        Calculate total nutrients from ration.

        Args:
            feeds: Dict of feed name to amount (kg as-fed)

        Returns:
            Dict of nutrient values
        """
        totals = {
            "dry_matter": 0.0,
            "crude_protein": 0.0,
            "energy_me": 0.0,
            "ndf": 0.0,
            "calcium": 0.0,
            "phosphorus": 0.0,
            "cost": 0.0,
        }

        for feed_name, amount in feeds.items():
            analysis = self.get_feed_analysis(feed_name)
            if analysis is None:
                logger.warning(f"Unknown feed: {feed_name}")
                continue

            dm = amount * analysis.dry_matter / 100

            totals["dry_matter"] += dm
            totals["crude_protein"] += dm * analysis.crude_protein / 100
            totals["energy_me"] += dm * analysis.energy_me
            totals["ndf"] += dm * analysis.ndf / 100
            totals["calcium"] += dm * analysis.calcium / 100 * 1000  # g
            totals["phosphorus"] += dm * analysis.phosphorus / 100 * 1000  # g
            totals["cost"] += amount * analysis.cost_per_kg

        return totals

    def evaluate_balance(
        self,
        requirements: NutrientRequirements,
        feeds: Dict[str, float],
    ) -> List[NutrientBalance]:
        """
        Evaluate nutrient balance.

        Args:
            requirements: Nutrient requirements
            feeds: Ration composition

        Returns:
            List of NutrientBalance assessments
        """
        supplied = self.calculate_ration_nutrients(feeds)
        balances = []

        # Check each nutrient
        checks = [
            ("Dry Matter", requirements.dry_matter, supplied.get("dry_matter", 0)),
            ("Crude Protein", requirements.crude_protein, supplied.get("crude_protein", 0)),
            ("Energy (ME)", requirements.energy_me, supplied.get("energy_me", 0)),
            ("NDF", requirements.ndf, supplied.get("ndf", 0)),
            ("Calcium", requirements.calcium, supplied.get("calcium", 0)),
            ("Phosphorus", requirements.phosphorus, supplied.get("phosphorus", 0)),
        ]

        for name, required, actual in checks:
            if required > 0:
                balance = actual - required
                balance_pct = balance / required * 100

                if balance_pct < -10:
                    status = "deficient"
                elif balance_pct > 20:
                    status = "excess"
                else:
                    status = "adequate"

                balances.append(NutrientBalance(
                    nutrient=name,
                    required=required,
                    supplied=actual,
                    balance=balance,
                    balance_pct=balance_pct,
                    status=status,
                ))

        return balances

    def identify_deficiencies(
        self,
        balances: List[NutrientBalance],
    ) -> List[Dict[str, Any]]:
        """
        Identify nutrient deficiencies.

        Args:
            balances: Nutrient balance list

        Returns:
            List of deficiency details
        """
        deficiencies = []

        for balance in balances:
            if balance.status == "deficient":
                severity = "mild" if balance.balance_pct > -20 else \
                          "moderate" if balance.balance_pct > -40 else "severe"

                deficiencies.append({
                    "nutrient": balance.nutrient,
                    "deficit": abs(balance.balance),
                    "deficit_pct": abs(balance.balance_pct),
                    "severity": severity,
                    "recommendation": self._get_deficiency_recommendation(
                        balance.nutrient
                    ),
                })

        return deficiencies

    def _get_deficiency_recommendation(
        self,
        nutrient: str,
    ) -> str:
        """Get recommendation for nutrient deficiency."""
        recommendations = {
            "Crude Protein": "Add soybean meal or other protein supplement",
            "Energy (ME)": "Add corn grain or other energy source",
            "Calcium": "Add limestone or calcium supplement",
            "Phosphorus": "Add dicalcium phosphate or monosodium phosphate",
            "NDF": "Add more forage to the ration",
        }
        return recommendations.get(nutrient, "Consult nutritionist")

    def calculate_ca_p_ratio(
        self,
        calcium: float,
        phosphorus: float,
    ) -> Dict[str, Any]:
        """
        Evaluate calcium:phosphorus ratio.

        Args:
            calcium: Calcium content
            phosphorus: Phosphorus content

        Returns:
            Dict with ratio analysis
        """
        if phosphorus <= 0:
            return {"status": "error", "message": "Phosphorus cannot be zero"}

        ratio = calcium / phosphorus

        # Ideal range is 1.5:1 to 2:1
        if 1.5 <= ratio <= 2.0:
            status = "optimal"
        elif 1.2 <= ratio < 1.5 or 2.0 < ratio <= 2.5:
            status = "acceptable"
        else:
            status = "imbalanced"

        return {
            "ratio": ratio,
            "status": status,
            "optimal_range": "1.5:1 to 2:1",
            "recommendation": "Adjust calcium or phosphorus" if status == "imbalanced" else "",
        }
