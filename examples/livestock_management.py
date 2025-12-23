#!/usr/bin/env python3
"""
Livestock Management System Example

This example demonstrates comprehensive livestock management using
the smartagri library, including:

- Animal registration and tracking
- Growth curve modeling and prediction
- Health monitoring and alerts
- Reproduction management
- Feed optimization
- Herd analytics and reporting

Usage:
    python examples/livestock_management.py

Author: SmartAgri Development Team
License: MIT
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnimalSpecies(Enum):
    """Supported livestock species."""
    CATTLE = "cattle"
    SHEEP = "sheep"
    SWINE = "swine"
    POULTRY = "poultry"
    GOAT = "goat"


class HealthStatus(Enum):
    """Animal health status classification."""
    HEALTHY = "healthy"
    MINOR_ISSUE = "minor_issue"
    REQUIRES_ATTENTION = "requires_attention"
    CRITICAL = "critical"


@dataclass
class GrowthParameters:
    """
    Growth model parameters for livestock.

    Stores species-specific growth curve parameters
    for Gompertz, von Bertalanffy, and logistic models.
    """
    species: str
    mature_weight: float      # A - asymptotic weight (kg)
    growth_rate: float        # k - growth rate constant
    inflection_time: float    # t* - time of maximum growth (days)
    birth_weight: float       # initial weight (kg)


@dataclass
class AnimalRecord:
    """
    Complete animal record with all attributes.

    Tracks identification, lineage, health, growth,
    and production information.
    """
    animal_id: str
    species: AnimalSpecies
    breed: str
    sex: str
    birth_date: date
    dam_id: Optional[str] = None
    sire_id: Optional[str] = None
    tag_number: Optional[str] = None
    weight_history: List[Tuple[date, float]] = field(default_factory=list)
    health_records: List[Dict] = field(default_factory=list)
    reproduction_records: List[Dict] = field(default_factory=list)
    current_location: Optional[str] = None
    status: str = "active"

    @property
    def age_days(self) -> int:
        """Calculate age in days."""
        return (date.today() - self.birth_date).days

    @property
    def age_months(self) -> float:
        """Calculate age in months."""
        return self.age_days / 30.44

    @property
    def current_weight(self) -> Optional[float]:
        """Get most recent weight."""
        if self.weight_history:
            return self.weight_history[-1][1]
        return None


class GrowthCurveModels:
    """
    Livestock growth curve modeling.

    Implements multiple growth models for different species
    and production systems. Supports:
    - Gompertz model (general purpose)
    - von Bertalanffy (cattle, fish)
    - Logistic (swine, poultry)
    - Richards (flexible, 4-parameter)
    """

    # Species-specific default parameters
    SPECIES_PARAMS = {
        'cattle_beef': GrowthParameters(
            species='cattle_beef',
            mature_weight=550.0,
            growth_rate=0.0038,
            inflection_time=280,
            birth_weight=35.0,
        ),
        'cattle_dairy': GrowthParameters(
            species='cattle_dairy',
            mature_weight=650.0,
            growth_rate=0.0032,
            inflection_time=320,
            birth_weight=40.0,
        ),
        'sheep': GrowthParameters(
            species='sheep',
            mature_weight=80.0,
            growth_rate=0.008,
            inflection_time=120,
            birth_weight=4.5,
        ),
        'swine': GrowthParameters(
            species='swine',
            mature_weight=120.0,
            growth_rate=0.012,
            inflection_time=150,
            birth_weight=1.5,
        ),
        'poultry_broiler': GrowthParameters(
            species='poultry_broiler',
            mature_weight=4.5,
            growth_rate=0.045,
            inflection_time=28,
            birth_weight=0.042,
        ),
    }

    @staticmethod
    def gompertz(
        t: np.ndarray,
        A: float,
        b: float,
        k: float,
    ) -> np.ndarray:
        """
        Gompertz growth model.

        W(t) = A * exp(-b * exp(-k * t))

        Args:
            t: Time (days)
            A: Asymptotic weight (kg)
            b: Integration constant
            k: Growth rate constant

        Returns:
            Predicted weight at each time point

        Example:
            >>> t = np.arange(0, 365)
            >>> weights = GrowthCurveModels.gompertz(t, A=550, b=4.2, k=0.004)
        """
        return A * np.exp(-b * np.exp(-k * t))

    @staticmethod
    def von_bertalanffy(
        t: np.ndarray,
        A: float,
        k: float,
        t0: float,
    ) -> np.ndarray:
        """
        von Bertalanffy growth model.

        W(t) = A * (1 - exp(-k * (t - t0)))³

        Commonly used for cattle and fish growth.

        Args:
            t: Time (days)
            A: Asymptotic weight (kg)
            k: Growth rate constant
            t0: Time at which weight is zero (days)

        Returns:
            Predicted weight at each time point
        """
        growth = np.maximum(1 - np.exp(-k * (t - t0)), 0)
        return A * np.power(growth, 3)

    @staticmethod
    def logistic(
        t: np.ndarray,
        A: float,
        k: float,
        t_inflection: float,
    ) -> np.ndarray:
        """
        Logistic growth model.

        W(t) = A / (1 + exp(-k * (t - t*)))

        Well-suited for swine and poultry.

        Args:
            t: Time (days)
            A: Asymptotic weight (kg)
            k: Growth rate constant
            t_inflection: Time of inflection point (days)

        Returns:
            Predicted weight at each time point
        """
        return A / (1 + np.exp(-k * (t - t_inflection)))

    @classmethod
    def predict_growth(
        cls,
        species: str,
        age_days: int,
        model: str = "gompertz",
        custom_params: Optional[GrowthParameters] = None,
    ) -> Dict[str, float]:
        """
        Predict weight for given species and age.

        Args:
            species: Species key (e.g., 'cattle_beef', 'swine')
            age_days: Age in days
            model: Model type ('gompertz', 'von_bertalanffy', 'logistic')
            custom_params: Optional custom parameters

        Returns:
            Dictionary with predicted weight and growth metrics

        Example:
            >>> prediction = GrowthCurveModels.predict_growth('cattle_beef', 365)
            >>> print(f"Predicted weight at 1 year: {prediction['weight']:.1f} kg")
        """
        params = custom_params or cls.SPECIES_PARAMS.get(species, cls.SPECIES_PARAMS['cattle_beef'])

        t = np.array([age_days])

        if model == "gompertz":
            b = np.log(params.mature_weight / params.birth_weight)
            weight = cls.gompertz(t, params.mature_weight, b, params.growth_rate)[0]
        elif model == "von_bertalanffy":
            weight = cls.von_bertalanffy(t, params.mature_weight, params.growth_rate, -30)[0]
        else:  # logistic
            weight = cls.logistic(t, params.mature_weight, params.growth_rate * 10, params.inflection_time)[0]

        # Calculate growth rate (kg/day)
        t_next = np.array([age_days + 1])
        if model == "gompertz":
            weight_next = cls.gompertz(t_next, params.mature_weight, b, params.growth_rate)[0]
        elif model == "von_bertalanffy":
            weight_next = cls.von_bertalanffy(t_next, params.mature_weight, params.growth_rate, -30)[0]
        else:
            weight_next = cls.logistic(t_next, params.mature_weight, params.growth_rate * 10, params.inflection_time)[0]

        daily_gain = weight_next - weight

        return {
            'weight': weight,
            'daily_gain': daily_gain,
            'mature_weight': params.mature_weight,
            'percent_mature': weight / params.mature_weight * 100,
            'days_to_market': cls._days_to_market_weight(species, weight),
        }

    @staticmethod
    def _days_to_market_weight(species: str, current_weight: float) -> int:
        """Estimate days to reach market weight."""
        market_weights = {
            'cattle_beef': 550,
            'cattle_dairy': 500,
            'sheep': 50,
            'swine': 110,
            'poultry_broiler': 2.5,
        }
        market_weight = market_weights.get(species, 500)

        if current_weight >= market_weight:
            return 0

        # Simplified linear estimate
        avg_daily_gain = {
            'cattle_beef': 1.2,
            'sheep': 0.25,
            'swine': 0.8,
            'poultry_broiler': 0.06,
        }
        adg = avg_daily_gain.get(species, 1.0)

        return int((market_weight - current_weight) / adg)


class HealthMonitor:
    """
    Livestock health monitoring system.

    Monitors vital signs, detects anomalies, and generates
    health alerts based on species-specific thresholds.
    """

    # Species-specific normal vital ranges
    VITAL_RANGES = {
        'cattle': {
            'temperature': (38.0, 39.5),      # °C
            'heart_rate': (48, 84),           # bpm
            'respiratory_rate': (12, 36),     # breaths/min
            'rumination': (400, 600),         # min/day
        },
        'sheep': {
            'temperature': (38.5, 40.0),
            'heart_rate': (70, 90),
            'respiratory_rate': (12, 25),
            'rumination': (300, 500),
        },
        'swine': {
            'temperature': (38.0, 39.5),
            'heart_rate': (60, 100),
            'respiratory_rate': (15, 25),
        },
        'poultry': {
            'temperature': (40.5, 42.0),
            'heart_rate': (200, 400),
            'respiratory_rate': (15, 30),
        },
    }

    @classmethod
    def assess_vitals(
        cls,
        species: str,
        vitals: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Assess vital signs against normal ranges.

        Args:
            species: Animal species
            vitals: Dictionary of vital measurements

        Returns:
            Assessment results with status and alerts

        Example:
            >>> vitals = {'temperature': 39.8, 'heart_rate': 92}
            >>> assessment = HealthMonitor.assess_vitals('cattle', vitals)
        """
        ranges = cls.VITAL_RANGES.get(species, cls.VITAL_RANGES['cattle'])

        alerts = []
        status = HealthStatus.HEALTHY

        for vital, value in vitals.items():
            if vital in ranges:
                low, high = ranges[vital]
                if value < low:
                    severity = "WARNING" if value > low * 0.9 else "CRITICAL"
                    alerts.append({
                        'vital': vital,
                        'value': value,
                        'status': severity,
                        'message': f"{vital} ({value}) below normal range ({low}-{high})"
                    })
                    if severity == "CRITICAL":
                        status = HealthStatus.CRITICAL
                    elif status != HealthStatus.CRITICAL:
                        status = HealthStatus.REQUIRES_ATTENTION

                elif value > high:
                    severity = "WARNING" if value < high * 1.1 else "CRITICAL"
                    alerts.append({
                        'vital': vital,
                        'value': value,
                        'status': severity,
                        'message': f"{vital} ({value}) above normal range ({low}-{high})"
                    })
                    if severity == "CRITICAL":
                        status = HealthStatus.CRITICAL
                    elif status != HealthStatus.CRITICAL:
                        status = HealthStatus.REQUIRES_ATTENTION

        return {
            'status': status,
            'alerts': alerts,
            'vitals_checked': list(vitals.keys()),
            'timestamp': datetime.now(),
        }

    @staticmethod
    def calculate_health_score(
        vital_assessments: List[Dict],
        weight_trend: float,
        activity_level: float,
    ) -> float:
        """
        Calculate overall health score (0-100).

        Combines vital sign assessments, weight trends,
        and activity levels into composite score.

        Args:
            vital_assessments: List of vital assessments
            weight_trend: Weight change trend (-1 to 1)
            activity_level: Relative activity (0 to 1)

        Returns:
            Health score (0-100)
        """
        score = 100.0

        # Deduct for vital sign alerts
        for assessment in vital_assessments:
            for alert in assessment.get('alerts', []):
                if alert['status'] == 'CRITICAL':
                    score -= 25
                elif alert['status'] == 'WARNING':
                    score -= 10

        # Weight trend component
        if weight_trend < -0.1:  # Losing weight
            score -= 15 * abs(weight_trend)
        elif weight_trend < 0:
            score -= 5

        # Activity component
        if activity_level < 0.5:
            score -= 20 * (0.5 - activity_level)

        return max(0, min(100, score))


class FeedOptimizer:
    """
    Feed ration optimization using linear programming.

    Formulates least-cost rations meeting nutritional
    requirements for different production stages.
    """

    # Nutrient requirements by species and stage (per kg feed)
    REQUIREMENTS = {
        'cattle_growing': {
            'crude_protein': 0.14,    # 14%
            'energy_mcal': 2.8,       # Mcal ME/kg
            'calcium': 0.006,         # 0.6%
            'phosphorus': 0.003,      # 0.3%
            'fiber_min': 0.15,        # minimum 15%
            'fiber_max': 0.35,        # maximum 35%
        },
        'cattle_finishing': {
            'crude_protein': 0.12,
            'energy_mcal': 3.0,
            'calcium': 0.004,
            'phosphorus': 0.003,
            'fiber_min': 0.10,
            'fiber_max': 0.25,
        },
        'swine_growing': {
            'crude_protein': 0.18,
            'energy_mcal': 3.3,
            'lysine': 0.011,
            'calcium': 0.007,
            'phosphorus': 0.006,
        },
    }

    # Feed ingredient database
    INGREDIENTS = {
        'corn': {
            'crude_protein': 0.088,
            'energy_mcal': 3.3,
            'calcium': 0.0003,
            'phosphorus': 0.003,
            'fiber': 0.022,
            'cost_per_kg': 0.20,
        },
        'soybean_meal': {
            'crude_protein': 0.44,
            'energy_mcal': 3.1,
            'calcium': 0.003,
            'phosphorus': 0.007,
            'fiber': 0.06,
            'cost_per_kg': 0.45,
        },
        'corn_silage': {
            'crude_protein': 0.08,
            'energy_mcal': 2.3,
            'calcium': 0.003,
            'phosphorus': 0.002,
            'fiber': 0.28,
            'cost_per_kg': 0.05,
        },
        'alfalfa_hay': {
            'crude_protein': 0.18,
            'energy_mcal': 2.2,
            'calcium': 0.014,
            'phosphorus': 0.002,
            'fiber': 0.30,
            'cost_per_kg': 0.15,
        },
        'mineral_premix': {
            'crude_protein': 0.0,
            'energy_mcal': 0.0,
            'calcium': 0.20,
            'phosphorus': 0.10,
            'fiber': 0.0,
            'cost_per_kg': 1.50,
        },
    }

    @classmethod
    def formulate_ration(
        cls,
        species_stage: str,
        available_ingredients: List[str] = None,
        daily_intake_kg: float = 10.0,
    ) -> Dict:
        """
        Formulate least-cost ration meeting requirements.

        Uses simplified linear programming approach to find
        optimal ingredient proportions.

        Args:
            species_stage: Species and stage (e.g., 'cattle_growing')
            available_ingredients: List of available ingredients
            daily_intake_kg: Target daily intake (kg)

        Returns:
            Ration formulation with proportions and analysis

        Example:
            >>> ration = FeedOptimizer.formulate_ration('cattle_growing')
            >>> print(f"Cost per day: ${ration['cost_per_day']:.2f}")
        """
        if available_ingredients is None:
            available_ingredients = list(cls.INGREDIENTS.keys())

        requirements = cls.REQUIREMENTS.get(species_stage, cls.REQUIREMENTS['cattle_growing'])

        # Simplified optimization using heuristics
        # (In production, use scipy.optimize.linprog or similar)

        # Start with base formulation
        if 'cattle' in species_stage:
            formulation = {
                'corn': 0.40,
                'corn_silage': 0.30,
                'soybean_meal': 0.15,
                'alfalfa_hay': 0.12,
                'mineral_premix': 0.03,
            }
        else:  # swine
            formulation = {
                'corn': 0.70,
                'soybean_meal': 0.27,
                'mineral_premix': 0.03,
            }

        # Filter to available ingredients
        formulation = {k: v for k, v in formulation.items() if k in available_ingredients}

        # Normalize to sum to 1
        total = sum(formulation.values())
        formulation = {k: v / total for k, v in formulation.items()}

        # Calculate ration analysis
        analysis = {
            'crude_protein': 0,
            'energy_mcal': 0,
            'calcium': 0,
            'phosphorus': 0,
            'fiber': 0,
        }

        cost_per_kg = 0
        for ingredient, proportion in formulation.items():
            ing_data = cls.INGREDIENTS[ingredient]
            for nutrient in analysis:
                if nutrient in ing_data:
                    analysis[nutrient] += proportion * ing_data[nutrient]
            cost_per_kg += proportion * ing_data['cost_per_kg']

        # Check if requirements are met
        deficiencies = []
        for nutrient, required in requirements.items():
            if nutrient in analysis:
                if analysis[nutrient] < required * 0.95:
                    deficiencies.append({
                        'nutrient': nutrient,
                        'required': required,
                        'provided': analysis[nutrient],
                        'deficit': required - analysis[nutrient],
                    })

        return {
            'formulation': formulation,
            'analysis': analysis,
            'cost_per_kg': cost_per_kg,
            'cost_per_day': cost_per_kg * daily_intake_kg,
            'daily_intake_kg': daily_intake_kg,
            'requirements_met': len(deficiencies) == 0,
            'deficiencies': deficiencies,
            'ingredients_kg_per_day': {k: v * daily_intake_kg for k, v in formulation.items()},
        }


class HerdManager:
    """
    Comprehensive herd management system.

    Manages animal inventory, tracking, and analytics
    for complete farm operation visibility.
    """

    def __init__(self):
        """Initialize herd manager."""
        self.animals: Dict[str, AnimalRecord] = {}
        self.groups: Dict[str, List[str]] = {}

    def register_animal(
        self,
        species: AnimalSpecies,
        breed: str,
        sex: str,
        birth_date: date,
        dam_id: Optional[str] = None,
        sire_id: Optional[str] = None,
        birth_weight: Optional[float] = None,
    ) -> AnimalRecord:
        """
        Register a new animal in the herd.

        Args:
            species: Animal species
            breed: Breed name
            sex: Sex ('M' or 'F')
            birth_date: Date of birth
            dam_id: Dam's ID
            sire_id: Sire's ID
            birth_weight: Birth weight (kg)

        Returns:
            New AnimalRecord

        Example:
            >>> animal = manager.register_animal(
            ...     AnimalSpecies.CATTLE, 'Angus', 'F',
            ...     date(2024, 3, 15), birth_weight=38.0
            ... )
        """
        animal_id = str(uuid.uuid4())[:8].upper()

        animal = AnimalRecord(
            animal_id=animal_id,
            species=species,
            breed=breed,
            sex=sex,
            birth_date=birth_date,
            dam_id=dam_id,
            sire_id=sire_id,
        )

        if birth_weight:
            animal.weight_history.append((birth_date, birth_weight))

        self.animals[animal_id] = animal
        logger.info(f"Registered animal {animal_id}: {species.value} {breed}")

        return animal

    def record_weight(
        self,
        animal_id: str,
        weight: float,
        record_date: date = None,
    ) -> None:
        """Record weight measurement."""
        if animal_id in self.animals:
            record_date = record_date or date.today()
            self.animals[animal_id].weight_history.append((record_date, weight))

    def record_health_event(
        self,
        animal_id: str,
        event_type: str,
        description: str,
        treatment: Optional[str] = None,
    ) -> None:
        """Record health event or treatment."""
        if animal_id in self.animals:
            self.animals[animal_id].health_records.append({
                'date': date.today(),
                'type': event_type,
                'description': description,
                'treatment': treatment,
            })

    def create_group(self, name: str, animal_ids: List[str] = None) -> str:
        """Create animal group (pen, pasture, etc.)."""
        group_id = name.lower().replace(' ', '_')
        self.groups[group_id] = animal_ids or []
        return group_id

    def get_herd_summary(self) -> Dict:
        """
        Generate comprehensive herd summary.

        Returns:
            Dictionary with herd statistics and analytics
        """
        active_animals = [a for a in self.animals.values() if a.status == 'active']

        by_species = {}
        by_sex = {}
        total_weight = 0
        weight_count = 0

        for animal in active_animals:
            species = animal.species.value
            by_species[species] = by_species.get(species, 0) + 1
            by_sex[animal.sex] = by_sex.get(animal.sex, 0) + 1

            if animal.current_weight:
                total_weight += animal.current_weight
                weight_count += 1

        return {
            'total_active': len(active_animals),
            'by_species': by_species,
            'by_sex': by_sex,
            'average_weight': total_weight / weight_count if weight_count else 0,
            'groups': len(self.groups),
            'total_registered': len(self.animals),
        }

    def get_growth_analysis(self, animal_id: str) -> Dict:
        """
        Analyze growth performance for specific animal.

        Returns growth metrics including ADG, feed efficiency,
        and comparison to breed standards.
        """
        animal = self.animals.get(animal_id)
        if not animal or len(animal.weight_history) < 2:
            return {'error': 'Insufficient data'}

        weights = sorted(animal.weight_history, key=lambda x: x[0])

        # Calculate ADG over entire period
        first_date, first_weight = weights[0]
        last_date, last_weight = weights[-1]
        days = (last_date - first_date).days

        if days == 0:
            return {'error': 'Insufficient time range'}

        adg = (last_weight - first_weight) / days

        # Get predicted weight from growth model
        species_key = f"{animal.species.value}_beef" if animal.species == AnimalSpecies.CATTLE else animal.species.value
        prediction = GrowthCurveModels.predict_growth(species_key, animal.age_days)

        return {
            'animal_id': animal_id,
            'current_weight': last_weight,
            'birth_weight': first_weight if animal.birth_date == first_date else None,
            'adg': adg,
            'age_days': animal.age_days,
            'predicted_weight': prediction['weight'],
            'performance_ratio': last_weight / prediction['weight'] if prediction['weight'] else 0,
            'days_to_market': prediction['days_to_market'],
            'percent_mature': prediction['percent_mature'],
        }


def run_demonstration():
    """
    Run comprehensive livestock management demonstration.

    Demonstrates all major features of the livestock
    management system.
    """
    print("=" * 70)
    print("Smart Agriculture - Livestock Management System Demo")
    print("=" * 70)
    print()

    # Initialize herd manager
    manager = HerdManager()

    # Register some animals
    print("Registering animals...")
    print("-" * 50)

    # Cattle
    cow1 = manager.register_animal(
        AnimalSpecies.CATTLE, 'Angus', 'F',
        date(2022, 4, 10), birth_weight=38.0
    )
    manager.record_weight(cow1.animal_id, 280.0, date(2023, 4, 10))
    manager.record_weight(cow1.animal_id, 450.0, date(2024, 4, 10))
    manager.record_weight(cow1.animal_id, 520.0, date(2024, 10, 15))

    cow2 = manager.register_animal(
        AnimalSpecies.CATTLE, 'Hereford', 'M',
        date(2023, 6, 20), birth_weight=42.0
    )
    manager.record_weight(cow2.animal_id, 180.0, date(2023, 12, 20))
    manager.record_weight(cow2.animal_id, 350.0, date(2024, 6, 20))

    # Sheep
    sheep1 = manager.register_animal(
        AnimalSpecies.SHEEP, 'Suffolk', 'F',
        date(2024, 2, 1), birth_weight=4.2
    )
    manager.record_weight(sheep1.animal_id, 35.0, date(2024, 8, 1))

    # Swine
    pig1 = manager.register_animal(
        AnimalSpecies.SWINE, 'Yorkshire', 'M',
        date(2024, 5, 1), birth_weight=1.4
    )
    manager.record_weight(pig1.animal_id, 85.0, date(2024, 9, 1))

    print(f"  • Registered {len(manager.animals)} animals")

    # Herd summary
    print("\nHerd Summary:")
    print("-" * 50)
    summary = manager.get_herd_summary()
    print(f"  • Total Active: {summary['total_active']}")
    print(f"  • By Species: {summary['by_species']}")
    print(f"  • By Sex: {summary['by_sex']}")
    print(f"  • Average Weight: {summary['average_weight']:.1f} kg")

    # Growth analysis
    print("\nGrowth Analysis - Angus Cow:")
    print("-" * 50)
    growth = manager.get_growth_analysis(cow1.animal_id)
    print(f"  • Current Weight: {growth['current_weight']:.1f} kg")
    print(f"  • Age: {growth['age_days']} days ({growth['age_days']/365:.1f} years)")
    print(f"  • Average Daily Gain: {growth['adg']:.2f} kg/day")
    print(f"  • Predicted Weight (model): {growth['predicted_weight']:.1f} kg")
    print(f"  • Performance vs Predicted: {growth['performance_ratio']*100:.1f}%")
    print(f"  • Mature Weight %: {growth['percent_mature']:.1f}%")

    # Growth curve predictions
    print("\nGrowth Curve Predictions (Beef Cattle):")
    print("-" * 50)
    for age in [180, 365, 540, 730]:
        pred = GrowthCurveModels.predict_growth('cattle_beef', age)
        print(f"  Age {age:3d} days: {pred['weight']:6.1f} kg, "
              f"ADG: {pred['daily_gain']:.2f} kg/day, "
              f"{pred['percent_mature']:.1f}% mature")

    # Health monitoring
    print("\nHealth Monitoring:")
    print("-" * 50)
    vitals = {
        'temperature': 39.2,
        'heart_rate': 72,
        'respiratory_rate': 28,
        'rumination': 480,
    }
    assessment = HealthMonitor.assess_vitals('cattle', vitals)
    print(f"  • Status: {assessment['status'].value}")
    print(f"  • Vitals Checked: {', '.join(assessment['vitals_checked'])}")
    if assessment['alerts']:
        print(f"  • Alerts: {len(assessment['alerts'])}")
        for alert in assessment['alerts']:
            print(f"    - {alert['message']}")
    else:
        print("  • No alerts - all vitals normal")

    # Abnormal vitals test
    print("\nHealth Alert Test (elevated temperature):")
    print("-" * 50)
    abnormal_vitals = {'temperature': 40.5, 'heart_rate': 95, 'respiratory_rate': 45}
    alert_assessment = HealthMonitor.assess_vitals('cattle', abnormal_vitals)
    print(f"  • Status: {alert_assessment['status'].value}")
    for alert in alert_assessment['alerts']:
        print(f"  • {alert['status']}: {alert['message']}")

    health_score = HealthMonitor.calculate_health_score(
        [assessment, alert_assessment],
        weight_trend=0.1,
        activity_level=0.8
    )
    print(f"  • Health Score: {health_score:.1f}/100")

    # Feed optimization
    print("\nFeed Ration Optimization:")
    print("-" * 50)

    print("\n  Growing Cattle Ration:")
    ration = FeedOptimizer.formulate_ration('cattle_growing', daily_intake_kg=12.0)
    print(f"    Formulation:")
    for ingredient, proportion in ration['formulation'].items():
        kg_per_day = ration['ingredients_kg_per_day'][ingredient]
        print(f"      • {ingredient}: {proportion*100:.1f}% ({kg_per_day:.2f} kg/day)")
    print(f"    Analysis:")
    print(f"      • Crude Protein: {ration['analysis']['crude_protein']*100:.1f}%")
    print(f"      • Energy: {ration['analysis']['energy_mcal']:.2f} Mcal/kg")
    print(f"    Cost: ${ration['cost_per_kg']:.2f}/kg (${ration['cost_per_day']:.2f}/day)")
    print(f"    Requirements Met: {'Yes' if ration['requirements_met'] else 'No'}")

    print("\n  Finishing Cattle Ration:")
    ration_finish = FeedOptimizer.formulate_ration('cattle_finishing', daily_intake_kg=10.0)
    print(f"    Cost: ${ration_finish['cost_per_kg']:.2f}/kg (${ration_finish['cost_per_day']:.2f}/day)")

    # Record health event
    print("\nRecording Health Events:")
    print("-" * 50)
    manager.record_health_event(
        cow1.animal_id,
        'vaccination',
        'Annual BVD vaccination',
        treatment='BVD vaccine 2ml IM'
    )
    print(f"  • Recorded vaccination for {cow1.animal_id}")
    print(f"  • Health records: {len(manager.animals[cow1.animal_id].health_records)}")

    # Create groups
    print("\nGroup Management:")
    print("-" * 50)
    manager.create_group('Breeding Herd', [cow1.animal_id])
    manager.create_group('Finishing Pen', [cow2.animal_id])
    manager.create_group('Sheep Flock', [sheep1.animal_id])
    print(f"  • Created {len(manager.groups)} groups")
    for group_name, members in manager.groups.items():
        print(f"    - {group_name}: {len(members)} animals")

    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


def main():
    """Main entry point."""
    run_demonstration()


if __name__ == "__main__":
    main()
