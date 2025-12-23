#!/usr/bin/env python3
"""
Integrated Farm Management System Example

This example demonstrates a complete farm management system
integrating all smartagri library modules:

- Crop production planning and monitoring
- Livestock management and tracking
- Weather data integration
- Irrigation scheduling
- Feed management
- Health monitoring
- Machine learning predictions
- Financial analysis

This represents a production-ready farm management
application architecture.

Usage:
    python examples/farm_integration.py

Author: SmartAgri Development Team
License: MIT
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
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


# ==============================================================================
# CORE DATA STRUCTURES
# ==============================================================================

@dataclass
class FarmConfiguration:
    """
    Master farm configuration.

    Stores all farm-level settings and parameters
    used across integrated modules.
    """
    farm_id: str
    name: str
    location: Tuple[float, float]  # (latitude, longitude)
    elevation: float               # meters
    total_area_ha: float
    timezone: str = "UTC"
    currency: str = "USD"
    measurement_system: str = "metric"


@dataclass
class FieldRecord:
    """Individual field/paddock record."""
    field_id: str
    name: str
    area_ha: float
    soil_type: str
    current_crop: Optional[str] = None
    planting_date: Optional[date] = None
    irrigation_zone: Optional[str] = None
    field_history: List[Dict] = field(default_factory=list)


@dataclass
class AnimalRecord:
    """Individual animal record."""
    animal_id: str
    species: str
    breed: str
    sex: str
    birth_date: date
    current_weight: Optional[float] = None
    current_location: Optional[str] = None
    status: str = "active"


@dataclass
class WeatherObservation:
    """Daily weather observation."""
    date: date
    t_max: float
    t_min: float
    precipitation: float
    humidity: float
    wind_speed: float
    solar_radiation: float


@dataclass
class FinancialRecord:
    """Financial transaction record."""
    transaction_id: str
    date: date
    category: str
    description: str
    amount: float
    type: str  # 'income' or 'expense'


# ==============================================================================
# INTEGRATED MODULES
# ==============================================================================

class WeatherModule:
    """
    Weather data management and analysis.

    Handles weather data collection, storage, and
    provides forecasting capabilities.
    """

    def __init__(self, latitude: float, elevation: float):
        """Initialize weather module."""
        self.latitude = latitude
        self.elevation = elevation
        self.observations: List[WeatherObservation] = []

    def add_observation(self, obs: WeatherObservation) -> None:
        """Add weather observation."""
        self.observations.append(obs)

    def get_recent_weather(self, days: int = 7) -> List[WeatherObservation]:
        """Get recent weather observations."""
        sorted_obs = sorted(self.observations, key=lambda x: x.date, reverse=True)
        return sorted_obs[:days]

    def calculate_eto(self, obs: WeatherObservation) -> float:
        """
        Calculate reference evapotranspiration.

        Uses simplified Hargreaves method when full
        Penman-Monteith data unavailable.
        """
        t_mean = (obs.t_max + obs.t_min) / 2
        doy = obs.date.timetuple().tm_yday

        # Extraterrestrial radiation
        lat_rad = np.radians(self.latitude)
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))

        ra = 24 * 60 / np.pi * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        # Hargreaves equation
        eto = 0.0023 * 0.408 * ra * (t_mean + 17.8) * np.sqrt(max(0, obs.t_max - obs.t_min))
        return max(0, eto)

    def get_weather_summary(self) -> Dict[str, float]:
        """Generate weather summary statistics."""
        if not self.observations:
            return {}

        recent = self.get_recent_weather(30)

        return {
            'avg_temp': np.mean([(o.t_max + o.t_min) / 2 for o in recent]),
            'total_precip': sum(o.precipitation for o in recent),
            'avg_humidity': np.mean([o.humidity for o in recent]),
            'max_temp': max(o.t_max for o in recent),
            'min_temp': min(o.t_min for o in recent),
            'days_recorded': len(recent),
        }


class CropModule:
    """
    Crop production management.

    Handles crop planning, growth tracking, and
    yield prediction.
    """

    # Crop parameters database
    CROP_DATABASE = {
        'corn': {
            'base_temp': 10.0,
            'gdd_maturity': 1400,
            'kc_stages': [0.3, 1.2, 0.35],  # initial, mid, late
            'typical_yield': 10.0,  # t/ha
        },
        'wheat': {
            'base_temp': 0.0,
            'gdd_maturity': 1300,
            'kc_stages': [0.3, 1.15, 0.25],
            'typical_yield': 6.0,
        },
        'soybean': {
            'base_temp': 10.0,
            'gdd_maturity': 1450,
            'kc_stages': [0.4, 1.15, 0.5],
            'typical_yield': 3.5,
        },
    }

    def __init__(self):
        """Initialize crop module."""
        self.fields: Dict[str, FieldRecord] = {}

    def register_field(self, field: FieldRecord) -> None:
        """Register a new field."""
        self.fields[field.field_id] = field

    def record_planting(
        self,
        field_id: str,
        crop: str,
        planting_date: date,
        variety: str = None,
    ) -> None:
        """Record crop planting."""
        if field_id in self.fields:
            self.fields[field_id].current_crop = crop
            self.fields[field_id].planting_date = planting_date
            self.fields[field_id].field_history.append({
                'event': 'planting',
                'date': planting_date,
                'crop': crop,
                'variety': variety,
            })

    def calculate_gdd(
        self,
        field_id: str,
        weather_data: List[WeatherObservation],
    ) -> float:
        """Calculate accumulated GDD for field."""
        field = self.fields.get(field_id)
        if not field or not field.planting_date or not field.current_crop:
            return 0.0

        crop_params = self.CROP_DATABASE.get(field.current_crop, self.CROP_DATABASE['corn'])
        base_temp = crop_params['base_temp']

        gdd = 0.0
        for obs in weather_data:
            if obs.date >= field.planting_date:
                t_mean = (obs.t_max + obs.t_min) / 2
                daily_gdd = max(0, t_mean - base_temp)
                gdd += daily_gdd

        return gdd

    def estimate_yield(
        self,
        field_id: str,
        gdd: float,
        total_precip: float,
    ) -> Dict[str, float]:
        """Estimate crop yield based on conditions."""
        field = self.fields.get(field_id)
        if not field or not field.current_crop:
            return {}

        crop_params = self.CROP_DATABASE.get(field.current_crop, self.CROP_DATABASE['corn'])

        # GDD factor (0-1)
        gdd_factor = min(1.0, gdd / crop_params['gdd_maturity'])

        # Water factor (simplified)
        optimal_precip = 500  # mm
        water_factor = min(1.0, total_precip / optimal_precip)

        # Yield estimate
        potential_yield = crop_params['typical_yield'] * gdd_factor * water_factor
        total_production = potential_yield * field.area_ha

        return {
            'field_id': field_id,
            'crop': field.current_crop,
            'area_ha': field.area_ha,
            'yield_t_ha': potential_yield,
            'total_production_t': total_production,
            'gdd_factor': gdd_factor,
            'water_factor': water_factor,
        }

    def get_crop_summary(self) -> Dict[str, Any]:
        """Generate crop production summary."""
        by_crop = {}
        total_area = 0

        for field in self.fields.values():
            if field.current_crop:
                crop = field.current_crop
                if crop not in by_crop:
                    by_crop[crop] = {'area': 0, 'fields': 0}
                by_crop[crop]['area'] += field.area_ha
                by_crop[crop]['fields'] += 1
            total_area += field.area_ha

        return {
            'total_fields': len(self.fields),
            'total_area_ha': total_area,
            'by_crop': by_crop,
        }


class LivestockModule:
    """
    Livestock management module.

    Handles animal registration, tracking, health
    monitoring, and production records.
    """

    def __init__(self):
        """Initialize livestock module."""
        self.animals: Dict[str, AnimalRecord] = {}
        self.weight_records: Dict[str, List[Tuple[date, float]]] = {}
        self.health_records: Dict[str, List[Dict]] = {}

    def register_animal(
        self,
        species: str,
        breed: str,
        sex: str,
        birth_date: date,
        birth_weight: float = None,
    ) -> AnimalRecord:
        """Register a new animal."""
        animal_id = str(uuid.uuid4())[:8].upper()

        animal = AnimalRecord(
            animal_id=animal_id,
            species=species,
            breed=breed,
            sex=sex,
            birth_date=birth_date,
            current_weight=birth_weight,
        )

        self.animals[animal_id] = animal
        self.weight_records[animal_id] = []
        self.health_records[animal_id] = []

        if birth_weight:
            self.weight_records[animal_id].append((birth_date, birth_weight))

        return animal

    def record_weight(
        self,
        animal_id: str,
        weight: float,
        record_date: date = None,
    ) -> None:
        """Record animal weight."""
        record_date = record_date or date.today()

        if animal_id in self.animals:
            self.weight_records[animal_id].append((record_date, weight))
            self.animals[animal_id].current_weight = weight

    def record_health_event(
        self,
        animal_id: str,
        event_type: str,
        description: str,
        treatment: str = None,
    ) -> None:
        """Record health event."""
        if animal_id in self.animals:
            self.health_records[animal_id].append({
                'date': date.today(),
                'type': event_type,
                'description': description,
                'treatment': treatment,
            })

    def calculate_adg(self, animal_id: str) -> float:
        """Calculate average daily gain."""
        records = self.weight_records.get(animal_id, [])
        if len(records) < 2:
            return 0.0

        records = sorted(records, key=lambda x: x[0])
        first_date, first_weight = records[0]
        last_date, last_weight = records[-1]

        days = (last_date - first_date).days
        if days == 0:
            return 0.0

        return (last_weight - first_weight) / days

    def get_livestock_summary(self) -> Dict[str, Any]:
        """Generate livestock summary."""
        active = [a for a in self.animals.values() if a.status == 'active']

        by_species = {}
        total_weight = 0
        weight_count = 0

        for animal in active:
            species = animal.species
            by_species[species] = by_species.get(species, 0) + 1

            if animal.current_weight:
                total_weight += animal.current_weight
                weight_count += 1

        return {
            'total_active': len(active),
            'by_species': by_species,
            'avg_weight': total_weight / weight_count if weight_count else 0,
            'total_weight': total_weight,
        }


class FeedModule:
    """
    Feed management module.

    Handles feed inventory, ration formulation,
    and consumption tracking.
    """

    FEED_INGREDIENTS = {
        'corn': {'protein': 8.8, 'energy': 3.3, 'cost': 0.20},
        'soybean_meal': {'protein': 44.0, 'energy': 3.1, 'cost': 0.45},
        'hay': {'protein': 12.0, 'energy': 2.0, 'cost': 0.12},
        'silage': {'protein': 8.0, 'energy': 2.3, 'cost': 0.05},
    }

    def __init__(self):
        """Initialize feed module."""
        self.inventory: Dict[str, float] = {}  # ingredient: kg
        self.consumption_log: List[Dict] = []

    def add_inventory(self, ingredient: str, quantity_kg: float, cost: float = None) -> None:
        """Add feed inventory."""
        if ingredient in self.FEED_INGREDIENTS:
            self.inventory[ingredient] = self.inventory.get(ingredient, 0) + quantity_kg

    def record_consumption(
        self,
        ingredient: str,
        quantity_kg: float,
        animal_group: str = None,
    ) -> None:
        """Record feed consumption."""
        if ingredient in self.inventory:
            self.inventory[ingredient] = max(0, self.inventory[ingredient] - quantity_kg)

        self.consumption_log.append({
            'date': date.today(),
            'ingredient': ingredient,
            'quantity_kg': quantity_kg,
            'group': animal_group,
        })

    def calculate_ration_cost(self, ration: Dict[str, float], daily_intake_kg: float) -> float:
        """Calculate daily ration cost."""
        cost = 0.0
        for ingredient, proportion in ration.items():
            if ingredient in self.FEED_INGREDIENTS:
                cost += proportion * daily_intake_kg * self.FEED_INGREDIENTS[ingredient]['cost']
        return cost

    def get_inventory_summary(self) -> Dict[str, Any]:
        """Generate inventory summary."""
        total_value = 0
        for ingredient, quantity in self.inventory.items():
            if ingredient in self.FEED_INGREDIENTS:
                total_value += quantity * self.FEED_INGREDIENTS[ingredient]['cost']

        return {
            'inventory': self.inventory.copy(),
            'total_kg': sum(self.inventory.values()),
            'total_value': total_value,
            'n_ingredients': len(self.inventory),
        }


class FinancialModule:
    """
    Financial tracking module.

    Handles income, expenses, and profitability
    analysis for farm operations.
    """

    def __init__(self):
        """Initialize financial module."""
        self.transactions: List[FinancialRecord] = []

    def record_transaction(
        self,
        category: str,
        description: str,
        amount: float,
        trans_type: str,
        trans_date: date = None,
    ) -> FinancialRecord:
        """Record financial transaction."""
        trans_id = str(uuid.uuid4())[:8]
        trans_date = trans_date or date.today()

        record = FinancialRecord(
            transaction_id=trans_id,
            date=trans_date,
            category=category,
            description=description,
            amount=amount,
            type=trans_type,
        )

        self.transactions.append(record)
        return record

    def get_summary(
        self,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Any]:
        """Generate financial summary."""
        transactions = self.transactions

        if start_date:
            transactions = [t for t in transactions if t.date >= start_date]
        if end_date:
            transactions = [t for t in transactions if t.date <= end_date]

        income = sum(t.amount for t in transactions if t.type == 'income')
        expenses = sum(t.amount for t in transactions if t.type == 'expense')

        by_category = {}
        for t in transactions:
            if t.category not in by_category:
                by_category[t.category] = {'income': 0, 'expense': 0}
            by_category[t.category][t.type] += t.amount

        return {
            'total_income': income,
            'total_expenses': expenses,
            'net_profit': income - expenses,
            'profit_margin': (income - expenses) / income * 100 if income > 0 else 0,
            'by_category': by_category,
            'transaction_count': len(transactions),
        }


class MLPredictions:
    """
    Machine learning predictions module.

    Provides predictive analytics for various
    farm operations.
    """

    @staticmethod
    def predict_yield(
        historical_yields: List[float],
        current_conditions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Predict crop yield using historical data.

        Implements a simple ensemble of trend analysis
        and condition-based adjustment.
        """
        if not historical_yields:
            return {'prediction': 0, 'confidence': 0}

        # Trend prediction
        n = len(historical_yields)
        if n >= 3:
            # Simple linear trend
            x = np.arange(n)
            slope = np.polyfit(x, historical_yields, 1)[0]
            trend_pred = historical_yields[-1] + slope
        else:
            trend_pred = np.mean(historical_yields)

        # Condition adjustment
        base = np.mean(historical_yields)
        gdd_factor = current_conditions.get('gdd_factor', 1.0)
        water_factor = current_conditions.get('water_factor', 1.0)
        condition_pred = base * gdd_factor * water_factor

        # Ensemble
        prediction = 0.4 * trend_pred + 0.6 * condition_pred

        # Confidence based on data and conditions
        data_conf = min(1.0, n / 5)
        condition_conf = (gdd_factor + water_factor) / 2
        confidence = data_conf * condition_conf

        return {
            'prediction': max(0, prediction),
            'confidence': confidence,
            'trend_component': trend_pred,
            'condition_component': condition_pred,
        }

    @staticmethod
    def predict_weight_gain(
        current_weight: float,
        age_days: int,
        species: str,
        days_ahead: int = 30,
    ) -> Dict[str, float]:
        """Predict animal weight gain."""
        # Species-specific growth parameters
        params = {
            'cattle': {'mature': 550, 'k': 0.0038},
            'sheep': {'mature': 80, 'k': 0.008},
            'swine': {'mature': 120, 'k': 0.012},
        }

        p = params.get(species, params['cattle'])

        # Gompertz model prediction
        b = np.log(p['mature'] / 35)  # Assume birth weight
        current_pred = p['mature'] * np.exp(-b * np.exp(-p['k'] * age_days))
        future_pred = p['mature'] * np.exp(-b * np.exp(-p['k'] * (age_days + days_ahead)))

        predicted_gain = future_pred - current_pred
        adg = predicted_gain / days_ahead

        return {
            'current_expected': current_pred,
            'future_weight': future_pred,
            'predicted_gain': predicted_gain,
            'adg': adg,
            'days': days_ahead,
        }


# ==============================================================================
# INTEGRATED FARM MANAGER
# ==============================================================================

class IntegratedFarmManager:
    """
    Central farm management system.

    Integrates all modules into a unified management
    platform with cross-module analytics.
    """

    def __init__(self, config: FarmConfiguration):
        """
        Initialize integrated farm manager.

        Args:
            config: Farm configuration
        """
        self.config = config
        self.weather = WeatherModule(config.location[0], config.elevation)
        self.crops = CropModule()
        self.livestock = LivestockModule()
        self.feed = FeedModule()
        self.finance = FinancialModule()
        self.ml = MLPredictions()

        logger.info(f"Initialized farm manager for {config.name}")

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive daily farm report.

        Aggregates data from all modules into a
        unified daily operations report.
        """
        report = {
            'farm': self.config.name,
            'date': date.today(),
            'weather': self.weather.get_weather_summary(),
            'crops': self.crops.get_crop_summary(),
            'livestock': self.livestock.get_livestock_summary(),
            'feed': self.feed.get_inventory_summary(),
            'finance': self.finance.get_summary(
                start_date=date.today() - timedelta(days=30)
            ),
        }

        # Add alerts
        report['alerts'] = self._generate_alerts()

        return report

    def _generate_alerts(self) -> List[str]:
        """Generate operational alerts."""
        alerts = []

        # Weather alerts
        weather = self.weather.get_weather_summary()
        if weather.get('total_precip', 0) < 20:  # Less than 20mm in 30 days
            alerts.append("LOW PRECIPITATION: Consider irrigation scheduling")

        # Feed alerts
        feed_inv = self.feed.get_inventory_summary()
        if feed_inv['total_kg'] < 1000:
            alerts.append("LOW FEED INVENTORY: Reorder recommended")

        # Financial alerts
        finance = self.finance.get_summary()
        if finance['net_profit'] < 0:
            alerts.append("NEGATIVE CASH FLOW: Review expenses")

        return alerts

    def run_seasonal_analysis(
        self,
        season_start: date,
        season_end: date,
    ) -> Dict[str, Any]:
        """
        Run comprehensive seasonal analysis.

        Analyzes performance across all operations
        for a given season.
        """
        # Crop analysis
        crop_yields = []
        for field_id, field in self.crops.fields.items():
            if field.current_crop:
                weather_data = [o for o in self.weather.observations
                               if season_start <= o.date <= season_end]
                gdd = self.crops.calculate_gdd(field_id, weather_data)
                precip = sum(o.precipitation for o in weather_data)
                yield_est = self.crops.estimate_yield(field_id, gdd, precip)
                crop_yields.append(yield_est)

        # Livestock analysis
        livestock_performance = []
        for animal_id, animal in self.livestock.animals.items():
            adg = self.livestock.calculate_adg(animal_id)
            livestock_performance.append({
                'animal_id': animal_id,
                'species': animal.species,
                'adg': adg,
                'current_weight': animal.current_weight,
            })

        # Financial analysis
        financial = self.finance.get_summary(season_start, season_end)

        return {
            'season': f"{season_start} to {season_end}",
            'crop_production': {
                'total_yield_t': sum(y.get('total_production_t', 0) for y in crop_yields),
                'fields_analyzed': len(crop_yields),
                'by_field': crop_yields,
            },
            'livestock_performance': {
                'animals_tracked': len(livestock_performance),
                'avg_adg': np.mean([l['adg'] for l in livestock_performance]) if livestock_performance else 0,
                'by_animal': livestock_performance,
            },
            'financial': financial,
        }


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

def run_demonstration():
    """
    Run comprehensive integrated farm demonstration.
    """
    print("=" * 70)
    print("Smart Agriculture - Integrated Farm Management System")
    print("=" * 70)
    print()

    # Configure farm
    config = FarmConfiguration(
        farm_id="FARM001",
        name="Green Valley Farm",
        location=(42.0, -89.5),
        elevation=250,
        total_area_ha=500,
    )

    # Initialize manager
    farm = IntegratedFarmManager(config)

    print(f"Farm: {config.name}")
    print(f"Location: {config.location[0]}°N, {config.location[1]}°W")
    print(f"Total Area: {config.total_area_ha} ha")
    print()

    # ==== SETUP: Register Fields ====
    print("Setting Up Farm Operations...")
    print("-" * 50)

    fields = [
        FieldRecord("F001", "North Field", 120, "loam"),
        FieldRecord("F002", "South Field", 95, "clay_loam"),
        FieldRecord("F003", "East Pasture", 80, "sandy_loam"),
        FieldRecord("F004", "West Field", 100, "loam"),
    ]

    for field in fields:
        farm.crops.register_field(field)

    # Record plantings
    farm.crops.record_planting("F001", "corn", date(2024, 4, 20), "Pioneer P1197")
    farm.crops.record_planting("F002", "soybean", date(2024, 5, 5), "Asgrow AG36X6")
    farm.crops.record_planting("F004", "wheat", date(2024, 3, 15), "WestBred WB4303")

    print(f"  • Registered {len(fields)} fields")
    print(f"  • Planted: Corn (F001), Soybean (F002), Wheat (F004)")
    print(f"  • Pasture: F003 (East Pasture)")

    # ==== SETUP: Register Animals ====
    print()

    animals_data = [
        ("cattle", "Angus", "F", date(2022, 4, 10), 38.0),
        ("cattle", "Angus", "F", date(2022, 5, 15), 40.0),
        ("cattle", "Hereford", "M", date(2023, 3, 20), 42.0),
        ("cattle", "Angus", "M", date(2023, 6, 1), 35.0),
        ("sheep", "Suffolk", "F", date(2024, 2, 1), 4.2),
        ("sheep", "Suffolk", "F", date(2024, 2, 5), 4.5),
        ("swine", "Yorkshire", "M", date(2024, 5, 1), 1.4),
    ]

    registered_animals = []
    for species, breed, sex, birth, weight in animals_data:
        animal = farm.livestock.register_animal(species, breed, sex, birth, weight)
        registered_animals.append(animal)

    # Add weight records
    farm.livestock.record_weight(registered_animals[0].animal_id, 520.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[1].animal_id, 495.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[2].animal_id, 380.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[3].animal_id, 320.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[4].animal_id, 42.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[5].animal_id, 45.0, date(2024, 10, 1))
    farm.livestock.record_weight(registered_animals[6].animal_id, 95.0, date(2024, 10, 1))

    print(f"  • Registered {len(registered_animals)} animals")
    print(f"  • Species: 4 cattle, 2 sheep, 1 swine")

    # ==== SETUP: Add Weather Data ====
    print()

    np.random.seed(42)
    for i in range(90):  # 90 days of weather
        obs_date = date(2024, 7, 1) + timedelta(days=i)
        doy = obs_date.timetuple().tm_yday

        t_base = 22 + 8 * np.sin(2 * np.pi * (doy - 172) / 365)
        obs = WeatherObservation(
            date=obs_date,
            t_max=t_base + np.random.uniform(6, 10),
            t_min=t_base - np.random.uniform(6, 10),
            precipitation=np.random.exponential(3) if np.random.random() < 0.2 else 0,
            humidity=50 + np.random.normal(0, 15),
            wind_speed=2 + np.random.exponential(1.5),
            solar_radiation=18 + np.random.normal(0, 4),
        )
        farm.weather.add_observation(obs)

    print(f"  • Loaded {len(farm.weather.observations)} days of weather data")

    # ==== SETUP: Feed Inventory ====
    print()

    farm.feed.add_inventory("corn", 5000)
    farm.feed.add_inventory("soybean_meal", 2000)
    farm.feed.add_inventory("hay", 8000)
    farm.feed.add_inventory("silage", 15000)

    print(f"  • Feed inventory initialized: {farm.feed.get_inventory_summary()['total_kg']:.0f} kg")

    # ==== SETUP: Financial Records ====
    print()

    # Income
    farm.finance.record_transaction("livestock", "Cattle sale - 2 steers", 4500, "income", date(2024, 8, 15))
    farm.finance.record_transaction("crops", "Wheat harvest sale", 25000, "income", date(2024, 7, 20))

    # Expenses
    farm.finance.record_transaction("feed", "Corn purchase", 2000, "expense", date(2024, 7, 1))
    farm.finance.record_transaction("feed", "Soybean meal purchase", 900, "expense", date(2024, 7, 1))
    farm.finance.record_transaction("veterinary", "Annual vaccinations", 350, "expense", date(2024, 7, 15))
    farm.finance.record_transaction("equipment", "Tractor maintenance", 800, "expense", date(2024, 8, 1))
    farm.finance.record_transaction("seed", "Corn seed purchase", 3500, "expense", date(2024, 4, 1))
    farm.finance.record_transaction("fertilizer", "NPK fertilizer", 4200, "expense", date(2024, 4, 15))

    print(f"  • Financial records: {len(farm.finance.transactions)} transactions")

    # ==== DAILY REPORT ====
    print("\n")
    print("=" * 70)
    print("DAILY OPERATIONS REPORT")
    print("=" * 70)

    report = farm.generate_daily_report()

    print(f"\nDate: {report['date']}")

    print(f"\n📊 Weather Summary (Last 30 Days):")
    w = report['weather']
    if w:
        print(f"   • Average Temperature: {w.get('avg_temp', 0):.1f}°C")
        print(f"   • Temperature Range: {w.get('min_temp', 0):.1f} - {w.get('max_temp', 0):.1f}°C")
        print(f"   • Total Precipitation: {w.get('total_precip', 0):.1f} mm")
        print(f"   • Average Humidity: {w.get('avg_humidity', 0):.1f}%")

    print(f"\n🌾 Crop Status:")
    c = report['crops']
    print(f"   • Total Fields: {c['total_fields']}")
    print(f"   • Total Area: {c['total_area_ha']:.1f} ha")
    for crop, data in c['by_crop'].items():
        print(f"   • {crop.capitalize()}: {data['area']:.1f} ha ({data['fields']} fields)")

    print(f"\n🐄 Livestock Status:")
    l = report['livestock']
    print(f"   • Active Animals: {l['total_active']}")
    for species, count in l['by_species'].items():
        print(f"   • {species.capitalize()}: {count}")
    print(f"   • Average Weight: {l['avg_weight']:.1f} kg")
    print(f"   • Total Weight: {l['total_weight']:.1f} kg")

    print(f"\n📦 Feed Inventory:")
    f = report['feed']
    print(f"   • Total Stock: {f['total_kg']:.0f} kg")
    print(f"   • Inventory Value: ${f['total_value']:.2f}")
    for ingredient, qty in f['inventory'].items():
        print(f"   • {ingredient}: {qty:.0f} kg")

    print(f"\n💰 Financial Summary (Last 30 Days):")
    fin = report['finance']
    print(f"   • Total Income: ${fin['total_income']:,.2f}")
    print(f"   • Total Expenses: ${fin['total_expenses']:,.2f}")
    print(f"   • Net Profit: ${fin['net_profit']:,.2f}")
    print(f"   • Profit Margin: {fin['profit_margin']:.1f}%")

    print(f"\n⚠️ Alerts:")
    if report['alerts']:
        for alert in report['alerts']:
            print(f"   • {alert}")
    else:
        print("   • No alerts")

    # ==== ML PREDICTIONS ====
    print("\n")
    print("=" * 70)
    print("ML PREDICTIONS & ANALYTICS")
    print("=" * 70)

    # Yield prediction
    print("\n📈 Crop Yield Predictions:")
    historical_corn = [9.5, 10.2, 9.8, 11.0, 10.5]
    conditions = {'gdd_factor': 0.92, 'water_factor': 0.85}
    yield_pred = farm.ml.predict_yield(historical_corn, conditions)
    print(f"   Corn Yield Prediction:")
    print(f"   • Predicted: {yield_pred['prediction']:.2f} t/ha")
    print(f"   • Confidence: {yield_pred['confidence']*100:.1f}%")
    print(f"   • Trend Component: {yield_pred['trend_component']:.2f} t/ha")
    print(f"   • Condition Component: {yield_pred['condition_component']:.2f} t/ha")

    # Weight gain prediction
    print("\n📈 Livestock Growth Predictions:")
    for animal in registered_animals[:3]:
        age = (date.today() - animal.birth_date).days
        pred = farm.ml.predict_weight_gain(
            animal.current_weight or 0,
            age,
            animal.species,
            days_ahead=60
        )
        print(f"   {animal.animal_id} ({animal.species} {animal.breed}):")
        print(f"   • Current: {animal.current_weight:.0f} kg (age: {age} days)")
        print(f"   • Predicted in 60 days: {pred['future_weight']:.0f} kg")
        print(f"   • Expected ADG: {pred['adg']:.2f} kg/day")

    # ==== SEASONAL ANALYSIS ====
    print("\n")
    print("=" * 70)
    print("SEASONAL ANALYSIS (Jul-Sep 2024)")
    print("=" * 70)

    seasonal = farm.run_seasonal_analysis(date(2024, 7, 1), date(2024, 9, 30))

    print(f"\n🌾 Crop Production:")
    print(f"   • Fields Analyzed: {seasonal['crop_production']['fields_analyzed']}")
    print(f"   • Estimated Total Yield: {seasonal['crop_production']['total_yield_t']:.1f} t")

    print(f"\n🐄 Livestock Performance:")
    print(f"   • Animals Tracked: {seasonal['livestock_performance']['animals_tracked']}")
    print(f"   • Average ADG: {seasonal['livestock_performance']['avg_adg']:.2f} kg/day")

    print(f"\n💰 Season Financial Performance:")
    print(f"   • Revenue: ${seasonal['financial']['total_income']:,.2f}")
    print(f"   • Expenses: ${seasonal['financial']['total_expenses']:,.2f}")
    print(f"   • Net Profit: ${seasonal['financial']['net_profit']:,.2f}")

    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    print("\nThis integrated system demonstrates how smartagri modules")
    print("work together to provide comprehensive farm management.")
    print("=" * 70)


def main():
    """Main entry point."""
    run_demonstration()


if __name__ == "__main__":
    main()
