"""
Herd Management Module

Comprehensive herd analytics and management:
- Herd composition analysis
- Performance benchmarking
- Culling decisions
- Inventory management
- Economic analysis

Example:
    >>> herd = HerdManager()
    >>> herd.add_animal(animal)
    >>> stats = herd.get_composition_stats()
    >>> culling_candidates = herd.identify_culling_candidates()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProductionSystem(Enum):
    """Livestock production system types."""
    COW_CALF = "cow_calf"
    FEEDLOT = "feedlot"
    DAIRY = "dairy"
    STOCKER = "stocker"
    BREEDING = "breeding"
    MIXED = "mixed"


class CullingReason(Enum):
    """Reasons for culling animals."""
    AGE = "age"
    REPRODUCTION = "reproduction_failure"
    PRODUCTION = "low_production"
    HEALTH = "health_issues"
    TEMPERAMENT = "temperament"
    STRUCTURE = "structural_issues"
    ECONOMICS = "economics"
    GENETICS = "genetic_improvement"
    OTHER = "other"


@dataclass
class AnimalRecord:
    """
    Complete animal record for herd management.

    Attributes:
        id: Unique identifier
        species: Animal species
        breed: Breed name
        sex: Animal sex
        birth_date: Date of birth
        status: Current status
        production_data: Production records
        health_data: Health records
        economic_data: Economic records
    """
    id: str
    species: str
    breed: str = ""
    sex: str = "unknown"
    birth_date: Optional[date] = None
    status: str = "active"
    sire_id: Optional[str] = None
    dam_id: Optional[str] = None
    production_data: Dict[str, Any] = field(default_factory=dict)
    health_data: Dict[str, Any] = field(default_factory=dict)
    economic_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_years(self) -> Optional[float]:
        """Calculate age in years."""
        if self.birth_date:
            return (date.today() - self.birth_date).days / 365.25
        return None


@dataclass
class HerdStats:
    """Herd statistics summary."""
    total_count: int
    by_sex: Dict[str, int]
    by_breed: Dict[str, int]
    by_age_group: Dict[str, int]
    avg_age: float
    active_count: int
    culled_count: int


@dataclass
class CullingCandidate:
    """Culling recommendation."""
    animal_id: str
    reason: CullingReason
    score: float  # 0-100, higher = more likely to cull
    details: str
    economic_impact: float


class HerdInventory:
    """
    Herd inventory tracking and management.

    Tracks animal inventory, movements, and
    provides real-time herd counts.

    Example:
        >>> inventory = HerdInventory()
        >>> inventory.add_animal(animal_record)
        >>> count = inventory.count(status="active", sex="female")
    """

    def __init__(self):
        """Initialize herd inventory."""
        self._animals: Dict[str, AnimalRecord] = {}
        self._groups: Dict[str, List[str]] = {}

    def add_animal(self, animal: AnimalRecord) -> None:
        """
        Add animal to inventory.

        Args:
            animal: AnimalRecord to add
        """
        self._animals[animal.id] = animal
        logger.info(f"Added animal {animal.id} to inventory")

    def remove_animal(
        self,
        animal_id: str,
        reason: str = "",
    ) -> Optional[AnimalRecord]:
        """
        Remove animal from inventory.

        Args:
            animal_id: Animal identifier
            reason: Removal reason

        Returns:
            Removed AnimalRecord or None
        """
        if animal_id in self._animals:
            animal = self._animals.pop(animal_id)
            logger.info(f"Removed animal {animal_id}: {reason}")

            # Remove from groups
            for group_id in self._groups:
                if animal_id in self._groups[group_id]:
                    self._groups[group_id].remove(animal_id)

            return animal
        return None

    def get_animal(self, animal_id: str) -> Optional[AnimalRecord]:
        """Get animal by ID."""
        return self._animals.get(animal_id)

    def update_animal(
        self,
        animal_id: str,
        **updates,
    ) -> Optional[AnimalRecord]:
        """
        Update animal attributes.

        Args:
            animal_id: Animal identifier
            **updates: Attributes to update

        Returns:
            Updated AnimalRecord
        """
        animal = self._animals.get(animal_id)
        if animal:
            for key, value in updates.items():
                if hasattr(animal, key):
                    setattr(animal, key, value)
        return animal

    def list_animals(
        self,
        status: Optional[str] = None,
        sex: Optional[str] = None,
        breed: Optional[str] = None,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
    ) -> List[AnimalRecord]:
        """
        List animals with filters.

        Args:
            status: Filter by status
            sex: Filter by sex
            breed: Filter by breed
            min_age: Minimum age in years
            max_age: Maximum age in years

        Returns:
            List of matching animals
        """
        results = list(self._animals.values())

        if status:
            results = [a for a in results if a.status == status]
        if sex:
            results = [a for a in results if a.sex.lower() == sex.lower()]
        if breed:
            results = [a for a in results if a.breed.lower() == breed.lower()]
        if min_age is not None:
            results = [a for a in results
                       if a.age_years and a.age_years >= min_age]
        if max_age is not None:
            results = [a for a in results
                       if a.age_years and a.age_years <= max_age]

        return results

    def count(self, **filters) -> int:
        """Count animals matching filters."""
        return len(self.list_animals(**filters))

    def create_group(
        self,
        group_id: str,
        animal_ids: List[str],
    ) -> None:
        """Create animal group."""
        self._groups[group_id] = [
            aid for aid in animal_ids if aid in self._animals
        ]

    def get_group(self, group_id: str) -> List[AnimalRecord]:
        """Get animals in group."""
        animal_ids = self._groups.get(group_id, [])
        return [self._animals[aid] for aid in animal_ids if aid in self._animals]


class HerdComposition:
    """
    Herd composition analysis.

    Analyzes herd structure, demographics, and
    provides composition recommendations.

    Example:
        >>> comp = HerdComposition(inventory)
        >>> stats = comp.get_stats()
        >>> optimal = comp.recommend_composition()
    """

    # Age group definitions by species (years)
    AGE_GROUPS = {
        "cattle": {
            "calf": (0, 0.5),
            "weaner": (0.5, 1),
            "yearling": (1, 2),
            "young_adult": (2, 4),
            "mature": (4, 8),
            "aged": (8, float('inf')),
        },
        "sheep": {
            "lamb": (0, 0.5),
            "hogget": (0.5, 1.5),
            "adult": (1.5, 6),
            "aged": (6, float('inf')),
        },
        "swine": {
            "piglet": (0, 0.17),
            "weaner": (0.17, 0.33),
            "grower": (0.33, 0.5),
            "finisher": (0.5, 0.67),
            "adult": (0.67, 3),
            "aged": (3, float('inf')),
        },
    }

    def __init__(
        self,
        inventory: HerdInventory,
        species: str = "cattle",
    ):
        """
        Initialize composition analyzer.

        Args:
            inventory: HerdInventory instance
            species: Animal species
        """
        self.inventory = inventory
        self.species = species.lower()
        self.age_groups = self.AGE_GROUPS.get(
            self.species,
            self.AGE_GROUPS["cattle"]
        )

    def get_stats(self) -> HerdStats:
        """
        Get comprehensive herd statistics.

        Returns:
            HerdStats object
        """
        animals = self.inventory.list_animals()
        active = [a for a in animals if a.status == "active"]

        # Count by sex
        by_sex = {}
        for animal in active:
            sex = animal.sex.lower()
            by_sex[sex] = by_sex.get(sex, 0) + 1

        # Count by breed
        by_breed = {}
        for animal in active:
            breed = animal.breed or "unknown"
            by_breed[breed] = by_breed.get(breed, 0) + 1

        # Count by age group
        by_age = {group: 0 for group in self.age_groups}
        ages = []
        for animal in active:
            if animal.age_years is not None:
                ages.append(animal.age_years)
                for group, (min_age, max_age) in self.age_groups.items():
                    if min_age <= animal.age_years < max_age:
                        by_age[group] += 1
                        break

        return HerdStats(
            total_count=len(animals),
            by_sex=by_sex,
            by_breed=by_breed,
            by_age_group=by_age,
            avg_age=np.mean(ages) if ages else 0,
            active_count=len(active),
            culled_count=len([a for a in animals if a.status == "culled"]),
        )

    def sex_ratio(self) -> Dict[str, float]:
        """Calculate sex ratio (females per male)."""
        active = self.inventory.list_animals(status="active")

        males = sum(1 for a in active if a.sex.lower() in ["male", "m"])
        females = sum(1 for a in active if a.sex.lower() in ["female", "f"])

        return {
            "males": males,
            "females": females,
            "ratio": females / males if males > 0 else float('inf'),
        }

    def age_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed age distribution."""
        active = self.inventory.list_animals(status="active")

        distribution = {}
        for group, (min_age, max_age) in self.age_groups.items():
            group_animals = [
                a for a in active
                if a.age_years is not None and min_age <= a.age_years < max_age
            ]

            ages = [a.age_years for a in group_animals]

            distribution[group] = {
                "count": len(group_animals),
                "pct": len(group_animals) / len(active) * 100 if active else 0,
                "avg_age": np.mean(ages) if ages else 0,
                "ids": [a.id for a in group_animals],
            }

        return distribution

    def recommend_composition(
        self,
        system: ProductionSystem = ProductionSystem.COW_CALF,
    ) -> Dict[str, Any]:
        """
        Recommend optimal herd composition.

        Args:
            system: Production system type

        Returns:
            Dict with recommendations
        """
        current = self.get_stats()

        # Target compositions by production system
        targets = {
            ProductionSystem.COW_CALF: {
                "sex_ratio": 25,  # cows per bull
                "replacement_rate": 0.15,
                "mature_pct": 0.70,
                "young_pct": 0.20,
            },
            ProductionSystem.DAIRY: {
                "sex_ratio": 40,
                "replacement_rate": 0.25,
                "mature_pct": 0.75,
                "young_pct": 0.20,
            },
            ProductionSystem.FEEDLOT: {
                "sex_ratio": None,  # All same sex typically
                "replacement_rate": 1.0,  # Complete turnover
                "mature_pct": 0.0,
                "young_pct": 1.0,
            },
        }

        target = targets.get(system, targets[ProductionSystem.COW_CALF])
        current_ratio = self.sex_ratio()

        recommendations = []

        # Sex ratio check
        if target["sex_ratio"] and current_ratio["ratio"] != float('inf'):
            if current_ratio["ratio"] < target["sex_ratio"] * 0.8:
                recommendations.append(
                    f"Consider reducing bulls; current ratio "
                    f"{current_ratio['ratio']:.1f} vs target {target['sex_ratio']}"
                )
            elif current_ratio["ratio"] > target["sex_ratio"] * 1.2:
                recommendations.append(
                    f"May need more breeding males; current ratio "
                    f"{current_ratio['ratio']:.1f} vs target {target['sex_ratio']}"
                )

        return {
            "current": {
                "total": current.active_count,
                "sex_ratio": current_ratio["ratio"],
                "age_distribution": current.by_age_group,
            },
            "target": target,
            "recommendations": recommendations,
        }


class PerformanceBenchmark:
    """
    Herd performance benchmarking.

    Compares herd performance against industry
    standards and internal metrics.

    Example:
        >>> benchmark = PerformanceBenchmark()
        >>> benchmark.add_metric("weaning_weight", animal_id, value)
        >>> comparison = benchmark.compare_to_industry(metric)
    """

    # Industry benchmarks by species
    BENCHMARKS = {
        "cattle": {
            "weaning_weight": {"mean": 250, "std": 40, "unit": "kg"},
            "yearling_weight": {"mean": 420, "std": 50, "unit": "kg"},
            "adg": {"mean": 1.2, "std": 0.3, "unit": "kg/day"},
            "calving_interval": {"mean": 365, "std": 30, "unit": "days"},
            "weaning_rate": {"mean": 0.88, "std": 0.08, "unit": "%"},
            "conception_rate": {"mean": 0.55, "std": 0.10, "unit": "%"},
        },
        "sheep": {
            "weaning_weight": {"mean": 30, "std": 5, "unit": "kg"},
            "lambing_rate": {"mean": 1.5, "std": 0.3, "unit": "lambs/ewe"},
            "wool_weight": {"mean": 4.5, "std": 1.0, "unit": "kg"},
        },
        "swine": {
            "litter_size": {"mean": 12, "std": 2, "unit": "piglets"},
            "days_to_market": {"mean": 165, "std": 15, "unit": "days"},
            "feed_conversion": {"mean": 2.8, "std": 0.3, "unit": "kg/kg"},
        },
    }

    def __init__(self, species: str = "cattle"):
        """
        Initialize benchmark system.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.benchmarks = self.BENCHMARKS.get(
            self.species,
            self.BENCHMARKS["cattle"]
        )
        self._metrics: Dict[str, Dict[str, List[float]]] = {}

    def add_metric(
        self,
        metric_name: str,
        animal_id: str,
        value: float,
    ) -> None:
        """
        Add performance metric for animal.

        Args:
            metric_name: Metric name
            animal_id: Animal identifier
            value: Metric value
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = {}

        if animal_id not in self._metrics[metric_name]:
            self._metrics[metric_name][animal_id] = []

        self._metrics[metric_name][animal_id].append(value)

    def get_herd_average(self, metric_name: str) -> Optional[float]:
        """Get herd average for metric."""
        if metric_name not in self._metrics:
            return None

        all_values = []
        for values in self._metrics[metric_name].values():
            all_values.extend(values)

        return np.mean(all_values) if all_values else None

    def compare_to_industry(
        self,
        metric_name: str,
    ) -> Dict[str, Any]:
        """
        Compare herd metric to industry benchmark.

        Args:
            metric_name: Metric name

        Returns:
            Dict with comparison results
        """
        herd_avg = self.get_herd_average(metric_name)
        benchmark = self.benchmarks.get(metric_name)

        if herd_avg is None or benchmark is None:
            return {"status": "insufficient_data"}

        industry_mean = benchmark["mean"]
        industry_std = benchmark["std"]

        # Calculate percentile (assuming normal distribution)
        z_score = (herd_avg - industry_mean) / industry_std
        percentile = self._norm_cdf(z_score) * 100

        # Classification
        if percentile >= 75:
            classification = "top_quartile"
        elif percentile >= 50:
            classification = "above_average"
        elif percentile >= 25:
            classification = "below_average"
        else:
            classification = "bottom_quartile"

        return {
            "metric": metric_name,
            "herd_value": herd_avg,
            "industry_mean": industry_mean,
            "industry_std": industry_std,
            "z_score": z_score,
            "percentile": percentile,
            "classification": classification,
            "unit": benchmark["unit"],
        }

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {}

        for metric_name in self._metrics:
            comparison = self.compare_to_industry(metric_name)
            if comparison.get("status") != "insufficient_data":
                report[metric_name] = comparison

        # Overall performance score (average percentile)
        percentiles = [
            r["percentile"] for r in report.values()
            if "percentile" in r
        ]

        report["overall"] = {
            "avg_percentile": np.mean(percentiles) if percentiles else None,
            "metrics_analyzed": len(report),
        }

        return report


class CullingDecision:
    """
    Culling decision support system.

    Identifies culling candidates based on
    production, reproduction, health, and economics.

    Example:
        >>> culler = CullingDecision(inventory, benchmark)
        >>> candidates = culler.identify_candidates(n=10)
    """

    def __init__(
        self,
        inventory: HerdInventory,
        benchmark: Optional[PerformanceBenchmark] = None,
    ):
        """
        Initialize culling decision system.

        Args:
            inventory: HerdInventory instance
            benchmark: PerformanceBenchmark instance
        """
        self.inventory = inventory
        self.benchmark = benchmark or PerformanceBenchmark()

        # Culling criteria weights
        self._weights = {
            "age": 0.20,
            "reproduction": 0.25,
            "production": 0.25,
            "health": 0.20,
            "economics": 0.10,
        }

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set custom culling criteria weights."""
        self._weights.update(weights)

    def score_animal(
        self,
        animal: AnimalRecord,
    ) -> Dict[str, Any]:
        """
        Calculate culling score for animal.

        Args:
            animal: AnimalRecord to evaluate

        Returns:
            Dict with scores by criterion
        """
        scores = {}

        # Age score (older = higher culling score)
        if animal.age_years is not None:
            # Sigmoid function for age scoring
            age_threshold = 8  # years for cattle
            scores["age"] = 100 / (1 + np.exp(-0.5 * (animal.age_years - age_threshold)))
        else:
            scores["age"] = 50  # Unknown age

        # Reproduction score
        repro_data = animal.production_data.get("reproduction", {})
        calving_interval = repro_data.get("calving_interval", 365)
        open_days = repro_data.get("open_days", 0)

        # Penalize long calving intervals and open days
        repro_score = min(100, (calving_interval - 365) / 3 + open_days / 5)
        scores["reproduction"] = max(0, repro_score)

        # Production score (lower production = higher culling score)
        prod_data = animal.production_data
        if "weaning_weight" in prod_data:
            herd_avg = self.benchmark.get_herd_average("weaning_weight") or 250
            deviation = (herd_avg - prod_data["weaning_weight"]) / herd_avg * 100
            scores["production"] = max(0, min(100, deviation + 50))
        else:
            scores["production"] = 50

        # Health score
        health_issues = animal.health_data.get("chronic_conditions", 0)
        vet_visits = animal.health_data.get("vet_visits_year", 0)
        scores["health"] = min(100, health_issues * 25 + vet_visits * 10)

        # Economics score
        net_value = animal.economic_data.get("net_value", 0)
        if net_value < 0:
            scores["economics"] = min(100, abs(net_value) / 10)
        else:
            scores["economics"] = 0

        # Calculate weighted total
        total_score = sum(
            scores[k] * self._weights.get(k, 0.2)
            for k in scores
        )

        return {
            "scores": scores,
            "total_score": total_score,
            "primary_reason": max(scores, key=scores.get),
        }

    def identify_candidates(
        self,
        n: int = 10,
        min_score: float = 50.0,
    ) -> List[CullingCandidate]:
        """
        Identify top culling candidates.

        Args:
            n: Number of candidates to return
            min_score: Minimum culling score

        Returns:
            List of CullingCandidate objects
        """
        active = self.inventory.list_animals(status="active")

        candidates = []
        for animal in active:
            evaluation = self.score_animal(animal)

            if evaluation["total_score"] >= min_score:
                reason = CullingReason[evaluation["primary_reason"].upper()]

                candidates.append(CullingCandidate(
                    animal_id=animal.id,
                    reason=reason,
                    score=evaluation["total_score"],
                    details=f"Primary issue: {reason.value}",
                    economic_impact=animal.economic_data.get("cull_value", 0),
                ))

        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:n]


class EconomicAnalyzer:
    """
    Herd economic analysis.

    Calculates costs, revenues, and profitability
    metrics for herd management decisions.

    Example:
        >>> analyzer = EconomicAnalyzer()
        >>> analyzer.set_costs(feed=2.50, vet=50)
        >>> profit = analyzer.calculate_profit(animal_id)
    """

    def __init__(self):
        """Initialize economic analyzer."""
        self._costs: Dict[str, float] = {
            "feed_per_day": 3.00,  # $/day
            "vet_per_year": 50.00,
            "labor_per_day": 1.00,
            "overhead_per_day": 0.50,
        }
        self._prices: Dict[str, float] = {
            "cull_cow_per_kg": 2.50,
            "feeder_calf_per_kg": 4.00,
            "finished_per_kg": 3.50,
        }
        self._revenues: Dict[str, Dict[str, float]] = {}

    def set_costs(self, **costs) -> None:
        """Set cost parameters."""
        self._costs.update(costs)

    def set_prices(self, **prices) -> None:
        """Set market prices."""
        self._prices.update(prices)

    def record_revenue(
        self,
        animal_id: str,
        revenue_type: str,
        amount: float,
        record_date: Optional[date] = None,
    ) -> None:
        """Record revenue from animal."""
        if animal_id not in self._revenues:
            self._revenues[animal_id] = {}

        if revenue_type not in self._revenues[animal_id]:
            self._revenues[animal_id][revenue_type] = 0

        self._revenues[animal_id][revenue_type] += amount

    def calculate_daily_cost(
        self,
        animal: AnimalRecord,
    ) -> float:
        """
        Calculate daily cost for animal.

        Args:
            animal: AnimalRecord

        Returns:
            Daily cost in $
        """
        # Base costs
        feed = self._costs["feed_per_day"]
        labor = self._costs["labor_per_day"]
        overhead = self._costs["overhead_per_day"]

        # Adjust for animal type
        weight = animal.production_data.get("current_weight", 500)
        feed_adjusted = feed * (weight / 500) ** 0.75

        # Add vet costs (distributed daily)
        vet_daily = self._costs["vet_per_year"] / 365

        return feed_adjusted + labor + overhead + vet_daily

    def calculate_lifetime_cost(
        self,
        animal: AnimalRecord,
    ) -> float:
        """Calculate total lifetime cost for animal."""
        if animal.age_years is None:
            return 0

        days = int(animal.age_years * 365)
        daily_cost = self.calculate_daily_cost(animal)

        return days * daily_cost

    def calculate_value(
        self,
        animal: AnimalRecord,
    ) -> Dict[str, float]:
        """
        Calculate animal value.

        Args:
            animal: AnimalRecord

        Returns:
            Dict with value components
        """
        weight = animal.production_data.get("current_weight", 500)

        # Base value (cull/slaughter)
        if animal.sex.lower() == "female":
            cull_price = self._prices.get("cull_cow_per_kg", 2.50)
        else:
            cull_price = self._prices.get("finished_per_kg", 3.50)

        cull_value = weight * cull_price

        # Breeding value premium
        breeding_premium = 0
        if animal.sex.lower() == "female":
            # Value remaining productive years
            if animal.age_years and animal.age_years < 8:
                remaining_years = 8 - animal.age_years
                calf_value = 250 * self._prices.get("feeder_calf_per_kg", 4.00)
                breeding_premium = remaining_years * calf_value * 0.88 * 0.5

        return {
            "cull_value": cull_value,
            "breeding_value": cull_value + breeding_premium,
            "replacement_cost": cull_value * 1.5,
        }

    def profit_analysis(
        self,
        animal: AnimalRecord,
    ) -> Dict[str, float]:
        """
        Calculate profit/loss for animal.

        Args:
            animal: AnimalRecord

        Returns:
            Dict with profit metrics
        """
        # Get all revenues
        revenues = self._revenues.get(animal.id, {})
        total_revenue = sum(revenues.values())

        # Get costs
        total_cost = self.calculate_lifetime_cost(animal)

        # Current value
        value = self.calculate_value(animal)

        return {
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "current_value": value["breeding_value"],
            "net_profit": total_revenue + value["breeding_value"] - total_cost,
            "roi": (total_revenue + value["breeding_value"] - total_cost) / total_cost
            if total_cost > 0 else 0,
        }


class HerdManager:
    """
    Comprehensive herd management system.

    Integrates inventory, composition, performance,
    culling, and economics for complete herd oversight.

    Example:
        >>> manager = HerdManager(species="cattle")
        >>> manager.add_animal(animal)
        >>> stats = manager.get_composition_stats()
        >>> candidates = manager.identify_culling_candidates()
    """

    def __init__(
        self,
        species: str = "cattle",
        system: ProductionSystem = ProductionSystem.COW_CALF,
    ):
        """
        Initialize herd manager.

        Args:
            species: Animal species
            system: Production system type
        """
        self.species = species.lower()
        self.system = system

        self.inventory = HerdInventory()
        self.composition = HerdComposition(self.inventory, species)
        self.benchmark = PerformanceBenchmark(species)
        self.culling = CullingDecision(self.inventory, self.benchmark)
        self.economics = EconomicAnalyzer()

    def add_animal(
        self,
        animal_id: str,
        breed: str = "",
        sex: str = "unknown",
        birth_date: Optional[date] = None,
        **kwargs,
    ) -> AnimalRecord:
        """
        Add animal to herd.

        Args:
            animal_id: Unique identifier
            breed: Breed name
            sex: Animal sex
            birth_date: Date of birth
            **kwargs: Additional attributes

        Returns:
            Created AnimalRecord
        """
        animal = AnimalRecord(
            id=animal_id,
            species=self.species,
            breed=breed,
            sex=sex,
            birth_date=birth_date,
            production_data=kwargs.get("production_data", {}),
            health_data=kwargs.get("health_data", {}),
            economic_data=kwargs.get("economic_data", {}),
        )

        self.inventory.add_animal(animal)
        return animal

    def remove_animal(
        self,
        animal_id: str,
        reason: str = "",
    ) -> Optional[AnimalRecord]:
        """Remove animal from herd."""
        return self.inventory.remove_animal(animal_id, reason)

    def get_animal(self, animal_id: str) -> Optional[AnimalRecord]:
        """Get animal by ID."""
        return self.inventory.get_animal(animal_id)

    def record_performance(
        self,
        animal_id: str,
        metric: str,
        value: float,
    ) -> None:
        """Record performance metric for animal."""
        self.benchmark.add_metric(metric, animal_id, value)

        # Also update animal's production data
        animal = self.inventory.get_animal(animal_id)
        if animal:
            animal.production_data[metric] = value

    def get_composition_stats(self) -> HerdStats:
        """Get herd composition statistics."""
        return self.composition.get_stats()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance benchmarking report."""
        return self.benchmark.generate_report()

    def identify_culling_candidates(
        self,
        n: int = 10,
    ) -> List[CullingCandidate]:
        """Identify top culling candidates."""
        return self.culling.identify_candidates(n)

    def get_economic_summary(self) -> Dict[str, Any]:
        """Get economic summary for herd."""
        animals = self.inventory.list_animals(status="active")

        total_value = 0
        total_cost = 0
        total_profit = 0

        for animal in animals:
            value = self.economics.calculate_value(animal)
            total_value += value["breeding_value"]

            cost = self.economics.calculate_lifetime_cost(animal)
            total_cost += cost

            profit = self.economics.profit_analysis(animal)
            total_profit += profit["net_profit"]

        return {
            "herd_size": len(animals),
            "total_value": total_value,
            "avg_value": total_value / len(animals) if animals else 0,
            "total_cost": total_cost,
            "total_profit": total_profit,
            "avg_profit_per_head": total_profit / len(animals) if animals else 0,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive herd dashboard."""
        stats = self.get_composition_stats()
        performance = self.get_performance_report()
        economics = self.get_economic_summary()

        return {
            "composition": {
                "total": stats.total_count,
                "active": stats.active_count,
                "by_sex": stats.by_sex,
                "avg_age": stats.avg_age,
            },
            "performance": {
                "overall_percentile": performance.get("overall", {}).get("avg_percentile"),
                "metrics": performance,
            },
            "economics": economics,
            "alerts": self._generate_alerts(stats, performance),
        }

    def _generate_alerts(
        self,
        stats: HerdStats,
        performance: Dict[str, Any],
    ) -> List[str]:
        """Generate management alerts."""
        alerts = []

        # Check sex ratio
        ratio = self.composition.sex_ratio()
        if ratio["ratio"] < 15:
            alerts.append("Low female:male ratio - consider reducing breeding males")

        # Check age distribution
        aged_pct = stats.by_age_group.get("aged", 0) / stats.active_count * 100 \
            if stats.active_count > 0 else 0
        if aged_pct > 20:
            alerts.append(f"High proportion of aged animals ({aged_pct:.1f}%)")

        # Check performance
        overall = performance.get("overall", {}).get("avg_percentile")
        if overall and overall < 40:
            alerts.append("Overall performance below industry average")

        return alerts
