"""
Livestock Reproduction Module

Reproduction management and breeding optimization:
- Estrus detection and prediction
- Breeding scheduling and optimization
- Pregnancy monitoring
- Calving/lambing prediction
- Genetic evaluation and selection

Example:
    >>> manager = ReproductionManager(species="cattle")
    >>> manager.record_heat(animal_id, date)
    >>> prediction = manager.predict_next_heat(animal_id)
    >>> breeding_value = manager.calculate_epd(animal_id, trait="weaning_weight")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReproductiveStatus(Enum):
    """Animal reproductive status."""
    OPEN = "open"  # Not pregnant
    BRED = "bred"  # Recently bred, unconfirmed
    PREGNANT = "pregnant"
    LACTATING = "lactating"
    DRY = "dry"
    PREPUBERTAL = "prepubertal"
    CULLED = "culled"


class BreedingMethod(Enum):
    """Breeding method types."""
    NATURAL = "natural"
    AI = "artificial_insemination"
    ET = "embryo_transfer"
    IVF = "in_vitro_fertilization"


@dataclass
class HeatRecord:
    """Estrus/heat observation record."""
    animal_id: str
    observed_date: date
    intensity: str = "normal"  # weak, normal, strong
    duration_hours: float = 12.0
    detected_by: str = "visual"  # visual, activity, hormone


@dataclass
class BreedingRecord:
    """Breeding event record."""
    animal_id: str
    breeding_date: date
    sire_id: str
    method: BreedingMethod
    technician: str = ""
    semen_lot: str = ""
    notes: str = ""


@dataclass
class PregnancyRecord:
    """Pregnancy check record."""
    animal_id: str
    check_date: date
    pregnant: bool
    days_bred: int
    method: str = "palpation"  # palpation, ultrasound, blood_test
    fetal_sex: Optional[str] = None
    twins: bool = False


@dataclass
class BirthRecord:
    """Birth/calving record."""
    dam_id: str
    sire_id: str
    birth_date: date
    offspring_id: str
    sex: str
    birth_weight: float
    ease_of_birth: int = 1  # 1=unassisted, 2=easy pull, 3=hard pull, 4=surgical
    presentation: str = "normal"
    vigor: int = 3  # 1-5 scale
    notes: str = ""


# Species-specific reproductive parameters
REPRODUCTIVE_PARAMS = {
    "cattle": {
        "heat_cycle_days": 21,
        "heat_duration_hours": 18,
        "gestation_days": 283,
        "puberty_months": 12,
        "breeding_season": None,  # Year-round
        "conception_rate": 0.55,
        "twins_rate": 0.02,
    },
    "sheep": {
        "heat_cycle_days": 17,
        "heat_duration_hours": 30,
        "gestation_days": 147,
        "puberty_months": 7,
        "breeding_season": (8, 12),  # Aug-Dec in N. hemisphere
        "conception_rate": 0.70,
        "twins_rate": 0.15,
    },
    "swine": {
        "heat_cycle_days": 21,
        "heat_duration_hours": 48,
        "gestation_days": 114,
        "puberty_months": 6,
        "breeding_season": None,
        "conception_rate": 0.85,
        "twins_rate": 0.0,  # Swine have litters
        "litter_size": 12,
    },
    "goat": {
        "heat_cycle_days": 21,
        "heat_duration_hours": 36,
        "gestation_days": 150,
        "puberty_months": 7,
        "breeding_season": (8, 1),  # Aug-Jan
        "conception_rate": 0.70,
        "twins_rate": 0.50,
    },
    "horse": {
        "heat_cycle_days": 21,
        "heat_duration_hours": 120,
        "gestation_days": 340,
        "puberty_months": 18,
        "breeding_season": (3, 7),  # Spring/summer
        "conception_rate": 0.60,
        "twins_rate": 0.01,
    },
}


class EstrusDetector:
    """
    Estrus (heat) detection and prediction.

    Uses historical heat records to predict future
    estrus events for breeding scheduling.

    Example:
        >>> detector = EstrusDetector(species="cattle")
        >>> detector.record_heat(animal_id, observed_date)
        >>> next_heat = detector.predict_next_heat(animal_id)
    """

    def __init__(self, species: str):
        """
        Initialize estrus detector.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.params = REPRODUCTIVE_PARAMS.get(
            self.species,
            REPRODUCTIVE_PARAMS["cattle"]
        )
        self._heat_records: Dict[str, List[HeatRecord]] = {}

    def record_heat(
        self,
        animal_id: str,
        observed_date: date,
        intensity: str = "normal",
        duration_hours: float = None,
    ) -> HeatRecord:
        """
        Record estrus observation.

        Args:
            animal_id: Animal identifier
            observed_date: Date heat observed
            intensity: Heat intensity
            duration_hours: Duration in hours

        Returns:
            HeatRecord object
        """
        if duration_hours is None:
            duration_hours = self.params["heat_duration_hours"]

        record = HeatRecord(
            animal_id=animal_id,
            observed_date=observed_date,
            intensity=intensity,
            duration_hours=duration_hours,
        )

        if animal_id not in self._heat_records:
            self._heat_records[animal_id] = []

        self._heat_records[animal_id].append(record)
        self._heat_records[animal_id].sort(key=lambda x: x.observed_date)

        return record

    def get_heat_history(
        self,
        animal_id: str,
    ) -> List[HeatRecord]:
        """Get heat history for animal."""
        return self._heat_records.get(animal_id, [])

    def calculate_cycle_length(
        self,
        animal_id: str,
    ) -> Optional[float]:
        """
        Calculate average estrus cycle length.

        Args:
            animal_id: Animal identifier

        Returns:
            Average cycle length in days
        """
        records = self.get_heat_history(animal_id)

        if len(records) < 2:
            return None

        # Calculate intervals between heats
        intervals = []
        for i in range(1, len(records)):
            days = (records[i].observed_date - records[i-1].observed_date).days

            # Filter out unlikely intervals (should be ~1 cycle)
            expected = self.params["heat_cycle_days"]
            if expected * 0.5 <= days <= expected * 2.5:
                intervals.append(days)

        if not intervals:
            return None

        return np.mean(intervals)

    def predict_next_heat(
        self,
        animal_id: str,
        from_date: Optional[date] = None,
    ) -> Optional[date]:
        """
        Predict next estrus date.

        Args:
            animal_id: Animal identifier
            from_date: Reference date

        Returns:
            Predicted next heat date
        """
        records = self.get_heat_history(animal_id)

        if not records:
            return None

        if from_date is None:
            from_date = date.today()

        last_heat = records[-1].observed_date

        # Use calculated cycle or default
        cycle_length = self.calculate_cycle_length(animal_id)
        if cycle_length is None:
            cycle_length = self.params["heat_cycle_days"]

        # Find next heat after from_date
        next_heat = last_heat
        while next_heat <= from_date:
            next_heat = next_heat + timedelta(days=int(cycle_length))

        return next_heat

    def optimal_breeding_window(
        self,
        heat_start: date,
    ) -> Tuple[datetime, datetime]:
        """
        Calculate optimal breeding window.

        Args:
            heat_start: Start of heat

        Returns:
            Tuple of (optimal_start, optimal_end) datetimes
        """
        # Optimal breeding is typically 12-18 hours after heat onset
        heat_duration = self.params["heat_duration_hours"]

        optimal_start = datetime.combine(heat_start, datetime.min.time())
        optimal_start += timedelta(hours=max(6, heat_duration * 0.3))

        optimal_end = datetime.combine(heat_start, datetime.min.time())
        optimal_end += timedelta(hours=min(24, heat_duration * 0.75))

        return optimal_start, optimal_end


class BreedingManager:
    """
    Breeding event management.

    Tracks breeding events, conception rates,
    and breeding performance.

    Example:
        >>> manager = BreedingManager(species="cattle")
        >>> manager.record_breeding(dam_id, sire_id, date, method)
        >>> stats = manager.conception_rate(sire_id)
    """

    def __init__(self, species: str):
        """
        Initialize breeding manager.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.params = REPRODUCTIVE_PARAMS.get(
            self.species,
            REPRODUCTIVE_PARAMS["cattle"]
        )
        self._breeding_records: List[BreedingRecord] = []
        self._pregnancy_results: Dict[str, bool] = {}  # breeding_key -> pregnant

    def record_breeding(
        self,
        animal_id: str,
        sire_id: str,
        breeding_date: date,
        method: BreedingMethod = BreedingMethod.NATURAL,
        technician: str = "",
        semen_lot: str = "",
    ) -> BreedingRecord:
        """
        Record breeding event.

        Args:
            animal_id: Female animal ID
            sire_id: Sire ID
            breeding_date: Date of breeding
            method: Breeding method
            technician: AI technician name
            semen_lot: Semen lot number

        Returns:
            BreedingRecord object
        """
        record = BreedingRecord(
            animal_id=animal_id,
            breeding_date=breeding_date,
            sire_id=sire_id,
            method=method,
            technician=technician,
            semen_lot=semen_lot,
        )

        self._breeding_records.append(record)
        return record

    def record_pregnancy_result(
        self,
        animal_id: str,
        breeding_date: date,
        pregnant: bool,
    ) -> None:
        """
        Record pregnancy check result.

        Args:
            animal_id: Animal ID
            breeding_date: Original breeding date
            pregnant: Whether pregnant
        """
        key = f"{animal_id}_{breeding_date}"
        self._pregnancy_results[key] = pregnant

    def get_breedings(
        self,
        animal_id: Optional[str] = None,
        sire_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[BreedingRecord]:
        """
        Get breeding records with filters.

        Args:
            animal_id: Filter by female
            sire_id: Filter by sire
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of matching BreedingRecord objects
        """
        results = self._breeding_records

        if animal_id:
            results = [r for r in results if r.animal_id == animal_id]
        if sire_id:
            results = [r for r in results if r.sire_id == sire_id]
        if start_date:
            results = [r for r in results if r.breeding_date >= start_date]
        if end_date:
            results = [r for r in results if r.breeding_date <= end_date]

        return results

    def conception_rate(
        self,
        sire_id: Optional[str] = None,
        method: Optional[BreedingMethod] = None,
    ) -> Dict[str, Any]:
        """
        Calculate conception rate.

        Args:
            sire_id: Filter by sire
            method: Filter by breeding method

        Returns:
            Dict with conception statistics
        """
        # Filter relevant breedings
        breedings = self._breeding_records

        if sire_id:
            breedings = [b for b in breedings if b.sire_id == sire_id]
        if method:
            breedings = [b for b in breedings if b.method == method]

        # Count confirmed pregnancies
        confirmed = 0
        open_count = 0
        unknown = 0

        for breeding in breedings:
            key = f"{breeding.animal_id}_{breeding.breeding_date}"
            if key in self._pregnancy_results:
                if self._pregnancy_results[key]:
                    confirmed += 1
                else:
                    open_count += 1
            else:
                unknown += 1

        total_checked = confirmed + open_count

        return {
            "total_breedings": len(breedings),
            "confirmed_pregnant": confirmed,
            "confirmed_open": open_count,
            "unknown": unknown,
            "conception_rate": confirmed / total_checked if total_checked > 0 else None,
        }

    def services_per_conception(
        self,
        animal_id: Optional[str] = None,
    ) -> float:
        """
        Calculate average services per conception.

        Args:
            animal_id: Filter by animal

        Returns:
            Average services per conception
        """
        breedings = self._breeding_records
        if animal_id:
            breedings = [b for b in breedings if b.animal_id == animal_id]

        # Group by animal
        animal_services: Dict[str, int] = {}
        animal_conceptions: Dict[str, int] = {}

        for breeding in breedings:
            aid = breeding.animal_id
            if aid not in animal_services:
                animal_services[aid] = 0
                animal_conceptions[aid] = 0

            animal_services[aid] += 1

            key = f"{aid}_{breeding.breeding_date}"
            if self._pregnancy_results.get(key, False):
                animal_conceptions[aid] += 1

        total_services = sum(animal_services.values())
        total_conceptions = sum(animal_conceptions.values())

        return total_services / total_conceptions if total_conceptions > 0 else float('inf')


class PregnancyManager:
    """
    Pregnancy monitoring and management.

    Tracks pregnancy status, predicts due dates,
    and monitors gestation progress.

    Example:
        >>> manager = PregnancyManager(species="cattle")
        >>> manager.record_pregnancy_check(animal_id, date, pregnant=True)
        >>> due_date = manager.predict_due_date(animal_id)
    """

    def __init__(self, species: str):
        """
        Initialize pregnancy manager.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.params = REPRODUCTIVE_PARAMS.get(
            self.species,
            REPRODUCTIVE_PARAMS["cattle"]
        )
        self._pregnancy_records: Dict[str, List[PregnancyRecord]] = {}
        self._breeding_dates: Dict[str, date] = {}  # Last breeding date per animal

    def set_breeding_date(
        self,
        animal_id: str,
        breeding_date: date,
    ) -> None:
        """Set last breeding date for animal."""
        self._breeding_dates[animal_id] = breeding_date

    def record_pregnancy_check(
        self,
        animal_id: str,
        check_date: date,
        pregnant: bool,
        method: str = "palpation",
        fetal_sex: Optional[str] = None,
        twins: bool = False,
    ) -> PregnancyRecord:
        """
        Record pregnancy check result.

        Args:
            animal_id: Animal identifier
            check_date: Date of check
            pregnant: Whether pregnant
            method: Detection method
            fetal_sex: Fetal sex if determined
            twins: Whether twins detected

        Returns:
            PregnancyRecord object
        """
        breeding_date = self._breeding_dates.get(animal_id)
        days_bred = (check_date - breeding_date).days if breeding_date else 0

        record = PregnancyRecord(
            animal_id=animal_id,
            check_date=check_date,
            pregnant=pregnant,
            days_bred=days_bred,
            method=method,
            fetal_sex=fetal_sex,
            twins=twins,
        )

        if animal_id not in self._pregnancy_records:
            self._pregnancy_records[animal_id] = []

        self._pregnancy_records[animal_id].append(record)
        return record

    def predict_due_date(
        self,
        animal_id: str,
    ) -> Optional[date]:
        """
        Predict due date based on breeding date.

        Args:
            animal_id: Animal identifier

        Returns:
            Predicted due date
        """
        breeding_date = self._breeding_dates.get(animal_id)

        if breeding_date is None:
            return None

        gestation = self.params["gestation_days"]
        return breeding_date + timedelta(days=gestation)

    def days_until_due(
        self,
        animal_id: str,
        as_of: Optional[date] = None,
    ) -> Optional[int]:
        """
        Calculate days until due date.

        Args:
            animal_id: Animal identifier
            as_of: Reference date

        Returns:
            Days until due
        """
        due_date = self.predict_due_date(animal_id)

        if due_date is None:
            return None

        if as_of is None:
            as_of = date.today()

        return (due_date - as_of).days

    def gestation_stage(
        self,
        animal_id: str,
        as_of: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Determine current gestation stage.

        Args:
            animal_id: Animal identifier
            as_of: Reference date

        Returns:
            Dict with gestation information
        """
        breeding_date = self._breeding_dates.get(animal_id)

        if breeding_date is None:
            return {"status": "not_bred"}

        if as_of is None:
            as_of = date.today()

        days_bred = (as_of - breeding_date).days
        gestation = self.params["gestation_days"]
        progress = days_bred / gestation

        # Determine trimester
        if progress < 0.33:
            trimester = 1
            stage = "early"
        elif progress < 0.67:
            trimester = 2
            stage = "mid"
        else:
            trimester = 3
            stage = "late"

        return {
            "days_bred": days_bred,
            "gestation_length": gestation,
            "progress_pct": progress * 100,
            "trimester": trimester,
            "stage": stage,
            "due_date": self.predict_due_date(animal_id),
            "days_remaining": gestation - days_bred,
        }

    def get_animals_due_soon(
        self,
        days_window: int = 14,
    ) -> List[Tuple[str, date, int]]:
        """
        Get animals due within specified window.

        Args:
            days_window: Days ahead to check

        Returns:
            List of (animal_id, due_date, days_until_due)
        """
        today = date.today()
        cutoff = today + timedelta(days=days_window)

        due_animals = []
        for animal_id in self._breeding_dates:
            due_date = self.predict_due_date(animal_id)
            if due_date and today <= due_date <= cutoff:
                days_until = (due_date - today).days
                due_animals.append((animal_id, due_date, days_until))

        # Sort by due date
        due_animals.sort(key=lambda x: x[1])
        return due_animals


class CalvingManager:
    """
    Birth/calving event management.

    Records and analyzes birth events, calving ease,
    and offspring performance.

    Example:
        >>> manager = CalvingManager()
        >>> manager.record_birth(dam_id, sire_id, date, offspring_id, ...)
        >>> stats = manager.calving_ease_score(sire_id)
    """

    def __init__(self):
        """Initialize calving manager."""
        self._birth_records: List[BirthRecord] = []

    def record_birth(
        self,
        dam_id: str,
        sire_id: str,
        birth_date: date,
        offspring_id: str,
        sex: str,
        birth_weight: float,
        ease_of_birth: int = 1,
        presentation: str = "normal",
        vigor: int = 3,
        notes: str = "",
    ) -> BirthRecord:
        """
        Record birth event.

        Args:
            dam_id: Mother's ID
            sire_id: Father's ID
            birth_date: Date of birth
            offspring_id: New animal ID
            sex: Offspring sex
            birth_weight: Birth weight (kg)
            ease_of_birth: Calving ease score (1-4)
            presentation: Birth presentation
            vigor: Newborn vigor score
            notes: Additional notes

        Returns:
            BirthRecord object
        """
        record = BirthRecord(
            dam_id=dam_id,
            sire_id=sire_id,
            birth_date=birth_date,
            offspring_id=offspring_id,
            sex=sex,
            birth_weight=birth_weight,
            ease_of_birth=ease_of_birth,
            presentation=presentation,
            vigor=vigor,
            notes=notes,
        )

        self._birth_records.append(record)
        return record

    def get_births(
        self,
        dam_id: Optional[str] = None,
        sire_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[BirthRecord]:
        """
        Get birth records with filters.

        Args:
            dam_id: Filter by dam
            sire_id: Filter by sire
            year: Filter by birth year

        Returns:
            List of BirthRecord objects
        """
        results = self._birth_records

        if dam_id:
            results = [r for r in results if r.dam_id == dam_id]
        if sire_id:
            results = [r for r in results if r.sire_id == sire_id]
        if year:
            results = [r for r in results if r.birth_date.year == year]

        return results

    def calving_ease_score(
        self,
        sire_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate calving ease statistics.

        Args:
            sire_id: Filter by sire

        Returns:
            Dict with calving ease stats
        """
        births = self.get_births(sire_id=sire_id)

        if not births:
            return {"n": 0}

        ease_scores = [b.ease_of_birth for b in births]

        # Distribution by score
        distribution = {}
        for score in range(1, 5):
            distribution[score] = sum(1 for e in ease_scores if e == score)

        # Percent unassisted (score 1)
        pct_unassisted = distribution.get(1, 0) / len(births) * 100

        return {
            "n": len(births),
            "mean_ease": np.mean(ease_scores),
            "distribution": distribution,
            "pct_unassisted": pct_unassisted,
            "pct_assisted": 100 - pct_unassisted,
        }

    def birth_weight_stats(
        self,
        sire_id: Optional[str] = None,
        sex: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calculate birth weight statistics.

        Args:
            sire_id: Filter by sire
            sex: Filter by sex

        Returns:
            Dict with birth weight stats
        """
        births = self.get_births(sire_id=sire_id)

        if sex:
            births = [b for b in births if b.sex.lower() == sex.lower()]

        if not births:
            return {"n": 0}

        weights = [b.birth_weight for b in births]

        return {
            "n": len(births),
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": np.min(weights),
            "max": np.max(weights),
            "cv": np.std(weights) / np.mean(weights) * 100,
        }


class GeneticEvaluator:
    """
    Genetic evaluation and EPD calculation.

    Calculates Expected Progeny Differences (EPDs)
    and breeding values for genetic selection.

    Example:
        >>> evaluator = GeneticEvaluator()
        >>> evaluator.add_performance(animal_id, trait, value)
        >>> epd = evaluator.calculate_epd(animal_id, trait)
    """

    def __init__(self):
        """Initialize genetic evaluator."""
        self._performance: Dict[str, Dict[str, List[float]]] = {}
        self._pedigree: Dict[str, Tuple[str, str]] = {}  # animal -> (sire, dam)

        # Trait heritabilities
        self._heritabilities = {
            "birth_weight": 0.40,
            "weaning_weight": 0.30,
            "yearling_weight": 0.40,
            "milk": 0.25,
            "marbling": 0.35,
            "ribeye_area": 0.40,
            "backfat": 0.40,
            "scrotal_circumference": 0.45,
            "calving_ease": 0.15,
            "docility": 0.40,
        }

    def set_pedigree(
        self,
        animal_id: str,
        sire_id: str,
        dam_id: str,
    ) -> None:
        """
        Set pedigree information.

        Args:
            animal_id: Animal identifier
            sire_id: Sire identifier
            dam_id: Dam identifier
        """
        self._pedigree[animal_id] = (sire_id, dam_id)

    def add_performance(
        self,
        animal_id: str,
        trait: str,
        value: float,
    ) -> None:
        """
        Add performance record.

        Args:
            animal_id: Animal identifier
            trait: Trait name
            value: Measured value
        """
        if animal_id not in self._performance:
            self._performance[animal_id] = {}

        if trait not in self._performance[animal_id]:
            self._performance[animal_id][trait] = []

        self._performance[animal_id][trait].append(value)

    def calculate_epd(
        self,
        animal_id: str,
        trait: str,
    ) -> Dict[str, float]:
        """
        Calculate Expected Progeny Difference.

        Args:
            animal_id: Animal identifier
            trait: Trait name

        Returns:
            Dict with EPD and accuracy
        """
        h2 = self._heritabilities.get(trait, 0.30)

        # Get own performance
        own_perf = self._performance.get(animal_id, {}).get(trait, [])

        # Get progeny performance
        progeny_perf = []
        for aid, (sire, dam) in self._pedigree.items():
            if sire == animal_id or dam == animal_id:
                perf = self._performance.get(aid, {}).get(trait, [])
                progeny_perf.extend(perf)

        # Calculate population mean
        all_values = []
        for aid, traits in self._performance.items():
            if trait in traits:
                all_values.extend(traits[trait])

        pop_mean = np.mean(all_values) if all_values else 0

        # Simple EPD calculation
        n_own = len(own_perf)
        n_prog = len(progeny_perf)

        own_dev = np.mean(own_perf) - pop_mean if own_perf else 0
        prog_dev = np.mean(progeny_perf) - pop_mean if progeny_perf else 0

        # Weighted combination
        if n_own + n_prog == 0:
            epd = 0
            accuracy = 0
        else:
            # Weight by reliability
            own_weight = h2 * n_own / (1 + (n_own - 1) * 0.25 * h2)
            prog_weight = 2 * h2 * n_prog / (4 + (n_prog - 1) * h2)

            total_weight = own_weight + prog_weight

            if total_weight > 0:
                epd = (own_weight * own_dev + prog_weight * prog_dev) / total_weight
                accuracy = min(0.99, np.sqrt(total_weight / (1 + total_weight)))
            else:
                epd = 0
                accuracy = 0

        return {
            "epd": epd,
            "accuracy": accuracy,
            "n_own_records": n_own,
            "n_progeny_records": n_prog,
        }

    def rank_animals(
        self,
        trait: str,
        top_n: int = 10,
    ) -> List[Tuple[str, float, float]]:
        """
        Rank animals by EPD for trait.

        Args:
            trait: Trait name
            top_n: Number of top animals

        Returns:
            List of (animal_id, epd, accuracy)
        """
        rankings = []

        for animal_id in self._performance:
            result = self.calculate_epd(animal_id, trait)
            rankings.append((
                animal_id,
                result["epd"],
                result["accuracy"],
            ))

        # Sort by EPD (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings[:top_n]


class ReproductionManager:
    """
    Comprehensive reproduction management system.

    Integrates estrus detection, breeding, pregnancy,
    and calving management for complete reproductive oversight.

    Example:
        >>> manager = ReproductionManager(species="cattle")
        >>> manager.record_heat(animal_id, date)
        >>> manager.record_breeding(dam_id, sire_id, date)
        >>> due_date = manager.predict_due_date(animal_id)
    """

    def __init__(self, species: str):
        """
        Initialize reproduction manager.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.estrus = EstrusDetector(species)
        self.breeding = BreedingManager(species)
        self.pregnancy = PregnancyManager(species)
        self.calving = CalvingManager()
        self.genetics = GeneticEvaluator()

        self._status: Dict[str, ReproductiveStatus] = {}

    def set_status(
        self,
        animal_id: str,
        status: ReproductiveStatus,
    ) -> None:
        """Set reproductive status for animal."""
        self._status[animal_id] = status

    def get_status(
        self,
        animal_id: str,
    ) -> ReproductiveStatus:
        """Get reproductive status for animal."""
        return self._status.get(animal_id, ReproductiveStatus.OPEN)

    def record_heat(
        self,
        animal_id: str,
        observed_date: date,
        intensity: str = "normal",
    ) -> HeatRecord:
        """Record heat observation."""
        return self.estrus.record_heat(animal_id, observed_date, intensity)

    def predict_next_heat(
        self,
        animal_id: str,
    ) -> Optional[date]:
        """Predict next heat date."""
        return self.estrus.predict_next_heat(animal_id)

    def record_breeding(
        self,
        animal_id: str,
        sire_id: str,
        breeding_date: date,
        method: BreedingMethod = BreedingMethod.AI,
    ) -> BreedingRecord:
        """Record breeding event."""
        self.set_status(animal_id, ReproductiveStatus.BRED)
        self.pregnancy.set_breeding_date(animal_id, breeding_date)

        # Set pedigree for future offspring
        return self.breeding.record_breeding(
            animal_id, sire_id, breeding_date, method
        )

    def confirm_pregnancy(
        self,
        animal_id: str,
        check_date: date,
        pregnant: bool,
    ) -> PregnancyRecord:
        """Confirm pregnancy status."""
        if pregnant:
            self.set_status(animal_id, ReproductiveStatus.PREGNANT)
        else:
            self.set_status(animal_id, ReproductiveStatus.OPEN)

        return self.pregnancy.record_pregnancy_check(
            animal_id, check_date, pregnant
        )

    def predict_due_date(
        self,
        animal_id: str,
    ) -> Optional[date]:
        """Predict due date."""
        return self.pregnancy.predict_due_date(animal_id)

    def record_birth(
        self,
        dam_id: str,
        sire_id: str,
        birth_date: date,
        offspring_id: str,
        sex: str,
        birth_weight: float,
        ease_of_birth: int = 1,
    ) -> BirthRecord:
        """Record birth event."""
        self.set_status(dam_id, ReproductiveStatus.LACTATING)

        # Set pedigree
        self.genetics.set_pedigree(offspring_id, sire_id, dam_id)

        # Record birth weight
        self.genetics.add_performance(offspring_id, "birth_weight", birth_weight)

        return self.calving.record_birth(
            dam_id, sire_id, birth_date, offspring_id,
            sex, birth_weight, ease_of_birth
        )

    def get_reproduction_summary(self) -> Dict[str, Any]:
        """Get summary of reproduction metrics."""
        # Count by status
        status_counts = {}
        for status in ReproductiveStatus:
            status_counts[status.value] = sum(
                1 for s in self._status.values() if s == status
            )

        # Conception rate
        cr = self.breeding.conception_rate()

        return {
            "status_counts": status_counts,
            "conception_rate": cr.get("conception_rate"),
            "services_per_conception": self.breeding.services_per_conception(),
            "animals_due_soon": len(self.pregnancy.get_animals_due_soon(14)),
        }
