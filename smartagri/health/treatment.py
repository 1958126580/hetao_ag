"""
Treatment Management Module

Medication, vaccination, and treatment tracking:
- Treatment record management
- Withdrawal period tracking
- Vaccination scheduling
- Drug inventory

Example:
    >>> manager = TreatmentManager()
    >>> manager.administer_treatment(animal_id, drug, dose)
    >>> withdrawal = manager.check_withdrawal(animal_id)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DrugCategory(Enum):
    """Drug category classification."""
    ANTIBIOTIC = "antibiotic"
    ANTIPARASITIC = "antiparasitic"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    VACCINE = "vaccine"
    HORMONE = "hormone"
    VITAMIN = "vitamin"
    MINERAL = "mineral"
    OTHER = "other"


class AdministrationRoute(Enum):
    """Drug administration routes."""
    ORAL = "oral"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    INTRAVENOUS = "intravenous"
    TOPICAL = "topical"
    INTRAMAMMARY = "intramammary"
    POUR_ON = "pour_on"


@dataclass
class Drug:
    """
    Drug/medication definition.

    Attributes:
        name: Drug name
        active_ingredient: Active ingredient
        category: Drug category
        withdrawal_meat: Meat withdrawal period (days)
        withdrawal_milk: Milk withdrawal period (days)
        dose_per_kg: Standard dose (mL or mg per kg)
        route: Administration route
        frequency: Treatment frequency (hours)
        duration: Treatment duration (days)
    """
    name: str
    active_ingredient: str
    category: DrugCategory
    withdrawal_meat: int = 0
    withdrawal_milk: int = 0
    dose_per_kg: float = 0.0
    route: AdministrationRoute = AdministrationRoute.INTRAMUSCULAR
    frequency: int = 24  # hours
    duration: int = 1  # days
    notes: str = ""


@dataclass
class TreatmentRecord:
    """
    Treatment administration record.

    Attributes:
        id: Record identifier
        animal_id: Animal treated
        drug_name: Drug administered
        dose: Dose given
        route: Administration route
        administered_by: Person administering
        timestamp: Administration time
        reason: Reason for treatment
        batch_number: Drug batch/lot number
        withdrawal_end: End of withdrawal period
    """
    id: str
    animal_id: str
    drug_name: str
    dose: float
    route: AdministrationRoute
    administered_by: str
    timestamp: datetime
    reason: str = ""
    batch_number: str = ""
    withdrawal_end_meat: Optional[date] = None
    withdrawal_end_milk: Optional[date] = None
    notes: str = ""


@dataclass
class VaccinationRecord:
    """
    Vaccination record.

    Attributes:
        animal_id: Animal vaccinated
        vaccine_name: Vaccine administered
        date: Vaccination date
        dose: Dose given
        batch_number: Vaccine lot number
        next_due: Next vaccination due date
        administered_by: Person administering
    """
    animal_id: str
    vaccine_name: str
    date: date
    dose: float
    batch_number: str = ""
    next_due: Optional[date] = None
    administered_by: str = ""
    reaction: str = ""


@dataclass
class MedicationLog:
    """
    Medication inventory log entry.

    Attributes:
        drug_name: Drug name
        quantity: Quantity in/out
        transaction_type: in/out/adjustment
        date: Transaction date
        batch_number: Batch number
        expiry_date: Expiration date
        notes: Additional notes
    """
    drug_name: str
    quantity: float
    transaction_type: str  # in, out, adjustment
    date: date
    batch_number: str = ""
    expiry_date: Optional[date] = None
    notes: str = ""


# Common drug database
DRUG_DATABASE = {
    "oxytetracycline": Drug(
        name="Oxytetracycline LA",
        active_ingredient="Oxytetracycline",
        category=DrugCategory.ANTIBIOTIC,
        withdrawal_meat=28,
        withdrawal_milk=96,  # hours, but stored as days (4)
        dose_per_kg=0.1,  # mL/kg
        route=AdministrationRoute.INTRAMUSCULAR,
        frequency=72,
        duration=1,
    ),
    "tulathromycin": Drug(
        name="Draxxin",
        active_ingredient="Tulathromycin",
        category=DrugCategory.ANTIBIOTIC,
        withdrawal_meat=18,
        withdrawal_milk=0,  # Not for lactating dairy
        dose_per_kg=0.025,
        route=AdministrationRoute.SUBCUTANEOUS,
        frequency=0,  # Single dose
        duration=1,
    ),
    "flunixin": Drug(
        name="Banamine",
        active_ingredient="Flunixin meglumine",
        category=DrugCategory.ANTI_INFLAMMATORY,
        withdrawal_meat=4,
        withdrawal_milk=36,  # hours
        dose_per_kg=0.022,
        route=AdministrationRoute.INTRAVENOUS,
        frequency=24,
        duration=3,
    ),
    "ivermectin": Drug(
        name="Ivomec",
        active_ingredient="Ivermectin",
        category=DrugCategory.ANTIPARASITIC,
        withdrawal_meat=35,
        withdrawal_milk=0,  # Not for dairy
        dose_per_kg=0.02,
        route=AdministrationRoute.SUBCUTANEOUS,
        frequency=0,
        duration=1,
    ),
    "ceftiofur": Drug(
        name="Excede",
        active_ingredient="Ceftiofur crystalline",
        category=DrugCategory.ANTIBIOTIC,
        withdrawal_meat=13,
        withdrawal_milk=0,
        dose_per_kg=0.066,
        route=AdministrationRoute.SUBCUTANEOUS,
        frequency=0,
        duration=1,
    ),
}


class TreatmentManager:
    """
    Treatment administration and tracking.

    Manages drug administration, withdrawal periods,
    and treatment history.

    Example:
        >>> manager = TreatmentManager()
        >>> manager.administer_treatment(
        ...     animal_id="A001",
        ...     drug_name="oxytetracycline",
        ...     weight=500,
        ...     administered_by="Dr. Smith"
        ... )
        >>> withdrawal = manager.check_withdrawal("A001")
    """

    def __init__(self):
        """Initialize treatment manager."""
        self._treatments: List[TreatmentRecord] = []
        self._drugs = DRUG_DATABASE.copy()
        self._counter = 0

    def register_drug(self, drug: Drug) -> None:
        """Register a new drug in the database."""
        key = drug.name.lower().replace(" ", "_")
        self._drugs[key] = drug

    def get_drug(self, name: str) -> Optional[Drug]:
        """Get drug by name."""
        key = name.lower().replace(" ", "_")
        return self._drugs.get(key)

    def administer_treatment(
        self,
        animal_id: str,
        drug_name: str,
        weight: float,
        administered_by: str,
        reason: str = "",
        custom_dose: Optional[float] = None,
        batch_number: str = "",
        timestamp: Optional[datetime] = None,
    ) -> TreatmentRecord:
        """
        Record treatment administration.

        Args:
            animal_id: Animal identifier
            drug_name: Drug name
            weight: Animal weight (kg)
            administered_by: Person administering
            reason: Treatment reason
            custom_dose: Override calculated dose
            batch_number: Drug batch number
            timestamp: Administration time

        Returns:
            TreatmentRecord
        """
        drug = self.get_drug(drug_name)
        if drug is None:
            raise ValueError(f"Unknown drug: {drug_name}")

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate dose
        if custom_dose is not None:
            dose = custom_dose
        else:
            dose = weight * drug.dose_per_kg

        # Calculate withdrawal end dates
        treatment_date = timestamp.date()
        withdrawal_meat = treatment_date + timedelta(days=drug.withdrawal_meat)
        withdrawal_milk = treatment_date + timedelta(days=drug.withdrawal_milk) \
            if drug.withdrawal_milk > 0 else None

        # Create record
        self._counter += 1
        record = TreatmentRecord(
            id=f"TRT-{self._counter:06d}",
            animal_id=animal_id,
            drug_name=drug.name,
            dose=dose,
            route=drug.route,
            administered_by=administered_by,
            timestamp=timestamp,
            reason=reason,
            batch_number=batch_number,
            withdrawal_end_meat=withdrawal_meat,
            withdrawal_end_milk=withdrawal_milk,
        )

        self._treatments.append(record)
        logger.info(
            f"Treatment {record.id}: {drug.name} administered to {animal_id}"
        )

        return record

    def get_treatments(
        self,
        animal_id: Optional[str] = None,
        drug_name: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[TreatmentRecord]:
        """
        Get treatment records with filters.

        Args:
            animal_id: Filter by animal
            drug_name: Filter by drug
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of TreatmentRecord
        """
        results = self._treatments

        if animal_id:
            results = [t for t in results if t.animal_id == animal_id]
        if drug_name:
            results = [t for t in results
                       if drug_name.lower() in t.drug_name.lower()]
        if start_date:
            results = [t for t in results if t.timestamp.date() >= start_date]
        if end_date:
            results = [t for t in results if t.timestamp.date() <= end_date]

        return results

    def check_withdrawal(
        self,
        animal_id: str,
        check_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Check withdrawal status for animal.

        Args:
            animal_id: Animal identifier
            check_date: Date to check

        Returns:
            Dict with withdrawal status
        """
        if check_date is None:
            check_date = date.today()

        treatments = self.get_treatments(animal_id)

        # Find active withdrawals
        meat_clear = True
        milk_clear = True
        meat_until = None
        milk_until = None
        blocking_treatments = []

        for treatment in treatments:
            if treatment.withdrawal_end_meat and \
               treatment.withdrawal_end_meat > check_date:
                meat_clear = False
                if meat_until is None or treatment.withdrawal_end_meat > meat_until:
                    meat_until = treatment.withdrawal_end_meat
                blocking_treatments.append(treatment.drug_name)

            if treatment.withdrawal_end_milk and \
               treatment.withdrawal_end_milk > check_date:
                milk_clear = False
                if milk_until is None or treatment.withdrawal_end_milk > milk_until:
                    milk_until = treatment.withdrawal_end_milk

        return {
            "animal_id": animal_id,
            "check_date": check_date,
            "meat_clear": meat_clear,
            "milk_clear": milk_clear,
            "meat_withdrawal_until": meat_until,
            "milk_withdrawal_until": milk_until,
            "days_until_meat_clear": (meat_until - check_date).days
                if meat_until else 0,
            "days_until_milk_clear": (milk_until - check_date).days
                if milk_until else 0,
            "blocking_treatments": list(set(blocking_treatments)),
        }

    def get_animals_in_withdrawal(
        self,
        withdrawal_type: str = "meat",
    ) -> List[Dict[str, Any]]:
        """
        Get all animals currently in withdrawal.

        Args:
            withdrawal_type: "meat" or "milk"

        Returns:
            List of animals in withdrawal
        """
        today = date.today()
        animals = {}

        for treatment in self._treatments:
            if withdrawal_type == "meat" and treatment.withdrawal_end_meat:
                if treatment.withdrawal_end_meat > today:
                    if treatment.animal_id not in animals:
                        animals[treatment.animal_id] = {
                            "animal_id": treatment.animal_id,
                            "withdrawal_end": treatment.withdrawal_end_meat,
                            "drugs": [],
                        }
                    animals[treatment.animal_id]["drugs"].append(treatment.drug_name)
                    if treatment.withdrawal_end_meat > \
                       animals[treatment.animal_id]["withdrawal_end"]:
                        animals[treatment.animal_id]["withdrawal_end"] = \
                            treatment.withdrawal_end_meat

            elif withdrawal_type == "milk" and treatment.withdrawal_end_milk:
                if treatment.withdrawal_end_milk > today:
                    if treatment.animal_id not in animals:
                        animals[treatment.animal_id] = {
                            "animal_id": treatment.animal_id,
                            "withdrawal_end": treatment.withdrawal_end_milk,
                            "drugs": [],
                        }
                    animals[treatment.animal_id]["drugs"].append(treatment.drug_name)

        return list(animals.values())


class VaccinationScheduler:
    """
    Vaccination schedule management.

    Manages vaccination protocols, scheduling,
    and compliance tracking.

    Example:
        >>> scheduler = VaccinationScheduler()
        >>> scheduler.create_protocol("BVD", interval_months=6)
        >>> scheduler.record_vaccination(animal_id, "BVD", date)
        >>> due = scheduler.get_vaccinations_due(days=30)
    """

    def __init__(self):
        """Initialize vaccination scheduler."""
        self._vaccinations: List[VaccinationRecord] = []
        self._protocols: Dict[str, Dict[str, Any]] = {}

    def create_protocol(
        self,
        vaccine_name: str,
        interval_months: int,
        initial_doses: int = 1,
        booster_interval_weeks: int = 4,
        start_age_months: int = 2,
    ) -> None:
        """
        Create vaccination protocol.

        Args:
            vaccine_name: Vaccine name
            interval_months: Months between boosters
            initial_doses: Initial series doses
            booster_interval_weeks: Weeks between initial doses
            start_age_months: Minimum age to vaccinate
        """
        self._protocols[vaccine_name] = {
            "interval_months": interval_months,
            "initial_doses": initial_doses,
            "booster_interval_weeks": booster_interval_weeks,
            "start_age_months": start_age_months,
        }

    def record_vaccination(
        self,
        animal_id: str,
        vaccine_name: str,
        vaccination_date: date,
        dose: float = 1.0,
        batch_number: str = "",
        administered_by: str = "",
        reaction: str = "",
    ) -> VaccinationRecord:
        """
        Record vaccination administration.

        Args:
            animal_id: Animal identifier
            vaccine_name: Vaccine name
            vaccination_date: Date of vaccination
            dose: Dose administered
            batch_number: Vaccine lot number
            administered_by: Person administering
            reaction: Any reaction noted

        Returns:
            VaccinationRecord
        """
        # Determine next due date
        protocol = self._protocols.get(vaccine_name)
        if protocol:
            # Check if this is initial series or booster
            prev_vaccinations = self.get_vaccinations(animal_id, vaccine_name)
            if len(prev_vaccinations) < protocol["initial_doses"]:
                next_due = vaccination_date + timedelta(
                    weeks=protocol["booster_interval_weeks"]
                )
            else:
                next_due = vaccination_date + timedelta(
                    days=protocol["interval_months"] * 30
                )
        else:
            next_due = None

        record = VaccinationRecord(
            animal_id=animal_id,
            vaccine_name=vaccine_name,
            date=vaccination_date,
            dose=dose,
            batch_number=batch_number,
            next_due=next_due,
            administered_by=administered_by,
            reaction=reaction,
        )

        self._vaccinations.append(record)
        return record

    def get_vaccinations(
        self,
        animal_id: Optional[str] = None,
        vaccine_name: Optional[str] = None,
    ) -> List[VaccinationRecord]:
        """Get vaccination records."""
        results = self._vaccinations

        if animal_id:
            results = [v for v in results if v.animal_id == animal_id]
        if vaccine_name:
            results = [v for v in results if v.vaccine_name == vaccine_name]

        return sorted(results, key=lambda x: x.date)

    def get_vaccinations_due(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get vaccinations due within period.

        Args:
            days: Days ahead to check

        Returns:
            List of due vaccinations
        """
        today = date.today()
        cutoff = today + timedelta(days=days)

        due_list = []

        # Track latest vaccination per animal per vaccine
        latest: Dict[str, Dict[str, VaccinationRecord]] = {}
        for vac in self._vaccinations:
            if vac.animal_id not in latest:
                latest[vac.animal_id] = {}
            if vac.vaccine_name not in latest[vac.animal_id] or \
               vac.date > latest[vac.animal_id][vac.vaccine_name].date:
                latest[vac.animal_id][vac.vaccine_name] = vac

        # Check for due vaccinations
        for animal_id, vaccines in latest.items():
            for vaccine_name, record in vaccines.items():
                if record.next_due and record.next_due <= cutoff:
                    days_until = (record.next_due - today).days
                    due_list.append({
                        "animal_id": animal_id,
                        "vaccine": vaccine_name,
                        "due_date": record.next_due,
                        "days_until": days_until,
                        "overdue": days_until < 0,
                        "last_vaccination": record.date,
                    })

        return sorted(due_list, key=lambda x: x["due_date"])

    def get_compliance_report(
        self,
        animal_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Generate vaccination compliance report.

        Args:
            animal_ids: Animals to check

        Returns:
            Dict with compliance stats
        """
        today = date.today()
        compliant = 0
        non_compliant = 0
        details = []

        for animal_id in animal_ids:
            animal_status = {"animal_id": animal_id, "vaccines": {}}
            animal_compliant = True

            for vaccine_name in self._protocols:
                vaccinations = self.get_vaccinations(animal_id, vaccine_name)

                if not vaccinations:
                    animal_status["vaccines"][vaccine_name] = "never_vaccinated"
                    animal_compliant = False
                else:
                    latest = vaccinations[-1]
                    if latest.next_due and latest.next_due < today:
                        animal_status["vaccines"][vaccine_name] = "overdue"
                        animal_compliant = False
                    else:
                        animal_status["vaccines"][vaccine_name] = "current"

            if animal_compliant:
                compliant += 1
            else:
                non_compliant += 1

            details.append(animal_status)

        return {
            "total_animals": len(animal_ids),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "compliance_rate": compliant / len(animal_ids) * 100
                if animal_ids else 0,
            "details": details,
        }


class DrugInventory:
    """
    Drug inventory management.

    Tracks drug stock levels, expiration dates,
    and usage patterns.

    Example:
        >>> inventory = DrugInventory()
        >>> inventory.add_stock("oxytetracycline", 100, batch, expiry)
        >>> inventory.use_stock("oxytetracycline", 10)
        >>> low_stock = inventory.get_low_stock_alerts()
    """

    def __init__(self):
        """Initialize drug inventory."""
        self._inventory: Dict[str, List[Dict[str, Any]]] = {}
        self._transactions: List[MedicationLog] = []
        self._reorder_levels: Dict[str, float] = {}

    def set_reorder_level(
        self,
        drug_name: str,
        level: float,
    ) -> None:
        """Set reorder alert level for drug."""
        self._reorder_levels[drug_name] = level

    def add_stock(
        self,
        drug_name: str,
        quantity: float,
        batch_number: str,
        expiry_date: date,
        notes: str = "",
    ) -> None:
        """
        Add drug stock.

        Args:
            drug_name: Drug name
            quantity: Quantity added
            batch_number: Batch number
            expiry_date: Expiration date
            notes: Additional notes
        """
        if drug_name not in self._inventory:
            self._inventory[drug_name] = []

        self._inventory[drug_name].append({
            "batch_number": batch_number,
            "quantity": quantity,
            "expiry_date": expiry_date,
            "added_date": date.today(),
        })

        self._transactions.append(MedicationLog(
            drug_name=drug_name,
            quantity=quantity,
            transaction_type="in",
            date=date.today(),
            batch_number=batch_number,
            expiry_date=expiry_date,
            notes=notes,
        ))

    def use_stock(
        self,
        drug_name: str,
        quantity: float,
    ) -> bool:
        """
        Use drug stock (FIFO by expiry).

        Args:
            drug_name: Drug name
            quantity: Quantity to use

        Returns:
            True if successful
        """
        if drug_name not in self._inventory:
            return False

        # Sort by expiry (FEFO - First Expired First Out)
        batches = sorted(
            self._inventory[drug_name],
            key=lambda x: x["expiry_date"]
        )

        remaining = quantity
        for batch in batches:
            if remaining <= 0:
                break
            if batch["quantity"] > 0:
                used = min(batch["quantity"], remaining)
                batch["quantity"] -= used
                remaining -= used

        if remaining > 0:
            logger.warning(f"Insufficient stock for {drug_name}")
            return False

        self._transactions.append(MedicationLog(
            drug_name=drug_name,
            quantity=-quantity,
            transaction_type="out",
            date=date.today(),
        ))

        return True

    def get_stock_level(
        self,
        drug_name: str,
    ) -> float:
        """Get current stock level for drug."""
        if drug_name not in self._inventory:
            return 0

        return sum(b["quantity"] for b in self._inventory[drug_name])

    def get_expiring_stock(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get stock expiring within period.

        Args:
            days: Days ahead to check

        Returns:
            List of expiring items
        """
        cutoff = date.today() + timedelta(days=days)
        expiring = []

        for drug_name, batches in self._inventory.items():
            for batch in batches:
                if batch["quantity"] > 0 and batch["expiry_date"] <= cutoff:
                    days_until = (batch["expiry_date"] - date.today()).days
                    expiring.append({
                        "drug_name": drug_name,
                        "batch_number": batch["batch_number"],
                        "quantity": batch["quantity"],
                        "expiry_date": batch["expiry_date"],
                        "days_until_expiry": days_until,
                        "expired": days_until < 0,
                    })

        return sorted(expiring, key=lambda x: x["expiry_date"])

    def get_low_stock_alerts(self) -> List[Dict[str, Any]]:
        """Get drugs below reorder level."""
        alerts = []

        for drug_name, level in self._reorder_levels.items():
            current = self.get_stock_level(drug_name)
            if current < level:
                alerts.append({
                    "drug_name": drug_name,
                    "current_stock": current,
                    "reorder_level": level,
                    "shortage": level - current,
                })

        return alerts

    def get_inventory_report(self) -> Dict[str, Any]:
        """Generate inventory report."""
        report = {
            "drugs": {},
            "total_items": 0,
            "expiring_soon": len(self.get_expiring_stock(30)),
            "low_stock": len(self.get_low_stock_alerts()),
        }

        for drug_name in self._inventory:
            level = self.get_stock_level(drug_name)
            if level > 0:
                report["drugs"][drug_name] = level
                report["total_items"] += 1

        return report
