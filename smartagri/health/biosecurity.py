"""
Biosecurity Management Module

Biosecurity protocols and risk management:
- Quarantine zone management
- Risk assessment
- Movement tracking
- Visitor management
- Disease prevention protocols

Example:
    >>> manager = BiosecurityManager()
    >>> manager.create_quarantine_zone("Isolation", capacity=10)
    >>> manager.add_to_quarantine(animal_id, zone, reason)
    >>> risk = manager.assess_risk()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Biosecurity risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class MovementType(Enum):
    """Animal movement types."""
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    INTERNAL = "internal"
    RETURN = "return"


class VisitorType(Enum):
    """Visitor classification."""
    EMPLOYEE = "employee"
    VETERINARIAN = "veterinarian"
    INSPECTOR = "inspector"
    DELIVERY = "delivery"
    CONTRACTOR = "contractor"
    OTHER = "other"


@dataclass
class QuarantineZone:
    """
    Quarantine zone definition.

    Attributes:
        id: Zone identifier
        name: Zone name
        capacity: Maximum capacity
        current_count: Current animal count
        animals: List of animal IDs
        created_date: Creation date
        min_days: Minimum quarantine days
        status: Zone status (active/inactive)
    """
    id: str
    name: str
    capacity: int
    current_count: int = 0
    animals: List[str] = field(default_factory=list)
    created_date: date = field(default_factory=date.today)
    min_days: int = 21
    status: str = "active"


@dataclass
class QuarantineRecord:
    """
    Quarantine record for animal.

    Attributes:
        animal_id: Animal identifier
        zone_id: Quarantine zone ID
        entry_date: Entry date
        reason: Quarantine reason
        release_date: Actual release date
        expected_release: Expected release date
        health_checks: Health check records
        status: Current status
    """
    animal_id: str
    zone_id: str
    entry_date: date
    reason: str
    release_date: Optional[date] = None
    expected_release: Optional[date] = None
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # active, cleared, extended, removed


@dataclass
class MovementRecord:
    """
    Animal movement record.

    Attributes:
        animal_id: Animal identifier
        movement_type: Type of movement
        date: Movement date
        origin: Origin location
        destination: Destination location
        purpose: Purpose of movement
        documentation: Required documents
        health_certificate: Health certificate ID
    """
    animal_id: str
    movement_type: MovementType
    date: date
    origin: str
    destination: str
    purpose: str = ""
    documentation: List[str] = field(default_factory=list)
    health_certificate: str = ""


@dataclass
class VisitorRecord:
    """
    Visitor log record.

    Attributes:
        name: Visitor name
        visitor_type: Type of visitor
        date: Visit date
        time_in: Check-in time
        time_out: Check-out time
        areas_visited: Areas accessed
        vehicle_id: Vehicle identification
        last_farm_visit: Last farm visited
        health_declaration: Health declaration signed
    """
    name: str
    visitor_type: VisitorType
    date: date
    time_in: datetime
    time_out: Optional[datetime] = None
    areas_visited: List[str] = field(default_factory=list)
    vehicle_id: str = ""
    last_farm_visit: str = ""
    health_declaration: bool = False


@dataclass
class RiskAssessment:
    """
    Biosecurity risk assessment result.

    Attributes:
        assessment_date: Date of assessment
        overall_risk: Overall risk level
        risk_factors: Identified risk factors
        score: Numerical score (0-100)
        recommendations: Risk mitigation recommendations
    """
    assessment_date: date
    overall_risk: RiskLevel
    risk_factors: List[Dict[str, Any]]
    score: float
    recommendations: List[str]


class QuarantineManager:
    """
    Quarantine zone and protocol management.

    Manages isolation areas, quarantine periods,
    and health monitoring for incoming animals.

    Example:
        >>> manager = QuarantineManager()
        >>> zone = manager.create_zone("Isolation A", capacity=20)
        >>> manager.add_animal(animal_id, zone.id, "new_purchase")
    """

    def __init__(self, default_quarantine_days: int = 21):
        """
        Initialize quarantine manager.

        Args:
            default_quarantine_days: Default quarantine period
        """
        self.default_days = default_quarantine_days
        self._zones: Dict[str, QuarantineZone] = {}
        self._records: Dict[str, QuarantineRecord] = {}
        self._zone_counter = 0

    def create_zone(
        self,
        name: str,
        capacity: int,
        min_days: Optional[int] = None,
    ) -> QuarantineZone:
        """
        Create quarantine zone.

        Args:
            name: Zone name
            capacity: Maximum capacity
            min_days: Minimum quarantine days

        Returns:
            QuarantineZone object
        """
        self._zone_counter += 1
        zone_id = f"QZ-{self._zone_counter:03d}"

        zone = QuarantineZone(
            id=zone_id,
            name=name,
            capacity=capacity,
            min_days=min_days or self.default_days,
        )

        self._zones[zone_id] = zone
        logger.info(f"Created quarantine zone {zone_id}: {name}")

        return zone

    def get_zone(self, zone_id: str) -> Optional[QuarantineZone]:
        """Get quarantine zone by ID."""
        return self._zones.get(zone_id)

    def list_zones(
        self,
        status: Optional[str] = None,
    ) -> List[QuarantineZone]:
        """List quarantine zones."""
        zones = list(self._zones.values())
        if status:
            zones = [z for z in zones if z.status == status]
        return zones

    def add_animal(
        self,
        animal_id: str,
        zone_id: str,
        reason: str,
        entry_date: Optional[date] = None,
    ) -> QuarantineRecord:
        """
        Add animal to quarantine.

        Args:
            animal_id: Animal identifier
            zone_id: Quarantine zone ID
            reason: Quarantine reason
            entry_date: Entry date

        Returns:
            QuarantineRecord
        """
        zone = self._zones.get(zone_id)
        if zone is None:
            raise ValueError(f"Unknown zone: {zone_id}")

        if zone.current_count >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")

        if entry_date is None:
            entry_date = date.today()

        expected_release = entry_date + timedelta(days=zone.min_days)

        record = QuarantineRecord(
            animal_id=animal_id,
            zone_id=zone_id,
            entry_date=entry_date,
            reason=reason,
            expected_release=expected_release,
        )

        self._records[animal_id] = record
        zone.animals.append(animal_id)
        zone.current_count += 1

        logger.info(f"Animal {animal_id} added to quarantine zone {zone_id}")

        return record

    def record_health_check(
        self,
        animal_id: str,
        check_date: date,
        healthy: bool,
        notes: str = "",
        checker: str = "",
    ) -> None:
        """
        Record health check during quarantine.

        Args:
            animal_id: Animal identifier
            check_date: Check date
            healthy: Health status
            notes: Check notes
            checker: Person performing check
        """
        record = self._records.get(animal_id)
        if record is None:
            raise ValueError(f"Animal {animal_id} not in quarantine")

        record.health_checks.append({
            "date": check_date,
            "healthy": healthy,
            "notes": notes,
            "checker": checker,
        })

        # Extend quarantine if unhealthy
        if not healthy and record.expected_release:
            zone = self._zones.get(record.zone_id)
            if zone:
                record.expected_release = check_date + timedelta(days=zone.min_days)
                record.status = "extended"

    def release_animal(
        self,
        animal_id: str,
        release_date: Optional[date] = None,
    ) -> bool:
        """
        Release animal from quarantine.

        Args:
            animal_id: Animal identifier
            release_date: Release date

        Returns:
            True if released successfully
        """
        record = self._records.get(animal_id)
        if record is None:
            return False

        if release_date is None:
            release_date = date.today()

        # Check if minimum period met
        zone = self._zones.get(record.zone_id)
        days_in = (release_date - record.entry_date).days

        if zone and days_in < zone.min_days:
            logger.warning(
                f"Early release: {days_in} days (min: {zone.min_days})"
            )

        # Check health status
        if record.health_checks:
            latest = record.health_checks[-1]
            if not latest.get("healthy", True):
                logger.warning("Releasing animal with unhealthy status")

        record.release_date = release_date
        record.status = "cleared"

        # Update zone
        if zone and animal_id in zone.animals:
            zone.animals.remove(animal_id)
            zone.current_count -= 1

        logger.info(f"Animal {animal_id} released from quarantine")

        return True

    def get_status(
        self,
        animal_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get quarantine status for animal.

        Args:
            animal_id: Animal identifier

        Returns:
            Dict with quarantine status
        """
        record = self._records.get(animal_id)
        if record is None:
            return None

        days_in = (date.today() - record.entry_date).days
        zone = self._zones.get(record.zone_id)

        return {
            "animal_id": animal_id,
            "zone": zone.name if zone else record.zone_id,
            "entry_date": record.entry_date,
            "days_in_quarantine": days_in,
            "expected_release": record.expected_release,
            "days_remaining": (record.expected_release - date.today()).days
                if record.expected_release else None,
            "status": record.status,
            "health_checks": len(record.health_checks),
            "reason": record.reason,
        }

    def get_due_for_release(
        self,
        days_ahead: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get animals due for release."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        due = []
        for animal_id, record in self._records.items():
            if record.status == "active" and record.expected_release:
                if today <= record.expected_release <= cutoff:
                    due.append({
                        "animal_id": animal_id,
                        "expected_release": record.expected_release,
                        "days_until": (record.expected_release - today).days,
                        "zone": record.zone_id,
                    })

        return sorted(due, key=lambda x: x["expected_release"])


class MovementTracker:
    """
    Animal movement tracking and control.

    Tracks all animal movements for traceability
    and disease control purposes.

    Example:
        >>> tracker = MovementTracker()
        >>> tracker.record_movement(
        ...     animal_id, MovementType.INCOMING,
        ...     origin="Farm A", destination="Main Herd"
        ... )
    """

    def __init__(self):
        """Initialize movement tracker."""
        self._movements: List[MovementRecord] = []
        self._high_risk_origins: Set[str] = set()

    def add_high_risk_origin(self, location: str) -> None:
        """Mark a location as high-risk."""
        self._high_risk_origins.add(location.lower())

    def record_movement(
        self,
        animal_id: str,
        movement_type: MovementType,
        origin: str,
        destination: str,
        movement_date: Optional[date] = None,
        purpose: str = "",
        health_certificate: str = "",
    ) -> MovementRecord:
        """
        Record animal movement.

        Args:
            animal_id: Animal identifier
            movement_type: Type of movement
            origin: Origin location
            destination: Destination location
            movement_date: Date of movement
            purpose: Purpose of movement
            health_certificate: Health certificate ID

        Returns:
            MovementRecord
        """
        if movement_date is None:
            movement_date = date.today()

        record = MovementRecord(
            animal_id=animal_id,
            movement_type=movement_type,
            date=movement_date,
            origin=origin,
            destination=destination,
            purpose=purpose,
            health_certificate=health_certificate,
        )

        self._movements.append(record)

        # Check for high-risk origin
        if origin.lower() in self._high_risk_origins:
            logger.warning(
                f"Movement from high-risk origin: {animal_id} from {origin}"
            )

        return record

    def get_movements(
        self,
        animal_id: Optional[str] = None,
        movement_type: Optional[MovementType] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[MovementRecord]:
        """
        Get movement records.

        Args:
            animal_id: Filter by animal
            movement_type: Filter by type
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of MovementRecord
        """
        results = self._movements

        if animal_id:
            results = [m for m in results if m.animal_id == animal_id]
        if movement_type:
            results = [m for m in results if m.movement_type == movement_type]
        if start_date:
            results = [m for m in results if m.date >= start_date]
        if end_date:
            results = [m for m in results if m.date <= end_date]

        return results

    def get_animal_history(
        self,
        animal_id: str,
    ) -> List[MovementRecord]:
        """Get complete movement history for animal."""
        return sorted(
            self.get_movements(animal_id),
            key=lambda x: x.date
        )

    def get_contact_trace(
        self,
        animal_id: str,
        days_back: int = 21,
    ) -> Dict[str, Any]:
        """
        Trace contacts for disease investigation.

        Args:
            animal_id: Animal to trace
            days_back: Days to trace back

        Returns:
            Dict with contact trace info
        """
        cutoff = date.today() - timedelta(days=days_back)
        movements = self.get_movements(animal_id, start_date=cutoff)

        locations = set()
        for m in movements:
            locations.add(m.origin)
            locations.add(m.destination)

        # Find other animals at same locations
        contacts = []
        for m in self._movements:
            if m.animal_id != animal_id and m.date >= cutoff:
                if m.origin in locations or m.destination in locations:
                    contacts.append({
                        "animal_id": m.animal_id,
                        "location": m.destination,
                        "date": m.date,
                    })

        return {
            "traced_animal": animal_id,
            "period_days": days_back,
            "locations_visited": list(locations),
            "potential_contacts": contacts,
            "contact_count": len(set(c["animal_id"] for c in contacts)),
        }


class VisitorManager:
    """
    Visitor access control and logging.

    Manages visitor registration, access control,
    and biosecurity compliance.

    Example:
        >>> manager = VisitorManager()
        >>> manager.check_in("John Doe", VisitorType.VETERINARIAN)
        >>> manager.check_out("John Doe")
    """

    def __init__(self, require_health_declaration: bool = True):
        """
        Initialize visitor manager.

        Args:
            require_health_declaration: Require health declaration
        """
        self.require_declaration = require_health_declaration
        self._visitors: List[VisitorRecord] = []
        self._active_visitors: Dict[str, VisitorRecord] = {}

    def check_in(
        self,
        name: str,
        visitor_type: VisitorType,
        areas: Optional[List[str]] = None,
        vehicle_id: str = "",
        last_farm_visit: str = "",
        health_declaration: bool = False,
    ) -> VisitorRecord:
        """
        Check in visitor.

        Args:
            name: Visitor name
            visitor_type: Type of visitor
            areas: Areas to visit
            vehicle_id: Vehicle ID
            last_farm_visit: Last farm visited
            health_declaration: Health declaration signed

        Returns:
            VisitorRecord
        """
        if self.require_declaration and not health_declaration:
            logger.warning(f"Visitor {name} has not signed health declaration")

        record = VisitorRecord(
            name=name,
            visitor_type=visitor_type,
            date=date.today(),
            time_in=datetime.now(),
            areas_visited=areas or [],
            vehicle_id=vehicle_id,
            last_farm_visit=last_farm_visit,
            health_declaration=health_declaration,
        )

        self._visitors.append(record)
        self._active_visitors[name] = record

        logger.info(f"Visitor checked in: {name} ({visitor_type.value})")

        return record

    def check_out(
        self,
        name: str,
        areas_visited: Optional[List[str]] = None,
    ) -> bool:
        """
        Check out visitor.

        Args:
            name: Visitor name
            areas_visited: Final list of areas visited

        Returns:
            True if successful
        """
        if name not in self._active_visitors:
            return False

        record = self._active_visitors[name]
        record.time_out = datetime.now()

        if areas_visited:
            record.areas_visited = areas_visited

        del self._active_visitors[name]

        logger.info(f"Visitor checked out: {name}")

        return True

    def get_active_visitors(self) -> List[VisitorRecord]:
        """Get currently active visitors."""
        return list(self._active_visitors.values())

    def get_visitor_log(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        visitor_type: Optional[VisitorType] = None,
    ) -> List[VisitorRecord]:
        """Get visitor log with filters."""
        results = self._visitors

        if start_date:
            results = [v for v in results if v.date >= start_date]
        if end_date:
            results = [v for v in results if v.date <= end_date]
        if visitor_type:
            results = [v for v in results if v.visitor_type == visitor_type]

        return results

    def get_high_risk_visits(
        self,
        days: int = 7,
    ) -> List[VisitorRecord]:
        """Get visits with biosecurity risks."""
        cutoff = date.today() - timedelta(days=days)
        high_risk = []

        for record in self._visitors:
            if record.date >= cutoff:
                risks = []

                if not record.health_declaration:
                    risks.append("no_health_declaration")

                if record.last_farm_visit:
                    risks.append("recent_farm_contact")

                if risks:
                    high_risk.append(record)

        return high_risk


class BiosecurityManager:
    """
    Comprehensive biosecurity management.

    Integrates quarantine, movement tracking,
    and visitor management for complete biosecurity.

    Example:
        >>> manager = BiosecurityManager()
        >>> zone = manager.create_quarantine_zone("Isolation", 20)
        >>> manager.add_to_quarantine(animal_id, zone.id, "new_arrival")
        >>> risk = manager.assess_risk()
    """

    def __init__(self):
        """Initialize biosecurity manager."""
        self.quarantine = QuarantineManager()
        self.movements = MovementTracker()
        self.visitors = VisitorManager()

        self._protocols: Dict[str, Dict[str, Any]] = {}
        self._incidents: List[Dict[str, Any]] = []

    def create_quarantine_zone(
        self,
        name: str,
        capacity: int,
        min_days: int = 21,
    ) -> QuarantineZone:
        """Create quarantine zone."""
        return self.quarantine.create_zone(name, capacity, min_days)

    def add_to_quarantine(
        self,
        animal_id: str,
        zone_id: str,
        reason: str,
    ) -> QuarantineRecord:
        """Add animal to quarantine."""
        return self.quarantine.add_animal(animal_id, zone_id, reason)

    def record_incident(
        self,
        incident_type: str,
        description: str,
        severity: RiskLevel,
        affected_animals: Optional[List[str]] = None,
        location: str = "",
    ) -> Dict[str, Any]:
        """
        Record biosecurity incident.

        Args:
            incident_type: Type of incident
            description: Incident description
            severity: Severity level
            affected_animals: Affected animal IDs
            location: Incident location

        Returns:
            Incident record
        """
        incident = {
            "id": f"INC-{len(self._incidents) + 1:04d}",
            "date": date.today(),
            "type": incident_type,
            "description": description,
            "severity": severity,
            "affected_animals": affected_animals or [],
            "location": location,
            "status": "open",
        }

        self._incidents.append(incident)
        logger.warning(f"Biosecurity incident: {incident['id']} - {incident_type}")

        return incident

    def assess_risk(
        self,
        herd_size: int = 100,
    ) -> RiskAssessment:
        """
        Assess overall biosecurity risk.

        Args:
            herd_size: Total herd size

        Returns:
            RiskAssessment object
        """
        risk_factors = []
        score = 0

        # Check quarantine status
        active_zones = self.quarantine.list_zones(status="active")
        quarantine_count = sum(z.current_count for z in active_zones)
        if quarantine_count > herd_size * 0.1:
            risk_factors.append({
                "factor": "high_quarantine_load",
                "description": f"{quarantine_count} animals in quarantine",
                "weight": 15,
            })
            score += 15

        # Check recent movements
        recent_incoming = self.movements.get_movements(
            movement_type=MovementType.INCOMING,
            start_date=date.today() - timedelta(days=30),
        )
        if len(recent_incoming) > herd_size * 0.2:
            risk_factors.append({
                "factor": "high_incoming_movement",
                "description": f"{len(recent_incoming)} incoming in last 30 days",
                "weight": 20,
            })
            score += 20

        # Check visitor activity
        high_risk_visits = self.visitors.get_high_risk_visits(14)
        if high_risk_visits:
            risk_factors.append({
                "factor": "high_risk_visitors",
                "description": f"{len(high_risk_visits)} high-risk visits",
                "weight": 10,
            })
            score += 10

        # Check open incidents
        open_incidents = [i for i in self._incidents if i["status"] == "open"]
        critical = [i for i in open_incidents if i["severity"] == RiskLevel.CRITICAL]
        if critical:
            risk_factors.append({
                "factor": "critical_incidents",
                "description": f"{len(critical)} critical incidents open",
                "weight": 30,
            })
            score += 30
        elif open_incidents:
            risk_factors.append({
                "factor": "open_incidents",
                "description": f"{len(open_incidents)} incidents open",
                "weight": 15,
            })
            score += 15

        # Determine overall risk level
        if score >= 50:
            overall_risk = RiskLevel.CRITICAL
        elif score >= 30:
            overall_risk = RiskLevel.HIGH
        elif score >= 15:
            overall_risk = RiskLevel.MODERATE
        else:
            overall_risk = RiskLevel.LOW

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors)

        return RiskAssessment(
            assessment_date=date.today(),
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            score=score,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        risk_factors: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []

        for factor in risk_factors:
            if factor["factor"] == "high_quarantine_load":
                recommendations.append(
                    "Review quarantine animals for release eligibility"
                )
            elif factor["factor"] == "high_incoming_movement":
                recommendations.append(
                    "Enhance incoming animal screening protocols"
                )
            elif factor["factor"] == "high_risk_visitors":
                recommendations.append(
                    "Enforce visitor biosecurity protocols strictly"
                )
            elif "incidents" in factor["factor"]:
                recommendations.append(
                    "Address and close open biosecurity incidents"
                )

        if not recommendations:
            recommendations.append("Maintain current biosecurity protocols")

        return recommendations

    def get_dashboard(self) -> Dict[str, Any]:
        """Get biosecurity dashboard summary."""
        risk = self.assess_risk()

        zones = self.quarantine.list_zones()
        quarantine_count = sum(z.current_count for z in zones)
        quarantine_capacity = sum(z.capacity for z in zones)

        return {
            "risk_level": risk.overall_risk.value,
            "risk_score": risk.score,
            "quarantine": {
                "total_zones": len(zones),
                "animals_in_quarantine": quarantine_count,
                "total_capacity": quarantine_capacity,
                "utilization": quarantine_count / quarantine_capacity * 100
                    if quarantine_capacity > 0 else 0,
            },
            "visitors": {
                "active": len(self.visitors.get_active_visitors()),
                "today": len(self.visitors.get_visitor_log(
                    start_date=date.today()
                )),
            },
            "incidents": {
                "open": len([i for i in self._incidents if i["status"] == "open"]),
                "total": len(self._incidents),
            },
            "recommendations": risk.recommendations[:3],
        }
