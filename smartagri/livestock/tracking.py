"""
Livestock Tracking Module

Provides animal identification, tracking, and movement recording:
- Individual animal registration and identification
- GPS-based location tracking
- Movement history and analysis
- Group management

Example:
    >>> tracker = AnimalTracker()
    >>> animal = tracker.register_animal(
    ...     species="cattle",
    ...     breed="Angus",
    ...     birth_date=date(2023, 3, 15)
    ... )
    >>> tracker.record_location(animal.id, latitude=45.0, longitude=-90.0)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class Sex(Enum):
    """Animal sex classification."""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AnimalStatus(Enum):
    """Animal status in the herd."""
    ACTIVE = "active"
    SOLD = "sold"
    DECEASED = "deceased"
    TRANSFERRED = "transferred"
    CULLED = "culled"


@dataclass
class Animal:
    """
    Individual animal record.

    Attributes:
        id: Unique identifier
        species: Animal species
        breed: Breed name
        sex: Animal sex
        birth_date: Date of birth
        dam_id: Mother's ID
        sire_id: Father's ID
        tag_number: Physical tag number
        electronic_id: RFID/electronic ID
        status: Current status
        birth_weight: Birth weight (kg)
        current_weight: Most recent weight (kg)
        location: Current location
        metadata: Additional attributes
    """
    id: str
    species: str
    breed: str = ""
    sex: Sex = Sex.UNKNOWN
    birth_date: Optional[date] = None
    dam_id: Optional[str] = None
    sire_id: Optional[str] = None
    tag_number: Optional[str] = None
    electronic_id: Optional[str] = None
    status: AnimalStatus = AnimalStatus.ACTIVE
    birth_weight: Optional[float] = None
    current_weight: Optional[float] = None
    location: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> Optional[int]:
        """Calculate age in days."""
        if self.birth_date:
            return (date.today() - self.birth_date).days
        return None

    @property
    def age_months(self) -> Optional[float]:
        """Calculate age in months."""
        if self.age_days:
            return self.age_days / 30.44
        return None


@dataclass
class AnimalGroup:
    """
    Group of animals (herd, flock, etc.).

    Attributes:
        id: Group identifier
        name: Group name
        animal_ids: List of animal IDs in group
        pasture_id: Current pasture/location ID
        created_date: Date group was created
    """
    id: str
    name: str
    animal_ids: List[str] = field(default_factory=list)
    pasture_id: Optional[str] = None
    created_date: date = field(default_factory=date.today)

    def add_animal(self, animal_id: str) -> None:
        """Add animal to group."""
        if animal_id not in self.animal_ids:
            self.animal_ids.append(animal_id)

    def remove_animal(self, animal_id: str) -> None:
        """Remove animal from group."""
        if animal_id in self.animal_ids:
            self.animal_ids.remove(animal_id)

    @property
    def size(self) -> int:
        """Number of animals in group."""
        return len(self.animal_ids)


@dataclass
class LocationRecord:
    """GPS location record."""
    timestamp: datetime
    latitude: float
    longitude: float
    accuracy: float = 10.0
    source: str = "gps"


@dataclass
class MovementRecord:
    """Animal movement event record."""
    animal_id: str
    timestamp: datetime
    from_location: Optional[str]
    to_location: str
    reason: str = ""
    notes: str = ""


class LocationTracker:
    """
    GPS-based animal location tracking.

    Tracks animal locations over time and provides
    analysis of movement patterns.

    Example:
        >>> tracker = LocationTracker()
        >>> tracker.record(animal_id, lat=45.0, lon=-90.0)
        >>> history = tracker.get_history(animal_id, days=7)
    """

    def __init__(self):
        """Initialize location tracker."""
        self._locations: Dict[str, List[LocationRecord]] = {}

    def record(
        self,
        animal_id: str,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
        accuracy: float = 10.0,
    ) -> None:
        """
        Record animal location.

        Args:
            animal_id: Animal identifier
            latitude: GPS latitude
            longitude: GPS longitude
            timestamp: Recording timestamp
            accuracy: GPS accuracy (meters)
        """
        if timestamp is None:
            timestamp = datetime.now()

        record = LocationRecord(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
        )

        if animal_id not in self._locations:
            self._locations[animal_id] = []

        self._locations[animal_id].append(record)

    def get_current_location(
        self,
        animal_id: str,
    ) -> Optional[LocationRecord]:
        """Get most recent location for animal."""
        if animal_id in self._locations and self._locations[animal_id]:
            return self._locations[animal_id][-1]
        return None

    def get_history(
        self,
        animal_id: str,
        days: int = 30,
    ) -> List[LocationRecord]:
        """Get location history for animal."""
        if animal_id not in self._locations:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [
            loc for loc in self._locations[animal_id]
            if loc.timestamp >= cutoff
        ]

    def calculate_distance_traveled(
        self,
        animal_id: str,
        days: int = 1,
    ) -> float:
        """
        Calculate total distance traveled.

        Args:
            animal_id: Animal identifier
            days: Number of days to analyze

        Returns:
            Total distance in meters
        """
        history = self.get_history(animal_id, days)

        if len(history) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(history)):
            dist = self._haversine_distance(
                history[i - 1].latitude,
                history[i - 1].longitude,
                history[i].latitude,
                history[i].longitude,
            )
            total_distance += dist

        return total_distance

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two GPS points (meters)."""
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def detect_grazing_behavior(
        self,
        animal_id: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Analyze grazing behavior from location data.

        Args:
            animal_id: Animal identifier
            hours: Hours to analyze

        Returns:
            Dict with behavior metrics
        """
        history = self.get_history(animal_id, days=hours // 24 + 1)

        if len(history) < 10:
            return {"status": "insufficient_data"}

        # Calculate movement speed
        speeds = []
        for i in range(1, len(history)):
            dist = self._haversine_distance(
                history[i - 1].latitude,
                history[i - 1].longitude,
                history[i].latitude,
                history[i].longitude,
            )
            time_diff = (history[i].timestamp - history[i - 1].timestamp).total_seconds()
            if time_diff > 0:
                speeds.append(dist / time_diff)

        avg_speed = np.mean(speeds) if speeds else 0

        # Classify behavior
        if avg_speed < 0.01:
            behavior = "resting"
        elif avg_speed < 0.1:
            behavior = "grazing"
        else:
            behavior = "moving"

        return {
            "behavior": behavior,
            "avg_speed_m_s": avg_speed,
            "distance_m": sum(self._haversine_distance(
                history[i - 1].latitude, history[i - 1].longitude,
                history[i].latitude, history[i].longitude
            ) for i in range(1, len(history))),
            "n_records": len(history),
        }


class MovementHistory:
    """
    Track animal movement between locations.

    Records pasture changes, sales, transfers, etc.
    """

    def __init__(self):
        """Initialize movement history."""
        self._movements: List[MovementRecord] = []

    def record_movement(
        self,
        animal_id: str,
        to_location: str,
        from_location: Optional[str] = None,
        reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> MovementRecord:
        """
        Record animal movement.

        Args:
            animal_id: Animal identifier
            to_location: Destination location/pasture
            from_location: Source location
            reason: Reason for movement
            timestamp: Movement timestamp

        Returns:
            MovementRecord
        """
        if timestamp is None:
            timestamp = datetime.now()

        record = MovementRecord(
            animal_id=animal_id,
            timestamp=timestamp,
            from_location=from_location,
            to_location=to_location,
            reason=reason,
        )

        self._movements.append(record)
        return record

    def get_animal_movements(
        self,
        animal_id: str,
    ) -> List[MovementRecord]:
        """Get all movements for an animal."""
        return [m for m in self._movements if m.animal_id == animal_id]

    def get_current_location(
        self,
        animal_id: str,
    ) -> Optional[str]:
        """Get current location for animal."""
        movements = self.get_animal_movements(animal_id)
        if movements:
            return movements[-1].to_location
        return None


class AnimalTracker:
    """
    Comprehensive animal tracking system.

    Manages animal registration, identification, and tracking
    for livestock operations.

    Example:
        >>> tracker = AnimalTracker()
        >>> animal = tracker.register_animal(
        ...     species="cattle",
        ...     breed="Angus",
        ...     sex=Sex.FEMALE,
        ...     birth_date=date(2023, 3, 15)
        ... )
        >>> tracker.record_weight(animal.id, 450.0)
        >>> tracker.record_location(animal.id, 45.0, -90.0)
    """

    def __init__(self):
        """Initialize animal tracker."""
        self._animals: Dict[str, Animal] = {}
        self._groups: Dict[str, AnimalGroup] = {}
        self._weights: Dict[str, List[Tuple[date, float]]] = {}
        self._location_tracker = LocationTracker()
        self._movement_history = MovementHistory()

    def register_animal(
        self,
        species: str,
        breed: str = "",
        sex: Sex = Sex.UNKNOWN,
        birth_date: Optional[date] = None,
        dam_id: Optional[str] = None,
        sire_id: Optional[str] = None,
        tag_number: Optional[str] = None,
        birth_weight: Optional[float] = None,
    ) -> Animal:
        """
        Register a new animal.

        Args:
            species: Animal species
            breed: Breed name
            sex: Animal sex
            birth_date: Date of birth
            dam_id: Mother's ID
            sire_id: Father's ID
            tag_number: Physical tag number
            birth_weight: Birth weight (kg)

        Returns:
            Registered Animal object
        """
        animal_id = str(uuid.uuid4())[:8]

        animal = Animal(
            id=animal_id,
            species=species,
            breed=breed,
            sex=sex,
            birth_date=birth_date,
            dam_id=dam_id,
            sire_id=sire_id,
            tag_number=tag_number,
            birth_weight=birth_weight,
            current_weight=birth_weight,
        )

        self._animals[animal_id] = animal
        self._weights[animal_id] = []

        if birth_weight:
            self._weights[animal_id].append((birth_date or date.today(), birth_weight))

        logger.info(f"Registered animal {animal_id}: {species} {breed}")
        return animal

    def get_animal(self, animal_id: str) -> Optional[Animal]:
        """Get animal by ID."""
        return self._animals.get(animal_id)

    def update_animal(
        self,
        animal_id: str,
        **updates,
    ) -> Optional[Animal]:
        """Update animal attributes."""
        animal = self._animals.get(animal_id)
        if animal:
            for key, value in updates.items():
                if hasattr(animal, key):
                    setattr(animal, key, value)
        return animal

    def record_weight(
        self,
        animal_id: str,
        weight: float,
        record_date: Optional[date] = None,
    ) -> None:
        """
        Record animal weight.

        Args:
            animal_id: Animal identifier
            weight: Weight in kg
            record_date: Date of weighing
        """
        if animal_id not in self._weights:
            self._weights[animal_id] = []

        record_date = record_date or date.today()
        self._weights[animal_id].append((record_date, weight))

        # Update current weight
        if animal_id in self._animals:
            self._animals[animal_id].current_weight = weight

    def get_weight_history(
        self,
        animal_id: str,
    ) -> List[Tuple[date, float]]:
        """Get weight history for animal."""
        return self._weights.get(animal_id, [])

    def record_location(
        self,
        animal_id: str,
        latitude: float,
        longitude: float,
    ) -> None:
        """Record animal GPS location."""
        self._location_tracker.record(animal_id, latitude, longitude)

        if animal_id in self._animals:
            self._animals[animal_id].location = (latitude, longitude)

    def create_group(
        self,
        name: str,
        animal_ids: Optional[List[str]] = None,
    ) -> AnimalGroup:
        """Create animal group."""
        group_id = str(uuid.uuid4())[:8]
        group = AnimalGroup(
            id=group_id,
            name=name,
            animal_ids=animal_ids or [],
        )
        self._groups[group_id] = group
        return group

    def get_herd_summary(self) -> Dict[str, Any]:
        """Get summary statistics for entire herd."""
        active_animals = [
            a for a in self._animals.values()
            if a.status == AnimalStatus.ACTIVE
        ]

        by_species = {}
        by_sex = {}

        for animal in active_animals:
            by_species[animal.species] = by_species.get(animal.species, 0) + 1
            by_sex[animal.sex.value] = by_sex.get(animal.sex.value, 0) + 1

        weights = [a.current_weight for a in active_animals if a.current_weight]

        return {
            "total_active": len(active_animals),
            "by_species": by_species,
            "by_sex": by_sex,
            "avg_weight": np.mean(weights) if weights else 0,
            "n_groups": len(self._groups),
        }
