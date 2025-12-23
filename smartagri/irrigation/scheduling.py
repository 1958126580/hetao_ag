"""
Irrigation Scheduling Module

Smart irrigation scheduling and optimization:
- Demand-based scheduling
- Weather-responsive adjustments
- Multi-zone management
- Optimization algorithms

Example:
    >>> scheduler = IrrigationScheduler()
    >>> scheduler.add_zone("zone_1", area=10, crop="corn")
    >>> schedule = scheduler.generate_schedule(days=7)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IrrigationType(Enum):
    """Irrigation system types."""
    DRIP = "drip"
    SPRINKLER = "sprinkler"
    CENTER_PIVOT = "center_pivot"
    SURFACE = "surface"
    SUBSURFACE = "subsurface"


class ScheduleStatus(Enum):
    """Irrigation event status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


@dataclass
class IrrigationZone:
    """
    Irrigation zone definition.

    Attributes:
        id: Zone identifier
        name: Zone name
        area: Zone area (ha)
        crop: Current crop
        soil_type: Soil classification
        system_type: Irrigation system type
        flow_rate: System flow rate (L/min)
        efficiency: Application efficiency (0-1)
        max_depth: Maximum application depth (mm)
        allowed_deficit: Allowable depletion (0-1)
    """
    id: str
    name: str
    area: float
    crop: str = ""
    soil_type: str = "loam"
    system_type: IrrigationType = IrrigationType.SPRINKLER
    flow_rate: float = 100.0  # L/min
    efficiency: float = 0.85
    max_depth: float = 50.0  # mm per application
    allowed_deficit: float = 0.50


@dataclass
class IrrigationEvent:
    """
    Scheduled irrigation event.

    Attributes:
        id: Event identifier
        zone_id: Target zone
        scheduled_start: Scheduled start time
        scheduled_end: Scheduled end time
        depth_mm: Target application depth (mm)
        volume_m3: Total volume (m³)
        status: Event status
        actual_start: Actual start time
        actual_end: Actual end time
        notes: Event notes
    """
    id: str
    zone_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    depth_mm: float
    volume_m3: float
    status: ScheduleStatus = ScheduleStatus.SCHEDULED
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    notes: str = ""


@dataclass
class WaterRequirement:
    """
    Crop water requirement calculation.

    Attributes:
        zone_id: Zone identifier
        date: Requirement date
        et_mm: Evapotranspiration (mm)
        effective_rain_mm: Effective rainfall (mm)
        net_requirement_mm: Net irrigation requirement (mm)
        gross_requirement_mm: Gross requirement (mm)
    """
    zone_id: str
    date: date
    et_mm: float
    effective_rain_mm: float
    net_requirement_mm: float
    gross_requirement_mm: float


class IrrigationScheduler:
    """
    Smart irrigation scheduling system.

    Creates optimized irrigation schedules based on
    crop water requirements and system constraints.

    Example:
        >>> scheduler = IrrigationScheduler()
        >>> scheduler.add_zone("zone_1", name="North Field", area=10)
        >>> schedule = scheduler.generate_schedule(days=7)
    """

    def __init__(self):
        """Initialize irrigation scheduler."""
        self._zones: Dict[str, IrrigationZone] = {}
        self._events: List[IrrigationEvent] = []
        self._event_counter = 0

        # Crop coefficients (Kc) by growth stage
        self._crop_kc = {
            "corn": {"initial": 0.3, "mid": 1.2, "late": 0.6},
            "wheat": {"initial": 0.4, "mid": 1.15, "late": 0.4},
            "soybean": {"initial": 0.4, "mid": 1.15, "late": 0.5},
            "cotton": {"initial": 0.35, "mid": 1.2, "late": 0.7},
            "alfalfa": {"initial": 0.4, "mid": 1.2, "late": 1.15},
            "potato": {"initial": 0.5, "mid": 1.15, "late": 0.75},
            "tomato": {"initial": 0.6, "mid": 1.15, "late": 0.8},
            "default": {"initial": 0.4, "mid": 1.0, "late": 0.5},
        }

    def add_zone(
        self,
        zone_id: str,
        name: str = "",
        area: float = 1.0,
        crop: str = "",
        soil_type: str = "loam",
        system_type: IrrigationType = IrrigationType.SPRINKLER,
        flow_rate: float = 100.0,
        efficiency: float = 0.85,
    ) -> IrrigationZone:
        """
        Add irrigation zone.

        Args:
            zone_id: Zone identifier
            name: Zone name
            area: Zone area (ha)
            crop: Current crop
            soil_type: Soil type
            system_type: Irrigation system type
            flow_rate: Flow rate (L/min)
            efficiency: Application efficiency

        Returns:
            IrrigationZone object
        """
        zone = IrrigationZone(
            id=zone_id,
            name=name or zone_id,
            area=area,
            crop=crop,
            soil_type=soil_type,
            system_type=system_type,
            flow_rate=flow_rate,
            efficiency=efficiency,
        )

        self._zones[zone_id] = zone
        return zone

    def get_zone(self, zone_id: str) -> Optional[IrrigationZone]:
        """Get zone by ID."""
        return self._zones.get(zone_id)

    def calculate_water_requirement(
        self,
        zone_id: str,
        et_reference: float,
        rainfall: float = 0.0,
        growth_stage: str = "mid",
    ) -> WaterRequirement:
        """
        Calculate irrigation water requirement.

        Args:
            zone_id: Zone identifier
            et_reference: Reference ET (mm/day)
            rainfall: Rainfall amount (mm)
            growth_stage: Crop growth stage

        Returns:
            WaterRequirement object
        """
        zone = self._zones.get(zone_id)
        if zone is None:
            raise ValueError(f"Unknown zone: {zone_id}")

        # Get crop coefficient
        crop_kc = self._crop_kc.get(zone.crop.lower(), self._crop_kc["default"])
        kc = crop_kc.get(growth_stage, crop_kc["mid"])

        # Crop ET
        etc = et_reference * kc

        # Effective rainfall (simplified)
        if rainfall < 5:
            effective_rain = 0
        elif rainfall > 75:
            effective_rain = rainfall * 0.6
        else:
            effective_rain = rainfall * 0.8

        # Net requirement
        net_req = max(0, etc - effective_rain)

        # Gross requirement (account for efficiency)
        gross_req = net_req / zone.efficiency if zone.efficiency > 0 else net_req

        return WaterRequirement(
            zone_id=zone_id,
            date=date.today(),
            et_mm=etc,
            effective_rain_mm=effective_rain,
            net_requirement_mm=net_req,
            gross_requirement_mm=gross_req,
        )

    def schedule_irrigation(
        self,
        zone_id: str,
        depth_mm: float,
        start_time: datetime,
    ) -> IrrigationEvent:
        """
        Schedule irrigation event.

        Args:
            zone_id: Zone identifier
            depth_mm: Application depth (mm)
            start_time: Scheduled start time

        Returns:
            IrrigationEvent object
        """
        zone = self._zones.get(zone_id)
        if zone is None:
            raise ValueError(f"Unknown zone: {zone_id}")

        # Calculate duration and volume
        volume_m3 = zone.area * 10 * depth_mm  # ha * mm = m³/10
        duration_min = volume_m3 * 1000 / zone.flow_rate  # Convert to L
        end_time = start_time + timedelta(minutes=duration_min)

        self._event_counter += 1
        event = IrrigationEvent(
            id=f"IRR-{self._event_counter:06d}",
            zone_id=zone_id,
            scheduled_start=start_time,
            scheduled_end=end_time,
            depth_mm=depth_mm,
            volume_m3=volume_m3,
        )

        self._events.append(event)
        return event

    def generate_schedule(
        self,
        days: int = 7,
        start_date: Optional[date] = None,
        et_forecast: Optional[List[float]] = None,
        rain_forecast: Optional[List[float]] = None,
        preferred_hours: Tuple[int, int] = (6, 18),
    ) -> List[IrrigationEvent]:
        """
        Generate irrigation schedule.

        Args:
            days: Number of days to schedule
            start_date: Schedule start date
            et_forecast: Daily ET forecast (mm)
            rain_forecast: Daily rainfall forecast (mm)
            preferred_hours: Preferred irrigation hours (start, end)

        Returns:
            List of IrrigationEvent objects
        """
        if start_date is None:
            start_date = date.today()

        if et_forecast is None:
            et_forecast = [5.0] * days  # Default 5 mm/day

        if rain_forecast is None:
            rain_forecast = [0.0] * days

        schedule = []
        accumulated_deficit: Dict[str, float] = {z: 0 for z in self._zones}

        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            et = et_forecast[day_offset] if day_offset < len(et_forecast) else 5.0
            rain = rain_forecast[day_offset] if day_offset < len(rain_forecast) else 0.0

            # Calculate requirements for each zone
            for zone_id, zone in self._zones.items():
                req = self.calculate_water_requirement(zone_id, et, rain)
                accumulated_deficit[zone_id] += req.gross_requirement_mm

                # Check if irrigation needed
                trigger = zone.max_depth * zone.allowed_deficit

                if accumulated_deficit[zone_id] >= trigger:
                    # Schedule irrigation
                    depth = min(accumulated_deficit[zone_id], zone.max_depth)

                    # Find available time slot
                    start_hour = preferred_hours[0]
                    start_time = datetime.combine(
                        current_date,
                        time(hour=start_hour)
                    )

                    event = self.schedule_irrigation(zone_id, depth, start_time)
                    schedule.append(event)

                    accumulated_deficit[zone_id] = 0

        return schedule

    def get_schedule(
        self,
        zone_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[ScheduleStatus] = None,
    ) -> List[IrrigationEvent]:
        """
        Get scheduled events with filters.

        Args:
            zone_id: Filter by zone
            start_date: Filter start date
            end_date: Filter end date
            status: Filter by status

        Returns:
            List of IrrigationEvent objects
        """
        events = self._events

        if zone_id:
            events = [e for e in events if e.zone_id == zone_id]
        if start_date:
            events = [e for e in events
                      if e.scheduled_start.date() >= start_date]
        if end_date:
            events = [e for e in events
                      if e.scheduled_start.date() <= end_date]
        if status:
            events = [e for e in events if e.status == status]

        return sorted(events, key=lambda x: x.scheduled_start)

    def update_event_status(
        self,
        event_id: str,
        status: ScheduleStatus,
        actual_start: Optional[datetime] = None,
        actual_end: Optional[datetime] = None,
    ) -> bool:
        """
        Update irrigation event status.

        Args:
            event_id: Event identifier
            status: New status
            actual_start: Actual start time
            actual_end: Actual end time

        Returns:
            True if updated successfully
        """
        for event in self._events:
            if event.id == event_id:
                event.status = status
                if actual_start:
                    event.actual_start = actual_start
                if actual_end:
                    event.actual_end = actual_end
                return True
        return False


class ScheduleOptimizer:
    """
    Irrigation schedule optimization.

    Optimizes irrigation schedules for water savings,
    energy efficiency, and crop productivity.

    Example:
        >>> optimizer = ScheduleOptimizer(scheduler)
        >>> optimized = optimizer.optimize(objective="water_savings")
    """

    def __init__(self, scheduler: IrrigationScheduler):
        """
        Initialize schedule optimizer.

        Args:
            scheduler: IrrigationScheduler instance
        """
        self.scheduler = scheduler

    def optimize(
        self,
        events: List[IrrigationEvent],
        objective: str = "water_savings",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[IrrigationEvent]:
        """
        Optimize irrigation schedule.

        Args:
            events: Events to optimize
            objective: Optimization objective
            constraints: Optimization constraints

        Returns:
            Optimized event list
        """
        if objective == "water_savings":
            return self._optimize_water(events)
        elif objective == "energy":
            return self._optimize_energy(events)
        elif objective == "time":
            return self._optimize_time(events)
        else:
            return events

    def _optimize_water(
        self,
        events: List[IrrigationEvent],
    ) -> List[IrrigationEvent]:
        """Optimize for water savings."""
        optimized = []

        for event in events:
            zone = self.scheduler.get_zone(event.zone_id)
            if zone is None:
                optimized.append(event)
                continue

            # Reduce depth if possible (deficit irrigation)
            reduced_depth = event.depth_mm * 0.9
            if reduced_depth >= zone.max_depth * 0.5:
                new_event = IrrigationEvent(
                    id=event.id,
                    zone_id=event.zone_id,
                    scheduled_start=event.scheduled_start,
                    scheduled_end=event.scheduled_end,
                    depth_mm=reduced_depth,
                    volume_m3=event.volume_m3 * 0.9,
                    status=event.status,
                    notes="Optimized for water savings",
                )
                optimized.append(new_event)
            else:
                optimized.append(event)

        return optimized

    def _optimize_energy(
        self,
        events: List[IrrigationEvent],
    ) -> List[IrrigationEvent]:
        """Optimize for energy efficiency (off-peak scheduling)."""
        # Move events to off-peak hours (night)
        optimized = []

        for event in events:
            # Shift to early morning (2-6 AM) if daytime
            if 8 <= event.scheduled_start.hour < 20:
                duration = event.scheduled_end - event.scheduled_start
                new_start = event.scheduled_start.replace(hour=4, minute=0)
                new_end = new_start + duration

                new_event = IrrigationEvent(
                    id=event.id,
                    zone_id=event.zone_id,
                    scheduled_start=new_start,
                    scheduled_end=new_end,
                    depth_mm=event.depth_mm,
                    volume_m3=event.volume_m3,
                    status=event.status,
                    notes="Shifted to off-peak hours",
                )
                optimized.append(new_event)
            else:
                optimized.append(event)

        return optimized

    def _optimize_time(
        self,
        events: List[IrrigationEvent],
    ) -> List[IrrigationEvent]:
        """Optimize for minimum total irrigation time."""
        # Consolidate events where possible
        by_zone: Dict[str, List[IrrigationEvent]] = {}

        for event in events:
            if event.zone_id not in by_zone:
                by_zone[event.zone_id] = []
            by_zone[event.zone_id].append(event)

        optimized = []

        for zone_id, zone_events in by_zone.items():
            # Combine events on same day
            by_date: Dict[date, List[IrrigationEvent]] = {}
            for event in zone_events:
                event_date = event.scheduled_start.date()
                if event_date not in by_date:
                    by_date[event_date] = []
                by_date[event_date].append(event)

            for event_date, day_events in by_date.items():
                if len(day_events) == 1:
                    optimized.append(day_events[0])
                else:
                    # Combine into single event
                    total_depth = sum(e.depth_mm for e in day_events)
                    total_volume = sum(e.volume_m3 for e in day_events)

                    combined = IrrigationEvent(
                        id=day_events[0].id,
                        zone_id=zone_id,
                        scheduled_start=day_events[0].scheduled_start,
                        scheduled_end=day_events[-1].scheduled_end,
                        depth_mm=total_depth,
                        volume_m3=total_volume,
                        status=ScheduleStatus.SCHEDULED,
                        notes="Combined events",
                    )
                    optimized.append(combined)

        return sorted(optimized, key=lambda x: x.scheduled_start)

    def analyze_efficiency(
        self,
        events: List[IrrigationEvent],
    ) -> Dict[str, Any]:
        """
        Analyze schedule efficiency.

        Args:
            events: Events to analyze

        Returns:
            Dict with efficiency metrics
        """
        if not events:
            return {"status": "no_events"}

        total_volume = sum(e.volume_m3 for e in events)
        total_depth = sum(e.depth_mm for e in events)
        n_events = len(events)

        # Time analysis
        durations = [
            (e.scheduled_end - e.scheduled_start).total_seconds() / 3600
            for e in events
        ]
        total_hours = sum(durations)

        # Zones covered
        zones = set(e.zone_id for e in events)

        return {
            "total_events": n_events,
            "total_volume_m3": total_volume,
            "total_depth_mm": total_depth,
            "total_hours": total_hours,
            "avg_duration_hours": np.mean(durations) if durations else 0,
            "zones_covered": len(zones),
            "avg_volume_per_event": total_volume / n_events if n_events > 0 else 0,
        }
