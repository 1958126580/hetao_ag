"""
Irrigation Systems Module

Irrigation system modeling and efficiency analysis:
- System type configurations
- Hydraulic calculations
- Efficiency analysis
- Maintenance scheduling

Example:
    >>> system = DripSystem(area=10, emitter_spacing=0.3)
    >>> flow = system.calculate_flow_rate()
    >>> efficiency = system.calculate_efficiency()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemComponent:
    """
    Irrigation system component.

    Attributes:
        name: Component name
        component_type: Type of component
        quantity: Number of units
        flow_rate: Flow rate per unit (L/hr)
        pressure: Operating pressure (bar)
        efficiency: Component efficiency (0-1)
    """
    name: str
    component_type: str
    quantity: int = 1
    flow_rate: float = 0.0
    pressure: float = 1.0
    efficiency: float = 0.95


@dataclass
class SystemEfficiency:
    """
    Irrigation system efficiency assessment.

    Attributes:
        conveyance_efficiency: Pipe/channel efficiency (0-1)
        application_efficiency: Field application efficiency (0-1)
        distribution_uniformity: Uniformity coefficient (0-1)
        overall_efficiency: Total system efficiency (0-1)
        water_losses: Water loss breakdown
    """
    conveyance_efficiency: float
    application_efficiency: float
    distribution_uniformity: float
    overall_efficiency: float
    water_losses: Dict[str, float] = field(default_factory=dict)


class IrrigationSystem(ABC):
    """
    Base class for irrigation systems.

    Provides common functionality for different
    irrigation system types.
    """

    def __init__(
        self,
        area: float,
        flow_rate: float = 0.0,
        pressure: float = 2.0,
    ):
        """
        Initialize irrigation system.

        Args:
            area: Irrigated area (ha)
            flow_rate: Total system flow rate (L/min)
            pressure: Operating pressure (bar)
        """
        self.area = area
        self.flow_rate = flow_rate
        self.pressure = pressure
        self._components: List[SystemComponent] = []
        self._maintenance_log: List[Dict[str, Any]] = []

    @abstractmethod
    def calculate_application_rate(self) -> float:
        """Calculate application rate (mm/hr)."""
        pass

    @abstractmethod
    def calculate_efficiency(self) -> SystemEfficiency:
        """Calculate system efficiency."""
        pass

    def add_component(self, component: SystemComponent) -> None:
        """Add system component."""
        self._components.append(component)

    def calculate_runtime(
        self,
        target_depth: float,
    ) -> float:
        """
        Calculate runtime for target application depth.

        Args:
            target_depth: Target application (mm)

        Returns:
            Runtime in hours
        """
        app_rate = self.calculate_application_rate()
        if app_rate <= 0:
            return 0.0

        return target_depth / app_rate

    def calculate_volume(
        self,
        depth_mm: float,
    ) -> float:
        """
        Calculate water volume for application depth.

        Args:
            depth_mm: Application depth (mm)

        Returns:
            Volume in m³
        """
        return self.area * depth_mm * 10  # ha * mm = m³/10

    def log_maintenance(
        self,
        maintenance_type: str,
        description: str,
        performed_by: str = "",
        cost: float = 0.0,
    ) -> None:
        """Log maintenance activity."""
        self._maintenance_log.append({
            "date": date.today(),
            "type": maintenance_type,
            "description": description,
            "performed_by": performed_by,
            "cost": cost,
        })

    def get_maintenance_history(self) -> List[Dict[str, Any]]:
        """Get maintenance history."""
        return self._maintenance_log


class DripSystem(IrrigationSystem):
    """
    Drip/micro-irrigation system.

    Models drip irrigation with emitters,
    laterals, and submains.

    Example:
        >>> system = DripSystem(area=5, emitter_spacing=0.3, row_spacing=1.0)
        >>> rate = system.calculate_application_rate()
    """

    def __init__(
        self,
        area: float,
        emitter_spacing: float = 0.3,
        row_spacing: float = 1.0,
        emitter_flow: float = 2.0,
        pressure: float = 1.0,
    ):
        """
        Initialize drip system.

        Args:
            area: Irrigated area (ha)
            emitter_spacing: Emitter spacing (m)
            row_spacing: Row/lateral spacing (m)
            emitter_flow: Emitter flow rate (L/hr)
            pressure: Operating pressure (bar)
        """
        super().__init__(area, pressure=pressure)

        self.emitter_spacing = emitter_spacing
        self.row_spacing = row_spacing
        self.emitter_flow = emitter_flow

        # Calculate number of emitters
        self.emitters_per_ha = 10000 / (emitter_spacing * row_spacing)
        self.total_emitters = int(self.emitters_per_ha * area)

        # Calculate total flow
        self.flow_rate = self.total_emitters * emitter_flow / 60  # L/min

    def calculate_application_rate(self) -> float:
        """Calculate application rate (mm/hr)."""
        # Emitter flow * emitters per m² / 1000
        emitters_per_m2 = self.emitters_per_ha / 10000
        return self.emitter_flow * emitters_per_m2

    def calculate_efficiency(self) -> SystemEfficiency:
        """Calculate drip system efficiency."""
        # Drip systems typically very efficient
        conveyance = 0.98
        application = 0.92
        uniformity = 0.90

        # Adjust for pressure
        if self.pressure < 0.8:
            uniformity *= 0.95  # Low pressure reduces uniformity
        elif self.pressure > 1.5:
            application *= 0.98  # High pressure increases losses

        overall = conveyance * application * uniformity

        return SystemEfficiency(
            conveyance_efficiency=conveyance,
            application_efficiency=application,
            distribution_uniformity=uniformity,
            overall_efficiency=overall,
            water_losses={
                "pipe_leakage": 0.02,
                "evaporation": 0.01,
                "deep_percolation": 0.07,
            },
        )

    def emitter_clogging_check(
        self,
        measured_flows: List[float],
    ) -> Dict[str, Any]:
        """
        Check for emitter clogging.

        Args:
            measured_flows: Measured emitter flows (L/hr)

        Returns:
            Dict with clogging assessment
        """
        if not measured_flows:
            return {"status": "no_data"}

        avg_flow = np.mean(measured_flows)
        std_flow = np.std(measured_flows)
        cv = std_flow / avg_flow * 100 if avg_flow > 0 else 0

        # Count clogged (< 50% of rated flow)
        clogged = sum(1 for f in measured_flows if f < self.emitter_flow * 0.5)
        partial = sum(1 for f in measured_flows
                      if self.emitter_flow * 0.5 <= f < self.emitter_flow * 0.8)

        return {
            "n_checked": len(measured_flows),
            "avg_flow": avg_flow,
            "rated_flow": self.emitter_flow,
            "flow_ratio": avg_flow / self.emitter_flow,
            "cv_pct": cv,
            "clogged_count": clogged,
            "partial_clog_count": partial,
            "clogging_pct": clogged / len(measured_flows) * 100,
            "maintenance_needed": clogged > len(measured_flows) * 0.1,
        }


class SprinklerSystem(IrrigationSystem):
    """
    Sprinkler irrigation system.

    Models overhead sprinkler systems with
    various sprinkler types and layouts.

    Example:
        >>> system = SprinklerSystem(area=20, sprinkler_spacing=12)
        >>> rate = system.calculate_application_rate()
    """

    def __init__(
        self,
        area: float,
        sprinkler_spacing: float = 12.0,
        lateral_spacing: float = 12.0,
        sprinkler_flow: float = 0.5,
        pressure: float = 3.0,
        sprinkler_type: str = "impact",
    ):
        """
        Initialize sprinkler system.

        Args:
            area: Irrigated area (ha)
            sprinkler_spacing: Sprinkler spacing on lateral (m)
            lateral_spacing: Lateral spacing (m)
            sprinkler_flow: Sprinkler flow rate (m³/hr)
            pressure: Operating pressure (bar)
            sprinkler_type: Type of sprinkler
        """
        super().__init__(area, pressure=pressure)

        self.sprinkler_spacing = sprinkler_spacing
        self.lateral_spacing = lateral_spacing
        self.sprinkler_flow = sprinkler_flow  # m³/hr
        self.sprinkler_type = sprinkler_type

        # Calculate sprinklers
        self.sprinklers_per_ha = 10000 / (sprinkler_spacing * lateral_spacing)
        self.total_sprinklers = int(self.sprinklers_per_ha * area)

        # Flow rate (L/min)
        self.flow_rate = self.total_sprinklers * sprinkler_flow * 1000 / 60

    def calculate_application_rate(self) -> float:
        """Calculate application rate (mm/hr)."""
        # Flow (m³/hr) / area covered per sprinkler (m²) * 1000
        area_per_sprinkler = self.sprinkler_spacing * self.lateral_spacing
        return self.sprinkler_flow * 1000 / area_per_sprinkler

    def calculate_efficiency(self) -> SystemEfficiency:
        """Calculate sprinkler system efficiency."""
        # Base efficiencies
        conveyance = 0.95

        # Application efficiency depends on conditions
        if self.sprinkler_type == "impact":
            application = 0.75
        elif self.sprinkler_type == "rotor":
            application = 0.80
        else:
            application = 0.70

        # Uniformity based on spacing ratio
        spacing_ratio = self.sprinkler_spacing / self.lateral_spacing
        if 0.9 <= spacing_ratio <= 1.1:
            uniformity = 0.85
        else:
            uniformity = 0.75

        overall = conveyance * application * uniformity

        return SystemEfficiency(
            conveyance_efficiency=conveyance,
            application_efficiency=application,
            distribution_uniformity=uniformity,
            overall_efficiency=overall,
            water_losses={
                "evaporation": 0.08,
                "wind_drift": 0.07,
                "deep_percolation": 0.10,
            },
        )

    def wind_adjustment(
        self,
        wind_speed: float,
    ) -> float:
        """
        Calculate efficiency adjustment for wind.

        Args:
            wind_speed: Wind speed (m/s)

        Returns:
            Efficiency multiplier (0-1)
        """
        if wind_speed < 2:
            return 1.0
        elif wind_speed < 4:
            return 0.95
        elif wind_speed < 6:
            return 0.85
        elif wind_speed < 8:
            return 0.70
        else:
            return 0.50  # Not recommended


class PivotSystem(IrrigationSystem):
    """
    Center pivot irrigation system.

    Models center pivot systems with variable
    speed and application control.

    Example:
        >>> system = PivotSystem(radius=400, flow_rate=3000)
        >>> rate = system.calculate_application_rate()
    """

    def __init__(
        self,
        radius: float,
        flow_rate: float,
        pressure: float = 4.0,
        spans: int = 8,
        end_gun: bool = True,
    ):
        """
        Initialize pivot system.

        Args:
            radius: Pivot radius (m)
            flow_rate: Total flow rate (L/min)
            pressure: Operating pressure (bar)
            spans: Number of spans
            end_gun: Whether end gun is installed
        """
        # Calculate area (circular)
        area = np.pi * (radius ** 2) / 10000  # ha

        super().__init__(area, flow_rate, pressure)

        self.radius = radius
        self.spans = spans
        self.end_gun = end_gun

        # Typical rotation time (hours per revolution)
        self.rotation_time = 24.0

    def calculate_application_rate(self) -> float:
        """Calculate application rate (mm/hr)."""
        # Flow (L/min) * 60 / area (m²) = mm/hr
        area_m2 = self.area * 10000
        return self.flow_rate * 60 / area_m2

    def calculate_application_depth(
        self,
        speed_percent: float = 100,
    ) -> float:
        """
        Calculate application depth at given speed.

        Args:
            speed_percent: Pivot speed (% of max)

        Returns:
            Application depth (mm)
        """
        # Higher speed = less water applied
        base_depth = self.flow_rate * self.rotation_time * 60 / (self.area * 10000)
        return base_depth * (100 / speed_percent)

    def calculate_efficiency(self) -> SystemEfficiency:
        """Calculate pivot system efficiency."""
        conveyance = 0.98  # Short pipe run
        application = 0.85
        uniformity = 0.88

        # End gun reduces uniformity
        if self.end_gun:
            uniformity *= 0.95

        overall = conveyance * application * uniformity

        return SystemEfficiency(
            conveyance_efficiency=conveyance,
            application_efficiency=application,
            distribution_uniformity=uniformity,
            overall_efficiency=overall,
            water_losses={
                "evaporation": 0.05,
                "runoff": 0.08,
                "deep_percolation": 0.02,
            },
        )

    def variable_rate_zones(
        self,
        n_zones: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Define VRI zones for variable rate irrigation.

        Args:
            n_zones: Number of radial zones

        Returns:
            List of zone definitions
        """
        zones = []
        zone_width = self.radius / n_zones

        for i in range(n_zones):
            inner_radius = i * zone_width
            outer_radius = (i + 1) * zone_width

            # Area increases with radius
            zone_area = np.pi * (outer_radius ** 2 - inner_radius ** 2) / 10000

            zones.append({
                "zone_id": i + 1,
                "inner_radius_m": inner_radius,
                "outer_radius_m": outer_radius,
                "area_ha": zone_area,
                "speed_adjust": 1.0,  # Default no adjustment
            })

        return zones

    def calculate_energy_use(
        self,
        hours: float,
        pump_efficiency: float = 0.75,
    ) -> Dict[str, float]:
        """
        Calculate energy consumption.

        Args:
            hours: Operating hours
            pump_efficiency: Pump efficiency

        Returns:
            Dict with energy metrics
        """
        # Hydraulic power (kW)
        # P = Q * H * ρ * g / (1000 * η)
        # Q = flow (m³/s), H = head (m), ρ = 1000 kg/m³, g = 9.81
        q_m3s = self.flow_rate / 60000
        head = self.pressure * 10.2  # bar to m
        hydraulic_power = q_m3s * head * 1000 * 9.81 / 1000

        # Shaft power
        shaft_power = hydraulic_power / pump_efficiency

        # Energy consumption
        energy_kwh = shaft_power * hours

        return {
            "flow_m3_hr": self.flow_rate * 60 / 1000,
            "total_head_m": head,
            "hydraulic_power_kw": hydraulic_power,
            "shaft_power_kw": shaft_power,
            "operating_hours": hours,
            "energy_kwh": energy_kwh,
            "cost_estimate": energy_kwh * 0.10,  # Assumed rate
        }
