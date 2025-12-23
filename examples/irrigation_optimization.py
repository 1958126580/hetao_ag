#!/usr/bin/env python3
"""
Smart Irrigation Optimization Example

This example demonstrates intelligent irrigation scheduling
using the smartagri library, including:

- Evapotranspiration calculation (FAO-56 Penman-Monteith)
- Soil water balance modeling
- Multi-zone irrigation scheduling
- Water use efficiency optimization
- Deficit irrigation strategies
- Real-time weather integration

Usage:
    python examples/irrigation_optimization.py

Author: SmartAgri Development Team
License: MIT
"""

import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SoilProfile:
    """
    Soil hydraulic properties for water balance.

    Stores soil water retention parameters and
    hydraulic conductivity information.
    """
    texture: str
    field_capacity: float       # θfc (m³/m³)
    wilting_point: float        # θwp (m³/m³)
    saturation: float           # θs (m³/m³)
    available_water: float      # TAW (mm/m depth)
    infiltration_rate: float    # mm/hour


@dataclass
class IrrigationZone:
    """
    Individual irrigation zone configuration.

    Represents a field or section with uniform
    irrigation requirements and equipment.
    """
    zone_id: str
    name: str
    area_ha: float
    crop_type: str
    soil: SoilProfile
    root_depth_m: float = 0.6
    management_allowable_depletion: float = 0.5
    kc: float = 1.0                    # Crop coefficient
    irrigation_efficiency: float = 0.85


class PenmanMonteithET:
    """
    FAO-56 Penman-Monteith evapotranspiration calculator.

    Implements the standard reference ET calculation
    recommended by FAO for irrigation scheduling.
    """

    def __init__(
        self,
        latitude: float,
        elevation: float = 0.0,
    ):
        """
        Initialize ET calculator.

        Args:
            latitude: Site latitude (degrees)
            elevation: Elevation above sea level (m)
        """
        self.latitude = latitude
        self.elevation = elevation

        # Atmospheric pressure (kPa) from elevation
        self.pressure = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26

        # Psychrometric constant (kPa/°C)
        self.gamma = 0.665e-3 * self.pressure

    def calculate_eto(
        self,
        t_max: float,
        t_min: float,
        humidity: float,
        wind_speed: float,
        solar_radiation: float,
        doy: int,
    ) -> float:
        """
        Calculate reference evapotranspiration (ETo).

        Implements FAO-56 Penman-Monteith equation for
        a hypothetical grass reference surface.

        Args:
            t_max: Maximum temperature (°C)
            t_min: Minimum temperature (°C)
            humidity: Mean relative humidity (%)
            wind_speed: Wind speed at 2m height (m/s)
            solar_radiation: Incoming solar radiation (MJ/m²/day)
            doy: Day of year (1-365)

        Returns:
            Reference evapotranspiration (mm/day)

        Example:
            >>> et = PenmanMonteithET(latitude=42.0, elevation=100)
            >>> eto = et.calculate_eto(
            ...     t_max=32, t_min=18, humidity=55,
            ...     wind_speed=2.5, solar_radiation=25, doy=180
            ... )
            >>> print(f"ETo = {eto:.2f} mm/day")
        """
        # Mean temperature
        t_mean = (t_max + t_min) / 2

        # Saturation vapor pressure (kPa)
        es_max = 0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
        es_min = 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))
        es = (es_max + es_min) / 2

        # Actual vapor pressure (kPa)
        ea = es * humidity / 100

        # Slope of saturation vapor pressure curve (kPa/°C)
        delta = 4098 * 0.6108 * np.exp(17.27 * t_mean / (t_mean + 237.3)) / (t_mean + 237.3) ** 2

        # Net radiation calculation
        rn = self._calculate_net_radiation(solar_radiation, t_max, t_min, ea, doy)

        # Soil heat flux (assumed zero for daily calculations)
        g = 0

        # FAO-56 Penman-Monteith equation
        numerator = (
            0.408 * delta * (rn - g) +
            self.gamma * 900 / (t_mean + 273) * wind_speed * (es - ea)
        )
        denominator = delta + self.gamma * (1 + 0.34 * wind_speed)

        eto = numerator / denominator

        return max(eto, 0)

    def _calculate_net_radiation(
        self,
        rs: float,
        t_max: float,
        t_min: float,
        ea: float,
        doy: int,
    ) -> float:
        """Calculate net radiation (MJ/m²/day)."""
        lat_rad = np.radians(self.latitude)

        # Extraterrestrial radiation
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta_solar = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

        # Handle polar regions
        tan_product = -np.tan(lat_rad) * np.tan(delta_solar)
        tan_product = np.clip(tan_product, -1, 1)
        ws = np.arccos(tan_product)

        ra = 24 * 60 / np.pi * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta_solar) +
            np.cos(lat_rad) * np.cos(delta_solar) * np.sin(ws)
        )

        # Clear-sky radiation
        rso = (0.75 + 2e-5 * self.elevation) * ra

        # Net shortwave radiation
        albedo = 0.23  # Reference grass albedo
        rns = (1 - albedo) * rs

        # Net longwave radiation
        sigma = 4.903e-9  # Stefan-Boltzmann constant
        tk_max = t_max + 273.16
        tk_min = t_min + 273.16

        rnl = sigma * (tk_max ** 4 + tk_min ** 4) / 2 * (
            0.34 - 0.14 * np.sqrt(ea)
        ) * (1.35 * min(rs / rso, 1.0) - 0.35)

        return rns - rnl


class SoilWaterBalance:
    """
    Daily soil water balance model.

    Tracks soil moisture in the root zone considering
    ET, precipitation, irrigation, and drainage.
    """

    # Standard soil parameters by texture
    SOIL_TYPES = {
        'sand': SoilProfile(
            texture='sand',
            field_capacity=0.12,
            wilting_point=0.04,
            saturation=0.43,
            available_water=80,    # mm/m
            infiltration_rate=50,
        ),
        'loamy_sand': SoilProfile(
            texture='loamy_sand',
            field_capacity=0.14,
            wilting_point=0.06,
            saturation=0.44,
            available_water=80,
            infiltration_rate=40,
        ),
        'sandy_loam': SoilProfile(
            texture='sandy_loam',
            field_capacity=0.23,
            wilting_point=0.10,
            saturation=0.45,
            available_water=130,
            infiltration_rate=25,
        ),
        'loam': SoilProfile(
            texture='loam',
            field_capacity=0.27,
            wilting_point=0.12,
            saturation=0.46,
            available_water=150,
            infiltration_rate=15,
        ),
        'clay_loam': SoilProfile(
            texture='clay_loam',
            field_capacity=0.32,
            wilting_point=0.15,
            saturation=0.47,
            available_water=170,
            infiltration_rate=10,
        ),
        'clay': SoilProfile(
            texture='clay',
            field_capacity=0.38,
            wilting_point=0.22,
            saturation=0.50,
            available_water=160,
            infiltration_rate=5,
        ),
    }

    def __init__(self, zone: IrrigationZone):
        """
        Initialize water balance for irrigation zone.

        Args:
            zone: Irrigation zone configuration
        """
        self.zone = zone
        self.soil = zone.soil

        # Total available water in root zone (mm)
        self.TAW = self.soil.available_water * zone.root_depth_m

        # Readily available water (before stress)
        self.RAW = self.TAW * zone.management_allowable_depletion

        # Initialize at field capacity
        self.current_depletion = 0  # Dr = 0 at FC

    def daily_balance(
        self,
        et0: float,
        precipitation: float,
        irrigation: float = 0,
    ) -> Dict[str, float]:
        """
        Calculate daily water balance update.

        Args:
            et0: Reference evapotranspiration (mm)
            precipitation: Precipitation (mm)
            irrigation: Irrigation applied (mm)

        Returns:
            Dictionary with balance components

        Example:
            >>> balance = swb.daily_balance(et0=5.0, precipitation=2.0)
            >>> print(f"Depletion: {balance['depletion']:.1f} mm")
        """
        # Crop evapotranspiration
        kc = self.zone.kc
        etc = et0 * kc

        # Water stress coefficient
        if self.current_depletion <= self.RAW:
            ks = 1.0
        else:
            ks = (self.TAW - self.current_depletion) / (self.TAW - self.RAW)
            ks = max(0, min(1, ks))

        # Actual ET under stress
        etc_adj = etc * ks

        # Deep percolation if precipitation exceeds soil capacity
        available_storage = self.current_depletion
        total_input = precipitation + irrigation
        effective_input = min(total_input, available_storage + self.TAW)
        deep_percolation = max(0, total_input - effective_input)

        # Update depletion
        self.current_depletion = self.current_depletion - total_input + deep_percolation + etc_adj
        self.current_depletion = max(0, min(self.TAW, self.current_depletion))

        # Soil moisture content
        theta = self.soil.field_capacity - (self.current_depletion / (self.zone.root_depth_m * 1000))

        return {
            'et0': et0,
            'etc': etc,
            'etc_adj': etc_adj,
            'ks': ks,
            'precipitation': precipitation,
            'irrigation': irrigation,
            'deep_percolation': deep_percolation,
            'depletion': self.current_depletion,
            'soil_moisture': theta,
            'relative_depletion': self.current_depletion / self.TAW,
            'TAW': self.TAW,
            'RAW': self.RAW,
        }

    def irrigation_required(self) -> float:
        """
        Calculate irrigation amount needed.

        Determines water needed to refill root zone
        to field capacity.

        Returns:
            Required irrigation depth (mm)
        """
        return self.current_depletion / self.zone.irrigation_efficiency

    def days_until_stress(self, daily_et: float) -> float:
        """
        Estimate days until water stress begins.

        Args:
            daily_et: Expected daily ET (mm)

        Returns:
            Estimated days until stress
        """
        if daily_et <= 0:
            return float('inf')

        remaining_raw = max(0, self.RAW - self.current_depletion)
        return remaining_raw / daily_et


class IrrigationScheduler:
    """
    Intelligent irrigation scheduling system.

    Develops optimal irrigation schedules based on
    crop water requirements, soil conditions, and
    system constraints.
    """

    def __init__(
        self,
        zones: List[IrrigationZone],
        latitude: float,
        elevation: float = 0,
    ):
        """
        Initialize scheduler with irrigation zones.

        Args:
            zones: List of irrigation zones
            latitude: Site latitude (degrees)
            elevation: Site elevation (m)
        """
        self.zones = {z.zone_id: z for z in zones}
        self.et_calculator = PenmanMonteithET(latitude, elevation)
        self.water_balances = {
            z.zone_id: SoilWaterBalance(z) for z in zones
        }

    def calculate_daily_requirements(
        self,
        weather: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        Calculate irrigation requirements for all zones.

        Args:
            weather: Weather data for the day

        Returns:
            Dictionary of requirements per zone

        Example:
            >>> weather = {'t_max': 32, 't_min': 18, ...}
            >>> reqs = scheduler.calculate_daily_requirements(weather)
        """
        # Calculate reference ET
        et0 = self.et_calculator.calculate_eto(
            t_max=weather['t_max'],
            t_min=weather['t_min'],
            humidity=weather.get('humidity', 60),
            wind_speed=weather.get('wind_speed', 2.0),
            solar_radiation=weather.get('solar_radiation', 20),
            doy=weather.get('doy', 180),
        )

        precipitation = weather.get('precipitation', 0)

        requirements = {}
        for zone_id, wb in self.water_balances.items():
            zone = self.zones[zone_id]

            # Run daily balance
            balance = wb.daily_balance(et0, precipitation)

            # Determine if irrigation needed
            needs_irrigation = wb.current_depletion > wb.RAW

            requirements[zone_id] = {
                'zone_name': zone.name,
                'et0': et0,
                'etc': balance['etc'],
                'current_depletion': balance['depletion'],
                'relative_depletion': balance['relative_depletion'],
                'needs_irrigation': needs_irrigation,
                'irrigation_depth_mm': wb.irrigation_required() if needs_irrigation else 0,
                'irrigation_volume_m3': (wb.irrigation_required() * zone.area_ha * 10) if needs_irrigation else 0,
                'days_to_stress': wb.days_until_stress(balance['etc']),
                'water_stress': balance['ks'] < 1.0,
                'stress_coefficient': balance['ks'],
            }

        return requirements

    def generate_schedule(
        self,
        weather_forecast: List[Dict[str, float]],
        max_daily_volume_m3: float = float('inf'),
    ) -> List[Dict]:
        """
        Generate multi-day irrigation schedule.

        Optimizes irrigation timing based on forecast
        and system constraints.

        Args:
            weather_forecast: List of daily weather dictionaries
            max_daily_volume_m3: Maximum daily water use (m³)

        Returns:
            List of daily schedules

        Example:
            >>> forecast = [{'t_max': 30, ...}, {'t_max': 32, ...}]
            >>> schedule = scheduler.generate_schedule(forecast)
        """
        schedule = []

        for day_idx, weather in enumerate(weather_forecast):
            requirements = self.calculate_daily_requirements(weather)

            # Sort zones by urgency (lowest days_to_stress first)
            zone_priority = sorted(
                requirements.items(),
                key=lambda x: (not x[1]['needs_irrigation'], x[1]['days_to_stress'])
            )

            # Allocate water within daily limit
            daily_volume_used = 0
            day_schedule = {
                'day': day_idx + 1,
                'date': weather.get('date', f"Day {day_idx + 1}"),
                'et0': requirements[list(requirements.keys())[0]]['et0'],
                'precipitation': weather.get('precipitation', 0),
                'zones': [],
            }

            for zone_id, req in zone_priority:
                if req['needs_irrigation']:
                    volume_needed = req['irrigation_volume_m3']

                    if daily_volume_used + volume_needed <= max_daily_volume_m3:
                        # Apply irrigation
                        wb = self.water_balances[zone_id]
                        irrigation_mm = wb.irrigation_required()
                        wb.current_depletion = 0  # Refill to FC

                        day_schedule['zones'].append({
                            'zone_id': zone_id,
                            'zone_name': req['zone_name'],
                            'irrigation_mm': irrigation_mm,
                            'volume_m3': volume_needed,
                            'runtime_hours': self._calculate_runtime(zone_id, irrigation_mm),
                        })

                        daily_volume_used += volume_needed

            day_schedule['total_volume_m3'] = daily_volume_used
            schedule.append(day_schedule)

        return schedule

    def _calculate_runtime(self, zone_id: str, depth_mm: float) -> float:
        """Calculate irrigation runtime (hours)."""
        zone = self.zones[zone_id]
        # Assume application rate of 10 mm/hour for sprinkler
        application_rate = zone.soil.infiltration_rate * 0.8
        return depth_mm / application_rate

    def get_water_use_summary(
        self,
        schedule: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate water use efficiency metrics.

        Args:
            schedule: Generated irrigation schedule

        Returns:
            Summary of water use metrics
        """
        total_volume = sum(day['total_volume_m3'] for day in schedule)
        irrigation_days = sum(1 for day in schedule if day['zones'])

        total_area = sum(z.area_ha for z in self.zones.values())

        return {
            'total_volume_m3': total_volume,
            'avg_daily_volume_m3': total_volume / len(schedule) if schedule else 0,
            'irrigation_days': irrigation_days,
            'irrigation_frequency': irrigation_days / len(schedule) if schedule else 0,
            'water_per_hectare_m3': total_volume / total_area if total_area else 0,
            'total_area_ha': total_area,
        }


def simulate_weather_forecast(
    start_date: date,
    n_days: int,
    base_temp: float = 25.0,
) -> List[Dict]:
    """
    Generate synthetic weather forecast.

    Args:
        start_date: Start date
        n_days: Number of days
        base_temp: Base temperature (°C)

    Returns:
        List of daily weather dictionaries
    """
    np.random.seed(42)

    forecast = []
    for i in range(n_days):
        current_date = start_date + timedelta(days=i)
        doy = current_date.timetuple().tm_yday

        # Seasonal temperature pattern
        seasonal_adj = 10 * np.sin(2 * np.pi * (doy - 172) / 365)
        t_mean = base_temp + seasonal_adj + np.random.normal(0, 2)

        # Precipitation (low probability in summer)
        precip_prob = 0.15 + 0.1 * np.cos(2 * np.pi * (doy - 172) / 365)
        precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

        forecast.append({
            'date': current_date,
            't_max': t_mean + np.random.uniform(5, 9),
            't_min': t_mean - np.random.uniform(5, 9),
            'humidity': 40 + 30 * np.random.random() + precipitation * 3,
            'wind_speed': 1.5 + np.random.exponential(1.5),
            'solar_radiation': 20 + 8 * np.sin(2 * np.pi * (doy - 172) / 365) * (1 - precipitation / 20),
            'precipitation': precipitation,
            'doy': doy,
        })

    return forecast


def run_demonstration():
    """
    Run comprehensive irrigation scheduling demonstration.
    """
    print("=" * 70)
    print("Smart Agriculture - Irrigation Optimization Demo")
    print("=" * 70)
    print()

    # Create irrigation zones
    zones = [
        IrrigationZone(
            zone_id="field_a",
            name="North Corn Field",
            area_ha=25.0,
            crop_type="corn",
            soil=SoilWaterBalance.SOIL_TYPES['loam'],
            root_depth_m=0.9,
            management_allowable_depletion=0.55,
            kc=1.15,
            irrigation_efficiency=0.85,
        ),
        IrrigationZone(
            zone_id="field_b",
            name="South Soybean Field",
            area_ha=18.0,
            crop_type="soybean",
            soil=SoilWaterBalance.SOIL_TYPES['sandy_loam'],
            root_depth_m=0.6,
            management_allowable_depletion=0.50,
            kc=1.05,
            irrigation_efficiency=0.80,
        ),
        IrrigationZone(
            zone_id="field_c",
            name="East Wheat Field",
            area_ha=30.0,
            crop_type="wheat",
            soil=SoilWaterBalance.SOIL_TYPES['clay_loam'],
            root_depth_m=0.8,
            management_allowable_depletion=0.60,
            kc=0.95,
            irrigation_efficiency=0.85,
        ),
    ]

    print("Irrigation Zones Configuration:")
    print("-" * 50)
    for zone in zones:
        print(f"\n  {zone.name} ({zone.zone_id}):")
        print(f"    • Area: {zone.area_ha} ha")
        print(f"    • Crop: {zone.crop_type} (Kc = {zone.kc})")
        print(f"    • Soil: {zone.soil.texture}")
        print(f"    • Root depth: {zone.root_depth_m} m")
        print(f"    • TAW: {zone.soil.available_water * zone.root_depth_m:.0f} mm")
        print(f"    • MAD: {zone.management_allowable_depletion * 100:.0f}%")

    # Initialize scheduler
    scheduler = IrrigationScheduler(
        zones=zones,
        latitude=42.0,
        elevation=250.0,
    )

    print("\n\nEvapotranspiration Calculator:")
    print("-" * 50)
    print(f"  • Latitude: 42.0°N")
    print(f"  • Elevation: 250 m")
    print(f"  • Atmospheric pressure: {scheduler.et_calculator.pressure:.1f} kPa")

    # Sample ET calculation
    sample_weather = {
        't_max': 32,
        't_min': 18,
        'humidity': 55,
        'wind_speed': 2.5,
        'solar_radiation': 24,
        'doy': 180,
    }
    sample_et0 = scheduler.et_calculator.calculate_eto(**sample_weather)
    print(f"\n  Sample ETo calculation (mid-summer day):")
    print(f"    • Temperature: {sample_weather['t_min']}°C - {sample_weather['t_max']}°C")
    print(f"    • Humidity: {sample_weather['humidity']}%")
    print(f"    • Wind: {sample_weather['wind_speed']} m/s")
    print(f"    • Solar radiation: {sample_weather['solar_radiation']} MJ/m²/day")
    print(f"    • Reference ET (ETo): {sample_et0:.2f} mm/day")

    # Generate 14-day forecast
    print("\n\n14-Day Weather Forecast:")
    print("-" * 50)
    forecast = simulate_weather_forecast(date(2024, 7, 15), 14)

    print(f"\n  {'Day':>3} | {'Date':>10} | {'Tmax':>5} | {'Tmin':>5} | {'Rain':>5} | {'ETo':>5}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    for i, day in enumerate(forecast[:7]):  # Show first week
        et0 = scheduler.et_calculator.calculate_eto(
            t_max=day['t_max'],
            t_min=day['t_min'],
            humidity=day['humidity'],
            wind_speed=day['wind_speed'],
            solar_radiation=day['solar_radiation'],
            doy=day['doy'],
        )
        print(f"  {i+1:>3} | {day['date']} | {day['t_max']:5.1f} | {day['t_min']:5.1f} | "
              f"{day['precipitation']:5.1f} | {et0:5.2f}")
    print(f"  ... (showing 7 of 14 days)")

    # Generate irrigation schedule
    print("\n\nIrrigation Schedule Generation:")
    print("-" * 50)

    # Set water constraint (simulate limited water supply)
    max_daily_volume = 2000  # m³/day

    schedule = scheduler.generate_schedule(
        weather_forecast=forecast,
        max_daily_volume_m3=max_daily_volume,
    )

    print(f"\n  Constraint: Maximum {max_daily_volume} m³/day")
    print(f"\n  Day-by-Day Schedule:")
    print()

    for day in schedule[:7]:  # Show first week
        print(f"  Day {day['day']} ({day['date']}):")
        print(f"    ETo: {day['et0']:.1f} mm | Precip: {day['precipitation']:.1f} mm")

        if day['zones']:
            print(f"    Irrigation events:")
            for zone in day['zones']:
                print(f"      • {zone['zone_name']}: {zone['irrigation_mm']:.1f} mm "
                      f"({zone['volume_m3']:.0f} m³, {zone['runtime_hours']:.1f} hrs)")
            print(f"    Total volume: {day['total_volume_m3']:.0f} m³")
        else:
            print(f"    No irrigation scheduled")
        print()

    # Water use summary
    summary = scheduler.get_water_use_summary(schedule)

    print("\nWater Use Summary (14 days):")
    print("-" * 50)
    print(f"  • Total irrigated area: {summary['total_area_ha']:.1f} ha")
    print(f"  • Total water volume: {summary['total_volume_m3']:.0f} m³")
    print(f"  • Average daily volume: {summary['avg_daily_volume_m3']:.0f} m³/day")
    print(f"  • Days with irrigation: {summary['irrigation_days']} of 14")
    print(f"  • Irrigation frequency: {summary['irrigation_frequency']*100:.1f}%")
    print(f"  • Water per hectare: {summary['water_per_hectare_m3']:.0f} m³/ha")

    # Deficit irrigation analysis
    print("\n\nDeficit Irrigation Strategy Comparison:")
    print("-" * 50)

    strategies = [
        ('Full irrigation (MAD 50%)', 0.50),
        ('Mild deficit (MAD 60%)', 0.60),
        ('Moderate deficit (MAD 70%)', 0.70),
    ]

    for name, mad in strategies:
        # Reset zones with new MAD
        test_zones = [
            IrrigationZone(
                zone_id=z.zone_id,
                name=z.name,
                area_ha=z.area_ha,
                crop_type=z.crop_type,
                soil=z.soil,
                root_depth_m=z.root_depth_m,
                management_allowable_depletion=mad,
                kc=z.kc,
                irrigation_efficiency=z.irrigation_efficiency,
            )
            for z in zones
        ]

        test_scheduler = IrrigationScheduler(test_zones, 42.0, 250.0)
        test_schedule = test_scheduler.generate_schedule(forecast)
        test_summary = test_scheduler.get_water_use_summary(test_schedule)

        print(f"\n  {name}:")
        print(f"    • Total water: {test_summary['total_volume_m3']:.0f} m³")
        print(f"    • Irrigation events: {test_summary['irrigation_days']}")
        print(f"    • Water savings vs full: "
              f"{(1 - test_summary['total_volume_m3']/summary['total_volume_m3'])*100:.1f}%")

    # Zone-specific recommendations
    print("\n\nZone-Specific Recommendations:")
    print("-" * 50)

    current_requirements = scheduler.calculate_daily_requirements(forecast[0])

    for zone_id, req in current_requirements.items():
        print(f"\n  {req['zone_name']}:")
        print(f"    • Current ETc: {req['etc']:.2f} mm/day")
        print(f"    • Soil depletion: {req['current_depletion']:.1f} mm ({req['relative_depletion']*100:.1f}%)")
        print(f"    • Days to stress: {req['days_to_stress']:.1f}")

        if req['needs_irrigation']:
            print(f"    • STATUS: Irrigation needed - {req['irrigation_depth_mm']:.1f} mm")
        elif req['days_to_stress'] < 3:
            print(f"    • STATUS: Monitor closely - approaching stress threshold")
        else:
            print(f"    • STATUS: Adequate moisture")

    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


def main():
    """Main entry point."""
    run_demonstration()


if __name__ == "__main__":
    main()
