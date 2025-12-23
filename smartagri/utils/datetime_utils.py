"""
Date and Time Utilities for Agricultural Applications

Provides specialized date/time functions for agricultural calendaring,
including growing degree days, solar calculations, and phenology timing.

Features:
    - Growing degree day (GDD) calculations
    - Solar position and day length calculations
    - Agricultural calendar management
    - Phenological stage timing
    - Frost date estimation
"""

import numpy as np
from typing import Union, Optional, List, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import math

ArrayLike = Union[float, np.ndarray, List[float]]


@dataclass
class SolarPosition:
    """
    Solar position parameters.

    Attributes:
        declination: Solar declination angle (degrees)
        hour_angle: Solar hour angle (degrees)
        altitude: Solar altitude angle (degrees)
        azimuth: Solar azimuth angle (degrees)
        zenith: Solar zenith angle (degrees)
    """
    declination: float
    hour_angle: float
    altitude: float
    azimuth: float
    zenith: float


def day_of_year(dt: Union[datetime, date, str]) -> int:
    """
    Get day of year (1-366).

    Args:
        dt: Date as datetime, date object, or string (YYYY-MM-DD)

    Returns:
        Day of year (1-366)

    Example:
        >>> doy = day_of_year("2024-03-21")
        >>> print(f"March 21 is day {doy}")  # 81
    """
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d")
    if isinstance(dt, datetime):
        dt = dt.date()

    return (dt - date(dt.year, 1, 1)).days + 1


def julian_day(dt: Union[datetime, date, str]) -> float:
    """
    Calculate Julian Day Number.

    Args:
        dt: Date as datetime, date object, or string

    Returns:
        Julian Day Number

    Example:
        >>> jd = julian_day("2024-01-01")
    """
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d")
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())

    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400

    if month <= 2:
        year -= 1
        month += 12

    a = int(year / 100)
    b = 2 - a + int(a / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5

    return jd


def solar_declination(doy: int) -> float:
    """
    Calculate solar declination angle.

    Args:
        doy: Day of year (1-366)

    Returns:
        Solar declination in degrees

    Example:
        >>> decl = solar_declination(172)  # Summer solstice
        >>> print(f"Declination: {decl:.2f}°")  # ~23.45°
    """
    # Convert day of year to radians for orbital position
    gamma = 2 * np.pi * (doy - 1) / 365

    # Solar declination (Spencer 1971)
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    return np.degrees(declination)


def daylight_hours(latitude: float, doy: int) -> float:
    """
    Calculate daylight hours for a given location and date.

    Args:
        latitude: Latitude in degrees (-90 to 90)
        doy: Day of year (1-366)

    Returns:
        Hours of daylight

    Example:
        >>> hours = daylight_hours(45.0, 172)  # Summer solstice at 45°N
        >>> print(f"Daylight: {hours:.2f} hours")
    """
    declination = np.radians(solar_declination(doy))
    lat_rad = np.radians(latitude)

    # Hour angle at sunrise/sunset
    cos_omega = -np.tan(lat_rad) * np.tan(declination)

    # Handle polar day/night
    if cos_omega < -1:
        return 24.0  # Polar day
    elif cos_omega > 1:
        return 0.0  # Polar night

    omega = np.arccos(cos_omega)

    # Convert to hours
    return 24 * omega / np.pi


def calculate_solar_position(
    latitude: float,
    longitude: float,
    dt: datetime,
) -> SolarPosition:
    """
    Calculate detailed solar position.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        dt: Date and time

    Returns:
        SolarPosition with all angular parameters

    Example:
        >>> pos = calculate_solar_position(45.0, -122.0, datetime(2024, 6, 21, 12, 0))
        >>> print(f"Solar altitude: {pos.altitude:.1f}°")
    """
    doy = day_of_year(dt)
    declination = solar_declination(doy)

    # Calculate time correction
    b = 2 * np.pi * (doy - 81) / 365

    # Equation of time (minutes)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    # Solar time
    solar_time = (
        dt.hour
        + dt.minute / 60
        + dt.second / 3600
        + eot / 60
        + longitude / 15
    )

    # Hour angle
    hour_angle = 15 * (solar_time - 12)

    # Convert to radians
    lat_rad = np.radians(latitude)
    decl_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    # Solar altitude
    sin_alt = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    )
    altitude = np.degrees(np.arcsin(sin_alt))

    # Solar azimuth
    cos_az = (
        np.sin(decl_rad) - np.sin(lat_rad) * sin_alt
    ) / (np.cos(lat_rad) * np.cos(np.arcsin(sin_alt)))
    cos_az = np.clip(cos_az, -1, 1)
    azimuth = np.degrees(np.arccos(cos_az))

    if hour_angle > 0:
        azimuth = 360 - azimuth

    return SolarPosition(
        declination=declination,
        hour_angle=hour_angle,
        altitude=altitude,
        azimuth=azimuth,
        zenith=90 - altitude,
    )


def growing_degree_days(
    t_max: ArrayLike,
    t_min: ArrayLike,
    t_base: float = 10.0,
    t_upper: float = 30.0,
    method: str = "averaging",
) -> np.ndarray:
    """
    Calculate growing degree days (GDD).

    GDD measures heat accumulation for crop development.

    Args:
        t_max: Maximum daily temperature(s) (°C)
        t_min: Minimum daily temperature(s) (°C)
        t_base: Base temperature (°C)
        t_upper: Upper threshold temperature (°C)
        method: Calculation method ('averaging', 'modified', 'single_sine')

    Returns:
        Array of GDD values

    Example:
        >>> t_max = [25, 28, 30, 22]
        >>> t_min = [15, 18, 20, 12]
        >>> gdd = growing_degree_days(t_max, t_min, t_base=10)
        >>> cumulative_gdd = np.cumsum(gdd)
    """
    t_max = np.asarray(t_max)
    t_min = np.asarray(t_min)

    if method == "averaging":
        # Simple averaging method
        t_avg = (t_max + t_min) / 2
        gdd = np.maximum(t_avg - t_base, 0)

    elif method == "modified":
        # Modified method with upper cutoff
        t_max_adj = np.minimum(t_max, t_upper)
        t_min_adj = np.maximum(t_min, t_base)
        t_min_adj = np.minimum(t_min_adj, t_max_adj)

        t_avg = (t_max_adj + t_min_adj) / 2
        gdd = np.maximum(t_avg - t_base, 0)

    elif method == "single_sine":
        # Single sine wave method
        gdd = np.zeros_like(t_max, dtype=float)

        for i in range(len(t_max)):
            if t_min[i] >= t_upper:
                gdd[i] = t_upper - t_base
            elif t_max[i] <= t_base:
                gdd[i] = 0
            elif t_min[i] >= t_base and t_max[i] <= t_upper:
                gdd[i] = (t_max[i] + t_min[i]) / 2 - t_base
            else:
                # Partial sine calculation
                amp = (t_max[i] - t_min[i]) / 2
                avg = (t_max[i] + t_min[i]) / 2

                if t_min[i] < t_base:
                    theta1 = np.arcsin((t_base - avg) / amp)
                else:
                    theta1 = -np.pi / 2

                if t_max[i] > t_upper:
                    theta2 = np.arcsin((t_upper - avg) / amp)
                else:
                    theta2 = np.pi / 2

                gdd[i] = (
                    (avg - t_base) * (theta2 - theta1)
                    + amp * (np.cos(theta1) - np.cos(theta2))
                ) / np.pi

    else:
        raise ValueError(f"Unknown GDD method: {method}")

    return gdd


class AgriculturalCalendar:
    """
    Agricultural calendar for tracking growing seasons and events.

    Manages planting dates, harvest windows, and phenological stages
    for different crops based on location and climate data.

    Example:
        >>> calendar = AgriculturalCalendar(latitude=40.0, longitude=-90.0)
        >>> planting = calendar.optimal_planting_date("corn", year=2024)
        >>> frost_free = calendar.frost_free_period(2024)
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation: float = 0.0,
    ):
        """
        Initialize agricultural calendar.

        Args:
            latitude: Location latitude (degrees)
            longitude: Location longitude (degrees)
            elevation: Elevation above sea level (meters)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

        # Crop-specific GDD requirements
        self._crop_gdd = {
            "corn": {"planting_gdd": 50, "maturity_gdd": 2700, "base_temp": 10},
            "wheat_winter": {"planting_gdd": 0, "maturity_gdd": 1800, "base_temp": 0},
            "wheat_spring": {"planting_gdd": 50, "maturity_gdd": 1500, "base_temp": 0},
            "soybean": {"planting_gdd": 100, "maturity_gdd": 2500, "base_temp": 10},
            "rice": {"planting_gdd": 100, "maturity_gdd": 2000, "base_temp": 15},
            "cotton": {"planting_gdd": 150, "maturity_gdd": 2200, "base_temp": 15},
        }

    def optimal_planting_date(
        self,
        crop: str,
        year: int,
        avg_last_frost: Optional[int] = None,
    ) -> date:
        """
        Calculate optimal planting date for a crop.

        Args:
            crop: Crop name
            year: Year for planting
            avg_last_frost: Average last frost day of year (auto-calculated if None)

        Returns:
            Optimal planting date

        Example:
            >>> cal = AgriculturalCalendar(40.0, -90.0)
            >>> planting = cal.optimal_planting_date("corn", 2024)
        """
        if avg_last_frost is None:
            avg_last_frost = self.estimate_last_frost_doy()

        crop_info = self._crop_gdd.get(crop, self._crop_gdd["corn"])

        # Add buffer after last frost
        soil_warmup_days = 14  # Days for soil to warm
        planting_doy = avg_last_frost + soil_warmup_days

        return date(year, 1, 1) + timedelta(days=planting_doy - 1)

    def estimate_last_frost_doy(self) -> int:
        """
        Estimate last spring frost day of year.

        Based on latitude and elevation.

        Returns:
            Estimated day of year for last frost
        """
        # Empirical formula based on US data
        base_doy = 90  # Late March baseline

        # Latitude adjustment (later frost at higher latitudes)
        lat_adjustment = (abs(self.latitude) - 35) * 1.5

        # Elevation adjustment (later frost at higher elevations)
        elev_adjustment = self.elevation / 300

        return int(base_doy + lat_adjustment + elev_adjustment)

    def estimate_first_frost_doy(self) -> int:
        """
        Estimate first fall frost day of year.

        Returns:
            Estimated day of year for first frost
        """
        base_doy = 280  # Early October baseline

        # Earlier frost at higher latitudes
        lat_adjustment = (abs(self.latitude) - 35) * 1.5

        # Earlier frost at higher elevations
        elev_adjustment = self.elevation / 300

        return int(base_doy - lat_adjustment - elev_adjustment)

    def frost_free_period(self, year: int) -> Tuple[date, date, int]:
        """
        Calculate frost-free growing period.

        Args:
            year: Year to calculate

        Returns:
            Tuple of (last_frost, first_frost, frost_free_days)
        """
        last_frost_doy = self.estimate_last_frost_doy()
        first_frost_doy = self.estimate_first_frost_doy()

        last_frost = date(year, 1, 1) + timedelta(days=last_frost_doy - 1)
        first_frost = date(year, 1, 1) + timedelta(days=first_frost_doy - 1)

        frost_free_days = first_frost_doy - last_frost_doy

        return last_frost, first_frost, frost_free_days

    def cumulative_gdd(
        self,
        t_max_series: ArrayLike,
        t_min_series: ArrayLike,
        crop: str,
        start_date: date,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative GDD and growth stage progression.

        Args:
            t_max_series: Daily maximum temperatures
            t_min_series: Daily minimum temperatures
            crop: Crop name
            start_date: Planting date

        Returns:
            Tuple of (cumulative_gdd, dates)
        """
        crop_info = self._crop_gdd.get(crop, self._crop_gdd["corn"])
        base_temp = crop_info["base_temp"]

        daily_gdd = growing_degree_days(
            t_max_series,
            t_min_series,
            t_base=base_temp,
        )

        cumulative = np.cumsum(daily_gdd)
        dates = np.array([
            start_date + timedelta(days=i)
            for i in range(len(daily_gdd))
        ])

        return cumulative, dates

    def predict_maturity_date(
        self,
        t_max_forecast: ArrayLike,
        t_min_forecast: ArrayLike,
        crop: str,
        planting_date: date,
        current_gdd: float = 0.0,
    ) -> Optional[date]:
        """
        Predict crop maturity date from weather forecast.

        Args:
            t_max_forecast: Forecasted max temperatures
            t_min_forecast: Forecasted min temperatures
            crop: Crop name
            planting_date: Planting date
            current_gdd: GDD already accumulated

        Returns:
            Predicted maturity date or None if not reached
        """
        crop_info = self._crop_gdd.get(crop, self._crop_gdd["corn"])
        target_gdd = crop_info["maturity_gdd"]

        cumulative, dates = self.cumulative_gdd(
            t_max_forecast,
            t_min_forecast,
            crop,
            planting_date,
        )

        cumulative += current_gdd

        # Find when target is reached
        maturity_idx = np.searchsorted(cumulative, target_gdd)

        if maturity_idx < len(dates):
            return dates[maturity_idx]
        return None

    def daylight_hours(self, doy: int) -> float:
        """
        Calculate daylight hours for this location.

        Args:
            doy: Day of year

        Returns:
            Hours of daylight
        """
        return daylight_hours(self.latitude, doy)

    def photoperiod_sensitivity(
        self,
        doy: int,
        crop: str,
        critical_day_length: float = 13.0,
    ) -> float:
        """
        Calculate photoperiod sensitivity factor for short-day plants.

        Args:
            doy: Day of year
            crop: Crop name
            critical_day_length: Critical day length for flowering

        Returns:
            Sensitivity factor (0-1)
        """
        day_length = self.daylight_hours(doy)

        if day_length <= critical_day_length:
            return 1.0
        elif day_length >= critical_day_length + 4:
            return 0.0
        else:
            return 1.0 - (day_length - critical_day_length) / 4
