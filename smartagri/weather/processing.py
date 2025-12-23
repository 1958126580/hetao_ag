"""
Weather Data Processing Module

Handles weather data acquisition, quality control, and preprocessing
for agricultural applications.

Features:
    - Multi-source data integration
    - Quality control and gap filling
    - Temporal aggregation
    - Spatial interpolation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeatherData:
    """
    Container for weather observations.

    Attributes:
        dates: Array of dates/timestamps
        t_max: Maximum temperature (°C)
        t_min: Minimum temperature (°C)
        t_mean: Mean temperature (°C)
        precipitation: Precipitation (mm)
        humidity: Relative humidity (%)
        wind_speed: Wind speed (m/s)
        solar_radiation: Solar radiation (MJ/m2/day)
        pressure: Atmospheric pressure (kPa)
        dew_point: Dew point temperature (°C)
    """
    dates: np.ndarray
    t_max: np.ndarray
    t_min: np.ndarray
    t_mean: Optional[np.ndarray] = None
    precipitation: Optional[np.ndarray] = None
    humidity: Optional[np.ndarray] = None
    wind_speed: Optional[np.ndarray] = None
    solar_radiation: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    dew_point: Optional[np.ndarray] = None

    def __post_init__(self):
        """Calculate derived variables if missing."""
        n = len(self.dates)

        if self.t_mean is None:
            self.t_mean = (self.t_max + self.t_min) / 2

        if self.precipitation is None:
            self.precipitation = np.zeros(n)

        if self.humidity is None:
            self.humidity = np.full(n, 70.0)

        if self.wind_speed is None:
            self.wind_speed = np.full(n, 2.0)

    def __len__(self) -> int:
        return len(self.dates)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary."""
        return {
            "dates": self.dates,
            "t_max": self.t_max,
            "t_min": self.t_min,
            "t_mean": self.t_mean,
            "precipitation": self.precipitation,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "solar_radiation": self.solar_radiation,
            "pressure": self.pressure,
        }


@dataclass
class WeatherStation:
    """
    Weather station metadata.

    Attributes:
        station_id: Unique identifier
        name: Station name
        latitude: Latitude (degrees)
        longitude: Longitude (degrees)
        elevation: Elevation (m)
        start_date: Data start date
        end_date: Data end date
    """
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float = 0.0
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class QualityControl:
    """
    Weather data quality control system.

    Implements range checks, consistency tests, and gap filling
    to ensure data quality for agricultural applications.

    Example:
        >>> qc = QualityControl()
        >>> clean_data, flags = qc.process(raw_data)
    """

    def __init__(self):
        """Initialize quality control system."""
        # Valid ranges for weather variables
        self._ranges = {
            "t_max": (-50, 60),
            "t_min": (-60, 50),
            "t_mean": (-55, 55),
            "precipitation": (0, 500),
            "humidity": (0, 100),
            "wind_speed": (0, 50),
            "solar_radiation": (0, 50),
            "pressure": (80, 110),
        }

    def range_check(
        self,
        data: WeatherData,
    ) -> Dict[str, np.ndarray]:
        """
        Perform range checks on weather data.

        Args:
            data: Weather data to check

        Returns:
            Dict of boolean flags (True = passed)
        """
        flags = {}

        for var, (min_val, max_val) in self._ranges.items():
            values = getattr(data, var, None)
            if values is not None:
                valid = (values >= min_val) & (values <= max_val)
                valid = valid | np.isnan(values)  # NaN passes range check
                flags[var] = valid

        return flags

    def consistency_check(
        self,
        data: WeatherData,
    ) -> Dict[str, np.ndarray]:
        """
        Check data consistency.

        Args:
            data: Weather data to check

        Returns:
            Dict of consistency flags
        """
        n = len(data)
        flags = {}

        # T_max >= T_min
        flags["temp_order"] = data.t_max >= data.t_min

        # T_mean within T_min and T_max
        if data.t_mean is not None:
            flags["temp_mean"] = (
                (data.t_mean >= data.t_min) &
                (data.t_mean <= data.t_max)
            )

        # Solar radiation positive during day
        if data.solar_radiation is not None:
            flags["radiation_positive"] = data.solar_radiation >= 0

        return flags

    def temporal_check(
        self,
        data: WeatherData,
        max_change: Dict[str, float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Check for unrealistic temporal changes.

        Args:
            data: Weather data
            max_change: Maximum allowed daily change

        Returns:
            Dict of temporal consistency flags
        """
        if max_change is None:
            max_change = {
                "t_max": 20,
                "t_min": 20,
                "pressure": 5,
            }

        flags = {}

        for var, max_delta in max_change.items():
            values = getattr(data, var, None)
            if values is not None and len(values) > 1:
                changes = np.abs(np.diff(values))
                flags[var] = np.concatenate([[True], changes <= max_delta])

        return flags

    def fill_gaps(
        self,
        data: WeatherData,
        method: str = "interpolate",
    ) -> WeatherData:
        """
        Fill gaps in weather data.

        Args:
            data: Weather data with gaps
            method: Gap filling method

        Returns:
            Gap-filled weather data
        """
        filled_data = WeatherData(
            dates=data.dates.copy(),
            t_max=self._fill_array(data.t_max, method),
            t_min=self._fill_array(data.t_min, method),
            t_mean=self._fill_array(data.t_mean, method),
            precipitation=self._fill_array(data.precipitation, "zero"),
            humidity=self._fill_array(data.humidity, method),
            wind_speed=self._fill_array(data.wind_speed, method),
            solar_radiation=self._fill_array(data.solar_radiation, method),
            pressure=self._fill_array(data.pressure, method),
        )

        return filled_data

    def _fill_array(
        self,
        arr: Optional[np.ndarray],
        method: str,
    ) -> Optional[np.ndarray]:
        """Fill gaps in single array."""
        if arr is None:
            return None

        arr = arr.copy()
        nan_mask = np.isnan(arr)

        if not np.any(nan_mask):
            return arr

        if method == "interpolate":
            # Linear interpolation
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 1:
                arr[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    valid_idx,
                    arr[valid_idx]
                )
            elif len(valid_idx) == 1:
                arr[nan_mask] = arr[valid_idx[0]]

        elif method == "zero":
            arr[nan_mask] = 0

        elif method == "mean":
            arr[nan_mask] = np.nanmean(arr)

        return arr

    def process(
        self,
        data: WeatherData,
        fill_gaps: bool = True,
    ) -> Tuple[WeatherData, Dict[str, Any]]:
        """
        Complete quality control processing.

        Args:
            data: Raw weather data
            fill_gaps: Whether to fill gaps

        Returns:
            Tuple of (processed_data, quality_report)
        """
        # Run all checks
        range_flags = self.range_check(data)
        consistency_flags = self.consistency_check(data)
        temporal_flags = self.temporal_check(data)

        # Compile report
        report = {
            "range_check": range_flags,
            "consistency_check": consistency_flags,
            "temporal_check": temporal_flags,
            "total_records": len(data),
        }

        # Count issues
        total_issues = 0
        for check_type in [range_flags, consistency_flags, temporal_flags]:
            for var, flags in check_type.items():
                total_issues += np.sum(~flags)

        report["total_issues"] = total_issues
        report["quality_score"] = 1 - total_issues / (len(data) * 10)

        # Fill gaps if requested
        if fill_gaps:
            processed = self.fill_gaps(data)
        else:
            processed = data

        return processed, report


class WeatherProcessor:
    """
    Comprehensive weather data processor.

    Handles data loading, quality control, aggregation,
    and derived variable calculation.

    Example:
        >>> processor = WeatherProcessor()
        >>> data = processor.load_data(file_path)
        >>> daily = processor.aggregate(data, "daily")
        >>> processor.calculate_derived(daily)
    """

    def __init__(self):
        """Initialize weather processor."""
        self.qc = QualityControl()

    def create_weather_data(
        self,
        t_max: np.ndarray,
        t_min: np.ndarray,
        start_date: date,
        **kwargs,
    ) -> WeatherData:
        """
        Create WeatherData object from arrays.

        Args:
            t_max: Maximum temperatures
            t_min: Minimum temperatures
            start_date: Start date
            **kwargs: Additional weather variables

        Returns:
            WeatherData object
        """
        n = len(t_max)
        dates = np.array([
            start_date + timedelta(days=i) for i in range(n)
        ])

        return WeatherData(
            dates=dates,
            t_max=np.asarray(t_max),
            t_min=np.asarray(t_min),
            **kwargs,
        )

    def aggregate(
        self,
        data: WeatherData,
        period: str = "monthly",
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate weather data to different time periods.

        Args:
            data: Daily weather data
            period: Aggregation period ('weekly', 'monthly', 'annual')

        Returns:
            Dict with aggregated values
        """
        dates = data.dates

        if period == "weekly":
            # Group by week
            week_nums = np.array([d.isocalendar()[1] for d in dates])
            unique_weeks = np.unique(week_nums)

            aggregated = {"period": unique_weeks}
            for var in ["t_max", "t_min", "t_mean", "precipitation"]:
                values = getattr(data, var)
                if values is not None:
                    agg_values = [values[week_nums == w].mean() for w in unique_weeks]
                    aggregated[var] = np.array(agg_values)

        elif period == "monthly":
            # Group by month
            months = np.array([d.month + d.year * 12 for d in dates])
            unique_months = np.unique(months)

            aggregated = {"period": unique_months}
            for var in ["t_max", "t_min", "t_mean"]:
                values = getattr(data, var)
                if values is not None:
                    agg_values = [values[months == m].mean() for m in unique_months]
                    aggregated[var] = np.array(agg_values)

            # Sum precipitation
            if data.precipitation is not None:
                aggregated["precipitation"] = np.array([
                    data.precipitation[months == m].sum() for m in unique_months
                ])

        else:  # annual
            years = np.array([d.year for d in dates])
            unique_years = np.unique(years)

            aggregated = {"period": unique_years}
            for var in ["t_max", "t_min", "t_mean"]:
                values = getattr(data, var)
                if values is not None:
                    agg_values = [values[years == y].mean() for y in unique_years]
                    aggregated[var] = np.array(agg_values)

            if data.precipitation is not None:
                aggregated["precipitation"] = np.array([
                    data.precipitation[years == y].sum() for y in unique_years
                ])

        return aggregated

    def calculate_derived(
        self,
        data: WeatherData,
        latitude: float = 45.0,
    ) -> WeatherData:
        """
        Calculate derived weather variables.

        Args:
            data: Weather data
            latitude: Site latitude for radiation estimates

        Returns:
            Weather data with derived variables
        """
        n = len(data)

        # Calculate solar radiation if missing
        if data.solar_radiation is None:
            # Hargreaves radiation estimate
            doy = np.array([d.timetuple().tm_yday for d in data.dates])
            ra = self._extraterrestrial_radiation(latitude, doy)
            kt = 0.17 * np.sqrt(data.t_max - data.t_min)
            data.solar_radiation = ra * kt

        # Calculate dew point if missing
        if data.dew_point is None and data.humidity is not None:
            # Magnus formula approximation
            a = 17.27
            b = 237.7
            alpha = (a * data.t_mean) / (b + data.t_mean) + np.log(data.humidity / 100)
            data.dew_point = (b * alpha) / (a - alpha)

        return data

    def _extraterrestrial_radiation(
        self,
        latitude: float,
        doy: np.ndarray,
    ) -> np.ndarray:
        """Calculate extraterrestrial radiation (MJ/m2/day)."""
        lat_rad = np.radians(latitude)

        # Solar constant
        gsc = 0.0820  # MJ/m2/min

        # Inverse relative distance Earth-Sun
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

        # Solar declination
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

        # Sunset hour angle
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))

        # Extraterrestrial radiation
        ra = (24 * 60 / np.pi) * gsc * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        return ra
