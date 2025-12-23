"""
Climate Analysis Module

Provides climate analysis tools for agricultural applications:
- Long-term climate statistics
- Drought indices (SPI, PDSI, etc.)
- Growing season analysis
- Extreme event detection and analysis

Example:
    >>> analyzer = ClimateAnalyzer(historical_data)
    >>> drought = analyzer.calculate_spi(precipitation, scale=3)
    >>> extreme_events = analyzer.detect_extremes(temperature)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class DroughtSeverity(Enum):
    """Drought severity classification."""
    EXCEPTIONAL = auto()   # D4
    EXTREME = auto()       # D3
    SEVERE = auto()        # D2
    MODERATE = auto()      # D1
    ABNORMAL = auto()      # D0
    NORMAL = auto()        # No drought


@dataclass
class DroughtCondition:
    """
    Drought condition assessment.

    Attributes:
        index_value: Drought index value
        severity: Drought severity classification
        percentile: Historical percentile
        duration_days: Duration of current condition
    """
    index_value: float
    severity: DroughtSeverity
    percentile: float
    duration_days: int = 0


@dataclass
class GrowingSeason:
    """
    Growing season characteristics.

    Attributes:
        start_date: Season start (last frost)
        end_date: Season end (first frost)
        length_days: Season length in days
        gdd_accumulated: Total GDD during season
        frost_free_days: Number of frost-free days
    """
    start_date: date
    end_date: date
    length_days: int
    gdd_accumulated: float
    frost_free_days: int


@dataclass
class ExtremeEvent:
    """
    Extreme weather event record.

    Attributes:
        event_type: Type of extreme event
        start_date: Event start date
        end_date: Event end date
        severity: Event severity (1-5 scale)
        peak_value: Peak value during event
        return_period: Estimated return period (years)
    """
    event_type: str
    start_date: date
    end_date: date
    severity: int
    peak_value: float
    return_period: float


class ClimateAnalyzer:
    """
    Long-term climate analysis for agricultural planning.

    Provides tools for analyzing historical climate data
    to support crop selection and management decisions.

    Example:
        >>> analyzer = ClimateAnalyzer()
        >>> normals = analyzer.calculate_normals(temp_data, 30)
        >>> trends = analyzer.detect_trends(temp_data)
    """

    def __init__(
        self,
        reference_period: Tuple[int, int] = (1991, 2020),
    ):
        """
        Initialize climate analyzer.

        Args:
            reference_period: Reference period for normals (start_year, end_year)
        """
        self.reference_period = reference_period

    def calculate_normals(
        self,
        data: np.ndarray,
        period_years: int = 30,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate climate normals.

        Args:
            data: Daily or monthly data
            period_years: Reference period length

        Returns:
            Dictionary with normal statistics
        """
        # Assume monthly data (12 values per year)
        n_years = len(data) // 12

        if n_years < period_years:
            period_years = n_years

        # Use last 'period_years' years
        start_idx = max(0, len(data) - period_years * 12)
        period_data = data[start_idx:]

        # Reshape to (years, months)
        if len(period_data) % 12 == 0:
            monthly = period_data.reshape(-1, 12)
            means = np.mean(monthly, axis=0)
            stds = np.std(monthly, axis=0)
        else:
            means = np.full(12, np.mean(period_data))
            stds = np.full(12, np.std(period_data))

        return {
            'monthly_means': means,
            'monthly_stds': stds,
            'annual_mean': np.mean(data),
            'annual_std': np.std(data),
            'period_years': period_years,
        }

    def calculate_anomaly(
        self,
        current: float,
        normal_mean: float,
        normal_std: float,
    ) -> Dict[str, float]:
        """
        Calculate climate anomaly.

        Args:
            current: Current value
            normal_mean: Normal (average) value
            normal_std: Normal standard deviation

        Returns:
            Anomaly statistics
        """
        anomaly = current - normal_mean
        z_score = anomaly / normal_std if normal_std > 0 else 0

        return {
            'anomaly': anomaly,
            'z_score': z_score,
            'percent_of_normal': current / normal_mean * 100 if normal_mean > 0 else 0,
        }

    def detect_trends(
        self,
        data: np.ndarray,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect climate trends using linear regression.

        Args:
            data: Time series data
            significance_level: Statistical significance level

        Returns:
            Trend analysis results
        """
        n = len(data)
        x = np.arange(n)

        # Linear regression
        slope, intercept = np.polyfit(x, data, 1)

        # Calculate trend significance (simplified)
        residuals = data - (slope * x + intercept)
        se_slope = np.sqrt(np.sum(residuals ** 2) / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
        t_stat = slope / se_slope if se_slope > 0 else 0

        # Simple significance test
        significant = abs(t_stat) > 1.96  # Approximate for large n

        return {
            'slope': slope,
            'intercept': intercept,
            'slope_per_decade': slope * 10,
            't_statistic': t_stat,
            'significant': significant,
            'total_change': slope * n,
        }


class DroughtIndex:
    """
    Drought index calculations.

    Implements multiple drought indices for agricultural
    drought monitoring and assessment.

    Example:
        >>> index = DroughtIndex()
        >>> spi = index.standardized_precipitation_index(precip, scale=3)
        >>> condition = index.classify_drought(spi[-1])
    """

    # SPI drought classification thresholds
    SPI_THRESHOLDS = {
        DroughtSeverity.EXCEPTIONAL: -2.0,
        DroughtSeverity.EXTREME: -1.6,
        DroughtSeverity.SEVERE: -1.3,
        DroughtSeverity.MODERATE: -0.8,
        DroughtSeverity.ABNORMAL: -0.5,
    }

    def standardized_precipitation_index(
        self,
        precipitation: np.ndarray,
        scale: int = 3,
    ) -> np.ndarray:
        """
        Calculate Standardized Precipitation Index (SPI).

        Args:
            precipitation: Monthly precipitation (mm)
            scale: Time scale in months (1, 3, 6, 12, 24)

        Returns:
            SPI values

        Example:
            >>> spi3 = index.standardized_precipitation_index(precip, scale=3)
            >>> print(f"Current SPI-3: {spi3[-1]:.2f}")
        """
        # Rolling sum for scale
        if scale > 1:
            precip_scaled = np.convolve(precipitation, np.ones(scale), mode='valid')
        else:
            precip_scaled = precipitation

        # Fit gamma distribution (simplified to normal for robustness)
        mean = np.mean(precip_scaled)
        std = np.std(precip_scaled)

        if std > 0:
            spi = (precip_scaled - mean) / std
        else:
            spi = np.zeros_like(precip_scaled)

        return spi

    def classify_drought(self, spi_value: float) -> DroughtCondition:
        """
        Classify drought condition based on SPI value.

        Args:
            spi_value: Current SPI value

        Returns:
            DroughtCondition with severity classification
        """
        severity = DroughtSeverity.NORMAL

        for sev, threshold in self.SPI_THRESHOLDS.items():
            if spi_value <= threshold:
                severity = sev
                break

        # Calculate approximate percentile
        from scipy import stats as scipy_stats
        percentile = scipy_stats.norm.cdf(spi_value) * 100

        return DroughtCondition(
            index_value=spi_value,
            severity=severity,
            percentile=percentile,
        )

    def palmer_drought_severity_index(
        self,
        precipitation: np.ndarray,
        temperature: np.ndarray,
        latitude: float,
    ) -> np.ndarray:
        """
        Calculate simplified Palmer Drought Severity Index (PDSI).

        This is a simplified version for demonstration.
        Full PDSI requires additional soil and water balance data.

        Args:
            precipitation: Monthly precipitation (mm)
            temperature: Monthly mean temperature (°C)
            latitude: Station latitude

        Returns:
            PDSI values (simplified)
        """
        # Simplified water balance approach
        n_months = len(precipitation)

        # Estimate potential ET (Thornthwaite method, simplified)
        heat_index = np.sum(np.maximum(temperature / 5, 0) ** 1.514)
        a = 6.75e-7 * heat_index ** 3 - 7.71e-5 * heat_index ** 2 + 1.79e-2 * heat_index + 0.49
        pet = 16 * (10 * np.maximum(temperature, 0) / heat_index) ** a

        # Moisture departure
        moisture_dep = precipitation - pet

        # Standardize
        mean_dep = np.mean(moisture_dep)
        std_dep = np.std(moisture_dep)

        if std_dep > 0:
            pdsi = (moisture_dep - mean_dep) / std_dep * 2
        else:
            pdsi = np.zeros(n_months)

        return pdsi


class GrowingSeasonAnalysis:
    """
    Growing season characterization.

    Analyzes frost dates, growing degree days, and
    seasonal patterns for crop planning.

    Example:
        >>> analysis = GrowingSeasonAnalysis(frost_threshold=-2.0)
        >>> season = analysis.determine_season(t_min_daily)
        >>> print(f"Season length: {season.length_days} days")
    """

    def __init__(
        self,
        frost_threshold: float = 0.0,
        base_temp: float = 10.0,
    ):
        """
        Initialize growing season analysis.

        Args:
            frost_threshold: Temperature threshold for frost (°C)
            base_temp: Base temperature for GDD calculation
        """
        self.frost_threshold = frost_threshold
        self.base_temp = base_temp

    def determine_season(
        self,
        t_min_daily: np.ndarray,
        dates: np.ndarray = None,
    ) -> GrowingSeason:
        """
        Determine growing season from daily minimum temperatures.

        Args:
            t_min_daily: Daily minimum temperatures (°C)
            dates: Corresponding dates

        Returns:
            GrowingSeason with start, end, and characteristics
        """
        n_days = len(t_min_daily)

        if dates is None:
            dates = np.array([date.today() - timedelta(days=n_days-i-1) for i in range(n_days)])

        # Find frost events
        frost_days = t_min_daily <= self.frost_threshold

        # Find last spring frost (first half of year)
        mid_year = n_days // 2
        spring_frost_idx = np.where(frost_days[:mid_year])[0]
        if len(spring_frost_idx) > 0:
            last_spring_frost = spring_frost_idx[-1]
        else:
            last_spring_frost = 0

        # Find first fall frost (second half of year)
        fall_frost_idx = np.where(frost_days[mid_year:])[0]
        if len(fall_frost_idx) > 0:
            first_fall_frost = mid_year + fall_frost_idx[0]
        else:
            first_fall_frost = n_days - 1

        # Season dates
        start_idx = last_spring_frost + 1 if last_spring_frost < mid_year else 0
        end_idx = first_fall_frost - 1 if first_fall_frost > mid_year else n_days - 1

        season_length = end_idx - start_idx

        # Calculate GDD during season (would need t_max for accurate)
        # Using simplified approach with t_min only
        season_temps = t_min_daily[start_idx:end_idx]
        gdd = np.sum(np.maximum(season_temps - self.base_temp, 0))

        return GrowingSeason(
            start_date=dates[start_idx] if start_idx < len(dates) else dates[0],
            end_date=dates[end_idx] if end_idx < len(dates) else dates[-1],
            length_days=season_length,
            gdd_accumulated=gdd,
            frost_free_days=np.sum(~frost_days),
        )

    def calculate_probability(
        self,
        historical_seasons: List[GrowingSeason],
        target_date: date,
    ) -> float:
        """
        Calculate frost probability for target date.

        Args:
            historical_seasons: List of historical seasons
            target_date: Date to assess

        Returns:
            Probability of frost (0-1)
        """
        if not historical_seasons:
            return 0.5

        frost_count = 0
        for season in historical_seasons:
            if target_date < season.start_date or target_date > season.end_date:
                frost_count += 1

        return frost_count / len(historical_seasons)


class ExtremeEventDetector:
    """
    Extreme weather event detection and analysis.

    Identifies heat waves, cold spells, heavy precipitation,
    and drought events from historical data.

    Example:
        >>> detector = ExtremeEventDetector()
        >>> heat_waves = detector.detect_heat_waves(t_max, dates)
        >>> for event in heat_waves:
        ...     print(f"Heat wave: {event.start_date} to {event.end_date}")
    """

    def __init__(
        self,
        percentile_threshold: float = 95.0,
        min_duration: int = 3,
    ):
        """
        Initialize event detector.

        Args:
            percentile_threshold: Percentile for extreme classification
            min_duration: Minimum event duration (days)
        """
        self.percentile_threshold = percentile_threshold
        self.min_duration = min_duration

    def detect_heat_waves(
        self,
        t_max: np.ndarray,
        dates: np.ndarray = None,
    ) -> List[ExtremeEvent]:
        """
        Detect heat wave events.

        Args:
            t_max: Daily maximum temperatures
            dates: Corresponding dates

        Returns:
            List of heat wave events
        """
        threshold = np.percentile(t_max, self.percentile_threshold)

        return self._detect_events(
            values=t_max,
            dates=dates,
            threshold=threshold,
            condition='above',
            event_type='heat_wave',
        )

    def detect_cold_spells(
        self,
        t_min: np.ndarray,
        dates: np.ndarray = None,
    ) -> List[ExtremeEvent]:
        """
        Detect cold spell events.

        Args:
            t_min: Daily minimum temperatures
            dates: Corresponding dates

        Returns:
            List of cold spell events
        """
        threshold = np.percentile(t_min, 100 - self.percentile_threshold)

        return self._detect_events(
            values=t_min,
            dates=dates,
            threshold=threshold,
            condition='below',
            event_type='cold_spell',
        )

    def detect_heavy_precipitation(
        self,
        precipitation: np.ndarray,
        dates: np.ndarray = None,
    ) -> List[ExtremeEvent]:
        """
        Detect heavy precipitation events.

        Args:
            precipitation: Daily precipitation (mm)
            dates: Corresponding dates

        Returns:
            List of heavy precipitation events
        """
        # Filter out dry days for threshold calculation
        wet_days = precipitation[precipitation > 0.1]
        if len(wet_days) > 0:
            threshold = np.percentile(wet_days, self.percentile_threshold)
        else:
            threshold = 0.0

        return self._detect_events(
            values=precipitation,
            dates=dates,
            threshold=threshold,
            condition='above',
            event_type='heavy_precipitation',
            min_duration=1,  # Single day events count
        )

    def _detect_events(
        self,
        values: np.ndarray,
        dates: np.ndarray,
        threshold: float,
        condition: str,
        event_type: str,
        min_duration: int = None,
    ) -> List[ExtremeEvent]:
        """Detect events based on threshold exceedance."""
        if min_duration is None:
            min_duration = self.min_duration

        n_days = len(values)
        if dates is None:
            dates = np.array([date.today() - timedelta(days=n_days-i-1) for i in range(n_days)])

        # Find exceedances
        if condition == 'above':
            extreme_mask = values > threshold
        else:
            extreme_mask = values < threshold

        events = []
        in_event = False
        event_start = 0

        for i in range(n_days):
            if extreme_mask[i] and not in_event:
                in_event = True
                event_start = i
            elif not extreme_mask[i] and in_event:
                in_event = False
                duration = i - event_start

                if duration >= min_duration:
                    event_values = values[event_start:i]
                    peak = np.max(event_values) if condition == 'above' else np.min(event_values)

                    # Simple return period estimate
                    return_period = n_days / (365 * len(events) + 1)

                    events.append(ExtremeEvent(
                        event_type=event_type,
                        start_date=dates[event_start],
                        end_date=dates[i - 1],
                        severity=min(5, duration // min_duration),
                        peak_value=peak,
                        return_period=return_period,
                    ))

        return events

    def calculate_return_period(
        self,
        values: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Estimate return period for threshold exceedance.

        Args:
            values: Historical values
            threshold: Threshold value

        Returns:
            Estimated return period in years
        """
        n = len(values)
        exceedances = np.sum(values > threshold)

        if exceedances == 0:
            return float('inf')

        # Simple empirical return period
        probability = exceedances / n
        return 1 / (probability * 365) if probability > 0 else float('inf')
