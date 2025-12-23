"""
Evapotranspiration Calculation Module

Implements standard methods for calculating reference and
crop evapotranspiration for irrigation management.

Methods:
    - FAO-56 Penman-Monteith (reference method)
    - Hargreaves-Samani (temperature only)
    - Priestley-Taylor (radiation-based)

Example:
    >>> et = ETCalculator(latitude=40.0, elevation=100)
    >>> eto = et.penman_monteith(weather_data)
    >>> etc = et.crop_et(eto, kc=1.15)
"""

import numpy as np
from typing import Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ETResult:
    """
    Evapotranspiration calculation results.

    Attributes:
        eto: Reference evapotranspiration (mm/day)
        etc: Crop evapotranspiration (mm/day)
        components: ET components breakdown
    """
    eto: np.ndarray
    etc: Optional[np.ndarray] = None
    components: Optional[Dict[str, np.ndarray]] = None


class PenmanMonteith:
    """
    FAO-56 Penman-Monteith reference ET calculation.

    The standard method for calculating reference evapotranspiration
    from a hypothetical grass reference surface.

    Example:
        >>> pm = PenmanMonteith(latitude=40.0, elevation=100)
        >>> eto = pm.calculate(
        ...     t_max=30, t_min=18, humidity=65,
        ...     wind_speed=2.0, solar_radiation=22, doy=180
        ... )
    """

    def __init__(
        self,
        latitude: float,
        elevation: float = 0.0,
    ):
        """
        Initialize Penman-Monteith calculator.

        Args:
            latitude: Site latitude (degrees)
            elevation: Elevation above sea level (m)
        """
        self.latitude = latitude
        self.elevation = elevation

        # Psychrometric constant
        self.pressure = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
        self.gamma = 0.665e-3 * self.pressure

    def calculate(
        self,
        t_max: Union[float, np.ndarray],
        t_min: Union[float, np.ndarray],
        humidity: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        solar_radiation: Union[float, np.ndarray],
        doy: Union[int, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate reference ET using FAO-56 Penman-Monteith.

        Args:
            t_max: Maximum temperature (°C)
            t_min: Minimum temperature (°C)
            humidity: Mean relative humidity (%)
            wind_speed: Wind speed at 2m (m/s)
            solar_radiation: Solar radiation (MJ/m2/day)
            doy: Day of year

        Returns:
            Reference ET (mm/day)
        """
        t_max = np.atleast_1d(t_max)
        t_min = np.atleast_1d(t_min)
        humidity = np.atleast_1d(humidity)
        wind_speed = np.atleast_1d(wind_speed)
        solar_radiation = np.atleast_1d(solar_radiation)
        doy = np.atleast_1d(doy)

        # Mean temperature
        t_mean = (t_max + t_min) / 2

        # Saturation vapor pressure
        es_max = 0.6108 * np.exp(17.27 * t_max / (t_max + 237.3))
        es_min = 0.6108 * np.exp(17.27 * t_min / (t_min + 237.3))
        es = (es_max + es_min) / 2

        # Actual vapor pressure
        ea = es * humidity / 100

        # Slope of saturation vapor pressure curve
        delta = 4098 * es / (t_mean + 237.3) ** 2

        # Net radiation
        rn = self._net_radiation(
            solar_radiation, t_max, t_min, ea, doy
        )

        # Soil heat flux (assumed zero for daily calculations)
        g = 0

        # FAO-56 Penman-Monteith equation
        numerator = (
            0.408 * delta * (rn - g) +
            self.gamma * 900 / (t_mean + 273) * wind_speed * (es - ea)
        )
        denominator = delta + self.gamma * (1 + 0.34 * wind_speed)

        eto = numerator / denominator

        return np.maximum(eto, 0)

    def _net_radiation(
        self,
        rs: np.ndarray,
        t_max: np.ndarray,
        t_min: np.ndarray,
        ea: np.ndarray,
        doy: np.ndarray,
    ) -> np.ndarray:
        """Calculate net radiation (MJ/m2/day)."""
        lat_rad = np.radians(self.latitude)

        # Extraterrestrial radiation
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta_solar = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta_solar))

        ra = 24 * 60 / np.pi * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta_solar) +
            np.cos(lat_rad) * np.cos(delta_solar) * np.sin(ws)
        )

        # Clear-sky radiation
        rso = (0.75 + 2e-5 * self.elevation) * ra

        # Net shortwave radiation
        albedo = 0.23
        rns = (1 - albedo) * rs

        # Net longwave radiation
        sigma = 4.903e-9  # Stefan-Boltzmann constant
        tk_max = t_max + 273.16
        tk_min = t_min + 273.16

        rnl = sigma * (tk_max ** 4 + tk_min ** 4) / 2 * (
            0.34 - 0.14 * np.sqrt(ea)
        ) * (1.35 * np.minimum(rs / rso, 1.0) - 0.35)

        return rns - rnl


class HargreavesSamani:
    """
    Hargreaves-Samani ET estimation.

    Temperature-based method suitable when only temperature
    data is available.

    Example:
        >>> hs = HargreavesSamani(latitude=40.0)
        >>> eto = hs.calculate(t_max=30, t_min=18, doy=180)
    """

    def __init__(self, latitude: float):
        """
        Initialize Hargreaves-Samani calculator.

        Args:
            latitude: Site latitude (degrees)
        """
        self.latitude = latitude

    def calculate(
        self,
        t_max: Union[float, np.ndarray],
        t_min: Union[float, np.ndarray],
        doy: Union[int, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate reference ET using Hargreaves-Samani.

        Args:
            t_max: Maximum temperature (°C)
            t_min: Minimum temperature (°C)
            doy: Day of year

        Returns:
            Reference ET (mm/day)
        """
        t_max = np.atleast_1d(t_max)
        t_min = np.atleast_1d(t_min)
        doy = np.atleast_1d(doy)

        t_mean = (t_max + t_min) / 2

        # Extraterrestrial radiation
        ra = self._extraterrestrial_radiation(doy)

        # Convert MJ/m2/day to mm/day equivalent
        ra_mm = ra * 0.408

        # Hargreaves-Samani equation
        eto = 0.0023 * ra_mm * (t_mean + 17.8) * np.sqrt(np.maximum(t_max - t_min, 0))

        return np.maximum(eto, 0)

    def _extraterrestrial_radiation(self, doy: np.ndarray) -> np.ndarray:
        """Calculate extraterrestrial radiation."""
        lat_rad = np.radians(self.latitude)

        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)

        ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))

        ra = 24 * 60 / np.pi * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        return ra


class PriestleyTaylor:
    """
    Priestley-Taylor ET estimation.

    Radiation-based method with empirical coefficient.
    """

    def __init__(self, alpha: float = 1.26):
        """
        Initialize Priestley-Taylor calculator.

        Args:
            alpha: Priestley-Taylor coefficient (default 1.26)
        """
        self.alpha = alpha

    def calculate(
        self,
        net_radiation: Union[float, np.ndarray],
        temperature: Union[float, np.ndarray],
        ground_heat_flux: float = 0,
    ) -> np.ndarray:
        """
        Calculate ET using Priestley-Taylor method.

        Args:
            net_radiation: Net radiation (MJ/m2/day)
            temperature: Mean temperature (°C)
            ground_heat_flux: Ground heat flux (MJ/m2/day)

        Returns:
            Reference ET (mm/day)
        """
        net_radiation = np.atleast_1d(net_radiation)
        temperature = np.atleast_1d(temperature)

        # Psychrometric constant
        gamma = 0.066

        # Slope of saturation vapor pressure curve
        es = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
        delta = 4098 * es / (temperature + 237.3) ** 2

        # Priestley-Taylor equation
        eto = self.alpha * delta / (delta + gamma) * (net_radiation - ground_heat_flux) * 0.408

        return np.maximum(eto, 0)


class ETCalculator:
    """
    Unified ET calculation interface.

    Automatically selects appropriate method based on
    available data and calculates crop ET.

    Example:
        >>> et = ETCalculator(latitude=40.0, elevation=100)
        >>> eto = et.calculate(weather_data)
        >>> etc = et.crop_et(eto, kc_series)
    """

    def __init__(
        self,
        latitude: float,
        elevation: float = 0.0,
        default_method: str = "penman_monteith",
    ):
        """
        Initialize ET calculator.

        Args:
            latitude: Site latitude (degrees)
            elevation: Elevation (m)
            default_method: Default calculation method
        """
        self.latitude = latitude
        self.elevation = elevation
        self.default_method = default_method

        # Initialize method calculators
        self.pm = PenmanMonteith(latitude, elevation)
        self.hs = HargreavesSamani(latitude)
        self.pt = PriestleyTaylor()

    def penman_monteith(
        self,
        weather_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate ET using Penman-Monteith.

        Args:
            weather_data: Dict with t_max, t_min, humidity,
                         wind_speed, solar_radiation, doy

        Returns:
            Reference ET (mm/day)
        """
        return self.pm.calculate(
            t_max=weather_data["t_max"],
            t_min=weather_data["t_min"],
            humidity=weather_data.get("humidity", 70),
            wind_speed=weather_data.get("wind_speed", 2),
            solar_radiation=weather_data.get("solar_radiation", 20),
            doy=weather_data.get("doy", 180),
        )

    def hargreaves(
        self,
        weather_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate ET using Hargreaves-Samani.

        Args:
            weather_data: Dict with t_max, t_min, doy

        Returns:
            Reference ET (mm/day)
        """
        return self.hs.calculate(
            t_max=weather_data["t_max"],
            t_min=weather_data["t_min"],
            doy=weather_data.get("doy", 180),
        )

    def calculate(
        self,
        weather_data: Dict[str, np.ndarray],
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Calculate reference ET with automatic method selection.

        Args:
            weather_data: Weather data dict
            method: Specific method to use

        Returns:
            Reference ET (mm/day)
        """
        method = method or self.default_method

        if method == "penman_monteith":
            # Check if we have all required data
            required = ["humidity", "wind_speed", "solar_radiation"]
            has_all = all(k in weather_data for k in required)

            if has_all:
                return self.penman_monteith(weather_data)
            else:
                logger.warning("Falling back to Hargreaves due to missing data")
                return self.hargreaves(weather_data)

        elif method == "hargreaves":
            return self.hargreaves(weather_data)

        else:
            raise ValueError(f"Unknown method: {method}")

    def crop_et(
        self,
        eto: np.ndarray,
        kc: Union[float, np.ndarray],
        ks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate crop evapotranspiration.

        Args:
            eto: Reference ET (mm/day)
            kc: Crop coefficient (single value or time series)
            ks: Water stress coefficient (0-1, optional)

        Returns:
            Crop ET (mm/day)
        """
        eto = np.atleast_1d(eto)
        kc = np.atleast_1d(kc)

        if len(kc) == 1:
            kc = np.full_like(eto, kc[0])

        etc = eto * kc

        if ks is not None:
            etc = etc * ks

        return etc

    def dual_crop_coefficient(
        self,
        eto: np.ndarray,
        kcb: np.ndarray,
        ke: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate ET using dual crop coefficient method.

        Separates transpiration (Kcb) and evaporation (Ke).

        Args:
            eto: Reference ET
            kcb: Basal crop coefficient
            ke: Soil evaporation coefficient

        Returns:
            Crop ET (mm/day)
        """
        return eto * (kcb + ke)


def calculate_eto(
    t_max: np.ndarray,
    t_min: np.ndarray,
    latitude: float,
    doy: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Quick ETo calculation function.

    Args:
        t_max: Maximum temperature (°C)
        t_min: Minimum temperature (°C)
        latitude: Site latitude
        doy: Day of year
        **kwargs: Additional weather variables

    Returns:
        Reference ET (mm/day)
    """
    et = ETCalculator(latitude=latitude)

    weather = {
        "t_max": t_max,
        "t_min": t_min,
        "doy": doy,
        **kwargs,
    }

    return et.calculate(weather)
