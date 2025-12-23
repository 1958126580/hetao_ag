"""
Unit Conversion Utilities for SmartAgri

Provides comprehensive unit conversion functions for agricultural
measurements, supporting both metric and imperial units.

Categories:
    - Temperature (Celsius, Fahrenheit, Kelvin)
    - Area (hectares, acres, square meters)
    - Mass (kg, pounds, tons, bushels)
    - Volume (liters, gallons, acre-feet)
    - Pressure (Pa, bar, psi, atm)
    - Yield (kg/ha, bushels/acre, tons/ha)
    - Flow rate (L/min, gal/min, m3/h)
    - Concentration (ppm, mg/L, mol/L)
"""

import numpy as np
from typing import Union, Dict, Optional
from dataclasses import dataclass
from enum import Enum

ArrayLike = Union[float, int, np.ndarray]


class TemperatureUnit(Enum):
    """Temperature unit types."""
    CELSIUS = "C"
    FAHRENHEIT = "F"
    KELVIN = "K"


class AreaUnit(Enum):
    """Area unit types."""
    HECTARE = "ha"
    ACRE = "ac"
    SQUARE_METER = "m2"
    SQUARE_KILOMETER = "km2"
    SQUARE_FOOT = "ft2"


class MassUnit(Enum):
    """Mass unit types."""
    KILOGRAM = "kg"
    GRAM = "g"
    METRIC_TON = "t"
    POUND = "lb"
    SHORT_TON = "ton_us"
    BUSHEL_CORN = "bu_corn"
    BUSHEL_WHEAT = "bu_wheat"
    BUSHEL_SOYBEAN = "bu_soy"


class VolumeUnit(Enum):
    """Volume unit types."""
    LITER = "L"
    MILLILITER = "mL"
    CUBIC_METER = "m3"
    GALLON_US = "gal"
    ACRE_FOOT = "ac_ft"
    CUBIC_FOOT = "ft3"


# Conversion factors to base units
_AREA_TO_M2 = {
    AreaUnit.HECTARE: 10000.0,
    AreaUnit.ACRE: 4046.8564224,
    AreaUnit.SQUARE_METER: 1.0,
    AreaUnit.SQUARE_KILOMETER: 1000000.0,
    AreaUnit.SQUARE_FOOT: 0.09290304,
}

_MASS_TO_KG = {
    MassUnit.KILOGRAM: 1.0,
    MassUnit.GRAM: 0.001,
    MassUnit.METRIC_TON: 1000.0,
    MassUnit.POUND: 0.45359237,
    MassUnit.SHORT_TON: 907.18474,
    MassUnit.BUSHEL_CORN: 25.4012,  # 56 lb
    MassUnit.BUSHEL_WHEAT: 27.2155,  # 60 lb
    MassUnit.BUSHEL_SOYBEAN: 27.2155,  # 60 lb
}

_VOLUME_TO_L = {
    VolumeUnit.LITER: 1.0,
    VolumeUnit.MILLILITER: 0.001,
    VolumeUnit.CUBIC_METER: 1000.0,
    VolumeUnit.GALLON_US: 3.785411784,
    VolumeUnit.ACRE_FOOT: 1233481.84,
    VolumeUnit.CUBIC_FOOT: 28.316846592,
}


def temperature_convert(
    value: ArrayLike,
    from_unit: Union[str, TemperatureUnit],
    to_unit: Union[str, TemperatureUnit],
) -> ArrayLike:
    """
    Convert temperature between units.

    Args:
        value: Temperature value(s)
        from_unit: Source unit ('C', 'F', 'K')
        to_unit: Target unit ('C', 'F', 'K')

    Returns:
        Converted temperature value(s)

    Example:
        >>> temp_f = temperature_convert(25, 'C', 'F')
        >>> print(f"25°C = {temp_f}°F")  # 77°F
    """
    if isinstance(from_unit, str):
        from_unit = TemperatureUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = TemperatureUnit(to_unit)

    value = np.asarray(value)

    # Convert to Celsius first
    if from_unit == TemperatureUnit.CELSIUS:
        celsius = value
    elif from_unit == TemperatureUnit.FAHRENHEIT:
        celsius = (value - 32) * 5 / 9
    elif from_unit == TemperatureUnit.KELVIN:
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_unit == TemperatureUnit.CELSIUS:
        result = celsius
    elif to_unit == TemperatureUnit.FAHRENHEIT:
        result = celsius * 9 / 5 + 32
    elif to_unit == TemperatureUnit.KELVIN:
        result = celsius + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")

    return float(result) if result.ndim == 0 else result


def area_convert(
    value: ArrayLike,
    from_unit: Union[str, AreaUnit],
    to_unit: Union[str, AreaUnit],
) -> ArrayLike:
    """
    Convert area between units.

    Args:
        value: Area value(s)
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted area value(s)

    Example:
        >>> acres = area_convert(100, 'ha', 'ac')
        >>> print(f"100 hectares = {acres:.2f} acres")  # ~247 acres
    """
    if isinstance(from_unit, str):
        from_unit = AreaUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = AreaUnit(to_unit)

    value = np.asarray(value)

    # Convert through base unit (m2)
    m2 = value * _AREA_TO_M2[from_unit]
    result = m2 / _AREA_TO_M2[to_unit]

    return float(result) if result.ndim == 0 else result


def mass_convert(
    value: ArrayLike,
    from_unit: Union[str, MassUnit],
    to_unit: Union[str, MassUnit],
) -> ArrayLike:
    """
    Convert mass between units.

    Supports agricultural-specific units like bushels.

    Args:
        value: Mass value(s)
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted mass value(s)

    Example:
        >>> kg = mass_convert(1000, 'bu_corn', 'kg')
        >>> print(f"1000 bushels corn = {kg:.0f} kg")
    """
    if isinstance(from_unit, str):
        from_unit = MassUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = MassUnit(to_unit)

    value = np.asarray(value)

    # Convert through base unit (kg)
    kg = value * _MASS_TO_KG[from_unit]
    result = kg / _MASS_TO_KG[to_unit]

    return float(result) if result.ndim == 0 else result


def volume_convert(
    value: ArrayLike,
    from_unit: Union[str, VolumeUnit],
    to_unit: Union[str, VolumeUnit],
) -> ArrayLike:
    """
    Convert volume between units.

    Args:
        value: Volume value(s)
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted volume value(s)

    Example:
        >>> gallons = volume_convert(1000, 'L', 'gal')
        >>> print(f"1000 liters = {gallons:.2f} gallons")
    """
    if isinstance(from_unit, str):
        from_unit = VolumeUnit(from_unit)
    if isinstance(to_unit, str):
        to_unit = VolumeUnit(to_unit)

    value = np.asarray(value)

    # Convert through base unit (L)
    liters = value * _VOLUME_TO_L[from_unit]
    result = liters / _VOLUME_TO_L[to_unit]

    return float(result) if result.ndim == 0 else result


def pressure_convert(
    value: ArrayLike,
    from_unit: str,
    to_unit: str,
) -> ArrayLike:
    """
    Convert pressure between units.

    Supported units: 'Pa', 'hPa', 'kPa', 'bar', 'atm', 'psi', 'mmHg'

    Args:
        value: Pressure value(s)
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted pressure value(s)

    Example:
        >>> psi = pressure_convert(101.325, 'kPa', 'psi')
        >>> print(f"101.325 kPa = {psi:.2f} psi")  # ~14.7 psi
    """
    # Conversion factors to Pa
    to_pa = {
        "Pa": 1.0,
        "hPa": 100.0,
        "kPa": 1000.0,
        "bar": 100000.0,
        "atm": 101325.0,
        "psi": 6894.757293168,
        "mmHg": 133.322387415,
    }

    value = np.asarray(value)

    if from_unit not in to_pa or to_unit not in to_pa:
        raise ValueError(f"Unknown pressure unit. Supported: {list(to_pa.keys())}")

    pa = value * to_pa[from_unit]
    result = pa / to_pa[to_unit]

    return float(result) if result.ndim == 0 else result


def yield_convert(
    value: ArrayLike,
    from_unit: str,
    to_unit: str,
    crop: str = "corn",
) -> ArrayLike:
    """
    Convert crop yield between units.

    Supported units:
        - 'kg/ha': kilograms per hectare
        - 't/ha': metric tons per hectare
        - 'bu/ac': bushels per acre
        - 'lb/ac': pounds per acre

    Args:
        value: Yield value(s)
        from_unit: Source unit
        to_unit: Target unit
        crop: Crop type for bushel conversion ('corn', 'wheat', 'soybean')

    Returns:
        Converted yield value(s)

    Example:
        >>> bu_ac = yield_convert(10, 't/ha', 'bu/ac', crop='corn')
        >>> print(f"10 t/ha = {bu_ac:.1f} bu/ac")
    """
    value = np.asarray(value)

    # Bushel weights in kg
    bushel_kg = {
        "corn": 25.4012,
        "wheat": 27.2155,
        "soybean": 27.2155,
        "barley": 21.772,
        "oats": 14.515,
    }

    # Conversion factors to kg/ha
    def to_kg_ha(val, unit):
        if unit == "kg/ha":
            return val
        elif unit == "t/ha":
            return val * 1000
        elif unit == "bu/ac":
            return val * bushel_kg.get(crop, 25.4) / 0.404686
        elif unit == "lb/ac":
            return val * 0.45359237 / 0.404686
        else:
            raise ValueError(f"Unknown yield unit: {unit}")

    def from_kg_ha(val, unit):
        if unit == "kg/ha":
            return val
        elif unit == "t/ha":
            return val / 1000
        elif unit == "bu/ac":
            return val * 0.404686 / bushel_kg.get(crop, 25.4)
        elif unit == "lb/ac":
            return val * 0.404686 / 0.45359237
        else:
            raise ValueError(f"Unknown yield unit: {unit}")

    kg_ha = to_kg_ha(value, from_unit)
    result = from_kg_ha(kg_ha, to_unit)

    return float(result) if result.ndim == 0 else result


class UnitConverter:
    """
    General-purpose unit converter for agricultural measurements.

    Provides a unified interface for all unit conversions with
    automatic unit detection and validation.

    Example:
        >>> converter = UnitConverter()
        >>> result = converter.convert(100, from_unit='ha', to_unit='ac')
        >>> print(f"100 ha = {result:.2f} acres")
    """

    def __init__(self):
        """Initialize unit converter with unit type mappings."""
        self._temperature_units = {"C", "F", "K"}
        self._area_units = {"ha", "ac", "m2", "km2", "ft2"}
        self._mass_units = {"kg", "g", "t", "lb", "ton_us", "bu_corn", "bu_wheat", "bu_soy"}
        self._volume_units = {"L", "mL", "m3", "gal", "ac_ft", "ft3"}
        self._pressure_units = {"Pa", "hPa", "kPa", "bar", "atm", "psi", "mmHg"}

    def convert(
        self,
        value: ArrayLike,
        from_unit: str,
        to_unit: str,
        **kwargs,
    ) -> ArrayLike:
        """
        Convert value between units.

        Automatically detects the unit category and applies
        the appropriate conversion.

        Args:
            value: Value(s) to convert
            from_unit: Source unit
            to_unit: Target unit
            **kwargs: Additional arguments for specific converters

        Returns:
            Converted value(s)
        """
        # Detect unit category
        if from_unit in self._temperature_units:
            return temperature_convert(value, from_unit, to_unit)
        elif from_unit in self._area_units:
            return area_convert(value, from_unit, to_unit)
        elif from_unit in self._mass_units:
            return mass_convert(value, from_unit, to_unit)
        elif from_unit in self._volume_units:
            return volume_convert(value, from_unit, to_unit)
        elif from_unit in self._pressure_units:
            return pressure_convert(value, from_unit, to_unit)
        else:
            raise ValueError(f"Unknown unit: {from_unit}")

    def get_supported_units(self, category: str) -> set:
        """
        Get supported units for a category.

        Args:
            category: Unit category ('temperature', 'area', 'mass', 'volume', 'pressure')

        Returns:
            Set of supported unit strings
        """
        categories = {
            "temperature": self._temperature_units,
            "area": self._area_units,
            "mass": self._mass_units,
            "volume": self._volume_units,
            "pressure": self._pressure_units,
        }
        return categories.get(category, set())

    @staticmethod
    def soil_moisture_convert(
        value: ArrayLike,
        from_unit: str,
        to_unit: str,
        bulk_density: float = 1.3,
    ) -> ArrayLike:
        """
        Convert soil moisture between units.

        Supported units:
            - 'gravimetric': g water / g dry soil
            - 'volumetric': m3 water / m3 soil
            - 'percent_vwc': volumetric water content %

        Args:
            value: Moisture value(s)
            from_unit: Source unit
            to_unit: Target unit
            bulk_density: Soil bulk density (g/cm3)

        Returns:
            Converted moisture value(s)
        """
        value = np.asarray(value)

        # Convert to volumetric first
        if from_unit == "gravimetric":
            volumetric = value * bulk_density
        elif from_unit == "volumetric":
            volumetric = value
        elif from_unit == "percent_vwc":
            volumetric = value / 100
        else:
            raise ValueError(f"Unknown soil moisture unit: {from_unit}")

        # Convert from volumetric to target
        if to_unit == "gravimetric":
            result = volumetric / bulk_density
        elif to_unit == "volumetric":
            result = volumetric
        elif to_unit == "percent_vwc":
            result = volumetric * 100
        else:
            raise ValueError(f"Unknown soil moisture unit: {to_unit}")

        return float(result) if result.ndim == 0 else result

    @staticmethod
    def nutrient_concentration_convert(
        value: ArrayLike,
        from_unit: str,
        to_unit: str,
        molecular_weight: float = 14.0,
    ) -> ArrayLike:
        """
        Convert nutrient concentration between units.

        Supported units: 'ppm', 'mg/L', 'mg/kg', 'mmol/L', 'kg/ha'

        Args:
            value: Concentration value(s)
            from_unit: Source unit
            to_unit: Target unit
            molecular_weight: Element molecular weight (g/mol)

        Returns:
            Converted concentration value(s)
        """
        value = np.asarray(value)

        # Conversion to mg/L
        if from_unit in ["ppm", "mg/L", "mg/kg"]:
            mg_l = value
        elif from_unit == "mmol/L":
            mg_l = value * molecular_weight
        elif from_unit == "kg/ha":
            # Assuming 1 ha at 20cm depth = 2000000 L of soil
            mg_l = value * 1000 / 2000
        else:
            raise ValueError(f"Unknown concentration unit: {from_unit}")

        # Conversion from mg/L to target
        if to_unit in ["ppm", "mg/L", "mg/kg"]:
            result = mg_l
        elif to_unit == "mmol/L":
            result = mg_l / molecular_weight
        elif to_unit == "kg/ha":
            result = mg_l * 2000 / 1000
        else:
            raise ValueError(f"Unknown concentration unit: {to_unit}")

        return float(result) if result.ndim == 0 else result
