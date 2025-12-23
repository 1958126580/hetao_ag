"""
Utility Functions and Classes for SmartAgri

This module provides common utility functions used across the library:
- Data validation and preprocessing
- Unit conversions for agricultural measurements
- Date/time handling for agricultural calendars
- Configuration management
- Logging utilities
- Input/Output operations
"""

from smartagri.utils.validation import (
    validate_array,
    validate_positive,
    validate_range,
    validate_coordinates,
    validate_date_range,
    DataValidator,
)

from smartagri.utils.conversions import (
    UnitConverter,
    temperature_convert,
    area_convert,
    mass_convert,
    volume_convert,
    pressure_convert,
    yield_convert,
)

from smartagri.utils.datetime_utils import (
    AgriculturalCalendar,
    growing_degree_days,
    day_of_year,
    julian_day,
    solar_declination,
    daylight_hours,
)

from smartagri.utils.io import (
    load_csv,
    save_csv,
    load_geotiff,
    save_geotiff,
    load_shapefile,
    DataExporter,
)

from smartagri.utils.config import (
    Config,
    get_config,
    set_config,
)

__all__ = [
    # Validation
    "validate_array",
    "validate_positive",
    "validate_range",
    "validate_coordinates",
    "validate_date_range",
    "DataValidator",
    # Conversions
    "UnitConverter",
    "temperature_convert",
    "area_convert",
    "mass_convert",
    "volume_convert",
    "pressure_convert",
    "yield_convert",
    # Date/time
    "AgriculturalCalendar",
    "growing_degree_days",
    "day_of_year",
    "julian_day",
    "solar_declination",
    "daylight_hours",
    # I/O
    "load_csv",
    "save_csv",
    "load_geotiff",
    "save_geotiff",
    "load_shapefile",
    "DataExporter",
    # Configuration
    "Config",
    "get_config",
    "set_config",
]
