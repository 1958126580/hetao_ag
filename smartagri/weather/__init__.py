"""
Weather Data Processing Module for SmartAgri

Comprehensive weather data handling including:
- Data acquisition and processing
- Evapotranspiration calculation
- Weather forecasting
- Climate analysis
- Extreme event detection

Example:
    >>> from smartagri.weather import WeatherProcessor, ETCalculator
    >>> processor = WeatherProcessor()
    >>> et = ETCalculator(latitude=40.0)
    >>> eto = et.penman_monteith(weather_data)
"""

from smartagri.weather.processing import (
    WeatherProcessor,
    WeatherData,
    WeatherStation,
    QualityControl,
)

from smartagri.weather.evapotranspiration import (
    ETCalculator,
    PenmanMonteith,
    HargreavesSamani,
    PriestleyTaylor,
    calculate_eto,
)

from smartagri.weather.forecast import (
    WeatherForecaster,
    EnsembleForecast,
    ForecastValidator,
)

from smartagri.weather.climate import (
    ClimateAnalyzer,
    DroughtIndex,
    GrowingSeasonAnalysis,
    ExtremeEventDetector,
)

__all__ = [
    "WeatherProcessor",
    "WeatherData",
    "WeatherStation",
    "QualityControl",
    "ETCalculator",
    "PenmanMonteith",
    "HargreavesSamani",
    "PriestleyTaylor",
    "calculate_eto",
    "WeatherForecaster",
    "EnsembleForecast",
    "ForecastValidator",
    "ClimateAnalyzer",
    "DroughtIndex",
    "GrowingSeasonAnalysis",
    "ExtremeEventDetector",
]
