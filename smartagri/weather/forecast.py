"""
Weather Forecasting Module

Provides weather forecasting capabilities for agricultural planning:
- Statistical forecasting methods
- Ensemble predictions with uncertainty
- Forecast validation and skill assessment
- Integration with external forecast services

Example:
    >>> forecaster = WeatherForecaster(historical_data)
    >>> forecast = forecaster.predict(days=7)
    >>> print(f"7-day forecast uncertainty: {forecast.uncertainty}")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Weather forecast result container.

    Attributes:
        dates: Forecast dates
        values: Predicted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence_level: Confidence level (0-1)
        method: Forecasting method used
    """
    dates: np.ndarray
    values: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float = 0.95
    method: str = "persistence"


@dataclass
class ForecastMetrics:
    """
    Forecast validation metrics.

    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Square Error
        bias: Systematic bias
        skill_score: Forecast skill score
        correlation: Correlation with observations
    """
    mae: float
    rmse: float
    bias: float
    skill_score: float
    correlation: float


class WeatherForecaster:
    """
    Statistical weather forecaster for agricultural applications.

    Implements multiple forecasting methods with uncertainty
    quantification for decision support.

    Example:
        >>> from datetime import date
        >>> historical = {'t_max': temps, 'dates': dates}
        >>> forecaster = WeatherForecaster(historical)
        >>> forecast = forecaster.forecast_temperature(days=7)
    """

    def __init__(
        self,
        historical_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize weather forecaster.

        Args:
            historical_data: Historical weather data for model fitting
        """
        self.historical_data = historical_data or {}
        self._models_fitted = False

    def fit(self, data: Dict[str, np.ndarray]) -> 'WeatherForecaster':
        """
        Fit forecasting models to historical data.

        Args:
            data: Historical weather data

        Returns:
            Fitted forecaster
        """
        self.historical_data = data
        self._models_fitted = True
        return self

    def forecast_temperature(
        self,
        days: int = 7,
        method: str = "climatology",
        confidence: float = 0.95,
    ) -> ForecastResult:
        """
        Forecast temperature for specified period.

        Args:
            days: Number of days to forecast
            method: Forecasting method ('persistence', 'climatology', 'regression')
            confidence: Confidence level for bounds

        Returns:
            ForecastResult with predictions and uncertainty

        Example:
            >>> forecast = forecaster.forecast_temperature(days=7)
            >>> print(f"Day 3: {forecast.values[2]:.1f}°C")
        """
        t_max = self.historical_data.get('t_max', np.array([]))
        t_min = self.historical_data.get('t_min', np.array([]))

        if len(t_max) == 0:
            # Return default forecast if no data
            forecast_dates = np.array([date.today() + timedelta(days=i) for i in range(1, days + 1)])
            values = np.full(days, 20.0)
            std = 5.0
        else:
            forecast_dates = np.array([date.today() + timedelta(days=i) for i in range(1, days + 1)])

            if method == "persistence":
                # Persistence: last observation continues
                values = np.full(days, t_max[-1])
                std = np.std(t_max)

            elif method == "climatology":
                # Climatological average for time of year
                values = np.full(days, np.mean(t_max))
                std = np.std(t_max)

            else:  # regression
                # Simple trend regression
                x = np.arange(len(t_max))
                if len(x) > 1:
                    slope, intercept = np.polyfit(x, t_max, 1)
                    future_x = np.arange(len(t_max), len(t_max) + days)
                    values = slope * future_x + intercept
                else:
                    values = np.full(days, t_max[-1] if len(t_max) > 0 else 20.0)
                std = np.std(t_max) if len(t_max) > 1 else 5.0

        # Calculate confidence bounds
        z_score = 1.96 if confidence == 0.95 else 2.58
        lower = values - z_score * std
        upper = values + z_score * std

        return ForecastResult(
            dates=forecast_dates,
            values=values,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            method=method,
        )

    def forecast_precipitation(
        self,
        days: int = 7,
    ) -> ForecastResult:
        """
        Forecast precipitation probability and amount.

        Args:
            days: Number of days to forecast

        Returns:
            ForecastResult with precipitation forecasts
        """
        precip = self.historical_data.get('precipitation', np.array([]))

        forecast_dates = np.array([date.today() + timedelta(days=i) for i in range(1, days + 1)])

        if len(precip) == 0:
            values = np.zeros(days)
            std = 5.0
        else:
            # Use historical average
            mean_precip = np.mean(precip)
            values = np.full(days, mean_precip)
            std = np.std(precip)

        return ForecastResult(
            dates=forecast_dates,
            values=values,
            lower_bound=np.zeros(days),
            upper_bound=values + 1.96 * std,
            confidence_level=0.95,
            method="climatology",
        )

    def predict(
        self,
        days: int = 7,
        variables: List[str] = None,
    ) -> Dict[str, ForecastResult]:
        """
        Generate multi-variable forecast.

        Args:
            days: Forecast horizon in days
            variables: Variables to forecast

        Returns:
            Dictionary of forecasts by variable
        """
        if variables is None:
            variables = ['t_max', 't_min', 'precipitation']

        forecasts = {}

        if 't_max' in variables or 't_min' in variables:
            temp_forecast = self.forecast_temperature(days)
            forecasts['t_max'] = temp_forecast

        if 'precipitation' in variables:
            forecasts['precipitation'] = self.forecast_precipitation(days)

        return forecasts


class EnsembleForecast:
    """
    Ensemble weather forecasting.

    Combines multiple forecasting methods to provide
    robust predictions with uncertainty estimates.

    Example:
        >>> ensemble = EnsembleForecast()
        >>> ensemble.add_member(forecaster1, weight=0.4)
        >>> ensemble.add_member(forecaster2, weight=0.6)
        >>> forecast = ensemble.predict(days=7)
    """

    def __init__(self):
        """Initialize ensemble forecaster."""
        self.members: List[Tuple[WeatherForecaster, float]] = []

    def add_member(
        self,
        forecaster: WeatherForecaster,
        weight: float = 1.0,
    ) -> None:
        """
        Add ensemble member.

        Args:
            forecaster: Forecaster instance
            weight: Member weight (importance)
        """
        self.members.append((forecaster, weight))

    def predict(
        self,
        days: int = 7,
        variable: str = 't_max',
    ) -> ForecastResult:
        """
        Generate ensemble forecast.

        Args:
            days: Forecast horizon
            variable: Variable to forecast

        Returns:
            Weighted ensemble forecast
        """
        if not self.members:
            raise ValueError("No ensemble members added")

        # Normalize weights
        total_weight = sum(w for _, w in self.members)
        normalized_weights = [w / total_weight for _, w in self.members]

        # Collect forecasts
        all_forecasts = []
        for (forecaster, _), weight in zip(self.members, normalized_weights):
            forecast = forecaster.forecast_temperature(days)
            all_forecasts.append((forecast.values, weight))

        # Weighted average
        ensemble_values = np.zeros(days)
        for values, weight in all_forecasts:
            ensemble_values += values * weight

        # Ensemble spread for uncertainty
        all_values = np.array([f[0] for f in all_forecasts])
        spread = np.std(all_values, axis=0)

        forecast_dates = np.array([date.today() + timedelta(days=i) for i in range(1, days + 1)])

        return ForecastResult(
            dates=forecast_dates,
            values=ensemble_values,
            lower_bound=ensemble_values - 1.96 * spread,
            upper_bound=ensemble_values + 1.96 * spread,
            confidence_level=0.95,
            method="ensemble",
        )

    def get_spread(self) -> np.ndarray:
        """Calculate ensemble spread (uncertainty measure)."""
        if not self.members:
            return np.array([])

        days = 7
        all_forecasts = []
        for forecaster, _ in self.members:
            forecast = forecaster.forecast_temperature(days)
            all_forecasts.append(forecast.values)

        return np.std(np.array(all_forecasts), axis=0)


class ForecastValidator:
    """
    Forecast validation and skill assessment.

    Evaluates forecast performance against observations
    using standard meteorological verification metrics.

    Example:
        >>> validator = ForecastValidator()
        >>> metrics = validator.validate(forecast, observations)
        >>> print(f"RMSE: {metrics.rmse:.2f}°C")
    """

    def __init__(self):
        """Initialize forecast validator."""
        self.validation_history: List[ForecastMetrics] = []

    def validate(
        self,
        forecast: np.ndarray,
        observations: np.ndarray,
        climatology: Optional[np.ndarray] = None,
    ) -> ForecastMetrics:
        """
        Validate forecast against observations.

        Args:
            forecast: Forecasted values
            observations: Observed values
            climatology: Climatological reference (for skill score)

        Returns:
            ForecastMetrics with validation results

        Example:
            >>> metrics = validator.validate(predicted, actual)
            >>> if metrics.skill_score > 0:
            ...     print("Forecast outperforms climatology")
        """
        forecast = np.atleast_1d(forecast)
        observations = np.atleast_1d(observations)

        # Ensure same length
        n = min(len(forecast), len(observations))
        forecast = forecast[:n]
        observations = observations[:n]

        # Calculate metrics
        errors = forecast - observations

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)

        # Correlation
        if np.std(forecast) > 0 and np.std(observations) > 0:
            correlation = np.corrcoef(forecast, observations)[0, 1]
        else:
            correlation = 0.0

        # Skill score (relative to climatology)
        if climatology is not None:
            clim_mse = np.mean((climatology[:n] - observations) ** 2)
            forecast_mse = np.mean(errors ** 2)
            skill_score = 1 - forecast_mse / clim_mse if clim_mse > 0 else 0
        else:
            skill_score = 0.0

        metrics = ForecastMetrics(
            mae=mae,
            rmse=rmse,
            bias=bias,
            skill_score=skill_score,
            correlation=correlation,
        )

        self.validation_history.append(metrics)

        return metrics

    def calculate_reliability(
        self,
        forecast_probs: np.ndarray,
        observations: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate forecast reliability (calibration).

        Args:
            forecast_probs: Forecasted probabilities
            observations: Binary observations (0/1)
            n_bins: Number of probability bins

        Returns:
            Reliability diagram data
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        observed_freq = np.zeros(n_bins)
        forecast_freq = np.zeros(n_bins)
        sample_size = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (forecast_probs >= bin_edges[i]) & (forecast_probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                observed_freq[i] = np.mean(observations[mask])
                forecast_freq[i] = np.mean(forecast_probs[mask])
                sample_size[i] = np.sum(mask)

        return {
            'bin_centers': bin_centers,
            'observed_freq': observed_freq,
            'forecast_freq': forecast_freq,
            'sample_size': sample_size,
        }

    def get_summary(self) -> Dict[str, float]:
        """Get summary of validation history."""
        if not self.validation_history:
            return {}

        return {
            'avg_mae': np.mean([m.mae for m in self.validation_history]),
            'avg_rmse': np.mean([m.rmse for m in self.validation_history]),
            'avg_skill': np.mean([m.skill_score for m in self.validation_history]),
            'n_validations': len(self.validation_history),
        }
