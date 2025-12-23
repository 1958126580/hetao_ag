#!/usr/bin/env python3
"""
Crop Yield Analysis and Prediction Example

This example demonstrates comprehensive crop yield analysis using
the smartagri library, including:

- Crop growth simulation with phenological stages
- Yield prediction using machine learning
- Weather data integration
- Soil analysis and recommendations
- Growing Degree Day (GDD) calculations
- Seasonal forecasting

Usage:
    python examples/crop_yield_analysis.py

Author: SmartAgri Development Team
License: MIT
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_weather_data(
    start_date: date,
    n_days: int,
    base_temp: float = 15.0,
    temp_amplitude: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic weather data for demonstration.

    Creates realistic daily weather patterns including temperature,
    precipitation, humidity, wind speed, and solar radiation.

    Args:
        start_date: Starting date for weather simulation
        n_days: Number of days to simulate
        base_temp: Average temperature baseline (°C)
        temp_amplitude: Temperature variation amplitude (°C)

    Returns:
        Dictionary containing weather variables as numpy arrays

    Example:
        >>> weather = simulate_weather_data(date(2024, 4, 1), 180)
        >>> print(f"Max temp range: {weather['t_max'].min():.1f} - {weather['t_max'].max():.1f}°C")
    """
    np.random.seed(42)  # Reproducibility

    # Day of year for seasonal patterns
    doy = np.array([
        (start_date + timedelta(days=i)).timetuple().tm_yday
        for i in range(n_days)
    ])

    # Seasonal temperature pattern
    seasonal = base_temp + temp_amplitude * np.sin(2 * np.pi * (doy - 80) / 365)

    # Daily temperature variations
    daily_noise = np.random.normal(0, 2, n_days)
    t_mean = seasonal + daily_noise
    t_max = t_mean + np.random.uniform(4, 8, n_days)
    t_min = t_mean - np.random.uniform(4, 8, n_days)

    # Precipitation (stochastic with seasonal pattern)
    precip_prob = 0.2 + 0.1 * np.sin(2 * np.pi * (doy - 100) / 365)
    precip_events = np.random.random(n_days) < precip_prob
    precipitation = np.zeros(n_days)
    precipitation[precip_events] = np.random.exponential(8, precip_events.sum())

    # Relative humidity (higher with precipitation)
    humidity = 50 + 20 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 10, n_days)
    humidity = np.clip(humidity + precipitation * 2, 30, 100)

    # Wind speed
    wind_speed = 2 + np.random.exponential(1.5, n_days)

    # Solar radiation (seasonal)
    max_radiation = 25 + 10 * np.sin(2 * np.pi * (doy - 172) / 365)
    cloud_factor = 1 - 0.5 * precip_events
    solar_radiation = max_radiation * cloud_factor * np.random.uniform(0.8, 1.0, n_days)

    return {
        'dates': np.array([start_date + timedelta(days=i) for i in range(n_days)]),
        't_max': t_max,
        't_min': t_min,
        't_mean': t_mean,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'solar_radiation': solar_radiation,
        'doy': doy,
    }


def calculate_growing_degree_days(
    t_max: np.ndarray,
    t_min: np.ndarray,
    base_temp: float = 10.0,
    upper_temp: float = 30.0,
) -> np.ndarray:
    """
    Calculate Growing Degree Days (GDD) accumulation.

    GDD is a measure of heat accumulation used to predict plant
    development stages. The calculation clips temperatures to
    biologically relevant bounds.

    Args:
        t_max: Maximum daily temperatures (°C)
        t_min: Minimum daily temperatures (°C)
        base_temp: Base temperature below which no growth occurs
        upper_temp: Upper temperature threshold for GDD calculation

    Returns:
        Cumulative GDD array

    Example:
        >>> gdd = calculate_growing_degree_days(t_max, t_min, base_temp=10)
        >>> print(f"Total GDD accumulated: {gdd[-1]:.1f}")
    """
    # Clip temperatures to bounds
    t_max_clipped = np.clip(t_max, base_temp, upper_temp)
    t_min_clipped = np.clip(t_min, base_temp, upper_temp)

    # Calculate daily GDD using averaging method
    daily_gdd = (t_max_clipped + t_min_clipped) / 2 - base_temp
    daily_gdd = np.maximum(daily_gdd, 0)

    # Cumulative GDD
    cumulative_gdd = np.cumsum(daily_gdd)

    return cumulative_gdd


def determine_crop_stage(
    gdd: float,
    crop_type: str = "corn",
) -> Tuple[str, float]:
    """
    Determine crop growth stage based on accumulated GDD.

    Uses crop-specific GDD thresholds to identify current
    phenological stage and progress within that stage.

    Args:
        gdd: Accumulated Growing Degree Days
        crop_type: Crop type ('corn', 'wheat', 'soybean')

    Returns:
        Tuple of (stage_name, progress_percentage)

    Example:
        >>> stage, progress = determine_crop_stage(850, 'corn')
        >>> print(f"Current stage: {stage} ({progress:.0f}% complete)")
    """
    # Crop-specific GDD requirements (base 10°C)
    crop_stages = {
        'corn': [
            (0, 'planting', 0),
            (120, 'emergence', 120),
            (340, 'V6', 220),
            (630, 'tasseling', 290),
            (870, 'silking', 240),
            (1150, 'grain_fill', 280),
            (1400, 'maturity', 250),
        ],
        'wheat': [
            (0, 'planting', 0),
            (100, 'emergence', 100),
            (300, 'tillering', 200),
            (600, 'jointing', 300),
            (800, 'heading', 200),
            (1100, 'grain_fill', 300),
            (1300, 'maturity', 200),
        ],
        'soybean': [
            (0, 'planting', 0),
            (130, 'emergence', 130),
            (380, 'flowering', 250),
            (780, 'pod_development', 400),
            (1100, 'seed_fill', 320),
            (1450, 'maturity', 350),
        ],
    }

    stages = crop_stages.get(crop_type, crop_stages['corn'])

    current_stage = 'planting'
    progress = 0.0

    for i, (threshold, stage_name, _) in enumerate(stages):
        if gdd >= threshold:
            current_stage = stage_name
            # Calculate progress to next stage
            if i < len(stages) - 1:
                next_threshold = stages[i + 1][0]
                progress = (gdd - threshold) / (next_threshold - threshold) * 100
            else:
                progress = 100.0

    return current_stage, min(progress, 100.0)


def estimate_yield_potential(
    cumulative_gdd: np.ndarray,
    precipitation: np.ndarray,
    solar_radiation: np.ndarray,
    crop_type: str = "corn",
    planting_density: float = 80000,  # plants/ha
) -> Dict[str, float]:
    """
    Estimate crop yield potential based on environmental factors.

    Uses a simplified radiation use efficiency (RUE) model combined
    with water stress factors to estimate potential yield.

    Args:
        cumulative_gdd: Accumulated GDD array
        precipitation: Daily precipitation (mm)
        solar_radiation: Daily solar radiation (MJ/m²/day)
        crop_type: Crop type
        planting_density: Planting density (plants/ha)

    Returns:
        Dictionary with yield estimates and components

    Example:
        >>> yield_data = estimate_yield_potential(gdd, precip, solar, 'corn')
        >>> print(f"Estimated yield: {yield_data['yield_potential']:.2f} t/ha")
    """
    # Crop-specific parameters
    crop_params = {
        'corn': {
            'rue': 3.8,           # g/MJ radiation use efficiency
            'hi': 0.50,           # harvest index
            'max_yield': 15.0,    # maximum yield (t/ha)
            'water_req': 550,     # water requirement (mm)
        },
        'wheat': {
            'rue': 2.8,
            'hi': 0.45,
            'max_yield': 10.0,
            'water_req': 450,
        },
        'soybean': {
            'rue': 2.0,
            'hi': 0.40,
            'max_yield': 5.0,
            'water_req': 500,
        },
    }

    params = crop_params.get(crop_type, crop_params['corn'])

    # Calculate intercepted radiation (simplified)
    # Assumes LAI development with GDD
    final_gdd = cumulative_gdd[-1]
    lai_max = 5.0

    # LAI development curve
    lai_development = lai_max * (1 - np.exp(-0.003 * cumulative_gdd))

    # Fraction of radiation intercepted (Beer's law)
    k = 0.65  # extinction coefficient
    f_intercept = 1 - np.exp(-k * lai_development)

    # Intercepted PAR (assume 50% of solar is PAR)
    par_intercepted = 0.5 * solar_radiation * f_intercept

    # Biomass accumulation
    daily_biomass = params['rue'] * par_intercepted  # g/m²/day
    total_biomass = np.sum(daily_biomass) / 100  # convert to t/ha

    # Water stress factor
    total_precip = np.sum(precipitation)
    water_stress = min(1.0, total_precip / params['water_req'])

    # Temperature stress (based on GDD accumulation)
    optimal_gdd = 1400  # for maturity
    temp_factor = min(1.0, final_gdd / optimal_gdd)

    # Yield calculation
    yield_potential = total_biomass * params['hi'] * water_stress * temp_factor
    yield_potential = min(yield_potential, params['max_yield'])

    # Yield components
    if crop_type == 'corn':
        ears_per_ha = planting_density * 0.95
        kernels_per_ear = 500 * water_stress * temp_factor
        kernel_weight = 0.35  # g
        yield_from_components = (ears_per_ha * kernels_per_ear * kernel_weight) / 1e6
    else:
        yield_from_components = yield_potential

    return {
        'yield_potential': yield_potential,
        'yield_from_components': yield_from_components,
        'total_biomass': total_biomass,
        'harvest_index': params['hi'],
        'water_stress_factor': water_stress,
        'temperature_factor': temp_factor,
        'total_precipitation': total_precip,
        'cumulative_gdd': final_gdd,
        'lai_max_achieved': np.max(lai_development),
        'par_intercepted_total': np.sum(par_intercepted),
    }


def generate_yield_recommendations(
    yield_analysis: Dict[str, float],
    soil_data: Dict[str, float] = None,
) -> List[str]:
    """
    Generate agronomic recommendations based on yield analysis.

    Analyzes yield limiting factors and provides actionable
    recommendations for optimization.

    Args:
        yield_analysis: Yield analysis results from estimate_yield_potential
        soil_data: Optional soil analysis data

    Returns:
        List of recommendation strings

    Example:
        >>> recommendations = generate_yield_recommendations(yield_data)
        >>> for rec in recommendations:
        ...     print(f"• {rec}")
    """
    recommendations = []

    # Water stress recommendations
    if yield_analysis['water_stress_factor'] < 0.8:
        recommendations.append(
            f"Water stress detected (factor: {yield_analysis['water_stress_factor']:.2f}). "
            "Consider implementing supplemental irrigation or drought-tolerant varieties."
        )

    # Temperature/GDD recommendations
    if yield_analysis['temperature_factor'] < 0.9:
        recommendations.append(
            f"Suboptimal heat accumulation ({yield_analysis['cumulative_gdd']:.0f} GDD). "
            "Consider earlier planting or shorter-season hybrid for your region."
        )

    # LAI optimization
    if yield_analysis['lai_max_achieved'] < 4.5:
        recommendations.append(
            f"Canopy development limited (LAI max: {yield_analysis['lai_max_achieved']:.1f}). "
            "Evaluate nitrogen availability and planting density."
        )

    # Yield gap analysis
    yield_gap = yield_analysis.get('yield_from_components', 0) - yield_analysis['yield_potential']
    if abs(yield_gap) > 0.5:
        recommendations.append(
            f"Yield component analysis suggests {abs(yield_gap):.1f} t/ha potential gap. "
            "Review kernel number per ear and kernel weight optimization strategies."
        )

    # Soil-based recommendations
    if soil_data:
        if soil_data.get('ph', 7.0) < 6.0:
            recommendations.append(
                f"Soil pH ({soil_data['ph']:.1f}) below optimal. "
                "Apply agricultural lime to improve nutrient availability."
            )
        if soil_data.get('organic_matter', 3.0) < 2.5:
            recommendations.append(
                f"Low organic matter ({soil_data['organic_matter']:.1f}%). "
                "Incorporate cover crops or organic amendments."
            )

    # General optimization
    if yield_analysis['yield_potential'] > 12:
        recommendations.append(
            "High yield potential zone. Ensure adequate nitrogen (200-250 kg N/ha) "
            "and micronutrient availability for maximum expression."
        )

    if not recommendations:
        recommendations.append(
            "Growing conditions appear favorable. Maintain current management "
            "practices and scout regularly for pest and disease pressure."
        )

    return recommendations


def run_seasonal_simulation(
    crop_type: str = "corn",
    planting_date: date = None,
    location_lat: float = 42.0,
) -> Dict:
    """
    Run complete seasonal crop simulation.

    Simulates an entire growing season from planting to harvest,
    tracking crop development, yield accumulation, and providing
    management insights.

    Args:
        crop_type: Type of crop to simulate
        planting_date: Planting date (default: April 15 of current year)
        location_lat: Location latitude for weather simulation

    Returns:
        Complete simulation results dictionary

    Example:
        >>> results = run_seasonal_simulation('corn', date(2024, 4, 20))
        >>> print(f"Final yield: {results['yield']['yield_potential']:.2f} t/ha")
    """
    if planting_date is None:
        planting_date = date(2024, 4, 15)

    logger.info(f"Starting {crop_type} simulation from {planting_date}")

    # Simulate 180-day growing season
    n_days = 180
    weather = simulate_weather_data(planting_date, n_days)

    # Calculate GDD accumulation
    base_temps = {'corn': 10.0, 'wheat': 0.0, 'soybean': 10.0}
    base_temp = base_temps.get(crop_type, 10.0)

    cumulative_gdd = calculate_growing_degree_days(
        weather['t_max'],
        weather['t_min'],
        base_temp=base_temp,
    )

    # Track stage progression
    stage_history = []
    for i, gdd in enumerate(cumulative_gdd):
        stage, progress = determine_crop_stage(gdd, crop_type)
        stage_history.append({
            'day': i,
            'date': weather['dates'][i],
            'gdd': gdd,
            'stage': stage,
            'progress': progress,
        })

    # Estimate yield
    yield_analysis = estimate_yield_potential(
        cumulative_gdd,
        weather['precipitation'],
        weather['solar_radiation'],
        crop_type=crop_type,
    )

    # Generate recommendations
    recommendations = generate_yield_recommendations(yield_analysis)

    # Compile results
    results = {
        'crop_type': crop_type,
        'planting_date': planting_date,
        'harvest_date': planting_date + timedelta(days=n_days),
        'weather_summary': {
            'avg_temp': float(np.mean(weather['t_mean'])),
            'max_temp': float(np.max(weather['t_max'])),
            'min_temp': float(np.min(weather['t_min'])),
            'total_precip': float(np.sum(weather['precipitation'])),
            'avg_radiation': float(np.mean(weather['solar_radiation'])),
        },
        'gdd_accumulation': {
            'total': float(cumulative_gdd[-1]),
            'daily_avg': float(cumulative_gdd[-1] / n_days),
            'base_temp': base_temp,
        },
        'final_stage': stage_history[-1]['stage'],
        'yield': yield_analysis,
        'recommendations': recommendations,
    }

    return results


def main():
    """
    Main demonstration function.

    Runs comprehensive crop yield analysis example showing
    all library capabilities.
    """
    print("=" * 70)
    print("Smart Agriculture - Crop Yield Analysis Example")
    print("=" * 70)
    print()

    # Run simulations for different crops
    crops = ['corn', 'wheat', 'soybean']

    for crop in crops:
        print(f"\n{'─' * 50}")
        print(f"Simulating {crop.upper()} growing season...")
        print(f"{'─' * 50}")

        results = run_seasonal_simulation(
            crop_type=crop,
            planting_date=date(2024, 4, 15) if crop != 'wheat' else date(2024, 3, 1),
        )

        # Print results
        print(f"\nPlanting: {results['planting_date']}")
        print(f"Harvest:  {results['harvest_date']}")

        print(f"\nWeather Summary:")
        ws = results['weather_summary']
        print(f"  • Average Temperature: {ws['avg_temp']:.1f}°C")
        print(f"  • Temperature Range:   {ws['min_temp']:.1f} - {ws['max_temp']:.1f}°C")
        print(f"  • Total Precipitation: {ws['total_precip']:.1f} mm")
        print(f"  • Avg Solar Radiation: {ws['avg_radiation']:.1f} MJ/m²/day")

        print(f"\nGrowing Degree Days:")
        gdd = results['gdd_accumulation']
        print(f"  • Base Temperature:    {gdd['base_temp']}°C")
        print(f"  • Total GDD:           {gdd['total']:.1f}")
        print(f"  • Daily Average:       {gdd['daily_avg']:.1f}")

        print(f"\nFinal Growth Stage: {results['final_stage']}")

        print(f"\nYield Analysis:")
        y = results['yield']
        print(f"  • Estimated Yield:     {y['yield_potential']:.2f} t/ha")
        print(f"  • Total Biomass:       {y['total_biomass']:.2f} t/ha")
        print(f"  • Harvest Index:       {y['harvest_index']:.2f}")
        print(f"  • Water Stress Factor: {y['water_stress_factor']:.2f}")
        print(f"  • Temperature Factor:  {y['temperature_factor']:.2f}")

        print(f"\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  → {rec}")

    print("\n" + "=" * 70)
    print("Simulation Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
