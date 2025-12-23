# Comprehensive Examples Guide

This document provides detailed, real-world examples for each module in the hetao_ag library.

---

## Table of Contents

1. [Complete Workflow Examples](#complete-workflow-examples)
2. [Core Module Examples](#core-module-examples)
3. [Soil Module Examples](#soil-module-examples)
4. [Water Module Examples](#water-module-examples)
5. [Crop Module Examples](#crop-module-examples)
6. [Livestock Module Examples](#livestock-module-examples)
7. [Space Module Examples](#space-module-examples)
8. [Optimization Module Examples](#optimization-module-examples)
9. [Integration Examples](#integration-examples)

---

## Complete Workflow Examples

### Example 1: Full Growing Season Simulation

This example demonstrates a complete growing season simulation integrating soil, water, and crop modules.

```python
#!/usr/bin/env python3
"""
Complete Growing Season Simulation
===================================

Simulates wheat growth for an entire season including:
- Daily soil moisture dynamics
- Evapotranspiration calculations
- Crop growth and stress
- Irrigation scheduling
- Yield estimation
"""

import numpy as np
from datetime import date, timedelta
from hetao_ag.soil import SoilMoistureModel, SoilType
from hetao_ag.water import (
    eto_penman_monteith, WeatherData,
    IrrigationScheduler, ScheduleType
)
from hetao_ag.crop import CropModel


def generate_weather(start_doy: int, days: int, latitude: float = 40.8) -> list:
    """Generate synthetic weather data for simulation."""
    weather_data = []
    np.random.seed(42)  # For reproducibility

    for day in range(days):
        doy = start_doy + day

        # Seasonal temperature pattern
        t_mean = 15 + 10 * np.sin((doy - 80) / 365 * 2 * np.pi)
        t_max = t_mean + 5 + np.random.randn()
        t_min = t_mean - 5 + np.random.randn()

        # Other weather parameters
        rh = 50 + 20 * np.random.rand()
        u2 = 1.5 + np.random.rand()
        rs = 15 + 10 * np.sin((doy - 80) / 365 * np.pi)

        weather_data.append(WeatherData(
            t_mean=t_mean,
            t_max=t_max,
            t_min=t_min,
            rh=rh,
            u2=u2,
            rs=rs,
            elevation=1050,
            latitude=latitude,
            doy=doy
        ))

    return weather_data


def simulate_growing_season():
    """Run the full growing season simulation."""

    # ========================================
    # 1. Initialize Models
    # ========================================
    print("=" * 60)
    print("Growing Season Simulation for Winter Wheat")
    print("=" * 60)

    # Soil model
    soil = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.28,
        root_depth_m=0.4,
        soil_type=SoilType.LOAM
    )

    # Crop model
    crop = CropModel("wheat")

    # Irrigation scheduler
    irrigator = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5,  # MAD = 50%
        max_application_mm=50,
        irrigation_efficiency=0.85
    )

    # ========================================
    # 2. Generate Weather Data
    # ========================================
    start_doy = 90  # Early April
    season_length = 120  # days
    weather = generate_weather(start_doy, season_length)

    # ========================================
    # 3. Run Daily Simulation
    # ========================================
    print("\nRunning simulation...")

    results = {
        'day': [],
        'moisture': [],
        'et0': [],
        'etc': [],
        'irrigation': [],
        'stress': [],
        'biomass': [],
        'stage': []
    }

    total_irrigation = 0
    total_et = 0

    for day, w in enumerate(weather, 1):
        # Calculate reference ET
        et0 = eto_penman_monteith(w)

        # Get crop coefficient from current growth stage
        kc = crop.phenology.get_kc_for_stage()
        etc = et0 * kc

        # Check if irrigation is needed
        rec = irrigator.recommend_by_moisture(
            current_moisture=soil.moisture,
            field_capacity=0.32,
            wilting_point=0.12
        )

        irrigation_mm = rec.amount_mm if rec.should_irrigate else 0

        # Simulate precipitation (occasional rain)
        rain_mm = 15 if np.random.rand() < 0.1 else 0

        # Update soil moisture
        soil.step_day(
            rain_mm=rain_mm,
            irrigation_mm=irrigation_mm,
            et_mm=etc * soil.stress_factor
        )

        # Update crop growth
        crop_result = crop.update_daily(
            t_max=w.t_max,
            t_min=w.t_min,
            et=et0,
            soil_moisture=soil.moisture,
            field_capacity=0.32,
            wilting_point=0.12,
            ECe=2.5  # Moderate salinity
        )

        # Record results
        results['day'].append(day)
        results['moisture'].append(soil.moisture)
        results['et0'].append(et0)
        results['etc'].append(etc)
        results['irrigation'].append(irrigation_mm)
        results['stress'].append(crop_result['stress_factor'])
        results['biomass'].append(crop_result['biomass_kg_ha'])
        results['stage'].append(crop_result['stage'])

        total_irrigation += irrigation_mm
        total_et += etc

        # Progress report every 30 days
        if day % 30 == 0:
            print(f"Day {day:3d}: Stage={crop_result['stage']:12s}, "
                  f"Moisture={soil.moisture:.3f}, "
                  f"Stress={crop_result['stress_factor']:.2f}, "
                  f"Biomass={crop_result['biomass_kg_ha']:,.0f} kg/ha")

    # ========================================
    # 4. Final Results
    # ========================================
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    final_yield = crop.estimate_yield()
    wue = final_yield / (total_et * 10)  # kg/m³

    print(f"\nCrop Performance:")
    print(f"  Final Stage: {crop.phenology.current_stage.value}")
    print(f"  Total Biomass: {crop.accumulated_biomass:,.0f} kg/ha")
    print(f"  Estimated Yield: {final_yield:,.0f} kg/ha")
    print(f"  Average Stress Factor: {np.mean(results['stress']):.3f}")

    print(f"\nWater Use:")
    print(f"  Total ET: {total_et:.1f} mm")
    print(f"  Total Irrigation: {total_irrigation:.1f} mm")
    print(f"  Water Use Efficiency: {wue:.2f} kg/m³")

    print(f"\nIrrigation Events: {sum(1 for i in results['irrigation'] if i > 0)}")

    return results


if __name__ == "__main__":
    results = simulate_growing_season()
```

---

## Core Module Examples

### Example 1: Physical Unit Calculations

```python
"""
Physical Units for Agricultural Calculations
=============================================

Demonstrates the unit system for common agricultural calculations.
"""

from hetao_ag.core import (
    Quantity, Unit, Dimension, DimensionError,
    meters, kilometers, hectares, celsius, kilopascals, ds_per_m,
    megajoules_per_m2, mm_per_day
)


def unit_system_demo():
    """Demonstrate unit system capabilities."""

    print("=" * 50)
    print("Physical Unit System Demo")
    print("=" * 50)

    # ==========================================
    # 1. Field Area Calculations
    # ==========================================
    print("\n1. Field Area Calculations")
    print("-" * 40)

    # Define field dimensions
    length = meters(500)
    width = meters(300)

    # Calculate area (manual)
    area_m2 = length.value * width.value
    area_ha = hectares(area_m2 / 10000)

    print(f"Field: {length.value}m × {width.value}m")
    print(f"Area: {area_m2:,.0f} m² = {area_ha.value:.2f} ha")

    # ==========================================
    # 2. Irrigation Calculations
    # ==========================================
    print("\n2. Irrigation Calculations")
    print("-" * 40)

    # Water volume calculation
    irrigation_depth = mm_per_day(50)  # 50 mm application
    field_area = hectares(10)

    # Convert: 1 mm over 1 ha = 10 m³
    water_volume_m3 = irrigation_depth.value * field_area.value * 10

    print(f"Irrigation depth: {irrigation_depth.value} mm")
    print(f"Field area: {field_area.value} ha")
    print(f"Water volume: {water_volume_m3:,.0f} m³")

    # ==========================================
    # 3. Salinity Calculations
    # ==========================================
    print("\n3. Salinity Calculations")
    print("-" * 40)

    # Soil salinity
    soil_ec = ds_per_m(4.5)
    print(f"Soil EC: {soil_ec.value} dS/m = {soil_ec.to(Unit.SIEMENS_PER_METER).value} S/m")

    # ==========================================
    # 4. Temperature Conversions
    # ==========================================
    print("\n4. Temperature Conversions")
    print("-" * 40)

    temp_c = celsius(25)
    temp_k = temp_c.to(Unit.KELVIN)

    print(f"Temperature: {temp_c.value}°C = {temp_k.value} K")

    # ==========================================
    # 5. Arithmetic with Units
    # ==========================================
    print("\n5. Arithmetic with Units")
    print("-" * 40)

    rain = mm_per_day(15)
    irrigation = mm_per_day(30)
    total_water = Quantity(rain.value + irrigation.value, Unit.MM_PER_DAY)

    print(f"Rain: {rain.value} mm + Irrigation: {irrigation.value} mm")
    print(f"Total water input: {total_water.value} mm")

    # ==========================================
    # 6. Dimension Safety
    # ==========================================
    print("\n6. Dimension Safety")
    print("-" * 40)

    try:
        # This will raise an error
        invalid = meters(100) + celsius(25)
    except DimensionError as e:
        print(f"Error (expected): Cannot add length to temperature")
        print(f"  {e}")


if __name__ == "__main__":
    unit_system_demo()
```

### Example 2: Configuration and Logging

```python
"""
Configuration Management and Logging
=====================================

Shows how to manage configurations and log experiments.
"""

from hetao_ag.core import (
    ConfigManager, create_default_config,
    get_logger, Logger, Timer
)
import numpy as np


def config_and_logging_demo():
    """Demonstrate configuration and logging."""

    # ==========================================
    # 1. Configuration Setup
    # ==========================================
    print("=" * 50)
    print("Configuration Management")
    print("=" * 50)

    # Create config with defaults
    config = ConfigManager(defaults=create_default_config())

    # Access nested configuration
    print("\nSoil Configuration:")
    print(f"  Field Capacity: {config.get('soil.field_capacity')}")
    print(f"  Wilting Point: {config.get('soil.wilting_point')}")
    print(f"  Bulk Density: {config.get('soil.bulk_density_kg_m3')}")

    print("\nCrop Configuration:")
    print(f"  Base Temperature: {config.get('crop.thermal_time_base_celsius')}°C")

    # Dynamic configuration updates
    config.set("custom.experiment_id", "EXP_2024_001")
    config.set("custom.irrigation_strategy", "deficit")
    print(f"\nCustom Settings:")
    print(f"  Experiment ID: {config.get('custom.experiment_id')}")
    print(f"  Strategy: {config.get('custom.irrigation_strategy')}")

    # ==========================================
    # 2. Logging Setup
    # ==========================================
    print("\n" + "=" * 50)
    print("Logging System")
    print("=" * 50)

    logger = get_logger("simulation_demo")

    # Standard logging
    logger.info("Starting simulation demo")
    logger.debug("This is debug info (may not show at default level)")
    logger.warning("Example warning message")

    # Structured logging with extra data
    logger.info("Irrigation event",
                field_id="F001",
                amount_mm=35.0,
                method="drip")

    # ==========================================
    # 3. Experiment Tracking
    # ==========================================
    print("\n" + "-" * 50)
    print("Experiment Tracking")
    print("-" * 50)

    # Log experiment start
    logger.log_experiment_start(
        experiment_name="Deficit Irrigation Trial",
        parameters={
            "crop": "wheat",
            "irrigation_strategy": "deficit",
            "target_etc": 0.8,
            "soil_type": "loam"
        },
        random_seed=42
    )

    # Simulate some work
    np.random.seed(42)
    results = {
        "yield_kg_ha": 5200 + np.random.randn() * 200,
        "water_saved_pct": 15.5,
        "stress_days": 12
    }

    # Log experiment end
    logger.log_experiment_end(
        success=True,
        results=results
    )

    # ==========================================
    # 4. Timer for Performance
    # ==========================================
    print("\n" + "-" * 50)
    print("Performance Timing")
    print("-" * 50)

    with Timer("Heavy calculation"):
        # Simulate heavy computation
        total = sum(range(1000000))

    with Timer("Data processing"):
        data = np.random.rand(1000, 1000)
        mean = np.mean(data)


if __name__ == "__main__":
    config_and_logging_demo()
```

---

## Soil Module Examples

### Example 1: Seasonal Soil Moisture Dynamics

```python
"""
Seasonal Soil Moisture Dynamics
===============================

Simulates soil moisture throughout a growing season with
varying precipitation and ET patterns.
"""

import numpy as np
from hetao_ag.soil import SoilMoistureModel, SoilType


def seasonal_moisture_simulation():
    """Simulate seasonal soil moisture dynamics."""

    print("=" * 60)
    print("Seasonal Soil Moisture Simulation")
    print("=" * 60)

    # Initialize model
    model = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.28,
        root_depth_m=0.5,
        soil_type=SoilType.CLAY_LOAM
    )

    # Generate synthetic data for 120 days
    np.random.seed(42)

    # Seasonal ET pattern (increasing towards mid-season)
    days = 120
    et_pattern = 2 + 4 * np.sin(np.linspace(0, np.pi, days))

    # Random precipitation events
    precip = np.zeros(days)
    rain_days = np.random.choice(days, size=15, replace=False)
    precip[rain_days] = np.random.exponential(15, size=15)

    print("\nSimulation Parameters:")
    print(f"  Duration: {days} days")
    print(f"  Soil Type: Clay Loam")
    print(f"  Field Capacity: 0.32")
    print(f"  Wilting Point: 0.12")
    print(f"  Rain Events: {len(rain_days)}")

    # Run simulation
    results = {
        'moisture': [],
        'stress': [],
        'drainage': [],
        'irrigation': []
    }

    print("\n" + "-" * 60)
    print("Daily Summary (every 20 days):")
    print("-" * 60)

    for day in range(days):
        # Check if irrigation needed
        if model.moisture < 0.18:  # Threshold
            irrigation = model.irrigation_need_mm
        else:
            irrigation = 0

        # Daily step
        result = model.step_day(
            rain_mm=precip[day],
            irrigation_mm=irrigation,
            et_mm=et_pattern[day]
        )

        results['moisture'].append(model.moisture)
        results['stress'].append(model.stress_factor)
        results['drainage'].append(result.get('drainage_mm', 0))
        results['irrigation'].append(irrigation)

        if (day + 1) % 20 == 0:
            print(f"Day {day+1:3d}: Moisture={model.moisture:.3f}, "
                  f"Stress={model.stress_factor:.2f}, "
                  f"Rain={precip[day]:.1f}mm, "
                  f"ET={et_pattern[day]:.1f}mm")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nSoil Moisture:")
    print(f"  Mean: {np.mean(results['moisture']):.3f}")
    print(f"  Min: {np.min(results['moisture']):.3f}")
    print(f"  Max: {np.max(results['moisture']):.3f}")

    print(f"\nStress Factor:")
    print(f"  Mean: {np.mean(results['stress']):.3f}")
    print(f"  Days with stress (<1.0): {sum(1 for s in results['stress'] if s < 1.0)}")

    print(f"\nWater Balance:")
    print(f"  Total Precipitation: {np.sum(precip):.1f} mm")
    print(f"  Total Irrigation: {np.sum(results['irrigation']):.1f} mm")
    print(f"  Total ET: {np.sum(et_pattern):.1f} mm")
    print(f"  Total Drainage: {np.sum(results['drainage']):.1f} mm")


if __name__ == "__main__":
    seasonal_moisture_simulation()
```

### Example 2: Salinity Management

```python
"""
Salinity Management Simulation
==============================

Demonstrates salt accumulation and leaching strategies.
"""

from hetao_ag.soil import (
    SalinityModel,
    classify_soil_salinity,
    classify_water_salinity
)
import numpy as np


def salinity_management_demo():
    """Demonstrate salinity modeling and management."""

    print("=" * 60)
    print("Salinity Management Simulation")
    print("=" * 60)

    # Initialize with moderate salinity
    model = SalinityModel(
        initial_ECe=3.0,  # dS/m
        root_depth_m=0.4,
        soil_water_content=0.25,
        bulk_density=1400
    )

    print(f"\nInitial Conditions:")
    print(f"  Soil ECe: {model.ECe:.2f} dS/m")
    print(f"  Classification: {classify_soil_salinity(model.ECe)}")

    # ==========================================
    # Scenario 1: Irrigation with moderately saline water
    # ==========================================
    print("\n" + "-" * 60)
    print("Scenario 1: Irrigation with Saline Water")
    print("-" * 60)

    irrigation_ec = 1.8  # dS/m
    print(f"Irrigation Water EC: {irrigation_ec} dS/m")
    print(f"Water Classification: {classify_water_salinity(irrigation_ec)}")

    for week in range(8):
        # Apply 50mm irrigation twice per week
        for _ in range(2):
            model.irrigate(amount_mm=50, ec_water=irrigation_ec)

        print(f"Week {week+1}: Soil ECe = {model.ECe:.2f} dS/m "
              f"({classify_soil_salinity(model.ECe)})")

    # ==========================================
    # Scenario 2: Leaching to recover soil
    # ==========================================
    print("\n" + "-" * 60)
    print("Scenario 2: Leaching to Reduce Salinity")
    print("-" * 60)

    # Calculate leaching requirement
    lr = model.leaching_requirement(ec_irrigation=1.8, ec_threshold=4.0)
    print(f"Leaching Requirement: {lr:.1%}")

    # Apply leaching irrigations
    print("\nApplying leaching irrigations...")
    for i in range(5):
        # Heavy irrigation with low EC water
        model.irrigate(amount_mm=100, ec_water=0.5)
        # Drainage/leaching
        model.leach(drainage_mm=60)
        print(f"Leaching {i+1}: Soil ECe = {model.ECe:.2f} dS/m")

    print(f"\nFinal Soil ECe: {model.ECe:.2f} dS/m")
    print(f"Classification: {classify_soil_salinity(model.ECe)}")


if __name__ == "__main__":
    salinity_management_demo()
```

---

## Water Module Examples

### Example 1: ET Calculation Methods Comparison

```python
"""
Evapotranspiration Methods Comparison
=====================================

Compares Penman-Monteith and Hargreaves methods.
"""

from hetao_ag.water import (
    eto_penman_monteith, eto_hargreaves,
    WeatherData, extraterrestrial_radiation
)
import numpy as np


def et_comparison():
    """Compare different ET calculation methods."""

    print("=" * 60)
    print("Evapotranspiration Methods Comparison")
    print("=" * 60)

    # Location parameters
    latitude = 40.8  # degrees N
    elevation = 1050  # meters

    # Generate monthly weather data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    doys = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]

    # Typical weather patterns (simplified)
    t_means = [-5, -2, 5, 12, 18, 23, 26, 25, 19, 12, 4, -3]
    t_ranges = [8, 9, 10, 12, 13, 14, 14, 13, 12, 11, 9, 8]

    print(f"\nLocation: Lat={latitude}°N, Elevation={elevation}m")
    print("\n" + "-" * 60)
    print(f"{'Month':^6} {'DOY':^5} {'T_mean':^7} {'Ra':^8} {'PM':^8} {'HG':^8}")
    print(f"{'':^6} {'':^5} {'(°C)':^7} {'MJ/m²':^8} {'mm/d':^8} {'mm/d':^8}")
    print("-" * 60)

    pm_annual = 0
    hg_annual = 0

    for i, month in enumerate(months):
        doy = doys[i]
        t_mean = t_means[i]
        t_range = t_ranges[i]
        t_max = t_mean + t_range / 2
        t_min = t_mean - t_range / 2

        # Extraterrestrial radiation
        Ra = extraterrestrial_radiation(latitude, doy)

        # Penman-Monteith
        weather = WeatherData(
            t_mean=t_mean,
            t_max=t_max,
            t_min=t_min,
            rh=60,
            u2=2.0,
            rs=Ra * 0.5,  # Approximate solar radiation
            elevation=elevation,
            latitude=latitude,
            doy=doy
        )

        et0_pm = max(0, eto_penman_monteith(weather))

        # Hargreaves
        et0_hg = max(0, eto_hargreaves(t_mean, t_max, t_min, Ra))

        print(f"{month:^6} {doy:^5} {t_mean:^7.1f} {Ra:^8.1f} {et0_pm:^8.2f} {et0_hg:^8.2f}")

        pm_annual += et0_pm * 30  # Approximate monthly total
        hg_annual += et0_hg * 30

    print("-" * 60)
    print(f"\nAnnual Totals:")
    print(f"  Penman-Monteith: {pm_annual:.0f} mm")
    print(f"  Hargreaves: {hg_annual:.0f} mm")
    print(f"  Difference: {(hg_annual - pm_annual) / pm_annual * 100:+.1f}%")


if __name__ == "__main__":
    et_comparison()
```

### Example 2: Irrigation Scheduling Strategies

```python
"""
Irrigation Scheduling Strategies
================================

Compares different irrigation scheduling methods.
"""

from hetao_ag.water import IrrigationScheduler, ScheduleType
import numpy as np


def irrigation_strategies_demo():
    """Demonstrate different irrigation scheduling strategies."""

    print("=" * 60)
    print("Irrigation Scheduling Strategies")
    print("=" * 60)

    # Soil parameters
    fc = 0.32
    wp = 0.12

    # ==========================================
    # Strategy 1: Soil Moisture Based
    # ==========================================
    print("\n1. SOIL MOISTURE BASED SCHEDULING")
    print("-" * 50)

    scheduler = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5,  # 50% MAD
        max_application_mm=50
    )

    # Simulate different moisture levels
    moisture_levels = [0.30, 0.25, 0.20, 0.18, 0.15]

    print(f"Trigger at MAD = 50% (Critical moisture = {fc - 0.5*(fc-wp):.3f})")
    print()

    for moisture in moisture_levels:
        rec = scheduler.recommend_by_moisture(moisture, fc, wp)
        status = "IRRIGATE" if rec.should_irrigate else "OK"
        print(f"Moisture {moisture:.2f}: {status:10s} "
              f"Amount={rec.amount_mm:.1f}mm, Urgency={rec.urgency}")

    # ==========================================
    # Strategy 2: ET-Based Scheduling
    # ==========================================
    print("\n2. ET-BASED SCHEDULING")
    print("-" * 50)

    scheduler = IrrigationScheduler(
        method=ScheduleType.ET_BASED,
        max_application_mm=40
    )

    # Simulate cumulative ET
    print("Trigger when cumulative ETc reaches 40mm")
    print()

    cumulative_etc = [10, 20, 30, 40, 50]
    last_irrigation = 0

    for etc in cumulative_etc:
        rec = scheduler.recommend_by_et(etc, last_irrigation, trigger_mm=40)
        status = "IRRIGATE" if rec.should_irrigate else "Wait"
        print(f"Cumulative ETc {etc:2d}mm: {status:10s} "
              f"Deficit={etc - last_irrigation}mm")
        if rec.should_irrigate:
            last_irrigation = etc

    # ==========================================
    # Strategy 3: Deficit Irrigation
    # ==========================================
    print("\n3. DEFICIT IRRIGATION SCHEDULE")
    print("-" * 50)

    scheduler = IrrigationScheduler(
        method=ScheduleType.DEFICIT,
        max_application_mm=50
    )

    # Define deficit factors by growth stage
    deficits = {
        "emergence": 1.0,    # Full irrigation
        "vegetative": 0.9,   # 10% deficit
        "flowering": 1.0,    # Full irrigation (critical)
        "grain_fill": 0.8,   # 20% deficit
        "maturity": 0.6      # 40% deficit
    }

    # Full ETc requirement per stage (mm)
    etc_per_stage = {
        "emergence": 40,
        "vegetative": 100,
        "flowering": 120,
        "grain_fill": 100,
        "maturity": 40
    }

    print("Stage-specific deficit irrigation:")
    print()

    total_full = 0
    total_deficit = 0

    for stage, full_etc in etc_per_stage.items():
        deficit_factor = deficits[stage]
        actual = full_etc * deficit_factor
        saved = full_etc - actual

        print(f"{stage:12s}: Full={full_etc:3d}mm × {deficit_factor:.1f} "
              f"= {actual:.0f}mm (saved {saved:.0f}mm)")

        total_full += full_etc
        total_deficit += actual

    print(f"\nTotal: {total_deficit:.0f}mm vs {total_full}mm full irrigation")
    print(f"Water saving: {(1 - total_deficit/total_full)*100:.1f}%")


if __name__ == "__main__":
    irrigation_strategies_demo()
```

---

## Crop Module Examples

### Example 1: Multi-Crop Stress Comparison

```python
"""
Multi-Crop Stress Response Comparison
=====================================

Compares salt and water stress responses across crops.
"""

from hetao_ag.crop import (
    yield_reduction_salinity_crop,
    water_stress_from_moisture,
    combined_stress_factor,
    yield_with_stress,
    classify_salt_tolerance,
    CROP_SALT_TOLERANCE
)
import numpy as np


def stress_comparison():
    """Compare stress responses across crops."""

    print("=" * 60)
    print("Multi-Crop Stress Response Comparison")
    print("=" * 60)

    crops = ["wheat", "maize", "cotton", "barley", "soybean", "alfalfa"]

    # ==========================================
    # 1. Salt Tolerance Comparison
    # ==========================================
    print("\n1. SALT TOLERANCE PARAMETERS")
    print("-" * 60)
    print(f"{'Crop':12s} {'Threshold':>10s} {'Slope':>10s} {'Classification':>18s}")
    print("-" * 60)

    for crop in crops:
        params = CROP_SALT_TOLERANCE.get(crop)
        if params:
            tolerance = classify_salt_tolerance(crop)
            print(f"{crop:12s} {params.threshold:>10.1f} {params.slope:>10.3f} {tolerance:>18s}")

    # ==========================================
    # 2. Yield Response to Salinity
    # ==========================================
    print("\n2. YIELD RESPONSE TO SALINITY")
    print("-" * 60)

    ec_levels = [2, 4, 6, 8, 10, 12]

    print(f"{'Crop':12s}", end="")
    for ec in ec_levels:
        print(f"{'EC='+str(ec):>8s}", end="")
    print()
    print("-" * 60)

    for crop in crops:
        print(f"{crop:12s}", end="")
        for ec in ec_levels:
            rel_yield = yield_reduction_salinity_crop(ec, crop)
            print(f"{rel_yield*100:>7.0f}%", end="")
        print()

    # ==========================================
    # 3. Combined Stress Analysis
    # ==========================================
    print("\n3. COMBINED STRESS SCENARIO")
    print("-" * 60)

    # Scenario: moderate water stress + high salinity
    moisture = 0.18
    fc = 0.32
    wp = 0.12
    ECe = 8.0

    water_stress = water_stress_from_moisture(moisture, fc, wp)
    print(f"\nConditions:")
    print(f"  Soil moisture: {moisture}")
    print(f"  Water stress factor: {water_stress:.3f}")
    print(f"  Soil ECe: {ECe} dS/m")

    print(f"\n{'Crop':12s} {'Salt Ks':>10s} {'Combined':>10s} {'Yield':>12s}")
    print(f"{'':12s} {'':>10s} {'Ks':>10s} {'(kg/ha)':>12s}")
    print("-" * 60)

    potential_yields = {
        "wheat": 6000, "maize": 10000, "cotton": 4000,
        "barley": 5500, "soybean": 3000, "alfalfa": 8000
    }

    for crop in crops:
        salt_stress = yield_reduction_salinity_crop(ECe, crop)
        combined = combined_stress_factor(water_stress, salt_stress)
        actual_yield = yield_with_stress(potential_yields[crop], water_stress, salt_stress)

        print(f"{crop:12s} {salt_stress:>10.3f} {combined:>10.3f} {actual_yield:>12,.0f}")


if __name__ == "__main__":
    stress_comparison()
```

---

## Livestock Module Examples

### Example 1: Herd Health Monitoring System

```python
"""
Herd Health Monitoring System
=============================

Demonstrates comprehensive livestock health monitoring.
"""

from hetao_ag.livestock import (
    HealthMonitor, HerdHealthMonitor,
    BehaviorClassifier, AnimalBehavior,
    HealthStatus, AlertType
)
import numpy as np


def herd_monitoring_demo():
    """Demonstrate herd health monitoring."""

    print("=" * 60)
    print("Herd Health Monitoring System")
    print("=" * 60)

    np.random.seed(42)

    # ==========================================
    # 1. Setup Herd Monitoring
    # ==========================================
    print("\n1. SETTING UP HERD MONITORING")
    print("-" * 50)

    herd = HerdHealthMonitor()
    n_animals = 10

    # Add animals to herd
    for i in range(n_animals):
        animal_id = f"cow_{i+1:03d}"
        herd.add_animal(animal_id)

    print(f"Monitoring {n_animals} animals")

    # ==========================================
    # 2. Establish Baselines (7 days of normal data)
    # ==========================================
    print("\n2. ESTABLISHING BASELINES")
    print("-" * 50)

    for day in range(7):
        for animal_id, monitor in herd.animals.items():
            # Normal activity patterns
            activity = 100 + np.random.randn() * 5
            feeding_time = 240 + np.random.randn() * 15

            monitor.update_activity(activity)
            monitor.update_feeding_time(feeding_time)

    print("7 days of baseline data collected")
    print("All animals showing normal patterns")

    # ==========================================
    # 3. Simulate Day with Anomalies
    # ==========================================
    print("\n3. SIMULATING ANOMALIES")
    print("-" * 50)

    # Animal 3: Reduced activity (possible illness)
    herd.animals["cow_003"].update_activity(55)
    herd.animals["cow_003"].update_temperature(40.2)

    # Animal 7: Reduced feeding (possible digestive issue)
    herd.animals["cow_007"].update_feeding_time(120)

    # Animal 9: Both issues
    herd.animals["cow_009"].update_activity(50)
    herd.animals["cow_009"].update_feeding_time(100)
    herd.animals["cow_009"].update_temperature(39.8)

    # Check all animals for alerts
    all_alerts = herd.check_all()

    print(f"\nDetected {len(all_alerts)} health alerts:")
    print()

    for alert in all_alerts:
        print(f"  [{alert.severity.value:^8s}] {alert.animal_id}: "
              f"{alert.alert_type.value} - {alert.message}")

    # ==========================================
    # 4. Herd Health Summary
    # ==========================================
    print("\n4. HERD HEALTH SUMMARY")
    print("-" * 50)

    summary = herd.get_summary()

    for status, count in summary.items():
        pct = count / n_animals * 100
        bar = "█" * int(pct / 5)
        print(f"  {status:12s}: {count:2d} ({pct:5.1f}%) {bar}")

    # ==========================================
    # 5. Behavior Analysis
    # ==========================================
    print("\n5. BEHAVIOR CLASSIFICATION")
    print("-" * 50)

    classifier = BehaviorClassifier()

    # Sample motion and head position data
    observations = [
        (0.05, "down"),
        (0.15, "level"),
        (0.25, "down"),
        (0.45, None),
        (0.70, "up"),
    ]

    print(f"{'Motion':>8s} {'Head':>8s} {'Behavior':>15s}")
    print("-" * 35)

    for motion, head in observations:
        behavior = classifier.classify_from_motion(motion, head)
        head_str = head if head else "N/A"
        print(f"{motion:>8.2f} {head_str:>8s} {behavior.value:>15s}")


if __name__ == "__main__":
    herd_monitoring_demo()
```

---

## Space Module Examples

### Example 1: Vegetation Monitoring

```python
"""
Vegetation Monitoring with Remote Sensing
==========================================

Demonstrates spectral index calculations and vegetation health assessment.
"""

from hetao_ag.space import (
    compute_ndvi, compute_savi, compute_evi,
    compute_lswi, compute_ndwi,
    classify_vegetation_health
)
import numpy as np


def vegetation_monitoring_demo():
    """Demonstrate vegetation monitoring with spectral indices."""

    print("=" * 60)
    print("Vegetation Monitoring with Remote Sensing")
    print("=" * 60)

    np.random.seed(42)

    # ==========================================
    # 1. Simulate Multispectral Image
    # ==========================================
    print("\n1. SIMULATED MULTISPECTRAL DATA")
    print("-" * 50)

    # Create image with different land cover zones
    height, width = 100, 100

    # Initialize bands
    blue = np.zeros((height, width), dtype=np.float32)
    green = np.zeros((height, width), dtype=np.float32)
    red = np.zeros((height, width), dtype=np.float32)
    nir = np.zeros((height, width), dtype=np.float32)
    swir = np.zeros((height, width), dtype=np.float32)

    # Zone 1: Dense vegetation (top-left quadrant)
    blue[:50, :50] = 80 + np.random.randn(50, 50) * 10
    green[:50, :50] = 100 + np.random.randn(50, 50) * 10
    red[:50, :50] = 70 + np.random.randn(50, 50) * 10
    nir[:50, :50] = 250 + np.random.randn(50, 50) * 20
    swir[:50, :50] = 100 + np.random.randn(50, 50) * 10

    # Zone 2: Moderate vegetation (top-right)
    blue[:50, 50:] = 100 + np.random.randn(50, 50) * 10
    green[:50, 50:] = 120 + np.random.randn(50, 50) * 10
    red[:50, 50:] = 120 + np.random.randn(50, 50) * 10
    nir[:50, 50:] = 180 + np.random.randn(50, 50) * 15
    swir[:50, 50:] = 130 + np.random.randn(50, 50) * 10

    # Zone 3: Bare soil (bottom-left)
    blue[50:, :50] = 130 + np.random.randn(50, 50) * 10
    green[50:, :50] = 140 + np.random.randn(50, 50) * 10
    red[50:, :50] = 160 + np.random.randn(50, 50) * 10
    nir[50:, :50] = 150 + np.random.randn(50, 50) * 15
    swir[50:, :50] = 180 + np.random.randn(50, 50) * 10

    # Zone 4: Water (bottom-right)
    blue[50:, 50:] = 100 + np.random.randn(50, 50) * 10
    green[50:, 50:] = 80 + np.random.randn(50, 50) * 10
    red[50:, 50:] = 50 + np.random.randn(50, 50) * 10
    nir[50:, 50:] = 20 + np.random.randn(50, 50) * 5
    swir[50:, 50:] = 10 + np.random.randn(50, 50) * 5

    print(f"Image size: {height} × {width} pixels")
    print(f"Zones: Dense veg, Moderate veg, Bare soil, Water")

    # ==========================================
    # 2. Calculate Spectral Indices
    # ==========================================
    print("\n2. SPECTRAL INDEX CALCULATION")
    print("-" * 50)

    ndvi = compute_ndvi(red, nir)
    savi = compute_savi(red, nir, L=0.5)
    evi = compute_evi(blue, red, nir)
    lswi = compute_lswi(nir, swir)
    ndwi = compute_ndwi(green, nir)

    zones = [
        ("Dense Vegetation", slice(0, 50), slice(0, 50)),
        ("Moderate Vegetation", slice(0, 50), slice(50, 100)),
        ("Bare Soil", slice(50, 100), slice(0, 50)),
        ("Water", slice(50, 100), slice(50, 100))
    ]

    print(f"{'Zone':^22s} {'NDVI':>8s} {'SAVI':>8s} {'EVI':>8s} {'LSWI':>8s}")
    print("-" * 60)

    for zone_name, row_slice, col_slice in zones:
        zone_ndvi = ndvi[row_slice, col_slice].mean()
        zone_savi = savi[row_slice, col_slice].mean()
        zone_evi = evi[row_slice, col_slice].mean()
        zone_lswi = lswi[row_slice, col_slice].mean()

        print(f"{zone_name:^22s} {zone_ndvi:>8.3f} {zone_savi:>8.3f} "
              f"{zone_evi:>8.3f} {zone_lswi:>8.3f}")

    # ==========================================
    # 3. Vegetation Health Classification
    # ==========================================
    print("\n3. VEGETATION HEALTH CLASSIFICATION")
    print("-" * 50)

    ndvi_thresholds = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]

    for threshold in ndvi_thresholds:
        classification = classify_vegetation_health(threshold)
        print(f"NDVI = {threshold:5.2f}: {classification}")


if __name__ == "__main__":
    vegetation_monitoring_demo()
```

---

## Optimization Module Examples

### Example 1: Farm Resource Optimization

```python
"""
Farm Resource Optimization
==========================

Demonstrates linear programming for optimal crop mix and water allocation.
"""

from hetao_ag.opt import (
    LinearOptimizer, optimize_crop_mix,
    ScenarioEvaluator, multi_objective_score
)


def farm_optimization_demo():
    """Demonstrate farm resource optimization."""

    print("=" * 60)
    print("Farm Resource Optimization")
    print("=" * 60)

    # ==========================================
    # 1. Crop Mix Optimization
    # ==========================================
    print("\n1. OPTIMAL CROP MIX")
    print("-" * 50)

    crops = [
        {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
        {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
        {"name": "sunflower", "profit_per_ha": 450, "water_per_ha": 3500},
        {"name": "alfalfa", "profit_per_ha": 350, "water_per_ha": 2000},
    ]

    total_land = 200  # hectares
    total_water = 500000  # m³

    print(f"Constraints:")
    print(f"  Total land: {total_land} ha")
    print(f"  Total water: {total_water:,} m³")

    print(f"\nCrop Economics:")
    for crop in crops:
        print(f"  {crop['name']:12s}: Profit={crop['profit_per_ha']} $/ha, "
              f"Water={crop['water_per_ha']} m³/ha")

    # Optimize
    solution = optimize_crop_mix(crops, total_land, total_water)

    print(f"\nOptimal Allocation:")
    total_profit = 0
    total_water_used = 0

    for crop in crops:
        area = solution.get(crop['name'], 0)
        if area > 0.1:
            profit = area * crop['profit_per_ha']
            water = area * crop['water_per_ha']
            total_profit += profit
            total_water_used += water
            print(f"  {crop['name']:12s}: {area:6.1f} ha "
                  f"(Profit: ${profit:,.0f}, Water: {water:,.0f} m³)")

    print(f"\nResults:")
    print(f"  Total Profit: ${total_profit:,.0f}")
    print(f"  Water Used: {total_water_used:,.0f} m³ ({total_water_used/total_water*100:.1f}%)")
    print(f"  Land Used: {sum(solution.values()):.1f} ha ({sum(solution.values())/total_land*100:.1f}%)")

    # ==========================================
    # 2. Scenario Analysis
    # ==========================================
    print("\n2. SCENARIO ANALYSIS")
    print("-" * 50)

    crop_params = {
        "wheat": {
            "yield_kg_ha": 6000,
            "water_need_mm": 400,
            "price_per_kg": 0.30,
            "cost_per_ha": 800
        },
        "maize": {
            "yield_kg_ha": 10000,
            "water_need_mm": 600,
            "price_per_kg": 0.25,
            "cost_per_ha": 1200
        }
    }

    evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=400000)

    # Define scenarios
    scenarios = [
        ("All Wheat", {"wheat": 100}, 450),
        ("All Maize", {"maize": 100}, 650),
        ("60-40 Mix", {"wheat": 60, "maize": 40}, 500),
        ("40-60 Mix", {"wheat": 40, "maize": 60}, 550),
        ("Equal Mix", {"wheat": 50, "maize": 50}, 520),
    ]

    print(f"{'Scenario':15s} {'Profit':>12s} {'WUE':>10s}")
    print("-" * 40)

    for name, crop_areas, irrigation in scenarios:
        scenario = evaluator.evaluate_scenario(name, crop_areas, irrigation)
        print(f"{name:15s} ${scenario.total_profit:>10,.0f} "
              f"{scenario.water_use_efficiency:>10.4f}")

    # Compare scenarios
    comparison = evaluator.compare_scenarios()
    print(f"\nBest profit: {comparison['best_profit'].name}")
    print(f"Best water efficiency: {comparison['best_water_efficiency'].name}")

    # ==========================================
    # 3. Multi-Objective Scoring
    # ==========================================
    print("\n3. MULTI-OBJECTIVE SCORING")
    print("-" * 50)

    # Define weights
    weights = {"profit": 0.4, "water": 0.35, "sustainability": 0.25}

    print(f"Weights: Profit={weights['profit']}, Water Efficiency={weights['water']}, "
          f"Sustainability={weights['sustainability']}")
    print()

    for scenario in evaluator.scenarios:
        # Normalize metrics (example scaling)
        profit_score = min(1, scenario.total_profit / 150000)
        water_score = min(1, scenario.water_use_efficiency * 100)
        sustain_score = 0.8  # Example fixed score

        score = multi_objective_score(profit_score, water_score, sustain_score, weights)
        print(f"{scenario.name:15s}: Score = {score:.3f}")


if __name__ == "__main__":
    farm_optimization_demo()
```

---

## Integration Examples

### Example: Complete Farm Management System

```python
"""
Complete Farm Management System
===============================

Integrates all modules for comprehensive farm management.
"""

from hetao_ag.core import get_logger, ConfigManager, create_default_config
from hetao_ag.soil import SoilMoistureModel, SalinityModel
from hetao_ag.water import eto_penman_monteith, WeatherData, IrrigationScheduler, ScheduleType
from hetao_ag.crop import CropModel, yield_reduction_salinity_crop
from hetao_ag.space import compute_ndvi
from hetao_ag.opt import optimize_crop_mix, ScenarioEvaluator
import numpy as np


def integrated_farm_system():
    """Complete farm management integration."""

    # Setup logging
    logger = get_logger("farm_system")
    logger.info("Starting Integrated Farm Management System")

    print("=" * 70)
    print("INTEGRATED FARM MANAGEMENT SYSTEM")
    print("=" * 70)

    # ==========================================
    # 1. Configuration
    # ==========================================
    config = ConfigManager(defaults=create_default_config())

    farm_config = {
        "total_area_ha": 100,
        "crops": ["wheat", "maize"],
        "soil_type": "loam",
        "latitude": 40.8,
        "elevation": 1050
    }

    logger.info("Farm configured", **farm_config)

    # ==========================================
    # 2. Soil Assessment
    # ==========================================
    print("\n1. SOIL ASSESSMENT")
    print("-" * 60)

    soil = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.25
    )

    salinity = SalinityModel(initial_ECe=3.5)

    print(f"Current moisture: {soil.moisture:.3f}")
    print(f"Stress factor: {soil.stress_factor:.3f}")
    print(f"Soil ECe: {salinity.ECe:.2f} dS/m")

    # ==========================================
    # 3. Weather and ET
    # ==========================================
    print("\n2. WEATHER AND EVAPOTRANSPIRATION")
    print("-" * 60)

    weather = WeatherData(
        t_mean=25, t_max=32, t_min=18,
        rh=55, u2=2.0, rs=22,
        elevation=1050, latitude=40.8, doy=180
    )

    et0 = eto_penman_monteith(weather)
    print(f"Reference ET0: {et0:.2f} mm/day")

    # ==========================================
    # 4. Crop Analysis
    # ==========================================
    print("\n3. CROP ANALYSIS")
    print("-" * 60)

    for crop_name in farm_config["crops"]:
        model = CropModel(crop_name)
        kc = model.phenology.get_kc_for_stage()
        etc = et0 * kc

        salt_stress = yield_reduction_salinity_crop(salinity.ECe, crop_name)

        print(f"\n{crop_name.capitalize()}:")
        print(f"  Current Kc: {kc:.2f}")
        print(f"  Crop ETc: {etc:.2f} mm/day")
        print(f"  Salt stress factor: {salt_stress:.3f}")

    # ==========================================
    # 5. Irrigation Decision
    # ==========================================
    print("\n4. IRRIGATION DECISION")
    print("-" * 60)

    scheduler = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5
    )

    recommendation = scheduler.recommend_by_moisture(
        soil.moisture, 0.32, 0.12
    )

    if recommendation.should_irrigate:
        print(f"IRRIGATION RECOMMENDED")
        print(f"  Amount: {recommendation.amount_mm:.1f} mm")
        print(f"  Urgency: {recommendation.urgency}")
        logger.info("Irrigation recommended",
                   amount_mm=recommendation.amount_mm,
                   urgency=recommendation.urgency)
    else:
        print("No irrigation needed at this time")

    # ==========================================
    # 6. Resource Optimization
    # ==========================================
    print("\n5. RESOURCE OPTIMIZATION")
    print("-" * 60)

    crops = [
        {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
        {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
    ]

    total_water = 300000  # m³ for the season

    optimal_mix = optimize_crop_mix(
        crops,
        total_land=farm_config["total_area_ha"],
        total_water=total_water
    )

    print("Optimal crop allocation:")
    for crop, area in optimal_mix.items():
        if area > 0.1:
            print(f"  {crop}: {area:.1f} ha")

    # ==========================================
    # 7. Summary Report
    # ==========================================
    print("\n" + "=" * 70)
    print("FARM MANAGEMENT SUMMARY")
    print("=" * 70)

    print(f"""
    Soil Status:
      - Moisture: {soil.moisture:.3f} (Stress factor: {soil.stress_factor:.2f})
      - Salinity: {salinity.ECe:.2f} dS/m

    Weather:
      - Temperature: {weather.t_min:.0f}-{weather.t_max:.0f}°C
      - ET0: {et0:.2f} mm/day

    Irrigation:
      - Recommended: {'Yes' if recommendation.should_irrigate else 'No'}
      - Amount: {recommendation.amount_mm:.1f} mm

    Optimal Crop Mix:
      - Wheat: {optimal_mix.get('wheat', 0):.1f} ha
      - Maize: {optimal_mix.get('maize', 0):.1f} ha
    """)

    logger.info("Farm management report generated")


if __name__ == "__main__":
    integrated_farm_system()
```

---

*hetao_ag v1.0.0 Examples Guide*
