# -*- coding: utf-8 -*-
"""
Precision Irrigation Example
============================

This example demonstrates a complete precision irrigation workflow using
the hetao_ag library. It integrates soil moisture monitoring, evapotranspiration
calculation, and intelligent irrigation scheduling.

Learning Objectives:
- How to use the water module for ET calculation
- How to model soil moisture dynamics
- How to implement smart irrigation scheduling
- How to track water balance over a growing season

Author: Hetao College
"""

import numpy as np
from datetime import date, timedelta

# Import hetao_ag modules
from hetao_ag.water import (
    eto_penman_monteith,
    WeatherData,
    WaterBalance,
    IrrigationScheduler,
    ScheduleType,
    crop_coefficient,
    etc_crop
)
from hetao_ag.soil import SoilMoistureModel, SoilType
from hetao_ag.core import get_logger


def generate_weather_data(doy: int, seed: int = None) -> WeatherData:
    """
    Generate synthetic weather data for a given day of year.

    In real applications, this would come from a weather station or API.

    Args:
        doy: Day of year (1-365)
        seed: Random seed for reproducibility

    Returns:
        WeatherData object with realistic values
    """
    if seed is not None:
        np.random.seed(seed + doy)

    # Temperature follows seasonal pattern (Northern hemisphere)
    seasonal_factor = np.sin((doy - 80) / 365 * 2 * np.pi)

    t_mean = 15 + 10 * seasonal_factor + np.random.randn() * 2
    t_max = t_mean + 7 + np.random.randn()
    t_min = t_mean - 7 + np.random.randn()

    # Relative humidity (inverse to temperature)
    rh = 60 - 15 * seasonal_factor + np.random.randn() * 5
    rh = np.clip(rh, 30, 95)

    # Solar radiation (higher in summer)
    rs = 15 + 8 * seasonal_factor + np.random.randn() * 2
    rs = max(5, rs)

    # Wind speed (relatively constant)
    u2 = 2.0 + np.random.randn() * 0.5
    u2 = max(0.5, u2)

    return WeatherData(
        t_mean=t_mean,
        t_max=t_max,
        t_min=t_min,
        rh=rh,
        u2=u2,
        rs=rs,
        elevation=1050,  # Hetao region elevation
        latitude=40.8,   # Hetao region latitude
        doy=doy
    )


def run_precision_irrigation_simulation():
    """
    Run a complete precision irrigation simulation for wheat.

    This simulation:
    1. Tracks daily weather and calculates ET
    2. Monitors soil moisture
    3. Makes intelligent irrigation decisions
    4. Reports water use efficiency
    """
    print("=" * 70)
    print("PRECISION IRRIGATION SIMULATION")
    print("Crop: Winter Wheat | Location: Hetao Irrigation District")
    print("=" * 70)

    # Initialize logger
    logger = get_logger("irrigation_sim")
    logger.info("Starting precision irrigation simulation")

    # =========================================================================
    # Step 1: Initialize Models
    # =========================================================================
    print("\n[1] Initializing simulation models...")

    # Soil moisture model - typical loam soil in Hetao
    soil = SoilMoistureModel(
        field_capacity=0.32,      # Maximum water holding capacity
        wilting_point=0.12,       # Permanent wilting point
        initial_moisture=0.25,    # Starting moisture (75% of available)
        root_depth_m=0.6,         # Effective root zone depth
        soil_type=SoilType.LOAM
    )
    print(f"   Soil: Field capacity={soil.field_capacity}, "
          f"Wilting point={soil.wilting_point}")

    # Water balance tracker
    water_balance = WaterBalance(
        initial_storage_mm=80,    # Initial water in root zone
        max_storage_mm=120,       # Field capacity in mm
        min_storage_mm=40         # Wilting point in mm
    )

    # Irrigation scheduler using soil moisture method
    scheduler = IrrigationScheduler(
        method=ScheduleType.SOIL_MOISTURE,
        trigger_threshold=0.5,     # Irrigate when 50% depleted
        max_application_mm=50,     # Maximum per irrigation event
        irrigation_efficiency=0.85  # 85% efficiency (drip/sprinkler)
    )
    print(f"   Scheduler: {scheduler.method.value} method, "
          f"trigger at {scheduler.trigger_threshold*100}% depletion")

    # =========================================================================
    # Step 2: Simulation Parameters
    # =========================================================================

    # Simulation period: March 15 to July 15 (122 days for wheat)
    start_doy = 74   # March 15
    end_doy = 196    # July 15
    num_days = end_doy - start_doy

    # Wheat growth stages for Kc values
    growth_stages = [
        (0, 20, "initial", 0.3),
        (21, 40, "development", 0.7),
        (41, 80, "mid-season", 1.15),
        (81, 110, "late-season", 0.4),
        (111, num_days, "maturity", 0.3)
    ]

    print(f"\n   Simulation: Day {start_doy} to {end_doy} ({num_days} days)")

    # =========================================================================
    # Step 3: Run Daily Simulation
    # =========================================================================
    print("\n[2] Running daily simulation...")

    # Results storage
    results = {
        'doy': [],
        'et0': [],
        'etc': [],
        'moisture': [],
        'irrigation': [],
        'precipitation': [],
        'stage': []
    }

    total_irrigation = 0
    total_precipitation = 0
    irrigation_events = 0

    for day_idx in range(num_days):
        doy = start_doy + day_idx

        # Get weather data
        weather = generate_weather_data(doy, seed=42)

        # Calculate reference ET (FAO-56 Penman-Monteith)
        et0 = eto_penman_monteith(weather)

        # Determine growth stage and Kc
        kc = 1.0
        stage_name = "unknown"
        for start, end, name, kc_value in growth_stages:
            if start <= day_idx <= end:
                kc = kc_value
                stage_name = name
                break

        # Calculate crop ET
        etc = etc_crop(et0, kc)

        # Simulate precipitation (stochastic - about 20% of days)
        precipitation = 0.0
        if np.random.random() < 0.15:
            precipitation = np.random.exponential(10)  # Mean 10mm events
            precipitation = min(precipitation, 40)  # Cap at 40mm

        # Update soil with precipitation
        if precipitation > 0:
            infiltration, runoff = soil.add_water(precipitation)
            water_balance.add_precipitation(precipitation)

        # Check irrigation recommendation
        rec = scheduler.recommend_by_moisture(
            current_moisture=soil.moisture,
            field_capacity=soil.field_capacity,
            wilting_point=soil.wilting_point,
            root_depth_m=soil.root_depth_m
        )

        irrigation = 0.0
        if rec.should_irrigate:
            irrigation = rec.amount_mm
            soil.add_water(irrigation)
            water_balance.add_irrigation(irrigation)
            total_irrigation += irrigation
            irrigation_events += 1

        # Remove ET
        actual_et = soil.remove_water(etc)
        water_balance.remove_et(actual_et)

        # Store results
        results['doy'].append(doy)
        results['et0'].append(et0)
        results['etc'].append(etc)
        results['moisture'].append(soil.moisture)
        results['irrigation'].append(irrigation)
        results['precipitation'].append(precipitation)
        results['stage'].append(stage_name)

        total_precipitation += precipitation

        # Progress report every 30 days
        if (day_idx + 1) % 30 == 0:
            print(f"   Day {day_idx+1}/{num_days}: "
                  f"Moisture={soil.moisture:.3f}, "
                  f"Stage={stage_name}, "
                  f"Cumulative irrigation={total_irrigation:.0f}mm")

    # =========================================================================
    # Step 4: Generate Summary Report
    # =========================================================================
    print("\n[3] Simulation Results Summary")
    print("=" * 70)

    # Calculate statistics
    avg_et0 = np.mean(results['et0'])
    avg_etc = np.mean(results['etc'])
    total_et = np.sum(results['etc'])

    print(f"\n   EVAPOTRANSPIRATION:")
    print(f"   - Average ET0: {avg_et0:.2f} mm/day")
    print(f"   - Average ETc: {avg_etc:.2f} mm/day")
    print(f"   - Total ETc: {total_et:.0f} mm")

    print(f"\n   WATER INPUTS:")
    print(f"   - Total precipitation: {total_precipitation:.0f} mm")
    print(f"   - Total irrigation: {total_irrigation:.0f} mm")
    print(f"   - Number of irrigation events: {irrigation_events}")
    print(f"   - Average irrigation per event: "
          f"{total_irrigation/max(1,irrigation_events):.1f} mm")

    print(f"\n   SOIL MOISTURE:")
    print(f"   - Initial: {results['moisture'][0]:.3f}")
    print(f"   - Final: {results['moisture'][-1]:.3f}")
    print(f"   - Minimum: {min(results['moisture']):.3f}")
    print(f"   - Maximum: {max(results['moisture']):.3f}")

    # Estimate yield (simplified)
    # Assuming 6000 kg/ha potential with no stress
    # Average moisture stress factor
    avg_moisture = np.mean(results['moisture'])
    fc, wp = soil.field_capacity, soil.wilting_point
    available_water = (avg_moisture - wp) / (fc - wp)
    stress_factor = min(1.0, available_water / 0.5) if available_water < 0.5 else 1.0

    estimated_yield = 6000 * stress_factor  # kg/ha

    print(f"\n   YIELD ESTIMATION:")
    print(f"   - Average stress factor: {stress_factor:.3f}")
    print(f"   - Estimated yield: {estimated_yield:.0f} kg/ha")

    # Water use efficiency
    total_water = total_precipitation + total_irrigation
    wue = estimated_yield / (total_water * 10)  # kg/m3

    print(f"\n   WATER USE EFFICIENCY:")
    print(f"   - Total water applied: {total_water:.0f} mm")
    print(f"   - Water use efficiency: {wue:.2f} kg/m3")

    # Get water balance summary
    print(f"\n   WATER BALANCE:")
    summary = water_balance.get_summary()
    for key, value in summary.items():
        print(f"   - {key}: {value:.1f}")

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)

    return results


def compare_irrigation_strategies():
    """
    Compare different irrigation scheduling strategies.

    Demonstrates:
    - Fixed interval irrigation
    - Soil moisture-based irrigation
    - ET-based irrigation
    - Deficit irrigation
    """
    print("\n" + "=" * 70)
    print("IRRIGATION STRATEGY COMPARISON")
    print("=" * 70)

    strategies = [
        ("Fixed Interval (7 days, 40mm)",
         IrrigationScheduler(method=ScheduleType.FIXED_INTERVAL)),
        ("Soil Moisture Based (50% depletion)",
         IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE,
                            trigger_threshold=0.5)),
        ("Soil Moisture Based (70% depletion - deficit)",
         IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE,
                            trigger_threshold=0.7)),
    ]

    print("\nComparing 60-day simulation for each strategy:\n")
    print(f"{'Strategy':<45} {'Total Irrig':>12} {'Events':>8} {'Avg/Event':>10}")
    print("-" * 75)

    for name, scheduler in strategies:
        soil = SoilMoistureModel(
            field_capacity=0.32,
            wilting_point=0.12,
            initial_moisture=0.25
        )

        total_irrigation = 0
        events = 0

        for day in range(60):
            # Simulate daily ET
            et = 5 + np.sin(day / 15) * 2  # 3-7 mm/day

            # Check irrigation
            if scheduler.method == ScheduleType.FIXED_INTERVAL:
                if day % 7 == 0:
                    irrigation = 40
                    soil.add_water(irrigation)
                    total_irrigation += irrigation
                    events += 1
            else:
                rec = scheduler.recommend_by_moisture(
                    soil.moisture, soil.field_capacity, soil.wilting_point
                )
                if rec.should_irrigate:
                    soil.add_water(rec.amount_mm)
                    total_irrigation += rec.amount_mm
                    events += 1

            # Remove ET
            soil.remove_water(et)

        avg_per_event = total_irrigation / max(1, events)
        print(f"{name:<45} {total_irrigation:>10.0f}mm {events:>8} {avg_per_event:>8.1f}mm")

    print("\nNote: Soil moisture-based scheduling typically uses less water")
    print("while maintaining crop productivity through precise timing.")


if __name__ == "__main__":
    # Run main simulation
    results = run_precision_irrigation_simulation()

    # Compare strategies
    compare_irrigation_strategies()

    print("\n" + "=" * 70)
    print("Example complete! See code comments for detailed explanations.")
    print("=" * 70)
