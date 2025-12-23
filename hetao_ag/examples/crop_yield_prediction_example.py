# -*- coding: utf-8 -*-
"""
Crop Yield Prediction Example
=============================

This example demonstrates crop yield prediction using the hetao_ag library.
It combines crop growth modeling, stress analysis, and phenology tracking
to estimate final yields under various environmental conditions.

Learning Objectives:
- Understanding the crop growth model
- Modeling water and salt stress effects
- Tracking phenological development with GDD
- Predicting yields under different scenarios

Author: Hetao College
"""

import numpy as np
from typing import Dict, List, Tuple

# Import hetao_ag modules
from hetao_ag.crop import (
    CropModel,
    CropConfig,
    PhenologyTracker,
    GrowthStage,
    CROP_CONFIGS,
    CROP_PHENOLOGY,
    yield_reduction_salinity_crop,
    water_stress_from_moisture,
    combined_stress_factor,
    classify_salt_tolerance,
    CROP_SALT_TOLERANCE
)


def demonstrate_salt_tolerance():
    """
    Demonstrate crop salt tolerance analysis.

    The Maas-Hoffman model predicts yield reduction due to soil salinity.
    Different crops have different tolerance levels.
    """
    print("=" * 70)
    print("CROP SALT TOLERANCE ANALYSIS")
    print("(Maas-Hoffman Model)")
    print("=" * 70)

    print("\n1. Available Crop Salt Tolerance Parameters:")
    print("-" * 50)
    print(f"{'Crop':<12} {'Threshold':>10} {'Slope':>10} {'Classification'}")
    print(f"{'':12} {'(dS/m)':>10} {'(per dS/m)':>10}")
    print("-" * 50)

    for crop, params in CROP_SALT_TOLERANCE.items():
        classification = classify_salt_tolerance(crop)
        print(f"{crop:<12} {params.threshold:>10.1f} {params.slope:>10.3f} {classification}")

    # Demonstrate yield reduction at different ECe levels
    print("\n2. Yield Reduction at Different Salinity Levels:")
    print("-" * 70)

    ece_levels = [2, 4, 6, 8, 10, 12]
    crops_to_analyze = ["wheat", "maize", "cotton", "barley"]

    # Header
    print(f"{'ECe (dS/m)':>12}", end="")
    for crop in crops_to_analyze:
        print(f"{crop:>12}", end="")
    print()
    print("-" * 70)

    # Data rows
    for ece in ece_levels:
        print(f"{ece:>12}", end="")
        for crop in crops_to_analyze:
            rel_yield = yield_reduction_salinity_crop(ece, crop)
            print(f"{rel_yield*100:>11.1f}%", end="")
        print()

    print("\nInterpretation:")
    print("  - Higher threshold = more tolerant to initial salinity")
    print("  - Lower slope = slower yield decline per unit salinity increase")
    print("  - Barley and cotton are more salt tolerant than maize")


def demonstrate_water_stress():
    """
    Demonstrate water stress modeling.

    Shows how soil moisture levels affect the water stress factor,
    which reduces transpiration and biomass accumulation.
    """
    print("\n" + "=" * 70)
    print("WATER STRESS ANALYSIS")
    print("=" * 70)

    # Soil parameters
    field_capacity = 0.32
    wilting_point = 0.12

    print("\nSoil Parameters:")
    print(f"  Field Capacity (FC): {field_capacity}")
    print(f"  Wilting Point (WP): {wilting_point}")
    print(f"  Available Water Capacity: {field_capacity - wilting_point:.2f}")

    # Calculate stress at various moisture levels
    moisture_levels = np.linspace(wilting_point, field_capacity, 11)

    print("\nWater Stress Factor (Ks) at Different Moisture Levels:")
    print("-" * 50)
    print(f"{'Moisture':>12} {'Ks (p=0.3)':>12} {'Ks (p=0.5)':>12} {'Ks (p=0.7)':>12}")
    print("-" * 50)

    for moisture in moisture_levels:
        ks_03 = water_stress_from_moisture(moisture, field_capacity, wilting_point, p=0.3)
        ks_05 = water_stress_from_moisture(moisture, field_capacity, wilting_point, p=0.5)
        ks_07 = water_stress_from_moisture(moisture, field_capacity, wilting_point, p=0.7)
        print(f"{moisture:>12.3f} {ks_03:>12.3f} {ks_05:>12.3f} {ks_07:>12.3f}")

    print("\nNote: p = fraction of available water that can be depleted")
    print("      before stress begins (lower p = more sensitive crop)")


def demonstrate_combined_stress():
    """
    Demonstrate combined water and salt stress effects.
    """
    print("\n" + "=" * 70)
    print("COMBINED STRESS ANALYSIS")
    print("=" * 70)

    print("\nCombination Methods:")
    print("  - Multiplicative: Ks = Ks_water * Ks_salt")
    print("  - Minimum: Ks = min(Ks_water, Ks_salt)")
    print("  - Additive: Ks = max(0, 1 - (1-Ks_water) - (1-Ks_salt))")

    # Example scenarios
    scenarios = [
        (1.0, 1.0, "No stress"),
        (0.8, 1.0, "Mild water stress only"),
        (1.0, 0.85, "Mild salt stress only"),
        (0.8, 0.85, "Both mild stresses"),
        (0.5, 0.7, "Moderate both stresses"),
        (0.3, 0.5, "Severe both stresses"),
    ]

    print("\nCombined Stress Factor Under Different Scenarios:")
    print("-" * 75)
    print(f"{'Scenario':<25} {'Ks_water':>10} {'Ks_salt':>10} {'Multiplicative':>15} {'Minimum':>10}")
    print("-" * 75)

    for ks_water, ks_salt, description in scenarios:
        mult = combined_stress_factor(ks_water, ks_salt, "multiplicative")
        minimum = combined_stress_factor(ks_water, ks_salt, "minimum")
        print(f"{description:<25} {ks_water:>10.2f} {ks_salt:>10.2f} {mult:>15.3f} {minimum:>10.3f}")


def demonstrate_phenology_tracking():
    """
    Demonstrate phenological development tracking using GDD.
    """
    print("\n" + "=" * 70)
    print("PHENOLOGY TRACKING WITH GROWING DEGREE DAYS")
    print("=" * 70)

    crops = ["wheat", "maize", "rice"]

    for crop in crops:
        print(f"\n{crop.upper()} Phenology Development:")
        print("-" * 50)

        tracker = PhenologyTracker(crop)
        config = CROP_PHENOLOGY.get(crop, tracker.config)

        print(f"Base Temperature: {config.base_temperature} C")
        print(f"GDD Requirements by Stage:")
        for stage, gdd in config.stage_gdd.items():
            print(f"  - {stage}: {gdd} GDD")

        # Simulate 120 days of growth
        print(f"\nSimulating 120 days of growth...")
        daily_temps = []
        for day in range(120):
            # Seasonal temperature pattern
            t_max = 25 + 8 * np.sin(day / 60 * np.pi) + np.random.randn() * 2
            t_min = 12 + 5 * np.sin(day / 60 * np.pi) + np.random.randn() * 2
            daily_temps.append((t_max, t_min))
            tracker.accumulate_gdd(t_max, t_min)

        print(f"Final Accumulated GDD: {tracker.accumulated_gdd:.0f}")
        print(f"Final Stage: {tracker.current_stage.value}")
        print(f"Progress to Maturity: {tracker.progress_to_maturity()*100:.1f}%")
        print(f"Current Kc: {tracker.get_kc_for_stage():.2f}")


def run_yield_prediction():
    """
    Run complete yield prediction simulation for multiple crops.
    """
    print("\n" + "=" * 70)
    print("YIELD PREDICTION SIMULATION")
    print("=" * 70)

    # Environmental scenarios
    scenarios = [
        {
            "name": "Optimal Conditions",
            "soil_moisture": 0.28,
            "ECe": 2.0,
            "description": "Good irrigation, low salinity"
        },
        {
            "name": "Moderate Stress",
            "soil_moisture": 0.20,
            "ECe": 5.0,
            "description": "Limited water, moderate salinity"
        },
        {
            "name": "High Salinity",
            "soil_moisture": 0.25,
            "ECe": 8.0,
            "description": "Adequate water, high salinity"
        },
        {
            "name": "Water Deficit",
            "soil_moisture": 0.16,
            "ECe": 3.0,
            "description": "Severe water stress, low salinity"
        },
        {
            "name": "Severe Stress",
            "soil_moisture": 0.16,
            "ECe": 10.0,
            "description": "Both stresses severe"
        }
    ]

    crops = ["wheat", "maize", "cotton"]

    print("\nSimulating 100-day growing season under different scenarios...")
    print()

    # Results table header
    print(f"{'Scenario':<20}", end="")
    for crop in crops:
        print(f"{crop:>15}", end="")
    print()
    print("-" * 65)

    for scenario in scenarios:
        print(f"{scenario['name']:<20}", end="")

        for crop in crops:
            model = CropModel(crop)

            # Run simulation
            np.random.seed(42)
            for day in range(100):
                t_max = 26 + 6 * np.sin(day / 50 * np.pi) + np.random.randn()
                t_min = 14 + 4 * np.sin(day / 50 * np.pi) + np.random.randn()
                et = 5 + 2 * np.sin(day / 50 * np.pi)

                model.update_daily(
                    t_max=t_max,
                    t_min=t_min,
                    et=et,
                    soil_moisture=scenario["soil_moisture"],
                    ECe=scenario["ECe"]
                )

            estimated_yield = model.estimate_yield()
            potential = CROP_CONFIGS.get(crop, CropConfig()).potential_yield_kg_ha

            print(f"{estimated_yield:>12.0f} kg", end="")

        print()

    print("-" * 65)

    print("\nScenario Descriptions:")
    for scenario in scenarios:
        print(f"  {scenario['name']}: {scenario['description']}")
        print(f"    (Moisture={scenario['soil_moisture']}, ECe={scenario['ECe']} dS/m)")


def sensitivity_analysis():
    """
    Perform sensitivity analysis on key parameters.
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: WHEAT YIELD")
    print("=" * 70)

    print("\n1. Yield Response to Soil Salinity (ECe):")
    print("-" * 50)

    ece_range = np.arange(0, 16, 2)

    for ece in ece_range:
        model = CropModel("wheat")

        for day in range(100):
            model.update_daily(
                t_max=26, t_min=14, et=5,
                soil_moisture=0.25, ECe=ece
            )

        yield_est = model.estimate_yield()
        bar_length = int(yield_est / 100)
        bar = "#" * bar_length

        print(f"ECe={ece:>4} dS/m: {yield_est:>6.0f} kg/ha {bar}")

    print("\n2. Yield Response to Average Soil Moisture:")
    print("-" * 50)

    moisture_range = np.arange(0.14, 0.34, 0.02)

    for moisture in moisture_range:
        model = CropModel("wheat")

        for day in range(100):
            model.update_daily(
                t_max=26, t_min=14, et=5,
                soil_moisture=moisture, ECe=3.0
            )

        yield_est = model.estimate_yield()
        bar_length = int(yield_est / 100)
        bar = "#" * bar_length

        print(f"Moisture={moisture:.2f}: {yield_est:>6.0f} kg/ha {bar}")


def main():
    """Main function to run all demonstrations."""

    print("\n" + "=" * 70)
    print("CROP YIELD PREDICTION - COMPREHENSIVE EXAMPLE")
    print("hetao_ag Library Demonstration")
    print("=" * 70)

    # Run each demonstration
    demonstrate_salt_tolerance()
    demonstrate_water_stress()
    demonstrate_combined_stress()
    demonstrate_phenology_tracking()
    run_yield_prediction()
    sensitivity_analysis()

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Salt stress follows the Maas-Hoffman model")
    print("  2. Water stress depends on soil moisture and crop sensitivity")
    print("  3. Combined stresses typically multiply (worst case)")
    print("  4. Phenology tracking helps time management decisions")
    print("  5. Yield prediction integrates all stress factors")


if __name__ == "__main__":
    main()
