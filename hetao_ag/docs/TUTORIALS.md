# hetao_ag Tutorials

Comprehensive tutorials for using the hetao_ag smart agriculture library.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Tutorial 1: Basic Soil and Water Management](#tutorial-1-basic-soil-and-water-management)
3. [Tutorial 2: Crop Growth Simulation](#tutorial-2-crop-growth-simulation)
4. [Tutorial 3: Remote Sensing Analysis](#tutorial-3-remote-sensing-analysis)
5. [Tutorial 4: Livestock Monitoring](#tutorial-4-livestock-monitoring)
6. [Tutorial 5: Farm Optimization](#tutorial-5-farm-optimization)
7. [Tutorial 6: Complete Workflow Integration](#tutorial-6-complete-workflow-integration)

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/hetao-college/hetao_ag.git
cd hetao_ag

# Install with all dependencies
pip install -e ".[full]"
```

### Verify Installation

```python
import hetao_ag
print(f"hetao_ag version: {hetao_ag.__version__}")

# Quick test
from hetao_ag.water import eto_penman_monteith, WeatherData
weather = WeatherData(t_mean=25, t_max=32, t_min=18, rh=55, u2=2.0, rs=22,
                      elevation=1050, latitude=40.8, doy=180)
print(f"Reference ET: {eto_penman_monteith(weather):.2f} mm/day")
```

### Import Conventions

```python
# Recommended import patterns
from hetao_ag.water import eto_penman_monteith, WeatherData
from hetao_ag.soil import SoilMoistureModel, SoilType
from hetao_ag.crop import CropModel, yield_reduction_salinity_crop
from hetao_ag.space import compute_ndvi, classify_vegetation_health
from hetao_ag.livestock import HealthMonitor
from hetao_ag.opt import optimize_crop_mix

# Or import entire modules
from hetao_ag import water, soil, crop, space, livestock, opt
```

---

## Tutorial 1: Basic Soil and Water Management

### Objective
Learn to model soil moisture dynamics and calculate irrigation requirements.

### Step 1: Create a Soil Moisture Model

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

# Create model with typical loam soil parameters
soil = SoilMoistureModel(
    field_capacity=0.32,      # Maximum water holding (m3/m3)
    wilting_point=0.12,       # Permanent wilting point (m3/m3)
    initial_moisture=0.25,    # Current moisture content
    root_depth_m=0.6,         # Root zone depth
    soil_type=SoilType.LOAM   # Soil texture class
)

print(f"Current moisture: {soil.moisture:.3f}")
print(f"Available water: {soil.moisture - soil.wilting_point:.3f}")
print(f"Stress factor: {soil.stress_factor:.3f}")
```

### Step 2: Simulate Daily Water Balance

```python
import numpy as np

# Simulate 30 days
for day in range(30):
    # Weather-driven ET (random for demo)
    daily_et = 4 + 2 * np.sin(day / 10) + np.random.randn() * 0.5
    daily_et = max(0, daily_et)

    # Check for rain (20% chance)
    rain = np.random.exponential(10) if np.random.random() < 0.2 else 0
    rain = min(rain, 30)  # Cap at 30mm

    # Update soil
    result = soil.step_day(
        rain_mm=rain,
        irrigation_mm=0,  # No irrigation yet
        et_mm=daily_et
    )

    if day % 7 == 0:  # Report weekly
        print(f"Day {day+1}: Moisture={result['moisture']:.3f}, "
              f"Rain={rain:.1f}mm, ET={daily_et:.1f}mm")
```

### Step 3: Determine Irrigation Needs

```python
from hetao_ag.water import IrrigationScheduler, ScheduleType

scheduler = IrrigationScheduler(
    method=ScheduleType.SOIL_MOISTURE,
    trigger_threshold=0.5,     # Start irrigation at 50% depletion
    max_application_mm=40,     # Max per event
    irrigation_efficiency=0.85 # 85% efficiency
)

# Get recommendation
rec = scheduler.recommend_by_moisture(
    current_moisture=soil.moisture,
    field_capacity=soil.field_capacity,
    wilting_point=soil.wilting_point
)

if rec.should_irrigate:
    print(f"IRRIGATION NEEDED: {rec.amount_mm:.1f} mm")
    print(f"Reason: {rec.reason}")
    print(f"Urgency: {rec.urgency}")
else:
    print(f"No irrigation needed: {rec.reason}")
```

### Step 4: Track Water Balance

```python
from hetao_ag.water import WaterBalance

wb = WaterBalance(
    initial_storage_mm=80,
    max_storage_mm=120,
    min_storage_mm=40
)

# Simulate with irrigation
for day in range(60):
    precip = 10 if day % 7 == 0 else 0
    et = 5
    irrig = 30 if day % 14 == 7 else 0

    record = wb.step_day(
        precip_mm=precip,
        irrig_mm=irrig,
        et_mm=et
    )

# Summary
summary = wb.get_summary()
print("\nWater Balance Summary:")
for key, value in summary.items():
    print(f"  {key}: {value:.1f}")
```

---

## Tutorial 2: Crop Growth Simulation

### Objective
Learn to simulate crop growth and predict yields under stress conditions.

### Step 1: Understanding Salt Stress

```python
from hetao_ag.crop import (
    yield_reduction_salinity_crop,
    classify_salt_tolerance,
    CROP_SALT_TOLERANCE
)

# Check different crops at ECe = 6 dS/m
crops = ["wheat", "maize", "cotton", "barley"]
ECe = 6.0

print(f"Relative yields at ECe = {ECe} dS/m:")
for crop in crops:
    rel_yield = yield_reduction_salinity_crop(ECe, crop)
    tolerance = classify_salt_tolerance(crop)
    print(f"  {crop}: {rel_yield*100:.1f}% ({tolerance})")
```

### Step 2: Track Phenology with GDD

```python
from hetao_ag.crop import PhenologyTracker, GrowthStage
import numpy as np

tracker = PhenologyTracker("wheat")

print("Tracking wheat phenology over 120 days:")
print("-" * 50)

np.random.seed(42)
for day in range(120):
    # Seasonal temperature pattern
    t_max = 22 + 10 * np.sin(day / 60 * np.pi) + np.random.randn() * 2
    t_min = 10 + 6 * np.sin(day / 60 * np.pi) + np.random.randn() * 2

    tracker.accumulate_gdd(t_max, t_min)

    if day % 20 == 0:
        print(f"Day {day:3d}: GDD={tracker.accumulated_gdd:>6.0f}, "
              f"Stage={tracker.current_stage.value:<12}, "
              f"Kc={tracker.get_kc_for_stage():.2f}")

print(f"\nFinal progress to maturity: {tracker.progress_to_maturity()*100:.1f}%")
```

### Step 3: Full Crop Growth Model

```python
from hetao_ag.crop import CropModel, CROP_CONFIGS
import numpy as np

# Create model
model = CropModel("wheat")

print("Simulating wheat growth for 100 days:")
print("-" * 60)

np.random.seed(42)
for day in range(100):
    # Daily weather
    t_max = 24 + 8 * np.sin(day / 50 * np.pi) + np.random.randn()
    t_min = 12 + 5 * np.sin(day / 50 * np.pi) + np.random.randn()
    et = 5 + 2 * np.sin(day / 50 * np.pi)

    result = model.update_daily(
        t_max=t_max,
        t_min=t_min,
        et=et,
        soil_moisture=0.25,  # Moderate moisture
        ECe=3.0              # Low-moderate salinity
    )

    if day % 25 == 0:
        print(f"Day {day:3d}: Stage={result['stage']:<12}, "
              f"Biomass={result['biomass_kg_ha']:>6.0f} kg/ha, "
              f"LAI={result['lai']:.2f}")

# Final yield
yield_est = model.estimate_yield()
potential = CROP_CONFIGS["wheat"].potential_yield_kg_ha

print(f"\nFinal Results:")
print(f"  Biomass: {model.accumulated_biomass:.0f} kg/ha")
print(f"  Estimated yield: {yield_est:.0f} kg/ha")
print(f"  Potential yield: {potential:.0f} kg/ha")
print(f"  Yield ratio: {yield_est/potential*100:.1f}%")
```

### Step 4: Stress Scenario Comparison

```python
scenarios = [
    ("Optimal", 0.28, 2.0),
    ("Moderate stress", 0.20, 5.0),
    ("High salinity", 0.25, 10.0),
    ("Water deficit", 0.16, 3.0),
]

print("Yield comparison under different stress scenarios:")
print("-" * 50)

for name, moisture, ece in scenarios:
    model = CropModel("wheat")

    for day in range(100):
        model.update_daily(
            t_max=25, t_min=14, et=5,
            soil_moisture=moisture, ECe=ece
        )

    yield_est = model.estimate_yield()
    print(f"{name:<20}: {yield_est:>6.0f} kg/ha")
```

---

## Tutorial 3: Remote Sensing Analysis

### Objective
Learn to calculate vegetation indices and classify crop conditions.

### Step 1: Calculate Vegetation Indices

```python
import numpy as np
from hetao_ag.space import (
    compute_ndvi, compute_savi, compute_evi,
    compute_lswi, compute_ndwi
)

# Simulated Sentinel-2 bands (4x4 pixel example)
np.random.seed(42)

# Different land cover types
# Row 0-1: Dense vegetation, Row 2-3: Sparse/bare
vegetation = np.array([
    [0.8, 0.7, 0.75, 0.85],
    [0.7, 0.65, 0.7, 0.8],
    [0.3, 0.25, 0.35, 0.4],
    [0.2, 0.15, 0.25, 0.3]
])

# Generate bands based on vegetation cover
blue = (800 + 200 * (1 - vegetation)).astype(np.uint16)
green = (700 + 300 * (1 - vegetation)).astype(np.uint16)
red = (1200 - 700 * vegetation).astype(np.uint16)
nir = (1500 + 2500 * vegetation).astype(np.uint16)
swir = (2000 - 500 * vegetation).astype(np.uint16)

# Calculate indices
ndvi = compute_ndvi(red, nir)
savi = compute_savi(red, nir, L=0.5)
evi = compute_evi(blue, red, nir)

print("Vegetation Indices (4x4 scene):")
print("\nNDVI (higher = more vegetation):")
print(np.round(ndvi, 3))

print("\nSAVI (soil-adjusted):")
print(np.round(savi, 3))

print("\nEVI (enhanced):")
print(np.round(evi, 3))
```

### Step 2: Classify Vegetation Health

```python
from hetao_ag.space import classify_vegetation_health

print("\nVegetation Health Classification:")
print("-" * 40)

ndvi_values = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]

for ndvi_val in ndvi_values:
    health = classify_vegetation_health(ndvi_val)
    print(f"NDVI = {ndvi_val:.2f}: {health}")
```

### Step 3: Process Time Series for Phenology

```python
from hetao_ag.space import PhenologyClassifier, temporal_smoothing

# Create 12-month NDVI time series (30x30 pixels)
np.random.seed(42)
n_months = 12
height, width = 30, 30

# Seasonal pattern (Northern hemisphere growing season)
t = np.linspace(0, 2 * np.pi, n_months)
base_curve = 0.3 + 0.35 * np.sin(t)

# Create series with spatial variation
ndvi_series = np.zeros((n_months, height, width))
for i in range(height):
    for j in range(width):
        shift = (i + j) / 60 * np.pi  # Spatial phenology variation
        noise = np.random.randn(n_months) * 0.05
        ndvi_series[:, i, j] = np.clip(base_curve + noise +
                                        np.sin(t + shift) * 0.1, 0, 1)

# Smooth the series
smoothed = temporal_smoothing(ndvi_series, window=3)

# Create classifier
classifier = PhenologyClassifier(smoothed)

# Classify crops
crop_map = classifier.classify_crops(n_classes=3)

print("Crop Classification Results:")
for class_id in range(4):
    count = np.sum(crop_map == class_id)
    pct = 100 * count / crop_map.size
    print(f"  Class {class_id}: {count} pixels ({pct:.1f}%)")

# Extract features for a sample pixel
features = classifier.extract_features(15, 15)
print(f"\nSample pixel (15,15) features:")
print(f"  Peak NDVI: {features.peak_value:.3f}")
print(f"  Peak month: {features.peak_time + 1}")
print(f"  Season length: {features.end_of_season - features.start_of_season} months")
```

---

## Tutorial 4: Livestock Monitoring

### Objective
Learn to monitor animal health and detect abnormal behaviors.

### Step 1: Animal Detection

```python
from hetao_ag.livestock import AnimalDetector

# Create detector (uses simulation mode without PyTorch)
detector = AnimalDetector(confidence_threshold=0.5)

# Detect animals in image
detections = detector.detect("farm_image.jpg")

print("Animal Detection Results:")
for det in detections:
    print(f"  {det.label}: confidence={det.confidence:.2%}, "
          f"bbox={det.bbox}")

# Count animals
counts = detector.count_animals("farm_image.jpg")
print(f"\nAnimal counts: {counts}")
```

### Step 2: Behavior Classification

```python
from hetao_ag.livestock import BehaviorClassifier, AnimalBehavior

classifier = BehaviorClassifier()

# Test different motion/head position combinations
test_cases = [
    (0.05, "down", "Low motion, head down"),
    (0.2, "level", "Moderate motion, head level"),
    (0.3, "down", "Active, head down"),
    (0.7, None, "High motion"),
]

print("Behavior Classification:")
print("-" * 50)

for motion, head, description in test_cases:
    behavior = classifier.classify_from_motion(motion, head)
    print(f"{description:<30} -> {behavior.value}")
```

### Step 3: Health Monitoring System

```python
from hetao_ag.livestock import HealthMonitor, HerdHealthMonitor
import numpy as np

# Create individual monitor
monitor = HealthMonitor("cow_001")

print("Building baseline over 7 days...")

# Build baseline with normal data
np.random.seed(42)
for day in range(7):
    monitor.update_activity(100 + np.random.randn() * 5)
    monitor.update_feeding_time(240 + np.random.randn() * 10)

print(f"Activity baseline: {monitor.activity_baseline:.1f}")
print(f"Feeding baseline: {monitor.feeding_baseline:.1f}")

# Simulate abnormal day
print("\nSimulating abnormal readings...")
monitor.update_activity(60)  # Low activity
monitor.update_temperature(40.2)  # Fever

# Check health
alerts = monitor.check_health()

print(f"\nHealth Status: {monitor.get_status().value}")
print(f"Active Alerts: {len(alerts)}")
for alert in alerts:
    print(f"  [{alert.severity.value}] {alert.alert_type.value}: {alert.message}")
```

### Step 4: Herd-Level Monitoring

```python
# Create herd monitor
herd = HerdHealthMonitor()

# Add animals
for i in range(10):
    herd.add_animal(f"cow_{i:03d}")

# Update with data
np.random.seed(42)
for animal_id, monitor in herd.animals.items():
    # Baseline
    for _ in range(7):
        monitor.update_activity(100 + np.random.randn() * 10)
        monitor.update_feeding_time(240 + np.random.randn() * 15)

    # Current day (some abnormal)
    if np.random.random() < 0.2:  # 20% chance of issue
        monitor.update_activity(50 + np.random.randn() * 10)
    else:
        monitor.update_activity(95 + np.random.randn() * 10)

# Check all animals
all_alerts = herd.check_all()

# Summary
summary = herd.get_summary()
print("\nHerd Health Summary:")
for status, count in summary.items():
    print(f"  {status}: {count} animals")
```

---

## Tutorial 5: Farm Optimization

### Objective
Learn to optimize resource allocation using linear programming and genetic algorithms.

### Step 1: Optimize Crop Mix

```python
from hetao_ag.opt import optimize_crop_mix

# Define crops with economic parameters
crops = [
    {
        "name": "wheat",
        "profit_per_ha": 500,    # $/ha
        "water_per_ha": 3000,    # m3/ha
    },
    {
        "name": "maize",
        "profit_per_ha": 600,
        "water_per_ha": 5000,
    },
    {
        "name": "cotton",
        "profit_per_ha": 800,
        "water_per_ha": 6000,
    },
    {
        "name": "alfalfa",
        "profit_per_ha": 400,
        "water_per_ha": 2000,
    },
]

# Constraints
total_land = 100    # hectares
total_water = 350000  # m3

# Optimize
solution = optimize_crop_mix(crops, total_land, total_water)

print("Optimal Crop Allocation:")
print("-" * 40)
total_profit = 0
total_water_used = 0

for crop in crops:
    area = solution.get(crop["name"], 0)
    if area and area > 0.1:
        profit = area * crop["profit_per_ha"]
        water = area * crop["water_per_ha"]
        total_profit += profit
        total_water_used += water
        print(f"  {crop['name']:<10}: {area:>6.1f} ha "
              f"(profit: ${profit:>8,.0f}, water: {water:>8,.0f} m3)")

print("-" * 40)
print(f"  Total:      {sum(v for v in solution.values() if v):>6.1f} ha")
print(f"  Profit:     ${total_profit:>8,.0f}")
print(f"  Water used: {total_water_used:>8,.0f} m3 ({100*total_water_used/total_water:.1f}%)")
```

### Step 2: Genetic Algorithm Optimization

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

# Define a custom optimization problem
# Maximize: f(x) = -(x1^2 + x2^2 + x3^2) + 10
# This is a simple sphere function (minimum at origin)

def fitness(x):
    return -(x[0]**2 + x[1]**2 + x[2]**2) + 10

# Configure GA
config = GAConfig(
    population_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=2
)

# Create and run optimizer
optimizer = GeneticOptimizer(
    fitness_func=fitness,
    n_vars=3,
    bounds=[(-5, 5), (-5, 5), (-5, 5)],
    config=config
)

result = optimizer.optimize()

print("Genetic Algorithm Results:")
print(f"  Best solution: [{', '.join(f'{x:.4f}' for x in result.best_solution)}]")
print(f"  Best fitness: {result.best_fitness:.6f}")
print(f"  Generations: {result.generations_run}")

# Verify (should be close to [0, 0, 0] with fitness ~10)
distance = sum(x**2 for x in result.best_solution) ** 0.5
print(f"  Distance from optimum: {distance:.6f}")
```

### Step 3: Scenario Analysis

```python
from hetao_ag.opt import ScenarioEvaluator, FarmScenario

# Define crop parameters
crop_params = {
    "wheat": {
        "yield_kg_ha": 6000,
        "water_need_mm": 400,
        "price_per_kg": 0.8,
        "cost_per_ha": 1200
    },
    "maize": {
        "yield_kg_ha": 10000,
        "water_need_mm": 600,
        "price_per_kg": 0.6,
        "cost_per_ha": 1500
    },
}

# Create evaluator
evaluator = ScenarioEvaluator(
    crop_params=crop_params,
    total_land=100,
    total_water=500000
)

# Evaluate different scenarios
scenarios = [
    ("All Wheat", {"wheat": 100}, 450),
    ("All Maize", {"maize": 100}, 650),
    ("50-50 Mix", {"wheat": 50, "maize": 50}, 500),
    ("70-30 Wheat Heavy", {"wheat": 70, "maize": 30}, 480),
]

print("Scenario Comparison:")
print("-" * 70)
print(f"{'Scenario':<20} {'Profit':>15} {'WUE':>15} {'Irrigation':>15}")
print("-" * 70)

for name, areas, irrigation in scenarios:
    scenario = evaluator.evaluate_scenario(name, areas, irrigation)
    print(f"{name:<20} ${scenario.total_profit:>14,.0f} "
          f"{scenario.water_use_efficiency:>14.3f} {irrigation:>14} mm")

# Compare
best = evaluator.compare_scenarios()
print("-" * 70)
print(f"Best profit: {best['best_profit'].name}")
print(f"Best water efficiency: {best['best_water_efficiency'].name}")
```

---

## Tutorial 6: Complete Workflow Integration

### Objective
Learn to integrate all modules for a complete precision agriculture workflow.

### Complete Example

```python
"""
Complete Precision Agriculture Workflow
Integrates soil, water, crop, and optimization modules.
"""

import numpy as np
from hetao_ag.core import get_logger
from hetao_ag.water import eto_penman_monteith, WeatherData, WaterBalance
from hetao_ag.soil import SoilMoistureModel, SalinityModel
from hetao_ag.crop import CropModel
from hetao_ag.water import IrrigationScheduler, ScheduleType

def run_season_simulation():
    """Simulate a complete growing season."""

    # Initialize logger
    logger = get_logger("integrated_simulation")
    logger.info("Starting integrated simulation")

    # Initialize models
    soil = SoilMoistureModel(
        field_capacity=0.32,
        wilting_point=0.12,
        initial_moisture=0.28
    )

    salinity = SalinityModel(initial_ECe=3.0)
    water_balance = WaterBalance(initial_storage_mm=90)
    crop = CropModel("wheat")
    scheduler = IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE)

    # Simulation parameters
    season_length = 120
    results = []

    print("Running 120-day wheat simulation...")
    print("-" * 60)

    np.random.seed(42)

    for day in range(season_length):
        # Generate weather
        doy = 90 + day  # Start April 1
        seasonal = np.sin((doy - 80) / 365 * 2 * np.pi)

        t_max = 20 + 12 * seasonal + np.random.randn() * 2
        t_min = 8 + 8 * seasonal + np.random.randn() * 2
        rh = 55 - 10 * seasonal + np.random.randn() * 5
        rs = 18 + 6 * seasonal + np.random.randn()

        weather = WeatherData(
            t_mean=(t_max + t_min) / 2,
            t_max=t_max, t_min=t_min,
            rh=max(30, min(90, rh)),
            u2=2.0, rs=max(5, rs),
            elevation=1050, latitude=40.8, doy=doy
        )

        # Calculate ET
        et0 = eto_penman_monteith(weather)

        # Check irrigation
        rec = scheduler.recommend_by_moisture(
            soil.moisture, soil.field_capacity, soil.wilting_point
        )
        irrigation = rec.amount_mm if rec.should_irrigate else 0

        # Precipitation (15% chance)
        precipitation = (np.random.exponential(12)
                        if np.random.random() < 0.15 else 0)
        precipitation = min(precipitation, 35)

        # Update soil
        if precipitation > 0:
            soil.add_water(precipitation)
            water_balance.add_precipitation(precipitation)

        if irrigation > 0:
            soil.add_water(irrigation)
            water_balance.add_irrigation(irrigation)

        # ET removal
        actual_et = soil.remove_water(et0)
        water_balance.remove_et(actual_et)

        # Update crop
        crop_result = crop.update_daily(
            t_max=t_max, t_min=t_min, et=et0,
            soil_moisture=soil.moisture, ECe=salinity.ECe
        )

        # Store results
        results.append({
            'day': day,
            'et0': et0,
            'moisture': soil.moisture,
            'irrigation': irrigation,
            'precipitation': precipitation,
            'biomass': crop.accumulated_biomass,
            'stage': crop_result['stage']
        })

        # Weekly report
        if (day + 1) % 30 == 0:
            print(f"Day {day+1}: Stage={crop_result['stage']}, "
                  f"Biomass={crop.accumulated_biomass:.0f} kg/ha, "
                  f"Moisture={soil.moisture:.3f}")

    # Final summary
    print("\n" + "=" * 60)
    print("SEASON SUMMARY")
    print("=" * 60)

    total_irrigation = sum(r['irrigation'] for r in results)
    total_precipitation = sum(r['precipitation'] for r in results)
    estimated_yield = crop.estimate_yield()

    print(f"Total irrigation: {total_irrigation:.0f} mm")
    print(f"Total precipitation: {total_precipitation:.0f} mm")
    print(f"Final biomass: {crop.accumulated_biomass:.0f} kg/ha")
    print(f"Estimated yield: {estimated_yield:.0f} kg/ha")

    # Water use efficiency
    total_water = total_irrigation + total_precipitation
    wue = estimated_yield / (total_water * 10)  # kg/m3
    print(f"Water use efficiency: {wue:.2f} kg/m3")

    return results

if __name__ == "__main__":
    results = run_season_simulation()
```

---

## Next Steps

After completing these tutorials, you should be able to:

1. Model soil-water-plant systems
2. Predict crop yields under various stress conditions
3. Analyze remote sensing data for vegetation monitoring
4. Implement livestock health monitoring
5. Optimize farm resource allocation
6. Integrate all components for precision agriculture

### Additional Resources

- API Reference: See `docs/API_REFERENCE.md`
- Architecture Guide: See `docs/ARCHITECTURE.md`
- Example Scripts: See `examples/` directory

### Getting Help

- GitHub Issues: Report bugs or request features
- Documentation: Check module docstrings with `help(function)`
- Examples: Run `python examples/demo.py` for comprehensive demos
