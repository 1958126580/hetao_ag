# Hetao Smart Agriculture Library (hetao_ag)

<p align="center">
  <strong>A Comprehensive Python Library for Smart Agriculture and Livestock Management</strong>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#modules">Modules</a> |
  <a href="#documentation">Documentation</a> |
  <a href="#examples">Examples</a>
</p>

---

## Overview

**hetao_ag** is a comprehensive Python library designed for smart agriculture and livestock management, specifically tailored for arid and semi-arid regions like the Hetao Irrigation District. The library provides integrated solutions spanning soil modeling, water cycle management, crop growth simulation, livestock monitoring, remote sensing analysis, and farm optimization.

### Key Highlights

- **Scientific Rigor**: Built on internationally recognized models (FAO-56, Maas-Hoffman, van Genuchten)
- **SI Unit System**: Automatic unit conversion to prevent unit confusion errors
- **Modular Design**: Each module works independently or in combination
- **Chinese-First Documentation**: Complete Chinese comments and documentation
- **Production Ready**: Comprehensive test coverage and real-world validation

---

## Features

| Module | Functionality | Use Cases |
|--------|--------------|-----------|
| **core** | Unit system, configuration management, logging | Scientific computing standardization, experiment tracking |
| **soil** | Soil moisture/salinity modeling, sensor calibration | Soil moisture monitoring, saline land management |
| **water** | FAO-56 evapotranspiration, water balance, irrigation scheduling | Precision irrigation, water resource management |
| **crop** | Crop growth simulation, stress response, phenology tracking | Yield prediction, planting decisions |
| **livestock** | Animal detection, behavior classification, health monitoring | Smart ranching, disease early warning |
| **space** | Remote sensing indices, image processing, phenology classification | Crop monitoring, vegetation mapping |
| **opt** | Linear programming, genetic algorithms, scenario analysis | Resource optimization, decision support |

---

## Installation

### System Requirements

- Python 3.10 or higher
- Operating System: Windows / Linux / macOS

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/1958126580/hetao_ag.git
cd hetao_ag

# Install basic version (numpy only)
pip install -e .
```

### Full Installation

```bash
# Install with all optional dependencies
pip install -e ".[full]"
```

### Modular Installation

```bash
# Remote sensing support (rasterio, etc.)
pip install -e ".[space]"

# Livestock AI support (torch, opencv)
pip install -e ".[livestock]"

# Optimization support (pulp, scipy)
pip install -e ".[opt]"
```

### Verify Installation

```python
import hetao_ag as hag
print(hag.__version__)  # Output: 1.0.0
```

---

## Quick Start

### 5-Minute Tutorial

```python
import hetao_ag as hag
import numpy as np

# ============================================
# 1. Calculate Reference Evapotranspiration
# ============================================
from hetao_ag.water import eto_penman_monteith, WeatherData

weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)
et0 = eto_penman_monteith(weather)
print(f"Reference ET (ET0): {et0:.2f} mm/day")

# ============================================
# 2. Soil Moisture Simulation
# ============================================
from hetao_ag.soil import SoilMoistureModel, SoilType

soil = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

# Simulate daily water balance
result = soil.step_day(rain_mm=15, irrigation_mm=0, et_mm=5)
print(f"Soil moisture: {result['moisture']:.3f}")
print(f"Water stress factor: {soil.stress_factor:.3f}")

# ============================================
# 3. Crop Salinity Stress Assessment
# ============================================
from hetao_ag.crop import yield_reduction_salinity_crop, classify_salt_tolerance

ECe = 6.0  # Soil electrical conductivity (dS/m)
rel_yield = yield_reduction_salinity_crop(ECe, crop="wheat")
tolerance = classify_salt_tolerance("wheat")

print(f"Wheat relative yield at ECe={ECe}: {rel_yield*100:.1f}%")
print(f"Wheat salt tolerance: {tolerance}")

# ============================================
# 4. Remote Sensing Vegetation Index
# ============================================
from hetao_ag.space import compute_ndvi, classify_vegetation_health

red = np.array([[120, 130], [110, 90]], dtype=np.uint16)
nir = np.array([[200, 210], [180, 160]], dtype=np.uint16)

ndvi = compute_ndvi(red, nir)
print(f"NDVI mean: {ndvi.mean():.3f}")
print(f"Vegetation status: {classify_vegetation_health(ndvi.mean())}")

# ============================================
# 5. Irrigation Scheduling
# ============================================
from hetao_ag.water import IrrigationScheduler, ScheduleType

scheduler = IrrigationScheduler(
    method=ScheduleType.SOIL_MOISTURE,
    trigger_threshold=0.5
)

recommendation = scheduler.recommend_by_moisture(
    current_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12
)

if recommendation.should_irrigate:
    print(f"Irrigation recommended: {recommendation.amount_mm:.1f} mm")
    print(f"Reason: {recommendation.reason}")
    print(f"Urgency: {recommendation.urgency}")
```

---

## Modules

### Core Module (`hetao_ag.core`)

The foundation of the library providing essential infrastructure.

#### Unit System

```python
from hetao_ag.core import Quantity, Unit, meters, hectares, celsius

# Create physical quantities
length = meters(100)
area = hectares(50)
temp = celsius(25)

# Unit conversion
length_km = length.to(Unit.KILOMETER)
print(length_km)  # 0.1 km

# Arithmetic operations
total = meters(100) + Quantity(0.5, Unit.KILOMETER)
print(total)  # 600 m
```

#### Configuration Management

```python
from hetao_ag.core import ConfigManager, create_default_config

config = ConfigManager(defaults=create_default_config())

# Nested key access
fc = config.get("soil.field_capacity")
et_method = config.get("water.et_method", default="penman_monteith")

# Dynamic configuration
config.set("irrigation.efficiency", 0.90)
```

#### Logging

```python
from hetao_ag.core import get_logger

logger = get_logger("experiment")
logger.info("Starting simulation", soil_ec=4.5, irrigation=True)
logger.log_experiment_start("Yield Prediction", parameters={"model": "CropModel"})
```

---

### Soil Module (`hetao_ag.soil`)

Comprehensive soil water and salinity modeling.

#### Soil Moisture Model

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

model = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

# Multi-day simulation
for day in range(10):
    result = model.step_day(rain_mm=5 if day % 3 == 0 else 0, et_mm=4)
    print(f"Day {day+1}: moisture={result['moisture']:.3f}")

# Key properties
print(f"Current stress factor: {model.stress_factor:.2f}")
print(f"Irrigation needed: {model.irrigation_need_mm:.1f} mm")
```

#### Salinity Model

```python
from hetao_ag.soil import SalinityModel, classify_soil_salinity

model = SalinityModel(initial_ECe=4.0, root_depth_m=0.3)

# Irrigation with saline water
model.irrigate(amount_mm=60, ec_water=1.5)
print(f"EC after irrigation: {model.ECe:.2f} dS/m")

# Leaching
model.leach(drainage_mm=40)
print(f"EC after leaching: {model.ECe:.2f} dS/m")
print(f"Classification: {classify_soil_salinity(model.ECe)}")
```

#### Sensor Calibration

```python
from hetao_ag.soil import SensorCalibrator
import numpy as np

calibrator = SensorCalibrator()

# Calibration with field data
raw_readings = np.array([300, 450, 600, 750])
ground_truth = np.array([0.10, 0.20, 0.30, 0.40])

result = calibrator.auto_calibrate(raw_readings, ground_truth)
print(f"Best method: {result.method.value}, R²={result.r_squared:.4f}")

# Apply calibration
calibrated = result.apply(500)
print(f"Raw 500 -> Calibrated {calibrated:.3f}")
```

---

### Water Module (`hetao_ag.water`)

FAO-56 compliant water cycle management.

#### Evapotranspiration Calculation

```python
from hetao_ag.water import (
    eto_penman_monteith, eto_hargreaves,
    WeatherData, crop_coefficient, etc_crop,
    extraterrestrial_radiation
)

# Full weather data - Penman-Monteith
weather = WeatherData(
    t_mean=25, t_max=32, t_min=18,
    rh=55, u2=2.0, rs=22,
    elevation=1050, latitude=40.8, doy=180
)
et0_pm = eto_penman_monteith(weather)

# Limited data - Hargreaves
Ra = extraterrestrial_radiation(latitude=40.8, doy=180)
et0_hg = eto_hargreaves(25, 32, 18, Ra)

print(f"Penman-Monteith ET0: {et0_pm:.2f} mm/day")
print(f"Hargreaves ET0: {et0_hg:.2f} mm/day")

# Crop evapotranspiration
kc = crop_coefficient("mid", "wheat")
etc = etc_crop(et0_pm, kc)
print(f"Wheat ETc (mid-season): {etc:.2f} mm/day")
```

#### Water Balance

```python
from hetao_ag.water import WaterBalance

wb = WaterBalance(
    initial_storage_mm=80,
    max_storage_mm=120,
    min_storage_mm=40
)

# 10-day simulation
weather_data = [
    {"precip": 0, "et": 5}, {"precip": 15, "et": 4},
    {"precip": 0, "et": 6}, {"precip": 0, "et": 5},
    {"precip": 0, "et": 5}, {"precip": 30, "et": 3},
    {"precip": 0, "et": 4}, {"precip": 0, "et": 6},
    {"precip": 0, "et": 5}, {"precip": 0, "et": 5},
]

for i, w in enumerate(weather_data, 1):
    record = wb.step_day(precip_mm=w["precip"], et_mm=w["et"])
    print(f"Day {i}: storage={record.soil_moisture:.1f}mm")

# Summary
summary = wb.get_summary()
print(f"\nTotal precipitation: {summary['total_precipitation_mm']:.1f} mm")
print(f"Total ET: {summary['total_et_mm']:.1f} mm")
```

---

### Crop Module (`hetao_ag.crop`)

Crop growth simulation and stress analysis.

#### Stress Response Models

```python
from hetao_ag.crop import (
    yield_reduction_salinity_crop,
    water_stress_from_moisture,
    combined_stress_factor,
    yield_with_stress,
    CROP_SALT_TOLERANCE
)

# Salinity stress
ECe = 8.0
for crop in ["wheat", "maize", "cotton", "barley"]:
    rel_yield = yield_reduction_salinity_crop(ECe, crop)
    print(f"{crop}: {rel_yield*100:.1f}% yield at ECe={ECe} dS/m")

# Water stress
ks_water = water_stress_from_moisture(
    soil_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12
)

# Combined stress
ks_combined = combined_stress_factor(ks_water, rel_yield)
actual_yield = yield_with_stress(
    potential_yield=6000,  # kg/ha
    water_stress=ks_water,
    salinity_stress=rel_yield
)
print(f"\nActual yield: {actual_yield:.0f} kg/ha")
```

#### Phenology Tracking

```python
from hetao_ag.crop import PhenologyTracker, GrowthStage

tracker = PhenologyTracker("wheat")

# Accumulate growing degree days
import numpy as np
np.random.seed(42)

for day in range(100):
    t_max = 25 + 8 * np.sin(day / 100 * np.pi) + np.random.randn()
    t_min = t_max - 10
    tracker.accumulate_gdd(t_max, t_min)

    if day % 20 == 0:
        print(f"Day {day}: GDD={tracker.accumulated_gdd:.0f}, "
              f"Stage={tracker.current_stage.value}")

print(f"\nProgress to maturity: {tracker.progress_to_maturity()*100:.1f}%")
print(f"Current Kc: {tracker.get_kc_for_stage()}")
```

#### Full Crop Model

```python
from hetao_ag.crop import CropModel

model = CropModel("wheat")

# Growing season simulation
for day in range(120):
    result = model.update_daily(
        t_max=25, t_min=15, et=5,
        soil_moisture=0.25,
        field_capacity=0.32,
        wilting_point=0.12,
        ECe=3.0
    )

print(f"Final biomass: {model.accumulated_biomass:.0f} kg/ha")
print(f"Estimated yield: {model.estimate_yield():.0f} kg/ha")
print(f"Average stress: {np.mean(model.stress_history):.3f}")
```

---

### Livestock Module (`hetao_ag.livestock`)

Smart livestock monitoring and health management.

#### Animal Detection

```python
from hetao_ag.livestock import AnimalDetector

detector = AnimalDetector(
    model_name="yolov5s",
    confidence_threshold=0.5
)

# Detect animals in image
detections = detector.detect("farm_image.jpg")
for det in detections:
    print(f"Detected {det.label} at {det.bbox}, confidence={det.confidence:.2f}")

# Count by species
counts = detector.count_animals("farm_image.jpg")
print(f"Animal counts: {counts}")
```

#### Behavior Classification

```python
from hetao_ag.livestock import BehaviorClassifier, AnimalBehavior

classifier = BehaviorClassifier()

# Classify from motion features
behavior = classifier.classify_from_motion(
    motion_magnitude=0.3,
    head_position="down"
)
print(f"Detected behavior: {behavior.value}")
```

#### Health Monitoring

```python
from hetao_ag.livestock import HealthMonitor, HerdHealthMonitor

# Individual monitoring
monitor = HealthMonitor("cow_001")

# Build baseline (7 days of normal data)
import numpy as np
for _ in range(7):
    monitor.update_activity(100 + np.random.randn() * 5)
    monitor.update_feeding_time(240 + np.random.randn() * 10)

# Detect anomaly
monitor.update_activity(60)  # Reduced activity
monitor.update_temperature(40.2)  # Fever

alerts = monitor.check_health()
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.alert_type.value}: {alert.message}")

# Herd-level monitoring
herd = HerdHealthMonitor()
for i in range(10):
    herd.add_animal(f"cow_{i:03d}")

summary = herd.get_summary()
print(f"Herd health summary: {summary}")
```

---

### Space Module (`hetao_ag.space`)

Remote sensing and vegetation analysis.

#### Spectral Indices

```python
from hetao_ag.space import (
    compute_ndvi, compute_savi, compute_evi,
    compute_lswi, compute_ndwi,
    classify_vegetation_health
)
import numpy as np

# Sample multispectral data
blue = np.random.randint(50, 150, (100, 100), dtype=np.uint16)
green = np.random.randint(80, 180, (100, 100), dtype=np.uint16)
red = np.random.randint(100, 200, (100, 100), dtype=np.uint16)
nir = np.random.randint(150, 250, (100, 100), dtype=np.uint16)
swir = np.random.randint(100, 200, (100, 100), dtype=np.uint16)

# Calculate indices
ndvi = compute_ndvi(red, nir)
savi = compute_savi(red, nir, L=0.5)
evi = compute_evi(blue, red, nir)
lswi = compute_lswi(nir, swir)

print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
print(f"SAVI mean: {savi.mean():.3f}")
print(f"EVI mean: {evi.mean():.3f}")
```

#### Phenology Classification

```python
from hetao_ag.space import PhenologyClassifier, temporal_smoothing

# Time series NDVI data (12 months, 100x100 pixels)
ndvi_series = np.random.rand(12, 100, 100) * 0.5 + 0.2

# Smooth time series
smoothed = temporal_smoothing(ndvi_series, window=3)

# Classify crops
classifier = PhenologyClassifier(smoothed)
crop_map = classifier.classify_crops(n_classes=3)

# Extract pixel features
features = classifier.extract_features(50, 50)
print(f"Peak NDVI: {features.peak_value:.3f}")
print(f"Peak time: month {features.peak_time + 1}")
print(f"Season length: {features.end_of_season - features.start_of_season} months")
```

---

### Optimization Module (`hetao_ag.opt`)

Resource optimization and decision support.

#### Linear Programming

```python
from hetao_ag.opt import LinearOptimizer, optimize_crop_mix

# Define crops with economic parameters
crops = [
    {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
    {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
    {"name": "alfalfa", "profit_per_ha": 400, "water_per_ha": 2000},
]

# Optimize allocation
solution = optimize_crop_mix(
    crops,
    total_land=100,  # hectares
    total_water=300000  # m³
)

print("Optimal crop mix:")
for crop, area in solution.items():
    print(f"  {crop}: {area:.1f} ha")
```

#### Genetic Algorithm

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

# Define optimization problem
def my_fitness(x):
    # Maximize profit while minimizing water use
    profit = x[0] * 500 + x[1] * 600  # wheat, maize areas
    water = x[0] * 3000 + x[1] * 5000
    water_penalty = max(0, water - 300000) * 0.01
    return profit - water_penalty

optimizer = GeneticOptimizer(
    fitness_func=my_fitness,
    n_vars=2,
    bounds=[(0, 100), (0, 100)],
    config=GAConfig(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
)

result = optimizer.optimize()
print(f"Best solution: wheat={result.best_solution[0]:.1f}ha, "
      f"maize={result.best_solution[1]:.1f}ha")
print(f"Best fitness: {result.best_fitness:.0f}")
```

#### Scenario Analysis

```python
from hetao_ag.opt import ScenarioEvaluator

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
    }
}

evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=500000)

# Evaluate different scenarios
s1 = evaluator.evaluate_scenario("All Wheat", {"wheat": 100}, 450)
s2 = evaluator.evaluate_scenario("All Maize", {"maize": 100}, 650)
s3 = evaluator.evaluate_scenario("Mixed", {"wheat": 50, "maize": 50}, 500)

# Compare
comparison = evaluator.compare_scenarios()
print(f"Best profit scenario: {comparison['best_profit'].name}")
print(f"Best water efficiency: {comparison['best_water_efficiency'].name}")
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation for all modules
- **[User Manual](docs/USER_MANUAL.md)**: Comprehensive usage guide with examples

### Getting Help

```python
# View function documentation
from hetao_ag.water import eto_penman_monteith
help(eto_penman_monteith)

# List module contents
from hetao_ag import soil
print(dir(soil))
```

---

## Examples

Example scripts are provided in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `demo.py` | Comprehensive demonstration of all modules |
| `example_core.py` | Unit system, configuration, and logging |
| `example_soil.py` | Soil moisture and salinity modeling |
| `example_water.py` | Evapotranspiration and irrigation |
| `example_crop.py` | Crop growth and stress simulation |
| `example_livestock.py` | Animal detection and health monitoring |
| `example_space.py` | Remote sensing indices and classification |
| `example_opt.py` | Optimization and scenario analysis |

### Running Examples

```bash
# Run all examples
python examples/demo.py

# Run specific module examples
python examples/example_soil.py
python examples/example_water.py
```

---

## Best Practices

### 1. Always Use the Unit System

```python
# Recommended
from hetao_ag.core import meters, hectares
area = hectares(50)

# Not recommended
area = 50  # Units unclear
```

### 2. Configuration Over Hard-Coding

```python
# Recommended
config = ConfigManager(config_file="farm_config.yaml")
fc = config.get("soil.field_capacity")

# Not recommended
fc = 0.32  # Hard-coded values are hard to maintain
```

### 3. Log Experiment Parameters

```python
logger = get_logger("experiment")
logger.log_experiment_start(
    "Irrigation Optimization",
    parameters={"method": "deficit", "target_etc": 0.8},
    random_seed=42
)
```

### 4. Validate Model Results

```python
from hetao_ag.core import validate_model
result = validate_model(observed, predicted)
print(f"RMSE: {result.rmse:.4f}, R²: {result.r_squared:.4f}")
```

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use hetao_ag in your research, please cite:

```bibtex
@software{hetao_ag,
  author = {Hetao College},
  title = {hetao_ag: A Smart Agriculture Python Library},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/1958126580/hetao_ag}
}
```

---

## Contact

- **Author**: Hetao College
- **Email**: 1958126580@qq.com
- **Repository**: https://github.com/1958126580/hetao_ag

---

<p align="center">
  <strong>hetao_ag v1.0.0</strong><br>
  Smart Agriculture for a Sustainable Future
</p>
