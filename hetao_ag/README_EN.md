# Hetao Smart Agriculture Library (hetao_ag)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](setup.py)

A comprehensive Python library for smart agriculture and animal husbandry, designed specifically for precision farming in arid and semi-arid regions like the Hetao Irrigation District. The library provides complete solutions for soil modeling, water management, crop growth simulation, livestock monitoring, remote sensing analysis, and farm optimization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
  - [Core Module](#core-module)
  - [Soil Module](#soil-module)
  - [Water Module](#water-module)
  - [Crop Module](#crop-module)
  - [Livestock Module](#livestock-module)
  - [Space Module](#space-module)
  - [Optimization Module](#optimization-module)
- [Architecture](#architecture)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**hetao_ag** is a modular Python library that provides scientifically-validated tools for agricultural decision support. It implements internationally recognized models including:

- **FAO-56 Penman-Monteith** for evapotranspiration calculation
- **Maas-Hoffman Model** for crop salt tolerance
- **Van Genuchten Equations** for soil hydraulic properties
- **Growing Degree Days (GDD)** for phenology tracking

The library is designed with the following principles:
1. **SI Unit Standard**: All calculations use international standard units
2. **Scientific Rigor**: Based on peer-reviewed agricultural science
3. **Modular Design**: Use modules independently or in combination
4. **Production Ready**: Comprehensive testing and error handling

---

## Features

| Module | Functionality | Use Cases |
|--------|--------------|-----------|
| **core** | Unit system, configuration, logging | Scientific computation standardization |
| **soil** | Moisture/salinity modeling, sensor calibration | Soil monitoring, saline land management |
| **water** | FAO-56 ET, water balance, irrigation scheduling | Precision irrigation, water resources |
| **crop** | Growth simulation, stress response, phenology | Yield prediction, planting decisions |
| **livestock** | Animal detection, behavior analysis, health | Smart ranching, disease early warning |
| **space** | Spectral indices, image processing, classification | Crop monitoring, land classification |
| **opt** | Linear programming, genetic algorithms | Resource optimization, decision support |

---

## Installation

### System Requirements

- Python 3.10 or higher
- Operating System: Windows / Linux / macOS

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/hetao-college/hetao_ag.git
cd hetao_ag

# Install basic version (numpy only)
pip install -e .
```

### Full Installation

```bash
# Install with all optional dependencies
pip install -e ".[full]"
```

### Optional Module Installation

```bash
# Remote sensing support (rasterio, geopandas)
pip install -e ".[space]"

# Livestock AI support (PyTorch, YOLO)
pip install -e ".[livestock]"

# Optimization support (PuLP)
pip install -e ".[opt]"

# Development tools (pytest, black, flake8)
pip install -e ".[dev]"
```

### Verify Installation

```python
import hetao_ag
print(hetao_ag.__version__)  # Output: 1.0.0
```

---

## Quick Start

### 5-Minute Tutorial

```python
import numpy as np
from hetao_ag.water import eto_penman_monteith, WeatherData
from hetao_ag.soil import SoilMoistureModel
from hetao_ag.crop import yield_reduction_salinity_crop, classify_salt_tolerance
from hetao_ag.space import compute_ndvi

# 1. Calculate Reference Evapotranspiration
weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)
et0 = eto_penman_monteith(weather)
print(f"Reference ET: {et0:.2f} mm/day")

# 2. Soil Moisture Simulation
soil = SoilMoistureModel(field_capacity=0.32, wilting_point=0.12)
result = soil.step_day(rain_mm=15, et_mm=5)
print(f"Soil Moisture: {result['moisture']:.3f}")

# 3. Salinity Stress Calculation
rel_yield = yield_reduction_salinity_crop(ECe=6.0, crop="wheat")
print(f"Wheat Relative Yield: {rel_yield*100:.1f}%")
print(f"Salt Tolerance: {classify_salt_tolerance('wheat')}")

# 4. NDVI Calculation from Remote Sensing
red = np.array([[120, 130], [110, 90]])
nir = np.array([[200, 210], [180, 160]])
ndvi = compute_ndvi(red, nir)
print(f"Mean NDVI: {ndvi.mean():.3f}")
```

---

## Module Documentation

### Core Module

The core module provides fundamental utilities for the entire library.

#### Unit System

```python
from hetao_ag.core import Quantity, Unit, meters, hectares, celsius

# Create physical quantities
distance = meters(500)
area = hectares(50)
temp = celsius(25)

# Unit conversion
distance_km = distance.to(Unit.KILOMETER)
temp_kelvin = temp.to(Unit.KELVIN)

# Arithmetic with units
total_distance = meters(100) + Quantity(0.5, Unit.KILOMETER)
```

#### Configuration Management

```python
from hetao_ag.core import ConfigManager, create_default_config

# Initialize with defaults
config = ConfigManager(defaults=create_default_config())

# Get nested configuration
field_capacity = config.get("soil.field_capacity")
kc = config.get("crop.kc_mid", default=1.15)

# Set configuration values
config.set("irrigation.efficiency", 0.90)
```

#### Logging System

```python
from hetao_ag.core import get_logger

logger = get_logger("experiment")
logger.info("Starting simulation", soil_ec=4.5, temperature=25)
logger.log_experiment_start("Yield Prediction", parameters={"model": "CropModel"})
```

---

### Soil Module

Models soil water dynamics and salinity for agricultural applications.

#### Soil Moisture Model

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

# Create model with soil parameters
model = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    root_depth_m=0.6,
    soil_type=SoilType.LOAM
)

# Daily simulation
result = model.step_day(rain_mm=20, irrigation_mm=0, et_mm=5)

# Key properties
print(f"Current moisture: {model.moisture}")
print(f"Stress factor: {model.stress_factor}")
print(f"Irrigation needed: {model.irrigation_need_mm} mm")
```

#### Salinity Model

```python
from hetao_ag.soil import SalinityModel, classify_soil_salinity

model = SalinityModel(initial_ECe=4.0)

# Irrigation adds salt
model.irrigate(amount_mm=60, ec_water=1.5)

# Leaching reduces salt
model.leach(drainage_mm=40)

# Classify salinity level
print(classify_soil_salinity(model.ECe))
```

#### Sensor Calibration

```python
from hetao_ag.soil import SensorCalibrator
import numpy as np

calibrator = SensorCalibrator()
raw = np.array([300, 450, 600, 750])
actual = np.array([0.10, 0.20, 0.30, 0.40])

result = calibrator.linear_calibration(raw, actual)
calibrated = result.apply(raw_value=500)
```

---

### Water Module

Implements FAO-56 evapotranspiration calculations and irrigation scheduling.

#### FAO-56 Penman-Monteith

```python
from hetao_ag.water import (
    eto_penman_monteith, WeatherData,
    crop_coefficient, etc_crop
)

# Weather data
weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)

# Reference evapotranspiration
et0 = eto_penman_monteith(weather)

# Crop evapotranspiration
kc = crop_coefficient("mid", "wheat")
etc = etc_crop(et0, kc)
print(f"Wheat ETc: {etc:.2f} mm/day")
```

#### Water Balance

```python
from hetao_ag.water import WaterBalance

wb = WaterBalance(
    initial_storage_mm=80,
    max_storage_mm=120,
    min_storage_mm=40
)

# Daily simulation
for day in range(10):
    record = wb.step_day(precip_mm=5, et_mm=6)

# Summary
summary = wb.get_summary()
print(f"Available water: {wb.available_water} mm")
print(f"Water deficit: {wb.deficit_mm} mm")
```

#### Irrigation Scheduling

```python
from hetao_ag.water import IrrigationScheduler, ScheduleType

scheduler = IrrigationScheduler(
    method=ScheduleType.SOIL_MOISTURE,
    trigger_threshold=0.5,
    max_application_mm=50
)

# Get recommendation based on soil moisture
recommendation = scheduler.recommend_by_moisture(
    current_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12,
    root_depth_m=0.3
)

if recommendation.should_irrigate:
    print(f"Irrigate: {recommendation.amount_mm:.1f} mm")
    print(f"Urgency: {recommendation.urgency}")
```

---

### Crop Module

Simulates crop growth, phenology, and stress responses.

#### Salt Stress (Maas-Hoffman Model)

```python
from hetao_ag.crop import (
    yield_reduction_salinity_crop,
    classify_salt_tolerance,
    CROP_SALT_TOLERANCE
)

# Calculate relative yield at given ECe
crops = ["wheat", "maize", "cotton", "barley"]
ECe = 8.0  # dS/m

for crop in crops:
    rel_yield = yield_reduction_salinity_crop(ECe, crop)
    tolerance = classify_salt_tolerance(crop)
    print(f"{crop}: {rel_yield*100:.1f}% yield ({tolerance})")
```

#### Phenology Tracking

```python
from hetao_ag.crop import PhenologyTracker, GrowthStage

tracker = PhenologyTracker("wheat")

# Accumulate growing degree days
for day in range(120):
    t_max = 25 + np.random.randn() * 3
    t_min = 15 + np.random.randn() * 2
    tracker.accumulate_gdd(t_max, t_min)

print(f"Accumulated GDD: {tracker.accumulated_gdd:.0f}")
print(f"Current stage: {tracker.current_stage.value}")
print(f"Progress to maturity: {tracker.progress_to_maturity()*100:.1f}%")
print(f"Crop coefficient Kc: {tracker.get_kc_for_stage()}")
```

#### Crop Growth Model

```python
from hetao_ag.crop import CropModel

model = CropModel("wheat")

# Simulate growing season
for day in range(120):
    result = model.update_daily(
        t_max=25, t_min=15, et=5,
        soil_moisture=0.25, ECe=3.0
    )

print(f"Final biomass: {model.accumulated_biomass:.0f} kg/ha")
print(f"Estimated yield: {model.estimate_yield():.0f} kg/ha")
```

---

### Livestock Module

Provides AI-based animal detection and health monitoring.

#### Animal Detection

```python
from hetao_ag.livestock import AnimalDetector

detector = AnimalDetector(
    confidence_threshold=0.5,
    use_gpu=True
)

# Detect animals in image
detections = detector.detect("farm_image.jpg")
for det in detections:
    print(f"{det.label}: confidence={det.confidence:.2f}")

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
print(f"Behavior: {behavior.value}")  # Output: grazing
```

#### Health Monitoring

```python
from hetao_ag.livestock import HealthMonitor, HerdHealthMonitor

# Individual animal monitoring
monitor = HealthMonitor("cow_001")

# Build baseline over 7 days
for _ in range(7):
    monitor.update_activity(100)
    monitor.update_feeding_time(240)

# Detect anomaly
monitor.update_activity(60)  # Low activity
monitor.update_temperature(40.2)  # Fever

alerts = monitor.check_health()
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.message}")

# Herd-level monitoring
herd = HerdHealthMonitor()
herd.add_animal("cow_001")
herd.add_animal("cow_002")
summary = herd.get_summary()
```

---

### Space Module

Remote sensing analysis for vegetation monitoring and crop classification.

#### Spectral Indices

```python
from hetao_ag.space import (
    compute_ndvi, compute_savi, compute_evi,
    compute_lswi, compute_ndwi,
    classify_vegetation_health
)
import numpy as np

# Band data (simulated)
red = np.array([[120, 130], [110, 90]], dtype=np.uint16)
nir = np.array([[200, 210], [180, 160]], dtype=np.uint16)
blue = np.array([[80, 85], [75, 70]], dtype=np.uint16)

# Calculate indices
ndvi = compute_ndvi(red, nir)
savi = compute_savi(red, nir, L=0.5)
evi = compute_evi(blue, red, nir)

print(f"NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
print(f"Vegetation health: {classify_vegetation_health(ndvi.mean())}")
```

#### Image Processing

```python
from hetao_ag.space import RasterImage, GeoMetadata, CloudMask

# Load from file (requires rasterio)
# img = RasterImage.from_file("sentinel2.tif")

# Or create from array
data = np.random.randint(0, 10000, (4, 100, 100), dtype=np.uint16)
img = RasterImage(
    data,
    band_names={"blue": 0, "green": 1, "red": 2, "nir": 3}
)

# Get bands
red = img.get_band("red")
nir = img.get_band("nir")

# Subset and mask
subset = img.subset(slice(0, 50), slice(0, 50))
```

#### Phenology Classification

```python
from hetao_ag.space import PhenologyClassifier, temporal_smoothing

# NDVI time series (12 dates, 50x50 pixels)
ndvi_series = np.random.rand(12, 50, 50) * 0.6 + 0.2

# Smooth time series
smoothed = temporal_smoothing(ndvi_series, window=3)

# Classify crops
classifier = PhenologyClassifier(smoothed)
crop_map = classifier.classify_crops(n_classes=3)

# Extract features for a pixel
features = classifier.extract_features(25, 25)
print(f"Peak NDVI: {features.peak_value:.3f}")
print(f"Peak time: day {features.peak_time}")
```

---

### Optimization Module

Agricultural resource optimization using linear programming and genetic algorithms.

#### Linear Programming

```python
from hetao_ag.opt import LinearOptimizer, optimize_crop_mix

# Define crops with economic data
crops = [
    {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
    {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
    {"name": "alfalfa", "profit_per_ha": 400, "water_per_ha": 2000},
]

# Optimize crop mix
solution = optimize_crop_mix(
    crops,
    total_land=100,  # hectares
    total_water=300000  # cubic meters
)

for crop, area in solution.items():
    print(f"{crop}: {area:.1f} ha")
```

#### Genetic Algorithm

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

# Define fitness function
def fitness(x):
    return -sum(xi**2 for xi in x)  # Minimize sphere function

# Configure and run
optimizer = GeneticOptimizer(
    fitness_func=fitness,
    n_vars=3,
    bounds=[(-5, 5)] * 3,
    config=GAConfig(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
)

result = optimizer.optimize()
print(f"Best solution: {result.best_solution}")
print(f"Best fitness: {result.best_fitness}")
```

#### Scenario Analysis

```python
from hetao_ag.opt import ScenarioEvaluator, FarmScenario

# Define crop parameters
crop_params = {
    "wheat": {"yield_kg_ha": 6000, "water_need_mm": 400,
              "price_per_kg": 0.8, "cost_per_ha": 1200},
    "maize": {"yield_kg_ha": 10000, "water_need_mm": 600,
              "price_per_kg": 0.6, "cost_per_ha": 1500},
}

# Evaluate scenarios
evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=500000)

s1 = evaluator.evaluate_scenario("All Wheat", {"wheat": 100}, 450)
s2 = evaluator.evaluate_scenario("All Maize", {"maize": 100}, 650)
s3 = evaluator.evaluate_scenario("Mixed", {"wheat": 50, "maize": 50}, 500)

# Compare
best = evaluator.compare_scenarios()
print(f"Best profit: {best['best_profit'].name}")
print(f"Best water efficiency: {best['best_water_efficiency'].name}")
```

---

## Architecture

```
hetao_ag/
├── __init__.py              # Package initialization
├── core/                    # Core utilities
│   ├── config.py           # Configuration management
│   ├── logger.py           # Logging system
│   ├── units.py            # Physical unit system
│   └── utils.py            # Utility functions
├── soil/                    # Soil science
│   ├── moisture.py         # Water dynamics
│   ├── salinity.py         # Salt accumulation
│   └── sensors.py          # Sensor calibration
├── water/                   # Hydrology
│   ├── evapotranspiration.py  # FAO-56 ET
│   ├── balance.py          # Water balance
│   └── irrigation.py       # Scheduling
├── crop/                    # Crop science
│   ├── growth.py           # Growth models
│   ├── phenology.py        # Development stages
│   └── stress.py           # Stress response
├── livestock/               # Animal science
│   ├── behavior.py         # Behavior analysis
│   ├── health.py           # Health monitoring
│   └── vision.py           # Detection
├── space/                   # Remote sensing
│   ├── indices.py          # Spectral indices
│   ├── imagery.py          # Image handling
│   └── classification.py   # Crop mapping
└── opt/                     # Optimization
    ├── linear.py           # LP solvers
    ├── genetic.py          # GA optimization
    └── planning.py         # Farm planning
```

### Data Flow

```
Input Sources                    Processing                      Output
────────────────────────────────────────────────────────────────────────
Weather Data    ──┐
                  ├── water/et ──┐
Sensor Data     ──┘              │
                                 ├── crop/growth ──┐
Satellite Images ── space/ndvi ──┘                 │
                                                   ├── opt/planning ── Decisions
Soil Samples    ── soil/salinity ── crop/stress ──┘                    Reports
                                                                        Alerts
Animal Video    ── livestock/vision ── livestock/health ──────────────┘
```

---

## Examples

### Complete Workflow Example

```python
"""
Complete precision agriculture workflow example.
Demonstrates integration of multiple modules.
"""
import numpy as np
from datetime import date

from hetao_ag.core import get_logger
from hetao_ag.water import eto_penman_monteith, WeatherData, WaterBalance
from hetao_ag.soil import SoilMoistureModel, SalinityModel
from hetao_ag.crop import CropModel, yield_reduction_salinity_crop
from hetao_ag.water import IrrigationScheduler, ScheduleType

# Initialize
logger = get_logger("precision_ag")
logger.info("Starting precision agriculture workflow")

# Create models
soil_moisture = SoilMoistureModel(field_capacity=0.32, wilting_point=0.12)
soil_salinity = SalinityModel(initial_ECe=3.0)
water_balance = WaterBalance(initial_storage_mm=80)
crop = CropModel("wheat")
scheduler = IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE)

# Simulate 60-day growing season
results = []
for day in range(60):
    # Weather (simulated)
    weather = WeatherData(
        t_mean=20 + 5*np.sin(day/30*np.pi),
        t_max=28 + 5*np.sin(day/30*np.pi),
        t_min=12 + 5*np.sin(day/30*np.pi),
        rh=50, u2=2.0, rs=20,
        elevation=1050, latitude=40.8, doy=100+day
    )

    # Calculate ET
    et0 = eto_penman_monteith(weather)

    # Check irrigation need
    rec = scheduler.recommend_by_moisture(
        soil_moisture.moisture, 0.32, 0.12
    )
    irrigation = rec.amount_mm if rec.should_irrigate else 0

    # Update models
    soil_moisture.step_day(rain_mm=0, irrigation_mm=irrigation, et_mm=et0)
    water_balance.step_day(irrig_mm=irrigation, et_mm=et0)
    crop.update_daily(
        t_max=weather.t_max, t_min=weather.t_min,
        et=et0, soil_moisture=soil_moisture.moisture,
        ECe=soil_salinity.ECe
    )

    results.append({
        'day': day,
        'et0': et0,
        'moisture': soil_moisture.moisture,
        'biomass': crop.accumulated_biomass
    })

# Final report
print(f"Final yield estimate: {crop.estimate_yield():.0f} kg/ha")
print(f"Total irrigation: {water_balance.total_irrigation:.0f} mm")
print(f"Water use efficiency: {water_balance.water_use_efficiency(crop.estimate_yield()):.2f} kg/m3")
```

### Run Demo Scripts

```bash
# Comprehensive demo
python examples/demo.py

# Module-specific examples
python examples/example_core.py
python examples/example_soil.py
python examples/example_water.py
python examples/example_crop.py
python examples/example_livestock.py
python examples/example_space.py
python examples/example_opt.py
```

---

## API Reference

For complete API documentation, see:

- [API Reference (English)](docs/API_REFERENCE.md)
- [User Manual (Chinese)](docs/USER_MANUAL.md)

Or use Python's help system:

```python
from hetao_ag.water import eto_penman_monteith
help(eto_penman_monteith)
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 code style
4. Add tests for new features
5. Submit a pull request

---

## References

- Allen, R.G., et al. (1998). FAO Irrigation and Drainage Paper No. 56
- Maas, E.V. & Hoffman, G.J. (1977). Crop Salt Tolerance - Current Assessment
- Van Genuchten, M.T. (1980). A Closed-form Equation for Predicting the Hydraulic Conductivity

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Project**: https://github.com/hetao-college/hetao_ag
- **Issues**: https://github.com/hetao-college/hetao_ag/issues
- **Email**: hetao@example.com

---

*Hetao Smart Agriculture Library v1.0.0*
*Copyright 2024 Hetao College*
