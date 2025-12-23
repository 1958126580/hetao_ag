# hetao_ag Architecture Documentation

This document provides a comprehensive overview of the hetao_ag library architecture, including module design, data flow, class relationships, and integration patterns.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Architecture](#module-architecture)
3. [Core Module Design](#core-module-design)
4. [Domain Modules](#domain-modules)
5. [Data Flow Patterns](#data-flow-patterns)
6. [Integration Guide](#integration-guide)
7. [Extension Points](#extension-points)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        hetao_ag Library                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────── Domain Layer ──────────────────────┐        │
│   │                                                         │        │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐  │        │
│   │  │  crop   │ │  soil   │ │  water  │ │  livestock   │  │        │
│   │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬───────┘  │        │
│   │       │           │           │             │          │        │
│   │  ┌────┴────┐ ┌────┴────┐                               │        │
│   │  │  space  │ │   opt   │                               │        │
│   │  └─────────┘ └─────────┘                               │        │
│   │                                                         │        │
│   └─────────────────────────┬───────────────────────────────┘        │
│                             │                                        │
│   ┌─────────────────── Core Layer ──────────────────────────┐        │
│   │                         │                                │        │
│   │  ┌──────────┐  ┌───────┴──────┐  ┌──────────┐           │        │
│   │  │  units   │  │   config     │  │  logger  │           │        │
│   │  └──────────┘  └──────────────┘  └──────────┘           │        │
│   │                                                          │        │
│   │  ┌─────────────────────────────────────────────┐        │        │
│   │  │                  utils                       │        │        │
│   │  └─────────────────────────────────────────────┘        │        │
│   └──────────────────────────────────────────────────────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Layered Architecture**: Core utilities at the bottom, domain modules on top
2. **Loose Coupling**: Modules can be used independently
3. **High Cohesion**: Related functionality grouped together
4. **Interface Segregation**: Clear, focused public APIs
5. **Dependency Inversion**: Core abstractions don't depend on details

---

## Module Architecture

### Package Structure

```
hetao_ag/
├── __init__.py                 # Main package entry point
│
├── core/                       # Foundation layer
│   ├── __init__.py            # Exports: Quantity, Unit, ConfigManager, Logger
│   ├── units.py               # Physical unit system
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging utilities
│   └── utils.py               # Common utilities
│
├── soil/                       # Soil science module
│   ├── __init__.py            # Public API exports
│   ├── moisture.py            # SoilMoistureModel, SoilType
│   ├── salinity.py            # SalinityModel, classification
│   └── sensors.py             # SensorCalibrator
│
├── water/                      # Hydrology module
│   ├── __init__.py            # Public API exports
│   ├── evapotranspiration.py  # ET calculations (FAO-56)
│   ├── balance.py             # WaterBalance model
│   └── irrigation.py          # IrrigationScheduler
│
├── crop/                       # Crop science module
│   ├── __init__.py            # Public API exports
│   ├── growth.py              # CropModel
│   ├── phenology.py           # PhenologyTracker
│   └── stress.py              # Stress response functions
│
├── livestock/                  # Animal science module
│   ├── __init__.py            # Public API exports
│   ├── behavior.py            # BehaviorClassifier
│   ├── health.py              # HealthMonitor
│   └── vision.py              # AnimalDetector
│
├── space/                      # Remote sensing module
│   ├── __init__.py            # Public API exports
│   ├── indices.py             # Spectral index functions
│   ├── imagery.py             # RasterImage class
│   └── classification.py      # PhenologyClassifier
│
└── opt/                        # Optimization module
    ├── __init__.py            # Public API exports
    ├── linear.py              # LinearOptimizer
    ├── genetic.py             # GeneticOptimizer
    └── planning.py            # ScenarioEvaluator
```

### Module Dependencies

```
                    External Libraries
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌───────┐        ┌────────┐       ┌──────────┐
    │ numpy │        │ scipy  │       │ pandas   │
    └───┬───┘        └────┬───┘       └────┬─────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   core    │ ◄── All modules depend on core
                    └─────┬─────┘
                          │
        ┌────────┬────────┼────────┬────────┐
        │        │        │        │        │
        ▼        ▼        ▼        ▼        ▼
    ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐
    │ soil │ │water │ │ crop │ │space  │ │ opt  │
    └──────┘ └──┬───┘ └──┬───┘ └───────┘ └──────┘
                │        │
                │   ┌────┴────┐
                └──►│livestock│
                    └─────────┘
```

---

## Core Module Design

### Unit System (`core/units.py`)

The unit system provides type-safe physical quantities with automatic conversion.

```
┌─────────────────────────────────────────────────────────────┐
│                     Unit System                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐     ┌────────────┐     ┌──────────────────┐ │
│  │ Dimension  │────►│    Unit    │────►│    Quantity      │ │
│  │   (Enum)   │     │   (Enum)   │     │    (Class)       │ │
│  └────────────┘     └────────────┘     └──────────────────┘ │
│       │                   │                     │           │
│  - LENGTH            - METER              - value: float    │
│  - AREA              - KILOMETER          - unit: Unit      │
│  - TEMPERATURE       - HECTARE            - to(Unit)        │
│  - PRESSURE          - CELSIUS            - to_si()         │
│  - ...               - ...                - __add__, etc.   │
│                                                              │
│  Helper Functions:                                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ meters(v) │ hectares(v) │ celsius(v) │ ds_per_m(v)    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Class Diagram:**

```python
class Dimension(Enum):
    LENGTH, AREA, VOLUME, MASS, TIME, TEMPERATURE, PRESSURE, ...

class Unit(Enum):
    """Each unit has: symbol, to_si_factor, dimension"""
    METER = ("m", 1.0, Dimension.LENGTH)
    KILOMETER = ("km", 1000.0, Dimension.LENGTH)
    HECTARE = ("ha", 10000.0, Dimension.AREA)
    CELSIUS = ("°C", 1.0, Dimension.TEMPERATURE)  # with offset
    ...

class Quantity:
    def __init__(self, value: float, unit: Unit): ...
    def to(self, target_unit: Unit) -> Quantity: ...
    def to_si(self) -> Quantity: ...
    def __add__(self, other: Quantity) -> Quantity: ...
    def __mul__(self, scalar: float) -> Quantity: ...
```

### Configuration Management (`core/config.py`)

Hierarchical configuration with environment variable and file support.

```
┌─────────────────────────────────────────────────────────────┐
│                  Configuration System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │ Configuration    │     │ Sources (priority order)     │  │
│  │ Manager          │◄────│                              │  │
│  │                  │     │ 1. Runtime set()             │  │
│  │ - config: Dict   │     │ 2. Environment variables     │  │
│  │ - defaults: Dict │     │ 3. YAML/JSON config file     │  │
│  │ - env_prefix: str│     │ 4. Default values            │  │
│  └──────────────────┘     └──────────────────────────────┘  │
│                                                              │
│  Methods:                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ load(file)   │ get(key)    │ set(key, val) │ save()   │ │
│  │ has(key)     │ validate()  │ merge(dict)   │          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Nested Key Access: config.get("soil.field_capacity")       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Logging System (`core/logger.py`)

Structured logging with experiment tracking capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                     Logging System                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │     Logger       │                                       │
│  │                  │                                       │
│  │ - name: str      │     Outputs:                          │
│  │ - level: int     │     ┌─────────┐   ┌─────────────────┐ │
│  │ - handlers: []   │────►│ Console │   │ File (optional) │ │
│  │                  │     └─────────┘   └─────────────────┘ │
│  └──────────────────┘                                       │
│                                                              │
│  Methods:                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ debug(msg, **kw) │ info(msg, **kw) │ warning/error    │ │
│  │ log_experiment_start(name, params, seed)              │ │
│  │ log_experiment_end(name, results)                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Factory: get_logger("name") -> Logger                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Utility Functions (`core/utils.py`)

Common mathematical and validation utilities.

```
┌─────────────────────────────────────────────────────────────┐
│                     Utilities                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Math Functions:                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ safe_divide(a, b) │ clamp(v, min, max) │ normalize()  │ │
│  │ linear_interpolate() │ moving_average() │             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Validation Metrics:                                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ rmse(obs, pred) │ mae(obs, pred) │ r_squared()        │ │
│  │ validate_model() -> ValidationResult                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Date/Time:                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ day_of_year(date) │ Timer context manager             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  File I/O:                                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ensure_path(p) │ ensure_directory(p)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Domain Modules

### Soil Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Soil Module                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 SoilMoistureModel                        ││
│  │                                                          ││
│  │  Attributes:                                             ││
│  │  - field_capacity: float                                 ││
│  │  - wilting_point: float                                  ││
│  │  - moisture: float (current state)                       ││
│  │  - soil_type: SoilType                                   ││
│  │                                                          ││
│  │  Properties:                                             ││
│  │  - stress_factor -> float (0-1)                         ││
│  │  - irrigation_need_mm -> float                          ││
│  │                                                          ││
│  │  Methods:                                                ││
│  │  - add_water(mm) -> (infiltration, runoff)              ││
│  │  - remove_water(mm) -> actual_removed                   ││
│  │  - step_day(rain, irrig, et) -> dict                    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   SalinityModel                          ││
│  │                                                          ││
│  │  - ECe: float (current soil electrical conductivity)    ││
│  │  - irrigate(amount_mm, ec_water) -> dict                ││
│  │  - leach(drainage_mm) -> dict                           ││
│  │  - leaching_requirement(ec_irrig, ec_thresh) -> float   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 SensorCalibrator                         ││
│  │                                                          ││
│  │  - linear_calibration(raw, truth) -> CalibrationResult  ││
│  │  - polynomial_calibration(raw, truth, deg) -> Result    ││
│  │  - auto_calibrate(raw, truth) -> CalibrationResult      ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  Supporting Types:                                           │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  SoilType    │  │ SOIL_PARAMETERS │  │CalibrationResult││
│  │  (Enum)      │  │ (Dict)          │  │ - apply(v)     ││
│  └──────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Water Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Water Module                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Evapotranspiration Functions                  │ │
│  │                                                          │ │
│  │  WeatherData (dataclass)          Core Functions:       │ │
│  │  - t_mean, t_max, t_min           - eto_penman_monteith │ │
│  │  - rh (relative humidity)         - eto_hargreaves      │ │
│  │  - u2 (wind speed 2m)             - extraterrestrial_rad│ │
│  │  - rs (solar radiation)           - crop_coefficient    │ │
│  │  - elevation, latitude, doy       - etc_crop            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   WaterBalance                          │ │
│  │                                                          │ │
│  │  State:                        Methods:                  │ │
│  │  - storage_mm                  - add_precipitation()    │ │
│  │  - max_storage_mm              - add_irrigation()       │ │
│  │  - min_storage_mm              - remove_et()            │ │
│  │  - history: []                 - step_day()             │ │
│  │                                - get_summary()          │ │
│  │  Properties:                   - water_use_efficiency() │ │
│  │  - available_water                                      │ │
│  │  - deficit_mm                                           │ │
│  │  - relative_storage                                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                IrrigationScheduler                      │ │
│  │                                                          │ │
│  │  ScheduleType (Enum):          Methods:                  │ │
│  │  - FIXED_INTERVAL              - recommend_by_moisture()│ │
│  │  - SOIL_MOISTURE               - recommend_by_et()      │ │
│  │  - ET_BASED                    - fixed_schedule()       │ │
│  │  - DEFICIT                     - deficit_irrig_schedule()│ │
│  │                                                          │ │
│  │  Returns: IrrigationRecommendation                      │ │
│  │  - should_irrigate: bool                                │ │
│  │  - amount_mm: float                                     │ │
│  │  - reason: str                                          │ │
│  │  - urgency: str                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Crop Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Crop Module                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Stress Functions                       │ │
│  │                                                          │ │
│  │  Salinity (Maas-Hoffman):                               │ │
│  │  - yield_reduction_salinity(ECe, thresh, slope)         │ │
│  │  - yield_reduction_salinity_crop(ECe, crop_name)        │ │
│  │  - classify_salt_tolerance(crop) -> str                 │ │
│  │  - CROP_SALT_TOLERANCE: Dict[str, CropSaltTolerance]    │ │
│  │                                                          │ │
│  │  Water Stress:                                          │ │
│  │  - water_stress_factor(actual_et, potential_et)         │ │
│  │  - water_stress_from_moisture(sm, fc, wp)               │ │
│  │                                                          │ │
│  │  Combined:                                               │ │
│  │  - combined_stress_factor(ks_water, ks_salt, method)    │ │
│  │  - yield_with_stress(potential, ks_w, ks_s, ks_other)   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 PhenologyTracker                        │ │
│  │                                                          │ │
│  │  GrowthStage (Enum):           State:                   │ │
│  │  - DORMANT                     - accumulated_gdd        │ │
│  │  - EMERGENCE                   - current_stage          │ │
│  │  - VEGETATIVE                  - days_after_planting    │ │
│  │  - FLOWERING                                            │ │
│  │  - GRAIN_FILL                  Methods:                 │ │
│  │  - MATURITY                    - accumulate_gdd(tmax,min)│ │
│  │  - HARVEST                     - progress_to_maturity() │ │
│  │                                - get_kc_for_stage()     │ │
│  │  CROP_PHENOLOGY: Dict[str, PhenologyConfig]             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    CropModel                            │ │
│  │                                                          │ │
│  │  Composition:           State:                          │ │
│  │  - phenology: Tracker   - accumulated_biomass           │ │
│  │  - config: CropConfig   - lai (leaf area index)         │ │
│  │                         - stress_history                │ │
│  │  Methods:                                               │ │
│  │  - update_daily(tmax, tmin, et, sm, ECe) -> dict       │ │
│  │  - estimate_yield() -> float                           │ │
│  │  - water_use_efficiency(total_et) -> float             │ │
│  │                                                          │ │
│  │  CROP_CONFIGS: Dict[str, CropConfig]                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Livestock Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Livestock Module                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  AnimalDetector                         │ │
│  │                                                          │ │
│  │  Based on YOLO (YOLOv5/v8)                              │ │
│  │                                                          │ │
│  │  Detection (dataclass):        Methods:                  │ │
│  │  - bbox: (x1,y1,x2,y2)        - load_model()            │ │
│  │  - confidence: float          - detect(image) -> [Det]  │ │
│  │  - class_id: int              - count_animals(img) -> {}│ │
│  │  - label: str                                           │ │
│  │                                                          │ │
│  │  Supported: cow, sheep, goat, horse, pig, chicken       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                BehaviorClassifier                       │ │
│  │                                                          │ │
│  │  AnimalBehavior (Enum):        Methods:                  │ │
│  │  - STANDING                    - classify_from_motion()  │ │
│  │  - LYING                       - classify_sequence()     │ │
│  │  - WALKING                     - analyze_daily_pattern() │ │
│  │  - GRAZING                                              │ │
│  │  - DRINKING                                             │ │
│  │  - RUMINATING                                           │ │
│  │  - RUNNING                                              │ │
│  │  - ABNORMAL                                             │ │
│  │                                                          │ │
│  │  ActivityMonitor: tracks daily activity levels          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   HealthMonitor                         │ │
│  │                                                          │ │
│  │  AlertType (Enum):             HealthStatus (Enum):     │ │
│  │  - REDUCED_ACTIVITY            - HEALTHY                │ │
│  │  - REDUCED_FEEDING             - ATTENTION              │ │
│  │  - ABNORMAL_BEHAVIOR           - WARNING                │ │
│  │  - TEMPERATURE                 - CRITICAL               │ │
│  │  - LAMENESS                                             │ │
│  │  - HEAT_DETECTION                                       │ │
│  │                                                          │ │
│  │  Methods:                                               │ │
│  │  - update_activity(value)                               │ │
│  │  - update_feeding_time(minutes)                         │ │
│  │  - update_temperature(celsius)                          │ │
│  │  - check_health() -> [HealthAlert]                      │ │
│  │  - get_status() -> HealthStatus                         │ │
│  │                                                          │ │
│  │  HerdHealthMonitor: manages multiple animals            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Space Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Space Module                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Spectral Indices                       │ │
│  │                                                          │ │
│  │  Functions (all return np.ndarray):                     │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │ compute_ndvi(red, nir)                             │ │ │
│  │  │ compute_savi(red, nir, L=0.5)                      │ │ │
│  │  │ compute_evi(blue, red, nir, G, C1, C2, L)          │ │ │
│  │  │ compute_lswi(nir, swir)                            │ │ │
│  │  │ compute_ndwi(green, nir)                           │ │ │
│  │  │ classify_vegetation_health(ndvi) -> str            │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    RasterImage                          │ │
│  │                                                          │ │
│  │  GeoMetadata:                  Methods:                  │ │
│  │  - crs: str                    - from_file(path)        │ │
│  │  - transform: tuple            - get_band(name/idx)     │ │
│  │  - bounds: tuple               - subset(row, col)       │ │
│  │  - resolution: float           - apply_mask(mask)       │ │
│  │                                                          │ │
│  │  CloudMask:                                             │ │
│  │  - from_qa_band(qa) -> mask                             │ │
│  │  - simple_detection(blue, nir) -> mask                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                PhenologyClassifier                      │ │
│  │                                                          │ │
│  │  Input: NDVI time series (time, height, width)          │ │
│  │                                                          │ │
│  │  PhenologyFeatures:            Methods:                  │ │
│  │  - peak_value                  - extract_features(r, c) │ │
│  │  - peak_time                   - classify_crops(n)      │ │
│  │  - start_of_season             - get_phenology_map()    │ │
│  │  - end_of_season                                        │ │
│  │  - amplitude                                            │ │
│  │                                                          │ │
│  │  temporal_smoothing(series, window) -> smoothed         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Optimization Module                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  LinearOptimizer                        │ │
│  │                                                          │ │
│  │  Uses PuLP library (fallback if not installed)          │ │
│  │                                                          │ │
│  │  Methods:                      Returns:                  │ │
│  │  - add_variable(name, lb, ub)  OptimizationResult:      │ │
│  │  - set_objective(coef, max)    - status: str            │ │
│  │  - add_constraint(coef, op, b) - objective_value: float │ │
│  │  - solve() -> Result           - variables: dict        │ │
│  │                                                          │ │
│  │  Helper:                                                 │ │
│  │  optimize_crop_mix(crops, land, water) -> allocation    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  GeneticOptimizer                       │ │
│  │                                                          │ │
│  │  GAConfig:                     GAResult:                 │ │
│  │  - population_size             - best_solution          │ │
│  │  - generations                 - best_fitness           │ │
│  │  - crossover_rate              - generations_run        │ │
│  │  - mutation_rate               - fitness_history        │ │
│  │  - elitism                                              │ │
│  │  - tournament_size                                      │ │
│  │                                                          │ │
│  │  Methods:                                               │ │
│  │  - optimize() -> GAResult                               │ │
│  │                                                          │ │
│  │  Helper:                                                 │ │
│  │  optimize_irrigation_schedule(et, max_irr, interval)    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  ScenarioEvaluator                      │ │
│  │                                                          │ │
│  │  FarmScenario:                 Methods:                  │ │
│  │  - name: str                   - evaluate_scenario()    │ │
│  │  - crop_areas: dict            - compare_scenarios()    │ │
│  │  - irrigation_mm               - sensitivity_analysis() │ │
│  │  - expected_yield: dict                                 │ │
│  │  - total_profit: float                                  │ │
│  │  - water_use_efficiency                                 │ │
│  │                                                          │ │
│  │  multi_objective_score(profit, water, sustain, weights) │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Patterns

### Typical Agricultural Workflow

```
                            Input Layer
    ┌─────────────────────────────────────────────────────────┐
    │  Weather     Soil       Satellite    Animal    Historical│
    │  Station     Sensors    Images       Cameras   Records   │
    └─────┬──────────┬───────────┬───────────┬──────────┬─────┘
          │          │           │           │          │
          ▼          ▼           ▼           ▼          ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   Processing Layer                       │
    │                                                          │
    │  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐   │
    │  │ water/ │ │  soil/   │ │  space/ │ │  livestock/  │   │
    │  │ ET     │ │ moisture │ │ indices │ │  detection   │   │
    │  └───┬────┘ └────┬─────┘ └────┬────┘ └──────┬───────┘   │
    │      │           │            │             │            │
    │      └───────────┼────────────┼─────────────┘            │
    │                  │            │                          │
    │                  ▼            ▼                          │
    │            ┌──────────────────────┐                      │
    │            │       crop/          │                      │
    │            │  growth + stress     │                      │
    │            └──────────┬───────────┘                      │
    │                       │                                  │
    │                       ▼                                  │
    │            ┌──────────────────────┐                      │
    │            │         opt/         │                      │
    │            │     optimization     │                      │
    │            └──────────┬───────────┘                      │
    │                       │                                  │
    └───────────────────────┼──────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Output Layer                          │
    │                                                          │
    │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
    │  │ Irrigation │  │   Health   │  │    Management      │ │
    │  │ Schedule   │  │   Alerts   │  │    Decisions       │ │
    │  └────────────┘  └────────────┘  └────────────────────┘ │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
```

### Module Integration Example

```python
# Example: Complete precision irrigation workflow

# 1. Input: Weather data
weather = WeatherData(t_max=32, t_min=18, ...)

# 2. Process: Calculate ET
et0 = eto_penman_monteith(weather)

# 3. Integrate: Soil moisture + Crop stage
soil = SoilMoistureModel(...)
crop = CropModel("wheat")

# Update soil state
soil.step_day(rain_mm=0, et_mm=et0)

# Get crop coefficient based on phenology
kc = crop.phenology.get_kc_for_stage()
etc = et0 * kc

# 4. Optimize: Irrigation decision
scheduler = IrrigationScheduler(method=ScheduleType.SOIL_MOISTURE)
recommendation = scheduler.recommend_by_moisture(
    current_moisture=soil.moisture,
    field_capacity=soil.field_capacity,
    wilting_point=soil.wilting_point
)

# 5. Output: Decision
if recommendation.should_irrigate:
    print(f"Apply {recommendation.amount_mm} mm irrigation")
```

---

## Integration Guide

### Using Multiple Modules Together

```python
"""
Integration example: Season-long crop simulation
"""
import numpy as np
from hetao_ag.core import get_logger, ConfigManager
from hetao_ag.water import eto_penman_monteith, WeatherData, WaterBalance
from hetao_ag.soil import SoilMoistureModel, SalinityModel
from hetao_ag.crop import CropModel
from hetao_ag.water import IrrigationScheduler, ScheduleType

class IntegratedSimulation:
    """Demonstrates module integration."""

    def __init__(self, config: dict):
        self.logger = get_logger("simulation")

        # Initialize models
        self.soil_moisture = SoilMoistureModel(
            field_capacity=config['soil']['fc'],
            wilting_point=config['soil']['wp']
        )
        self.soil_salinity = SalinityModel(
            initial_ECe=config['soil']['ece']
        )
        self.water_balance = WaterBalance(
            initial_storage_mm=config['water']['initial']
        )
        self.crop = CropModel(config['crop']['name'])
        self.scheduler = IrrigationScheduler(
            method=ScheduleType.SOIL_MOISTURE
        )

    def simulate_day(self, weather: WeatherData) -> dict:
        """Simulate one day with full integration."""

        # 1. Calculate ET
        et0 = eto_penman_monteith(weather)

        # 2. Check irrigation need
        rec = self.scheduler.recommend_by_moisture(
            self.soil_moisture.moisture,
            self.soil_moisture.field_capacity,
            self.soil_moisture.wilting_point
        )
        irrigation = rec.amount_mm if rec.should_irrigate else 0

        # 3. Update soil moisture
        self.soil_moisture.step_day(
            rain_mm=0, irrigation_mm=irrigation, et_mm=et0
        )

        # 4. Update water balance
        self.water_balance.step_day(irrig_mm=irrigation, et_mm=et0)

        # 5. Update crop
        result = self.crop.update_daily(
            t_max=weather.t_max,
            t_min=weather.t_min,
            et=et0,
            soil_moisture=self.soil_moisture.moisture,
            ECe=self.soil_salinity.ECe
        )

        return {
            'et0': et0,
            'irrigation': irrigation,
            'moisture': self.soil_moisture.moisture,
            'crop_stage': result['stage'],
            'biomass': result['biomass_kg_ha']
        }
```

---

## Extension Points

### Adding New Crop Types

```python
from hetao_ag.crop import CropConfig, PhenologyConfig, CropSaltTolerance

# 1. Define crop parameters
NEW_CROP_CONFIG = CropConfig(
    name="sorghum",
    potential_yield_kg_ha=8000,
    harvest_index=0.45,
    transpiration_efficiency=22,
    max_lai=5.5,
    salt_threshold=6.8,
    salt_slope=0.16
)

# 2. Define phenology
NEW_PHENOLOGY = PhenologyConfig(
    base_temperature=10.0,
    stage_gdd={
        "emergence": 80,
        "vegetative": 400,
        "flowering": 700,
        "grain_fill": 1000,
        "maturity": 1300
    }
)

# 3. Register (in actual code, add to the respective dicts)
CROP_CONFIGS["sorghum"] = NEW_CROP_CONFIG
CROP_PHENOLOGY["sorghum"] = NEW_PHENOLOGY
CROP_SALT_TOLERANCE["sorghum"] = CropSaltTolerance(6.8, 0.16)
```

### Adding New Spectral Indices

```python
import numpy as np

def compute_gndvi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Green Normalized Difference Vegetation Index.

    GNDVI = (NIR - Green) / (NIR + Green)

    Useful for chlorophyll content estimation.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    numerator = nir - green
    denominator = nir + green

    gndvi = np.where(denominator == 0, 0, numerator / denominator)
    return np.clip(gndvi, -1, 1)
```

### Custom Optimization Problems

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

# Define custom fitness for nitrogen optimization
def nitrogen_fitness(params):
    """
    Optimize nitrogen application.
    params: [timing1, amount1, timing2, amount2]
    """
    t1, a1, t2, a2 = params
    total_n = a1 + a2

    # Simulate yield response
    yield_estimate = simulate_yield(t1, a1, t2, a2)

    # Cost function
    n_cost = total_n * 1.5
    yield_value = yield_estimate * 0.8

    # Maximize profit
    return yield_value - n_cost

optimizer = GeneticOptimizer(
    fitness_func=nitrogen_fitness,
    n_vars=4,
    bounds=[(0, 60), (0, 100), (30, 90), (0, 100)],
    config=GAConfig(generations=200)
)
```

---

## Summary

The hetao_ag library provides a comprehensive, modular architecture for smart agriculture applications:

1. **Core Layer**: Foundation utilities (units, config, logging)
2. **Domain Modules**: Specialized agricultural science (soil, water, crop, etc.)
3. **Integration**: Modules work together through well-defined interfaces
4. **Extension**: Easy to add new crops, indices, or optimization problems

The design prioritizes:
- Scientific accuracy through validated models
- Usability through clear APIs and comprehensive documentation
- Flexibility through modular, loosely-coupled design
- Performance through efficient numerical implementations
