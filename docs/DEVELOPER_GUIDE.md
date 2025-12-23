# Developer Guide

This guide provides detailed information about the hetao_ag codebase structure, architecture, and how to extend the library.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Module Deep Dive](#module-deep-dive)
4. [Code Patterns](#code-patterns)
5. [Extending the Library](#extending-the-library)
6. [Testing](#testing)
7. [Contributing Guidelines](#contributing-guidelines)

---

## Project Structure

```
hetao_ag/
├── README.md                    # Main documentation
├── hetao_ag.rar                 # Packaged source archive
├── docs/
│   ├── API_REFERENCE.md         # Complete API documentation
│   ├── USER_MANUAL.md           # User guide with examples
│   └── DEVELOPER_GUIDE.md       # This file
├── examples/
│   ├── demo.py                  # Comprehensive demo
│   ├── example_core.py          # Core module examples
│   ├── example_soil.py          # Soil module examples
│   ├── example_water.py         # Water module examples
│   ├── example_crop.py          # Crop module examples
│   ├── example_livestock.py     # Livestock module examples
│   ├── example_space.py         # Remote sensing examples
│   └── example_opt.py           # Optimization examples
├── tests/
│   └── test_all.py              # Test suite
├── hetao_ag/
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core infrastructure
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── logger.py            # Logging system
│   │   ├── units.py             # SI unit system
│   │   └── utils.py             # Utility functions
│   ├── soil/                    # Soil science module
│   │   ├── __init__.py
│   │   ├── moisture.py          # Soil moisture modeling
│   │   ├── salinity.py          # Salinity modeling
│   │   └── sensors.py           # Sensor calibration
│   ├── water/                   # Water management module
│   │   ├── __init__.py
│   │   ├── evapotranspiration.py # ET calculations
│   │   ├── balance.py           # Water balance
│   │   └── irrigation.py        # Irrigation scheduling
│   ├── crop/                    # Crop science module
│   │   ├── __init__.py
│   │   ├── stress.py            # Stress response models
│   │   ├── phenology.py         # Phenology tracking
│   │   └── growth.py            # Growth simulation
│   ├── livestock/               # Livestock module
│   │   ├── __init__.py
│   │   ├── vision.py            # Animal detection
│   │   ├── behavior.py          # Behavior classification
│   │   └── health.py            # Health monitoring
│   ├── space/                   # Remote sensing module
│   │   ├── __init__.py
│   │   ├── indices.py           # Spectral indices
│   │   ├── imagery.py           # Image processing
│   │   └── classification.py    # Crop classification
│   └── opt/                     # Optimization module
│       ├── __init__.py
│       ├── linear.py            # Linear programming
│       ├── genetic.py           # Genetic algorithms
│       └── planning.py          # Farm planning
├── hetao_ag_colab.py            # Google Colab bundle
├── generate_colab_bundle.py     # Colab bundle generator
└── deploy_to_github.bat         # Deployment script
```

---

## Architecture Overview

### Design Principles

1. **Modularity**: Each module is self-contained and can be used independently
2. **SI Units**: All physical quantities use SI units internally
3. **Scientific Rigor**: Based on peer-reviewed models (FAO-56, Maas-Hoffman, van Genuchten)
4. **Graceful Degradation**: Optional dependencies fallback to simpler implementations
5. **Chinese-First**: Complete Chinese documentation and comments

### Module Dependencies

```
                    ┌──────────────────────────────────────┐
                    │              hetao_ag                │
                    │           (main package)             │
                    └──────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│     core      │◄──────────│     soil      │           │    water      │
│  (foundation) │           │ (soil science)│           │(water mgmt)   │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        │                             └──────────┬──────────────────┘
        │                                        │
        ▼                                        ▼
┌───────────────┐                       ┌───────────────┐
│     crop      │◄──────────────────────│  livestock    │
│(crop science) │                       │ (monitoring)  │
└───────────────┘                       └───────────────┘
        │                                        │
        ▼                                        │
┌───────────────┐                                │
│    space      │                                │
│(remote sensing)│                               │
└───────────────┘                                │
        │                                        │
        └────────────────────┬───────────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │      opt      │
                    │(optimization) │
                    └───────────────┘
```

### External Dependencies

| Module | Required | Optional |
|--------|----------|----------|
| **core** | numpy | - |
| **soil** | numpy | - |
| **water** | numpy | - |
| **crop** | numpy | - |
| **livestock** | numpy | torch, opencv-python |
| **space** | numpy | rasterio, gdal |
| **opt** | numpy | pulp, scipy |

---

## Module Deep Dive

### Core Module (`hetao_ag/core/`)

The foundation layer providing essential infrastructure.

#### units.py - Physical Unit System

**Key Components:**

```python
# Enumeration of physical dimensions
class Dimension(Enum):
    LENGTH = "length"
    AREA = "area"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"
    TEMPERATURE = "temperature"
    # ... more dimensions

# Unit definitions with SI conversion factors
class Unit(Enum):
    METER = ("m", 1.0, Dimension.LENGTH)
    KILOMETER = ("km", 1000.0, Dimension.LENGTH)
    HECTARE = ("ha", 10000.0, Dimension.AREA)
    CELSIUS = ("°C", 1.0, Dimension.TEMPERATURE, 273.15)  # with offset
    # ... more units

# Physical quantity with unit
@dataclass
class Quantity:
    value: float
    unit: Unit

    def to(self, target_unit: Unit) -> 'Quantity':
        """Convert to target unit"""
        # Validates dimensions match
        # Applies conversion factor and offset
```

**Usage Pattern:**
```python
from hetao_ag.core import meters, hectares, celsius

# Create quantities
length = meters(100)
area = hectares(50)
temp = celsius(25)

# Arithmetic (auto-converts units)
total = meters(100) + meters(50)  # 150 m
```

#### config.py - Configuration Management

**Key Components:**

```python
class ConfigManager:
    """Hierarchical configuration with environment variable override"""

    def __init__(self, config_file=None, defaults=None, env_prefix="HETAO"):
        self._config = {}
        self._defaults = defaults or {}
        self._env_prefix = env_prefix

    def get(self, key: str, default=None):
        """Get config value with dot notation support"""
        # Priority: env var > file config > defaults

    def set(self, key: str, value):
        """Set config value dynamically"""

    def validate(self, required_keys: List[str]) -> bool:
        """Validate all required keys exist"""
```

**Configuration File Format (YAML):**
```yaml
soil:
  field_capacity: 0.32
  wilting_point: 0.12

crop:
  thermal_time_base_celsius: 10.0

irrigation:
  efficiency: 0.85
```

#### logger.py - Logging System

**Key Components:**

```python
class Logger:
    """Structured logging with experiment tracking"""

    def __init__(self, name, level="INFO", log_file=None, colored=True):
        self._logger = logging.getLogger(name)
        # Configure handlers and formatters

    def info(self, msg, **kwargs):
        """Log info with structured data"""
        # kwargs are formatted as key=value pairs

    def log_experiment_start(self, name, parameters, random_seed=None):
        """Log experiment metadata for reproducibility"""

    def log_experiment_end(self, success, results):
        """Log experiment results"""
```

#### utils.py - Utility Functions

**Categories:**

1. **Math Utilities:**
   - `safe_divide(num, denom, default=0.0)` - Division without ZeroDivisionError
   - `clamp(value, min_val, max_val)` - Constrain value to range
   - `linear_interpolate(x, x1, y1, x2, y2)` - Linear interpolation

2. **Array Operations:**
   - `moving_average(data, window)` - Sliding window average
   - `normalize(data, min_val, max_val)` - Normalize to [0, 1]

3. **Model Validation:**
   - `rmse(observed, predicted)` - Root Mean Square Error
   - `mae(observed, predicted)` - Mean Absolute Error
   - `r_squared(observed, predicted)` - Coefficient of Determination
   - `validate_model(observed, predicted)` - Complete validation report

4. **File Utilities:**
   - `ensure_path(path)` - Convert to Path object
   - `ensure_directory(path)` - Create directory if not exists

5. **Date Utilities:**
   - `day_of_year(dt)` - Convert date to DOY (1-366)

---

### Soil Module (`hetao_ag/soil/`)

Soil water and salinity dynamics modeling.

#### moisture.py - Soil Moisture Model

**Key Class: `SoilMoistureModel`**

```python
class SoilMoistureModel:
    """Bucket model for soil water balance"""

    def __init__(self,
                 field_capacity: float = 0.32,
                 wilting_point: float = 0.12,
                 initial_moisture: float = 0.25,
                 root_depth_m: float = 0.3,
                 soil_type: SoilType = SoilType.LOAM):
        self.fc = field_capacity
        self.wp = wilting_point
        self._moisture = initial_moisture
        # ...

    @property
    def moisture(self) -> float:
        """Current volumetric water content"""

    @property
    def stress_factor(self) -> float:
        """Water stress coefficient (0-1, 1=no stress)"""
        taw = self.fc - self.wp  # Total available water
        raw = 0.5 * taw  # Readily available water
        if self._moisture >= self.fc - raw:
            return 1.0
        return (self._moisture - self.wp) / (self.fc - raw - self.wp)

    def step_day(self, rain_mm, irrigation_mm, et_mm) -> dict:
        """Simulate one day of water balance"""
        # Add precipitation and irrigation
        # Remove ET
        # Calculate runoff and deep percolation
```

**Physical Model:**
- Based on FAO-56 soil water balance
- Accounts for runoff when exceeding saturation
- Calculates deep percolation

#### salinity.py - Salinity Model

**Key Class: `SalinityModel`**

```python
class SalinityModel:
    """Soil salinity dynamics with leaching"""

    def __init__(self, initial_ECe, root_depth_m, soil_water_content):
        self._ECe = initial_ECe
        self.depth = root_depth_m
        # ...

    def irrigate(self, amount_mm: float, ec_water: float) -> dict:
        """Add irrigation water with dissolved salts"""
        # Mass balance calculation

    def leach(self, drainage_mm: float) -> dict:
        """Remove salts through drainage"""
        # Leaching fraction model

    def leaching_requirement(self, ec_irrigation, ec_threshold) -> float:
        """Calculate leaching requirement (LR)"""
        # LR = ECw / (5 * ECe - ECw)  (simplified FAO approach)
```

#### sensors.py - Sensor Calibration

**Key Class: `SensorCalibrator`**

```python
class SensorCalibrator:
    """Multi-method sensor calibration"""

    def linear_calibration(self, raw, truth) -> CalibrationResult:
        """y = mx + b calibration"""

    def polynomial_calibration(self, raw, truth, degree=2) -> CalibrationResult:
        """y = a0 + a1*x + a2*x^2 + ... calibration"""

    def auto_calibrate(self, raw, truth) -> CalibrationResult:
        """Select best method by R²"""
```

---

### Water Module (`hetao_ag/water/`)

Evapotranspiration and irrigation management.

#### evapotranspiration.py - ET Calculations

**FAO-56 Penman-Monteith Implementation:**

```python
def eto_penman_monteith(weather: WeatherData) -> float:
    """
    Calculate reference evapotranspiration using FAO-56 PM equation.

    ET0 = (0.408 * Δ * (Rn - G) + γ * (900/(T+273)) * u2 * (es - ea)) /
          (Δ + γ * (1 + 0.34 * u2))

    Parameters:
        weather: WeatherData with t_mean, t_max, t_min, rh, u2, rs,
                 elevation, latitude, doy

    Returns:
        ET0 in mm/day
    """
    # Step 1: Psychrometric constant
    P = 101.3 * ((293 - 0.0065 * weather.elevation) / 293) ** 5.26
    gamma = 0.665e-3 * P

    # Step 2: Saturation vapor pressure
    es = (sat_vapor_pressure(weather.t_max) + sat_vapor_pressure(weather.t_min)) / 2
    ea = es * weather.rh / 100

    # Step 3: Slope of saturation vapor pressure curve
    delta = 4098 * sat_vapor_pressure(weather.t_mean) / (weather.t_mean + 237.3) ** 2

    # Step 4: Net radiation (from Rs)
    Rn = calculate_net_radiation(weather)

    # Step 5: Apply PM equation
    # ...
```

**Hargreaves Method (for limited data):**

```python
def eto_hargreaves(t_mean, t_max, t_min, Ra) -> float:
    """
    Hargreaves-Samani equation for ET0.

    ET0 = 0.0023 * (T_mean + 17.8) * (T_max - T_min)^0.5 * Ra
    """
```

#### irrigation.py - Irrigation Scheduling

**Key Class: `IrrigationScheduler`**

```python
class ScheduleType(Enum):
    FIXED_INTERVAL = "fixed"
    SOIL_MOISTURE = "soil_moisture"
    ET_BASED = "et_based"
    DEFICIT = "deficit"

class IrrigationScheduler:
    """Multi-strategy irrigation scheduling"""

    def recommend_by_moisture(self, current_moisture, field_capacity,
                              wilting_point) -> IrrigationRecommendation:
        """MAD-based irrigation trigger"""

    def recommend_by_et(self, cumulative_etc, last_irrigation_mm,
                       trigger_mm) -> IrrigationRecommendation:
        """ET-based irrigation trigger"""

    def deficit_irrigation_schedule(self, growth_stages,
                                   deficits: Dict[str, float]) -> List[IrrigationEvent]:
        """Controlled deficit irrigation by growth stage"""
```

---

### Crop Module (`hetao_ag/crop/`)

Crop growth simulation and stress response.

#### stress.py - Stress Response Models

**Maas-Hoffman Salinity Model:**

```python
def yield_reduction_salinity(ECe: float, threshold: float, slope: float) -> float:
    """
    Maas-Hoffman linear threshold-slope model.

    Yr = 1 - slope * (ECe - threshold) for ECe > threshold
    Yr = 1.0 for ECe <= threshold

    Reference: Maas & Hoffman (1977)
    """
    if ECe <= threshold:
        return 1.0
    return max(0.0, 1.0 - slope * (ECe - threshold))
```

**Crop-Specific Salt Tolerance Data:**

```python
CROP_SALT_TOLERANCE = {
    "wheat": CropSaltTolerance(threshold=6.0, slope=0.071),
    "maize": CropSaltTolerance(threshold=1.7, slope=0.120),
    "cotton": CropSaltTolerance(threshold=7.7, slope=0.052),
    "barley": CropSaltTolerance(threshold=8.0, slope=0.050),
    # ... more crops
}
```

#### phenology.py - Phenology Tracking

**Growing Degree Days (GDD) Model:**

```python
class PhenologyTracker:
    """GDD-based phenology tracking"""

    def calculate_daily_gdd(self, t_max, t_min) -> float:
        """
        Calculate daily GDD using average method.

        GDD = max(0, (T_max + T_min) / 2 - T_base)
        """
        t_mean = (t_max + t_min) / 2
        return max(0, t_mean - self.config.base_temperature)

    def _update_stage(self):
        """Update growth stage based on accumulated GDD"""
        stage_gdd = self.config.stage_gdd
        if self.accumulated_gdd >= stage_gdd["maturity"]:
            self.current_stage = GrowthStage.MATURITY
        elif self.accumulated_gdd >= stage_gdd["grain_fill"]:
            self.current_stage = GrowthStage.GRAIN_FILL
        # ... etc
```

**Crop-Specific Phenology Parameters:**

```python
CROP_PHENOLOGY = {
    "wheat": PhenologyConfig(
        base_temperature=0.0,
        stage_gdd={
            "emergence": 120,
            "vegetative": 450,
            "flowering": 800,
            "grain_fill": 1200,
            "maturity": 1600
        }
    ),
    # ... more crops
}
```

#### growth.py - Crop Growth Model

**Integrated Crop Model:**

```python
class CropModel:
    """Integrated crop growth simulation"""

    def update_daily(self, t_max, t_min, et, soil_moisture,
                    field_capacity, wilting_point, ECe) -> dict:
        """
        Daily update combining:
        1. GDD accumulation (phenology)
        2. Water stress calculation
        3. Salt stress calculation
        4. Biomass accumulation
        5. LAI development
        """
        # 1. Update phenology
        self.phenology.accumulate_gdd(t_max, t_min)

        # 2. Get crop coefficient for current stage
        kc = self.phenology.get_kc_for_stage()

        # 3. Calculate actual ET
        etc = et * kc
        ks_water = water_stress_from_moisture(soil_moisture, fc, wp)
        actual_et = etc * ks_water

        # 4. Calculate salt stress
        ks_salt = yield_reduction_salinity_crop(ECe, self.crop)

        # 5. Accumulate biomass (transpiration efficiency model)
        biomass = self.config.transpiration_efficiency * actual_et * ks_salt
        self.accumulated_biomass += biomass

        return {...}
```

---

### Livestock Module (`hetao_ag/livestock/`)

Animal detection and health monitoring.

#### vision.py - Animal Detection

**YOLO-based Detector:**

```python
class AnimalDetector:
    """Deep learning animal detector"""

    SUPPORTED_ANIMALS = ["cow", "sheep", "goat", "horse", "pig", "chicken"]

    def __init__(self, model_name="yolov5s", confidence_threshold=0.5, use_gpu=True):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None  # Lazy loading

    def load_model(self):
        """Load YOLO model from torch hub"""
        import torch
        self.model = torch.hub.load('ultralytics/yolov5', self.model_name)

    def detect(self, image) -> List[Detection]:
        """Detect animals in image"""
        if not self._model_loaded:
            return self._simulate_detection()  # Fallback

        results = self.model(image)
        detections = []
        for *bbox, conf, cls in results.xyxy[0].tolist():
            if conf >= self.confidence_threshold:
                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=int(cls),
                    label=results.names[int(cls)]
                ))
        return detections
```

#### health.py - Health Monitoring

**Multi-Source Health Monitor:**

```python
class HealthMonitor:
    """Individual animal health monitoring"""

    def __init__(self, animal_id):
        self.animal_id = animal_id
        self.activity_history = []
        self.feeding_history = []
        self.temperature_history = []

        # Baseline values (established after 7 days)
        self.activity_baseline = None
        self.feeding_baseline = None

        # Alert thresholds
        self.thresholds = {
            "activity_drop": 0.30,  # 30% reduction
            "feeding_drop": 0.25,   # 25% reduction
            "temp_high": 39.5,      # Fever threshold
        }

    def check_health(self) -> List[HealthAlert]:
        """Generate alerts based on current vs baseline"""
        alerts = []

        # Check activity anomaly
        if self.activity_baseline:
            current = self.activity_history[-1]
            if current < self.activity_baseline * (1 - self.thresholds["activity_drop"]):
                alerts.append(HealthAlert(
                    animal_id=self.animal_id,
                    alert_type=AlertType.REDUCED_ACTIVITY,
                    severity=HealthStatus.WARNING,
                    message=f"Activity dropped {drop_percent:.0f}%"
                ))

        return alerts
```

---

### Space Module (`hetao_ag/space/`)

Remote sensing and vegetation analysis.

#### indices.py - Spectral Indices

**Vegetation Index Calculations:**

```python
def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index

    NDVI = (NIR - Red) / (NIR + Red)

    Range: [-1, 1]
    - < 0: water, bare soil
    - 0-0.2: sparse vegetation
    - 0.2-0.4: moderate vegetation
    - > 0.6: dense vegetation
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    return np.clip((nir - red) / (nir + red + 1e-10), -1, 1)

def compute_savi(red, nir, L=0.5) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    L = 0.5 for moderate vegetation cover
    L = 0.25 for dense vegetation
    L = 1.0 for sparse vegetation
    """

def compute_evi(blue, red, nir, G=2.5, C1=6.0, C2=7.5, L=1.0) -> np.ndarray:
    """
    Enhanced Vegetation Index

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Reduces atmospheric and soil background noise
    """
```

#### classification.py - Crop Classification

**Phenology-Based Classifier:**

```python
class PhenologyClassifier:
    """Time-series based crop classification"""

    def __init__(self, time_series: np.ndarray, dates=None):
        """
        Parameters:
            time_series: NDVI array (time, height, width)
            dates: Optional date labels
        """
        self.data = time_series
        self.peak = np.max(time_series, axis=0)
        self.peak_time = np.argmax(time_series, axis=0)

    def classify_crops(self, n_classes=3) -> np.ndarray:
        """
        Rule-based classification by peak timing:
        - Class 1: Early peak (winter wheat)
        - Class 2: Late peak (maize, sunflower)
        - Class 3: Mid peak
        """
        classes = np.zeros_like(self.peak_time)

        early_threshold = self.data.shape[0] // 3
        late_threshold = 2 * self.data.shape[0] // 3

        classes[(self.peak_time < early_threshold) & (self.peak > 0.5)] = 1
        classes[(self.peak_time >= late_threshold) & (self.peak > 0.5)] = 2
        classes[(self.peak_time >= early_threshold) &
                (self.peak_time < late_threshold) & (self.peak > 0.5)] = 3

        return classes
```

---

### Optimization Module (`hetao_ag/opt/`)

Resource allocation and decision support.

#### linear.py - Linear Programming

**LP Optimizer:**

```python
class LinearOptimizer:
    """Linear programming for resource allocation"""

    def add_variable(self, name, lower=0, upper=None):
        """Add decision variable with bounds"""

    def set_objective(self, coefficients: Dict[str, float], maximize=True):
        """Set objective function coefficients"""

    def add_constraint(self, coefficients: Dict[str, float], sense: str, rhs: float):
        """Add linear constraint (<=, >=, ==)"""

    def solve(self) -> OptimizationResult:
        """Solve using PuLP or fallback"""
        try:
            import pulp
            return self._solve_pulp()
        except ImportError:
            return self._solve_simple()  # Approximate solution
```

#### genetic.py - Genetic Algorithm

**GA Optimizer:**

```python
class GeneticOptimizer:
    """Genetic algorithm for complex optimization"""

    def __init__(self, fitness_func, n_vars, bounds, config=None):
        self.fitness_func = fitness_func
        self.n_vars = n_vars
        self.bounds = bounds
        self.config = config or GAConfig()

    def _initialize_population(self):
        """Random initialization within bounds"""

    def _tournament_select(self) -> List[float]:
        """Tournament selection"""

    def _crossover(self, parent1, parent2) -> Tuple[List, List]:
        """Single-point crossover"""

    def _mutate(self, individual) -> List[float]:
        """Random reset mutation"""

    def optimize(self) -> GAResult:
        """Main optimization loop"""
        self._initialize_population()

        for gen in range(self.config.generations):
            # Evaluate fitness
            self._evaluate_population()

            # Elitism
            elite = self._select_elite()

            # Selection, crossover, mutation
            offspring = self._create_offspring()

            self.population = elite + offspring

        return GAResult(best_solution=..., best_fitness=...)
```

---

## Code Patterns

### 1. Dataclass for Configuration

```python
from dataclasses import dataclass

@dataclass
class CropConfig:
    """Crop configuration parameters"""
    name: str = "wheat"
    potential_yield_kg_ha: float = 6000.0
    harvest_index: float = 0.45
    # ... more parameters
```

### 2. Enum for States and Types

```python
from enum import Enum

class GrowthStage(Enum):
    DORMANT = "dormant"
    EMERGENCE = "emergence"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    GRAIN_FILL = "grain_fill"
    MATURITY = "maturity"
```

### 3. Optional Dependencies Pattern

```python
def solve(self):
    """Solve with optional dependency fallback"""
    try:
        import pulp
        return self._solve_pulp()
    except ImportError:
        # Fallback to simple implementation
        return self._solve_simple()
```

### 4. Property for Computed Values

```python
class SoilMoistureModel:
    @property
    def stress_factor(self) -> float:
        """Computed water stress (0-1)"""
        # Calculation based on internal state
        return self._calculate_stress()
```

### 5. Method Chaining

```python
config = (ConfigManager()
          .load("config.yaml")
          .set("soil.fc", 0.32)
          .set("irrigation.efficiency", 0.85))
```

---

## Extending the Library

### Adding a New Crop

1. Add salt tolerance parameters to `crop/stress.py`:
```python
CROP_SALT_TOLERANCE["sorghum"] = CropSaltTolerance(
    threshold=6.8,
    slope=0.16
)
```

2. Add phenology parameters to `crop/phenology.py`:
```python
CROP_PHENOLOGY["sorghum"] = PhenologyConfig(
    base_temperature=10.0,
    stage_gdd={
        "emergence": 80,
        "vegetative": 400,
        "flowering": 700,
        "grain_fill": 1000,
        "maturity": 1300
    }
)
```

3. Add growth parameters to `crop/growth.py`:
```python
CROP_CONFIGS["sorghum"] = CropConfig(
    name="sorghum",
    potential_yield_kg_ha=8000,
    harvest_index=0.40,
    transpiration_efficiency=18.0,
    max_lai=5.5
)
```

### Adding a New Spectral Index

Add to `space/indices.py`:

```python
def compute_gndvi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Green Normalized Difference Vegetation Index

    GNDVI = (NIR - Green) / (NIR + Green)

    More sensitive to chlorophyll concentration than NDVI.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    return np.clip((nir - green) / (nir + green + 1e-10), -1, 1)
```

Update `space/__init__.py`:
```python
from .indices import compute_gndvi
__all__.append("compute_gndvi")
```

### Adding a New Optimization Algorithm

Add to `opt/` directory:

```python
# opt/pso.py
class ParticleSwarmOptimizer:
    """Particle Swarm Optimization"""

    def __init__(self, fitness_func, n_vars, bounds, config=None):
        # Implementation

    def optimize(self) -> PSOResult:
        # PSO algorithm
```

Update `opt/__init__.py`:
```python
from .pso import ParticleSwarmOptimizer, PSOResult
__all__.extend(["ParticleSwarmOptimizer", "PSOResult"])
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=hetao_ag

# Run specific test file
python -m pytest tests/test_all.py
```

### Test Structure

```python
# tests/test_soil.py
import pytest
from hetao_ag.soil import SoilMoistureModel

class TestSoilMoistureModel:

    def test_initial_moisture(self):
        model = SoilMoistureModel(initial_moisture=0.25)
        assert model.moisture == 0.25

    def test_stress_factor_at_fc(self):
        model = SoilMoistureModel(
            field_capacity=0.32,
            initial_moisture=0.32
        )
        assert model.stress_factor == 1.0

    def test_stress_factor_at_wp(self):
        model = SoilMoistureModel(
            field_capacity=0.32,
            wilting_point=0.12,
            initial_moisture=0.12
        )
        assert model.stress_factor == 0.0
```

---

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Write Chinese docstrings (English for code)
- Maximum line length: 100 characters

### Commit Messages

```
[module] Brief description

Detailed explanation if needed.

Refs: #issue-number
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

---

## References

### Scientific Models

- FAO-56: Crop evapotranspiration - Guidelines for computing crop water requirements
- Maas & Hoffman (1977): Crop salt tolerance
- van Genuchten (1980): Soil water retention curves

### External Libraries

- NumPy: https://numpy.org/
- PuLP: https://coin-or.github.io/pulp/
- PyTorch: https://pytorch.org/
- Rasterio: https://rasterio.readthedocs.io/

---

*hetao_ag v1.0.0 Developer Guide*
