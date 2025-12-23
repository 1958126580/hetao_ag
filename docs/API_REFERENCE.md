# API 参考文档

本文档提供 `hetao_ag` 库的完整 API 参考。

---

## hetao_ag.core

### 单位系统

#### `class Dimension(Enum)`

物理维度枚举。

| 值            | 描述   |
| ------------- | ------ |
| LENGTH        | 长度   |
| AREA          | 面积   |
| VOLUME        | 体积   |
| MASS          | 质量   |
| TIME          | 时间   |
| TEMPERATURE   | 温度   |
| PRESSURE      | 压强   |
| VELOCITY      | 速度   |
| CONDUCTIVITY  | 电导率 |
| ENERGY        | 能量   |
| POWER         | 功率   |
| DIMENSIONLESS | 无量纲 |

#### `class Unit(Enum)`

单位枚举，包含符号、SI 转换因子和维度。

**长度单位**: METER, KILOMETER, CENTIMETER, MILLIMETER  
**面积单位**: SQUARE_METER, HECTARE, SQUARE_KILOMETER  
**体积单位**: CUBIC_METER, LITER, MILLILITER  
**质量单位**: KILOGRAM, GRAM, TON  
**时间单位**: SECOND, MINUTE, HOUR, DAY  
**温度单位**: KELVIN, CELSIUS  
**压强单位**: PASCAL, KILOPASCAL, BAR  
**电导率单位**: SIEMENS_PER_METER, DECISIEMENS_PER_METER

**属性**:

- `symbol` - 单位符号
- `to_si_factor` - 转换为 SI 的系数
- `dimension` - 物理维度

#### `class Quantity`

物理量类。

**构造参数**:

- `value: float` - 数值
- `unit: Unit` - 单位

**方法**:

- `to(target_unit: Unit) -> Quantity` - 转换到目标单位
- `to_si() -> Quantity` - 转换到 SI 基本单位

**运算符**: `+`, `-`, `*`, `/`, `==`, `<`, `<=`, `>`, `>=`

#### 便捷函数

```python
meters(value) -> Quantity
kilometers(value) -> Quantity
hectares(value) -> Quantity
celsius(value) -> Quantity
kilopascals(value) -> Quantity
ds_per_m(value) -> Quantity
megajoules_per_m2(value) -> Quantity
mm_per_day(value) -> Quantity
```

---

### 配置管理

#### `class ConfigManager`

配置管理器。

**构造参数**:

- `config_file: Optional[Union[str, Path]]` - 配置文件路径
- `defaults: Optional[Dict[str, Any]]` - 默认配置
- `env_prefix: str` - 环境变量前缀
- `auto_reload: bool` - 自动重载

**方法**:

- `load(config_file) -> ConfigManager` - 加载配置文件
- `get(key: str, default=None) -> Any` - 获取配置项
- `set(key: str, value: Any) -> ConfigManager` - 设置配置项
- `has(key: str) -> bool` - 检查配置项存在
- `validate(required_keys: List[str]) -> bool` - 验证必需配置
- `save(path=None)` - 保存配置

#### `create_default_config() -> Dict`

创建默认配置字典。

---

### 日志工具

#### `class Logger`

日志记录器。

**构造参数**:

- `name: str` - 日志器名称
- `level: Union[str, int]` - 日志级别
- `log_file: Optional[Union[str, Path]]` - 日志文件
- `console_output: bool` - 控制台输出
- `colored: bool` - 彩色输出

**方法**:

- `debug(msg, **kwargs)` - 调试日志
- `info(msg, **kwargs)` - 信息日志
- `warning(msg, **kwargs)` - 警告日志
- `error(msg, **kwargs)` - 错误日志
- `log_experiment_start(name, parameters, random_seed)` - 记录实验开始
- `log_experiment_end(name, results)` - 记录实验结束

#### `get_logger(name: str) -> Logger`

获取或创建日志器。

---

### 工具函数

| 函数                                     | 说明           |
| ---------------------------------------- | -------------- |
| `safe_divide(num, denom, default=0.0)`   | 安全除法       |
| `clamp(value, min_val, max_val)`         | 值限制         |
| `linear_interpolate(x, x1, y1, x2, y2)`  | 线性插值       |
| `array_interpolate(x, x_array, y_array)` | 数组插值       |
| `day_of_year(dt)`                        | 年积日         |
| `moving_average(data, window)`           | 移动平均       |
| `normalize(data, min_val, max_val)`      | 归一化         |
| `rmse(observed, predicted)`              | 均方根误差     |
| `mae(observed, predicted)`               | 平均绝对误差   |
| `r_squared(observed, predicted)`         | 决定系数       |
| `validate_model(observed, predicted)`    | 模型验证       |
| `ensure_path(path)`                      | 确保 Path 对象 |
| `ensure_directory(path)`                 | 确保目录存在   |

#### `class Timer`

计时器上下文管理器。

```python
with Timer("操作名"):
    # 代码
```

---

## hetao_ag.soil

### 土壤水分

#### `class SoilMoistureModel`

土壤水分模型。

**构造参数**:

- `field_capacity: float` - 田间持水量
- `wilting_point: float` - 凋萎点
- `initial_moisture: float` - 初始含水量
- `root_depth_m: float` - 根区深度(m)
- `soil_type: SoilType` - 土壤类型

**属性**:

- `moisture` - 当前含水量
- `stress_factor` - 水分胁迫因子(0-1)
- `irrigation_need_mm` - 需灌溉量(mm)

**方法**:

- `add_water(amount_mm) -> Tuple[float, float]` - 添加水分
- `remove_water(amount_mm) -> float` - 移除水分(ET)
- `deep_percolation() -> float` - 深层渗透
- `step_day(rain_mm, irrigation_mm, et_mm) -> dict` - 模拟一天

#### `class SoilType(Enum)`

土壤类型: SAND, LOAMY_SAND, SANDY_LOAM, LOAM, SILT_LOAM, SILT, SANDY_CLAY_LOAM, CLAY_LOAM, SILTY_CLAY_LOAM, SANDY_CLAY, SILTY_CLAY, CLAY

#### `SOIL_PARAMETERS`

van Genuchten 土壤水力参数字典。

#### `van_genuchten_theta(h, theta_r, theta_s, alpha, n) -> float`

van Genuchten 土壤水分特征曲线。

---

### 土壤盐分

#### `class SalinityModel`

土壤盐分模型。

**构造参数**:

- `initial_ECe: float` - 初始 EC(dS/m)
- `root_depth_m: float` - 根区深度
- `soil_water_content: float` - 土壤含水量
- `bulk_density: float` - 土壤容重

**方法**:

- `irrigate(amount_mm, ec_water) -> dict` - 灌溉
- `leach(drainage_mm) -> dict` - 淋洗
- `leaching_requirement(ec_irrigation, ec_threshold) -> float` - 淋洗需求
- `step_day(...) -> dict` - 模拟一天

#### `classify_soil_salinity(ECe: float) -> str`

土壤盐分分级。

#### `classify_water_salinity(EC: float) -> str`

灌溉水质分级。

---

### 传感器校准

#### `class SensorCalibrator`

传感器校准器。

**方法**:

- `linear_calibration(raw, truth) -> CalibrationResult` - 线性校准
- `polynomial_calibration(raw, truth, degree) -> CalibrationResult` - 多项式校准
- `auto_calibrate(raw, truth) -> CalibrationResult` - 自动选择最佳方法

#### `class CalibrationResult`

校准结果。

**属性**: `method`, `coefficients`, `r_squared`, `rmse`  
**方法**: `apply(raw_value) -> float` - 应用校准

#### `capacitive_sensor_formula(raw, dry_value, wet_value) -> float`

电容式传感器通用公式。

---

## hetao_ag.water

### 蒸散发

#### `eto_penman_monteith(weather: WeatherData) -> float`

FAO-56 Penman-Monteith 参考蒸散发(mm/day)。

#### `eto_hargreaves(t_mean, t_max, t_min, Ra) -> float`

Hargreaves 蒸散发(mm/day)。

#### `extraterrestrial_radiation(latitude, doy) -> float`

天文辐射(MJ/m²/day)。

#### `crop_coefficient(growth_stage, crop) -> float`

作物系数 Kc。

#### `etc_crop(et0, kc, ks=1.0) -> float`

作物蒸散发。

#### `class WeatherData`

气象数据。

**属性**: `t_mean`, `t_max`, `t_min`, `rh`, `u2`, `rs`, `elevation`, `latitude`, `doy`

---

### 水量平衡

#### `class WaterBalance`

水量平衡模型。

**构造参数**:

- `initial_storage_mm: float`
- `max_storage_mm: float`
- `min_storage_mm: float`

**方法**:

- `add_precipitation(amount_mm) -> tuple`
- `add_irrigation(amount_mm) -> float`
- `remove_et(amount_mm) -> float`
- `step_day(...) -> WaterBalanceRecord`
- `get_summary() -> dict`
- `water_use_efficiency(yield_kg_ha) -> float`

---

### 灌溉调度

#### `class IrrigationScheduler`

灌溉调度器。

**构造参数**:

- `method: ScheduleType`
- `trigger_threshold: float`
- `max_application_mm: float`
- `irrigation_efficiency: float`

**方法**:

- `recommend_by_moisture(...) -> IrrigationRecommendation`
- `recommend_by_et(...) -> IrrigationRecommendation`
- `fixed_schedule(...) -> List[IrrigationEvent]`
- `deficit_irrigation_schedule(...) -> List[IrrigationEvent]`

#### `class ScheduleType(Enum)`

FIXED_INTERVAL, SOIL_MOISTURE, ET_BASED, DEFICIT

---

## hetao_ag.crop

### 胁迫响应

#### `yield_reduction_salinity(ECe, threshold, slope) -> float`

Maas-Hoffman 盐分胁迫。

#### `yield_reduction_salinity_crop(ECe, crop) -> float`

根据作物计算盐分胁迫。

#### `water_stress_from_moisture(soil_moisture, fc, wp, p=0.5) -> float`

基于土壤水分的胁迫因子。

#### `combined_stress_factor(water, salinity, method) -> float`

组合胁迫因子。

#### `yield_with_stress(potential, water_stress, salinity_stress, other_stress) -> float`

胁迫条件下产量。

#### `classify_salt_tolerance(crop) -> str`

作物耐盐分级。

#### `CROP_SALT_TOLERANCE`

作物盐分耐受参数字典。

---

### 物候期

#### `class PhenologyTracker`

物候期跟踪器。

**构造参数**:

- `crop: str`
- `config: Optional[PhenologyConfig]`

**属性**:

- `accumulated_gdd` - 累积积温
- `current_stage` - 当前阶段
- `days_after_planting` - 播后天数

**方法**:

- `accumulate_gdd(t_max, t_min) -> float`
- `progress_to_maturity() -> float`
- `days_to_maturity(avg_gdd) -> int`
- `get_kc_for_stage() -> float`

#### `class GrowthStage(Enum)`

DORMANT, EMERGENCE, VEGETATIVE, FLOWERING, GRAIN_FILL, MATURITY, HARVEST

#### `growing_degree_days(t_max, t_min, t_base, t_upper) -> ndarray`

批量计算积温。

---

### 生长模型

#### `class CropModel`

作物生长模型。

**方法**:

- `update_daily(...) -> dict` - 更新一天生长
- `estimate_yield() -> float` - 估算产量
- `water_use_efficiency(total_et) -> float` - 水分利用效率

---

## hetao_ag.livestock

### 视觉检测

#### `class AnimalDetector`

动物检测器。

**方法**:

- `detect(image) -> List[Detection]`
- `count_animals(image, species=None) -> Dict[str, int]`

#### `calculate_iou(box1, box2) -> float`

计算 IoU。

---

### 行为分类

#### `class BehaviorClassifier`

行为分类器。

**方法**:

- `classify_from_motion(motion, head_position) -> AnimalBehavior`
- `analyze_daily_pattern(records) -> Dict[str, float]`

#### `class AnimalBehavior(Enum)`

STANDING, LYING, WALKING, GRAZING, DRINKING, RUMINATING, RUNNING, ABNORMAL

---

### 健康监测

#### `class HealthMonitor`

健康监测器。

**方法**:

- `update_activity(value)`
- `update_feeding_time(minutes)`
- `update_temperature(temp)`
- `check_health() -> List[HealthAlert]`
- `get_status() -> HealthStatus`

#### `class HerdHealthMonitor`

群体健康监测。

---

## hetao_ag.space

### 光谱指数

| 函数                               | 说明     |
| ---------------------------------- | -------- |
| `compute_ndvi(red, nir)`           | NDVI     |
| `compute_savi(red, nir, L)`        | SAVI     |
| `compute_lswi(nir, swir)`          | LSWI     |
| `compute_evi(blue, red, nir)`      | EVI      |
| `compute_ndwi(green, nir)`         | NDWI     |
| `classify_vegetation_health(ndvi)` | 植被分级 |

---

### 影像处理

#### `class RasterImage`

栅格影像类。

**方法**:

- `from_file(path) -> RasterImage`
- `get_band(name_or_index) -> ndarray`
- `subset(row_slice, col_slice) -> RasterImage`
- `apply_mask(mask) -> RasterImage`

---

### 物候分类

#### `class PhenologyClassifier`

物候分类器。

**方法**:

- `extract_features(row, col) -> PhenologyFeatures`
- `classify_crops(n_classes) -> ndarray`
- `get_phenology_map() -> Dict[str, ndarray]`

#### `temporal_smoothing(time_series, window) -> ndarray`

时序平滑。

---

## hetao_ag.opt

### 线性规划

#### `class LinearOptimizer`

线性规划优化器。

**方法**:

- `add_variable(name, lower, upper)`
- `set_objective(coefficients, maximize)`
- `add_constraint(coefficients, sense, rhs)`
- `solve() -> OptimizationResult`

#### `optimize_crop_mix(crops, total_land, total_water) -> Dict`

优化作物组合。

---

### 遗传算法

#### `class GeneticOptimizer`

遗传算法优化器。

**构造参数**:

- `fitness_func: Callable`
- `n_vars: int`
- `bounds: List[Tuple]`
- `config: GAConfig`

**方法**:

- `optimize() -> GAResult`

#### `class GAConfig`

GA 配置: `population_size`, `generations`, `crossover_rate`, `mutation_rate`, `elitism`, `tournament_size`

#### `optimize_irrigation_schedule(daily_et, max_irrigation, min_interval) -> Tuple`

优化灌溉计划。

---

### 农场规划

#### `class ScenarioEvaluator`

场景评估器。

**方法**:

- `evaluate_scenario(name, crop_areas, irrigation_mm) -> FarmScenario`
- `compare_scenarios() -> Dict`
- `sensitivity_analysis(...) -> List[FarmScenario]`

#### `multi_objective_score(profit, water_use, sustainability, weights) -> float`

多目标评分。

---

_hetao_ag v1.0.0 API Reference_
