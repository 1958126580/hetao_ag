# 河套智慧农牧业库 (hetao_ag) 用户手册

## 目录

1. [简介](#简介)
2. [安装指南](#安装指南)
3. [快速开始](#快速开始)
4. [模块详解](#模块详解)
5. [API 参考](#api参考)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)

---

## 简介

### 概述

`hetao_ag` 是一个面向智慧农业和畜牧业的综合 Python 库，专为河套灌区等干旱半干旱地区的农业生产管理设计。库提供了从土壤建模、水循环管理到作物生长模拟、畜牧监测、遥感分析和农场优化的完整解决方案。

### 主要特性

| 模块          | 功能                              | 典型应用场景             |
| ------------- | --------------------------------- | ------------------------ |
| **core**      | 单位系统、配置管理、日志工具      | 科学计算标准化、实验追踪 |
| **soil**      | 土壤水分/盐分建模、传感器校准     | 土壤墒情监测、盐碱地改良 |
| **water**     | FAO-56 蒸散发、水量平衡、灌溉调度 | 精准灌溉、水资源管理     |
| **crop**      | 作物生长模拟、胁迫响应、物候跟踪  | 产量预测、种植决策       |
| **livestock** | 动物检测、行为分类、健康监测      | 智慧牧场、疫病预警       |
| **space**     | 遥感指数、影像处理、物候分类      | 长势监测、作物识别       |
| **opt**       | 线性规划、遗传算法、场景分析      | 资源优化、决策支持       |

### 设计理念

1. **SI 单位标准**: 全面采用国际单位制，避免单位混乱导致的错误
2. **科学严谨**: 基于 FAO-56、Maas-Hoffman 等国际权威模型
3. **模块化设计**: 各模块独立可用，也可组合使用
4. **中文优先**: 完整的中文注释和文档
5. **生产就绪**: 经过 100%测试覆盖验证

---

## 安装指南

### 系统要求

- Python 3.10 或更高版本
- 操作系统: Windows / Linux / macOS

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/hetao-college/hetao_ag.git
cd hetao_ag

# 安装基础版本(仅numpy依赖)
pip install -e .
```

### 完整安装

```bash
# 安装所有可选依赖
pip install -e ".[full]"
```

### 按需安装

```bash
# 遥感模块支持
pip install -e ".[space]"

# 畜牧AI模块支持
pip install -e ".[livestock]"

# 优化模块支持
pip install -e ".[opt]"
```

### 验证安装

```bash
python -c "import hetao_ag; print(hetao_ag.__version__)"
# 输出: 1.0.0
```

---

## 快速开始

### 5 分钟入门

```python
import hetao_ag as hag

# 1. 计算参考蒸散发
from hetao_ag.water import eto_penman_monteith, WeatherData

weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)
et0 = eto_penman_monteith(weather)
print(f"参考蒸散发: {et0:.2f} mm/day")

# 2. 土壤水分模拟
from hetao_ag.soil import SoilMoistureModel

soil = SoilMoistureModel(field_capacity=0.32, wilting_point=0.12)
result = soil.step_day(rain_mm=15, et_mm=5)
print(f"土壤含水量: {result['moisture']:.3f}")

# 3. 盐分胁迫计算
from hetao_ag.crop import yield_reduction_salinity_crop

rel_yield = yield_reduction_salinity_crop(ECe=6.0, crop="wheat")
print(f"小麦相对产量: {rel_yield*100:.1f}%")

# 4. 遥感NDVI计算
import numpy as np
from hetao_ag.space import compute_ndvi

red = np.array([[120], [110]])
nir = np.array([[200], [180]])
ndvi = compute_ndvi(red, nir)
print(f"NDVI: {ndvi.mean():.3f}")
```

### 运行示例

```bash
# 运行综合演示
python examples/demo.py

# 运行特定模块示例
python examples/example_core.py
python examples/example_soil.py
python examples/example_water.py
python examples/example_crop.py
python examples/example_livestock.py
python examples/example_space.py
python examples/example_opt.py
```

---

## 模块详解

### Core 模块 - 核心基础设施

#### 单位系统

```python
from hetao_ag.core import Quantity, Unit, meters, hectares, celsius

# 创建物理量
length = meters(100)
area = hectares(50)
temp = celsius(25)

# 单位转换
length_km = length.to(Unit.KILOMETER)
temp_k = temp.to(Unit.KELVIN)

# 算术运算(自动单位处理)
total = meters(100) + Quantity(0.5, Unit.KILOMETER)  # 600m
```

#### 配置管理

```python
from hetao_ag.core import ConfigManager, create_default_config

# 加载配置
config = ConfigManager(defaults=create_default_config())

# 获取配置项(支持嵌套键)
fc = config.get("soil.field_capacity")
kc = config.get("crop.kc_mid", default=1.15)

# 设置配置项
config.set("irrigation.efficiency", 0.90)
```

#### 日志系统

```python
from hetao_ag.core import get_logger

logger = get_logger("my_experiment")
logger.info("实验开始", soil_ec=4.5, irrigation=True)
logger.log_experiment_start("产量预测", parameters={"model": "CropModel"})
```

---

### Soil 模块 - 土壤科学

#### 土壤水分模型

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

# 创建模型
model = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

# 逐日模拟
result = model.step_day(rain_mm=20, irrigation_mm=0, et_mm=5)

# 关键属性
print(model.moisture)        # 当前含水量
print(model.stress_factor)   # 水分胁迫因子(0-1)
print(model.irrigation_need_mm)  # 需灌溉量
```

#### 土壤盐分模型

```python
from hetao_ag.soil import SalinityModel, classify_soil_salinity

model = SalinityModel(initial_ECe=4.0)

# 灌溉带入盐分
model.irrigate(amount_mm=60, ec_water=1.5)

# 淋洗降低盐分
model.leach(drainage_mm=40)

# 盐分分级
print(classify_soil_salinity(model.ECe))
```

#### 传感器校准

```python
from hetao_ag.soil import SensorCalibrator
import numpy as np

calibrator = SensorCalibrator()
result = calibrator.linear_calibration(
    raw_readings=np.array([300, 450, 600]),
    ground_truth=np.array([0.10, 0.20, 0.30])
)

# 应用校准
calibrated_value = result.apply(raw_value=500)
```

---

### Water 模块 - 水循环管理

#### FAO-56 Penman-Monteith

```python
from hetao_ag.water import eto_penman_monteith, WeatherData, crop_coefficient

weather = WeatherData(
    t_mean=25, t_max=32, t_min=18,
    rh=55, u2=2.0, rs=22,
    elevation=1050, latitude=40.8, doy=180
)

# 参考蒸散发
et0 = eto_penman_monteith(weather)

# 作物蒸散发
kc = crop_coefficient("mid", "wheat")
etc = et0 * kc
```

#### 水量平衡

```python
from hetao_ag.water import WaterBalance

wb = WaterBalance(initial_storage_mm=80, max_storage_mm=120)

# 逐日模拟
wb.step_day(precip_mm=15, et_mm=5)

# 获取汇总
summary = wb.get_summary()
print(wb.available_water)  # 可用水分
print(wb.deficit_mm)       # 水分亏缺
```

#### 灌溉调度

```python
from hetao_ag.water import IrrigationScheduler, ScheduleType

scheduler = IrrigationScheduler(
    method=ScheduleType.SOIL_MOISTURE,
    trigger_threshold=0.5
)

# 获取灌溉建议
rec = scheduler.recommend_by_moisture(
    current_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12
)

if rec.should_irrigate:
    print(f"建议灌溉 {rec.amount_mm:.1f} mm")
```

---

### Crop 模块 - 作物科学

#### 盐分胁迫

```python
from hetao_ag.crop import yield_reduction_salinity_crop, classify_salt_tolerance

# 计算相对产量
rel_yield = yield_reduction_salinity_crop(ECe=8.0, crop="wheat")

# 作物耐盐分级
tolerance = classify_salt_tolerance("wheat")  # "中度耐盐"
```

#### 物候期跟踪

```python
from hetao_ag.crop import PhenologyTracker

tracker = PhenologyTracker("wheat")

# 累积积温
for day in range(120):
    tracker.accumulate_gdd(t_max=25, t_min=15)

print(tracker.accumulated_gdd)    # 累积积温
print(tracker.current_stage)      # 当前生长阶段
print(tracker.progress_to_maturity())  # 成熟进度
```

#### 作物生长模型

```python
from hetao_ag.crop import CropModel

model = CropModel("wheat")

for day in range(100):
    model.update_daily(
        t_max=25, t_min=15, et=5,
        soil_moisture=0.25, ECe=3.0
    )

print(model.estimate_yield())  # 预估产量
```

---

### Livestock 模块 - 畜牧监测

#### 动物检测

```python
from hetao_ag.livestock import AnimalDetector

detector = AnimalDetector(confidence_threshold=0.5)
detections = detector.detect("farm_image.jpg")
counts = detector.count_animals("farm_image.jpg")
```

#### 行为分类

```python
from hetao_ag.livestock import BehaviorClassifier, AnimalBehavior

classifier = BehaviorClassifier()
behavior = classifier.classify_from_motion(motion_magnitude=0.3, head_position="down")
# 输出: AnimalBehavior.GRAZING
```

#### 健康监测

```python
from hetao_ag.livestock import HealthMonitor

monitor = HealthMonitor("cow_001")

# 建立基线
for _ in range(7):
    monitor.update_activity(100)
    monitor.update_feeding_time(240)

# 检测异常
monitor.update_activity(60)  # 活动量下降
alerts = monitor.check_health()
```

---

### Space 模块 - 遥感分析

#### 光谱指数

```python
from hetao_ag.space import compute_ndvi, compute_savi, classify_vegetation_health
import numpy as np

red = np.array([[120, 130], [110, 90]])
nir = np.array([[200, 210], [180, 160]])

ndvi = compute_ndvi(red, nir)
savi = compute_savi(red, nir, L=0.5)

health = classify_vegetation_health(0.65)  # "茂密植被"
```

#### 物候分类

```python
from hetao_ag.space import PhenologyClassifier

# ndvi_series: (time, height, width)
classifier = PhenologyClassifier(ndvi_series)
crop_map = classifier.classify_crops(n_classes=3)
```

---

### Opt 模块 - 优化决策

#### 线性规划

```python
from hetao_ag.opt import optimize_crop_mix

crops = [
    {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
    {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
]

solution = optimize_crop_mix(crops, total_land=100, total_water=300000)
```

#### 遗传算法

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

def my_fitness(x):
    return -sum(xi**2 for xi in x)

optimizer = GeneticOptimizer(
    fitness_func=my_fitness,
    n_vars=3,
    bounds=[(-5, 5)] * 3,
    config=GAConfig(generations=100)
)
result = optimizer.optimize()
```

#### 场景分析

```python
from hetao_ag.opt import ScenarioEvaluator

evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=500000)
scenario = evaluator.evaluate_scenario("方案A", {"wheat": 60}, 500)
comparison = evaluator.compare_scenarios()
```

---

## API 参考

完整 API 文档请参考各模块的 docstring:

```python
from hetao_ag.water import eto_penman_monteith
help(eto_penman_monteith)
```

---

## 最佳实践

### 1. 始终使用单位系统

```python
# 推荐
from hetao_ag.core import meters, hectares
area = hectares(50)

# 不推荐
area = 50  # 单位不明确
```

### 2. 配置优于硬编码

```python
# 推荐
config = ConfigManager(config_file="farm_config.yaml")
fc = config.get("soil.field_capacity")

# 不推荐
fc = 0.32  # 硬编码值难以维护
```

### 3. 记录实验参数

```python
logger = get_logger("experiment")
logger.log_experiment_start(
    "灌溉优化",
    parameters={"method": "deficit", "target_etc": 0.8},
    random_seed=42
)
```

### 4. 验证模型结果

```python
from hetao_ag.core import validate_model
result = validate_model(observed, predicted)
print(f"RMSE: {result.rmse:.4f}, R²: {result.r_squared:.4f}")
```

---

## 常见问题

### Q: numpy 未安装怎么办？

```bash
pip install numpy
```

### Q: 如何获取特定作物的参数？

```python
from hetao_ag.crop import CROP_SALT_TOLERANCE, CROP_PHENOLOGY

# 盐分耐受性
tolerance = CROP_SALT_TOLERANCE["wheat"]
print(tolerance.threshold, tolerance.slope)

# 物候参数
phenology = CROP_PHENOLOGY["wheat"]
print(phenology.base_temperature, phenology.stage_gdd)
```

### Q: 如何处理缺失的气象数据？

如果缺少太阳辐射数据，可使用 Hargreaves 方法:

```python
from hetao_ag.water import eto_hargreaves, extraterrestrial_radiation

Ra = extraterrestrial_radiation(latitude=40.8, doy=180)
et0 = eto_hargreaves(t_mean=25, t_max=32, t_min=18, Ra=Ra)
```

---

## 联系方式

- 项目主页: https://github.com/hetao-college/hetao_ag
- 问题反馈: https://github.com/hetao-college/hetao_ag/issues
- 邮箱: hetao@example.com

---

_河套智慧农牧业库 v1.0.0_
_© 2024 Hetao College_
