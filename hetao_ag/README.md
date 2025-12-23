# 河套智慧农牧业库 (hetao_ag)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

面向智慧农业和畜牧业的综合 Python 库，涵盖土壤建模、水循环管理、作物生长模拟、畜牧监测、遥感分析和农场优化。

## 特性

- 🌱 **土壤模块** - 水分和盐分建模、IoT 传感器校准
- 💧 **水循环模块** - FAO-56 Penman-Monteith 蒸散发、智能灌溉调度
- 🌾 **作物模块** - 生长模拟、水分/盐分胁迫响应、物候期跟踪
- 🐄 **畜牧模块** - 基于 AI 的动物检测和健康监测
- 🛰️ **遥感模块** - NDVI/SAVI/LSWI 指数计算、物候分类
- 📊 **优化模块** - 线性规划和遗传算法农场优化

## 安装

```bash
# 基础安装
pip install -e .

# 完整安装
pip install -e ".[full]"

# 包含遥感支持
pip install -e ".[space]"

# 包含畜牧AI支持
pip install -e ".[livestock]"
```

## 快速开始

### 计算参考蒸散发

```python
from hetao_ag.water import eto_penman_monteith, WeatherData

weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)

et0 = eto_penman_monteith(weather)
print(f"参考蒸散发: {et0:.2f} mm/day")
```

### 土壤水分模拟

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

model = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

result = model.step_day(rain_mm=15, et_mm=5)
print(f"土壤含水量: {result['moisture']:.3f}")
```

### 盐分胁迫分析

```python
from hetao_ag.crop import yield_reduction_salinity_crop, classify_salt_tolerance

# 土壤EC=6 dS/m时的小麦产量
rel_yield = yield_reduction_salinity_crop(6.0, "wheat")
print(f"相对产量: {rel_yield*100:.1f}%")
print(f"耐盐等级: {classify_salt_tolerance('wheat')}")
```

### 光谱指数计算

```python
import numpy as np
from hetao_ag.space import compute_ndvi, classify_vegetation_health

red = np.array([[120, 130], [110, 90]])
nir = np.array([[200, 210], [180, 160]])

ndvi = compute_ndvi(red, nir)
print(f"平均NDVI: {ndvi.mean():.3f}")
print(f"植被状态: {classify_vegetation_health(0.65)}")
```

### 作物组合优化

```python
from hetao_ag.opt import optimize_crop_mix

crops = [
    {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
    {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
]

solution = optimize_crop_mix(crops, total_land=100, total_water=300000)
for crop, area in solution.items():
    print(f"{crop}: {area:.1f} ha")
```

## 模块结构

```
hetao_ag/
├── core/          # 核心工具：单位系统、配置、日志
├── soil/          # 土壤水分和盐分建模
├── water/         # 蒸散发计算和灌溉调度
├── crop/          # 作物生长模拟和胁迫响应
├── livestock/     # 畜牧监测和健康预警
├── space/         # 遥感分析和物候分类
└── opt/           # 农场优化和决策支持
```

## 运行示例

```bash
python examples/demo.py
```

## 技术规范

- **Python**: 3.10+
- **代码规范**: PEP 8
- **单位系统**: SI 国际标准
- **时间格式**: ISO 8601

## 参考文献

- Allen et al. (1998) FAO Irrigation & Drainage Paper 56
- Maas & Hoffman (1977) 作物盐分耐受性模型

## 许可证

MIT License

## 作者

Hetao College
