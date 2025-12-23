# -*- coding: utf-8 -*-
"""
Opt模块使用示例
===============

演示hetao_ag.opt模块的农业优化和决策支持功能。

作者: Hetao College
"""

import numpy as np
from hetao_ag.opt import (
    # 线性规划
    LinearOptimizer, OptimizationResult, optimize_crop_mix,
    # 遗传算法
    GeneticOptimizer, GAConfig, GAResult, optimize_irrigation_schedule,
    # 农场规划
    ScenarioEvaluator, FarmScenario, multi_objective_score
)


def example_linear_optimization():
    """
    示例1: 线性规划优化
    ===================
    
    使用线性规划解决农业资源分配问题。
    """
    print("\n" + "=" * 60)
    print("示例1: 线性规划优化")
    print("=" * 60)
    
    # 1.1 问题描述
    print("\n【问题描述】")
    print("  某农场有100公顷土地和30万立方米水资源")
    print("  可种植小麦、玉米、苜蓿三种作物")
    print("  目标: 最大化总利润")
    
    # 1.2 作物参数
    crops = [
        {"name": "wheat", "profit_per_ha": 800, "water_per_ha": 3500},
        {"name": "maize", "profit_per_ha": 1000, "water_per_ha": 5500},
        {"name": "alfalfa", "profit_per_ha": 600, "water_per_ha": 2500},
    ]
    
    print("\n【作物参数】")
    print(f"{'作物':>10} {'利润(元/ha)':>14} {'需水量(m³/ha)':>16}")
    print("-" * 44)
    for c in crops:
        print(f"{c['name']:>10} {c['profit_per_ha']:>14} {c['water_per_ha']:>16}")
    
    # 1.3 使用便捷函数优化
    solution = optimize_crop_mix(
        crops=crops,
        total_land=100,
        total_water=300000
    )
    
    print("\n【优化结果】")
    total_profit = 0
    total_water = 0
    for crop in crops:
        area = solution.get(crop['name'], 0)
        if area and area > 0:
            profit = area * crop['profit_per_ha']
            water = area * crop['water_per_ha']
            total_profit += profit
            total_water += water
            print(f"  {crop['name']}: {area:.1f} ha (利润: {profit:.0f}元, 用水: {water:.0f}m³)")
    
    print(f"\n  总利润: {total_profit:.0f} 元")
    print(f"  总用水: {total_water:.0f} m³")
    
    # 1.4 使用LinearOptimizer类(更灵活)
    print("\n【使用LinearOptimizer类】")
    
    optimizer = LinearOptimizer()
    
    # 添加决策变量
    optimizer.add_variable("wheat", lower=0, upper=100)
    optimizer.add_variable("maize", lower=0, upper=100)
    optimizer.add_variable("alfalfa", lower=0, upper=100)
    
    # 设置目标函数(最大化利润)
    optimizer.set_objective(
        {"wheat": 800, "maize": 1000, "alfalfa": 600},
        maximize=True
    )
    
    # 添加约束
    optimizer.add_constraint(
        {"wheat": 1, "maize": 1, "alfalfa": 1}, "<=", 100  # 土地约束
    )
    optimizer.add_constraint(
        {"wheat": 3500, "maize": 5500, "alfalfa": 2500}, "<=", 300000  # 水资源约束
    )
    optimizer.add_constraint(
        {"wheat": 1}, ">=", 20  # 小麦最少种20ha
    )
    
    result = optimizer.solve()
    
    print(f"  求解状态: {result.status}")
    print(f"  目标函数值: {result.objective_value:.0f} 元")
    print(f"  决策变量:")
    for name, value in result.variables.items():
        if value and value > 0:
            print(f"    {name}: {value:.1f} ha")


def example_genetic_algorithm():
    """
    示例2: 遗传算法优化
    ===================
    
    使用遗传算法解决复杂的非线性优化问题。
    """
    print("\n" + "=" * 60)
    print("示例2: 遗传算法优化")
    print("=" * 60)
    
    # 2.1 简单函数优化
    print("\n【函数优化示例】")
    print("  目标: 最小化 f(x,y) = x² + y²")
    print("  搜索范围: x,y ∈ [-10, 10]")
    
    def sphere_function(x):
        """球函数(全局最小值在原点)"""
        return -sum(xi**2 for xi in x)  # 负号因为GA是最大化
    
    # 配置遗传算法
    config = GAConfig(
        population_size=50,   # 种群大小
        generations=100,      # 迭代代数
        crossover_rate=0.8,   # 交叉概率
        mutation_rate=0.1,    # 变异概率
        elitism=2,            # 精英保留数
        tournament_size=3     # 锦标赛选择大小
    )
    
    optimizer = GeneticOptimizer(
        fitness_func=sphere_function,
        n_vars=2,
        bounds=[(-10, 10), (-10, 10)],
        config=config
    )
    
    result = optimizer.optimize()
    
    print(f"\n【优化结果】")
    print(f"  最优解: x={result.best_solution[0]:.4f}, y={result.best_solution[1]:.4f}")
    print(f"  目标函数值: {-result.best_fitness:.6f}")  # 转回正值
    print(f"  迭代代数: {result.generations_run}")
    
    # 2.2 高维优化
    print("\n【高维优化示例】")
    print("  目标: 10维Rosenbrock函数")
    
    def rosenbrock(x):
        """Rosenbrock函数"""
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return -total  # 最大化负值 = 最小化正值
    
    optimizer_10d = GeneticOptimizer(
        fitness_func=rosenbrock,
        n_vars=10,
        bounds=[(-5, 5)] * 10,
        config=GAConfig(population_size=100, generations=200)
    )
    
    result_10d = optimizer_10d.optimize()
    
    print(f"  最优解 (前3维): {result_10d.best_solution[:3]}")
    print(f"  目标函数值: {-result_10d.best_fitness:.2f}")
    
    # 2.3 灌溉计划优化
    print("\n【灌溉计划优化】")
    
    # 模拟30天的ET需求
    daily_et = np.array([4.0 + 2.0 * np.sin(i/30 * np.pi) for i in range(30)])
    
    print(f"  模拟周期: 30天")
    print(f"  平均日ET: {daily_et.mean():.1f} mm")
    print(f"  总ET需求: {daily_et.sum():.0f} mm")
    
    schedule, total_irrigation = optimize_irrigation_schedule(
        daily_et=daily_et,
        max_irrigation=40,
        min_interval=3
    )
    
    irrigation_days = np.where(schedule > 0)[0]
    
    print(f"\n  优化灌溉计划:")
    print(f"  灌溉天数: {len(irrigation_days)}天")
    print(f"  灌溉日期: {irrigation_days + 1}")
    print(f"  总灌溉量: {total_irrigation:.0f} mm")


def example_farm_planning():
    """
    示例3: 农场规划与场景分析
    =========================
    
    评估和比较不同农场管理方案。
    """
    print("\n" + "=" * 60)
    print("示例3: 农场规划与场景分析")
    print("=" * 60)
    
    # 3.1 作物参数
    crop_params = {
        "wheat": {
            "yield_kg_ha": 6000,
            "water_need_mm": 450,
            "price_per_kg": 2.5,
            "cost_per_ha": 3000
        },
        "maize": {
            "yield_kg_ha": 10000,
            "water_need_mm": 600,
            "price_per_kg": 2.0,
            "cost_per_ha": 4000
        },
        "cotton": {
            "yield_kg_ha": 4500,
            "water_need_mm": 700,
            "price_per_kg": 7.0,
            "cost_per_ha": 5000
        },
    }
    
    print("\n【作物参数】")
    print(f"{'作物':>8} {'产量':>10} {'需水':>8} {'价格':>8} {'成本':>8}")
    print("-" * 50)
    for crop, params in crop_params.items():
        print(f"{crop:>8} {params['yield_kg_ha']:>10} {params['water_need_mm']:>8} "
              f"{params['price_per_kg']:>8.1f} {params['cost_per_ha']:>8}")
    
    # 3.2 创建场景评估器
    evaluator = ScenarioEvaluator(
        crop_params=crop_params,
        total_land=100,
        total_water=500000
    )
    
    # 3.3 定义多个方案
    scenarios = [
        ("全小麦", {"wheat": 100}, 500),
        ("全玉米", {"maize": 100}, 650),
        ("全棉花", {"cotton": 100}, 750),
        ("均衡种植", {"wheat": 40, "maize": 40, "cotton": 20}, 550),
        ("粮食优先", {"wheat": 60, "maize": 40}, 520),
        ("经济作物", {"wheat": 30, "cotton": 70}, 650),
    ]
    
    print("\n【方案评估结果】")
    print(f"{'方案':>12} {'利润(万元)':>12} {'WUE':>8}")
    print("-" * 36)
    
    for name, areas, irrig in scenarios:
        scenario = evaluator.evaluate_scenario(name, areas, irrig)
        profit_wan = scenario.total_profit / 10000
        print(f"{name:>12} {profit_wan:>12.1f} {scenario.water_use_efficiency:>8.3f}")
    
    # 3.4 方案比较
    comparison = evaluator.compare_scenarios()
    
    print("\n【最优方案】")
    print(f"  最高利润方案: {comparison['best_profit'].name}")
    print(f"    利润: {comparison['best_profit'].total_profit/10000:.1f} 万元")
    print(f"  最优水效方案: {comparison['best_water_efficiency'].name}")
    print(f"    WUE: {comparison['best_water_efficiency'].water_use_efficiency:.3f}")
    
    # 3.5 敏感性分析
    print("\n【敏感性分析 - 灌溉量变化】")
    
    base_scenario = evaluator.scenarios[3]  # 均衡种植
    variations = [-0.2, -0.1, 0, 0.1, 0.2]
    
    print(f"  基准方案: {base_scenario.name}")
    print(f"  基准灌溉: {base_scenario.irrigation_mm} mm")
    print(f"\n{'变化':>8} {'灌溉量':>10} {'利润(万元)':>14}")
    print("-" * 36)
    
    sensitivity_results = evaluator.sensitivity_analysis(
        base_scenario, "irrigation", variations
    )
    
    for var, result in zip(variations, sensitivity_results):
        profit_wan = result.total_profit / 10000
        print(f"{var:>+8.0%} {result.irrigation_mm:>10.0f} {profit_wan:>14.1f}")
    
    # 3.6 多目标评分
    print("\n【多目标综合评分】")
    
    # 归一化指标
    profits = [s.total_profit for s in evaluator.scenarios]
    wues = [s.water_use_efficiency for s in evaluator.scenarios]
    
    max_profit = max(profits)
    max_wue = max(wues)
    
    print(f"{'方案':>12} {'利润分':>8} {'水效分':>8} {'综合分':>8}")
    print("-" * 40)
    
    for scenario in evaluator.scenarios:
        profit_score = scenario.total_profit / max_profit
        wue_score = scenario.water_use_efficiency / max_wue
        sustainability = 0.8  # 假设固定
        
        total_score = multi_objective_score(
            profit=profit_score,
            water_use=wue_score,
            sustainability=sustainability,
            weights={"profit": 0.4, "water": 0.4, "sustainability": 0.2}
        )
        
        print(f"{scenario.name:>12} {profit_score:>8.2f} {wue_score:>8.2f} {total_score:>8.2f}")


def example_integrated_optimization():
    """
    示例4: 综合优化案例
    ===================
    
    结合多种方法解决实际问题。
    """
    print("\n" + "=" * 60)
    print("示例4: 综合优化案例")
    print("=" * 60)
    
    print("\n【问题】")
    print("  河套灌区某家庭农场优化")
    print("  - 土地: 50公顷")
    print("  - 水权: 25万m³/年")
    print("  - 目标: 最大化经济效益和水资源效率")
    
    # 4.1 第一阶段: 线性规划确定种植结构
    print("\n【阶段1: 种植结构优化】")
    
    crops = [
        {"name": "spring_wheat", "profit_per_ha": 750, "water_per_ha": 3200},
        {"name": "sunflower", "profit_per_ha": 900, "water_per_ha": 3800},
        {"name": "sugar_beet", "profit_per_ha": 1100, "water_per_ha": 4500},
    ]
    
    solution = optimize_crop_mix(crops, total_land=50, total_water=200000)
    
    print("  优化种植面积:")
    for c in crops:
        area = solution.get(c['name'], 0)
        if area:
            print(f"    {c['name']}: {area:.1f} ha")
    
    # 4.2 第二阶段: 遗传算法优化灌溉策略
    print("\n【阶段2: 灌溉策略优化】")
    
    def irrigation_fitness(params):
        """灌溉参数适应度函数"""
        irrigation_amount = params[0] * 100  # 0-100 -> 0-100mm
        trigger_threshold = params[1]  # 0-1
        
        # 简化的产量-用水模型
        if irrigation_amount < 30:
            yield_ratio = irrigation_amount / 30 * 0.6
        elif irrigation_amount < 60:
            yield_ratio = 0.6 + (irrigation_amount - 30) / 30 * 0.35
        else:
            yield_ratio = 0.95 - (irrigation_amount - 60) / 40 * 0.1
        
        water_saved = (100 - irrigation_amount) / 100 * 0.3
        yield_ratio = min(1.0, yield_ratio + trigger_threshold * 0.1)
        
        # 综合得分
        return 0.7 * yield_ratio + 0.3 * water_saved
    
    optimizer = GeneticOptimizer(
        fitness_func=irrigation_fitness,
        n_vars=2,
        bounds=[(0.3, 0.8), (0.4, 0.7)],
        config=GAConfig(generations=50)
    )
    
    result = optimizer.optimize()
    
    optimal_irrigation = result.best_solution[0] * 100
    optimal_trigger = result.best_solution[1]
    
    print(f"  最优灌溉定额: {optimal_irrigation:.0f} mm/次")
    print(f"  最优触发阈值: {optimal_trigger:.2f} (可用水消耗比)")
    print(f"  综合得分: {result.best_fitness:.3f}")
    
    # 4.3 总结
    print("\n【优化方案总结】")
    print("  ┌────────────────────────────────────┐")
    print("  │ 春小麦: 按优化面积种植             │")
    print("  │ 向日葵: 按优化面积种植             │")
    print(f"  │ 灌溉量: {optimal_irrigation:.0f}mm/次                    │")
    print(f"  │ 灌溉触发: 可用水消耗{optimal_trigger*100:.0f}%时         │")
    print("  └────────────────────────────────────┘")


if __name__ == "__main__":
    print("=" * 60)
    print("  hetao_ag.opt 模块使用示例")
    print("=" * 60)
    
    example_linear_optimization()
    example_genetic_algorithm()
    example_farm_planning()
    example_integrated_optimization()
    
    print("\n" + "=" * 60)
    print("Opt模块示例完成")
    print("=" * 60)
