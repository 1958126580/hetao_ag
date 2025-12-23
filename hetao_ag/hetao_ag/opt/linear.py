# -*- coding: utf-8 -*-
"""
hetao_ag.opt.linear - 线性规划

农业资源优化的线性规划工具。

作者: Hetao College
版本: 1.0.0
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class OptimizationResult:
    """优化结果"""
    status: str
    objective_value: float
    variables: Dict[str, float]
    message: str = ""


class LinearOptimizer:
    """线性规划优化器
    
    示例:
        >>> optimizer = LinearOptimizer()
        >>> optimizer.add_variable("wheat_area", 0, 100)
        >>> optimizer.add_variable("maize_area", 0, 100)
        >>> optimizer.set_objective({"wheat_area": 500, "maize_area": 600}, maximize=True)
        >>> optimizer.add_constraint({"wheat_area": 1, "maize_area": 1}, "<=", 150)
        >>> result = optimizer.solve()
    """
    
    def __init__(self):
        self.variables: Dict[str, tuple] = {}  # name: (lower, upper)
        self.objective: Dict[str, float] = {}
        self.maximize = True
        self.constraints: List[tuple] = []  # (coeffs, sense, rhs)
    
    def add_variable(self, name: str, lower: float = 0, upper: float = None):
        """添加决策变量"""
        self.variables[name] = (lower, upper)
    
    def set_objective(self, coefficients: Dict[str, float], maximize: bool = True):
        """设置目标函数"""
        self.objective = coefficients
        self.maximize = maximize
    
    def add_constraint(self, coefficients: Dict[str, float], sense: str, rhs: float):
        """添加约束
        
        参数:
            coefficients: 变量系数
            sense: "<=", ">=", "=="
            rhs: 右侧常数
        """
        self.constraints.append((coefficients, sense, rhs))
    
    def solve(self) -> OptimizationResult:
        """求解优化问题"""
        try:
            import pulp
            return self._solve_pulp()
        except ImportError:
            return self._solve_simple()
    
    def _solve_pulp(self) -> OptimizationResult:
        """使用PuLP求解"""
        import pulp
        
        sense = pulp.LpMaximize if self.maximize else pulp.LpMinimize
        prob = pulp.LpProblem("AgOptimization", sense)
        
        # 创建变量
        lp_vars = {}
        for name, (lower, upper) in self.variables.items():
            lp_vars[name] = pulp.LpVariable(name, lowBound=lower, upBound=upper)
        
        # 目标函数
        prob += pulp.lpSum(coeff * lp_vars[name] for name, coeff in self.objective.items())
        
        # 约束
        for coeffs, sense_str, rhs in self.constraints:
            expr = pulp.lpSum(c * lp_vars[n] for n, c in coeffs.items())
            if sense_str == "<=":
                prob += expr <= rhs
            elif sense_str == ">=":
                prob += expr >= rhs
            else:
                prob += expr == rhs
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return OptimizationResult(
            status=pulp.LpStatus[prob.status],
            objective_value=pulp.value(prob.objective),
            variables={name: var.value() for name, var in lp_vars.items()},
            message=""
        )
    
    def _solve_simple(self) -> OptimizationResult:
        """简单求解(无PuLP时)"""
        # 返回边界可行解作为近似
        variables = {}
        for name, (lower, upper) in self.variables.items():
            if self.maximize:
                variables[name] = upper if upper else 100
            else:
                variables[name] = lower
        
        obj_value = sum(self.objective.get(n, 0) * v for n, v in variables.items())
        
        return OptimizationResult(
            status="Approximate",
            objective_value=obj_value,
            variables=variables,
            message="PuLP未安装,使用简单近似"
        )


def optimize_crop_mix(
    crops: List[Dict],
    total_land: float,
    total_water: float
) -> Dict[str, float]:
    """优化作物组合
    
    参数:
        crops: [{"name", "profit_per_ha", "water_per_ha"}, ...]
        total_land: 总土地(ha)
        total_water: 总水量(m³)
        
    返回:
        各作物面积
    """
    optimizer = LinearOptimizer()
    
    for crop in crops:
        optimizer.add_variable(crop["name"], 0, total_land)
    
    # 目标: 最大化利润
    optimizer.set_objective(
        {c["name"]: c["profit_per_ha"] for c in crops},
        maximize=True
    )
    
    # 约束: 总面积
    optimizer.add_constraint({c["name"]: 1 for c in crops}, "<=", total_land)
    
    # 约束: 总水量
    optimizer.add_constraint(
        {c["name"]: c["water_per_ha"] for c in crops},
        "<=",
        total_water
    )
    
    result = optimizer.solve()
    return result.variables


if __name__ == "__main__":
    print("=" * 50)
    print("线性规划优化演示")
    print("=" * 50)
    
    crops = [
        {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
        {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
        {"name": "alfalfa", "profit_per_ha": 400, "water_per_ha": 2000},
    ]
    
    solution = optimize_crop_mix(crops, total_land=100, total_water=300000)
    
    print("\n优化种植方案:")
    for crop, area in solution.items():
        print(f"  {crop}: {area:.1f} ha")
