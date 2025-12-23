# -*- coding: utf-8 -*-
"""
hetao_ag.opt.genetic - 遗传算法

农业优化问题的遗传算法实现。

作者: Hetao College
版本: 1.0.0
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple


@dataclass
class GAConfig:
    """遗传算法配置"""
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism: int = 2
    tournament_size: int = 3


@dataclass
class GAResult:
    """遗传算法结果"""
    best_solution: List[float]
    best_fitness: float
    generations_run: int
    fitness_history: List[float]


class GeneticOptimizer:
    """遗传算法优化器
    
    示例:
        >>> def fitness(x):
        ...     return -(x[0]**2 + x[1]**2)
        >>> optimizer = GeneticOptimizer(fitness, n_vars=2, bounds=[(-5,5), (-5,5)])
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        fitness_func: Callable[[List[float]], float],
        n_vars: int,
        bounds: List[Tuple[float, float]],
        config: Optional[GAConfig] = None
    ):
        self.fitness_func = fitness_func
        self.n_vars = n_vars
        self.bounds = bounds
        self.config = config or GAConfig()
        
        self.population: List[List[float]] = []
        self.fitness_values: List[float] = []
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = [
                random.uniform(low, high)
                for low, high in self.bounds
            ]
            self.population.append(individual)
    
    def _evaluate_population(self):
        """评估种群适应度"""
        self.fitness_values = [self.fitness_func(ind) for ind in self.population]
    
    def _tournament_select(self) -> List[float]:
        """锦标赛选择"""
        indices = random.sample(range(len(self.population)), self.config.tournament_size)
        best_idx = max(indices, key=lambda i: self.fitness_values[i])
        return self.population[best_idx].copy()
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """交叉操作"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        point = random.randint(1, self.n_vars - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """变异操作"""
        result = individual.copy()
        for i in range(self.n_vars):
            if random.random() < self.config.mutation_rate:
                low, high = self.bounds[i]
                result[i] = random.uniform(low, high)
        return result
    
    def _clip_bounds(self, individual: List[float]) -> List[float]:
        """边界约束"""
        return [
            max(low, min(high, val))
            for val, (low, high) in zip(individual, self.bounds)
        ]
    
    def optimize(self) -> GAResult:
        """运行优化"""
        self._initialize_population()
        self._evaluate_population()
        
        fitness_history = []
        
        for gen in range(self.config.generations):
            # 记录当代最佳
            best_fitness = max(self.fitness_values)
            fitness_history.append(best_fitness)
            
            # 精英保留
            elite_indices = sorted(range(len(self.fitness_values)),
                                  key=lambda i: self.fitness_values[i],
                                  reverse=True)[:self.config.elitism]
            new_population = [self.population[i].copy() for i in elite_indices]
            
            # 生成新个体
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                child1 = self._clip_bounds(child1)
                child2 = self._clip_bounds(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
            self._evaluate_population()
        
        # 最终结果
        best_idx = max(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i])
        
        return GAResult(
            best_solution=self.population[best_idx],
            best_fitness=self.fitness_values[best_idx],
            generations_run=self.config.generations,
            fitness_history=fitness_history
        )


def optimize_irrigation_schedule(
    daily_et: np.ndarray,
    max_irrigation: float = 50,
    min_interval: int = 3
) -> Tuple[np.ndarray, float]:
    """优化灌溉计划
    
    参数:
        daily_et: 逐日ET需求
        max_irrigation: 单次最大灌溉量
        min_interval: 最小灌溉间隔
        
    返回:
        (灌溉计划, 总水量)
    """
    n_days = len(daily_et)
    
    def fitness(schedule):
        # 评估灌溉计划
        soil_water = 100  # 初始土壤水分
        stress_days = 0
        total_irrig = 0
        
        for i, (et, irrig) in enumerate(zip(daily_et, schedule)):
            soil_water += irrig * max_irrigation
            soil_water -= et
            total_irrig += irrig * max_irrigation
            
            if soil_water < 30:  # 胁迫阈值
                stress_days += 1
                soil_water = max(0, soil_water)
        
        return -stress_days - total_irrig * 0.01
    
    bounds = [(0, 1) for _ in range(n_days)]
    optimizer = GeneticOptimizer(fitness, n_days, bounds, GAConfig(generations=50))
    result = optimizer.optimize()
    
    schedule = np.array(result.best_solution)
    schedule = (schedule > 0.5).astype(float) * max_irrigation
    
    return schedule, np.sum(schedule)


if __name__ == "__main__":
    print("=" * 50)
    print("遗传算法优化演示")
    print("=" * 50)
    
    # 测试函数: 球函数最小化
    def sphere(x):
        return -sum(xi**2 for xi in x)
    
    bounds = [(-5, 5) for _ in range(3)]
    optimizer = GeneticOptimizer(sphere, 3, bounds, GAConfig(generations=50))
    result = optimizer.optimize()
    
    print(f"\n最优解: {[f'{x:.4f}' for x in result.best_solution]}")
    print(f"最优适应度: {result.best_fitness:.6f}")
    print(f"运行代数: {result.generations_run}")
