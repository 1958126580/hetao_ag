# -*- coding: utf-8 -*-
"""
hetao_ag.opt - 优化模块

农业资源优化和决策支持。

作者: Hetao College
版本: 1.0.0
"""

from .linear import (
    LinearOptimizer,
    OptimizationResult,
    optimize_crop_mix,
)

from .genetic import (
    GeneticOptimizer,
    GAConfig,
    GAResult,
    optimize_irrigation_schedule,
)

from .planning import (
    ScenarioEvaluator,
    FarmScenario,
    multi_objective_score,
)

__all__ = [
    # linear
    "LinearOptimizer",
    "OptimizationResult",
    "optimize_crop_mix",
    # genetic
    "GeneticOptimizer",
    "GAConfig",
    "GAResult",
    "optimize_irrigation_schedule",
    # planning
    "ScenarioEvaluator",
    "FarmScenario",
    "multi_objective_score",
]
