"""
Ration Formulation Module

Feed ration optimization and formulation:
- Linear programming optimization
- Least-cost formulation
- Constraint management
- Ration analysis

Example:
    >>> optimizer = RationOptimizer(species="cattle")
    >>> ration = optimizer.formulate_least_cost(
    ...     requirements=requirements,
    ...     available_feeds=["corn_grain", "soybean_meal", "alfalfa_hay"]
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .nutrition import (
    NutrientRequirements,
    NutrientCalculator,
    NutrientAnalyzer,
    FeedAnalysis,
    FEED_DATABASE,
)

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Formulation constraint types."""
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"
    RANGE = "range"


@dataclass
class FeedIngredient:
    """
    Feed ingredient with constraints.

    Attributes:
        name: Ingredient name
        analysis: Nutritional analysis
        min_inclusion: Minimum inclusion (kg or %)
        max_inclusion: Maximum inclusion (kg or %)
        available: Quantity available
        fixed_amount: Fixed amount (overrides optimization)
    """
    name: str
    analysis: FeedAnalysis
    min_inclusion: float = 0.0
    max_inclusion: float = float('inf')
    available: float = float('inf')
    fixed_amount: Optional[float] = None


@dataclass
class FormulationConstraint:
    """
    Ration formulation constraint.

    Attributes:
        nutrient: Nutrient name
        constraint_type: Type of constraint
        value: Constraint value
        upper_value: Upper value (for range)
    """
    nutrient: str
    constraint_type: ConstraintType
    value: float
    upper_value: Optional[float] = None


@dataclass
class Ration:
    """
    Formulated ration.

    Attributes:
        ingredients: Dict of ingredient amounts (kg)
        total_dm: Total dry matter (kg)
        nutrients: Nutrient composition
        cost: Total cost
        analysis: Detailed analysis
    """
    ingredients: Dict[str, float]
    total_dm: float
    nutrients: Dict[str, float]
    cost: float
    analysis: Dict[str, Any] = field(default_factory=dict)


class SimplexSolver:
    """
    Simple linear programming solver.

    Implements the simplex method for least-cost
    feed formulation optimization.
    """

    def __init__(self, max_iterations: int = 1000):
        """
        Initialize solver.

        Args:
            max_iterations: Maximum iterations
        """
        self.max_iterations = max_iterations

    def solve(
        self,
        c: np.ndarray,  # Cost coefficients
        A_ub: np.ndarray,  # Inequality constraint matrix
        b_ub: np.ndarray,  # Inequality constraint bounds
        A_eq: Optional[np.ndarray] = None,  # Equality constraint matrix
        b_eq: Optional[np.ndarray] = None,  # Equality constraint bounds
        bounds: Optional[List[Tuple[float, float]]] = None,  # Variable bounds
    ) -> Dict[str, Any]:
        """
        Solve linear programming problem.

        Minimize c'x subject to:
            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            bounds[i][0] <= x[i] <= bounds[i][1]

        Args:
            c: Cost coefficients
            A_ub: Inequality constraints
            b_ub: Inequality bounds
            A_eq: Equality constraints
            b_eq: Equality bounds
            bounds: Variable bounds

        Returns:
            Dict with solution
        """
        n_vars = len(c)

        # Apply bounds as constraints
        if bounds is not None:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None and lb > 0:
                    # x[i] >= lb -> -x[i] <= -lb
                    constraint = np.zeros(n_vars)
                    constraint[i] = -1
                    A_ub = np.vstack([A_ub, constraint]) if A_ub.size else constraint.reshape(1, -1)
                    b_ub = np.append(b_ub, -lb)

                if ub is not None and ub < float('inf'):
                    # x[i] <= ub
                    constraint = np.zeros(n_vars)
                    constraint[i] = 1
                    A_ub = np.vstack([A_ub, constraint]) if A_ub.size else constraint.reshape(1, -1)
                    b_ub = np.append(b_ub, ub)

        # Convert to standard form with slack variables
        n_slack = len(b_ub) if b_ub.size else 0
        n_total = n_vars + n_slack

        # Initialize tableau
        if n_slack > 0:
            tableau = np.zeros((n_slack + 1, n_total + 1))
            tableau[:-1, :n_vars] = A_ub
            tableau[:-1, n_vars:n_vars + n_slack] = np.eye(n_slack)
            tableau[:-1, -1] = b_ub
            tableau[-1, :n_vars] = c
        else:
            tableau = np.zeros((1, n_vars + 1))
            tableau[-1, :n_vars] = c

        # Simplex iterations
        for _ in range(self.max_iterations):
            # Find pivot column (most negative in objective row)
            obj_row = tableau[-1, :-1]
            if np.all(obj_row >= -1e-10):
                break  # Optimal

            pivot_col = np.argmin(obj_row)

            # Find pivot row (minimum ratio test)
            column = tableau[:-1, pivot_col]
            rhs = tableau[:-1, -1]

            valid_rows = column > 1e-10
            if not np.any(valid_rows):
                return {"success": False, "message": "Unbounded"}

            ratios = np.full(len(column), np.inf)
            ratios[valid_rows] = rhs[valid_rows] / column[valid_rows]
            pivot_row = np.argmin(ratios)

            # Pivot
            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot_val

            for i in range(len(tableau)):
                if i != pivot_row:
                    tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        # Extract solution
        solution = np.zeros(n_vars)
        for j in range(n_vars):
            column = tableau[:-1, j]
            if np.sum(np.abs(column) > 1e-10) == 1:
                row = np.argmax(np.abs(column))
                if np.abs(column[row] - 1) < 1e-10:
                    solution[j] = tableau[row, -1]

        return {
            "success": True,
            "x": solution,
            "fun": np.dot(c, solution),
        }


class RationOptimizer:
    """
    Ration formulation and optimization.

    Uses linear programming to formulate
    least-cost rations meeting nutrient requirements.

    Example:
        >>> optimizer = RationOptimizer(species="cattle")
        >>> requirements = calc.calculate_requirements(weight=500)
        >>> ration = optimizer.formulate_least_cost(
        ...     requirements=requirements,
        ...     target_dmi=12.0,
        ...     available_feeds=["corn_grain", "soybean_meal", "alfalfa_hay"]
        ... )
    """

    def __init__(self, species: str = "cattle"):
        """
        Initialize ration optimizer.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.calculator = NutrientCalculator(species)
        self.analyzer = NutrientAnalyzer()
        self.solver = SimplexSolver()

        self._ingredients: List[FeedIngredient] = []
        self._constraints: List[FormulationConstraint] = []

    def add_ingredient(
        self,
        name: str,
        min_pct: float = 0.0,
        max_pct: float = 100.0,
        available: float = float('inf'),
        custom_analysis: Optional[FeedAnalysis] = None,
    ) -> None:
        """
        Add ingredient to formulation.

        Args:
            name: Ingredient name
            min_pct: Minimum inclusion (% of DM)
            max_pct: Maximum inclusion (% of DM)
            available: Available quantity (kg)
            custom_analysis: Custom nutritional analysis
        """
        if custom_analysis:
            analysis = custom_analysis
        else:
            analysis = self.analyzer.get_feed_analysis(name)
            if analysis is None:
                raise ValueError(f"Unknown feed: {name}")

        ingredient = FeedIngredient(
            name=name,
            analysis=analysis,
            min_inclusion=min_pct,
            max_inclusion=max_pct,
            available=available,
        )

        self._ingredients.append(ingredient)

    def add_constraint(
        self,
        nutrient: str,
        constraint_type: str,
        value: float,
        upper_value: Optional[float] = None,
    ) -> None:
        """
        Add formulation constraint.

        Args:
            nutrient: Nutrient name
            constraint_type: "minimum", "maximum", "exact", or "range"
            value: Constraint value
            upper_value: Upper value for range
        """
        self._constraints.append(FormulationConstraint(
            nutrient=nutrient,
            constraint_type=ConstraintType(constraint_type),
            value=value,
            upper_value=upper_value,
        ))

    def formulate_least_cost(
        self,
        requirements: NutrientRequirements,
        target_dmi: Optional[float] = None,
        available_feeds: Optional[List[str]] = None,
    ) -> Ration:
        """
        Formulate least-cost ration.

        Args:
            requirements: Nutrient requirements
            target_dmi: Target dry matter intake (kg)
            available_feeds: List of available feed names

        Returns:
            Optimized Ration
        """
        # Set up ingredients
        if available_feeds:
            self._ingredients = []
            for feed in available_feeds:
                self.add_ingredient(feed)

        if not self._ingredients:
            raise ValueError("No ingredients available")

        n_feeds = len(self._ingredients)

        if target_dmi is None:
            target_dmi = requirements.dry_matter

        # Cost coefficients (per kg DM)
        c = np.array([
            ing.analysis.cost_per_kg * 100 / ing.analysis.dry_matter
            for ing in self._ingredients
        ])

        # Build constraint matrices
        A_ub_list = []
        b_ub_list = []

        # DM constraint: sum = target_dmi
        # Implemented as <= target_dmi and >= target_dmi (using -sum <= -target_dmi)
        A_ub_list.append(np.ones(n_feeds))
        b_ub_list.append(target_dmi * 1.05)  # Allow 5% overage

        A_ub_list.append(-np.ones(n_feeds))
        b_ub_list.append(-target_dmi * 0.95)  # Require 95% of target

        # Protein constraint: sum(CP) >= requirement
        cp_coefs = np.array([
            ing.analysis.crude_protein / 100
            for ing in self._ingredients
        ])
        A_ub_list.append(-cp_coefs)
        b_ub_list.append(-requirements.crude_protein)

        # Energy constraint: sum(ME) >= requirement
        me_coefs = np.array([
            ing.analysis.energy_me
            for ing in self._ingredients
        ])
        A_ub_list.append(-me_coefs)
        b_ub_list.append(-requirements.energy_me)

        # NDF minimum (for ruminants)
        if self.species in ["cattle", "sheep", "goat"]:
            ndf_coefs = np.array([
                ing.analysis.ndf / 100
                for ing in self._ingredients
            ])
            min_ndf = target_dmi * 0.28  # Minimum 28% NDF
            A_ub_list.append(-ndf_coefs)
            b_ub_list.append(-min_ndf)

        # NDF maximum
        max_ndf = target_dmi * 0.50  # Maximum 50% NDF
        A_ub_list.append(ndf_coefs)
        b_ub_list.append(max_ndf)

        # Calcium constraint
        ca_coefs = np.array([
            ing.analysis.calcium / 100 * 1000  # Convert to g
            for ing in self._ingredients
        ])
        A_ub_list.append(-ca_coefs)
        b_ub_list.append(-requirements.calcium)

        # Phosphorus constraint
        p_coefs = np.array([
            ing.analysis.phosphorus / 100 * 1000
            for ing in self._ingredients
        ])
        A_ub_list.append(-p_coefs)
        b_ub_list.append(-requirements.phosphorus)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Variable bounds (ingredient inclusion limits)
        bounds = []
        for ing in self._ingredients:
            lb = target_dmi * ing.min_inclusion / 100
            ub = min(
                target_dmi * ing.max_inclusion / 100,
                ing.available * ing.analysis.dry_matter / 100
            )
            bounds.append((lb, ub))

        # Solve
        result = self.solver.solve(c, A_ub, b_ub, bounds=bounds)

        if not result["success"]:
            logger.warning("Optimization failed, using heuristic solution")
            return self._heuristic_formulation(requirements, target_dmi)

        # Build ration
        amounts = result["x"]
        ingredients = {
            ing.name: amt * 100 / ing.analysis.dry_matter  # Convert to as-fed
            for ing, amt in zip(self._ingredients, amounts)
            if amt > 0.001
        }

        # Calculate nutrients
        nutrients = self.analyzer.calculate_ration_nutrients(ingredients)

        return Ration(
            ingredients=ingredients,
            total_dm=sum(amounts),
            nutrients=nutrients,
            cost=result["fun"],
            analysis={
                "optimization_success": True,
                "method": "linear_programming",
            },
        )

    def _heuristic_formulation(
        self,
        requirements: NutrientRequirements,
        target_dmi: float,
    ) -> Ration:
        """Fallback heuristic formulation."""
        # Simple proportional formulation
        ingredients = {}
        remaining_dm = target_dmi

        # Start with forage (50% of DMI)
        forage_dm = target_dmi * 0.5
        for ing in self._ingredients:
            if ing.analysis.ndf > 40:  # Forage
                amount = min(forage_dm, remaining_dm)
                ingredients[ing.name] = amount * 100 / ing.analysis.dry_matter
                remaining_dm -= amount
                break

        # Add protein source
        for ing in self._ingredients:
            if ing.analysis.crude_protein > 30:  # Protein supplement
                amount = min(target_dmi * 0.15, remaining_dm)
                ingredients[ing.name] = amount * 100 / ing.analysis.dry_matter
                remaining_dm -= amount
                break

        # Fill with energy source
        for ing in self._ingredients:
            if ing.analysis.energy_me > 12:  # Energy source
                ingredients[ing.name] = remaining_dm * 100 / ing.analysis.dry_matter
                break

        nutrients = self.analyzer.calculate_ration_nutrients(ingredients)

        return Ration(
            ingredients=ingredients,
            total_dm=target_dmi,
            nutrients=nutrients,
            cost=nutrients.get("cost", 0),
            analysis={
                "optimization_success": False,
                "method": "heuristic",
            },
        )

    def analyze_ration(
        self,
        ration: Ration,
        requirements: NutrientRequirements,
    ) -> Dict[str, Any]:
        """
        Analyze formulated ration.

        Args:
            ration: Ration to analyze
            requirements: Nutrient requirements

        Returns:
            Dict with analysis results
        """
        balances = self.analyzer.evaluate_balance(requirements, ration.ingredients)
        deficiencies = self.analyzer.identify_deficiencies(balances)

        # Calculate ration composition
        dm_basis = {}
        for name, amount in ration.ingredients.items():
            analysis = self.analyzer.get_feed_analysis(name)
            if analysis:
                dm = amount * analysis.dry_matter / 100
                dm_basis[name] = dm / ration.total_dm * 100

        # Ca:P ratio
        ca = ration.nutrients.get("calcium", 0)
        p = ration.nutrients.get("phosphorus", 0)
        ca_p_ratio = self.analyzer.calculate_ca_p_ratio(ca, p)

        return {
            "ingredient_composition": dm_basis,
            "nutrient_content": {
                "crude_protein_pct": ration.nutrients.get("crude_protein", 0) /
                                     ration.total_dm * 100 if ration.total_dm > 0 else 0,
                "energy_density": ration.nutrients.get("energy_me", 0) /
                                  ration.total_dm if ration.total_dm > 0 else 0,
                "ndf_pct": ration.nutrients.get("ndf", 0) /
                           ration.total_dm * 100 if ration.total_dm > 0 else 0,
            },
            "balance_summary": [
                {"nutrient": b.nutrient, "status": b.status, "balance_pct": b.balance_pct}
                for b in balances
            ],
            "deficiencies": deficiencies,
            "ca_p_ratio": ca_p_ratio,
            "cost_per_kg_dm": ration.cost / ration.total_dm if ration.total_dm > 0 else 0,
            "cost_per_day": ration.cost,
        }

    def optimize_for_target(
        self,
        weight: float,
        production: str = "maintenance",
        target_dmi: Optional[float] = None,
        available_feeds: Optional[List[str]] = None,
        **production_params,
    ) -> Ration:
        """
        Optimize ration for production target.

        Args:
            weight: Animal body weight (kg)
            production: Production stage
            target_dmi: Target DMI (auto-calculated if None)
            available_feeds: Available feed list
            **production_params: Additional production parameters

        Returns:
            Optimized Ration
        """
        requirements = self.calculator.calculate_requirements(
            weight=weight,
            production=production,
            **production_params,
        )

        if target_dmi is None:
            target_dmi = requirements.dry_matter

        return self.formulate_least_cost(
            requirements=requirements,
            target_dmi=target_dmi,
            available_feeds=available_feeds,
        )
