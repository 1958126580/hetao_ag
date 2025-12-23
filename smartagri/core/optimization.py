"""
Optimization Algorithms Module for SmartAgri

This module provides advanced optimization algorithms for agricultural
and animal husbandry applications, including:
- Gradient-based optimization for model training
- Evolutionary algorithms for complex optimization
- Constrained optimization for resource allocation
- Multi-objective optimization for farm planning

Applications:
    - Crop yield model parameter estimation
    - Irrigation schedule optimization
    - Feed formulation optimization
    - Resource allocation planning
    - Harvest timing optimization

Example:
    >>> from smartagri.core.optimization import AdamOptimizer, GeneticAlgorithm
    >>> # Train yield prediction model
    >>> optimizer = AdamOptimizer(learning_rate=0.001)
    >>> for epoch in range(100):
    ...     loss, gradients = compute_loss_and_gradients(model, data)
    ...     optimizer.step(model.parameters, gradients)
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Callable,
    Any,
    Dict,
    TypeVar,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import copy

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, List, Tuple]
ObjectiveFunc = Callable[[np.ndarray], float]
GradientFunc = Callable[[np.ndarray], np.ndarray]


@dataclass
class OptimizationResult:
    """
    Container for optimization results.

    Attributes:
        x: Optimal solution
        fun: Objective function value at solution
        success: Whether optimization converged
        message: Status message
        n_iterations: Number of iterations performed
        n_function_evals: Number of function evaluations
        history: Optimization history (optional)
    """

    x: np.ndarray
    fun: float
    success: bool
    message: str
    n_iterations: int
    n_function_evals: int
    history: Optional[Dict[str, List]] = None


class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """Perform one optimization step."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state."""
        pass


class GradientDescent(BaseOptimizer):
    """
    Gradient Descent optimizer with momentum support.

    Implements vanilla gradient descent with optional momentum
    and learning rate scheduling.

    Attributes:
        learning_rate: Step size for parameter updates
        momentum: Momentum coefficient (0 = no momentum)
        nesterov: Use Nesterov momentum

    Example:
        >>> optimizer = GradientDescent(learning_rate=0.01, momentum=0.9)
        >>> for epoch in range(100):
        ...     gradients = compute_gradients(params, data)
        ...     params = optimizer.step(params, gradients)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        decay: float = 0.0,
    ):
        """
        Initialize Gradient Descent optimizer.

        Args:
            learning_rate: Initial learning rate
            momentum: Momentum coefficient [0, 1)
            nesterov: Use Nesterov accelerated gradient
            decay: Learning rate decay per step
        """
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay

        self._velocity: Optional[np.ndarray] = None
        self._step_count = 0

    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one gradient descent step.

        Args:
            params: Current parameters
            gradients: Gradients of loss with respect to params

        Returns:
            Updated parameters
        """
        self._step_count += 1

        # Apply learning rate decay
        if self.decay > 0:
            self.learning_rate = self.initial_lr / (
                1 + self.decay * self._step_count
            )

        # Initialize velocity
        if self._velocity is None:
            self._velocity = np.zeros_like(params)

        if self.momentum > 0:
            # Update velocity
            self._velocity = (
                self.momentum * self._velocity - self.learning_rate * gradients
            )

            if self.nesterov:
                # Nesterov momentum
                params = params + self.momentum * self._velocity - self.learning_rate * gradients
            else:
                # Standard momentum
                params = params + self._velocity
        else:
            # Vanilla gradient descent
            params = params - self.learning_rate * gradients

        return params

    def reset(self) -> None:
        """Reset optimizer state."""
        self._velocity = None
        self._step_count = 0
        self.learning_rate = self.initial_lr


class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer for adaptive learning rate optimization.

    Combines momentum with adaptive per-parameter learning rates.
    Recommended optimizer for deep learning in agricultural applications.

    Reference:
        Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization"

    Attributes:
        learning_rate: Initial learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Numerical stability constant

    Example:
        >>> optimizer = AdamOptimizer(learning_rate=0.001)
        >>> for epoch in range(1000):
        ...     loss, grads = model.forward_backward(batch)
        ...     params = optimizer.step(params, grads)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Initial learning rate
            beta1: First moment decay (momentum)
            beta2: Second moment decay (RMSprop)
            epsilon: Numerical stability
            weight_decay: L2 regularization coefficient
            amsgrad: Use AMSGrad variant
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self._m: Optional[np.ndarray] = None  # First moment
        self._v: Optional[np.ndarray] = None  # Second moment
        self._v_max: Optional[np.ndarray] = None  # Max second moment (AMSGrad)
        self._step_count = 0

    def step(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one Adam optimization step.

        Args:
            params: Current parameters
            gradients: Gradients of loss

        Returns:
            Updated parameters
        """
        self._step_count += 1

        # Apply weight decay (AdamW style)
        if self.weight_decay > 0:
            params = params - self.learning_rate * self.weight_decay * params

        # Initialize moments
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
            if self.amsgrad:
                self._v_max = np.zeros_like(params)

        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1 - self.beta1) * gradients

        # Update biased second moment estimate
        self._v = self.beta2 * self._v + (1 - self.beta2) * (gradients ** 2)

        # Bias correction
        m_hat = self._m / (1 - self.beta1 ** self._step_count)
        v_hat = self._v / (1 - self.beta2 ** self._step_count)

        if self.amsgrad:
            # Use maximum of current and past v_hat
            self._v_max = np.maximum(self._v_max, v_hat)
            v_hat = self._v_max

        # Update parameters
        params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

    def reset(self) -> None:
        """Reset optimizer state."""
        self._m = None
        self._v = None
        self._v_max = None
        self._step_count = 0


class LBFGSOptimizer:
    """
    Limited-memory BFGS optimizer for large-scale optimization.

    Quasi-Newton method suitable for optimizing with many parameters.
    Efficient for convex optimization problems in agriculture.

    Applications:
        - Crop model calibration
        - Soil parameter estimation
        - Spatial interpolation (kriging)

    Example:
        >>> optimizer = LBFGSOptimizer(max_iter=100)
        >>> result = optimizer.minimize(objective_func, initial_params)
        >>> optimal_params = result.x
    """

    def __init__(
        self,
        max_iter: int = 100,
        memory_size: int = 10,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search: str = "strong_wolfe",
    ):
        """
        Initialize L-BFGS optimizer.

        Args:
            max_iter: Maximum number of iterations
            memory_size: Number of past gradients to store
            tolerance_grad: Gradient norm tolerance for convergence
            tolerance_change: Function change tolerance
            line_search: Line search method ('strong_wolfe', 'backtracking')
        """
        self.max_iter = max_iter
        self.memory_size = memory_size
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.line_search = line_search

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: np.ndarray,
        gradient: Optional[GradientFunc] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizationResult:
        """
        Minimize objective function using L-BFGS.

        Args:
            objective: Objective function to minimize
            x0: Initial parameter values
            gradient: Gradient function (computed numerically if None)
            bounds: Parameter bounds [(low, high), ...]

        Returns:
            OptimizationResult containing optimal solution
        """
        from scipy.optimize import minimize

        if gradient is None:
            jac = "2-point"
        else:
            jac = gradient

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={
                "maxiter": self.max_iter,
                "gtol": self.tolerance_grad,
                "ftol": self.tolerance_change,
                "maxcor": self.memory_size,
            },
        )

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            success=result.success,
            message=result.message,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
        )


class GeneticAlgorithm:
    """
    Genetic Algorithm for global optimization.

    Evolutionary optimization suitable for non-convex problems
    with multiple local optima.

    Applications:
        - Crop rotation optimization
        - Farm layout planning
        - Multi-objective scheduling
        - Feature selection

    Attributes:
        population_size: Number of individuals in population
        n_generations: Number of generations to evolve
        crossover_prob: Probability of crossover
        mutation_prob: Probability of mutation

    Example:
        >>> ga = GeneticAlgorithm(population_size=100, n_generations=200)
        >>> result = ga.minimize(objective, bounds)
        >>> print(f"Optimal solution: {result.x}")
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        elite_size: int = 2,
        tournament_size: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize Genetic Algorithm.

        Args:
            population_size: Size of population
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per gene
            elite_size: Number of elite individuals to preserve
            tournament_size: Tournament selection size
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        self._rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: ObjectiveFunc,
        bounds: List[Tuple[float, float]],
        constraints: Optional[List[Callable]] = None,
    ) -> OptimizationResult:
        """
        Minimize objective using genetic algorithm.

        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds [(low, high), ...]
            constraints: Optional constraint functions (must return >= 0)

        Returns:
            OptimizationResult with optimal solution
        """
        n_vars = len(bounds)
        bounds = np.array(bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        # Initialize population
        population = self._initialize_population(lower, upper)
        fitness = np.array([objective(ind) for ind in population])
        n_evals = self.population_size

        history = {
            "best_fitness": [],
            "mean_fitness": [],
            "diversity": [],
        }

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for generation in range(self.n_generations):
            # Selection
            selected = self._tournament_selection(population, fitness)

            # Crossover
            offspring = self._crossover(selected, lower, upper)

            # Mutation
            offspring = self._mutate(offspring, lower, upper)

            # Evaluate offspring
            offspring_fitness = np.array([objective(ind) for ind in offspring])
            n_evals += len(offspring)

            # Apply constraints (penalty method)
            if constraints is not None:
                penalties = self._compute_penalties(offspring, constraints)
                offspring_fitness = offspring_fitness + penalties

            # Elitism: preserve best individuals
            elite_idx = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_idx].copy()
            elite_fitness = fitness[elite_idx].copy()

            # Combine and select next generation
            combined_pop = np.vstack([offspring, elite])
            combined_fitness = np.concatenate([offspring_fitness, elite_fitness])

            # Select best for next generation
            best_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            # Update best solution
            if fitness[0] < best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness[0]

            # Record history
            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(np.mean(fitness))
            history["diversity"].append(np.std(population))

            logger.debug(
                f"Generation {generation}: best={best_fitness:.6f}, "
                f"mean={np.mean(fitness):.6f}"
            )

        return OptimizationResult(
            x=best_individual,
            fun=best_fitness,
            success=True,
            message=f"Optimization completed after {self.n_generations} generations",
            n_iterations=self.n_generations,
            n_function_evals=n_evals,
            history=history,
        )

    def _initialize_population(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Initialize random population within bounds."""
        return self._rng.uniform(
            lower,
            upper,
            size=(self.population_size, len(lower)),
        )

    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Select individuals using tournament selection."""
        selected = []
        n_select = self.population_size - self.elite_size

        for _ in range(n_select):
            tournament_idx = self._rng.choice(
                len(population),
                size=self.tournament_size,
                replace=False,
            )
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return np.array(selected)

    def _crossover(
        self,
        parents: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Perform crossover on selected parents."""
        offspring = parents.copy()
        n_pairs = len(parents) // 2

        for i in range(n_pairs):
            if self._rng.random() < self.crossover_prob:
                p1, p2 = parents[2 * i], parents[2 * i + 1]

                # Simulated Binary Crossover (SBX)
                eta = 2.0
                u = self._rng.random(len(p1))

                beta = np.where(
                    u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)),
                )

                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

                # Clip to bounds
                offspring[2 * i] = np.clip(c1, lower, upper)
                offspring[2 * i + 1] = np.clip(c2, lower, upper)

        return offspring

    def _mutate(
        self,
        population: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Apply polynomial mutation."""
        eta_m = 20.0

        for i in range(len(population)):
            for j in range(len(population[i])):
                if self._rng.random() < self.mutation_prob:
                    x = population[i, j]
                    delta1 = (x - lower[j]) / (upper[j] - lower[j])
                    delta2 = (upper[j] - x) / (upper[j] - lower[j])

                    u = self._rng.random()

                    if u < 0.5:
                        xy = 1 - delta1
                        val = 2 * u + (1 - 2 * u) * (xy ** (eta_m + 1))
                        delta_q = val ** (1 / (eta_m + 1)) - 1
                    else:
                        xy = 1 - delta2
                        val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta_m + 1))
                        delta_q = 1 - val ** (1 / (eta_m + 1))

                    population[i, j] = np.clip(
                        x + delta_q * (upper[j] - lower[j]),
                        lower[j],
                        upper[j],
                    )

        return population

    def _compute_penalties(
        self,
        population: np.ndarray,
        constraints: List[Callable],
    ) -> np.ndarray:
        """Compute constraint violation penalties."""
        penalties = np.zeros(len(population))
        penalty_coeff = 1000.0

        for i, ind in enumerate(population):
            for constraint in constraints:
                violation = constraint(ind)
                if violation < 0:
                    penalties[i] += penalty_coeff * (violation ** 2)

        return penalties


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for continuous optimization.

    Swarm intelligence algorithm inspired by bird flocking behavior.
    Good for multimodal optimization with many local optima.

    Applications:
        - Neural network hyperparameter tuning
        - Sensor placement optimization
        - Irrigation system design

    Example:
        >>> pso = ParticleSwarmOptimizer(n_particles=30, n_iterations=100)
        >>> result = pso.minimize(objective, bounds)
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_ratio: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        Initialize Particle Swarm Optimizer.

        Args:
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive coefficient (personal best attraction)
            c2: Social coefficient (global best attraction)
            v_max_ratio: Maximum velocity as ratio of search space
            seed: Random seed
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio

        self._rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: ObjectiveFunc,
        bounds: List[Tuple[float, float]],
    ) -> OptimizationResult:
        """
        Minimize objective using particle swarm optimization.

        Args:
            objective: Objective function
            bounds: Parameter bounds

        Returns:
            OptimizationResult with optimal solution
        """
        n_vars = len(bounds)
        bounds = np.array(bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        range_size = upper - lower

        # Maximum velocity
        v_max = self.v_max_ratio * range_size

        # Initialize particles
        positions = self._rng.uniform(
            lower, upper, size=(self.n_particles, n_vars)
        )
        velocities = self._rng.uniform(
            -v_max, v_max, size=(self.n_particles, n_vars)
        )

        # Evaluate initial fitness
        fitness = np.array([objective(p) for p in positions])
        n_evals = self.n_particles

        # Initialize personal and global bests
        personal_best_pos = positions.copy()
        personal_best_fit = fitness.copy()

        global_best_idx = np.argmin(fitness)
        global_best_pos = positions[global_best_idx].copy()
        global_best_fit = fitness[global_best_idx]

        history = {
            "best_fitness": [global_best_fit],
            "mean_fitness": [np.mean(fitness)],
        }

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1 = self._rng.random(n_vars)
                r2 = self._rng.random(n_vars)

                cognitive = self.c1 * r1 * (personal_best_pos[i] - positions[i])
                social = self.c2 * r2 * (global_best_pos - positions[i])

                velocities[i] = self.w * velocities[i] + cognitive + social

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]

                # Enforce bounds
                positions[i] = np.clip(positions[i], lower, upper)

                # Evaluate
                fit = objective(positions[i])
                n_evals += 1

                # Update personal best
                if fit < personal_best_fit[i]:
                    personal_best_fit[i] = fit
                    personal_best_pos[i] = positions[i].copy()

                    # Update global best
                    if fit < global_best_fit:
                        global_best_fit = fit
                        global_best_pos = positions[i].copy()

            history["best_fitness"].append(global_best_fit)
            history["mean_fitness"].append(np.mean(personal_best_fit))

            logger.debug(
                f"Iteration {iteration}: best={global_best_fit:.6f}"
            )

        return OptimizationResult(
            x=global_best_pos,
            fun=global_best_fit,
            success=True,
            message=f"PSO completed after {self.n_iterations} iterations",
            n_iterations=self.n_iterations,
            n_function_evals=n_evals,
            history=history,
        )


class SimulatedAnnealing:
    """
    Simulated Annealing for global optimization.

    Probabilistic optimization inspired by metallurgical annealing.
    Can escape local minima by accepting worse solutions with
    decreasing probability.

    Applications:
        - Combinatorial optimization
        - Scheduling problems
        - Layout optimization

    Example:
        >>> sa = SimulatedAnnealing(initial_temp=100, final_temp=0.01)
        >>> result = sa.minimize(objective, x0, bounds)
    """

    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        n_iterations_per_temp: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Initialize Simulated Annealing optimizer.

        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature decay rate per epoch
            n_iterations_per_temp: Iterations at each temperature
            seed: Random seed
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.n_iterations_per_temp = n_iterations_per_temp

        self._rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        step_size: float = 0.5,
    ) -> OptimizationResult:
        """
        Minimize objective using simulated annealing.

        Args:
            objective: Objective function
            x0: Initial solution
            bounds: Parameter bounds
            step_size: Step size as fraction of range

        Returns:
            OptimizationResult with optimal solution
        """
        bounds = np.array(bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        range_size = upper - lower

        # Initialize
        current = np.clip(x0.copy(), lower, upper)
        current_energy = objective(current)
        n_evals = 1

        best = current.copy()
        best_energy = current_energy

        temperature = self.initial_temp
        n_accepted = 0
        total_iterations = 0

        history = {
            "temperature": [],
            "best_energy": [],
            "current_energy": [],
            "acceptance_rate": [],
        }

        while temperature > self.final_temp:
            accepted_at_temp = 0

            for _ in range(self.n_iterations_per_temp):
                # Generate neighbor
                perturbation = self._rng.normal(0, step_size, len(current))
                neighbor = current + perturbation * range_size
                neighbor = np.clip(neighbor, lower, upper)

                # Evaluate neighbor
                neighbor_energy = objective(neighbor)
                n_evals += 1

                # Metropolis criterion
                delta = neighbor_energy - current_energy

                if delta < 0 or self._rng.random() < np.exp(-delta / temperature):
                    current = neighbor
                    current_energy = neighbor_energy
                    accepted_at_temp += 1
                    n_accepted += 1

                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy

                total_iterations += 1

            # Record history
            history["temperature"].append(temperature)
            history["best_energy"].append(best_energy)
            history["current_energy"].append(current_energy)
            history["acceptance_rate"].append(
                accepted_at_temp / self.n_iterations_per_temp
            )

            # Cool down
            temperature *= self.cooling_rate

            logger.debug(
                f"T={temperature:.4f}: best={best_energy:.6f}, "
                f"acceptance={accepted_at_temp / self.n_iterations_per_temp:.2%}"
            )

        return OptimizationResult(
            x=best,
            fun=best_energy,
            success=True,
            message=f"SA completed with final temperature {temperature:.6f}",
            n_iterations=total_iterations,
            n_function_evals=n_evals,
            history=history,
        )


class DifferentialEvolution:
    """
    Differential Evolution for robust global optimization.

    Evolutionary strategy that uses differences between population
    members for mutation. Excellent for noisy objective functions.

    Example:
        >>> de = DifferentialEvolution(population_size=50)
        >>> result = de.minimize(noisy_objective, bounds)
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = "best1bin",
        seed: Optional[int] = None,
    ):
        """
        Initialize Differential Evolution.

        Args:
            population_size: Population size (should be >= 4)
            n_generations: Number of generations
            F: Mutation factor [0, 2]
            CR: Crossover rate [0, 1]
            strategy: Mutation strategy ('best1bin', 'rand1bin', 'best2bin')
            seed: Random seed
        """
        self.population_size = max(4, population_size)
        self.n_generations = n_generations
        self.F = F
        self.CR = CR
        self.strategy = strategy

        self._rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: ObjectiveFunc,
        bounds: List[Tuple[float, float]],
    ) -> OptimizationResult:
        """
        Minimize objective using differential evolution.

        Args:
            objective: Objective function
            bounds: Parameter bounds

        Returns:
            OptimizationResult with optimal solution
        """
        n_vars = len(bounds)
        bounds = np.array(bounds)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        # Initialize population
        population = self._rng.uniform(
            lower, upper, size=(self.population_size, n_vars)
        )
        fitness = np.array([objective(ind) for ind in population])
        n_evals = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        history = {"best_fitness": [], "mean_fitness": []}

        for generation in range(self.n_generations):
            for i in range(self.population_size):
                # Select mutation indices
                idxs = [j for j in range(self.population_size) if j != i]

                if self.strategy == "best1bin":
                    r1, r2 = self._rng.choice(idxs, size=2, replace=False)
                    mutant = best + self.F * (population[r1] - population[r2])
                elif self.strategy == "rand1bin":
                    r1, r2, r3 = self._rng.choice(idxs, size=3, replace=False)
                    mutant = population[r1] + self.F * (
                        population[r2] - population[r3]
                    )
                else:  # best2bin
                    r1, r2, r3, r4 = self._rng.choice(idxs, size=4, replace=False)
                    mutant = best + self.F * (
                        population[r1] - population[r2] +
                        population[r3] - population[r4]
                    )

                # Clip to bounds
                mutant = np.clip(mutant, lower, upper)

                # Crossover
                trial = np.zeros(n_vars)
                j_rand = self._rng.integers(n_vars)

                for j in range(n_vars):
                    if self._rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i, j]

                # Selection
                trial_fitness = objective(trial)
                n_evals += 1

                if trial_fitness <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial.copy()
                        best_fitness = trial_fitness

            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(np.mean(fitness))

            logger.debug(
                f"Generation {generation}: best={best_fitness:.6f}"
            )

        return OptimizationResult(
            x=best,
            fun=best_fitness,
            success=True,
            message=f"DE completed after {self.n_generations} generations",
            n_iterations=self.n_generations,
            n_function_evals=n_evals,
            history=history,
        )
